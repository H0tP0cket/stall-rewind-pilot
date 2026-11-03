"""Prefix-bounded, visibility-checked access to recorded trajectory events.

Every observer (incremental T, full-prefix audit, independent summary) reads
the recorded trajectory only through a PrefixView. The view enforces:

* an event interval (after_seq, through_seq] that never exceeds the target
  checkpoint's event cut. The store is queried with through_sequence=cut, and
  the view additionally drops anything later;
* worker visibility: only content delivered to, or expressed by, the worker is
  readable. Raw tool output, grades, detector output, annotations, captures and
  research-private records are never eligible;
* an access log, so each derived artifact records the source events it consumed.

Observer-source event convention (contracts.EventRecord, confirmed with the
execution session on 2026-09-15; `source_event_from_record` is the single
mapping point):

* kind="task_disclosure", WORKER_VISIBLE, delivered_ref = exact delivered
  system/user text (one per delivered task/stage/restart message).
* kind="model_response", WORKER_VISIBLE, call_id, output_ref = normalized JSON
  {"content", "tool_calls": [{"id","name","arguments"}], "reasoning_content",
   "finish_reason"}; only accepted responses carry this kind.
* kind="format_error", WORKER_VISIBLE, delivered_ref = exact error message
  delivered to the worker (an observation).
* kind="observation_delivery", WORKER_VISIBLE, call_id, delivered_ref = exact
  tool content appended to H; parent_event_id binds its raw tool_result.
* Legacy kind="tool_result", WORKER_VISIBLE with delivered_ref is eligible only
  with an explicit append receipt or prefix-bounded H/API delivery proof.
  Current raw tool_result is RESEARCH_PRIVATE and never read here.
* kind="treatment_packet", WORKER_VISIBLE, delivered_ref = injected packet
  bytes on branches. Deliberately NOT a T source: T must not be re-extracted
  from a packet that was itself derived from T.
* model_request / tool_call are ignored; every RESEARCH_PRIVATE kind
  (model_attempt, mutation, capture, grade, detector, annotation, lifecycle,
  terminal, incident) is refused.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from types import MappingProxyType

from ..contracts import ArtifactStoreProtocol, EventRecord, Visibility
from ._canonical import sha256_hex

WORKER_EXPRESSED = "worker_expressed"
OBSERVATION_DIRECT = "observation_direct"
SESSION_LINEAGE_REVISION = "native-session-primary/v1"

_TOOL_ARG_FIELD = re.compile(r"^tool_call\[(\d+)\]\.arguments$")

# Kinds that must never be read by an observer, whatever their visibility flag.
FORBIDDEN_KINDS = frozenset({
    "model_attempt", "mutation", "capture", "grade", "evaluation", "detector", "annotation",
    "lifecycle", "terminal", "incident", "private_evaluator", "reference", "treatment_packet",
})


class PrefixViolation(Exception):
    """Raised when an observer tries to read outside its allowed prefix."""


@dataclass(frozen=True)
class SourceEvent:
    seq: int
    kind: str
    worker_visible: bool
    stage_id: str | None
    session_id: str | None
    texts: Mapping[str, str]
    tool_call_id: str | None = None
    raw_differs_from_delivered: bool | None = None
    event_id: str | None = None
    delivery_basis: str | None = None


def field_category(kind: str, field_name: str) -> str | None:
    """Source category of an eligible field, or None when not eligible."""
    if kind == "model_response":
        if field_name == "assistant_text" or _TOOL_ARG_FIELD.match(field_name):
            return WORKER_EXPRESSED
        if field_name == "returned_reasoning_text":
            return WORKER_EXPRESSED
        return None
    if kind in {"tool_result", "observation_delivery"} and field_name == "delivered_content":
        return OBSERVATION_DIRECT
    if kind in {"task_disclosure", "format_error"} and field_name == "delivered_text":
        return OBSERVATION_DIRECT
    return None


@dataclass(frozen=True)
class SourceScope:
    """Declared source universe for one observer call.

    session_scope: "full_trajectory" keeps historically allowed content from
    earlier native sessions; "current_session" restricts to the session active
    at the target checkpoint. include_returned_reasoning is an unratified
    source choice and defaults off.
    """

    session_scope: str = "full_trajectory"
    include_returned_reasoning: bool = False
    # Absent on historical scopes; do not alter their wire representation.
    lineage_revision: str | None = None

    def __post_init__(self) -> None:
        if self.session_scope not in {"full_trajectory", "current_session"}:
            raise ValueError(f"unknown session scope {self.session_scope!r}")
        if type(self.include_returned_reasoning) is not bool:
            raise ValueError("include_returned_reasoning must be an explicit boolean")
        if self.lineage_revision not in (None, SESSION_LINEAGE_REVISION):
            raise ValueError("Unsupported observer session lineage revision")

    def as_record(self) -> dict[str, Any]:
        result = {
            "session_scope": self.session_scope,
            "include_returned_reasoning": self.include_returned_reasoning,
        }
        if self.lineage_revision is not None:
            result["lineage_revision"] = self.lineage_revision
        return result


@dataclass
class PrefixView:
    run_id: str
    events_by_seq: Mapping[int, SourceEvent]
    after_seq: int
    through_seq: int
    target_session_id: str | None
    scope: SourceScope = field(default_factory=SourceScope)
    accessed: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.through_seq < self.after_seq:
            raise PrefixViolation(f"inverted interval ({self.after_seq}, {self.through_seq}]")
        if self.scope.session_scope not in {"full_trajectory", "current_session"}:
            raise ValueError(f"unknown session scope {self.scope.session_scope!r}")
        if (self.scope.lineage_revision == SESSION_LINEAGE_REVISION and
                (type(self.target_session_id) is not str or not self.target_session_id)):
            raise PrefixViolation("Versioned observer lineage requires an identified native session")
        late = [s for s in self.events_by_seq if s > self.through_seq]
        if late:
            raise PrefixViolation(f"view holds events after its cut: {sorted(late)[:3]}")
        safe = {}
        for seq, event in self.events_by_seq.items():
            if type(seq) is not int or seq < 0 or event.seq != seq:
                raise PrefixViolation("source event sequence/key mismatch or invalid sequence")
            fields = self._eligible_fields(event)
            safe[seq] = SourceEvent(
                seq=seq, kind=event.kind, worker_visible=event.worker_visible,
                stage_id=event.stage_id, session_id=event.session_id,
                texts=MappingProxyType({name: event.texts[name] for name in fields}),
                tool_call_id=event.tool_call_id,
                raw_differs_from_delivered=event.raw_differs_from_delivered,
                event_id=event.event_id,
                delivery_basis=event.delivery_basis,
            )
        self.events_by_seq = MappingProxyType(safe)

    def _eligible_fields(self, event: SourceEvent) -> dict[str, str]:
        if event.kind in FORBIDDEN_KINDS or not event.worker_visible:
            return {}
        if (
            self.scope.session_scope == "current_session"
            and event.session_id != self.target_session_id
        ):
            return {}
        out: dict[str, str] = {}
        for name in event.texts:
            category = field_category(event.kind, name)
            if category is None:
                continue
            if name == "returned_reasoning_text" and not self.scope.include_returned_reasoning:
                continue
            out[name] = category
        return out

    def in_interval(self, seq: int) -> bool:
        return self.after_seq < seq <= self.through_seq

    def events(self) -> list[SourceEvent]:
        """Eligible events in the interval, in order, restricted to eligible fields."""
        out: list[SourceEvent] = []
        for seq in sorted(self.events_by_seq):
            if not self.in_interval(seq):
                continue
            event = self.events_by_seq[seq]
            fields = self._eligible_fields(event)
            if not fields:
                continue
            self.accessed.add(seq)
            out.append(
                SourceEvent(
                    seq=event.seq,
                    kind=event.kind,
                    worker_visible=True,
                    stage_id=event.stage_id,
                    session_id=event.session_id,
                    texts={name: event.texts[name] for name in event.texts if name in fields},
                    tool_call_id=event.tool_call_id,
                    raw_differs_from_delivered=event.raw_differs_from_delivered,
                    event_id=event.event_id,
                    delivery_basis=event.delivery_basis,
                )
            )
        return out

    def resolve(self, seq: int, field_name: str, *, allow_before_interval: bool = False) -> tuple[str, str]:
        """Return (text, source_category) for an eligible field.

        allow_before_interval permits citing earlier allowed events (seq <=
        after_seq), used when validating inherited entries. The cut is never
        relaxed.
        """
        if seq > self.through_seq:
            raise PrefixViolation(f"seq {seq} is after the cut {self.through_seq}")
        if not allow_before_interval and seq <= self.after_seq:
            raise PrefixViolation(f"seq {seq} precedes interval start {self.after_seq}")
        event = self.events_by_seq.get(seq)
        if event is None:
            raise PrefixViolation(f"seq {seq} does not exist in the allowed prefix")
        fields = self._eligible_fields(event)
        if field_name not in fields:
            raise PrefixViolation(
                f"field {field_name!r} of seq {seq} (kind {event.kind}) is not an eligible source"
            )
        self.accessed.add(seq)
        return event.texts[field_name], fields[field_name]

    def source_event(self, seq: int) -> SourceEvent:
        if seq > self.through_seq or seq not in self.events_by_seq:
            raise PrefixViolation(f"seq {seq} is not in the allowed prefix")
        return self.events_by_seq[seq]

    def narrowed(self, after_seq: int, through_seq: int) -> "PrefixView":
        if through_seq > self.through_seq:
            raise PrefixViolation("cannot widen a prefix view beyond its cut")
        return PrefixView(
            run_id=self.run_id,
            events_by_seq={s: e for s, e in self.events_by_seq.items() if s <= through_seq},
            after_seq=after_seq,
            through_seq=through_seq,
            target_session_id=self.target_session_id,
            scope=self.scope,
        )

    def visibility_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "interval": {"after_seq": self.after_seq, "through_seq": self.through_seq},
            "target_session_id": self.target_session_id,
            "scope": self.scope.as_record(),
            "eligible_sources": {
                "task_disclosure": ["delivered_text"],
                "format_error": ["delivered_text"],
                "model_response": ["assistant_text", "tool_call[i].arguments"]
                + (["returned_reasoning_text"] if self.scope.include_returned_reasoning else []),
                "tool_result": ["delivered_content"],
                "observation_delivery": ["delivered_content"],
            },
            "accessed_seqs": sorted(self.accessed),
            "tool_delivery_policy": "explicit_history_append_receipt_or_prefix_bounded_H_API_membership",
            "unverified_tool_event_seqs": [s for s, e in self.events_by_seq.items()
                                            if e.kind in {"tool_result", "observation_delivery"}
                                            and e.worker_visible and not e.texts],
        }


def build_view(
    run_id: str,
    events: Iterable[SourceEvent],
    *,
    after_seq: int,
    through_seq: int,
    target_session_id: str | None,
    scope: SourceScope | None = None,
) -> PrefixView:
    """Build a view that physically holds no event beyond the cut."""
    by_seq: dict[int, SourceEvent] = {}
    for event in events:
        if event.seq > through_seq:
            continue
        if event.seq in by_seq:
            raise ValueError(f"duplicate seq {event.seq} in run {run_id}")
        by_seq[event.seq] = event
    return PrefixView(
        run_id=run_id,
        events_by_seq=by_seq,
        after_seq=after_seq,
        through_seq=through_seq,
        target_session_id=target_session_id,
        scope=scope or SourceScope(),
    )


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8")


def _delivered_tool_events(
    store: ArtifactStoreProtocol, run_id: str, records: Iterable[EventRecord], *, through_sequence: int | None,
) -> set[str]:
    """Read a bounded delivery index once; raw output and observation-ref lists are not proof.

    Legacy event flags were emitted before H append. Match exact delivered bytes,
    call identity and native session against H/API snapshots within the cut.
    Duplicate legacy identities are ambiguous and fail closed. No cache outlives
    this prefix read, so a growing run cannot reuse stale or future membership.
    """
    records = list(records)
    candidates: dict[tuple[str | None, str | None, str], list[EventRecord]] = defaultdict(list)
    for event in records:
        if event.kind == "tool_result" and event.delivered_ref is not None:
            key = (event.scope.session_id if event.scope else None, event.call_id, event.delivered_ref.sha256)
            candidates[key].append(event)
    matched: set[str] = set()
    if not candidates:
        return matched

    def inspect(messages: Any, session_id: str | None, proof_sequence: int) -> None:
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "tool" or not isinstance(message.get("content"), str):
                continue
            key = (session_id, message.get("tool_call_id"), sha256_hex(message["content"]))
            events = candidates.get(key, ())
            if len(events) == 1 and events[0].sequence <= proof_sequence:
                matched.add(events[0].event_id)

    lister = getattr(store, "list_checkpoints", None)
    if lister is not None:
        for checkpoint in lister(run_id):
            if checkpoint.run_id != run_id:
                raise PrefixViolation("store returned a checkpoint from another run")
            if through_sequence is not None and checkpoint.event_sequence > through_sequence:
                continue
            if checkpoint.state.history is not None:
                body = store.get_json(checkpoint.state.history)
                inspect(body.get("messages"), checkpoint.scope.session_id, checkpoint.event_sequence)
    for event in records:
        if event.kind == "model_request" and event.input_ref is not None:
            body = store.get_json(event.input_ref)
            inspect(body.get("messages"), event.scope.session_id if event.scope else None, event.sequence)
    return matched


def source_event_from_record(
    record: EventRecord, store: ArtifactStoreProtocol, *,
    verified_observation_events: set[str] | None = None,
    records_by_id: Mapping[str, EventRecord] | None = None,
) -> SourceEvent:
    """Map one EventRecord to a SourceEvent, reading only eligible artifacts."""
    visible = record.visibility == Visibility.WORKER_VISIBLE
    scope = record.scope
    texts: dict[str, str] = {}
    raw_differs = None
    delivery_basis = None
    if visible and record.kind not in FORBIDDEN_KINDS:
        if record.kind in {"task_disclosure", "format_error"} and record.delivered_ref is not None:
            texts["delivered_text"] = _decode_text(store.get_bytes(record.delivered_ref))
        elif record.kind == "model_response" and record.output_ref is not None \
                and record.payload.get("accepted", True) is not False:
            # A rejected response (format error) was never accepted into the
            # worker session; only the delivered format_error is a source.
            body = store.get_json(record.output_ref)
            if isinstance(body.get("content"), str):
                texts["assistant_text"] = body["content"]
            for i, call in enumerate(body.get("tool_calls") or ()):
                args = call.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args, sort_keys=True, ensure_ascii=False)
                texts[f"tool_call[{i}].arguments"] = args
            if isinstance(body.get("reasoning_content"), str):
                texts["returned_reasoning_text"] = body["reasoning_content"]
        elif record.kind in {"tool_result", "observation_delivery"} and record.delivered_ref is not None:
            declared_delivery = record.payload.get("delivered_to_history")
            if record.kind == "observation_delivery":
                if records_by_id is None:
                    records_by_id = {e.event_id: e for e in store.iter_events(record.run_id, through_sequence=record.sequence)}
                parent = records_by_id.get(record.parent_event_id)
                if (record.payload.get("delivery_status") == "appended_to_history"
                        and record.payload.get("role") == "tool" and parent is not None
                        and parent.kind == "tool_result" and parent.run_id == record.run_id
                        and parent.sequence < record.sequence and parent.call_id == record.call_id
                        and parent.scope == record.scope):
                    delivery_basis = "observation_delivery_after_history_append"
            elif declared_delivery is True:
                delivery_basis = "explicit_history_append_receipt"
            elif declared_delivery is not False:
                if verified_observation_events is None:
                    # Direct mapper use is a read-only whole-recorded-run diagnostic.
                    # Production views always pass their prefix-bounded index below.
                    records = list(store.iter_events(record.run_id))
                    verified_observation_events = _delivered_tool_events(
                        store, record.run_id, records, through_sequence=None)
                if record.event_id in verified_observation_events:
                    delivery_basis = "prefix_H_or_API_membership"
            # Only the delivered view; output_ref (raw) is never dereferenced.
            if delivery_basis is not None:
                texts["delivered_content"] = _decode_text(store.get_bytes(record.delivered_ref))
            declared = record.payload.get("raw_equals_delivered")
            if isinstance(declared, bool):
                raw_differs = not declared
            elif record.output_ref is not None:
                raw_differs = record.output_ref.sha256 != record.delivered_ref.sha256
    return SourceEvent(
        seq=record.sequence,
        kind=record.kind,
        worker_visible=visible,
        stage_id=scope.stage_id if scope else None,
        session_id=scope.session_id if scope else None,
        texts=texts,
        tool_call_id=record.call_id,
        raw_differs_from_delivered=raw_differs,
        event_id=record.event_id,
        delivery_basis=delivery_basis,
    )


def source_events_from_store(
    store: ArtifactStoreProtocol, run_id: str, *, through_sequence: int
) -> list[SourceEvent]:
    out: list[SourceEvent] = []
    records = list(store.iter_events(run_id, through_sequence=through_sequence))
    for record in records:
        if record.run_id != run_id:
            raise PrefixViolation("store returned an event from another run")
        if record.sequence > through_sequence:
            raise PrefixViolation(
                f"store returned seq {record.sequence} beyond requested cut {through_sequence}"
            )
    verified = _delivered_tool_events(store, run_id, records, through_sequence=through_sequence)
    by_id = {record.event_id: record for record in records}
    for record in records:
        out.append(source_event_from_record(record, store, verified_observation_events=verified, records_by_id=by_id))
    return out


def view_for_checkpoint(
    store: ArtifactStoreProtocol,
    run_id: str,
    *,
    through_seq: int,
    after_seq: int = -1,
    target_session_id: str | None,
    scope: SourceScope | None = None,
) -> PrefixView:
    events = source_events_from_store(store, run_id, through_sequence=through_seq)
    return build_view(
        run_id, events, after_seq=after_seq, through_seq=through_seq,
        target_session_id=target_session_id, scope=scope,
    )
