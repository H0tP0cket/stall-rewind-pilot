"""Q4: mask retained delivered O in the recorded detector window (A, S1].

Raw archival results are never delivery evidence. Mapping is to exact tool
messages in source H, with call identity and delivery receipts. Compaction text
remains H. This module prepares/replays a mask; it never runs a worker or tool.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Mapping, Sequence

from ..contracts import ArtifactStoreProtocol, Checkpoint, EventRecord, Visibility, canonical_json, to_jsonable
from .selection import first_stall_annotation

REVISION = "retained-observation-mask/v1"
from ..detector.stall_clock_v7 import STALL_CLOCK_RULE_VERSIONS

PRIMARY_WINDOW_POLICY = "o-mask-detector-anchor-window.v1"
SECONDARY_ALL_HISTORY_POLICY = "o-mask-secondary-all-retained-deliveries.v1"
ENGINEERING_ALL_HISTORY_POLICY = "o-mask-engineering-all-retained-deliveries.v1"
MANUAL_CHECKPOINT_WINDOW_POLICY = "o-mask-manual-checkpoint-window.v1"
MASK_PLACEHOLDER = "[TOOL OUTPUT ELIDED BY EXPERIMENT]"
_POLICIES = {PRIMARY_WINDOW_POLICY, SECONDARY_ALL_HISTORY_POLICY, ENGINEERING_ALL_HISTORY_POLICY,
             MANUAL_CHECKPOINT_WINDOW_POLICY}


class MaskError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "mask_id": _digest(record)}


def _delivery_candidates(events: Sequence[EventRecord]) -> list[dict[str, Any]]:
    by_id = {event.event_id: event for event in events}
    candidates = []
    for event in events:
        if event.delivered_ref is None or event.visibility != Visibility.WORKER_VISIBLE:
            continue
        if event.kind == "observation_delivery":
            parent = by_id.get(event.parent_event_id)
            if (event.payload.get("delivery_status") != "appended_to_history" or event.payload.get("role") != "tool"
                    or parent is None or parent.kind != "tool_result" or parent.sequence >= event.sequence
                    or parent.scope != event.scope or parent.call_id != event.call_id or parent.run_id != event.run_id):
                continue
            start = by_id.get(parent.parent_event_id)
            basis = "observation_delivery_after_history_append_and_exact_current_H_membership"
        elif event.kind == "tool_result" and event.payload.get("delivered_to_history") is not False:
            parent, start = event, by_id.get(event.parent_event_id)
            basis = "legacy_delivered_ref_and_exact_current_H_membership"
        else:
            continue
        candidates.append({"event": event, "result_event_id": parent.event_id,
                           "model_call_id": start.payload.get("model_call_id") if start is not None and start.kind == "tool_call" else None,
                           "basis": basis})
    # A later explicit append receipt supersedes its raw/prepared parent as the
    # delivery occurrence. Do not manufacture two deliveries from one operation.
    explicit_parents = {item["result_event_id"] for item in candidates
                        if item["event"].kind == "observation_delivery"}
    return [item for item in candidates if item["event"].kind != "tool_result"
            or item["event"].event_id not in explicit_parents]


def build_observation_mask(store: ArtifactStoreProtocol, source: Checkpoint, *, selection_kind: str,
                           policy: str | None = None, after_checkpoint_id: str | None = None,
                           through_checkpoint_id: str | None = None,
                           decision_checkpoint_id: str | None = None) -> dict[str, Any]:
    """Produce a fully bound available spec or explicit unavailable-arm record.

    Engineering and secondary all-history policies must be named by the caller.
    The default is the prospective primary detector window, never a best donor
    approximation or an implicit all-history fallback.

    The manual secondary policy masks (after checkpoint, through checkpoint]
    in the selected source H. Through defaults to H's own checkpoint. All cuts
    and H must precede the explicit decision in one natural run/session/target.
    An O cut is an evidence boundary, not a claim that its W/R is restorable.
    """
    policy = policy or PRIMARY_WINDOW_POLICY
    if policy not in _POLICIES or selection_kind not in {"primary_s1", "engineering_checkpoint", "secondary"}:
        raise MaskError("Unknown observation mask policy or source selection kind")
    if policy != MANUAL_CHECKPOINT_WINDOW_POLICY and any(
            value is not None for value in (after_checkpoint_id, through_checkpoint_id, decision_checkpoint_id)):
        raise MaskError("Checkpoint range selectors apply only to the explicit manual secondary policy")
    record = {"revision": REVISION, "policy_id": policy, "selection_kind": selection_kind,
              "comparison_role": {PRIMARY_WINDOW_POLICY: "primary", SECONDARY_ALL_HISTORY_POLICY: "secondary",
                                  ENGINEERING_ALL_HISTORY_POLICY: "engineering",
                                  MANUAL_CHECKPOINT_WINDOW_POLICY: "secondary"}[policy],
              "source_run_id": source.run_id, "source_checkpoint_id": source.checkpoint_id,
              "source_scope": to_jsonable(source.scope), "source_history_ref": to_jsonable(source.state.history),
              "through_event_sequence": source.event_sequence, "placeholder": MASK_PLACEHOLDER,
              "anchor_checkpoint_id": None, "after_event_sequence": None,
              "selected_messages": [], "delivered_but_not_retained": [],
              "preserve": ["assistant_turns", "tool_calls", "arguments", "call_result_pairing", "compaction_summaries"]}
    if policy == MANUAL_CHECKPOINT_WINDOW_POLICY:
        record.update(after_checkpoint_id=after_checkpoint_id,
                      through_checkpoint_id=source.checkpoint_id if through_checkpoint_id is None else through_checkpoint_id,
                      decision_checkpoint_id=decision_checkpoint_id)

    def unavailable(reason: str) -> dict[str, Any]:
        return _seal({**record, "status": "unavailable", "reason": reason})

    if policy == PRIMARY_WINDOW_POLICY and selection_kind != "primary_s1":
        return unavailable("No primary S1 window declared; engineering/secondary masks require an explicit mask policy")
    if policy == ENGINEERING_ALL_HISTORY_POLICY and selection_kind != "engineering_checkpoint":
        return unavailable("Engineering all-history mask requires an engineering checkpoint selection")
    if policy == SECONDARY_ALL_HISTORY_POLICY and selection_kind != "secondary":
        return unavailable("All-history O masking requires an explicitly secondary comparison")
    if policy == MANUAL_CHECKPOINT_WINDOW_POLICY and selection_kind != "secondary":
        return unavailable("Manual checkpoint-window O masking requires an explicitly secondary comparison")
    if source.state.history is None:
        return unavailable("Source checkpoint has no retained H artifact")
    if any(type(value) is not str or not value for value in to_jsonable(source.scope).values()):
        return unavailable("Source checkpoint has unidentified session/target scope fields")
    try:
        saved_source = store.get_checkpoint(source.checkpoint_id)
        if canonical_json(saved_source) != canonical_json(source):
            return unavailable("Source checkpoint differs from its immutable record")
        history = store.get_json(source.state.history)
        if not isinstance(history, Mapping):
            return unavailable("Source H does not contain a supported recorded message sequence")
        messages = history.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
            return unavailable("Source H does not contain a supported recorded message sequence")
        record["source_messages_sha256"] = _digest(messages)
        events = None
        if policy == PRIMARY_WINDOW_POLICY:
            annotations = [store.get_record("annotation", item["record_id"])
                for item in store.list_records("annotation", run_id=source.run_id)]
            stall = first_stall_annotation(annotations)
            if stall is None or stall.get("checkpoint_id") != source.checkpoint_id:
                return unavailable("Source is not the recorded first online S1; no primary detector window")
            if (stall.get("run_id") != source.run_id or stall.get("scope") != to_jsonable(source.scope)
                    or type(stall.get("evidence_cutoff")) is not int or stall["evidence_cutoff"] < source.event_sequence):
                return unavailable("Recorded S1 annotation does not bind the selected source scope/cut")
            evidence = stall.get("evidence") or {}
            if stall.get("rule_version") in STALL_CLOCK_RULE_VERSIONS:
                # Stall-clock S records carry no anchor field; the anchor is the checkpoint of the last recorded
                # progress (or scope-baseline) decision in the same detector scope, strictly before the S cut.
                anchor_record = _stall_clock_anchor(store, source, stall)
                if anchor_record is None:
                    return unavailable("Recorded stall-clock progress anchor is unavailable")
            else:
                anchor_record = evidence.get("anchor")
                if not isinstance(anchor_record, Mapping) or not anchor_record.get("checkpoint_id"):
                    return unavailable("Recorded S1 detector anchor is unavailable")
                effective = (evidence.get("stagnation") or {}).get("effective") or {}
                if effective.get("anchor_checkpoint_id") != anchor_record["checkpoint_id"]:
                    return unavailable("Recorded S1 detector anchor conflicts with its effective stagnation window")
            anchor_id = anchor_record["checkpoint_id"]
            anchor = store.get_checkpoint(anchor_id)
            if (anchor.run_id != source.run_id or anchor.scope != source.scope
                    or anchor.ordinal >= source.ordinal or anchor.event_sequence >= source.event_sequence):
                return unavailable("Detector anchor is not strictly prior in the selected session/target scope")
            record.update(anchor_checkpoint_id=anchor_id, after_event_sequence=anchor.event_sequence,
                          detector_anchor=dict(anchor_record), stall_annotation_id=stall["annotation_id"],
                          stall_evidence_cutoff=stall["evidence_cutoff"],
                          detector_rule_version=stall["rule_version"], interval="(detector_anchor, S1]")
        elif policy == MANUAL_CHECKPOINT_WINDOW_POLICY:
            if any(type(record[name]) is not str or not record[name] for name in (
                    "after_checkpoint_id", "through_checkpoint_id", "decision_checkpoint_id")):
                return unavailable("Manual O window requires explicit after and decision checkpoint identities")
            after, through, decision = (store.get_checkpoint(record[name]) for name in (
                "after_checkpoint_id", "through_checkpoint_id", "decision_checkpoint_id"))
            if any(checkpoint.run_id != source.run_id or checkpoint.scope != source.scope
                   for checkpoint in (after, through, decision)):
                return unavailable("Manual O checkpoint selectors must share the selected H natural run/session/target scope")
            ordered = (after, through, source, decision)
            if any(a.ordinal > b.ordinal or a.event_sequence > b.event_sequence for a, b in zip(ordered, ordered[1:])):
                return unavailable("Manual O checkpoint order must be after <= through <= H <= decision; future or inverted range")
            events = list(store.iter_events(source.run_id, through_sequence=source.event_sequence))
            starts = [event.payload for event in events if event.kind == "lifecycle"
                      and event.payload.get("transition") == "run_started"]
            if (len(starts) != 1 or not {"resume_from", "treatment", "config"} <= set(starts[0])
                    or starts[0].get("resume_from") is not None or starts[0].get("treatment") is not None
                    or not isinstance(starts[0]["config"], Mapping)
                    or starts[0]["config"].get("run_id") != source.run_id):
                return unavailable("Manual O window requires one retained natural run start; resumed or unidentified sources are unavailable")
            record.update(after_event_sequence=after.event_sequence, through_event_sequence=through.event_sequence,
                          decision_event_sequence=decision.event_sequence,
                          checkpoint_ordinals={"after": after.ordinal, "through": through.ordinal,
                                               "history": source.ordinal, "decision": decision.ordinal},
                          interval="(manual_after_checkpoint, manual_through_checkpoint]",
                          cutoff_eligibility="same natural run/session/target evidence boundary; no W/R restoration asserted")
        else:
            record.update(after_event_sequence=-1, interval="all_retained_deliveries_in_selected_session_target")
        if events is None:
            events = list(store.iter_events(source.run_id, through_sequence=source.event_sequence))
        if any(event.run_id != source.run_id or event.sequence > source.event_sequence for event in events):
            return unavailable("Event store violated the source run/prefix binding")
        candidates = _delivery_candidates(events)
        selected, mapping, used = [], [], set()
        pending: dict[str, dict[str, Any]] = {}
        for index, message in enumerate(messages):
            if message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
                return unavailable("Source H contains an unsupported message role")
            if pending and message.get("role") not in {"assistant", "tool"}:
                return unavailable("Source H has an interrupted tool-call/result sequence")
            if message.get("role") == "assistant":
                if pending:
                    return unavailable("Source H has an incomplete tool-call/result sequence")
                for call in message.get("tool_calls") or ():
                    if not isinstance(call, Mapping) or type(call.get("id")) is not str or call["id"] in pending:
                        return unavailable("Source H has unsupported tool-call identities")
                    extra = message.get("extra") or {}
                    if not isinstance(extra, Mapping):
                        return unavailable("Source H has malformed assistant delivery provenance")
                    pending[call["id"]] = {"assistant_index": index, "model_call_id": extra.get("call_id")}
            if message.get("role") != "tool":
                continue
            call_id, content = message.get("tool_call_id"), message.get("content")
            if type(call_id) is not str or call_id not in pending or type(content) is not str:
                return unavailable("Retained tool message lacks supported call pairing or string result content")
            context = pending.pop(call_id)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            matches = [candidate for candidate in candidates if candidate["event"].call_id == call_id
                       and candidate["event"].delivered_ref.sha256 == content_hash]
            if context["model_call_id"] is not None:
                bound = [candidate for candidate in matches if candidate["model_call_id"] == context["model_call_id"]]
                # A legacy receipt without model-call linkage can still be
                # uniquely proven by exact current H membership.
                matches = bound if bound else (matches if all(candidate["model_call_id"] is None for candidate in matches) else [])
            if len(matches) != 1 or matches[0]["event"].event_id in used:
                return unavailable("Retained H tool message has missing or ambiguous delivery evidence")
            candidate, delivery = matches[0], matches[0]["event"]
            if delivery.scope is None:
                return unavailable("Retained H delivery has unavailable native session/target scope")
            if store.get_bytes(delivery.delivered_ref) != content.encode("utf-8"):
                return unavailable("Delivered artifact bytes differ from retained source H")
            used.add(delivery.event_id)
            in_scope = delivery.scope == source.scope
            in_window = record["after_event_sequence"] < delivery.sequence <= record["through_event_sequence"]
            item = {"message_index": index, "assistant_message_index": context["assistant_index"],
                    "tool_call_id": call_id, "delivery_event_id": delivery.event_id,
                    "delivery_sequence": delivery.sequence, "result_event_id": candidate["result_event_id"],
                    "delivered_ref": to_jsonable(delivery.delivered_ref), "message_sha256": _digest(message),
                    "mapping_basis": candidate["basis"], "in_selected_scope": in_scope, "in_window": in_window}
            mapping.append(item)
            if in_scope and in_window:
                selected.append(item)
        if pending:
            return unavailable("Source H ends with pending tool calls")
        record.update(selected_messages=selected, retained_tool_mapping=mapping,
            delivered_but_not_retained=[{"delivery_event_id": candidate["event"].event_id,
                "delivery_sequence": candidate["event"].sequence, "call_id": candidate["event"].call_id}
                for candidate in candidates if candidate["event"].event_id not in used
                and candidate["event"].scope == source.scope
                and record["after_event_sequence"] < candidate["event"].sequence <= record["through_event_sequence"]],
            status="available", reason="Recorded detector/delivery/H bindings verified" if policy == PRIMARY_WINDOW_POLICY else
                "Explicit manual checkpoint range; natural source/order and exact current H delivery mapping verified" if policy == MANUAL_CHECKPOINT_WINDOW_POLICY else
                "Explicit all-history policy; exact current H delivery mapping verified",
            effect="mask_retained_results" if selected else "no_retained_deliveries_in_selected_window")
        return _seal(record)
    except (KeyError, TypeError, ValueError) as exc:
        return unavailable(f"Observation mask source evidence unavailable: {type(exc).__name__}: {exc}")


def apply_observation_mask(messages: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    """Apply a verified selection to exact source H; retain call/assistant bytes."""
    if spec.get("revision") != REVISION or spec.get("policy_id") not in _POLICIES or spec.get("status") != "available":
        raise MaskError("Observation mask is unavailable or uses an unsupported policy")
    if spec.get("mask_id") != _digest({key: value for key, value in spec.items() if key != "mask_id"}):
        raise MaskError("Observation mask identity mismatch")
    if spec.get("placeholder") != MASK_PLACEHOLDER:
        raise MaskError("Observation mask requires the exact ratified placeholder")
    if _digest(messages) != spec.get("source_messages_sha256"):
        raise MaskError("Observation mask history differs from the bound source H")
    masked = deepcopy(list(messages))
    seen = set()
    for item in spec["selected_messages"]:
        index = item["message_index"]
        if type(index) is not int or index in seen or not 0 <= index < len(masked):
            raise MaskError("Observation mask has an invalid message selection")
        message = masked[index]
        if (message.get("role") != "tool" or message.get("tool_call_id") != item["tool_call_id"]
                or _digest(message) != item["message_sha256"]):
            raise MaskError("Observation mask selection differs from the bound tool result")
        # Exit flags, return codes, diagnostics and result metadata are O too.
        # Preserve only protocol pairing; calls, names and arguments remain in the
        # untouched assistant message. No outcome survives in extra fields.
        masked[index] = {key: deepcopy(message[key]) for key in ("role", "tool_call_id") if key in message}
        masked[index]["content"] = MASK_PLACEHOLDER
        seen.add(index)
    report = {"revision": REVISION, "policy_id": spec["policy_id"], "mask_id": spec["mask_id"],
                    "placeholder": MASK_PLACEHOLDER, "masked_tool_results": len(seen),
                    "selected_message_indexes": sorted(seen), "source_messages_sha256": spec["source_messages_sha256"],
                    "masked_messages_sha256": _digest(masked), "anchor_checkpoint_id": spec["anchor_checkpoint_id"],
                    "preserved": spec["preserve"]}
    if spec["policy_id"] == MANUAL_CHECKPOINT_WINDOW_POLICY:
        report.update({key: spec[key] for key in ("after_checkpoint_id", "through_checkpoint_id", "decision_checkpoint_id",
            "after_event_sequence", "through_event_sequence", "decision_event_sequence")})
    return masked, report


def validate_observation_mask(store: ArtifactStoreProtocol, source: Checkpoint,
                              spec: Mapping[str, Any], *, selection_kind: str) -> list[str]:
    try:
        expected = build_observation_mask(store, source, selection_kind=selection_kind, policy=spec.get("policy_id"),
            after_checkpoint_id=spec.get("after_checkpoint_id"), through_checkpoint_id=spec.get("through_checkpoint_id"),
            decision_checkpoint_id=spec.get("decision_checkpoint_id"))
    except MaskError as exc:
        return [str(exc)]
    if expected["status"] != "available":
        return ["Observation mask unavailable: " + expected["reason"]]
    if canonical_json(spec) != canonical_json(expected):
        return ["Observation mask differs from current immutable anchor/delivery/H evidence"]
    return []


def _stall_clock_anchor(store: Any, source: Any, stall: Mapping[str, Any]) -> dict[str, Any] | None:
    """Last recorded stall-clock progress/baseline decision before the S cut, in the S's detector scope."""
    cut = stall.get("evidence_cutoff")
    last = None
    for event in store.iter_events(source.run_id, through_sequence=cut):
        if event.kind != "detector" or event.payload.get("role") != "output":
            continue
        out = event.payload.get("output") or {}
        if out.get("record") != "detector_decision" or out.get("decision") not in ("progress", "scope_baseline"):
            continue
        if out.get("rule_version") not in STALL_CLOCK_RULE_VERSIONS or not out.get("checkpoint_id"):
            continue
        last = {"checkpoint_id": out["checkpoint_id"], "decision": out["decision"], "scope_id": out.get("scope_id"),
                "event_sequence": event.sequence, "source": "stall-clock detector_decision record"}
    return last
