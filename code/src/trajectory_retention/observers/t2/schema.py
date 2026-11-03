"""T2 records: verbatim items, labels with probabilities, links and derived views.

T2 replaces free-form LLM extraction with deterministic segmentation of the
allowed prefix into verbatim candidate items. A classifier only assigns labels
to those items, so every item is an exact span of a worker-visible source by
construction. Labels keep their full probability vectors; a label below the
declared confidence threshold is recorded as ``unknown`` rather than guessed.

Nothing here is a frozen scientific setting. Thresholds and label sets are
development choices recorded with every artifact (see ``T2_VERSION``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

T2_VERSION = "t2/dev-2026-09-23-v1"

# Where the item's text came from. Deterministic from the source event/field.
SOURCE_KINDS = (
    "worker_text",        # assistant message content
    "worker_reasoning",   # plaintext reasoning returned by the provider and replayed to the worker
    "provider_summary",   # provider-written reasoning summary replayed to the worker
    "worker_action",      # tool-call command (the executed action)
    "worker_code_comment",  # comment lines the worker wrote inside tool-call arguments
    "observation",        # delivered tool output
    "format_error",       # delivered harness format-error message
)
WORKER_SOURCES = frozenset({"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"})
OBSERVATION_SOURCES = frozenset({"observation", "format_error"})

ROLES = ("evidence", "approach", "intended_action", "question", "takeaway", "working_note", "narration")
STANCES = ("asserted", "uncertain", "question", "not_a_claim")
PURSUIT_STATUSES = ("active", "withdrawn", "replaced", "unknown")
LINK_TIERS = ("structural", "worker_stated", "observer_inferred")
UNKNOWN = "unknown"

# Development confidence gates (not calibrated scientific thresholds).
CHOICE_THRESHOLD = 0.6
KEEP_YES, KEEP_NO = 0.6, 0.4
LINK_THRESHOLD = 0.6
STATUS_THRESHOLD = 0.7


@dataclass(frozen=True)
class Span:
    """Exact location of an item's text.

    ``decoder`` names the deterministic transform from the raw field text to the
    text the span indexes (``raw`` or a JSON field decoder). Re-applying the
    decoder to the recorded field reproduces ``text[start:end]`` exactly.
    """

    seq: int
    field: str
    decoder: str
    start: int
    end: int

    def as_record(self) -> dict[str, Any]:
        return {"seq": self.seq, "field": self.field, "decoder": self.decoder, "start": self.start, "end": self.end}


@dataclass
class Label:
    value: str
    probabilities: Mapping[str, float] | None
    method: str  # "deterministic:<rule>" or "jev:<model>"
    confidence: float | None = None

    def as_record(self) -> dict[str, Any]:
        return {"value": self.value, "probabilities": dict(self.probabilities) if self.probabilities else None,
                "method": self.method, "confidence": self.confidence}


@dataclass
class Item:
    item_id: str
    run_id: str
    session_id: str | None
    stage_id: str | None
    seq: int
    order: int  # position within the source event
    available_seq: int  # first event cut at which this item is an allowed source (>= seq)
    source_kind: str
    text: str
    span: Span
    tool_call_id: str | None = None
    call_group: str | None = None  # model_response call id that produced/received it
    task_copy: bool = False
    tags: tuple[str, ...] = ()
    labels: dict[str, Label] = field(default_factory=dict)

    def label(self, name: str) -> str:
        lab = self.labels.get(name)
        return lab.value if lab is not None else UNKNOWN

    def as_record(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "run_id": self.run_id, "session_id": self.session_id, "stage_id": self.stage_id,
            "seq": self.seq, "order": self.order, "available_seq": self.available_seq, "source_kind": self.source_kind, "text": self.text,
            "span": self.span.as_record(), "tool_call_id": self.tool_call_id, "call_group": self.call_group,
            "task_copy": self.task_copy, "tags": list(self.tags),
            "labels": {k: v.as_record() for k, v in self.labels.items()},
        }


@dataclass(frozen=True)
class Link:
    source_id: str      # later item
    target_id: str      # earlier item
    relation: str       # result_of, same_turn, refers_to, abandons, replaces, confirms
    tier: str
    probability: float | None
    evidence_seq: int   # the link becomes known at this event (never earlier than source)
    method: str

    def as_record(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "target_id": self.target_id, "relation": self.relation,
                "tier": self.tier, "probability": self.probability, "evidence_seq": self.evidence_seq,
                "method": self.method}
