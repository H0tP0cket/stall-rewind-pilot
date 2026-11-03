"""Typed trajectory representation T: entries, update operations and versions.

Development schema `t-content/0-dev`. It is the observer session's proposed
content payload; the shared store owns the envelope (artifact IDs, storage,
accounting). Nothing here is frozen: types, statuses and fields freeze
through calibration.

Design rules implemented here (state-recording.md, Q11/Q14/Q18):

* one store with typed entries: verified_fact, hypothesis, plan_step, scratch,
  lesson, open_question;
* every entry cites exact source spans (seq, field, char offsets, quote hash)
  resolved against immutable source text; normalized text never replaces the
  source;
* source_category separates worker_expressed and observation_direct content
  (primary retention) from extractor_inference (advisor-only);
* worker statement status is separate from applicability at a target
  checkpoint; applicability defaults to unknown;
* versions are immutable and append-only: add, set_status and supersede
  operations; nothing is deleted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..ledger.text import locate_quote  # noqa: F401
from ._canonical import content_id, sha256_hex

SCHEMA_VERSION = "t-content/0-dev"

ENTRY_TYPES = (
    "verified_fact",
    "hypothesis",
    "plan_step",
    "scratch",
    "lesson",
    "open_question",
)
SOURCE_CATEGORIES = ("worker_expressed", "observation_direct", "extractor_inference")
RETENTION_CATEGORIES = frozenset({"worker_expressed", "observation_direct"})
SCOPES = ("task_requirement", "implementation_behavior", "runtime_behavior", "worker_strategy")
# Worker-statement / evidence status. Changes require new cited evidence.
WORKER_STATUSES = (
    "observed",  # direct observation fact
    "stated",  # worker expressed it, no later statement about it
    "worker_confirmed",  # worker later explicitly said it holds / is done
    "worker_rejected",  # worker later explicitly said it is wrong
    "worker_abandoned",  # worker explicitly dropped the plan/question
)
APPLICABILITY = ("supported", "unknown", "inapplicable", "superseded")


@dataclass(frozen=True)
class SourceSpan:
    seq: int
    field: str
    start: int
    end: int
    quote_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "field": self.field,
            "start": self.start,
            "end": self.end,
            "quote_sha256": self.quote_sha256,
        }

    @staticmethod
    def from_record(rec: Mapping[str, Any]) -> "SourceSpan":
        return SourceSpan(
            int(rec["seq"]), str(rec["field"]), int(rec["start"]), int(rec["end"]),
            str(rec["quote_sha256"]),
        )


def make_span(seq: int, field_name: str, text: str, quote: str, occurrence: int = 0) -> SourceSpan | None:
    located = locate_quote(text, quote, occurrence)
    if located is None:
        return None
    return SourceSpan(seq, field_name, located[0], located[1], sha256_hex(quote))


@dataclass(frozen=True)
class Applicability:
    target_checkpoint_id: str
    value: str
    method: str
    evidence: tuple[SourceSpan, ...] = ()
    recorded_in_version: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "target_checkpoint_id": self.target_checkpoint_id,
            "value": self.value,
            "method": self.method,
            "evidence": [s.as_record() for s in self.evidence],
            "recorded_in_version": self.recorded_in_version,
        }


@dataclass(frozen=True)
class StatusChange:
    status: str
    evidence: tuple[SourceSpan, ...]
    update_index: int

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "evidence": [s.as_record() for s in self.evidence],
            "update_index": self.update_index,
        }


@dataclass(frozen=True)
class TEntry:
    entry_id: str
    type: str
    text: str
    text_is_exact_quote: bool
    source_category: str
    spans: tuple[SourceSpan, ...]
    scope: str
    worker_status: str
    first_source_seq: int
    stage_id: str | None
    session_id: str | None
    requirement_version: str | None
    # Latest checkpoint whose event cut precedes the first source event: the
    # W/R state the source content was about (observed after that boundary).
    observed_after_checkpoint_id: str | None
    introduced_update_index: int
    depends_on: tuple[str, ...] = ()
    superseded_by: str | None = None
    status_history: tuple[StatusChange, ...] = ()
    applicability: tuple[Applicability, ...] = ()
    # Added by validation revision v2. Omit when absent to preserve old hashes.
    supersession_evidence: tuple[SourceSpan, ...] = ()

    def as_record(self) -> dict[str, Any]:
        record = {
            "entry_id": self.entry_id,
            "type": self.type,
            "text": self.text,
            "text_is_exact_quote": self.text_is_exact_quote,
            "source_category": self.source_category,
            "spans": [s.as_record() for s in self.spans],
            "scope": self.scope,
            "worker_status": self.worker_status,
            "first_source_seq": self.first_source_seq,
            "stage_id": self.stage_id,
            "session_id": self.session_id,
            "requirement_version": self.requirement_version,
            "observed_after_checkpoint_id": self.observed_after_checkpoint_id,
            "introduced_update_index": self.introduced_update_index,
            "depends_on": list(self.depends_on),
            "superseded_by": self.superseded_by,
            "status_history": [c.as_record() for c in self.status_history],
            "applicability": [a.as_record() for a in self.applicability],
        }
        if self.supersession_evidence:
            record["supersession_evidence"] = [s.as_record() for s in self.supersession_evidence]
        return record

    @staticmethod
    def from_record(rec: Mapping[str, Any]) -> "TEntry":
        return TEntry(
            entry_id=rec["entry_id"],
            type=rec["type"],
            text=rec["text"],
            text_is_exact_quote=bool(rec["text_is_exact_quote"]),
            source_category=rec["source_category"],
            spans=tuple(SourceSpan.from_record(s) for s in rec["spans"]),
            scope=rec["scope"],
            worker_status=rec["worker_status"],
            first_source_seq=int(rec["first_source_seq"]),
            stage_id=rec.get("stage_id"),
            session_id=rec.get("session_id"),
            requirement_version=rec.get("requirement_version"),
            observed_after_checkpoint_id=rec.get("observed_after_checkpoint_id"),
            introduced_update_index=int(rec["introduced_update_index"]),
            depends_on=tuple(rec.get("depends_on") or ()),
            superseded_by=rec.get("superseded_by"),
            status_history=tuple(
                StatusChange(
                    c["status"], tuple(SourceSpan.from_record(s) for s in c["evidence"]),
                    int(c["update_index"]),
                )
                for c in rec.get("status_history") or ()
            ),
            applicability=tuple(
                Applicability(
                    a["target_checkpoint_id"], a["value"], a["method"],
                    tuple(SourceSpan.from_record(s) for s in a.get("evidence") or ()),
                    a.get("recorded_in_version"),
                )
                for a in rec.get("applicability") or ()
            ),
            supersession_evidence=tuple(SourceSpan.from_record(s) for s in rec.get("supersession_evidence") or ()),
        )


@dataclass(frozen=True)
class RejectedOp:
    op: Mapping[str, Any]
    reasons: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {"op": dict(self.op), "reasons": list(self.reasons)}


@dataclass(frozen=True)
class TVersion:
    """Immutable T content at one checkpoint.

    role: t_retention (primary incremental chain), t_audit_fullprefix
    (independent reference) or advisor_t (extractor inference only).
    status: valid, valid_with_rejections, failed, missing. A failed/missing
    version carries its parent's entries unchanged and must not be used for
    T-dependent treatments.
    """

    role: str
    run_id: str
    target_checkpoint_id: str
    update_index: int
    parent_version_id: str | None
    after_seq: int
    through_seq: int
    entries: tuple[TEntry, ...]
    ops_applied: tuple[Mapping[str, Any], ...]
    rejected_ops: tuple[RejectedOp, ...]
    status: str
    extractor_config_id: str | None
    call_record_ids: tuple[str, ...]
    source_visibility: Mapping[str, Any]
    failure: str | None = None
    notes: tuple[str, ...] = ()
    schema: str = SCHEMA_VERSION

    def content_record(self) -> dict[str, Any]:
        """Identity-bearing content (excludes call receipts so identical content shares a hash)."""
        return {
            "schema": self.schema,
            "role": self.role,
            "run_id": self.run_id,
            "entries": [e.as_record() for e in self.entries],
        }

    def content_hash(self) -> str:
        return content_id(self.content_record())

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": self.role,
            "run_id": self.run_id,
            "target_checkpoint_id": self.target_checkpoint_id,
            "update_index": self.update_index,
            "parent_version_id": self.parent_version_id,
            "interval": {"after_seq": self.after_seq, "through_seq": self.through_seq},
            "entries": [e.as_record() for e in self.entries],
            "ops_applied": [dict(o) for o in self.ops_applied],
            "rejected_ops": [r.as_record() for r in self.rejected_ops],
            "status": self.status,
            "extractor_config_id": self.extractor_config_id,
            "call_record_ids": list(self.call_record_ids),
            "source_visibility": dict(self.source_visibility),
            "failure": self.failure,
            "notes": list(self.notes),
            "content_hash": self.content_hash(),
        }

    def version_id(self) -> str:
        """Occurrence identity: distinct per checkpoint update even when content is unchanged."""
        rec = self.as_record()
        return content_id(rec)

    @staticmethod
    def from_record(rec: Mapping[str, Any]) -> "TVersion":
        version = TVersion(
            role=rec["role"],
            run_id=rec["run_id"],
            target_checkpoint_id=rec["target_checkpoint_id"],
            update_index=int(rec["update_index"]),
            parent_version_id=rec.get("parent_version_id"),
            after_seq=int(rec["interval"]["after_seq"]),
            through_seq=int(rec["interval"]["through_seq"]),
            entries=tuple(TEntry.from_record(e) for e in rec["entries"]),
            ops_applied=tuple(rec.get("ops_applied") or ()),
            rejected_ops=tuple(
                RejectedOp(r["op"], tuple(r["reasons"])) for r in rec.get("rejected_ops") or ()
            ),
            status=rec["status"],
            extractor_config_id=rec.get("extractor_config_id"),
            call_record_ids=tuple(rec.get("call_record_ids") or ()),
            source_visibility=rec.get("source_visibility") or {},
            failure=rec.get("failure"),
            notes=tuple(rec.get("notes") or ()),
            schema=rec.get("schema", SCHEMA_VERSION),
        )
        if "content_hash" in rec and rec["content_hash"] != version.content_hash():
            raise ValueError("T version content hash mismatch")
        return version

    def entry(self, entry_id: str) -> TEntry:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        raise KeyError(entry_id)


def entry_id_for(run_id: str, update_index: int, ordinal: int) -> str:
    return f"t:{run_id}:u{update_index}:e{ordinal}"


@dataclass
class OpResult:
    entries: list[TEntry] = field(default_factory=list)
    applied: list[Mapping[str, Any]] = field(default_factory=list)
    rejected: list[RejectedOp] = field(default_factory=list)
    advisor: list[Mapping[str, Any]] = field(default_factory=list)
