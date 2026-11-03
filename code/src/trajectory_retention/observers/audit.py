"""Audits: provenance re-validation, primary-vs-full-prefix comparison, repeat drift.

Findings are records, not corrections. Nothing here writes into the primary
incremental chain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contracts import ArtifactStoreProtocol
from ._canonical import content_id
from .prefix import PrefixView, SourceScope, view_for_checkpoint
from .tstore import TEntry, TVersion
from .update import VALIDATION_SCOPE, validate_version

AUDIT_REVISION = "audit/dev-2026-09-15-v2-all-evidence"

_WORD = re.compile(r"[a-z0-9_]+")


def _key(entry: TEntry) -> frozenset[tuple[int, str, int, int]]:
    return frozenset((s.seq, s.field, s.start, s.end) for s in entry.spans)


def _overlap(a: TEntry, b: TEntry) -> bool:
    """Same source event and overlapping character range on any span."""
    for sa in a.spans:
        for sb in b.spans:
            if sa.seq == sb.seq and sa.field == sb.field and sa.start < sb.end and sb.start < sa.end:
                return True
    return False


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


@dataclass(frozen=True)
class Match:
    primary_id: str
    reference_id: str
    kind: str  # exact_span / overlapping_span
    same_type: bool
    same_status: bool
    text_similarity: float


@dataclass(frozen=True)
class AuditReport:
    run_id: str
    target_checkpoint_id: str
    primary_version_id: str
    reference_version_id: str
    primary_problems: tuple[str, ...]
    reference_problems: tuple[str, ...]
    matches: tuple[Match, ...]
    omitted_in_primary: tuple[str, ...]  # reference entries with no primary counterpart
    unmatched_in_primary: tuple[str, ...]  # primary entries the reference did not produce
    type_disagreements: int
    status_disagreements: int
    supersession_disagreements: tuple[str, ...]
    primary_rejections: int
    counts: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "revision": AUDIT_REVISION,
            "run_id": self.run_id,
            "target_checkpoint_id": self.target_checkpoint_id,
            "primary_version_id": self.primary_version_id,
            "reference_version_id": self.reference_version_id,
            "primary_problems": list(self.primary_problems),
            "reference_problems": list(self.reference_problems),
            "matches": [m.__dict__ for m in self.matches],
            "omitted_in_primary": list(self.omitted_in_primary),
            "unmatched_in_primary": list(self.unmatched_in_primary),
            "type_disagreements": self.type_disagreements,
            "status_disagreements": self.status_disagreements,
            "supersession_disagreements": list(self.supersession_disagreements),
            "primary_rejections": self.primary_rejections,
            "counts": dict(self.counts),
            "validation_scope": dict(VALIDATION_SCOPE),
            "agreement_interpretation": "source-span and lexical agreement, not correctness or entailment",
        }


def compare_versions(primary: TVersion, reference: TVersion, view: PrefixView) -> AuditReport:
    if primary.run_id != reference.run_id or primary.run_id != view.run_id:
        raise ValueError("audit requires the same run")
    if primary.target_checkpoint_id != reference.target_checkpoint_id:
        raise ValueError("audit requires the same target checkpoint")
    if primary.through_seq != reference.through_seq:
        raise ValueError("audit requires the same prefix cut")
    if primary.source_visibility.get("scope") != reference.source_visibility.get("scope"):
        raise ValueError("audit requires the same permitted source scope")
    p_problems = validate_version(primary, view)
    r_problems = validate_version(reference, view)

    matches: list[Match] = []
    used_ref: set[str] = set()
    for pe in primary.entries:
        best: tuple[int, TEntry] | None = None
        for re_ in reference.entries:
            if re_.entry_id in used_ref:
                continue
            if _key(pe) == _key(re_):
                best = (2, re_)
                break
            if _overlap(pe, re_) and (best is None or best[0] < 1):
                best = (1, re_)
        if best is None:
            continue
        rank, re_ = best
        used_ref.add(re_.entry_id)
        matches.append(Match(
            primary_id=pe.entry_id, reference_id=re_.entry_id,
            kind="exact_span" if rank == 2 else "overlapping_span",
            same_type=pe.type == re_.type, same_status=pe.worker_status == re_.worker_status,
            text_similarity=round(_jaccard(pe.text, re_.text), 3),
        ))
    matched_primary = {m.primary_id for m in matches}
    omitted = [e.entry_id for e in reference.entries if e.entry_id not in used_ref]
    unmatched = [e.entry_id for e in primary.entries if e.entry_id not in matched_primary]

    sup_dis: list[str] = []
    matched_ids = {m.primary_id: m.reference_id for m in matches}
    for m in matches:
        pe, re_ = primary.entry(m.primary_id), reference.entry(m.reference_id)
        if ((pe.superseded_by is None) != (re_.superseded_by is None)
                or (pe.superseded_by is not None and matched_ids.get(pe.superseded_by) != re_.superseded_by)):
            sup_dis.append(m.primary_id)

    def counts(v: TVersion) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in v.entries:
            out[e.type] = out.get(e.type, 0) + 1
        return out

    return AuditReport(
        run_id=primary.run_id, target_checkpoint_id=primary.target_checkpoint_id,
        primary_version_id=primary.version_id(), reference_version_id=reference.version_id(),
        primary_problems=tuple(p_problems), reference_problems=tuple(r_problems),
        matches=tuple(matches), omitted_in_primary=tuple(omitted), unmatched_in_primary=tuple(unmatched),
        type_disagreements=sum(not m.same_type for m in matches),
        status_disagreements=sum(not m.same_status for m in matches),
        supersession_disagreements=tuple(sup_dis), primary_rejections=len(primary.rejected_ops),
        counts={"primary": counts(primary), "reference": counts(reference),
                "primary_total": len(primary.entries), "reference_total": len(reference.entries)},
    )


def repeat_agreement(repeats: Sequence[TVersion]) -> dict[str, Any]:
    """Pairwise span-set agreement across repeated independent extractions at one checkpoint."""
    if len(repeats) < 2:
        return {"pairs": 0, "note": "need >=2 repeats"}
    keys = {(v.run_id, v.target_checkpoint_id, v.through_seq,
             content_id(v.source_visibility.get("scope")), v.extractor_config_id) for v in repeats}
    if len(keys) != 1:
        raise ValueError("repeat agreement requires matched run/checkpoint/cut/scope/config")
    sets = [{_key(e) for e in v.entries} for v in repeats]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            pairs.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return {"pairs": len(pairs), "span_jaccard_mean": round(sum(pairs) / len(pairs), 3),
            "span_jaccard_min": round(min(pairs), 3), "entry_counts": [len(s) for s in sets],
            "distinct_version_occurrences": len({v.version_id() for v in repeats}),
            "duplicate_versions_supplied": len({v.version_id() for v in repeats}) != len(repeats),
            "quality_interpretation": "span agreement only; independent full-prefix T is not ground truth",
            "scientific_quality_gate": "pending_calibration"}


def audit_recorded_chain(store: ArtifactStoreProtocol, version_ids: Sequence[str]) -> dict[str, Any]:
    """Deterministically audit saved versions and propagate input exposure through ancestry.

    Supply a contiguous chain from its root. Reports only metadata and source
    IDs, with no prompts, tool contents, grading data or model calls. Incomplete
    ancestry is itself a failing provenance check, never assumed clean.
    """
    rows = []
    parent: TVersion | None = None
    inherited_exposure: set[int] = set()
    inherited_problems = False
    extraction_gaps: list[str] = []
    for version_id in version_ids:
        version = TVersion.from_record(store.get_record("t_version", version_id))
        checkpoint = store.get_checkpoint(version.target_checkpoint_id)
        raw_scope = version.source_visibility.get("scope")
        problems = []
        if not isinstance(raw_scope, Mapping):
            problems.append("missing source scope")
            raw_scope = SourceScope().as_record()
        scope = SourceScope(**raw_scope)
        view = view_for_checkpoint(store, version.run_id, through_seq=version.through_seq,
                                   target_session_id=checkpoint.scope.session_id, scope=scope)
        problems.extend(validate_version(version, view))
        if version.version_id() != version_id:
            problems.append("stored version identity mismatch")
        if checkpoint.run_id != version.run_id or checkpoint.event_sequence != version.through_seq:
            problems.append("checkpoint run/cut mismatch")
        if version.role != "t_retention":
            problems.append("chain version is not retention T")
        if parent is None:
            if version.parent_version_id is not None or version.update_index != 0 or version.after_seq != -1:
                problems.append("incomplete ancestry: supplied chain does not begin at root")
        else:
            if (version.parent_version_id != parent.version_id() or version.run_id != parent.run_id
                    or version.after_seq != parent.through_seq or version.update_index != parent.update_index + 1):
                problems.append("noncontiguous or mismatched ancestry")
            if (version.extractor_config_id != parent.extractor_config_id
                    or version.source_visibility.get("scope") != parent.source_visibility.get("scope")):
                problems.append("extractor configuration/source scope changed within chain")
        exposed = {seq for seq in version.source_visibility.get("accessed_seqs", ())
                   if type(seq) is int and (seq > version.through_seq or seq not in view.events_by_seq
                                          or not view.source_event(seq).texts)}
        inherited_exposure.update(exposed)
        inherited_problems = inherited_problems or bool(problems)
        if version.status in {"failed", "missing"}:
            extraction_gaps.append(version_id)
        rows.append({
            "version_id": version_id, "checkpoint_id": version.target_checkpoint_id,
            "update_index": version.update_index, "status": version.status,
            "parent_version_id": version.parent_version_id, "entry_count": len(version.entries),
            "direct_problems": problems,
            "direct_ineligible_input_seqs": sorted(exposed),
            "inherited_ineligible_input_seqs": sorted(inherited_exposure),
            "prefix_and_ancestry_valid": not inherited_problems,
            "extraction_gap_version_ids": list(extraction_gaps),
            "quality_gate": "pending_calibration",
        })
        parent = version
    return {"revision": AUDIT_REVISION, "validation_scope": dict(VALIDATION_SCOPE),
            "versions": rows, "version_count": len(rows),
            "prefix_and_ancestry_valid": bool(rows) and all(row["prefix_and_ancestry_valid"] for row in rows),
            "direct_input_exposure_versions": sum(bool(row["direct_ineligible_input_seqs"]) for row in rows),
            "input_exposure_affected_versions": sum(bool(row["inherited_ineligible_input_seqs"]) for row in rows),
            "failed_or_missing_version_ids": extraction_gaps,
            "note": "Source and lineage checks do not measure semantic extraction correctness."}


def audit_recorded_ancestry(store: ArtifactStoreProtocol, version_id: str) -> dict[str, Any]:
    """Load the complete immutable ancestry for a treatment's selected version."""
    reverse_ids: list[str] = []
    seen: set[str] = set()
    current: str | None = version_id
    while current is not None:
        if current in seen:
            raise ValueError("T parent-version cycle")
        seen.add(current)
        reverse_ids.append(current)
        version = TVersion.from_record(store.get_record("t_version", current))
        current = version.parent_version_id
    return audit_recorded_chain(store, list(reversed(reverse_ids)))
