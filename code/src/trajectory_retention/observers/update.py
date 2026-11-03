"""Incremental T update: validate extractor-proposed operations against the prefix.

The extractor model proposes operations as JSON. This module is the
provenance gate: nothing enters a T version unless every claim resolves to an
exact quote inside an eligible source field of the allowed interval. Quotes
are located deterministically and stored as spans with a hash of the quote.

Proposed operation shapes (extractor output, schema `t-ops/0-dev`):

  {"op": "add", "type": <ENTRY_TYPES>, "text": str,
   "source_category": "worker_expressed"|"observation_direct"|"extractor_inference",
   "quotes": [{"seq": int, "field": str, "quote": str, "occurrence": int?}],
   "scope": <SCOPES>, "depends_on": [entry_id...]?, "tmp_id": str?}

  {"op": "set_status", "entry_id": str, "status": <WORKER_STATUSES>,
   "quotes": [...]}

  {"op": "supersede", "entry_id": str, "by": entry_id|tmp_id, "quotes": [...]}

Rules enforced:
* worker_expressed: `text` must equal one quote exactly (exact-span policy);
* observation_direct: >=1 quote from tool_result/task_disclosure fields; text
  may be normalized but every number/identifier token in the text must appear
  in the quoted material (a cheap unsupported-content check);
* extractor_inference: never enters retention T; diverted to the advisor list;
* status/supersede changes need quotes from the current interval;
* statuses that strengthen certainty (worker_confirmed) need a worker_expressed
  quote, not an observation;
* every quote must be inside the interval (add) — inherited entries may be
  cited with allow_before_interval for dependency links only.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

from ._canonical import sha256_hex
from .prefix import OBSERVATION_DIRECT, WORKER_EXPRESSED, PrefixView, PrefixViolation, SESSION_LINEAGE_REVISION
from .tstore import (
    ENTRY_TYPES,
    APPLICABILITY,
    RETENTION_CATEGORIES,
    SCOPES,
    SOURCE_CATEGORIES,
    WORKER_STATUSES,
    OpResult,
    RejectedOp,
    SourceSpan,
    StatusChange,
    TEntry,
    TVersion,
    entry_id_for,
    make_span,
)

OPS_SCHEMA_VERSION = "t-ops/0-dev"
VALIDATION_REVISION = "retention-validation/3"
VALIDATION_SCOPE = {
    "structural_source_validation": VALIDATION_REVISION,
    "semantic_entailment": "not_measured",
    "status_and_type_interpretation_accuracy": "not_measured",
    "observation_normalization_check": "lexical_support_only_not_entailment",
    "scientific_quality_gate": "pending_calibration",
}

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+(?:\.\d+)?")
# Tokens that must be present verbatim in cited material: numbers, identifiers,
# paths, error names and constants — not ordinary lowercase English words.
_CHECKABLE = re.compile(r"\d|[_./:-]|^[A-Z]{2,}$|^.+[A-Z]")
_OBSERVATION_KINDS = frozenset({"tool_result", "observation_delivery", "task_disclosure", "format_error"})


def _resolve_quotes(
    view: PrefixView, quotes: Any, *, allow_before_interval: bool
) -> tuple[list[SourceSpan], list[str], list[str], list[str]]:
    """Return (spans, categories, kinds, errors)."""
    spans: list[SourceSpan] = []
    categories: list[str] = []
    kinds: list[str] = []
    errors: list[str] = []
    if not isinstance(quotes, list) or not quotes:
        return spans, categories, kinds, ["no quotes supplied"]
    for q in quotes:
        if not isinstance(q, Mapping):
            errors.append("quote is not an object")
            continue
        seq, field_name, quote = q.get("seq"), q.get("field"), q.get("quote")
        occurrence = q.get("occurrence", 0)
        if (type(seq) is not int or seq < 0 or not isinstance(field_name, str)
                or not isinstance(quote, str) or not quote
                or type(occurrence) is not int or occurrence < 0):
            errors.append(f"malformed quote {q!r}")
            continue
        try:
            text, category = view.resolve(seq, field_name, allow_before_interval=allow_before_interval)
        except PrefixViolation as exc:
            errors.append(str(exc))
            continue
        span = make_span(seq, field_name, text, quote, occurrence)
        if span is None:
            errors.append(f"quote not found verbatim in seq {seq} field {field_name!r}")
            continue
        spans.append(span)
        categories.append(category)
        kinds.append(view.source_event(seq).kind)
    return spans, categories, kinds, errors


def _quoted_text(view: PrefixView, span: SourceSpan) -> str:
    text, _ = view.resolve(span.seq, span.field, allow_before_interval=True)
    return text[span.start : span.end]


def _unsupported_tokens(text: str, quoted: Sequence[str]) -> list[str]:
    pool = " ".join(quoted)
    def present(token: str) -> bool:
        if token[0].isdigit():
            # A final sentence period is punctuation; a decimal/qualified
            # continuation is part of the token and cannot support a substring.
            return re.search(r"(?<![A-Za-z0-9_.])" + re.escape(token) + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])", pool) is not None
        # A name may legitimately be retained from a qualified path/error label.
        return token in pool
    return sorted({
        tok for tok in _TOKEN.findall(text)
        if _CHECKABLE.search(tok) and tok.strip(".:/-") and not present(tok)
    })


def _checkpoint_before(seq: int, checkpoint_cuts: Mapping[str, int]) -> str | None:
    best: tuple[int, str] | None = None
    for cid, cut in checkpoint_cuts.items():
        if cut < seq and (best is None or cut > best[0]):
            best = (cut, cid)
    return best[1] if best else None


def apply_ops(
    parent: TVersion | None,
    view: PrefixView,
    ops: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    update_index: int,
    checkpoint_cuts: Mapping[str, int],
    requirement_version: str | None,
) -> OpResult:
    """Validate and apply proposed ops. Never raises on bad ops; records rejections."""
    result = OpResult(entries=list(parent.entries) if parent else [])
    by_id: dict[str, TEntry] = {e.entry_id: e for e in result.entries}
    tmp_ids: dict[str, str] = {}
    ordinal = 0

    def lookup(ref: Any) -> TEntry | None:
        if not isinstance(ref, str):
            return None
        ref = tmp_ids.get(ref, ref)
        return by_id.get(ref)

    for op in ops:
        if not isinstance(op, Mapping) or "op" not in op:
            result.rejected.append(RejectedOp(op if isinstance(op, Mapping) else {"raw": repr(op)}, ("malformed op",)))
            continue
        name = op["op"]
        reasons: list[str] = []

        if name == "add":
            etype = op.get("type")
            category = op.get("source_category")
            text = op.get("text")
            scope = op.get("scope")
            if not isinstance(etype, str) or etype not in ENTRY_TYPES:
                reasons.append(f"unknown type {etype!r}")
            if not isinstance(category, str) or category not in SOURCE_CATEGORIES:
                reasons.append(f"unknown source_category {category!r}")
            if not isinstance(text, str) or not text.strip():
                reasons.append("empty text")
            if not isinstance(scope, str) or scope not in SCOPES:
                reasons.append(f"unknown scope {scope!r}")
            if category == "extractor_inference":
                # Advisor content: recorded separately, never in retention T.
                result.advisor.append(dict(op))
                result.applied.append({"op": "divert_advisor", "source": dict(op)})
                continue
            if reasons:
                result.rejected.append(RejectedOp(op, tuple(reasons)))
                continue
            spans, cats, kinds, errors = _resolve_quotes(view, op.get("quotes"), allow_before_interval=False)
            reasons.extend(errors)
            if not spans:
                reasons.append("no resolvable quote")
                result.rejected.append(RejectedOp(op, tuple(reasons)))
                continue
            quoted = [_quoted_text(view, s) for s in spans]
            exact = text in quoted
            if category == WORKER_EXPRESSED:
                if not all(c == WORKER_EXPRESSED for c in cats):
                    reasons.append("worker_expressed entry cites a non-worker source")
                if not exact:
                    reasons.append("worker_expressed text must equal one cited quote exactly")
            elif category == OBSERVATION_DIRECT:
                if not all(k in _OBSERVATION_KINDS and c == OBSERVATION_DIRECT for k, c in zip(kinds, cats)):
                    reasons.append("observation_direct entry must cite delivered tool/task content only")
                if etype not in {"verified_fact", "open_question"}:
                    reasons.append(f"observation_direct cannot introduce a {etype}")
                missing = _unsupported_tokens(text, quoted)
                if missing:
                    reasons.append(f"tokens absent from cited material: {missing[:5]}")
            deps: list[str] = []
            raw_deps = op.get("depends_on", [])
            if not isinstance(raw_deps, list) or not all(isinstance(d, str) for d in raw_deps):
                reasons.append("depends_on must be an array of entry IDs")
                raw_deps = []
            tmp_id = op.get("tmp_id")
            if tmp_id is not None and (not isinstance(tmp_id, str) or not tmp_id or tmp_id in tmp_ids or tmp_id in by_id):
                reasons.append("tmp_id must be a unique nonempty string")
            for d in raw_deps:
                dep = lookup(d)
                if dep is None:
                    reasons.append(f"unknown dependency {d!r}")
                else:
                    deps.append(dep.entry_id)
            if reasons:
                result.rejected.append(RejectedOp(op, tuple(reasons)))
                continue
            first_seq = min(s.seq for s in spans)
            first_event = view.source_event(first_seq)
            entry_namespace = run_id
            if view.scope.lineage_revision == SESSION_LINEAGE_REVISION:
                entry_namespace += ":native-session:" + str(first_event.session_id)
            entry_id = entry_id_for(entry_namespace, update_index, ordinal)
            ordinal += 1
            entry = TEntry(
                entry_id=entry_id,
                type=etype,
                text=text,
                text_is_exact_quote=exact,
                source_category=category,
                spans=tuple(spans),
                scope=scope,
                worker_status="observed" if category == OBSERVATION_DIRECT else "stated",
                first_source_seq=first_seq,
                stage_id=first_event.stage_id,
                session_id=first_event.session_id,
                requirement_version=requirement_version,
                observed_after_checkpoint_id=_checkpoint_before(first_seq, checkpoint_cuts),
                introduced_update_index=update_index,
                depends_on=tuple(deps),
            )
            result.entries.append(entry)
            by_id[entry_id] = entry
            if isinstance(op.get("tmp_id"), str):
                tmp_ids[op["tmp_id"]] = entry_id
            result.applied.append({"op": "add", "entry_id": entry_id, "source": dict(op)})

        elif name == "set_status":
            entry = lookup(op.get("entry_id"))
            status = op.get("status")
            if entry is None:
                reasons.append(f"unknown entry {op.get('entry_id')!r}")
            if not isinstance(status, str) or status not in WORKER_STATUSES:
                reasons.append(f"unknown status {status!r}")
            spans, cats, kinds, errors = _resolve_quotes(view, op.get("quotes"), allow_before_interval=False)
            reasons.extend(errors)
            required_category = OBSERVATION_DIRECT if status == "observed" else WORKER_EXPRESSED
            if not cats or not all(c == required_category for c in cats):
                reasons.append(f"{status} requires only {required_category} quotes")
            if entry is not None:
                if spans and min(s.seq for s in spans) < entry.first_source_seq:
                    reasons.append("status evidence precedes entry evidence")
                if status in ("observed", "stated") and status != (
                    "observed" if entry.source_category == OBSERVATION_DIRECT else "stated"
                ):
                    reasons.append("status cannot reinterpret the entry's original source category")
            if reasons or not spans:
                result.rejected.append(RejectedOp(op, tuple(reasons or ("no resolvable quote",))))
                continue
            assert entry is not None
            updated = replace(
                entry,
                worker_status=status,
                status_history=entry.status_history + (StatusChange(status, tuple(spans), update_index),),
            )
            _replace_entry(result.entries, by_id, updated)
            result.applied.append({"op": "set_status", "entry_id": entry.entry_id, "status": status,
                                   "spans": [s.as_record() for s in spans]})

        elif name == "supersede":
            entry = lookup(op.get("entry_id"))
            successor = lookup(op.get("by"))
            if entry is None:
                reasons.append(f"unknown entry {op.get('entry_id')!r}")
            if successor is None:
                reasons.append(f"unknown successor {op.get('by')!r}")
            elif entry is not None and successor.entry_id == entry.entry_id:
                reasons.append("entry cannot supersede itself")
            elif entry is not None:
                if successor.first_source_seq < entry.first_source_seq:
                    reasons.append("successor evidence precedes entry evidence")
                cursor = successor
                visited = set()
                while cursor is not None and cursor.entry_id not in visited:
                    if cursor.entry_id == entry.entry_id:
                        reasons.append("supersession cycle")
                        break
                    visited.add(cursor.entry_id)
                    cursor = lookup(cursor.superseded_by)
            spans, _, _, errors = _resolve_quotes(view, op.get("quotes"), allow_before_interval=False)
            reasons.extend(errors)
            if successor is not None and spans and min(s.seq for s in spans) < successor.first_source_seq:
                reasons.append("supersession evidence precedes successor evidence")
            if reasons or not spans:
                result.rejected.append(RejectedOp(op, tuple(reasons or ("no resolvable quote",))))
                continue
            assert entry is not None and successor is not None
            updated = replace(entry, superseded_by=successor.entry_id, supersession_evidence=tuple(spans))
            _replace_entry(result.entries, by_id, updated)
            result.applied.append({"op": "supersede", "entry_id": entry.entry_id,
                                   "by": successor.entry_id, "spans": [s.as_record() for s in spans]})
        else:
            result.rejected.append(RejectedOp(op, (f"unknown op {name!r}",)))
    return result


def _replace_entry(entries: list[TEntry], by_id: dict[str, TEntry], updated: TEntry) -> None:
    for i, e in enumerate(entries):
        if e.entry_id == updated.entry_id:
            entries[i] = updated
            break
    by_id[updated.entry_id] = updated


def validate_version(version: TVersion, view: PrefixView) -> list[str]:
    """Revalidate every evidence span and structural binding, never entailment.

    Old artifacts remain readable. Historical supersession evidence absent from
    entry records is reported, rather than fabricated or repaired in place.
    """
    problems: list[str] = []
    if version.run_id != view.run_id:
        problems.append("version run does not match prefix view")
    if version.through_seq != view.through_seq:
        problems.append("version cut does not match prefix view")
    if version.after_seq > version.through_seq:
        problems.append("inverted version interval")
    if version.status not in {"valid", "valid_with_rejections", "failed", "missing"}:
        problems.append("unknown version status")
    context = version.source_visibility.get("context_preflight")
    if context is not None:
        if not isinstance(context, Mapping) or type(context.get("client_call_allowed")) is not bool:
            problems.append("invalid request context preflight evidence")
        elif not context["client_call_allowed"]:
            if version.status != "failed" or not version.failure:
                problems.append("blocked request context requires an explicit failed version")
            if version.call_record_ids or version.ops_applied or version.rejected_ops:
                problems.append("blocked request context cannot contain calls or newly processed ops")
    for seq in version.source_visibility.get("accessed_seqs", ()):
        if type(seq) is not int or seq > version.through_seq or seq not in view.events_by_seq or not view.source_event(seq).texts:
            problems.append(f"extractor input accessed seq {seq!r} without an eligible prefix source")
    ids = {e.entry_id for e in version.entries}
    if len(ids) != len(version.entries):
        problems.append("duplicate entry IDs")

    def spans_check(label: str, spans: Sequence[SourceSpan], *, category: str | None = None,
                    earliest: int | None = None, required: bool = True) -> list[str]:
        quoted = []
        if required and not spans:
            problems.append(f"{label}: no spans")
        for span in spans:
            if (type(span.seq) is not int or type(span.start) is not int or type(span.end) is not int
                    or span.seq < 0 or span.start < 0 or span.end <= span.start):
                problems.append(f"{label}: invalid span bounds")
                continue
            if span.seq > version.through_seq:
                problems.append(f"{label}: span seq {span.seq} after cut {version.through_seq}")
                continue
            if earliest is not None and span.seq < earliest:
                problems.append(f"{label}: evidence precedes the content it updates")
            try:
                text, actual_category = view.resolve(span.seq, span.field, allow_before_interval=True)
            except PrefixViolation as exc:
                problems.append(f"{label}: {exc}")
                continue
            if span.end > len(text):
                problems.append(f"{label}: span extends beyond source text")
                continue
            quote = text[span.start:span.end]
            quoted.append(quote)
            if sha256_hex(quote) != span.quote_sha256:
                problems.append(f"{label}: quote hash mismatch at seq {span.seq}")
            if category is not None and actual_category != category:
                problems.append(f"{label}: {category} evidence cites {actual_category}")
        return quoted

    for e in version.entries:
        if e.type not in ENTRY_TYPES or e.scope not in SCOPES or e.worker_status not in WORKER_STATUSES:
            problems.append(f"{e.entry_id}: invalid type, scope or status")
        if not e.text.strip():
            problems.append(f"{e.entry_id}: empty text")
        if version.role in {"t_retention", "t_audit_fullprefix"} and e.source_category not in RETENTION_CATEGORIES:
            problems.append(f"{e.entry_id}: {e.source_category} inside {version.role}")
        quotes = spans_check(e.entry_id, e.spans, category=e.source_category)
        if e.source_category == WORKER_EXPRESSED and (not e.text_is_exact_quote or e.text not in quotes):
            problems.append(f"{e.entry_id}: worker text must be an exact source quote")
        if e.text_is_exact_quote and e.text not in quotes:
            problems.append(f"{e.entry_id}: exact-quote flag but text differs from every span")
        if e.source_category == OBSERVATION_DIRECT:
            if e.type not in {"verified_fact", "open_question"}:
                problems.append(f"{e.entry_id}: observation cannot introduce {e.type}")
            if _unsupported_tokens(e.text, quotes):
                problems.append(f"{e.entry_id}: tokens absent from cited material")
        if e.spans and e.first_source_seq != min(s.seq for s in e.spans):
            problems.append(f"{e.entry_id}: first_source_seq disagrees with spans")
        if e.first_source_seq in view.events_by_seq:
            event = view.source_event(e.first_source_seq)
            if (e.stage_id, e.session_id) != (event.stage_id, event.session_id):
                problems.append(f"{e.entry_id}: original stage/session differs from source")
        if not 0 <= e.introduced_update_index <= version.update_index:
            problems.append(f"{e.entry_id}: introduction outside version lineage")
        prior_update = e.introduced_update_index
        for change in e.status_history:
            label = f"{e.entry_id}: status {change.status}"
            if change.status not in WORKER_STATUSES:
                problems.append(f"{label}: unknown status")
            if not prior_update <= change.update_index <= version.update_index:
                problems.append(f"{label}: update index outside ordered lineage")
            prior_update = change.update_index
            required_category = OBSERVATION_DIRECT if change.status == "observed" else WORKER_EXPRESSED
            spans_check(label, change.evidence, category=required_category, earliest=e.first_source_seq)
        initial_status = "observed" if e.source_category == OBSERVATION_DIRECT else "stated"
        expected_status = e.status_history[-1].status if e.status_history else initial_status
        if expected_status != e.worker_status:
            problems.append(f"{e.entry_id}: status does not match evidence history")
        for assessment in e.applicability:
            label = f"{e.entry_id}: applicability"
            if assessment.value not in APPLICABILITY or not assessment.target_checkpoint_id or not assessment.method:
                problems.append(f"{label}: invalid assessment metadata")
            spans_check(label, assessment.evidence, earliest=e.first_source_seq,
                        required=assessment.value != "unknown")
        for d in e.depends_on:
            if d not in ids or d == e.entry_id:
                problems.append(f"{e.entry_id}: dangling or self dependency {d}")
        if e.superseded_by is not None:
            if e.superseded_by not in ids or e.superseded_by == e.entry_id:
                problems.append(f"{e.entry_id}: dangling or self supersession {e.superseded_by}")
            evidence = e.supersession_evidence
            if not evidence:
                # Legacy versions kept only the current update's operations.
                evidence = tuple(SourceSpan.from_record(s) for op in version.ops_applied
                                 if op.get("op") == "supersede" and op.get("entry_id") == e.entry_id
                                 for s in op.get("spans", ()))
            spans_check(f"{e.entry_id}: supersession", evidence, earliest=e.first_source_seq)
        elif e.supersession_evidence:
            problems.append(f"{e.entry_id}: supersession evidence without successor")
    # Applied status/supersession records can also be altered independently of entries.
    for op in version.ops_applied:
        if op.get("op") in {"set_status", "supersede"}:
            try:
                spans = tuple(SourceSpan.from_record(s) for s in op.get("spans", ()))
                spans_check(f"applied {op.get('op')}", spans, earliest=version.after_seq + 1)
            except (KeyError, TypeError, ValueError):
                problems.append("malformed applied operation evidence")
    by_id = {e.entry_id: e for e in version.entries}
    for relation in ("depends_on", "superseded_by"):
        def cyclic(node: str, active: set[str], done: set[str]) -> bool:
            if node in active:
                return True
            if node in done or node not in by_id:
                return False
            active.add(node)
            raw = getattr(by_id[node], relation)
            children = raw if relation == "depends_on" else (() if raw is None else (raw,))
            found = any(cyclic(child, active, done) for child in children)
            active.remove(node)
            done.add(node)
            return found
        if any(cyclic(node, set(), set()) for node in by_id):
            problems.append(f"{relation} cycle")
    return problems
