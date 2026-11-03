"""Read-only natural suffix measurements, with explicit observation boundaries.

This exporter does not replay/select a new detector, manufacture an S or N, or
turn a natural suffix into a matched recovery rollout. Numerical horizons and
the optional near-budget threshold are caller parameters, not calibrated defaults.
Every input run remains in the denominator, including no-warning and partial runs.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping, Sequence

from ..accounting import CostLedger, CostRecord, summarize
from ..contracts import ArtifactStoreProtocol, Checkpoint, EventRecord, to_jsonable
from ..experiments.selection import FIRST_STALL_SELECTION_RULE, first_stall_annotation

VERSION = "natural-followup.v4"
NATURAL_ROLES = frozenset({"worker", "grading", "restore", "infrastructure"})
RESOURCE_AXES = frozenset({"generated_tokens", "cost_usd", "elapsed_seconds", "model_attempts", "tool_calls"})


def _scope_key(scope: Any) -> tuple[Any, ...]:
    data = to_jsonable(scope)
    return tuple(data.get(k) for k in ("task_id", "stage_id", "target_id", "session_id",
                                     "detector_scope_id", "requirement_version", "evaluator_version"))


def _event_ref(event: EventRecord | None) -> dict[str, Any] | None:
    return None if event is None else {"event_id": event.event_id, "sequence": event.sequence,
                                       "timestamp": event.timestamp}


def _elapsed(events: Mapping[int, EventRecord], start: int, end: int) -> float | None:
    if start not in events or end not in events:
        return None
    try:
        value = (datetime.fromisoformat(events[end].timestamp) -
                 datetime.fromisoformat(events[start].timestamp)).total_seconds()
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _resources(records: Sequence[CostRecord], events: Mapping[int, EventRecord],
               start: int, end: int) -> dict[str, Any]:
    """Resources in (start, end]; unplaced records cannot be assigned to an interval."""
    selected = [r for r in records if r.source_event_sequence is not None
                and start < r.source_event_sequence <= end]
    unplaced = [r for r in records if r.source_event_sequence is None]
    totals = summarize(selected)

    def measured(metric: str, relevant: Sequence[CostRecord]) -> dict[str, Any]:
        value = dict(totals[metric])
        value["unplaced_operations"] = len(relevant)
        field = {"worker_generated_tokens": "generated_tokens", "cost_usd": "cost_usd",
                 "operation_seconds": "elapsed_seconds"}[metric]
        measured_records = ([r for r in selected if r.role == "worker" and r.kind == "model"]
                            if metric == "worker_generated_tokens" else selected)
        ordered = sorted(measured_records, key=lambda r: (r.source_event_sequence, r.operation_id))
        first_missing = next((r.source_event_sequence for r in ordered if getattr(r.usage, field) is None), None)
        value["first_missing_sequence"] = first_missing
        value["known_before_first_missing"] = (sum(getattr(r.usage, field) for r in ordered
            if first_missing is not None and r.source_event_sequence < first_missing)
            if not relevant and first_missing is not None else None)
        if relevant:
            value.update(value=None, complete=False)
        return value

    workers = [r for r in unplaced if r.role == "worker" and r.kind == "model"]
    seconds = _elapsed(events, start, end)
    return {
        "event_interval": {"start_exclusive": start, "end_inclusive": end},
        "generated_tokens": measured("worker_generated_tokens", workers),
        "cost_usd": measured("cost_usd", unplaced),
        "operation_seconds": measured("operation_seconds", unplaced),
        "elapsed_seconds": {"value": seconds, "complete": seconds is not None,
                            "basis": "recorded_event_wall_timestamps; not summed operation durations"},
        "model_attempts": {"value": sum(r.role == "worker" and r.kind == "model" for r in selected)
                           if not workers else None,
                           "known_sum": sum(r.role == "worker" and r.kind == "model" for r in selected),
                           "complete": not workers, "unplaced_operations": len(workers)},
        "tool_calls": {"value": sum(r.kind == "tool" for r in selected)
                       if not any(r.kind == "tool" for r in unplaced) else None,
                       "known_sum": sum(r.kind == "tool" for r in selected),
                       "complete": not any(r.kind == "tool" for r in unplaced),
                       "unplaced_operations": sum(r.kind == "tool" for r in unplaced)},
        "non_ok_operations": sum(r.status != "ok" for r in selected),
        "roles": {role: summarize(r for r in selected if r.role == role)
                  for role in sorted({r.role for r in selected})},
        "operation_ids": [r.operation_id for r in selected],
        "unplaced_operation_ids": [r.operation_id for r in unplaced],
    }


def _grade(store: ArtifactStoreProtocol, cp: Checkpoint, events: Sequence[EventRecord]) -> dict[str, Any]:
    raw = store.get_json(cp.evaluation_ref) if cp.evaluation_ref else {}
    reasons = []
    if cp.evaluation_ref is None:
        reasons.append("ungraded")
    if cp.validation.get("evaluation_reused"):
        reasons.append("reused_evaluation_not_fresh_grade")
    if raw.get("valid") is not True:
        reasons.append("invalid_or_missing_grade")
    if raw.get("score") is not None and (type(raw["score"]) not in (int, float)
            or not math.isfinite(raw["score"]) or not 0 <= raw["score"] <= 1):
        reasons.append("invalid_score_type_or_range")
    if raw.get("full_solve") is not None and type(raw["full_solve"]) is not bool:
        reasons.append("invalid_solve_type")
    for field in ("task_id", "stage_id", "target_id", "evaluator_version"):
        if raw and raw.get(field) != getattr(cp.scope, field):
            reasons.append(f"grade_{field}_binding_mismatch")
    receipts = [e for e in events if e.kind == "grade" and e.payload.get("checkpoint_id") == cp.checkpoint_id
                and e.payload.get("evaluation_id") == raw.get("evaluation_id")]
    receipt = receipts[0] if len(receipts) == 1 else None
    if raw and receipt is None:
        reasons.append("grade_receipt_missing_or_ambiguous")
    return {"checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal, "event_sequence": cp.event_sequence,
            "scope": to_jsonable(cp.scope), "evaluation_ref": to_jsonable(cp.evaluation_ref),
            "evaluation_id": raw.get("evaluation_id"), "score": raw.get("score"),
            "grade_event": _event_ref(receipt),
            "available_at_sequence": receipt.sequence if receipt else None,
            "full_solve": raw.get("full_solve"), "valid": not reasons, "exclusions": reasons,
            "raw_score": raw.get("raw_score"), "raw_max_score": raw.get("raw_max_score"),
            "check_vector": raw.get("checks"), "reachable_milestones": raw.get("reachable_milestones")}


def _bound(events: Sequence[EventRecord], source: Checkpoint, start: int) -> tuple[int, str, EventRecord | None]:
    """Stop comparable observation before a native reset or changed target/session."""
    terminal_sequences = [e.sequence for e in events if e.kind == "terminal" and e.scope is not None
                          and _scope_key(e.scope) == _scope_key(source.scope)]
    for event in events:
        if event.sequence <= start:
            continue
        payload = event.payload
        if event.kind == "lifecycle" and payload.get("transition") == "stage_transition_planned":
            return event.sequence - 1, "native_stage_transition", event
        if event.scope is not None and _scope_key(event.scope) != _scope_key(source.scope):
            return event.sequence - 1, "scope_or_target_changed", event
        if event.kind == "detector" and payload.get("role") == "input":
            item = payload.get("input", {})
            if item.get("type") == "scope_start" and item.get("scope_id") != source.scope.detector_scope_id:
                return event.sequence - 1, "detector_scope_closed", event
            if item.get("type") == "terminal":
                # The durable terminal follows detector closure and carries
                # skipped/failed final measurement evidence. Keep it in view.
                if any(seq > event.sequence for seq in terminal_sequences):
                    continue
                return event.sequence, f"trajectory_terminal:{item.get('reason')}", event
        if event.kind == "terminal":
            return event.sequence, f"trajectory_terminal:{payload.get('reason')}", event
    return (events[-1].sequence if events else start), "recording_ends_without_terminal", None


def _remaining(allowance: Any, resources: Mapping[str, Any]) -> dict[str, Any]:
    valid_allowance = type(allowance) is int and allowance > 0
    used = resources["generated_tokens"]
    exact = valid_allowance and used["complete"]
    known = used["known_sum"]
    return {"nominal_allowance": allowance if valid_allowance else None,
            "known_generated_tokens": known, "usage_complete": used["complete"],
            "remaining_tokens": max(0, allowance - known) if exact else None,
            "remaining_tokens_upper_bound": max(0, allowance - known) if valid_allowance else None,
            "overshoot_tokens": max(0, known - allowance) if exact else None,
            "overshoot_tokens_lower_bound": max(0, known - allowance) if valid_allowance else None}


def _cost_source(record: CostRecord, by_seq: Mapping[int, EventRecord],
                 by_id: Mapping[str, EventRecord]) -> tuple[EventRecord | None, str]:
    """Resolve explicit cost links without guessing from neighboring events."""
    event_id, sequence = record.source_event_id, record.source_event_sequence
    if event_id is None and sequence is None:
        return None, "source_event_link_missing"
    if event_id is not None and event_id not in by_id:
        return None, "source_event_id_not_found"
    if sequence is not None and sequence not in by_seq:
        return None, "source_event_sequence_not_found"
    if event_id is not None and sequence is not None:
        if by_id[event_id].event_id != by_seq[sequence].event_id:
            return None, "source_event_identity_mismatch"
        return by_id[event_id], "recorded_source_event_id_and_sequence"
    if event_id is not None:
        return by_id[event_id], "recorded_source_event_id"
    return by_seq[sequence], "recorded_source_event_sequence"


def _stage_usage(records: Sequence[CostRecord], events: Sequence[EventRecord],
                 cost_refs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Partition the complete natural ledger by source-event scope, once per cost.

    Scope is an attribution recorded by the producer, not proof that an operation
    ran entirely inside a stage. In particular, transition operations retain
    their recorded old/new scope. Missing or conflicting links remain unassigned.
    No terminal cut is applied: cleanup and unplaced known charges still count.
    """
    by_seq, by_id = {e.sequence: e for e in events}, {e.event_id: e for e in events}
    refs = {r["record_id"]: r for r in cost_refs}
    records = sorted(records, key=lambda r: (r.source_event_sequence is None,
                                            r.source_event_sequence or 0, r.operation_id))
    scopes: dict[tuple[Any, ...], dict[str, Any]] = {}
    scoped_records: dict[tuple[Any, ...], list[CostRecord]] = {}
    for event in events:
        if event.scope is not None:
            key = _scope_key(event.scope)
            scopes.setdefault(key, {"scope": to_jsonable(event.scope), "source_events": []})
            scoped_records.setdefault(key, [])
    unassigned, operations = [], []
    reasons: dict[str, int] = {}
    for record in records:
        event, basis = _cost_source(record, by_seq, by_id)
        reason = basis if event is None else "source_event_unscoped" if event.scope is None else None
        if reason is None:
            key = _scope_key(event.scope)
            scoped_records[key].append(record)
            scopes[key]["source_events"].append(_event_ref(event))
        else:
            unassigned.append(record)
            reasons[reason] = reasons.get(reason, 0) + 1
        cost_ref = refs.get(f"{record.run_id}:{record.operation_id}")
        operations.append({"cost": to_jsonable(record), "cost_record": cost_ref,
            "ledger_origin": "persisted_cost_record" if cost_ref else "read_only_cost_ledger_derivation",
            "source_event": _event_ref(event),
            "recorded_scope": to_jsonable(event.scope) if event else None,
            "attribution_basis": basis if event is not None else None,
            "unassigned_reason": reason})

    def usage(items: Sequence[CostRecord]) -> dict[str, Any]:
        return {"totals": summarize(items),
                "roles": {role: summarize(r for r in items if r.role == role)
                          for role in sorted({r.role for r in items})},
                "operation_ids": [r.operation_id for r in items]}

    stages: dict[tuple[str, str], dict[str, Any]] = {}
    stage_records: dict[tuple[str, str], list[CostRecord]] = {}
    for key, scope_row in scopes.items():
        stage_key = key[:2]
        stage = stages.setdefault(stage_key, {"task_id": key[0], "stage_id": key[1],
                                              "recorded_scopes": []})
        items = scoped_records[key]
        stage_records.setdefault(stage_key, []).extend(items)
        # Several charges may share a source event, but each source is listed once.
        scope_row["source_events"] = list({e["event_id"]: e for e in scope_row["source_events"]}.values())
        stage["recorded_scopes"].append({**scope_row, **usage(items)})
    for key, stage in stages.items():
        stage.update(usage(stage_records[key]))
        stage["attribution_complete"] = not unassigned
        stage["elapsed_seconds"] = {"value": None, "complete": False,
            "basis": "stage_wall_time_not_recorded; operation_seconds sums attributed durations"}
    return {"version": "natural-stage-usage.v1", **usage(records),
            "stages": list(stages.values()),
            "unassigned": {**usage(unassigned), "reasons": reasons},
            "operations": operations, "attribution_complete": not unassigned,
            "basis": "exact recorded source-event scope; no neighboring-event or lifecycle-payload inference",
            "coverage": "all recorded natural-role ledger operations, including unplaced and post-terminal operations",
            "interpretation": "Stage totals partition attributed operations; nested scope totals are subtotals. "
                              "Unassigned operations may add unknown stage contributions. Metric completeness "
                              "describes only each listed set of operations. No per-stage allowance is assigned. "
                              "Whole-run elapsed time is not apportioned to stages."}


def _horizon(episode: dict[str, Any], axis: str, horizon: int | float) -> dict[str, Any]:
    coverage = episode["comparable_resources"][axis]
    fully_observed = coverage["complete"] and coverage["value"] >= horizon
    if not coverage["complete"] and coverage.get("known_before_first_missing") is not None:
        # An unknown increment may be zero. At equality its later grade may also
        # sit on the queried horizon; only strictly earlier horizons are placed.
        fully_observed = horizon < coverage["known_before_first_missing"]
    points = [p for p in episode["points"] if p["valid"] and p["resources_since_s"][axis]["complete"]
              and p["resources_since_s"][axis]["value"] <= horizon]
    missing_grade_points = [p for p in episode["points"] if not p["valid"] and p["evaluation_ref"] is not None
                            and (not p["resources_since_s"][axis]["complete"]
                                 or p["resources_since_s"][axis]["value"] <= horizon)]
    missing_quality_barriers = []
    for barrier in episode.get("quality_barriers", []):
        at = barrier["resources_since_s"][axis]
        before_missing = at.get("known_before_first_missing")
        if (at["complete"] and at["value"] <= horizon or not at["complete"]
                and not (before_missing is not None and horizon < before_missing)):
            missing_quality_barriers.append(barrier)
    # A later valid grade can recover current quality, but cannot establish
    # that no transient escape/solve occurred inside an unmeasured interval.
    negative_outcomes_observed = fully_observed and not missing_grade_points and not missing_quality_barriers
    n = episode["natural_escape"]
    escape_observed = (n is not None and n["resources_since_s"][axis]["complete"]
                       and n["resources_since_s"][axis]["value"] <= horizon)
    score_escape = any(p["score"] is not None and episode["best_score_before_s"] is not None
                       and p["score"] > episode["best_score_before_s"] for p in points)
    solved = any(p["full_solve"] is True for p in points)
    known_scores = [p["score"] for p in points if p["score"] is not None]
    baseline = episode["score_at_s"] if episode["source_grade_available_at_s"] else None
    scores_with_baseline = ([baseline] if baseline is not None else []) + known_scores
    cuts = [p for p in episode["points"] if p["resources_since_s"][axis]["complete"]
            and p["resources_since_s"][axis]["value"] <= horizon]
    current = baseline
    current_checkpoint = episode["source_checkpoint_id"] if baseline is not None else None
    quality_cuts = [(p["available_at_sequence"] or p["event_sequence"], p["score"] if p["valid"] else None,
                     p["checkpoint_id"] if p["valid"] and p["score"] is not None else None) for p in cuts]
    quality_cuts += [(b["event"]["sequence"], None, None) for b in episode.get("quality_barriers", [])
                    if b["resources_since_s"][axis]["complete"] and b["resources_since_s"][axis]["value"] <= horizon]
    for _, score, checkpoint_id in sorted(quality_cuts):
        # An invalid/ungraded boundary ends the earlier grade's supported
        # interval. A later fresh valid grade can establish a new interval.
        current, current_checkpoint = score, checkpoint_id
    if not fully_observed:
        current = current_checkpoint = None
    # These are observed event times, so success remains known beyond censoring.
    # The primary quality view is current score on actual measured resources;
    # best observed quality remains a separate historical summary.
    return {"run_id": episode["run_id"], "s_annotation_id": episode["s_annotation_id"],
            "source_checkpoint_id": episode["source_checkpoint_id"], "primary": episode["primary"],
            "technical_eligibility": episode["technical_eligibility"],
            "technical_eligibility_problems": episode["technical_eligibility_problems"],
            "resource_axis": axis, "horizon": horizon, "fully_observed_horizon": fully_observed,
            "outcome_observation_complete": negative_outcomes_observed,
            "unusable_grade_checkpoint_ids": [p["checkpoint_id"] for p in missing_grade_points],
            "unmeasured_outcome_event_ids": [b["event"]["event_id"] for b in missing_quality_barriers],
            "censored": not negative_outcomes_observed,
            "censoring_reason": ("grade_measurement_missing" if missing_grade_points or missing_quality_barriers else
                None if fully_observed else "resource_measurement_missing" if not coverage["complete"]
                else episode["comparable_end_reason"]),
            "escape_by": True if escape_observed else (False if negative_outcomes_observed else None),
            "new_best_by": True if score_escape else (False if negative_outcomes_observed else None),
            "solved_by": True if solved else (False if negative_outcomes_observed else None),
            "current_score_at_horizon": current,
            "current_score_checkpoint_id": current_checkpoint,
            "quality_curve": "actual_compute_current_score.v1",
            "current_score_status": ("observed_current_grade" if current is not None else
                "grade_measurement_missing" if fully_observed else "resource_measurement_missing" if not coverage["complete"]
                else "stopped_before_horizon"),
            "best_observed_score_within_horizon": max(scores_with_baseline) if scores_with_baseline else None,
            "source_score_included": baseline is not None,
            "quality_observation_count": len(known_scores),
            "no_escape_within_horizon": False if escape_observed else (True if negative_outcomes_observed else None),
            "original_remaining_allowance_covers_horizon": (
                episode["budget_at_s"]["remaining_tokens"] >= horizon
                if axis == "generated_tokens" and episode["budget_at_s"]["remaining_tokens"] is not None else None),
            "grade_checkpoint_ids": [p["checkpoint_id"] for p in points]}


def natural_followup_report(store: ArtifactStoreProtocol, run_ids: Sequence[str], *,
                            horizons: Sequence[int] = (), near_budget_tokens: int | None = None,
                            resource_horizons: Mapping[str, Sequence[int | float]] | None = None) -> dict[str, Any]:
    """Export natural-run and post-S rows with raw-record provenance.

    ``horizons`` are additional generated worker tokens since each S. Additional
    axes accepted by ``resource_horizons`` are ``generated_tokens``, ``cost_usd``,
    ``elapsed_seconds``, ``model_attempts`` and ``tool_calls``. Near-budget means
    *remaining original generated-token allowance <= near_budget_tokens*; with
    no threshold supplied it stays unclassified. No new budget is introduced.

    The first online S is selected prospectively for the complete natural
    trajectory, independent of capture/grade eligibility. Technical failures
    remain attached to that primary source and never select a later replacement.
    All warnings remain rows; offline labels cannot select a primary source.
    Recovery/resumed runs are rejected. Partial runs and missing metadata remain
    visible, with technical eligibility reported separately from selection.

    Cost intervals include only the recorded natural worker/grading/restore/
    infrastructure operations. External observer preparation is separately
    excluded by role; this is not a standalone treatment-policy cost allocation.
    Stage attribution and the original whole-task budget use the complete
    natural ledger, including unplaced charges and post-terminal cleanup.
    """
    if near_budget_tokens is not None and (type(near_budget_tokens) is not int or near_budget_tokens < 0):
        raise ValueError("near_budget_tokens must be a nonnegative integer or None")
    requested: dict[str, list[int | float]] = {"generated_tokens": list(horizons)}
    for axis, values in (resource_horizons or {}).items():
        if axis not in RESOURCE_AXES:
            raise ValueError(f"unsupported resource axis: {axis}")
        requested.setdefault(axis, []).extend(values)
    for axis, values in requested.items():
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("resource horizons must be finite and nonnegative")
        if axis in {"generated_tokens", "model_attempts", "tool_calls"} and any(int(v) != v for v in values):
            raise ValueError(f"{axis} horizons must be whole counts")
    requested = {k: sorted(set(v)) for k, v in sorted(requested.items())}
    report: dict[str, Any] = {"version": VERSION,
        "parameters": {"resource_horizons": requested, "near_budget_tokens": near_budget_tokens,
                       "numerical_calibration_status": "caller_supplied_not_asserted_calibrated",
                       "scope_comparison": "exact task/stage/target/session/detector/requirements/evaluator identity",
                       "primary_source_selection_rule": FIRST_STALL_SELECTION_RULE,
                       "cost_roles": sorted(NATURAL_ROLES),
                       "quality_rule": "current fresh valid grade on actual measured resources; missing boundaries interrupt support; historical best is separate",
                       "event_rule": "observed escape/solve stays known past censoring; absence needs complete horizon"},
        "runs": [], "episodes": [], "horizons": []}
    for run_id in sorted(set(run_ids)):
        events = list(store.iter_events(run_id))
        event_by_seq = {e.sequence: e for e in events}
        cps = sorted(store.list_checkpoints(run_id), key=lambda c: (c.ordinal, c.checkpoint_id))
        by_cp = {c.checkpoint_id: c for c in cps}
        if not events and not cps:
            raise ValueError(f"run has no recorded events or checkpoints: {run_id}")
        starts = [e for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"]
        if len(starts) > 1:
            raise ValueError(f"multiple run_started events: {run_id}")
        start = starts[0] if starts else None
        payload = start.payload if start else {}
        if payload.get("treatment") is not None or payload.get("resume_from") is not None:
            raise ValueError(f"natural follow-up cannot include recovery/resumed run: {run_id}")
        config, task = payload.get("config", {}), payload.get("task", {})
        try:
            result = store.get_record("run_result", run_id)
        except KeyError:
            result = {}
        result_refs = [x for x in store.list_records("run_result", run_id=run_id) if x["record_id"] == run_id]
        annotations = [{**store.get_record("annotation", x["record_id"]), "record_ref": x["ref"]}
                       for x in store.list_records("annotation", run_id=run_id)]
        annotations.sort(key=lambda a: (a["evidence_cutoff"], a["annotation_id"]))
        warnings = [a for a in annotations if a["kind"] == "S"]
        first_warning = first_stall_annotation(annotations)
        grades = {c.checkpoint_id: _grade(store, c, events) for c in cps}
        ledger = CostLedger(store).records(run_id)  # type: ignore[arg-type]
        records = [r for r in ledger if r.role in NATURAL_ROLES]
        terminal = next((e for e in reversed(events) if e.kind == "terminal"), None)
        end_seq = terminal.sequence if terminal else (events[-1].sequence if events else 0)
        run_resources = _resources(records, event_by_seq, 0, end_seq)
        cost_refs = store.list_records("cost", run_id=run_id)
        stage_usage = _stage_usage(records, events, cost_refs)
        # Full-run elapsed time has a separately recorded monotonic measurement.
        run_row = {"run_id": run_id, "task_id": task.get("task_id"), "family_id": task.get("family_id"),
                   "worker_model": config.get("model"), "repeat_id": config.get("repeat_id"),
                   "purpose": config.get("purpose"), "natural_origin_verified": start is not None,
                   "status": result.get("status"),
                   "stop_reason": result.get("termination_reason", terminal.payload.get("reason") if terminal else None),
                   "run_elapsed_seconds": result.get("details", {}).get("elapsed_seconds"),
                   "checkpoint_count": len(cps), "warning_count": len(warnings),
                   "no_s_observed": not warnings,
                   "online_warning_count": sum(a.get("origin") == "online" for a in warnings),
                   "no_online_s_observed": first_warning is None,
                   "primary_s_annotation_id": first_warning["annotation_id"] if first_warning else None,
                   "primary_technical_eligibility": None,
                   "terminal_recorded": terminal is not None, "resources": run_resources,
                   "stage_usage": stage_usage,
                   "original_budget": {**_remaining(config.get("generated_token_limit"),
                       {"generated_tokens": stage_usage["totals"]["worker_generated_tokens"]}),
                       "basis": "all recorded worker model operations; placement is not required for whole-task spend"},
                   "max_model_calls": config.get("max_model_calls"),
                   "run_started_event": _event_ref(start), "terminal_event": _event_ref(terminal),
                   "run_result_ref": result_refs[0] if result_refs else None,
                   "excluded_external_operation_ids": [r.operation_id for r in ledger if r.role not in NATURAL_ROLES],
                   "cost_record_refs": cost_refs,
                   "missingness": ([] if start else ["run_started_missing"]) + ([] if result else ["run_result_missing"])}
        report["runs"].append(run_row)
        for warning in warnings:
            source = by_cp.get(warning["checkpoint_id"])
            source_grade = grades.get(warning["checkpoint_id"])
            problems = []
            selection_exclusions = [] if warning.get("origin") == "online" else ["warning_not_online"]
            if start is None:
                problems.append("natural_origin_unverified")
            if source is None:
                problems.append("source_checkpoint_missing")
            else:
                if not source.branchable:
                    problems.append("source_checkpoint_not_branchable")
                if not source_grade["valid"]:
                    problems.append("source_grade_not_valid_fresh_bound")
                elif source_grade["available_at_sequence"] > warning["evidence_cutoff"]:
                    problems.append("source_grade_not_available_at_warning")
                if _scope_key(warning["scope"]) != _scope_key(source.scope):
                    problems.append("warning_scope_mismatch")
                if not source.event_sequence <= warning["evidence_cutoff"] <= end_seq:
                    problems.append("warning_cutoff_outside_recorded_source")
            primary = warning["annotation_id"] == run_row["primary_s_annotation_id"]
            if primary:
                run_row["primary_technical_eligibility"] = not problems
            if source is None:
                report["episodes"].append({"run_id": run_id, "s_annotation_id": warning["annotation_id"],
                    "source_checkpoint_id": warning["checkpoint_id"], "primary": primary,
                    "technical_eligibility": False, "technical_eligibility_problems": problems,
                    "selection_exclusions": selection_exclusions, "s_record": warning})
                continue
            s_cut = warning["evidence_cutoff"]
            cutoff, end_reason, boundary = _bound(events, source, s_cut)
            before = _resources(records, event_by_seq, 0, s_cut)
            budget = _remaining(config.get("generated_token_limit"), before)
            comparable = [c for c in cps if s_cut < c.event_sequence <= cutoff
                          and _scope_key(c.scope) == _scope_key(source.scope)]
            points = [{**grades[c.checkpoint_id], "resources_since_s": _resources(
                        records, event_by_seq, s_cut, grades[c.checkpoint_id]["available_at_sequence"] or c.event_sequence)}
                      for c in comparable]
            received_tools = {e.parent_event_id for e in events if e.kind == "tool_result"}
            quality_barriers = []
            for event in events:
                if not s_cut < event.sequence <= cutoff or event.scope is None or _scope_key(event.scope) != _scope_key(source.scope):
                    continue
                payload = event.payload
                missing = ((event.kind == "tool_call" and event.event_id not in received_tools)
                    or (event.kind == "tool_result" and payload.get("status") in {"transport_error", "transport_timeout", "interrupted", "error"})
                    or (event.kind in {"capture", "grade"} and payload.get("status") == "failed")
                    or (event.kind == "incident" and payload.get("what") == "terminal_capture_failed")
                    or (event.kind == "terminal" and (payload.get("terminal_capture_skipped") or payload.get("terminal_recording_failure")
                        or isinstance(payload.get("termination"), Mapping) and payload["termination"].get("policy_version") == "termination.v1"
                           and payload["termination"].get("terminal_checkpoint_id") is None)))
                if missing:
                    quality_barriers.append({"event": _event_ref(event), "reason": "state_or_grade_unconfirmed",
                        "resources_since_s": _resources(records, event_by_seq, s_cut, event.sequence)})
            for point in points:
                if point["available_at_sequence"] is not None and point["available_at_sequence"] > cutoff:
                    point["valid"] = False
                    point["exclusions"] = [*point["exclusions"], "grade_not_available_before_comparable_end"]
            valid_points = [p for p in points if p["valid"]]
            n_candidates = [a for a in annotations if a["kind"] == "N" and (
                a.get("linked_stall_id") == warning["annotation_id"] or
                a.get("evidence", {}).get("linked_s_annotation_id") == warning["annotation_id"])]
            n_records, rejected_n = [], []
            frozen_best = warning.get("evidence", {}).get("best_score_before_s")
            for n in n_candidates:
                ncp = by_cp.get(n["checkpoint_id"])
                ng = grades.get(n["checkpoint_id"])
                ev = n.get("evidence", {})
                kind = ev.get("escape_type")
                score_relief = (ng is not None and ng["score"] is not None and frozen_best is not None
                                and ng["score"] > frozen_best)
                frontier_relief = bool(ev.get("newly_reached"))
                if (ncp is None or ng is None or not ng["valid"] or n.get("origin") != "online"
                    or ng["available_at_sequence"] > n["evidence_cutoff"]
                    or _scope_key(ncp.scope) != _scope_key(source.scope)
                    or _scope_key(n["scope"]) != _scope_key(source.scope)
                    or not s_cut < n["evidence_cutoff"] <= cutoff
                    or kind not in {"new_best", "frontier_only", "both"}
                    or kind in {"new_best", "both"} and not score_relief
                    or kind in {"frontier_only", "both"} and not frontier_relief):
                    rejected_n.append(n)
                    continue
                n_records.append({"annotation_id": n["annotation_id"], "checkpoint_id": n["checkpoint_id"],
                                  "kind": kind, "record": n,
                                  "resources_since_s": _resources(records, event_by_seq, s_cut, n["evidence_cutoff"]),
                                  "evaluation_ref": ng["evaluation_ref"]})
            natural_escape = n_records[0] if n_records else None
            score_points = [p for p in valid_points if p["score"] is not None]
            new_bests = [p for p in score_points if frozen_best is not None and p["score"] > frozen_best]
            solves = [p for p in valid_points if p["full_solve"] is True]
            closure_records = [{"event": _event_ref(e), "output": e.payload["output"]} for e in events
                               if e.kind == "detector" and e.payload.get("role") == "output"
                               and e.payload.get("output", {}).get("record") == "followup_closed"
                               and e.payload["output"].get("s_annotation_id") == warning["annotation_id"]]
            episode = {"run_id": run_id, "task_id": source.scope.task_id, "family_id": task.get("family_id"),
                       "worker_model": config.get("model"), "repeat_id": config.get("repeat_id"),
                       "s_annotation_id": warning["annotation_id"], "source_checkpoint_id": source.checkpoint_id,
                       "primary": primary, "technical_eligibility": not problems,
                       "technical_eligibility_problems": problems,
                       "selection_exclusions": selection_exclusions, "s_record": warning,
                       "source_scope": to_jsonable(source.scope), "source_evaluation_ref": to_jsonable(source.evaluation_ref),
                       "best_score_before_s": frozen_best, "score_at_s": source_grade["score"],
                       "source_grade_available_at_s": bool(source_grade["valid"] and
                                                          source_grade["available_at_sequence"] <= s_cut),
                       "budget_at_s": budget, "near_budget_tokens": near_budget_tokens,
                       "near_budget": (budget["remaining_tokens"] <= near_budget_tokens
                                       if near_budget_tokens is not None and budget["remaining_tokens"] is not None else None),
                       "stop_reason": run_row["stop_reason"], "comparable_end_reason": end_reason,
                       "comparable_end_sequence": cutoff, "scope_boundary_event": _event_ref(boundary),
                       "comparable_resources": _resources(records, event_by_seq, s_cut, cutoff),
                       "full_natural_suffix_resources": _resources(records, event_by_seq, s_cut, end_seq),
                       "natural_escape": natural_escape, "rejected_n_records": rejected_n,
                       "additional_n_records": n_records[1:], "detector_followup_closed_records": closure_records,
                       "first_new_best_checkpoint_id": new_bests[0]["checkpoint_id"] if new_bests else None,
                       "first_new_best_resources": new_bests[0]["resources_since_s"] if new_bests else None,
                       "first_solve_checkpoint_id": solves[0]["checkpoint_id"] if solves else None,
                       "first_solve_resources": solves[0]["resources_since_s"] if solves else None,
                       "best_future_observed_score": max(p["score"] for p in score_points) if score_points else None,
                       "best_future_score_gain": (max(p["score"] for p in score_points) - frozen_best
                                                  if score_points and frozen_best is not None else None),
                       "comparable_grade_count": len(valid_points), "points": points, "quality_barriers": quality_barriers,
                       "excluded_later_checkpoint_ids": [c.checkpoint_id for c in cps
                                                         if c.event_sequence > cutoff],
                       "interpretation": "One unchanged natural suffix; not an independent matched Continue rollout."}
            report["episodes"].append(episode)
            report["horizons"].extend(_horizon(episode, axis, horizon)
                                      for axis, values in requested.items() for horizon in values)
    report["denominators"] = {
        "input_run_ids": len(run_ids), "distinct_runs": len(report["runs"]),
        "runs_without_s": sum(r["no_s_observed"] for r in report["runs"]),
        "runs_with_primary_s": sum(r["primary_s_annotation_id"] is not None for r in report["runs"]),
        "runs_with_eligible_primary_s": sum(r["primary_technical_eligibility"] is True for r in report["runs"]),
        "all_warning_episodes": len(report["episodes"]),
        "primary_episodes": sum(e["primary"] for e in report["episodes"]),
        "eligible_primary_episodes": sum(e["primary"] and e["technical_eligibility"] for e in report["episodes"]),
        "known_families": len({r["family_id"] for r in report["runs"] if r["family_id"] is not None}),
        "runs_missing_family": sum(r["family_id"] is None for r in report["runs"]),
    }
    return report
