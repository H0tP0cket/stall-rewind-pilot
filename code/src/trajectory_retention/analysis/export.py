"""Traceable data exports: long-format rows tied to treatment/branch/checkpoint IDs.

Every exported row carries identifiers that resolve to stored records
(treatment manifest, branch run, graded checkpoint, T version, packet). Plots
made from these files are downstream of the raw run records, never a
substitute for them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import TreatmentManifest
from .continuations import ContinuationRecord
from .curves import arm_curve
from .conditions import condition_identity, execution_problems


def continuation_rows(records: Sequence[ContinuationRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        base = {"treatment_id": r.treatment_id, "branch_run_id": r.branch_run_id, "source_checkpoint_id": r.source_checkpoint_id,
                "arm_id": r.arm_id, "repeat_id": r.repeat_id, "worker_model": r.worker_model,
                "allowance": r.allowance, "stop_reason": r.stop_reason, "technical_failure": r.technical_failure,
                "observed_horizon": r.observed_horizon, "baseline_score": r.baseline_score,
                "horizon_complete": r.horizon_complete, "unknown_from_tokens": r.unknown_from_tokens,
                "termination_json": json.dumps(dict(r.termination), sort_keys=True),
                "termination_binding_problems_json": json.dumps(r.termination_binding_problems),
                "quality_barriers_json": json.dumps(r.quality_barriers, sort_keys=True),
                "allowance_absorption_qualified": r.absorption_point() is not None,
                "target_id": r.lineage.get("target_id"), "selection_kind": r.lineage.get("selection_kind"),
                "execution_condition_json": json.dumps(condition_identity(r.lineage), sort_keys=True),
                "execution_binding_problems_json": json.dumps(execution_problems(r.lineage))}
        if not r.points:
            rows.append({**base, "checkpoint_id": None, "generated_tokens": None, "score": None, "valid": None,
                         "full_solve": None, "checks_json": None, "tokens_complete": None,
                         "checkpoint_event_sequence": None, "grade_available_at_sequence": None,
                         "grade_event_id": None, "evaluation_id": None, "grade_exclusions_json": None})
        for p in r.points:
            rows.append({**base, "checkpoint_id": p.checkpoint_id, "generated_tokens": p.generated_tokens,
                         "score": p.score, "valid": p.valid, "full_solve": p.full_solve,
                         "checks_json": json.dumps(dict(p.check_vector), sort_keys=True),
                         "tokens_complete": p.tokens_complete, "checkpoint_event_sequence": p.event_sequence,
                         "grade_available_at_sequence": p.available_at_sequence, "grade_event_id": p.grade_event_id,
                         "evaluation_id": p.evaluation_id, "grade_exclusions_json": json.dumps(p.exclusions)})
    return rows


def horizon_rows(records: Sequence[ContinuationRecord], horizons: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        for h in horizons:
            rows.append({"treatment_id": r.treatment_id, "source_checkpoint_id": r.source_checkpoint_id, "arm_id": r.arm_id,
                         "repeat_id": r.repeat_id, "horizon": h, "score_at": r.score_at(h),
                         "curve": "allowance", "best_observed_score_at": r.best_score_at(h),
                         "actual_compute_score_at": r.score_at(h, curve="actual_compute"),
                         "actual_compute_status": r.horizon_status(h, curve="actual_compute"),
                         "current_full_solve": r.current_solve_at(h),
                         "allowance_absorbed": h > r.observed_horizon and r.observed_at(h),
                         "worker_model": r.worker_model, "branch_run_id": r.branch_run_id,
                         "target_id": r.lineage.get("target_id"), "selection_kind": r.lineage.get("selection_kind"),
                         "last_observed_grade": r.last_graded_score_at(h), "status": r.horizon_status(h),
                         "solved_by": r.solved_by(h), "observed": r.score_at(h) is not None})
    return rows


def curve_rows(records: Sequence[ContinuationRecord], horizons: Sequence[int]) -> list[dict[str, Any]]:
    by_arm: dict[tuple[str, str, int, str], list[ContinuationRecord]] = {}
    for r in records:
        identity = json.dumps({**{k: r.lineage.get(k) for k in ("model_settings", "selection_kind", "runtime_policy")},
                               **condition_identity(r.lineage),
                               "execution_binding_problems": execution_problems(r.lineage),
                               "unmatched_run": r.branch_run_id if execution_problems(r.lineage) else None}, sort_keys=True)
        by_arm.setdefault((r.arm_id, r.worker_model, r.allowance, identity), []).append(r)
    rows = []
    for (arm, model, allowance, identity), rs in sorted(by_arm.items()):
        for curve in ("allowance", "actual_compute"):
            for cp in arm_curve(rs, horizons, curve=curve):
                rows.append({"arm_id": arm, "worker_model": model, "allowance": allowance,
                             "condition_json": identity, "solve_rate_basis": "ever observed solve",
                             **cp.__dict__})
    return rows


def manifest_rows(manifests: Sequence[TreatmentManifest]) -> list[dict[str, Any]]:
    return [{
        "treatment_id": m.treatment_id, "source_checkpoint_id": m.source_checkpoint_id, "arm_id": m.metadata.get("arm_id"),
        "repeat_id": m.repeat_id, "selection_kind": m.selection_kind, "workspace_donor": m.component_donors.get("workspace"),
        "history_donor": m.component_donors.get("history"), "history_policy": m.history_policy,
        "runtime_policy": m.runtime_policy, "packet_kind": (m.metadata.get("packet") or {}).get("kind"),
        "packet_id": (m.metadata.get("packet") or {}).get("packet_id"),
        "packet_approx_tokens": (m.metadata.get("packet") or {}).get("approx_token_count"),
        "target_manifest_id": m.metadata.get("recovery_target_manifest_id"), "worker_model": m.worker_model,
        "allowance": m.generated_token_limit, "intended_mismatches": ";".join(m.intended_mismatches),
    } for m in manifests]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return path


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    return path


def export_bundle(out_dir: Path, *, manifests: Sequence[TreatmentManifest], records: Sequence[ContinuationRecord],
                  horizons: Sequence[int], analysis: Mapping[str, Any] | None = None,
                  family_by_source: Mapping[str, str] | None = None, analysis_spec: Any = None) -> dict[str, str]:
    from .effects import AnalysisSpec, matched_effects

    out_dir = Path(out_dir)
    effects = matched_effects(manifests=manifests, records=records,
        spec=analysis_spec or AnalysisSpec(tuple(horizons)), family_by_source=family_by_source)
    if analysis is not None:
        effects["upstream_selection"] = analysis.get("selection")
        effects["upstream_skipped"] = analysis.get("skipped", [])
    written = {
        "manifests": str(write_csv(out_dir / "treatments.csv", manifest_rows(manifests))),
        "continuations": str(write_csv(out_dir / "continuation_points.csv", continuation_rows(records))),
        "horizons": str(write_csv(out_dir / "horizon_scores.csv", horizon_rows(records, horizons))),
        "curves": str(write_csv(out_dir / "arm_curves.csv", curve_rows(records, horizons))),
        "records": str(write_json(out_dir / "continuations.json", [r.as_record() for r in records])),
        "matched_effects": str(write_json(out_dir / "matched_effects.json", effects)),
        "matched_blocks": str(write_csv(out_dir / "matched_blocks.csv", [
            {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in effects["matched_blocks"]])),
    }
    if analysis is not None:
        written["analysis"] = str(write_json(out_dir / "analysis.json", analysis))
    return written


def export_campaign_analysis(store: Any, campaign_id: str, out_dir: Path, *, analysis_spec: Any = None,
                             near_budget_tokens: int | None = None,
                             resource_horizons: Mapping[str, Sequence[float]] | None = None,
                             dollar_horizons: Sequence[float] | None = None) -> dict[str, str]:
    """Read-only reanalysis of existing real or fixture records; no model calls.

    Use a new versioned directory. Frozen campaign accounting cuts supply the
    actual operation union; standalone preparation follows the exact packet's
    ancestral T/summary calls. Original campaign reports and artifacts stay
    untouched. No-S and engineering-only selections remain explicit. Unspecified
    analysis settings inherit the recorded campaign configuration; explicit
    overrides are recorded in the new receipt.
    """
    from ..campaign import analysis_stop_reason
    from ..contracts import ArtifactRef, RunResult, from_jsonable
    from .continuations import from_run_records
    from .costs import policy_cost_report
    from .effects import AnalysisSpec
    from .natural import natural_followup_report
    from ..observers.audit import audit_recorded_chain, audit_recorded_ancestry

    report = store.get_record("campaign_result", campaign_id)
    config = store.get_record("campaign", campaign_id)
    task = store.get_record("campaign_task", campaign_id)
    try:
        schedule = store.get_record("campaign_schedule", campaign_id)
    except KeyError:
        schedule = {"treatments": []}
    manifests = [from_jsonable(TreatmentManifest, m) for m in schedule.get("treatments", [])]
    by_id = {m.treatment_id: m for m in manifests}
    records = []
    for item in report["branch_results"]:
        result = from_jsonable(RunResult, item)
        rows = store.list_records("treatment", run_id=result.run_id)
        if len(rows) != 1:
            raise ValueError("branch requires exactly one recorded treatment")
        m = from_jsonable(TreatmentManifest, store.get_record("treatment", rows[0]["record_id"]))
        if m.treatment_id not in by_id or m != by_id[m.treatment_id]:
            raise ValueError("executed treatment differs from campaign schedule")
        records.append(from_run_records(store, m, result.run_id, stop_reason=analysis_stop_reason(store, result),
            technical_failure=result.status in {"error", "failed", "invalid", "technical_failure"}))
    spec = analysis_spec or AnalysisSpec(tuple(config["horizons"]), **config.get("analysis_settings", {}))
    if near_budget_tokens is None:
        near_budget_tokens = config.get("near_budget_tokens")
    if resource_horizons is None:
        resource_horizons = config.get("resource_horizons", {})
    if dollar_horizons is None:
        dollar_horizons = config.get("dollar_horizons", ())
    out_dir = Path(out_dir)
    # Prevent accidental overwriting of an older analysis with changed code/settings.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("use an empty versioned output directory for reanalysis")
    family = {m.source_checkpoint_id: task["family_id"] for m in manifests}
    written = export_bundle(out_dir, manifests=manifests, records=records, horizons=spec.horizons,
                            analysis=report, family_by_source=family, analysis_spec=spec)
    natural_run = report["natural"]["run_id"]
    natural = natural_followup_report(store, [natural_run], horizons=spec.horizons,
                                     near_budget_tokens=near_budget_tokens, resource_horizons=resource_horizons)
    written["natural_followup"] = str(write_json(out_dir / "natural_followup.json", natural))
    chain_audit = audit_recorded_chain(store, report["observer"]["version_ids"])
    selected_cp = report["selection"].get("checkpoint_id")
    selected_version = next((row["version_id"] for row in chain_audit["versions"]
                             if row["checkpoint_id"] == selected_cp), None)
    observer_audit = {"declared_chain": chain_audit,
        "selected_ancestry": audit_recorded_ancestry(store, selected_version) if selected_version else None,
        "semantic_quality": "not_measured; source validity and lexical agreement are not entailment"}
    written["observer_source_audit"] = str(write_json(out_dir / "observer_source_audit.json", observer_audit))
    operation_keys = []
    for run_id, ref in sorted(report["accounting_refs"].items()):
        snapshot = store.get_json(from_jsonable(ArtifactRef, ref))
        operation_keys.extend((run_id, op) for op in snapshot["operation_ids"])
    costs = policy_cost_report(store, manifests=manifests, records=records,
                               actual_operation_keys=operation_keys, dollar_horizons=dollar_horizons)
    written["policy_costs"] = str(write_json(out_dir / "policy_costs.json", costs))
    written["cost_quality_points"] = str(write_csv(out_dir / "cost_quality_points.csv", [
        {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
        for row in costs["cost_quality_points"]]))
    written.update(export_trace_tables(out_dir, store=store,
        run_ids=[natural_run, *(r.branch_run_id for r in records)], version_ids=report["observer"]["version_ids"]))
    import hashlib
    package_root = Path(__file__).resolve().parent.parent
    modules = [p for folder in ("analysis", "observers", "experiments") for p in (package_root / folder).glob("*.py")]
    modules.extend(package_root / name for name in ("accounting.py", "contracts.py", "store.py", "campaign.py"))
    receipt = {"version": "campaign-analysis-receipt.v1", "campaign_id": campaign_id,
        "selection": report["selection"], "family_id": task["family_id"], "analysis_spec": spec.__dict__,
        "followup_parameters": {"near_budget_tokens": near_budget_tokens,
                                "resource_horizons": resource_horizons},
        "dollar_horizons": list(dollar_horizons),
        "executions_repeated": False, "source_campaign_mutated": False,
        "outputs": {k: {"path": str(Path(p)), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                    for k, p in sorted(written.items())},
        "source_modules": {str(p.relative_to(package_root.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(modules)}}
    written["receipt"] = str(write_json(out_dir / "receipt.json", receipt))
    return written


def checkpoint_resource_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One CSV row per checkpoint; detailed identities stay in typed JSON."""
    rows = []
    for checkpoint in report["checkpoints"]:
        capture, grade = checkpoint["capture"], checkpoint["grade_available"]
        row = {k: checkpoint[k] for k in ("run_id", "checkpoint_id", "ordinal", "parent_checkpoint_id",
            "previous_same_run_checkpoint_id", "previous_capture_sequence", "source_checkpoint_id")}
        row.update({"capture_event_sequence": capture["through_sequence"], "capture_event_id": capture["source_event_id"],
            "grade_available_at_sequence": grade["through_sequence"] if grade else None,
            "grade_event_id": grade["source_event_id"] if grade else None,
            "grade_availability_status": checkpoint["grade_availability"]["status"],
            "snapshot_matches_derived_capture": checkpoint["snapshot_matches_derived_capture"]})
        for field in ("scope", "source_checkpoint_bindings", "evaluation_ref", "accounting_ref",
                      "recorded_accounting_snapshot", "snapshot_binding_problems", "grade_availability"):
            row[field + "_json"] = json.dumps(checkpoint[field], sort_keys=True)
        for name, boundary, delta_name in (("capture", capture, "delta_from_previous_capture"),
                                           ("grade_available", grade, "delta_since_capture")):
            row[name + "_attribution_complete"] = boundary["attribution_complete"] if boundary else None
            row[name + "_json"] = json.dumps(boundary, sort_keys=True)
            for view in ("cumulative", "delta"):
                summary = boundary["cumulative" if view == "cumulative" else delta_name] if boundary else None
                prefix = f"{name}_{view}_"
                for field in ("operations", "model_calls", "tool_calls"):
                    row[prefix + field] = summary["totals"][field] if summary else None
                for category in ("grading", "overhead"):
                    row[prefix + category + "_operations"] = summary["categories"][category]["operations"] if summary else None
                for metric in ("worker_generated_tokens", "cost_usd", "operation_seconds"):
                    fields = ("value", "known_sum", "missing_operations", "complete") if view == "cumulative" else (
                        "value", "known_sum_change", "missing_operations_before", "missing_operations_after", "complete")
                    for field in fields:
                        row[prefix + metric + "_" + field] = summary["totals"][metric][field] if summary else None
        rows.append(row)
    return rows


def export_trace_tables(out_dir: Path, *, store: Any, run_ids: Sequence[str],
                        version_ids: Sequence[str]) -> dict[str, str]:
    """Lossless-index tables for grades, T evolution and operation accounting.

    Every checkpoint occurrence remains a row. Nested evaluator vectors and
    provenance stay typed JSON; missing scalar values are blank, never zero.
    Cost rows include all recorded operations on the supplied runs, including
    later observer audits/repairs; operation identity prevents double counting.
    """
    from collections import Counter
    from ..accounting import CostLedger
    from ..contracts import to_jsonable
    from .costs import checkpoint_resource_report

    rows, costs, annotation_rows = [], [], []
    versions = [store.get_record("t_version", vid) for vid in version_ids]
    version_by_cp = {v["target_checkpoint_id"]: (vid, v) for vid, v in zip(version_ids, versions)}
    if len(version_by_cp) != len(versions):
        raise ValueError("Supply one declared observer chain, not multiple T versions per checkpoint")
    for run_id in dict.fromkeys(run_ids):
        labels: dict[str, list[dict]] = {}
        for item in store.list_records("annotation", run_id=run_id):
            a = store.get_record("annotation", item["record_id"])
            annotation_rows.append(a)
            labels.setdefault(a["checkpoint_id"], []).append(a)
        for cp in store.list_checkpoints(run_id):
            grade = store.get_json(cp.evaluation_ref) if cp.evaluation_ref else {}
            cost = store.get_json(cp.accounting_ref).get("totals", {}) if cp.accounting_ref else {}
            vid, version = version_by_cp.get(cp.checkpoint_id, (None, None))
            rows.append({"run_id": run_id, "checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
                "event_cut": cp.event_sequence, "stage_id": cp.scope.stage_id, "target_id": cp.scope.target_id,
                "branchable_recorded": cp.branchable, "labels_json": json.dumps(labels.get(cp.checkpoint_id, []), sort_keys=True),
                "score": grade.get("score"), "raw_score": grade.get("raw_score"), "raw_max_score": grade.get("raw_max_score"),
                "grade_valid": grade.get("valid"), "full_solve": grade.get("full_solve"),
                "evaluation_reused": bool(cp.validation.get("evaluation_reused")),
                "checks_json": json.dumps(grade.get("checks"), sort_keys=True),
                "capabilities_json": json.dumps(grade.get("capabilities"), sort_keys=True),
                "groups_json": json.dumps(grade.get("group_scores"), sort_keys=True),
                "frontier_json": json.dumps(grade.get("reachable_milestones"), sort_keys=True),
                "worker_generated_tokens": cost.get("worker_generated_tokens", {}).get("value"),
                "cost_known_usd": cost.get("cost_usd", {}).get("known_sum"),
                "unpriced_operations": cost.get("cost_usd", {}).get("missing_operations"),
                "state_refs_json": json.dumps(to_jsonable(cp.state), sort_keys=True),
                "evaluation_ref_json": json.dumps(to_jsonable(cp.evaluation_ref), sort_keys=True),
                "accounting_ref_json": json.dumps(to_jsonable(cp.accounting_ref), sort_keys=True),
                "T_version_id": vid, "T_status": version["status"] if version else None,
                "T_entry_types_json": json.dumps(dict(Counter(e["type"] for e in version["entries"])), sort_keys=True) if version else None,
                "T_rejected_operations": len(version["rejected_ops"]) if version else None})
        costs.extend(to_jsonable(c) for c in CostLedger(store).records(run_id))
    out_dir = Path(out_dir)
    checkpoint_path = write_csv(out_dir / "checkpoint_trace.csv", rows)
    cost_path = out_dir / "cost_operations.jsonl"
    with cost_path.open("w") as f:
        for c in costs:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    annotation_path = out_dir / "annotations.jsonl"
    with annotation_path.open("w") as f:
        for annotation in annotation_rows:
            f.write(json.dumps(annotation, sort_keys=True) + "\n")
    resources = checkpoint_resource_report(store, run_ids=run_ids)
    resource_path = write_json(out_dir / "checkpoint_resources.json", resources)
    resource_csv = write_csv(out_dir / "checkpoint_resources.csv", checkpoint_resource_rows(resources))
    return {"checkpoint_trace": str(checkpoint_path), "cost_operations": str(cost_path),
            "annotations": str(annotation_path), "checkpoint_resources": str(resource_path),
            "checkpoint_resource_rows": str(resource_csv)}
