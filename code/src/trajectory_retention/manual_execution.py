"""One explicit manual treatment through the shared runner, with durable reuse."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from .accounting import CostLedger, summarize
from .campaign import PhaseFailed, PhaseJournal, PhaseUncertain, analysis_stop_reason
from .contracts import (ArtifactRef, RunConfig, RunResult, TreatmentManifest, WorkerRequestPolicy, WORKER_OPERATIONAL_FIELDS,
                        canonical_json, from_jsonable, to_jsonable, worker_operational_settings)

VERSION = "manual-execution.v1"
EXECUTION_PHASE = "manual_execution"


def _source(store: Any, manifest: TreatmentManifest):
    decision = store.get_checkpoint(manifest.source_checkpoint_id)
    starts = [event for event in store.iter_events(decision.run_id)
              if event.kind == "lifecycle" and event.payload.get("transition") == "run_started"]
    if len(starts) != 1 or starts[0].sequence > decision.event_sequence:
        raise ValueError("Manual execution requires one natural run_started binding before the decision checkpoint")
    start = starts[0]
    from .resample_source import recovery_source_ok, store_start_lookup
    if not recovery_source_ok(start.payload, store_start_lookup(store)):
        raise ValueError("Manual execution source must be a natural run or a natural resample, not a branch/resume")
    config = start.payload.get("config")
    if not isinstance(config, dict) or config.get("run_id") != decision.run_id:
        raise ValueError("Natural run_started config does not bind the source run")
    return decision, start, config


def _identity(experiment_id: str) -> None:
    if type(experiment_id) is not str or not experiment_id.strip():
        raise ValueError("Manual execution requires an explicit nonempty experiment identity")


def manual_run_config(store: Any, manifest: TreatmentManifest, experiment_id: str) -> RunConfig:
    """Derive exact source worker settings, with only declared recovery changes.

    This reader does not initialize a runner/client or write store evidence.
    CLI source qualification uses the retained source_binding in metadata.
    """
    from .experiments.manual import FIXED_TO_SOURCE, MANUAL_VERSION
    from .experiments.treatments import recovery_settings_known
    _identity(experiment_id)
    if manifest.metadata.get("manual_experiment", {}).get("version") != MANUAL_VERSION:
        raise ValueError("Expected a built manual-experiment.v1 treatment manifest")
    decision, _, original = _source(store, manifest)
    if experiment_id in {decision.run_id, original.get("campaign_id")}:
        raise ValueError("Manual execution requires a new experiment identity")
    if original.get("model") != manifest.worker_model or canonical_json(original.get("model_settings")) != canonical_json(manifest.metadata.get("model_settings")):
        raise ValueError("Manual treatment worker model/settings differ from the natural source")
    if type(original.get("model")) is not str or not original["model"] or not isinstance(original.get("model_settings"), dict):
        raise ValueError("Natural source lacks explicit worker model/settings")
    if original.get("purpose") not in {"development", "calibration", "main"}:
        raise ValueError("Natural source lacks a supported recorded purpose")
    settings = manifest.metadata.get("recovery_settings")
    if not recovery_settings_known(settings) or set(settings) != set(WORKER_OPERATIONAL_FIELDS):
        raise ValueError("Manual treatment requires complete explicit recovery settings")
    if any(key not in original or canonical_json(original[key]) != canonical_json(settings[key]) for key in FIXED_TO_SOURCE):
        raise ValueError("Manual recovery settings differ from runner-fixed natural source settings")
    if not isinstance(original.get("metadata", {}), dict):
        raise ValueError("Natural source metadata must be an object")
    values = {"run_id": f"{experiment_id}:branch:{manifest.treatment_id}",
        "campaign_id": experiment_id, "purpose": original["purpose"], "model": original["model"],
        "model_settings": original["model_settings"], "generated_token_limit": manifest.generated_token_limit,
        "stage_id": manifest.stage_id, "repeat_id": manifest.repeat_id, **settings,
        "metadata": {**original.get("metadata", {}), "treatment_id": manifest.treatment_id,
            "selection_kind": manifest.selection_kind, "execution_role": "recovery", "manual_experiment_id": experiment_id,
            "comparison_role": "manual", "manual_comparison_label": manifest.metadata.get("arm_id")}}
    # Retain recorded numeric representations as well as values; the generic
    # reader normalizes integers in float-annotated fields during reconstruction.
    values = json.loads(canonical_json(values))
    values["worker_request_policy"] = WorkerRequestPolicy(**values["worker_request_policy"])
    config = RunConfig(**values)
    if canonical_json(worker_operational_settings(config)) != canonical_json(settings):
        raise ValueError("Derived manual config does not preserve declared recovery settings")
    return config


def _request(manifest: TreatmentManifest, experiment_id: str, horizons: Sequence[int]) -> dict[str, Any]:
    _identity(experiment_id)
    if (not isinstance(horizons, (list, tuple)) or not horizons
            or any(type(h) is not int or h <= 0 for h in horizons) or list(horizons) != sorted(set(horizons))):
        raise ValueError("Manual horizons must be an explicit nonempty increasing list of positive integers")
    return {"version": VERSION, "experiment_id": experiment_id, "manifest": to_jsonable(manifest), "horizons": list(horizons)}


def _readback(store: Any, saved: dict[str, Any], export_dir: Path) -> dict[str, str]:
    export_dir = Path(export_dir).resolve()
    if export_dir.is_relative_to(store.root):
        raise ValueError("Manual exports must be outside the retained store")
    prepared = {}
    for key, item in saved.items():
        if type(item.get("filename")) is not str or Path(item["filename"]).name != item["filename"]:
            raise ValueError("Saved manual export has an invalid filename")
        path = export_dir / item["filename"]
        data = store.get_bytes(from_jsonable(ArtifactRef, item["ref"]))
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite changed completed manual export: {path}")
        prepared[key] = path, data
    export_dir.mkdir(parents=True, exist_ok=True)
    for path, data in prepared.values():
        path.write_bytes(data)
    return {key: str(path) for key, (path, _) in prepared.items()}


def reuse_manual_experiment(store: Any, manifest: TreatmentManifest, *, experiment_id: str,
                            horizons: Sequence[int], export_dir: Path) -> dict[str, Any] | None:
    """Read completed exports before current task/provider qualification.

    Failed/uncertain paid phases also refuse a repeat before any live setup.
    No source revalidation, client, PhaseJournal or store write occurs here.
    """
    request = _request(manifest, experiment_id, horizons)
    try:
        prior = store.get_record("manual_experiment", experiment_id)
    except KeyError:
        return None
    if canonical_json(prior["request"]) != canonical_json(request):
        raise ValueError("Manual experiment identity is already bound to different inputs")
    try:
        result = store.get_record("manual_experiment_result", experiment_id)
    except KeyError:
        phase_id = f"{experiment_id}:{EXECUTION_PHASE}"
        try:
            finished = store.get_record("phase_finished", phase_id)
        except KeyError:
            try:
                store.get_record("phase_started", phase_id)
            except KeyError:
                return None
            raise PhaseUncertain(f"Phase {phase_id} started without a final receipt; inspect before a new attempt")
        if finished["status"] != "ok":
            raise PhaseFailed(f"Phase {phase_id} previously failed; receipt preserved")
        return None  # Runner result is reusable; derived exports have not completed.
    saved = store.get_record("manual_experiment_exports", experiment_id)
    return {**result, "exports": _readback(store, saved, export_dir)}


def _information_lineage(store: Any, manifest: TreatmentManifest) -> list[dict[str, Any]]:
    """Preserve exact packet/freeze/T parent/call records without reallocating them."""
    packet = manifest.metadata.get("packet", {})
    records = []
    if packet.get("packet_id"):
        kind = {"summary": "summary", "research_packet": "research_packet"}.get(packet.get("kind"), "packet")
        records.append({"kind": kind, "record_id": packet["packet_id"], "record": store.get_record(kind, packet["packet_id"])})
    if packet.get("freeze_id"):
        records.append({"kind": "frozen_packets_v2", "record_id": packet["freeze_id"],
            "record": store.get_record("frozen_packets_v2", packet["freeze_id"])})
    version_id, seen = packet.get("source_version_id"), set()
    while version_id:
        if version_id in seen:
            raise ValueError("Cycle in manual packet T lineage")
        seen.add(version_id)
        version = store.get_record("t_version", version_id)
        records.append({"kind": "t_version", "record_id": version_id, "record": version})
        version_id = version.get("parent_version_id")
    return records


def _implementation(store: Any, experiment_id: str, branch_run_id: str) -> dict[str, Any]:
    from .provenance import implementation_binding, same_execution_binding
    try:
        prior = store.get_record("manual_experiment_implementation", experiment_id)
    except KeyError:
        prior = None
    try:
        finished = store.get_record("phase_finished", f"{experiment_id}:{EXECUTION_PHASE}")
    except KeyError:
        finished = None
    if finished is not None and finished["status"] == "ok":
        if prior is None:
            raise ValueError("Completed manual branch lacks its execution implementation binding")
        return prior  # Repair derived exports under the original branch binding.
    actual = implementation_binding(store)
    if prior is not None and not same_execution_binding(actual, prior):
        raise ValueError("Manual experiment implementation changed before branch execution")
    if prior is None:
        store.put_record("manual_experiment_implementation", experiment_id, actual, run_id=branch_run_id)
    return prior if prior is not None else to_jsonable(actual)


def run_manual_experiment(store: Any, task: Any, runner: Any, manifest: TreatmentManifest, *,
                          experiment_id: str, horizons: Sequence[int], export_dir: Path) -> dict[str, Any]:
    """Validate one manual intervention and dispatch the shared Runner once."""
    previous = reuse_manual_experiment(store, manifest, experiment_id=experiment_id, horizons=horizons, export_dir=export_dir)
    if previous is not None:
        return previous
    request = _request(manifest, experiment_id, horizons)
    config = manual_run_config(store, manifest, experiment_id)
    decision, start, _ = _source(store, manifest)
    task_record = task.describe()
    if canonical_json(to_jsonable(task_record)) != canonical_json(start.payload.get("task")):
        raise ValueError("Manual task identity differs from the immutable natural source")
    stage = task.stage(manifest.stage_id)
    recorded_stages = start.payload["task"].get("stages", [])
    if (to_jsonable(stage) not in recorded_stages or (stage.stage_id, stage.target_id, stage.requirement_version, stage.evaluator_version)
            != (decision.scope.stage_id, decision.scope.target_id, decision.scope.requirement_version, decision.scope.evaluator_version)
            or task_record.task_id != decision.scope.task_id):
        raise ValueError("Manual task stage differs from the fixed decision target")
    from .experiments.treatments import validate_manifest
    problems = validate_manifest(manifest, store, decision.run_id)
    if problems:
        raise ValueError("Manual treatment is not executable: " + "; ".join(problems))
    information = _information_lineage(store, manifest)
    export_dir = Path(export_dir).resolve()
    if export_dir.is_relative_to(store.root):
        raise ValueError("Manual exports must be outside the retained store")
    try:
        existing = store.get_record("treatment", manifest.treatment_id)
    except KeyError:
        store.put_record("treatment", manifest.treatment_id, manifest, run_id=config.run_id)
    else:
        if canonical_json(existing) != canonical_json(manifest):
            raise ValueError("Treatment identity is already bound to a different manifest")
    store.put_record("manual_experiment", experiment_id, {"request": request, "run_config": config,
        "source_run_id": decision.run_id, "source_run_started_event_id": start.event_id}, run_id=config.run_id)
    implementation = _implementation(store, experiment_id, config.run_id)
    phases = PhaseJournal(store, experiment_id)

    def execute():
        if next(iter(store.iter_events(config.run_id)), None) is not None:
            raise ValueError("Manual branch run identity already has execution evidence")
        result = runner.run(task, config, store=store, resume=decision, treatment=manifest)
        if result.run_id != config.run_id:
            raise ValueError("Runner result identifies a different manual branch")
        return result

    result = from_jsonable(RunResult, phases.execute(EXECUTION_PHASE,
        {"request": request, "run_config": config, "implementation_fingerprint_sha256": implementation["fingerprint_sha256"]}, execute))
    from .analysis.continuations import from_run_records
    from .analysis.export import (continuation_rows, export_trace_tables, horizon_rows, manifest_rows, write_csv, write_json)
    continuation = from_run_records(store, manifest, result.run_id, stop_reason=analysis_stop_reason(store, result),
        technical_failure=result.status in {"error", "failed", "invalid", "technical_failure"})
    continuation = replace(continuation, lineage={**continuation.lineage, "manual_experiment_id": experiment_id,
        "comparison_role": "manual", "manual_comparison_label": manifest.metadata["arm_id"],
        "raw_termination_reason": result.termination_reason, "run_status": result.status})
    costs = CostLedger(store).records(result.run_id)
    joins = {"experiment_id": experiment_id, "comparison_role": "manual", "manual_comparison_label": manifest.metadata["arm_id"],
        "treatment_id": manifest.treatment_id, "source_run_id": decision.run_id,
        "source_checkpoint_id": decision.checkpoint_id, "branch_run_id": result.run_id}
    cost_report = {"version": VERSION, **joins, "scope": "actual branch operations, including terminal closure",
        "totals": summarize(costs), "operation_keys": [[r.run_id, r.operation_id] for r in costs],
        "roles": {role: summarize(r for r in costs if r.role == role) for role in sorted({r.role for r in costs})},
        "shared_preparation": {"allocated_cost_usd": None, "status": "unallocated",
            "reason": "Historical preparation is shared; no counterfactual allocator is inferred for a manual intervention.",
            "source_scope": to_jsonable(manifest.source_scope), "packet": to_jsonable(manifest.metadata.get("packet")),
            "packet_refs": to_jsonable(manifest.packet_refs)}}
    report = {"version": VERSION, **joins, "pipeline_status": "outcomes_recorded" if result.status == "completed" else "partial_failure",
        "run_result": to_jsonable(result),
        "horizons": list(horizons), "run_config": to_jsonable(config), "manifest": to_jsonable(manifest),
        "source_run_started_event_id": start.event_id, "actual_branch_costs": cost_report,
        "implementation_record_id": experiment_id,
        "implementation_fingerprint_sha256": implementation["fingerprint_sha256"],
        "limitations": ["Manual comparison; no core-arm pooling or inferential defaults.",
            "Shared preparation remains unallocated; exact source and branch operation keys are exported."]}
    outputs = {"manual_result": str(write_json(export_dir / "manual_result.json", report)),
        "treatment": str(write_csv(export_dir / "treatment.csv", [{**row, **joins} for row in manifest_rows([manifest])])),
        "continuations": str(write_json(export_dir / "continuations.json", [continuation.as_record()])),
        "continuation_points": str(write_csv(export_dir / "continuation_points.csv", [{**row, **joins} for row in continuation_rows([continuation])])),
        "horizon_scores": str(write_csv(export_dir / "horizon_scores.csv", [{**row, **joins} for row in horizon_rows([continuation], horizons)])),
        "manual_costs": str(write_json(export_dir / "manual_costs.json", cost_report)),
        "information_lineage": str(write_json(export_dir / "information_lineage.json", {**joins,
            "source_scope": to_jsonable(manifest.source_scope), "records": information}))}
    outputs.update(export_trace_tables(export_dir, store=store, run_ids=[decision.run_id, result.run_id], version_ids=[]))
    saved = {key: {"filename": Path(path).name, "ref": store.put_bytes(Path(path).read_bytes())} for key, path in outputs.items()}
    store.put_record("manual_experiment_exports", experiment_id, saved, run_id=config.run_id)
    store.put_record("manual_experiment_result", experiment_id, report, run_id=config.run_id)
    return {**report, "exports": outputs}
