"""Evidence-bound main-study gate, separate from admission and pilot execution."""
from __future__ import annotations

import hashlib
from typing import Any

from .contracts import ArtifactRef, canonical_json, from_jsonable, to_jsonable
from .store import ArtifactStore


REQUIRED_DECISIONS = (
    "population_and_family_split", "task_budget_assignment", "detector_rules",
    "observer_prompts_and_audits", "token_matching", "repeats_and_horizons",
    "contrasts_and_multiplicity", "curve_and_censoring_estimands",
    "missingness_and_failures", "total_policy_cost", "baseline_policy",
)
REQUIRED_GATES = (
    "restore_and_isolation", "recording_and_accounting", "detector_and_lifecycle",
    "observer_and_controls", "analysis_and_repeats", "model_routes",
    "independent_integration_audit",
)


def settings_binding(config: Any, task: Any) -> dict[str, Any]:
    """Experimental conditions; run names/paths never substitute for settings."""
    return to_jsonable({
        "task_id": task.task_id, "task_version": task.task_version,
        "environment_version": task.environment_version,
        "targets": [{"target_id": s.target_id, "requirement_version": s.requirement_version,
                     "evaluator_version": s.evaluator_version} for s in task.stages],
        "model": config.mining.model, "model_settings": config.mining.model_settings, "mode": config.mode,
        "initial_stage_id": config.mining.stage_id,
        "mining_generated_tokens": config.mining.generated_token_limit,
        "max_model_calls": config.mining.max_model_calls,
        "tool_timeout_seconds": config.mining.tool_timeout_seconds,
        "worker_request_policy": config.mining.worker_request_policy,
        "max_consecutive_format_errors": config.mining.max_consecutive_format_errors,
        "detector_settings": config.mining.detector_settings,
        "recovery_generated_tokens": config.recovery_generated_tokens,
        "horizons": config.horizons, "repeat_count": len(config.repeats),
        "arms": config.arms, "observer_settings": config.observer_settings,
        "observer_audit": config.observer_audit, "packet_settings": config.packet_settings,
        "observer_source_policy": config.observer_source_policy,
        "observer_include_returned_reasoning": config.observer_include_returned_reasoning,
        "observer_context_policy": config.observer_context_policy,
        "live_observer": config.live_observer,
        "observer_context_counter": config.observer_context_counter,
        "observation_mask_policy": config.observation_mask_policy,
        "schedule_seed": config.schedule_seed,
        "analysis_settings": config.analysis_settings, "near_budget_tokens": config.near_budget_tokens,
        "resource_horizons": config.resource_horizons, "dollar_horizons": config.dollar_horizons,
    })


def validate_main_freeze(store: ArtifactStore, config: Any, task: Any) -> dict[str, Any] | None:
    if config.mining.purpose != "main":
        return None
    if config.freeze_ref is None:
        raise ValueError("Main execution requires an evidence-bound study freeze, not only calibration_status")
    freeze = store.get_json(config.freeze_ref)
    if freeze.get("version") == "study-freeze.v1":
        raise ValueError("Legacy study-freeze.v1 lacks enforced implementation bindings; issue a reviewed v2 freeze")
    if freeze.get("version") != "study-freeze.v2" or freeze.get("status") != "frozen":
        raise ValueError("Unsupported or unfrozen study manifest")
    if task.family_id in freeze.get("development_calibration_families", []):
        raise ValueError("Main family was used for development/calibration")
    if task.family_id not in freeze.get("main_families", []):
        raise ValueError("Main family absent from the frozen population")
    if settings_binding(config, task) not in freeze.get("settings_profiles", []):
        raise ValueError("Campaign conditions differ from the frozen study profiles")

    def checked_ref(value: Any) -> dict[str, Any]:
        ref = from_jsonable(ArtifactRef, value)
        data = store.get_bytes(ref)
        if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError("Study freeze evidence digest mismatch")
        return store.get_json(ref)

    from .provenance import fingerprint_digest, implementation_fingerprint, same_execution_binding
    if not freeze.get("implementation_ref"):
        raise ValueError("Study freeze lacks a declared implementation binding")
    implementation = checked_ref(freeze["implementation_ref"])
    frozen_fingerprint = implementation.get("fingerprint")
    if (implementation.get("version") != "implementation-binding.v2"
            or not isinstance(frozen_fingerprint, dict)
            or frozen_fingerprint.get("version") != "implementation-fingerprint.v2"
            or implementation.get("fingerprint_sha256") != fingerprint_digest(frozen_fingerprint)):
        raise ValueError("Study implementation binding is missing or inconsistent")
    implementation_digest = implementation["fingerprint_sha256"]
    source_artifacts = implementation.get("source_artifacts", {})
    sources = frozen_fingerprint.get("sources", [])
    if not sources or set(source_artifacts) != {s["path"] for s in sources}:
        raise ValueError("Study implementation binding lacks stored source evidence")
    for source in sources:
        ref = from_jsonable(ArtifactRef, source_artifacts[source["path"]])
        data = store.get_bytes(ref)
        if len(data) != source["byte_length"] or hashlib.sha256(data).hexdigest() != source["content_sha256"]:
            raise ValueError("Study implementation source evidence digest mismatch")
    if implementation_fingerprint() != frozen_fingerprint:
        raise ValueError("Current implementation differs from the frozen source/dependency conditions")

    checked_implementations = {from_jsonable(ArtifactRef, freeze["implementation_ref"])}
    def associated_implementation(value: Any) -> bool:
        if not value:
            return False
        ref = from_jsonable(ArtifactRef, value)
        if ref not in checked_implementations:
            if not same_execution_binding(checked_ref(value), implementation):
                return False
            checked_implementations.add(ref)
        return True

    for decision in REQUIRED_DECISIONS:
        value = freeze.get("decisions", {}).get(decision)
        if not value or not checked_ref(value):
            raise ValueError(f"Missing frozen scientific decision: {decision}")
    for gate in REQUIRED_GATES:
        value = freeze.get("gates", {}).get(gate)
        evidence = checked_ref(value) if value else {}
        if evidence.get("passed") is not True:
            raise ValueError(f"Study readiness evidence has not passed: {gate}")
        if not associated_implementation(evidence.get("implementation_ref")):
            raise ValueError(f"Study readiness evidence has a missing or different implementation: {gate}")
    # Keep all calibration outcomes. Coverage is about executing the same cohort,
    # not selectively retaining solves or stalls. The evidence gates above decide
    # whether that cohort actually calibrated the instrument satisfactorily.
    tasks = freeze.get("calibration_task_ids", [])
    models = freeze.get("calibration_model_ids", [])
    if len(set(tasks)) != 10 or len(tasks) != 10 or len(set(models)) < 2:
        raise ValueError("Freeze requires the ratified shared ten-task, multiple-model calibration cohort")
    coverage = set()
    for ref in freeze.get("calibration_run_receipts", []):
        receipt = checked_ref(ref)
        run_id = receipt["run_id"]
        result = store.get_record("run_result", run_id)
        if result != receipt:
            raise ValueError("Calibration receipt differs from its recorded result")
        start = next((e.payload for e in store.iter_events(run_id)
                      if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"), None)
        if not start or start.get("resume_from") or start.get("treatment"):
            raise ValueError("Calibration coverage requires a natural run")
        if start["config"].get("purpose") != "calibration":
            raise ValueError("Development fixtures are not calibration coverage")
        if start["config"]["model"].startswith(("fixture/", "scripted/", "dev/")):
            raise ValueError("Calibration coverage cannot be supplied by scripted workers")
        if start["task"]["family_id"] not in freeze.get("development_calibration_families", []):
            raise ValueError("Calibration family absent from exclusion list")
        campaign_id = start["config"].get("campaign_id")
        if not campaign_id:
            raise ValueError("Calibration receipt lacks its recorded implementation campaign association")
        try:
            calibration_implementation = store.get_record("campaign_implementation", campaign_id)
        except KeyError:
            raise ValueError("Calibration campaign lacks recorded implementation evidence") from None
        if not same_execution_binding(calibration_implementation, implementation):
            raise ValueError("Calibration implementation differs from the frozen source/dependency conditions")
        coverage.add((start["task"]["task_id"], start["config"]["model"]))
    if any((task_id, model) not in coverage for task_id in tasks for model in models):
        raise ValueError("Calibration cohort is incomplete across the frozen worker routes")
    return {"freeze_ref": config.freeze_ref, "implementation_ref": freeze["implementation_ref"],
        "implementation_digest": implementation_digest, "settings_digest": hashlib.sha256(
            canonical_json(settings_binding(config, task))).hexdigest(), "validated": True,
        "claim": "Frozen execution conditions and stored evidence associations; human approvals are trusted, not authenticated."}
