"""Declared and executed recovery conditions, without inferred run defaults."""
from typing import Any, Mapping

from ..contracts import TreatmentManifest
from ..experiments.treatments import recovery_settings_known


def execution_problems(lineage: Mapping[str, Any]) -> list[str]:
    problems = []
    if lineage.get("worker_policy_bindings_verified") is not True:
        problems.append("worker_policy_binding_missing_or_inconsistent")
    declared = lineage.get("declared_recovery_settings")
    actual = lineage.get("executed_recovery_settings")
    if not recovery_settings_known(declared):
        problems.append("declared_recovery_settings_unknown")
    if not recovery_settings_known(actual):
        problems.append("executed_recovery_settings_unknown")
    if recovery_settings_known(declared) and recovery_settings_known(actual) and declared != actual:
        problems.append("executed_recovery_settings_mismatch")
    if not isinstance(lineage.get("effective_detector_config"), Mapping):
        problems.append("effective_detector_config_unknown")
    if not isinstance(lineage.get("executed_model_settings"), Mapping):
        problems.append("executed_model_settings_unknown")
    elif lineage["executed_model_settings"] != lineage.get("model_settings"):
        problems.append("executed_model_settings_mismatch")
    if not lineage.get("executed_worker_model"):
        problems.append("executed_worker_model_unknown")
    elif lineage["executed_worker_model"] != lineage.get("declared_worker_model"):
        problems.append("executed_worker_model_mismatch")
    return problems


def condition_identity(lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Fields that cannot be pooled in a curve or matched comparison."""
    return {key: lineage.get(key) for key in (
        "declared_recovery_settings", "executed_recovery_settings", "effective_detector_config",
        "executed_model_settings", "executed_worker_model", "worker_policy_bindings_verified")}


def manifest_execution_problems(manifest: TreatmentManifest, lineage: Mapping[str, Any]) -> list[str]:
    problems = execution_problems(lineage)
    if lineage.get("declared_recovery_settings") != manifest.metadata.get("recovery_settings"):
        problems.append("continuation_declared_recovery_settings_mismatch")
    if lineage.get("declared_worker_model") != manifest.worker_model:
        problems.append("continuation_declared_worker_model_mismatch")
    return problems
