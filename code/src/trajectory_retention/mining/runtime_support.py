"""Bind task-declared R support to the state actually captured at a boundary."""
from __future__ import annotations

from typing import Any, Mapping
import hashlib

from ..contracts import ArtifactRef, RuntimeClass, StageSpec, TaskRecord, canonical_json, from_jsonable


def bind_runtime_support(task: TaskRecord, stage: StageSpec, capture: Mapping[str, Any], *, store: Any = None) -> dict[str, Any]:
    # Rebinding an already classified export uses its original measurement.
    prior = capture.get("runtime_support", {})
    measured = RuntimeClass(prior.get("measured_class", capture["runtime_class"]))
    declared = task.runtime_class
    evidence = task.evidence.get("runtime_irrelevance")
    valid_irrelevance = (isinstance(evidence, dict)
        and evidence.get("version") == "runtime-irrelevance.v1"
        and evidence.get("task_id") == task.task_id
        and evidence.get("task_version") == task.task_version
        and isinstance(evidence.get("stage_ids"), (list, tuple)) and stage.stage_id in evidence["stage_ids"]
        and isinstance(evidence.get("reason"), str) and bool(evidence["reason"].strip())
        and isinstance(evidence.get("evidence_refs"), (list, tuple)) and bool(evidence["evidence_refs"]))
    if valid_irrelevance:
        try:
            if store is None:
                raise ValueError("Runtime irrelevance evidence must be readable")
            for raw in evidence["evidence_refs"]:
                store.get_bytes(from_jsonable(ArtifactRef, raw))
        except (KeyError, ValueError, TypeError, OSError):
            valid_irrelevance = False
    effective = RuntimeClass.UNSUPPORTED
    if measured == RuntimeClass.UNSUPPORTED:
        reason = "captured_state_contains_unsupported_runtime"
    elif declared == RuntimeClass.UNSUPPORTED:
        reason = "task_declares_runtime_unsupported"
    elif declared == RuntimeClass.IRRELEVANT:
        reason = "task_bound_irrelevance_evidence" if valid_irrelevance else "runtime_irrelevance_not_demonstrated"
        if valid_irrelevance:
            effective = RuntimeClass.IRRELEVANT
    elif measured == RuntimeClass.CAPTURED:
        effective, reason = RuntimeClass.CAPTURED, "task_declaration_and_measured_capture_agree"
    else:
        reason = "required_runtime_capture_not_demonstrated"
    binding = {"revision": "declared-measured-runtime.v1", "task_id": task.task_id,
        "task_version": task.task_version, "stage_id": stage.stage_id,
        "declared_class": declared.value, "measured_class": measured.value,
        "effective_class": effective.value, "reason": reason,
        "irrelevance_evidence": evidence if valid_irrelevance else None}
    binding["binding_id"] = hashlib.sha256(canonical_json(binding)).hexdigest()
    result = {**capture, "runtime_class": effective.value, "runtime_support": binding}
    if valid_irrelevance:
        result["irrelevance_evidence"] = evidence
    return result
