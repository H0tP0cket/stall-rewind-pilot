"""Load admitted adapters through explicit local registry entrypoints.

The registry selects code; the adapter still verifies its own admission seal.
An ADMITTED string in a registry is not a substitute for that verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import AdmissionStatus, TaskAdapter, TaskRecord, from_jsonable


@dataclass(frozen=True)
class RegisteredTask:
    entry: Mapping[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        value = self.entry[key]
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError(f"Registry {key} must be a relative path")
        result = (self.root / value).resolve()
        if not result.is_relative_to(self.root.resolve()):
            raise ValueError(f"Registry {key} escapes its artifact root")
        if not result.exists():
            raise ValueError(f"Registry {key} is missing: {value}")
        return result

    def load(self, *, scratch_root: Path, purpose: str) -> TaskAdapter:
        if self.entry.get("calibration_compatibility") is not None:
            from .tasks.calibration_compatibility import load_calibration_task
            return load_calibration_task(self, scratch_root=scratch_root, purpose=purpose)
        if self.entry.get("admission") != "ADMITTED":
            raise ValueError("Registry task is not admitted")
        module, separator, name = self.entry["adapter"].partition(":")
        if not separator or not name.isidentifier() or not module.startswith("trajectory_retention.tasks."):
            raise ValueError("Expected an explicit trajectory_retention.tasks module:class entrypoint")
        cls = getattr(importlib.import_module(module), name)
        adapter = cls(self.path("package"), scratch_root=scratch_root)
        actual = adapter.describe()
        declared = from_jsonable(TaskRecord, json.loads(self.path("task_record").read_text()))
        if actual.admission != AdmissionStatus.ADMITTED:
            raise ValueError("Adapter did not verify ADMITTED status")
        if (actual.task_id, actual.family_id) != (self.entry["task_id"], self.entry["family_id"]):
            raise ValueError("Registry identity differs from the loaded adapter")
        for field in ("task_id", "family_id", "source", "source_revision", "task_version",
                      "environment_version", "stages", "lifecycle", "runtime_class"):
            if getattr(actual, field) != getattr(declared, field):
                raise ValueError(f"Loaded adapter differs from admitted task record: {field}")
        validate_task_purpose(actual, purpose, eligibility=self.entry.get("eligibility", {}))
        return adapter


def validate_task_purpose(task: TaskRecord, purpose: str, *, eligibility: Mapping[str, Any] | None = None) -> None:
    if purpose not in {"development", "calibration", "main"}:
        raise ValueError(f"Unknown execution purpose: {purpose}")
    if purpose != "development" and task.admission != AdmissionStatus.ADMITTED:
        raise ValueError("Only admitted tasks may enter calibration/main")
    bindings = [task.evidence.get("eligibility", {})]
    if eligibility is not None:
        bindings.append(eligibility)
    if purpose in {"development", "calibration"}:
        # Legacy records without partition flags retain their original policy.
        # An explicit restriction is never bypassed by technical admission.
        if any("development_and_calibration" in b and b["development_and_calibration"] is not True for b in bindings):
            raise ValueError("Task partition does not permit development/calibration execution")
        split = task.evidence.get("family_split")
        if purpose == "calibration" and split is not None and split != "calibration":
            raise ValueError("Only calibration-partition tasks may enter calibration")
        if split == "held_out":
            raise ValueError("Held-out tasks cannot enter development/calibration")
    if purpose == "main":
        if task.evidence.get("family_split") in {"development", "calibration"}:
            raise ValueError("Development/calibration families cannot enter held-out main evaluation")
        if any(b.get("held_out_main_evaluation") is not True for b in bindings):
            raise ValueError("Main eligibility must be explicitly established by admission")


def read_registry(path: Path, *, root: Path | None = None) -> dict[str, RegisteredTask]:
    data = json.loads(path.read_text())
    if data.get("registry_version") != "task-admission.registry.v1":
        raise ValueError("Unsupported task registry version")
    root = root or path.resolve().parents[2]
    result = {}
    for entry in data["tasks"]:
        for key in ("task_id", "family_id", "adapter", "admission", "package", "task_record"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"Registry entry lacks {key}")
        if entry["task_id"] in result:
            raise ValueError("Duplicate registry task identity")
        result[entry["task_id"]] = RegisteredTask(entry, root)
    return result
