"""Portable evidence binding for task-owned admission packages.

This is an admission loader, not a scientific admission decision or a storage
schema. Issuers must review every gate and provide durable receipts first.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

from trajectory_retention.contracts import AdmissionStatus, TaskRecord, from_jsonable, to_jsonable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value: Any, root: Path) -> Any:
    """Normalize only local paths within the checkout; preserve all other text."""
    if isinstance(value, str):
        prefix = str(root.resolve()) + "/"
        return value[len(prefix):] if value.startswith(prefix) else value
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v, root) for v in value]
    return value


def _safe_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Admission evidence must use repository-relative paths")
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Admission evidence escaped its repository")
    return path


def load_admission(package: Path, provisional: TaskRecord) -> TaskRecord:
    """Return HOLD until a matching reviewed attestation exists; fail on tamper.

    The canonical layout is benchmark_tasks/packages/<id> alongside admission/.
    Candidate directories deliberately do not acquire admission from a name match.
    """
    from .calibration_compatibility import approved_calibration_record
    approved = approved_calibration_record(package, provisional)
    if approved is not None:
        return approved
    package = Path(package).resolve()
    if package.parent.name != "packages" or package.parent.parent.name != "benchmark_tasks":
        return provisional
    root = package.parent.parent.parent
    attestation = package.parent.parent / "admission" / (provisional.task_id + ".json")
    if not attestation.exists():
        return provisional
    data = json.loads(attestation.read_text())
    if data.get("record_version") != "task.admission.attestation.v1" or data.get("admission") != "ADMITTED":
        raise ValueError("Unrecognized or incomplete task attestation")
    if data["task_id"] != provisional.task_id or data["package_sha256"] != sha256(package / "package.json"):
        raise ValueError("Admission package identity changed")
    if data["package"] != package.relative_to(root).as_posix():
        raise ValueError("Admission package location changed")
    if not data.get("evidence") or not data.get("source_files"):
        raise ValueError("Admission requires durable evidence and implementation bindings")
    for binding in [*data["evidence"], *data["source_files"]]:
        if sha256(_safe_file(root, binding["path"])) != binding["sha256"]:
            raise ValueError("Admission evidence or implementation changed: " + binding["path"])
    record = from_jsonable(TaskRecord, data["task_record"])
    actual = portable(to_jsonable(provisional), root)
    expected = to_jsonable(record)
    for field in ("task_id", "family_id", "source", "source_revision", "task_version",
                  "environment_version", "stages", "lifecycle"):
        if actual[field] != expected[field]:
            raise ValueError("Admission task contract changed: " + field)
    if record.admission != AdmissionStatus.ADMITTED or record.unresolved:
        raise ValueError("Attested TaskRecord is not admitted")
    return record
