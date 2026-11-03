"""Explicit calibration-only execution of unchanged tasks on a revised instrument.

Historical admission is retained, not reissued. A dedicated registry pins every
package byte and the actual current implementation. Normal/main loading never
uses this path. Context is local to one adapter call, including private grading.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import hashlib
import importlib
import json
from pathlib import Path

from ..contracts import AdmissionStatus, TaskRecord, from_jsonable, to_jsonable

VERSION = "calibration-instrument-compatibility.v1"
KEY = "calibration_compatibility"
_active = ContextVar("calibration_task_binding", default=None)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    """All files and symlink targets, including non-public package metadata."""
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[name] = {"symlink": str(path.readlink())}
        elif path.is_file():
            result[name] = {"sha256": digest(path)}
    return result


def _path(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Calibration binding must use repository-relative paths")
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Calibration binding escapes repository")
    return path


def _read_ref(root, ref):
    if type(ref) is not dict or set(ref) != {"path", "sha256"}:
        raise ValueError("Exact calibration reference required")
    path = _path(root, ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError("Calibration reference changed: " + ref["path"])
    return json.loads(path.read_bytes())


class CalibrationBinding:
    def __init__(self, registered):
        self.root = registered.root.resolve()
        self.ref = dict(registered.entry[KEY])
        self.document = _read_ref(self.root, self.ref)
        doc = self.document
        if (doc.get("version") != VERSION or doc.get("purpose") != "calibration"
                or doc.get("scientific_freeze") is not False):
            raise ValueError("Explicit unfrozen calibration association required")
        self.row = doc["tasks"][registered.entry["task_id"]]
        self.package = registered.path("package").resolve()
        if self.package != _path(self.root, self.row["package"]):
            raise ValueError("Calibration package location differs")
        if registered.entry["adapter"] != self.row["adapter"]:
            raise ValueError("Calibration adapter differs")
        self.attestation = _read_ref(self.root, self.row["historical_attestation"])
        self.record = from_jsonable(TaskRecord, self.attestation["task_record"])
        if (self.attestation.get("record_version") != "task.admission.attestation.v1"
                or self.attestation.get("admission") != "ADMITTED"
                or self.attestation.get("task_id") != registered.entry["task_id"]
                or self.attestation.get("package") != self.row["package"]
                or self.record.admission != AdmissionStatus.ADMITTED
                or self.record.unresolved
                or self.record.task_id != registered.entry["task_id"]
                or self.record.family_id != registered.entry["family_id"]):
            raise ValueError("Calibration historical task identity differs")
        from .source_closure import source_closure
        prefix = "src/trajectory_retention/"
        required = {r["path"] for r in self.attestation["source_files"] if r["path"].startswith(prefix)}
        required.update(prefix + p for p in ("tasks/calibration_compatibility.py", "registry.py", "factory.py"))
        required.update(prefix + p for p in source_closure(self.root / prefix,
            [p[len(prefix):] for p in required if p.endswith(".py")]))
        if not required <= set(doc.get("current_sources", {})):
            raise ValueError("Calibration current source closure is incomplete")
        declared = _read_ref(self.root, self.row["task_record"])
        if declared != to_jsonable(self.record):
            raise ValueError("Calibration historical TaskRecord differs")
        if registered.path("task_record") != _path(self.root, self.row["task_record"]["path"]):
            raise ValueError("Calibration registry TaskRecord differs")
        self.manifest = None
        self.validate()

    def validate(self):
        if _read_ref(self.root, self.ref) != self.document:
            raise ValueError("Calibration association changed")
        for path, expected in self.document["current_sources"].items():
            if digest(_path(self.root, path)) != expected:
                raise ValueError("Calibration implementation changed: " + path)
        if inventory(self.package) != self.row["package_files"]:
            raise ValueError("Calibration package bytes changed")
        manifest_path = self.package / "package.json"
        if digest(manifest_path) != self.attestation["package_sha256"]:
            raise ValueError("Calibration historical package hash differs")
        self.manifest = json.loads(manifest_path.read_bytes())
        # Historical receipts are immutable evidence, never current-run claims.
        if _read_ref(self.root, self.row["historical_attestation"]) != self.attestation:
            raise ValueError("Historical admission changed")
        for ref in self.attestation["evidence"]:
            if digest(_path(self.root, ref["path"])) != ref["sha256"]:
                raise ValueError("Historical admission evidence changed: " + ref["path"])
        for ref in self.document.get("validation_evidence", []):
            path = _path(self.root, ref["path"])
            if digest(path) != ref["sha256"]:
                raise ValueError("Calibration validation evidence changed")

    @contextmanager
    def activate(self):
        if _active.get() is self:
            yield
            return
        self.validate()
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    def annotation(self):
        return {"version": VERSION, "association": self.ref,
                "historical_attestation": self.row["historical_attestation"],
                "current_implementation": "association.current_sources",
                "scope": "Unchanged admitted task on revised calibration instrument; no new scientific admission or freeze",
                "scientific_freeze": False, "purpose": "calibration"}


def approved_calibration_package(package):
    binding = _active.get()
    if binding is None:
        return None
    if Path(package).resolve() != binding.package:
        raise ValueError("Calibration verifier requested a foreign package")
    # Return a fresh value so mutable adapter state cannot change the binding.
    return json.loads(json.dumps(binding.manifest))


def approved_calibration_record(package, provisional):
    binding = _active.get()
    if binding is None:
        return None
    approved_calibration_package(package)
    from .evidence import portable
    actual, expected = portable(to_jsonable(provisional), binding.root), to_jsonable(binding.record)
    for field in ("task_id", "family_id", "source", "source_revision", "task_version",
                  "environment_version", "stages", "lifecycle"):
        if actual[field] != expected[field]:
            raise ValueError("Calibration task semantics changed: " + field)
    return replace(binding.record, evidence={**binding.record.evidence,
        KEY: binding.annotation(), "eligibility": {"development_and_calibration": True,
        "held_out_main_evaluation": False, "calibration_settings_frozen": False}})


def load_calibration_task(registered, *, scratch_root, purpose):
    if purpose != "calibration":
        raise ValueError("Calibration compatibility cannot authorize development/main")
    binding = CalibrationBinding(registered)
    module, _, name = registered.entry["adapter"].partition(":")
    if not module.startswith("trajectory_retention.tasks.") or not name.isidentifier():
        raise ValueError("Explicit task entrypoint required")
    cls = getattr(importlib.import_module(module), name)
    with binding.activate():
        adapter = cls(binding.package, scratch_root=scratch_root)
        record = adapter.describe()
        if record.evidence.get(KEY) != binding.annotation():
            raise ValueError("Adapter did not expose current calibration association")
        for field in ("task_id", "family_id", "source", "source_revision", "task_version",
                      "environment_version", "stages", "lifecycle", "runtime_class"):
            if getattr(record, field) != getattr(binding.record, field):
                raise ValueError("Loaded calibration task contract differs: " + field)
    # Instance-local wrappers retain the original adapter type and detector
    # policy identities. No shared/global verifier or task class is patched.
    for method in ("describe", "stage", "materialize", "grade", "next_stage", "worker_prompt"):
        if method == "worker_prompt" and not hasattr(adapter, method):
            continue
        original = getattr(adapter, method)
        def wrap(function):
            @wraps(function)
            def invoke(*args, **kwargs):
                with binding.activate():
                    return function(*args, **kwargs)
            return invoke
        setattr(adapter, method, wrap(original))
    return adapter
