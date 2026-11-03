"""One declared-source guard for fresh batch, preflight and single-run entry."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping


def _binding(config: Any) -> Mapping[str, Any] | None:
    binding = config.mining.metadata.get("source_binding")
    if binding is not None and not isinstance(binding, Mapping):
        raise ValueError("Launch source binding must be a declared object")
    return binding


def verify_declared_sources(config: Any, registered: Any) -> None:
    binding = _binding(config)
    if binding is None:
        return  # Legacy inputs still undergo the adapter's admission checks.
    for name in ("task_id", "family_id", "adapter"):
        if binding.get(name) != registered.entry.get(name):
            raise ValueError(f"Launch source binding differs from registry: {name}")
    for name in ("task_record", "package_manifest", "attestation"):
        ref = binding.get(name)
        if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("sha256"), str):
            raise ValueError(f"Launch source reference is malformed: {name}")
        source = (registered.root / ref["path"]).resolve()
        if not source.is_relative_to(registered.root.resolve()) or not source.is_file():
            raise ValueError(f"Launch source is missing or escapes artifact root: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError(f"Launch source hash changed: {name}")
        if name == "task_record" and source != registered.path("task_record"):
            raise ValueError("Launch task record differs from registry task record")


def verify_loaded_task(config: Any, record: Any) -> None:
    binding = _binding(config)
    if binding is None:
        return
    if any(getattr(record, name) != binding.get(name) for name in ("task_id", "family_id", "environment_version")):
        raise ValueError("Loaded task differs from pinned launch source identity")
    versions = binding.get("evaluator_versions")
    if versions is not None and sorted({s.evaluator_version for s in record.stages}) != sorted(versions):
        raise ValueError("Loaded task differs from pinned launch evaluator versions")
