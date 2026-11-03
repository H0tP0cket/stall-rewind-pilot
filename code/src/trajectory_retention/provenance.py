"""Read-only execution fingerprints and separately stored source evidence.

Dependency identity hashes installed distribution bytes, not only versions or
RECORD's claimed hashes. It is not an archived wheel or an authentication of
human approval. Host labels are diagnostic and excluded from the fingerprint.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import sysconfig

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .contracts import canonical_json, to_jsonable
from .store import ArtifactStore


DEPENDENCY_ROOTS = ("mini-swe-agent", "requests", "pydantic", "jinja2", "packaging")
OPTIONAL_DEPENDENCY_ROOTS = ("tiktoken",)


def _digest(path: Path) -> dict:
    data = path.read_bytes()
    # These are measurements of external bytes, not content-store references.
    return {"content_sha256": hashlib.sha256(data).hexdigest(), "byte_length": len(data)}


def _source_files() -> tuple[Path, list[Path]]:
    package_root = Path(__file__).resolve().parent
    repository = package_root.parent.parent
    files = sorted(package_root.rglob("*.py"))
    files += [repository / "pyproject.toml", repository / "uv.lock"]
    return repository, sorted(files)


def _dependencies() -> list[dict]:
    pending = [(name, frozenset(), name in OPTIONAL_DEPENDENCY_ROOTS)
               for name in (*DEPENDENCY_ROOTS, *OPTIONAL_DEPENDENCY_ROOTS)]
    expanded, records = set(), {}
    while pending:
        name, extras, optional = pending.pop()
        name = canonicalize_name(name)
        key = (name, extras)
        if key in expanded:
            continue
        expanded.add(key)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if not optional:
                raise ValueError(f"Required installed dependency is missing: {name}") from None
            records[name] = {"name": name, "installed": False}
            continue
        if name not in records:
            direct_url = getattr(distribution, "read_text", lambda name: None)("direct_url.json")
            if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable") is True:
                raise ValueError(f"Editable dependency requires an explicit source binding: {name}")
            if not distribution.files:
                raise ValueError(f"Installed dependency lacks a file inventory: {name}")
            files = []
            for relative in sorted(distribution.files, key=str):
                # Console-script shebangs, bytecode and installation bookkeeping
                # contain host paths or are regenerated; imported library files,
                # native extensions, package data and METADATA/WHEEL are bound.
                if (".." in relative.parts or "__pycache__" in relative.parts
                        or relative.suffix in {".pyc", ".pyo"}
                        or (any(part.endswith(".dist-info") for part in relative.parts)
                            and relative.name in {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"})):
                    continue
                path = Path(distribution.locate_file(relative))
                if not path.is_file():
                    raise ValueError(f"Installed dependency file is missing: {name}/{relative}")
                files.append({"path": relative.as_posix(), **_digest(path)})
            if not files:
                raise ValueError(f"Installed dependency has no bindable library files: {name}")
            records[name] = {"name": name, "installed": True, "version": distribution.version, "files": files}
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)):
                continue
            pending.append((requirement.name, frozenset(requirement.extras), False))
    return [records[name] for name in sorted(records)]


def implementation_fingerprint() -> dict:
    """Measure current conditions without writing artifacts, files or a store.

    Requires a source checkout with its declared lock. This binds project source,
    installed library distributions and interpreter identity, not the entire host
    filesystem. Task OCI/evaluator bindings remain separate admission conditions.
    """
    repository, files = _source_files()
    return {"version": "implementation-fingerprint.v2",
        "sources": [{"path": p.relative_to(repository).as_posix(), **_digest(p)} for p in files],
        "dependency_roots": list(DEPENDENCY_ROOTS), "optional_dependency_roots": list(OPTIONAL_DEPENDENCY_ROOTS),
        "dependencies": _dependencies(),
        "python": {"implementation": sys.implementation.name, "version": list(sys.version_info[:3]),
                   "cache_tag": sys.implementation.cache_tag, "abi": sysconfig.get_config_var("SOABI"),
                   "platform": sys.platform, "machine": platform.machine(),
                   "executable": _digest(Path(sys.executable).resolve())}}


def fingerprint_digest(fingerprint: dict) -> str:
    return hashlib.sha256(canonical_json(fingerprint)).hexdigest()


def same_execution_binding(actual: dict, expected: dict) -> bool:
    """Compare measured/stored conditions, excluding diagnostic host metadata."""
    return (actual.get("version") == expected.get("version") == "implementation-binding.v2"
            and actual.get("fingerprint_sha256") == expected.get("fingerprint_sha256")
            and actual.get("fingerprint") == expected.get("fingerprint")
            and to_jsonable(actual.get("source_artifacts")) == to_jsonable(expected.get("source_artifacts")))


def implementation_binding(store: ArtifactStore) -> dict:
    fingerprint = implementation_fingerprint()
    repository, _ = _source_files()
    artifacts = {}
    for entry in fingerprint["sources"]:
        data = (repository / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["content_sha256"] or len(data) != entry["byte_length"]:
            raise ValueError("Implementation changed while source evidence was captured")
        artifacts[entry["path"]] = store.put_bytes(data, media_type="text/plain")
    return {"version": "implementation-binding.v2", "fingerprint": fingerprint,
            "fingerprint_sha256": fingerprint_digest(fingerprint), "source_artifacts": artifacts,
            "dependency_evidence": "measured installed file digests; dependency bytes are not archived",
            "host_metadata": {"python_build": sys.version, "system": platform.system(),
                              "release": platform.release(), "node": platform.node()}}
