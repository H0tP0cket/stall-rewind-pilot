"""v5 lane loader for the tasks first issued in the v5 release (published-hard-band selection).

v5 packages are built by the unchanged v4 pipeline steps (v3 package format:
schema ``swesmith-long.package.v3``, the v3 evaluator sources, ``grade_driver_v3``,
the one-commit ``public/workspace-git.tar``, the git-enabled family image,
``benchmark-task-prompt.v3``), so ``adapter_v3`` grades, materializes and
prompts them unchanged. The only difference is where the admission seal lives:
``admission/swesmith-long-v5`` (v3/v4 sealed evidence must not change); v5 seals
also bind this file's SHA-256. v3 and v4 tasks keep loading through their own
adapters.

Package: benchmark_tasks/packages/swesmith-long-v5/<task_id>/. Seal:
benchmark_tasks/admission/swesmith-long-v5/tasks/<task_id>/admission.json.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .adapter import sha256
from .adapter_v3 import SCHEMA, SWESmithLongV3TaskAdapter, evaluator_sources

LANE = "swesmith-long-v5"
ADAPTER_ENTRY = "trajectory_retention.tasks.swesmith_long.adapter_v5:SWESmithLongV5TaskAdapter"
SEAL_SCHEMA = "swesmith-long.admission.v5"
PACKAGE_FORMAT = SCHEMA  # swesmith-long.package.v3


def loader_sources() -> dict[str, str]:
    return {"adapter_v5.py": sha256(Path(__file__))}


def seal_path(package: Path, task_id: str) -> Path:
    return package.parents[2] / "admission" / LANE / "tasks" / task_id / "admission.json"


def verify_seal(package: Path, manifest: dict) -> dict | None:
    path = seal_path(package, manifest["task_id"])
    if not path.is_file():
        return None
    seal = json.loads(path.read_text())
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("admission") != "ADMITTED":
        raise ValueError("Unrecognized admission seal")
    if seal["task_id"] != manifest["task_id"] or seal["package_sha256"] != sha256(package / "package.json"):
        raise ValueError("Admission seal does not bind this package")
    if seal.get("package_format") != PACKAGE_FORMAT:
        raise ValueError("Admission seal binds a different package format")
    root = package.parents[2]
    for binding in seal["evidence"]:
        target = (root / binding["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != binding["sha256"]:
            raise ValueError("Admission evidence changed: " + binding["path"])
    if seal["evaluator_sources"] != evaluator_sources():
        raise ValueError("Admission seal binds different evaluator sources")
    if seal.get("loader_sources") != loader_sources():
        raise ValueError("Admission seal binds a different v5 loader")
    if seal["split"] not in ("calibration", "held_out"):
        raise ValueError("Unknown split")
    return seal


class SWESmithLongV5TaskAdapter(SWESmithLongV3TaskAdapter):
    """adapter_v3 behaviour; seal read from the v5 lane admission directory."""

    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        super().__init__(package, scratch_root=scratch_root or Path("artifacts/admission-scratch/swesmith-long-v5"))
        if self.manifest.get("adapter_entry") != ADAPTER_ENTRY:
            raise ValueError("Package does not declare the v5 loader")
        if self.seal is not None:
            raise ValueError("A v5-release package must not carry a v3 seal")
        self.seal = verify_seal(self.package, self.manifest)

    def describe(self):
        record = super().describe()
        return replace(record, evidence={**record.evidence, "lane_release": LANE, "package_format": PACKAGE_FORMAT,
                                         "loader_sources": loader_sources()})
