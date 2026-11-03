"""One generic adapter for every packaged multi-bug SWE-smith repair task.

Package layout (benchmark_tasks/packages/swesmith-long-v1/<task_id>/):
  package.json            manifest binding every public/private file and evaluator source
  public/instructions.md  the dataset problem_statement, verbatim
  public/workspace/       natural start: clean mirror tree + install-generated files + bug patch
  private/                bug.patch, source-row.json, checks.json, grading.json, pristine/ test support
The admission seal lives outside the package, at
benchmark_tasks/admission/swesmith-long-v1/tasks/<task_id>/admission.json.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, LifecyclePolicy, RuntimeClass, StageSpec, TaskRecord,
                                            to_jsonable)
from .grading import CONFIG_NAMES, observe, run_suite, translate
from .scoring import F2PFractionScorer

LANE = "swesmith-long-v1"
ADAPTER_ENTRY = "trajectory_retention.tasks.swesmith_long.adapter:SWESmithLongTaskAdapter"
EVALUATOR_SOURCES = ("adapter.py", "grading.py", "scoring.py", "grader/grade_driver.py", "grader/swesmith_long_recorder.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(Path(root).rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[rel] = "symlink:" + str(path.readlink())
        elif path.is_file():
            result[rel] = sha256(path)
    return result


def evaluator_sources() -> dict[str, str]:
    here = Path(__file__).parent
    return {name: sha256(here / name) for name in EVALUATOR_SOURCES}


def verify_package(package: Path) -> dict:
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("schema_version") != "swesmith-long.package.v1":
        raise ValueError("Unsupported package schema")
    for section in ("public", "private"):
        if file_manifest(package / section) != manifest[section + "_files"]:
            raise ValueError(f"Frozen package changed: {section}")
    if evaluator_sources() != manifest["evaluator_sources"]:
        raise ValueError("Frozen evaluator source changed; issue a new package version")
    return manifest


def seal_path(package: Path, task_id: str) -> Path:
    return package.parents[2] / "admission" / LANE / "tasks" / task_id / "admission.json"


def verify_seal(package: Path, manifest: dict) -> dict | None:
    """Return the verified seal, None when absent; raise on any binding mismatch."""
    path = seal_path(package, manifest["task_id"])
    if not path.is_file():
        return None
    seal = json.loads(path.read_text())
    if seal.get("schema_version") != "swesmith-long.admission.v1" or seal.get("admission") != "ADMITTED":
        raise ValueError("Unrecognized admission seal")
    if seal["task_id"] != manifest["task_id"] or seal["package_sha256"] != sha256(package / "package.json"):
        raise ValueError("Admission seal does not bind this package")
    root = package.parents[2]
    for binding in seal["evidence"]:
        target = (root / binding["path"]).resolve()
        if not target.is_relative_to(root.resolve()) or sha256(target) != binding["sha256"]:
            raise ValueError("Admission evidence changed: " + binding["path"])
    if seal["evaluator_sources"] != evaluator_sources():
        raise ValueError("Admission seal binds different evaluator sources")
    if seal["split"] not in ("calibration", "held_out"):
        raise ValueError("Unknown split")
    return seal


class SWESmithLongTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.task_id = self.manifest["task_id"]
        self.seal = verify_seal(self.package, self.manifest)
        self.scratch_root = Path(scratch_root or "artifacts/admission-scratch/swesmith-long").resolve()
        self.grading = json.loads((self.package / "private/grading.json").read_text())
        self.scorer = F2PFractionScorer(self.package / "private/checks.json", task_id=self.task_id,
                                        binding=json.dumps({"image": self.manifest["environment_image_id"],
                                                            "sources": self.manifest["evaluator_sources"],
                                                            "grading": self.grading}, sort_keys=True))
        text = (self.package / "public/instructions.md").read_text()
        self._stage = StageSpec(stage_id="repair", target_id=self.task_id + ":repair", ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.scorer.version,
            metadata={"upstream_instance_id": self.manifest["upstream_instance_id"], "private_grade_feedback": "none",
                      "visible_feedback_command": self.manifest["visible_feedback_command"],
                      "public_tests": "All upstream tests, including FAIL_TO_PASS files, are present in W (disclosed adaptation); "
                                      "grading restores the frozen test-support set first."})

    def describe(self) -> TaskRecord:
        m = self.manifest
        seal = self.seal or {}
        admitted = bool(seal)
        split = seal.get("split", m["planned_split"])
        return TaskRecord(task_id=self.task_id, family_id=m["family_id"], source=m["source_url"],
            source_revision=m["dataset_revision"], task_version=m["package_version"],
            environment_version=m["environment_image_id"], stages=(self._stage,),
            lifecycle=LifecyclePolicy(policy_id="swesmith-long.single-repair.v1", version="1", origin="benchmark_native",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry_as_H_substream", "T": "external_only"},
                detector_scope_rule="One fixed repair target per trajectory; no native stage boundary.",
                stage_release_rule="Single repair episode; no additional specifications or reference insertion.",
                source_evidence=(m["swesmith_source"],)),
            admission=AdmissionStatus.ADMITTED if admitted else AdmissionStatus.HOLD,
            runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": split,
                      "eligibility": {"development_and_calibration": admitted and split == "calibration",
                                      "held_out_main_evaluation": False},
                      "package_sha256": sha256(self.package / "package.json"),
                      "upstream_instance_id": m["upstream_instance_id"], "adaptation_version": m["adaptation"]["version"],
                      "grading_runtime_policy": "Only W is graded, in a fresh copy of the frozen image (SWE-smith native "
                                                "evaluation); engine-captured R is restored for worker continuation only.",
                      "network_policy": "worker and grader containers run with network none",
                      "main_study_frozen": False},
            unresolved=() if admitted else ("Lane admission seal absent or pending",))

    def worker_prompt(self, stage_id: str):
        from ..prompts import swesmith_prompt
        return swesmith_prompt(self.stage(stage_id))

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != self._stage.stage_id:
            raise ValueError("Unknown stage")
        return self._stage

    def next_stage(self, stage_id: str, *, termination_reason: str) -> StageSpec | None:
        self.stage(stage_id)
        return None

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or "repair")
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir()))):
            raise ValueError("Natural start requires an empty real workspace")
        verify_package(self.package)
        shutil.copytree(self.package / "public/workspace", workspace, symlinks=True, dirs_exist_ok=True)
        return stage

    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None, runtime=None):
        stage = self.stage(stage_id)
        verify_package(self.package)
        workspace = Path(workspace).resolve()
        scratch = self.scratch_root
        if scratch == workspace or scratch.is_relative_to(workspace):
            raise ValueError("Private scratch must be outside submitted W")
        if private_output_dir is None:
            scratch.mkdir(parents=True, exist_ok=True)
            private_output_dir = Path(tempfile.mkdtemp(dir=scratch, prefix="grade-")) / "private"
        output = Path(private_output_dir).resolve()
        if output == workspace or output.is_relative_to(workspace):
            raise ValueError("Private grade output must be outside submitted W")
        spec = {**{k: self.grading[k] for k in ("protected_dirs", "protected_files", "pytest_args", "test_files",
                                                 "timeout_seconds")}, "config_names": list(CONFIG_NAMES),
                "restore_test_support": True}
        receipt = run_suite(self.manifest["environment_image_id"], workspace, output,
                            pristine=self.package / "private/pristine", spec=spec)
        observation = observe(output)
        checks = translate(self.scorer.expected, observation, receipt)
        result = self.scorer.score(checks, stage=stage)
        result = replace(result, details={**result.details, "private_output_dir": str(output),
            "grade_wall_seconds": receipt.get("wall_seconds"),
            "runtime_export_consumed": False,
            "runtime_policy": "SWE-smith native: W graded in a fresh frozen image; R not part of the submission",
            "collection_errors": [e["nodeid"] for e in observation["collect_errors"]],
            "timed_out": bool((receipt.get("driver") or {}).get("timed_out")),
            "infrastructure_error": receipt.get("infrastructure_error")})
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
        return result
