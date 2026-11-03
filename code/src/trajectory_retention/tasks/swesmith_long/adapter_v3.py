"""v3 adapter: v2's native SWE-smith start inside a fresh Git repository, native instance template.

v1/v2 adapters, packages, seals and registries are unchanged. Relative to v2:

* W is v2's worker start (FAIL_TO_PASS test files withheld) plus ``.git``: a
  fresh repository whose only commit holds exactly the tracked start files
  (fixed neutral identity and date; no upstream history). The frozen ``.git``
  bytes are ``public/workspace-git.tar``; ``materialize`` extracts them and
  gives every start file the fixed modification time START_MTIME that the
  frozen index records, so ``git status`` is clean without rewriting the index.
* The worker image is the v1/v2 image plus Debian's ``git`` and a system Git
  configuration (``environment_git`` in package.json).
* Grading skips every ``.git`` path of the submitted W (grader/grade_driver_v3.py);
  everything else is v2's grader.
* The worker prompt is the native SWE-smith instance template
  (``prompts.swesmith_prompt_v3``, benchmark-task-prompt.v3); the rendered text is
  bound by SHA-256 in package.json.

Package: benchmark_tasks/packages/swesmith-long-v3/<task_id>/. Seal:
benchmark_tasks/admission/swesmith-long-v3/tasks/<task_id>/admission.json.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, LifecyclePolicy, RuntimeClass, StageSpec, TaskRecord,
                                            to_jsonable)
from .adapter import file_manifest, sha256
from .grading import CONFIG_NAMES, observe, translate
from .grading_v3 import run_suite
from .scoring import F2PFractionScorer

LANE = "swesmith-long-v3"
ADAPTER_ENTRY = "trajectory_retention.tasks.swesmith_long.adapter_v3:SWESmithLongV3TaskAdapter"
EVALUATOR_SOURCES = ("adapter_v3.py", "grading.py", "grading_v3.py", "scoring.py", "grader/grade_driver.py",
                     "grader/grade_driver_v3.py", "grader/swesmith_long_recorder.py")
SCHEMA = "swesmith-long.package.v3"
GIT_TAR = "public/workspace-git.tar"
START_MTIME = 946684800          # 2000-01-01T00:00:00Z: every start file and directory
GIT_MTIME = START_MTIME + 60     # every .git entry; later than the files, so no index entry is "racily clean"


def evaluator_sources() -> dict[str, str]:
    here = Path(__file__).parent
    return {name: sha256(here / name) for name in EVALUATOR_SOURCES}


def git_members(tar_path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(tar_path, mode="r:") as archive:
        return _checked_members(archive)


def _checked_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Members of the frozen .git archive; only regular files and directories under .git/."""
    members = archive.getmembers()
    for member in members:
        name = member.name
        if not (name == ".git" or name.startswith(".git/")) or ".." in name.split("/") or not (member.isfile() or member.isdir()):
            raise ValueError("Frozen .git archive holds an unexpected entry: " + name)
        if member.mtime != GIT_MTIME or member.uid or member.gid:
            raise ValueError("Frozen .git archive entry has unexpected metadata: " + name)
    return members


def verify_package(package: Path) -> dict:
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported package schema")
    for section in ("public", "private"):
        if file_manifest(package / section) != manifest[section + "_files"]:
            raise ValueError(f"Frozen package changed: {section}")
    if evaluator_sources() != manifest["evaluator_sources"]:
        raise ValueError("Frozen evaluator source changed; issue a new package version")
    workspace = package / "public/workspace"
    if any(p.name == ".git" for p in workspace.rglob("*")):
        raise ValueError("Git metadata must come only from the frozen archive")
    if sha256(package / GIT_TAR) != manifest["git_start"]["archive_sha256"]:
        raise ValueError("Frozen .git archive differs from the manifest")
    git_members(package / GIT_TAR)
    withheld = json.loads((package / "private/withheld.json").read_text())
    for row in withheld["files"]:
        if (workspace / row["path"]).exists() or (workspace / row["path"]).is_symlink():
            raise ValueError("Withheld FAIL_TO_PASS test file present in the worker start: " + row["path"])
        if sha256(package / "private/pristine" / row["path"]) != row["sha256"]:
            raise ValueError("Private copy of a withheld test file changed: " + row["path"])
    return manifest


def seal_path(package: Path, task_id: str) -> Path:
    return package.parents[2] / "admission" / LANE / "tasks" / task_id / "admission.json"


def verify_seal(package: Path, manifest: dict) -> dict | None:
    path = seal_path(package, manifest["task_id"])
    if not path.is_file():
        return None
    seal = json.loads(path.read_text())
    if seal.get("schema_version") != "swesmith-long.admission.v3" or seal.get("admission") != "ADMITTED":
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


def set_start_mtimes(root: Path) -> None:
    """Give every entry below root (and root) the fixed start time; never follow links."""
    for path in [*sorted(Path(root).rglob("*")), Path(root)]:
        os.utime(path, (START_MTIME, START_MTIME), follow_symlinks=False)


def extract_git(tar_path: Path, workspace: Path) -> None:
    if (workspace / ".git").exists() or (workspace / ".git").is_symlink():
        raise ValueError("Start tree already holds .git")
    with tarfile.open(tar_path, mode="r:") as archive:
        members = _checked_members(archive)
        archive.extractall(workspace, members=members, filter="tar")
    for member in members:  # directories and read-only objects keep their archived mode and time
        os.utime(workspace / member.name, (GIT_MTIME, GIT_MTIME), follow_symlinks=False)


class SWESmithLongV3TaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.task_id = self.manifest["task_id"]
        self.seal = verify_seal(self.package, self.manifest)
        self.scratch_root = Path(scratch_root or "artifacts/admission-scratch/swesmith-long-v3").resolve()
        self.grading = json.loads((self.package / "private/grading.json").read_text())
        self.withheld = json.loads((self.package / "private/withheld.json").read_text())
        self.git_start = self.manifest["git_start"]
        self.scorer = F2PFractionScorer(self.package / "private/checks.json", task_id=self.task_id,
                                        binding=json.dumps({"image": self.manifest["environment_image_id"],
                                                            "sources": self.manifest["evaluator_sources"],
                                                            "grading": self.grading, "withheld": self.withheld,
                                                            "git_start": self.git_start}, sort_keys=True))
        text = (self.package / "public/instructions.md").read_text()
        self._stage = StageSpec(stage_id="repair", target_id=self.task_id + ":repair", ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.scorer.version,
            metadata={"upstream_instance_id": self.manifest["upstream_instance_id"], "private_grade_feedback": "none",
                      "visible_feedback_command": self.manifest["visible_feedback_command"],
                      "public_tests": "SWE-smith native: FAIL_TO_PASS test files are withheld from W; the remaining "
                                      "upstream tests stay visible. Grading restores the frozen test-support set, "
                                      "including the withheld files, and runs every tracked check.",
                      "workspace_vcs": "W is a fresh Git repository with one commit of the tracked start files "
                                       f"({self.git_start['commit']}); no upstream history. Grading ignores .git."})

    def describe(self) -> TaskRecord:
        m = self.manifest
        seal = self.seal or {}
        admitted = bool(seal)
        split = seal.get("split", m["planned_split"])
        return TaskRecord(task_id=self.task_id, family_id=m["family_id"], source=m["source_url"],
            source_revision=m["dataset_revision"], task_version=m["package_version"],
            environment_version=m["environment_image_id"], stages=(self._stage,),
            lifecycle=LifecyclePolicy(policy_id="swesmith-long.single-repair.v3", version="3", origin="benchmark_native",
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
                      "predecessor": m["predecessor"]["task_id"],
                      "upstream_instance_id": m["upstream_instance_id"], "adaptation_version": m["adaptation"]["version"],
                      "withheld_test_files": [r["path"] for r in self.withheld["files"]],
                      "git_start": {k: self.git_start[k] for k in ("commit", "tree", "branch", "native_tree_equal")},
                      "worker_prompt_version": m["worker_prompt"]["version"],
                      "grading_runtime_policy": "Only W (without .git) is graded, in a fresh copy of the frozen image "
                                                "(SWE-smith native evaluation); engine-captured R is restored for "
                                                "worker continuation only.",
                      "network_policy": "worker and grader containers run with network none",
                      "main_study_frozen": False},
            unresolved=() if admitted else ("Lane admission seal absent or pending",))

    def worker_prompt(self, stage_id: str):
        from ..prompts import swesmith_prompt_v3
        prompt = swesmith_prompt_v3(self.stage(stage_id))
        bound = self.manifest["worker_prompt"]
        if (prompt.version, hashlib.sha256(prompt.text.encode()).hexdigest()) != (bound["version"], bound["rendered_sha256"]):
            raise ValueError("Rendered worker prompt differs from the package binding")
        return prompt

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
        set_start_mtimes(workspace)
        extract_git(self.package / GIT_TAR, workspace)
        os.utime(workspace, (START_MTIME, START_MTIME))
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
            "vcs_metadata_skipped": (receipt.get("driver") or {}).get("vcs_metadata_skipped"),
            "collection_errors": [e["nodeid"] for e in observation["collect_errors"]],
            "timed_out": bool((receipt.get("driver") or {}).get("timed_out")),
            "infrastructure_error": receipt.get("infrastructure_error")})
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(result), indent=2) + "\n")
        return result
