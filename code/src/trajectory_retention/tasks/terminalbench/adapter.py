"""Generic adapter for every packaged Terminal-Bench 2 task (lane terminalbench-v1).

Package layout (benchmark_tasks/packages/terminalbench-v1/<task_id>/):
  package.json               manifest binding every public/private file, images and evaluator sources
  public/instruction.md      upstream instruction.md, byte-for-byte
  public/workspace.tar.gz    natural start W: the native image's /app after the upstream Dockerfile build
  private/tests/             upstream tests/ (hidden; streamed only into disposable grading containers)
  private/solution/          upstream solution/ (reference; never enters W or an image)
  private/environment/       upstream environment/ build context (may hold hidden references, e.g. orig.c)
  private/task.toml          upstream task.toml
  private/checks.json        frozen native check ids (pytest node ids)
  private/metrics.json       optional declared native-metric regexes (annotation only)
The admission seal lives at benchmark_tasks/admission/terminalbench-v1/tasks/<task_id>/admission.json.

Lifecycle (benchmark-native, single shot): one task episode in one container;
the agent works until it stops (submits) or a limit fires; there is no
feedback, no resubmission and no stage. Hidden tests run once afterwards. The
engine's intermediate grades are research-private.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, LifecyclePolicy, RuntimeClass, RuntimeExport, StageSpec,
                                            TaskRecord, to_jsonable)
from ..prompts import terminal_bench_prompt
from ..swesmith_long.grading import observe, translate
from . import source as S
from .grading import run_private_grade
from .scoring import NativeCheckFractionScorer

LANE = "terminalbench-v1"
SCHEMA = "terminalbench.package.v1"
SEAL_SCHEMA = "terminalbench.admission.v1"
ADAPTER_ENTRY = "trajectory_retention.tasks.terminalbench.adapter:TerminalBenchTaskAdapter"
STAGE_ID = "task"
EVALUATOR_SOURCES = ("adapter.py", "grading.py", "scoring.py", "environment.py", "grader/tb_recorder.py",
                     "grader/run_native.py", "grader/pytest.ini", "../swesmith_long/grading.py")
LIFECYCLE_EVIDENCE = (
    "https://harborframework.com/docs (Harbor task format: instruction.md, task.toml, environment/, tests/test.sh, solution/)",
    S.SOURCE_URL + "/blob/" + S.SOURCE_REVISION + "/README.md",
)


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
    return {name: sha256((here / name).resolve()) for name in EVALUATOR_SOURCES}


def verify_package(package: Path) -> dict:
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("schema_version") != SCHEMA:
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
    path = seal_path(package, manifest["task_id"])
    if not path.is_file():
        return None
    seal = json.loads(path.read_text())
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("admission") != "ADMITTED":
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


def extract_workspace(archive: Path, workspace: Path) -> None:
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if m.name not in (".", "./")]
        for m in members:
            if not m.name.startswith("./") or ".." in Path(m.name).parts:
                raise ValueError("Unexpected workspace archive member: " + m.name)
        tf.extractall(workspace, members=members, filter="tar")


class TerminalBenchTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.task_id = self.manifest["task_id"]
        self.upstream_task = self.manifest["upstream_task"]
        self.seal = verify_seal(self.package, self.manifest)
        self.scratch_root = Path(scratch_root or "artifacts/admission-scratch/terminalbench-v1").resolve()
        env = self.manifest["environment"]
        self.worker_image_id = env["worker"]["image_id"]
        self.grader_image_id = env["grader"]["image_id"]
        self.grading = self.manifest["grading"]
        binding = json.dumps({"worker": self.worker_image_id, "grader": self.grader_image_id,
                              "sources": self.manifest["evaluator_sources"], "grading": self.grading,
                              "tests": {k: v for k, v in self.manifest["private_files"].items() if k.startswith("tests/")}},
                             sort_keys=True)
        metrics = self.package / "private/metrics.json"
        self.scorer = NativeCheckFractionScorer(self.package / "private/checks.json", task_id=self.task_id,
                                                binding=binding, metrics_path=metrics if metrics.is_file() else None)
        text = (self.package / "public/instruction.md").read_text(encoding="utf-8")
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.task_id + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.scorer.version,
            metadata={"source_task": self.upstream_task, "source_benchmark": "terminal-bench@2.0",
                      "native_stage_count": 1, "native_working_directory": "/app (symlink to /workspace)",
                      "native_agent_timeout_seconds": self.manifest["native_limits"]["agent_timeout_sec"],
                      "native_feedback": "none (hidden tests run once after the agent stops)",
                      "private_grade_feedback": "none", "checks": len(self.scorer.expected)})

    # ------------------------------------------------------------ identity
    def describe(self) -> TaskRecord:
        m = self.manifest
        seal = self.seal or {}
        admitted = bool(seal)
        split = seal.get("split", m["planned_split"])
        return TaskRecord(
            task_id=self.task_id, family_id=m["family_id"], source=S.SOURCE_URL, source_revision=S.SOURCE_REVISION,
            task_version=m["package_version"], environment_version=self.worker_image_id, stages=(self._stage,),
            lifecycle=LifecyclePolicy(
                policy_id="terminalbench.single-shot.v1", version="1", origin="benchmark_native",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry_as_H_substream",
                                  "T": "external_only"},
                detector_scope_rule="One task episode per trajectory; no native stage boundary.",
                stage_release_rule=("Single shot: the agent works until it submits or a limit fires; no feedback, no "
                                    "resubmission; hidden tests run once afterwards on the submitted W/R."),
                source_evidence=LIFECYCLE_EVIDENCE),
            admission=AdmissionStatus.ADMITTED if admitted else AdmissionStatus.HOLD,
            runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": split,
                      "eligibility": {"development_and_calibration": admitted and split == "calibration",
                                      "held_out_main_evaluation": False},
                      "package_sha256": sha256(self.package / "package.json"),
                      "upstream_task": self.upstream_task, "registry_identity": m["registry_identity"],
                      "adaptation_version": m["adaptation"]["version"],
                      "grading_runtime_policy": ("Native TB grades the live container; here captured W (/workspace = "
                                                 "/app) and filesystem R are restored into a fresh grader container "
                                                 "(worker image + verifier layer); live processes are not restored."),
                      "network_policy": "worker and grader containers run with network none",
                      "resources": m["native_limits"], "main_study_frozen": False},
            unresolved=() if admitted else ("Lane admission seal absent or pending",))

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != STAGE_ID:
            raise ValueError("Unknown stage")
        return self._stage

    def worker_prompt(self, stage_id: str):
        return terminal_bench_prompt(self.stage(stage_id))

    def next_stage(self, stage_id: str, *, termination_reason: str):
        self.stage(stage_id)
        return None

    # ------------------------------------------------------------ lifecycle
    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or STAGE_ID)
        verify_package(self.package)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir()))):
            raise ValueError("Natural start requires an empty real workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        extract_workspace(self.package / "public/workspace.tar.gz", workspace)
        return stage

    # ------------------------------------------------------------ grading
    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None):
        stage = self.stage(stage_id)
        verify_package(self.package)
        workspace = Path(workspace).resolve()
        if self.scratch_root == workspace or self.scratch_root.is_relative_to(workspace):
            raise ValueError("Private scratch must be outside submitted W")
        if private_output_dir is None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            private_output_dir = Path(tempfile.mkdtemp(prefix="grade-", dir=self.scratch_root)) / "private"
        output = Path(private_output_dir).resolve()
        if output == workspace or output.is_relative_to(workspace):
            raise ValueError("Private grade output must be outside submitted W")
        g = self.grading
        result = run_private_grade(grader_image_id=self.grader_image_id, worker_image_id=self.worker_image_id,
                                   tests_dir=self.package / "private/tests", workspace=workspace, output=output,
                                   runtime=runtime, timeout=g["timeout_seconds"], pytest_flags=g["pytest_flags"],
                                   cpus=g["cpus"], memory=g["memory"])
        infra = result["infrastructure_errors"]
        receipt = {"driver": {"timed_out": bool(result.get("timed_out")), "returncode": result.get("returncode")},
                   "infrastructure_error": "; ".join(e["message"] for e in infra) if infra else None}
        observation = observe(output)
        checks = translate(self.scorer.expected, observation, receipt)
        native_exit = (result.get("exit") or {}).get("exit_code") if not infra else None
        if native_exit is None and not infra:
            native_exit = result.get("returncode")
        extra = {"private_output_dir": str(output), "grade_wall_seconds": result.get("seconds"),
                 "pytest_seconds": result.get("pytest_seconds"), "timed_out": bool(result.get("timed_out")),
                 "runtime_export_consumed": runtime is not None,
                 "runtime_policy": "captured W + filesystem R restored into a fresh grader container; processes not restored",
                 "reserved_runtime_paths_dropped": result.get("reserved_runtime_paths_dropped", []),
                 "collection_errors": [e["nodeid"] for e in observation["collect_errors"]],
                 "infrastructure_error": receipt["infrastructure_error"]}
        value = self.scorer.score(checks, stage=stage, native_exit=native_exit, extra=extra)
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
        return value
