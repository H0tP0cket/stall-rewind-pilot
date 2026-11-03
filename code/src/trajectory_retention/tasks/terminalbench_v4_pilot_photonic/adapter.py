"""Generic adapter for every packaged Terminal-Bench 4 task of the terminalbench-v4-pilot-photonic lane.

Package layout:
  benchmark_tasks/packages/terminalbench-v4-pilot-photonic/<task_id>/   (in Git)
    package.json            manifest binding every public and private file, images and evaluator sources
    public/instruction.md   upstream instruction.md, byte-for-byte (the worker gets it with Harbor's canary strip)
    public/workspace.tar.gz natural start W: the native image's /app as a tar with numeric owners and modes ('./' too)
  <PRIVATE_ROOT>/<task_id>/                               (outside Git; rebuilt from the pinned checkout)
    tests/ solution/ environment/ task.toml README.md [cheat/]   upstream files (hidden tests, reference, build context)
    checks.json             frozen hidden-test ids (pytest node ids collected by the task's own test.sh in the verifier)
The admission seal lives at benchmark_tasks/admission/terminalbench-v4-pilot-photonic/tasks/<task_id>/admission.json.

Lifecycle (an explicit adaptation of the native single-shot episode): one task episode in one container. When the
worker submits, the submitted W and R are captured and graded privately in a separate verifier container, as Harbor
grades TB4 (``grading.py``); the worker is told only how many hidden tests still fail and keeps working. The run ends
at a passing submission, after 3 consecutive identical failing sets, or at a run guard (``protocol.py``). References,
hidden tests and grades never reach the worker.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, LifecyclePolicy, RuntimeClass, RuntimeExport, StageSpec,
                                            TaskRecord, to_jsonable)
from trajectory_retention.execution.archives import extract_directory
from ..prompts import terminal_bench_prompt
from . import protocol as P
from . import source as S
from .grading import run_private_grade
from .observe import observe, translate
from .scoring import NativeCheckFractionScorer

LANE = S.LANE
SCHEMA = "terminalbench-v4-pilot-photonic.package.v1"
SEAL_SCHEMA = "terminalbench-v4-pilot-photonic.admission.v1"
ADAPTER_ENTRY = "trajectory_retention.tasks.terminalbench_v4_pilot_photonic.adapter:TerminalBenchV4TaskAdapter"
STAGE_ID = "task"
EVALUATOR_SOURCES = ("adapter.py", "protocol.py", "grading.py", "scoring.py", "observe.py", "environment.py",
                     "source.py", "grader/tb_recorder.py")
PRIVATE_ROOT = Path(os.environ.get("TBV4_PHOTONIC_PRIVATE_ROOT", Path.home() / ".cache/trajectory-retention/terminalbench-v4-pilot-photonic/private"))
LIFECYCLE_EVIDENCE = (
    "https://harborframework.com/docs/tasks (separate verifier environments; artifacts)",
    S.SOURCE_URL + "/blob/" + S.SOURCE_REVISION + "/CONTRIBUTING.md",
    S.HARBOR_URL + "/tree/" + S.HARBOR_REVISION + " (separate verifier: trial.py, artifact_handler.py, verifier.py)",
    "docs/research/experiment.md (Natural runs: submission loop with failing-count feedback)",
)
# Harbor's strip_canary (src/harbor/models/task/task.py at source.HARBOR_REVISION): leading canary comment lines and the
# blank lines after them are not shown to the agent.
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def strip_canary(text: str) -> str:
    lines = text.split("\n")
    idx = 0
    while idx < len(lines) and _CANARY_LINE.match(lines[idx].strip()):
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return "\n".join(lines[idx:])


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


def private_dir(task_id: str) -> Path:
    return PRIVATE_ROOT / task_id


def verify_package(package: Path) -> dict:
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported package schema")
    if file_manifest(package / "public") != manifest["public_files"]:
        raise ValueError("Frozen package changed: public")
    private = private_dir(manifest["task_id"])
    if not private.is_dir():
        raise ValueError(f"Private package material absent at {private}; run the lane's restore-private command")
    if file_manifest(private) != manifest["private_files"]:
        raise ValueError("Frozen package changed: private")
    if evaluator_sources() != manifest["evaluator_sources"]:
        raise ValueError("Frozen evaluator source changed; issue a new lane")
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
    if not all(item.get("passed") is True for item in seal["gate"].values()):
        raise ValueError("Admission seal records a gate item that did not pass")
    return seal


class TerminalBenchV4TaskAdapter:
    native_submission_protocol = P.PROTOCOL

    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.task_id = self.manifest["task_id"]
        self.upstream_task = self.manifest["upstream_task"]
        self.private = private_dir(self.task_id)
        self.seal = verify_seal(self.package, self.manifest)
        self.scratch_root = Path(scratch_root or Path.home() / ".cache/trajectory-retention/v4-pilot-photonic/scratch").resolve()
        env = self.manifest["environment"]
        self.worker_image_id = env["worker"]["image_id"]
        self.verifier_image_id = env["verifier"]["image_id"]
        self.grading = self.manifest["grading"]
        limits = self.manifest["native_limits"]
        # Benchmark-native worker resources from task.toml (the runner passes these to the worker container).
        self.worker_environment = {"cpus": str(limits["cpus"] or 1), "memory": f"{limits['memory_mb'] or 2048}m"}
        binding = json.dumps({"worker": self.worker_image_id, "verifier": self.verifier_image_id,
                              "sources": self.manifest["evaluator_sources"], "grading": self.grading,
                              "tests": {k: v for k, v in self.manifest["private_files"].items() if k.startswith("tests/")}},
                             sort_keys=True)
        self.scorer = NativeCheckFractionScorer(self.private / "checks.json", task_id=self.task_id, binding=binding)
        text = strip_canary((self.package / "public/instruction.md").read_text(encoding="utf-8"))
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.task_id + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.scorer.version,
            metadata={"source_task": self.upstream_task, "source_benchmark": "terminal-bench@4.0.0",
                      "native_stage_count": 1, "native_working_directory": "/app (symlink to /workspace)",
                      "native_agent_timeout_seconds": limits["agent_timeout_sec"],
                      "instruction_delivery": "upstream instruction.md with Harbor's strip_canary applied",
                      "submission_feedback": "count of hidden tests still failing (" + P.VERSION + ")",
                      "private_grade_feedback": "none", "hidden_tests": len(self.scorer.expected)})

    # ------------------------------------------------------------ identity
    def describe(self) -> TaskRecord:
        m = self.manifest
        admitted = self.seal is not None
        return TaskRecord(
            task_id=self.task_id, family_id=m["family_id"], source=S.SOURCE_URL, source_revision=S.SOURCE_REVISION,
            task_version=m["package_version"], environment_version=self.worker_image_id, stages=(self._stage,),
            lifecycle=LifecyclePolicy(
                policy_id=P.VERSION, version="1", origin="explicit_adaptation",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry_as_H_substream",
                                  "T": "external_only"},
                detector_scope_rule="One task episode per trajectory; submissions and feedback do not reset scope.",
                stage_release_rule=("Submit and continue: each submission grades the captured W and R in a separate "
                                    "verifier container; the worker hears only the failing-test count; the run ends at a "
                                    "passing submission or after 3 consecutive identical failing sets."),
                source_evidence=LIFECYCLE_EVIDENCE),
            admission=AdmissionStatus.ADMITTED if admitted else AdmissionStatus.HOLD,
            runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": None, "split_policy": S.SPLIT_POLICY,
                      "eligibility": {"development_and_calibration": admitted, "held_out_main_evaluation": admitted},
                      "package_sha256": sha256(self.package / "package.json"),
                      "upstream_task": self.upstream_task, "source_tag": S.SOURCE_TAG,
                      "adaptation_version": m["adaptation"]["version"], "deviations": m["deviations"],
                      "hidden_tests": len(self.scorer.expected),
                      "grading_runtime_policy": ("captured W (/workspace = /app) and captured filesystem R restored "
                                                 "into a fresh worker container; the declared artifacts are copied from "
                                                 "it into a fresh verifier container (Harbor separate mode); live "
                                                 "processes are not restored; baseline R is never graded"),
                      "network_policy": "worker, source and verifier containers run with network none",
                      "resources": m["native_limits"]},
            unresolved=() if admitted else ("terminalbench-v4-pilot-photonic admission seal absent",))

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
    def initial_workspace_archive(self, *, stage_id: str) -> bytes:
        """The native /app, byte-for-byte with numeric owners and modes (the runner restores it as the natural start)."""
        self.stage(stage_id)
        verify_package(self.package)
        return gzip.decompress((self.package / "public/workspace.tar.gz").read_bytes())

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        """Host copy of the natural-start W for inspection tools (runs start from ``initial_workspace_archive``)."""
        stage = self.stage(stage_id or STAGE_ID)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir()))):
            raise ValueError("Natural start requires an empty real workspace")
        extract_directory(self.initial_workspace_archive(stage_id=stage.stage_id), workspace)
        return stage

    # ------------------------------------------------------------ submissions
    def native_submission_verdict(self, evaluation) -> dict:
        return P.verdict(evaluation)

    # ------------------------------------------------------------ grading
    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None):
        stage = self.stage(stage_id)
        verify_package(self.package)
        if runtime is None:
            raise ValueError("terminalbench-v4-pilot-photonic grades captured W and R only; baseline R is not the measured state")
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
        result = run_private_grade(verifier_image_id=self.verifier_image_id, worker_image_id=self.worker_image_id,
                                   artifacts=list(g["artifacts"]), workspace=workspace, output=output, runtime=runtime,
                                   timeout=g["timeout_seconds"], cpus=g["cpus"], memory=g["memory"])
        infra = result["infrastructure_errors"]
        receipt = {"driver": {"timed_out": bool(result.get("timed_out")), "returncode": result.get("returncode")},
                   "infrastructure_error": "; ".join(e["message"] for e in infra) if infra else None}
        observation = observe(output)
        checks = translate(self.scorer.expected, observation, receipt)
        reward = (result.get("reward") or {}).get("value") if not infra else None
        extra = {"private_output_dir": str(output), "grade_wall_seconds": result.get("seconds"),
                 "test_sh_seconds": result.get("test_sh_seconds"), "timed_out": bool(result.get("timed_out")),
                 "runtime_export_consumed": True,
                 "runtime_policy": "captured W + captured filesystem R restored into a fresh worker container; declared "
                                   "artifacts copied into a fresh verifier container; processes not restored",
                 "artifacts": [{k: a.get(k) for k in ("source", "kind", "uploaded", "sha256", "size")}
                               for a in result.get("artifacts", [])],
                 "reward_source": (result.get("reward") or {}).get("source"),
                 "collection_errors": [e["nodeid"] for e in observation["collect_errors"]],
                 "infrastructure_error": receipt["infrastructure_error"]}
        value = self.scorer.score(checks, stage=stage, native_reward=reward,
                                  test_sh_exit=None if infra else result.get("returncode"), extra=extra)
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
        return value
