"""Generic MettleBench task adapter: one native assignment, native Continue feedback.

Lifecycle (benchmark-native, paper Appendix B.3 "Continue"): a single
assignment in one container; W, R, H and O all carry across submissions.
On each submission the runner grades the submitted state and delivers only the
native first-unsatisfied-criterion feedback string as an ordinary user message;
the run ends on success or after five consecutive submissions failing at the
same criterion. Research-private per-criterion grades never reach the worker.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, LifecyclePolicy, RuntimeClass, RuntimeExport, StageSpec,
                                            TaskRecord, to_jsonable)
from trajectory_retention.tasks.evidence import load_admission
from trajectory_retention.tasks.prompts import WorkerTaskPrompt, _prompt
from . import source as S
from .grading import run_private_grade
from .package import extract_workspace, verify_package
from .scoring import MettleBenchScorer, native_consistency

STAGE_ID = "task"
PROMPT_SOURCE = S.SOURCE_URL + "/blob/" + S.SOURCE_REVISION + "/tasks/{task}/prompt.md"


class MettleBenchTaskAdapter:
    native_submission_protocol = S.NATIVE_PROTOCOL

    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.upstream_id = self.manifest["upstream_task_id"]
        self.scratch_root = Path(scratch_root or "artifacts/admission-scratch/mettlebench-v1").resolve()
        self.scorer = MettleBenchScorer(self.package / "private/rubric.json", task_id=self.manifest["task_id"],
                                        binding=self.manifest["implementation_binding"])
        text = (self.package / "public/instruction.md").read_text(encoding="utf-8")
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.manifest["task_id"] + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.scorer.version,
            metadata={"source_task": self.upstream_id, "source_benchmark": self.manifest["source_benchmark"],
                      "native_stage_count": 1, "criteria": self.manifest["criteria"],
                      "native_submission_feedback": "first unsatisfied criterion feedback string only",
                      "native_termination": "success or 5 identical consecutive first failures",
                      "private_grade_feedback": "none"})

    # ------------------------------------------------------------ identity
    def describe(self) -> TaskRecord:
        m = self.manifest
        record = TaskRecord(
            task_id=m["task_id"], family_id=m["family_id"], source=m["source_url"], source_revision=m["source_revision"],
            task_version=m["package_version"], environment_version=m["environment_image_id"], stages=(self._stage,),
            lifecycle=LifecyclePolicy(
                policy_id="mettlebench.native-continue.v1", version="1", origin="benchmark_native",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry", "T": "external_only"},
                detector_scope_rule="single native assignment; submissions and feedback do not reset scope",
                stage_release_rule=("no stages; on submission deliver native first-failure feedback verbatim and continue; "
                                    "end on success or 5 consecutive identical first failures"),
                source_evidence=tuple(S.NATIVE_PROTOCOL["source_evidence"])),
            admission=AdmissionStatus.HOLD, runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": m["family_split"], "package_sha256": hashlib.sha256(
                          (self.package / "package.json").read_bytes()).hexdigest(),
                      "eligibility": {"development_and_calibration": m["family_split"] == "calibration",
                                      "held_out_main_evaluation": False},
                      "native_start": "upstream dataset archive extracted with upstream filter='tar' semantics",
                      "reference_insertion": False,
                      "runtime_scope": "filesystem W/R capture; live processes unsupported (runner r-capture/v4)"},
            unresolved=("Admission requires the qualification seal",))
        return load_admission(self.package, record)

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != STAGE_ID:
            raise ValueError("Unknown stage")
        return self._stage

    def worker_prompt(self, stage_id: str) -> WorkerTaskPrompt:
        # MettleBench prompt.md is "a natural-language assignment, exactly as the agent receives it".
        return _prompt(self.stage(stage_id), "mettlebench", "", source=PROMPT_SOURCE.format(task=self.upstream_id),
                       adaptations=("Upstream prompt.md delivered byte-for-byte with the shared mini-SWE tool/submission interface.",))

    # ------------------------------------------------------------ lifecycle
    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or STAGE_ID)
        verify_package(self.package)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and any(workspace.iterdir())):
            raise ValueError("Natural start requires an empty real workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        extract_workspace(self.package / "public/workspace.tar.gz", workspace)
        return stage

    def next_stage(self, stage_id: str, *, termination_reason: str):
        self.stage(stage_id)
        return None

    # ------------------------------------------------------------ grading
    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None):
        stage = self.stage(stage_id)
        workspace = Path(workspace).resolve()
        if self.scratch_root == workspace or self.scratch_root.is_relative_to(workspace):
            raise ValueError("Private scratch must be outside W")
        if private_output_dir is not None and Path(private_output_dir).resolve().is_relative_to(workspace):
            raise ValueError("Private output must be outside W")
        if private_output_dir is None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            private_output_dir = Path(tempfile.mkdtemp(prefix="mettle-grade-", dir=self.scratch_root)) / "grade"
        output = Path(private_output_dir).resolve()
        result = run_private_grade(package=self.package, upstream_id=self.upstream_id,
                                   image_id=self.manifest["grader_image_id"], workspace=workspace, output=output,
                                   runtime=runtime, runner_source=Path(__file__).with_name("criteria_runner.py"))
        infra = result["infrastructure_errors"]
        checks = self.scorer.checks_from_observation(result.get("criteria"),
                                                     infra_error="; ".join(e["message"] for e in infra) if infra else None)
        native = result.get("native") if not infra else None
        extra = {"grade_seconds": result.get("seconds"), "native_seconds": result.get("native_seconds"),
                 "criteria_seconds": result.get("criteria_seconds"), "private_output_dir": str(output),
                 "infrastructure_errors": infra, "native_consistency": native_consistency(checks, native)}
        value = self.scorer.score(checks, stage=stage, native=native, extra=extra)
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
        return value

    def native_submission_verdict(self, evaluation) -> dict:
        """Worker-deliverable native verdict extracted from a valid grade of the submitted state."""
        native = (evaluation.details or {}).get("native") if evaluation is not None else None
        if evaluation is None or not evaluation.valid or not native or native.get("status") not in {"pass", "fail", "error"}:
            return {"status": "unavailable"}
        if native["status"] == "pass":
            return {"status": "pass", "criterion_id": None, "feedback": None}
        return {"status": native["status"], "criterion_id": native.get("first_failure_id"),
                "feedback": native.get("feedback") or ""}
