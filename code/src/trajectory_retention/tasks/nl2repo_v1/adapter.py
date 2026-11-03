"""NL2Repo task adapter: build a whole Python repository from start.md; native single submission.

The worker runs in an offline image (official python at the grading version + the task's third-party packages, no target package, no tests,
no network, no package index) with W0 = start.md only. One voluntary submission (or a guard) ends the run; the final
W goes through the native post-processor in a disposable container of the grading image, which the worker never
enters. Score = JUnit passed / native test_case_count (capped at 1); the native regex score is kept alongside.

Loop phase: `loop_verdict` returns the worker-safe summary (pass count and first failing test file); not wired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, CapabilityResult, CheckResult, CheckStatus, Evaluation,
                                            LifecyclePolicy, RuntimeClass, RuntimeExport, StageSpec, TaskRecord,
                                            to_jsonable)
from trajectory_retention.tasks.prompts import WorkerTaskPrompt, _prompt
from . import source as S
from .grading import run_private_grade
from .package import verify_package

STAGE_ID = "task"
SCORING_VERSION = "nl2repo.scoring.v1"


class NL2RepoTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.upstream_id = self.manifest["upstream_task_id"]
        self.scratch_root = Path(scratch_root or Path.home() / ".cache/trajectory-retention/nl2repo-grades").resolve()
        self.evaluator_version = SCORING_VERSION + ":" + hashlib.sha256(
            json.dumps(self.manifest["files"], sort_keys=True).encode() + self.manifest["grading_image_id"].encode()).hexdigest()
        text = (self.package / "public/instruction.md").read_text(encoding="utf-8")
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.manifest["task_id"] + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(
                text.encode() + (self.package / "public/workspace/start.md").read_bytes()).hexdigest(),
            evaluator_version=self.evaluator_version,
            metadata={"source_task": self.upstream_id, "source_benchmark": "NL2RepoBench", "native_stage_count": 1,
                      "test_case_count": self.manifest["test_case_count"], "native_submission_feedback": "none",
                      "private_grade_feedback": "none", "native_termination": "first voluntary submission or guard"})

    @property
    def worker_environment(self) -> dict:
        return {"cpus": "4", "memory": "8g"}

    def describe(self) -> TaskRecord:
        m = self.manifest
        return TaskRecord(
            task_id=m["task_id"], family_id=m["family_id"], source=m["source_url"], source_revision=m["source_revision"],
            task_version=m["package_version"], environment_version=m["environment_image_id"], stages=(self._stage,),
            lifecycle=LifecyclePolicy(
                policy_id=S.LIFECYCLE["policy_id"], version="1", origin="benchmark_native",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry", "T": "external_only"},
                detector_scope_rule="single native assignment", stage_release_rule=S.LIFECYCLE["stage_release_rule"],
                source_evidence=tuple(S.LIFECYCLE["source_evidence"])),
            admission=AdmissionStatus.HOLD, runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": m["family_split"],
                      "package_sha256": hashlib.sha256((self.package / "package.json").read_bytes()).hexdigest(),
                      "eligibility": {"development_and_calibration": m["family_split"] == "calibration",
                                      "held_out_main_evaluation": False},
                      "native_start": "workspace holding only start.md", "reference_insertion": False,
                      "deviations": m["deviations"], "context_overflow": S.LIFECYCLE["context_overflow"]},
            unresolved=("Admission requires the x86 qualification seal (Mac-emulated gates are development only)",))

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != STAGE_ID:
            raise ValueError("Unknown stage")
        return self._stage

    def worker_prompt(self, stage_id: str) -> WorkerTaskPrompt:
        return _prompt(self.stage(stage_id), "nl2repo", "", source=S.LIFECYCLE["source_evidence"][0],
                       adaptations=(S.DEVIATIONS[0], S.DEVIATIONS[1]))

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or STAGE_ID)
        verify_package(self.package)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and any(workspace.iterdir())):
            raise ValueError("Natural start requires an empty real workspace")
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.package / "public/workspace/start.md", workspace / "start.md")
        return stage

    def next_stage(self, stage_id: str, *, termination_reason: str):
        self.stage(stage_id)
        return None

    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None) -> Evaluation:
        stage = self.stage(stage_id)
        workspace = Path(workspace).resolve()
        if private_output_dir is None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            private_output_dir = Path(tempfile.mkdtemp(prefix="nl2repo-grade-", dir=self.scratch_root)) / "grade"
        output = Path(private_output_dir).resolve()
        if output.is_relative_to(workspace) or workspace.is_relative_to(output):
            raise ValueError("Private output must be outside W")
        result = run_private_grade(package=self.package, manifest=self.manifest, workspace=workspace, output=output)
        value = self._score(result, stage, output)
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
        return value

    def _score(self, result: dict, stage: StageSpec, output: Path) -> Evaluation:
        base = {"task_id": self.manifest["task_id"], "stage_id": stage.stage_id, "target_id": stage.target_id,
                "evaluator_version": self.evaluator_version}
        details = {"grade_seconds": result.get("seconds"), "private_output_dir": str(output),
                   "infrastructure_errors": result.get("infrastructure_errors") or [], "submission": result.get("submission"),
                   "native": result.get("native")}
        if details["infrastructure_errors"]:
            return Evaluation(**base, score=None, full_solve=None, valid=False, checks=(), details=details)
        count = self.manifest["test_case_count"]
        cases = result.get("cases") or []
        per_file: dict[str, tuple[int, int]] = {}
        checks = []
        for i, case in enumerate(cases):
            ok = case["outcome"] == "passed"
            status = CheckStatus.PASS if ok else CheckStatus.NOT_RUN if case["outcome"] == "skipped" else CheckStatus.FAIL
            checks.append(CheckResult(check_id=f"case::{i}", status=status, raw_status=case["outcome"],
                                      capability_id=case["file"]))
            done, total = per_file.get(case["file"], (0, 0))
            per_file[case["file"]] = (done + ok, total + 1)
        passed = sum(c.status == CheckStatus.PASS for c in checks)
        failing = sorted(f for f, (d, t) in per_file.items() if d < t) or ([] if passed >= count else ["collection"])
        details.update({"passed": passed, "total": count, "observed_cases": len(cases), "failing_groups": failing})
        return Evaluation(**base, score=min(passed / count, 1.0) if count else 0.0, full_solve=passed >= count > 0,
                          valid=True, checks=tuple(checks),
                          capabilities=tuple(CapabilityResult(capability_id=f, completion=d / t, reachable=None,
                                                              requirement_group=f) for f, (d, t) in sorted(per_file.items())),
                          group_scores={f: d / t for f, (d, t) in sorted(per_file.items())},
                          raw_score=(result.get("native") or {}).get("passed_regex"), raw_max_score=count, details=details)

    def loop_verdict(self, evaluation: Evaluation) -> dict:
        if evaluation is None or not evaluation.valid:
            return {"status": "unavailable"}
        d = evaluation.details
        if evaluation.full_solve:
            return {"status": "pass", "passed": d["passed"], "total": d["total"], "group": None}
        return {"status": "fail", "passed": d["passed"], "total": d["total"], "group": (d["failing_groups"] or [None])[0]}
