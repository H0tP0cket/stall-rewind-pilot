"""ProgramBench task adapter: native single-submission lifecycle on the cleanroom image.

The worker runs as the image's unprivileged `agent` user with SYS_PTRACE dropped and no network, starting from
the image's own /workspace (documentation plus a root-owned, execute-only reference ./executable). One voluntary
submission (or a guard) ends the run; the final W is graded once by the upstream Evaluator on the host. No grade
or feedback reaches the worker. Score = passed / frozen scored tests (active branches minus ignored tests).

Loop phase: `loop_verdict` exposes the worker-safe summary a later, separately ratified submit loop may deliver
(pass count and module-stem group, never test names' functions, inputs or expected outputs). It is not wired to
the runner: this adapter declares no `native_submission_protocol`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from trajectory_retention.contracts import (AdmissionStatus, CapabilityResult, CheckResult, CheckStatus, Evaluation,
                                            LifecyclePolicy, RuntimeClass, RuntimeExport, StageSpec, TaskRecord,
                                            to_jsonable)
from trajectory_retention.execution.archives import extract_directory
from trajectory_retention.tasks.prompts import WorkerTaskPrompt, _prompt
from . import source as S
from .grading import WORKER_ERROR_CODES, run_private_grade
from .package import frozen_tests, group_of, verify_package

STAGE_ID = "task"
SCORING_VERSION = "programbench.scoring.v1"


class ProgramBenchTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.upstream_id = self.manifest["upstream_task_id"]
        self.scratch_root = Path(scratch_root or Path.home() / ".cache/trajectory-retention/programbench-grades").resolve()
        tests = json.loads((self.package / "private/tests.json").read_text())
        frozen = frozen_tests(tests)
        self.scored = frozen["scored"]
        self.groups = {key: group_of(key.split("/", 1)[1]) for key in self.scored}
        self.evaluator_version = SCORING_VERSION + ":" + hashlib.sha256(
            (self.package / "private/tests.json").read_bytes() + self.manifest["environment_image_id"].encode()).hexdigest()
        text = (self.package / "public/instruction.md").read_text(encoding="utf-8")
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.manifest["task_id"] + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
            evaluator_version=self.evaluator_version,
            metadata={"source_task": self.upstream_id, "source_benchmark": "ProgramBench", "native_stage_count": 1,
                      "scored_tests": len(self.scored), "groups": len(set(self.groups.values())),
                      "native_submission_feedback": "none", "private_grade_feedback": "none",
                      "native_termination": "first voluntary submission or guard"})

    # ------------------------------------------------------------ identity
    @property
    def worker_environment(self) -> dict:
        return dict(S.WORKER_ENVIRONMENT)

    def describe(self) -> TaskRecord:
        m = self.manifest
        return TaskRecord(
            task_id=m["task_id"], family_id=m["family_id"], source=m["source_url"], source_revision=m["source_revision"],
            task_version=m["package_version"], environment_version=m["environment_image_id"], stages=(self._stage,),
            lifecycle=LifecyclePolicy(
                policy_id=S.LIFECYCLE["policy_id"], version="1", origin="benchmark_native",
                stage_operations={"W": "carry", "R": "carry", "H": "carry", "O": "carry", "T": "external_only"},
                detector_scope_rule="single native assignment",
                stage_release_rule=S.LIFECYCLE["stage_release_rule"],
                source_evidence=tuple(S.LIFECYCLE["source_evidence"])),
            admission=AdmissionStatus.HOLD, runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": m["family_split"],
                      "package_sha256": hashlib.sha256((self.package / "package.json").read_bytes()).hexdigest(),
                      "eligibility": {"development_and_calibration": m["family_split"] == "calibration",
                                      "held_out_main_evaluation": False},
                      "native_start": "cleanroom image /workspace exported byte-for-byte (numeric owners and modes)",
                      "reference_insertion": False, "deviations": list(S.DEVIATIONS),
                      "context_overflow": S.LIFECYCLE["context_overflow"]},
            unresolved=("Admission requires the x86 qualification seal (Mac-emulated gates are development only)",))

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != STAGE_ID:
            raise ValueError("Unknown stage")
        return self._stage

    def worker_prompt(self, stage_id: str) -> WorkerTaskPrompt:
        return _prompt(self.stage(stage_id), "programbench", "", source=S.AGENT_CONFIG_URL, adaptations=S.DEVIATIONS[3:6])

    # ------------------------------------------------------------ lifecycle
    def initial_workspace_archive(self, *, stage_id: str) -> bytes:
        self.stage(stage_id)
        verify_package(self.package)
        return (self.package / "public/workspace.tar").read_bytes()

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        """Host copy of the native W for inspection tools. Runs start from `initial_workspace_archive`, which keeps
        numeric ownership and modes that a host extraction cannot."""
        stage = self.stage(stage_id or STAGE_ID)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and any(workspace.iterdir())):
            raise ValueError("Natural start requires an empty real workspace")
        extract_directory(self.initial_workspace_archive(stage_id=stage.stage_id), workspace)
        return stage

    def next_stage(self, stage_id: str, *, termination_reason: str):
        self.stage(stage_id)
        return None

    # ------------------------------------------------------------ grading
    def grade(self, workspace: Path, *, stage_id: str, private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None) -> Evaluation:
        stage = self.stage(stage_id)
        workspace = Path(workspace).resolve()
        if private_output_dir is None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            private_output_dir = Path(tempfile.mkdtemp(prefix="programbench-grade-", dir=self.scratch_root)) / "grade"
        output = Path(private_output_dir).resolve()
        if output.is_relative_to(workspace) or workspace.is_relative_to(output):
            raise ValueError("Private output must be outside W")
        result = run_private_grade(package=self.package, manifest=self.manifest, workspace=workspace, output=output)
        value = self._score(result, stage, output)
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(value), indent=2) + "\n")
        return value

    def _score(self, result: dict, stage: StageSpec, output: Path) -> Evaluation:
        infra = result.get("infrastructure_errors") or []
        summary = result.get("summary") or {}
        base = {"task_id": self.manifest["task_id"], "stage_id": stage.stage_id, "target_id": stage.target_id,
                "evaluator_version": self.evaluator_version}
        details = {"grade_seconds": result.get("seconds"), "private_output_dir": str(output), "infrastructure_errors": infra,
                   "submission": result.get("submission"),
                   "native": {k: summary.get(k) for k in ("programbench_version", "error_code", "executable_hash",
                                                           "native_score", "native_n_resolved", "native_n_tests",
                                                           "branch_errors", "seconds", "grader_docker_cpus",
                                                           "built_equals_reference")}}
        # A test branch that ends in an error (results never read, executable not restored, ...) means the grader did
        # not finish that branch; per-test timeouts stay inside pytest, so a crashing or hanging submission still
        # yields results. Unless the submission already failed to build, the grade is an infrastructure fault.
        branch_errors = summary.get("branch_errors") or {}
        if branch_errors and not infra and summary.get("error_code") not in WORKER_ERROR_CODES:
            infra = [{"message": "test branches errored: " + ", ".join(
                f"{b}:{'+'.join(map(str, e))}" for b, e in sorted(branch_errors.items()))}]
            details["infrastructure_errors"] = infra
        if infra:
            checks = tuple(CheckResult(check_id="test::" + k, status=CheckStatus.INFRA_ERROR, capability_id=self.groups[k])
                           for k in self.scored)
            return Evaluation(**base, score=None, full_solve=None, valid=False, checks=checks, details=details)
        statuses = summary["statuses"]
        tamper = result.get("tamper") or []
        details["tamper"] = tamper
        worker_failed = summary.get("error_code") in WORKER_ERROR_CODES or bool(tamper)
        checks, per_group = [], {}
        for key in self.scored:
            raw = statuses.get(key)
            status = (CheckStatus.ERROR_TASK if tamper else CheckStatus.PASS if raw == "passed" else CheckStatus.ERROR_TASK if worker_failed
                      else CheckStatus.NOT_RUN if raw in (None, "not_run") else CheckStatus.FAIL)
            group = self.groups[key]
            checks.append(CheckResult(check_id="test::" + key, status=status, raw_status=raw, capability_id=group))
            done, total = per_group.get(group, (0, 0))
            per_group[group] = (done + (status == CheckStatus.PASS), total + 1)
        passed = sum(c.status == CheckStatus.PASS for c in checks)
        capabilities = tuple(CapabilityResult(capability_id=g, completion=d / t, reachable=None, requirement_group=g)
                             for g, (d, t) in sorted(per_group.items()))
        details.update({"passed": passed, "total": len(checks),
                        "failing_groups": sorted(g for g, (d, t) in per_group.items() if d < t)})
        return Evaluation(**base, score=passed / len(checks), full_solve=passed == len(checks), valid=True,
                          checks=tuple(checks), capabilities=capabilities,
                          group_scores={g: d / t for g, (d, t) in sorted(per_group.items())},
                          raw_score=summary.get("native_n_resolved"), raw_max_score=summary.get("native_n_tests"),
                          details=details)

    def loop_verdict(self, evaluation: Evaluation) -> dict:
        """Worker-safe summary for a future, separately ratified submit loop. Not wired to the runner."""
        if evaluation is None or not evaluation.valid:
            return {"status": "unavailable"}
        d = evaluation.details
        if evaluation.full_solve:
            return {"status": "pass", "passed": d["passed"], "total": d["total"], "group": None}
        return {"status": "fail", "passed": d["passed"], "total": d["total"],
                "group": (d["failing_groups"] or [None])[0]}
