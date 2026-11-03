"""FeatureBench level-1 task adapter: re-implement masked features of a real repository; native single submission.

The worker runs in the derived worker image (the pinned original image after the native level-1 initialisation, with
/root/my_repo, the F2P test file, bytecode and package caches gone) as the unprivileged `fbagent`, without network,
starting from the native W0 (the masked /testbed with a fresh one-commit git repository). One voluntary submission
(or a guard) ends the run; the final W becomes the native `git diff` submission and is graded natively in fresh
containers of the original image. Score = frozen F2P node ids passed / frozen count; full_solve = native `resolved`.

Loop phase: `loop_verdict` returns the worker-safe summary (pass count and first failing test class, or none when
the F2P file has no classes); not wired.
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
from .grading import group_of, run_private_grade
from .package import verify_package

STAGE_ID = "task"
SCORING_VERSION = "featurebench.scoring.v2"


class FeatureBenchTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.upstream_id = self.manifest["upstream_task_id"]
        self.frozen = json.loads((self.package / "private/f2p_ids.json").read_text())
        self.scratch_root = Path(scratch_root or Path.home() / ".cache/trajectory-retention/featurebench-grades").resolve()
        self.evaluator_version = SCORING_VERSION + ":" + hashlib.sha256(
            json.dumps(self.manifest["files"], sort_keys=True).encode() + self.manifest["grading_image_id"].encode()).hexdigest()
        text = (self.package / "public/instruction.md").read_text(encoding="utf-8")
        self._stage = StageSpec(
            stage_id=STAGE_ID, target_id=self.manifest["task_id"] + ":" + STAGE_ID, ordinal=1, instructions=text,
            requirement_version="sha256:" + hashlib.sha256(
                text.encode() + self.manifest["files"]["public/workspace.tar"].encode()).hexdigest(),
            evaluator_version=self.evaluator_version,
            metadata={"source_task": self.upstream_id, "source_benchmark": "FeatureBench", "native_stage_count": 1,
                      "f2p_count": len(self.frozen), "native_submission_feedback": "none",
                      "private_grade_feedback": "none", "native_termination": "first voluntary submission or guard"})

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
                detector_scope_rule="single native assignment", stage_release_rule=S.LIFECYCLE["stage_release_rule"],
                source_evidence=tuple(S.LIFECYCLE["source_evidence"])),
            admission=AdmissionStatus.HOLD, runtime_class=RuntimeClass.CAPTURED,
            evidence={"family_split": m["family_split"],
                      "package_sha256": hashlib.sha256((self.package / "package.json").read_bytes()).hexdigest(),
                      "eligibility": {"development_and_calibration": m["family_split"] == "calibration",
                                      "held_out_main_evaluation": False},
                      "native_start": "masked /testbed, F2P file deleted, one-commit git repository",
                      "reference_insertion": False, "deviations": m["deviations"],
                      "worker_image_note": ("derived from the grading image with the native deletions applied, as "
                                            "natively; the worker never runs a grading container"),
                      "leak_scan_clean": m["leak_scan_clean"], "context_overflow": S.LIFECYCLE["context_overflow"]},
            unresolved=("Admission requires the x86 qualification seal (Mac-emulated gates are development only)",))

    def stage(self, stage_id: str) -> StageSpec:
        if stage_id != STAGE_ID:
            raise ValueError("Unknown stage")
        return self._stage

    def worker_prompt(self, stage_id: str) -> WorkerTaskPrompt:
        return _prompt(self.stage(stage_id), "featurebench", "", source=S.LIFECYCLE["source_evidence"][0],
                       adaptations=(S.DEVIATIONS[1], S.DEVIATIONS[2]))

    def initial_workspace_archive(self, *, stage_id: str) -> bytes:
        self.stage(stage_id)
        verify_package(self.package)
        return (self.package / "public/workspace.tar").read_bytes()

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or STAGE_ID)
        workspace = Path(workspace)
        if workspace.is_symlink() or (workspace.exists() and any(workspace.iterdir())):
            raise ValueError("Natural start requires an empty real workspace")
        extract_directory(self.initial_workspace_archive(stage_id=stage.stage_id), workspace)
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
            private_output_dir = Path(tempfile.mkdtemp(prefix="featurebench-grade-", dir=self.scratch_root)) / "grade"
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
                   "infrastructure_errors": result.get("infrastructure_errors") or [],
                   "submission": result.get("submission"), "native": result.get("native"),
                   "native_error": result.get("native_error"), "guarded": result.get("guarded")}
        if details["infrastructure_errors"]:
            return Evaluation(**base, score=None, full_solve=None, valid=False, checks=(), details=details)
        status = result.get("f2p_status") or {}
        checks, per_group = [], {}
        for i, node in enumerate(self.frozen):
            raw = status.get(node, "missing")
            ok = raw in ("PASSED", "XFAIL")
            checks.append(CheckResult(check_id=f"f2p::{i}", status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                                      raw_status=raw, capability_id=group_of(node)))
            g = group_of(node) or "_"
            done, total = per_group.get(g, (0, 0))
            per_group[g] = (done + ok, total + 1)
        passed = sum(c.status == CheckStatus.PASS for c in checks)
        total = len(self.frozen)
        failing = sorted(g for g, (d, t) in per_group.items() if d < t and g != "_")
        resolved = bool((result.get("native") or {}).get("resolved"))
        regressions = sorted(set(result.get("p2p_failed_ids") or []) - set(self.manifest["reference_p2p_failed_ids"]))
        details.update({"passed": passed, "total": total, "failing_groups": failing,
                        "p2p_regressions": len(regressions), "native_resolved": resolved})
        full = passed == total and not regressions and result.get("native_error") is None
        return Evaluation(**base, score=passed / total if total else 0.0, full_solve=full,
                          valid=True, checks=tuple(checks),
                          capabilities=tuple(CapabilityResult(capability_id=g, completion=d / t, reachable=None,
                                                              requirement_group=g) for g, (d, t) in sorted(per_group.items())),
                          group_scores={g: d / t for g, (d, t) in sorted(per_group.items())},
                          raw_score=(result.get("native") or {}).get("native_pass_rate"), raw_max_score=1.0,
                          details=details)

    def loop_verdict(self, evaluation: Evaluation) -> dict:
        if evaluation is None or not evaluation.valid:
            return {"status": "unavailable"}
        d = evaluation.details
        if evaluation.full_solve:
            return {"status": "pass", "passed": d["passed"], "total": d["total"], "group": None}
        return {"status": "fail", "passed": d["passed"], "total": d["total"], "group": (d["failing_groups"] or [None])[0]}
