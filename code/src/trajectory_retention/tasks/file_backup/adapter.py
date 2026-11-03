"""The canonical TaskAdapter for a legitimate four-stage file_backup run."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from trajectory_retention.contracts import (
    AdmissionStatus, CheckResult, CheckStatus, Evaluation, LifecyclePolicy,
    RuntimeClass, RuntimeExport, StageSpec, TaskRecord, from_jsonable, to_jsonable,
)
from trajectory_retention.scoring.file_backup import FileBackupScoringAdapter
from .grading import run_grade
from .packaging import grading_contract_projection, verify_package


class FileBackupTaskAdapter:
    def __init__(self, package: Path, *, scratch_root: Path | None = None,
                 attestation: Path | None = None):
        self.package = Path(package).resolve()
        self.manifest = verify_package(self.package)
        self.scratch_root = (scratch_root or Path("artifacts/admission-scratch")).resolve()
        self.task_id = self.manifest["task_id"]
        contract_binding = self.manifest.get("evaluator_binding", {}).get("shared_grading_contract_projection")
        if contract_binding and contract_binding != grading_contract_projection():
            raise ValueError("Shared grading record/interface semantics changed after seal")
        self.attestation = attestation or self.package.parent.parent / "admission" / (self.task_id + ".json")
        for relative, digest in self.manifest.get("evaluator_binding", {}).get("source_files", {}).items():
            current = Path(__file__).parents[2] / relative
            if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Sealed evaluator source changed: {relative}")
        packaged_rubric = self.package / "private/rubric.json"
        self.scorer = FileBackupScoringAdapter(
            packaged_rubric if packaged_rubric.exists() else None, task_id=self.task_id)
        bound_version = self.manifest.get("evaluator_binding", {}).get("evaluator_version")
        if bound_version is not None and bound_version != self.scorer.version:
            raise ValueError("Sealed evaluator version differs from current scoring implementation")
        self._stages: dict[str, StageSpec] = {}
        for ordinal, stage_id in enumerate(self.manifest["stages"], 1):
            text = (self.package / self.manifest["requirements"] / (stage_id + ".md")).read_text()
            requirement_hash = hashlib.sha256(text.encode()).hexdigest()
            self._stages[stage_id] = StageSpec(
                stage_id=stage_id, target_id=f"{self.task_id}:{stage_id}", instructions=text,
                requirement_version=f"sha256:{requirement_hash}", evaluator_version=self.scorer.version,
                ordinal=ordinal, prerequisites=tuple(self.manifest["stages"][:ordinal - 1]),
                metadata={"task_id": self.task_id, "instruction_delivery": "current_stage_only",
                          "recovery_target": "active stage plus prerequisite obligations; no later disclosure",
                          "entrypoint": self.manifest["entrypoint"],
                          "private_grade_feedback": "none", "stage_release": "native_any",
                          "comparability": "within fixed target and evaluator only"},
            )

    def describe(self) -> TaskRecord:
        if self.attestation.exists():
            data = json.loads(self.attestation.read_text())
            if data["package_sha256"] != hashlib.sha256((self.package / "package.json").read_bytes()).hexdigest():
                raise ValueError("Admission attestation has a different package binding")
            if data["evaluator_version"] != self.scorer.version:
                raise ValueError("Admission attestation has a different evaluator binding")
            for evidence in data["evidence"]:
                if hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
                    raise ValueError("Admission evidence bytes changed")
            record = from_jsonable(TaskRecord, data["task_record"])
            if record.task_id != self.task_id or record.stages != tuple(self._stages.values()):
                raise ValueError("Admission record does not match the current task/stage identity")
            return record
        lifecycle_path = self.package / "private/lifecycle.json"
        if not lifecycle_path.exists():
            lifecycle_path = Path(__file__).with_name("lifecycle_draft.json")
        draft = json.loads(lifecycle_path.read_text())
        return TaskRecord(
            task_id=self.task_id, family_id=self.manifest["family_id"],
            source=self.manifest["source_url"], source_revision=self.manifest["source_revision"],
            task_version=self.manifest["package_version"],
            environment_version=self.manifest["environment_image_id"], stages=tuple(self._stages.values()),
            lifecycle=LifecyclePolicy(
                policy_id=draft["policy_id"], version="scb.file_backup.lifecycle.v1",
                origin="benchmark_native",
                stage_operations={"W": "carry", "R": "reset_to_fresh_container",
                                  "O": "reset_with_history", "H": "reset_to_current_stage_prompt",
                                  "T": "external_history_retained; not delivered to worker"},
                detector_scope_rule="new stage-local scope; retain trajectory-wide ever-reached ledger",
                stage_release_rule="native any: legitimate stage finish releases next stage regardless of hidden score; caller checks whole-run allowance",
                source_evidence=(str(self.package / "package.json"),
                                 str(lifecycle_path)),
            ),
            admission=AdmissionStatus.HOLD, runtime_class=RuntimeClass.UNSUPPORTED,
            evidence={"package": str(self.package / "package.json"),
                      "rubric_sha256": self.scorer.rubric_sha256,
                      "scoring_mapping_frozen": self.scorer.rubric["mapping_frozen"],
                      "worker_start": "official assets only, no implementation or reference",
                      "family_split": "development; excluded from held-out main evaluation"},
            unresolved=("final adapted evaluator/reference preflight",
                        "shared runner capture/restore/runtime and isolation validation",
                        "calibration coverage and final evidence-bound admission"),
        )

    def worker_prompt(self, stage_id: str):
        from ..prompts import scb_prompt
        return scb_prompt(self.stage(stage_id))

    def stage(self, stage_id: str) -> StageSpec:
        try:
            return self._stages[stage_id]
        except KeyError:
            raise ValueError(f"Unknown stage: {stage_id}") from None

    def materialize(self, workspace: Path, *, stage_id: str | None = None) -> StageSpec:
        stage = self.stage(stage_id or self.manifest["stages"][0])
        workspace = Path(workspace)
        if workspace.is_symlink():
            raise ValueError("Materialization target must not be a symlink")
        if stage.ordinal > 1:
            if not workspace.is_dir() or not any(workspace.iterdir()):
                raise ValueError("A later stage requires the natural run's existing W")
            # Only the execution runner can establish the incoming W lineage.
            # Disclosure never overwrites worker code or adds cumulative specs.
            return stage
        if workspace.exists() and (not workspace.is_dir() or any(workspace.iterdir())):
            raise ValueError("Initial materialization requires an empty workspace")
        verify_package(self.package)
        shutil.copytree(self.package / self.manifest["initial_workspace"], workspace, dirs_exist_ok=True)
        return stage

    def grade(self, workspace: Path, *, stage_id: str,
              private_output_dir: Path | None = None,
              runtime: RuntimeExport | None = None) -> Evaluation:
        stage = self.stage(stage_id)
        if private_output_dir is None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
            # Persistent private receipt: caller imports it into ArtifactStore.
            parent = Path(tempfile.mkdtemp(prefix="file-backup-receipt-", dir=self.scratch_root))
            private_output_dir = parent / "grade"
        output = Path(private_output_dir).resolve()
        receipt = run_grade(self.package, Path(workspace), stage=stage.ordinal, output=output,
                            scratch_root=self.scratch_root,
                            adapted=bool(self.manifest.get("adaptation_version")), runtime=runtime)
        raw_path = output / "results.json"
        parse_error = None
        try:
            native = json.loads(raw_path.read_text()) if raw_path.exists() else None
        except (OSError, ValueError) as exc:
            native, parse_error = None, repr(exc)
        checks = self._translate(native, receipt, stage, parse_error=parse_error)
        evaluation = self.scorer.score(checks, stage=stage)
        evaluation = replace(evaluation, details={
            **evaluation.details, "private_output_dir": str(output),
            "receipt_path": str(output / "receipt.json"), "receipt": receipt,
            "raw_output_path": str(raw_path) if raw_path.exists() else None,
            "parse_error": parse_error,
        })
        (output / "evaluation.json").write_text(json.dumps(to_jsonable(evaluation), indent=2) + "\n")
        return evaluation

    def _translate(self, native: dict | None, receipt: dict, stage: StageSpec,
                   *, parse_error: str | None = None) -> tuple[CheckResult, ...]:
        expected = self.scorer.expected_checks(stage.stage_id)
        infrastructure = (native is None or bool(parse_error) or not receipt["source_unchanged"] or
                          receipt["launch_error"] is not None or bool(receipt["timeout"]) or
                          receipt["returncode"] not in (0, 1) or bool(native.get("collection_errors")) or
                          bool(receipt.get("runtime_errors")))
        if infrastructure:
            return tuple(CheckResult(check_id=c, status=CheckStatus.INFRA_ERROR,
                details={"reason": "grader_execution_invalid", "reachable": None,
                         "raw_result_preserved": native is not None}) for c in expected)
        translated: list[CheckResult] = []
        for row in native["outcomes"]:
            check_id = self._canonical_check_id(row["check_id"])
            raw_status = row["status"]
            status = {"pass": CheckStatus.PASS, "passed": CheckStatus.PASS,
                      "fail": CheckStatus.FAIL, "failed": CheckStatus.FAIL,
                      "not_run": CheckStatus.NOT_RUN, "skipped": CheckStatus.NOT_RUN,
                      "error": CheckStatus.ERROR_TASK}.get(raw_status)
            if status is None:
                raise ValueError(f"Unknown native check status: {raw_status}")
            details: dict[str, Any] = {"native_record": row, "reachable": None}
            phases = [r for r in native.get("reports", ()) if r["nodeid"] == row["check_id"]]
            # Capture output is evidence, never hidden worker feedback. More
            # specific task/infrastructure attribution is applied below when
            # the plugin supplies invocation/boundary evidence.
            if row.get("generic_task_crash"):
                status = CheckStatus.ERROR_TASK
                details.update(generic_task_crash=True, reachable=False,
                               blocking_reason=row.get("generic_task_crash_types", []))
            if row.get("infrastructure_error"):
                status = CheckStatus.INFRA_ERROR
            # In this pinned evaluator, setup/teardown creates or destroys trusted
            # fixtures; submission invocation is confined to the call phase.
            # Do not translate those fixture failures into implementation errors.
            if any(r["phase"] != "call" and r["raw_outcome"] == "failed" for r in phases):
                status = CheckStatus.INFRA_ERROR
                details["blocking_reason"] = "trusted_fixture_setup_or_teardown_failure"
            details["raw_phases"] = phases
            if "current_reachable" in row:
                details["reachable"] = row["current_reachable"]
            if row.get("blocked_by"):
                details["blocked_by"] = row["blocked_by"]
            translated.append(CheckResult(check_id=check_id, status=status, raw_status=raw_status,
                                           details=details))
        return tuple(translated)

    @staticmethod
    def _canonical_check_id(check_id: str) -> str:
        bare = check_id.lstrip("/")
        if "test_supplemental.py::test_" in bare:
            return "supplemental::" + bare.split("::test_", 1)[1]
        if bare.startswith("test_checkpoint_"):
            return "tests/" + bare
        return bare

    def next_stage(self, stage_id: str, *, termination_reason: str) -> StageSpec | None:
        stage = self.stage(stage_id)
        releases = {"submitted", "Submitted", "stage_completed", "stage_step_limit", "stage_cost_limit",
                    "MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
        terminal = {"budget_exhausted", "worker_budget_exhausted", "call_limit", "cancelled",
                    "worker_error", "infra_error", "technical_failure", "sequence_completed"}
        if termination_reason in terminal:
            return None
        if termination_reason not in releases:
            raise ValueError(f"Unbound stage termination reason: {termination_reason}")
        if stage.ordinal == len(self._stages):
            return None
        return self.stage(self.manifest["stages"][stage.ordinal])
