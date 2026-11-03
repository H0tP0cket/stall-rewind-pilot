"""Offline launch checks, not a worker run or scientific-readiness certificate.

Only registry adapter construction/description, local artifact reads and Docker
image inspection are performed. No image is pulled, container started, model
client constructed, credential loaded, or campaign phase recorded.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from typing import Any, Callable, Iterator

from .campaign import CampaignConfig
from .contracts import TaskAdapter, canonical_json, from_jsonable, to_jsonable
from .detector.core import DetectorConfig
from .factory import BatchPlan, CampaignJob
from .freeze import settings_binding, validate_main_freeze
from .registry import read_registry
from .store import ArtifactStore


class _ReadOnlyStore(ArtifactStore):
    """Reuse integrity-checking readers without initializing or changing a store."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.blobs = self.root / "blobs"
        self.database = self.root / "journal.sqlite3"
        if not self.database.is_file() or not self.blobs.is_dir():
            raise ValueError("An existing artifact store is required to inspect main-study freeze evidence")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            yield db
        finally:
            db.close()


def inspect_local_image(image: str) -> dict[str, Any]:
    """Inspect exactly one local image; never invoke Docker run/build/pull."""
    if not isinstance(image, str) or not image.strip() or image.startswith("-"):
        raise ValueError("Task must declare a nonempty Docker image reference")
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                            text=True, timeout=20, check=False)
    if result.returncode:
        raise ValueError("Local Docker image inspection failed: " + (result.stderr.strip()[:2000]
                         or f"exit status {result.returncode}"))
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != 1 or not rows[0].get("Id"):
        raise ValueError("Docker inspect did not return exactly one image identity")
    row = rows[0]
    return {"declared_image": image, "local_image_id": row["Id"],
            "repo_digests": row.get("RepoDigests") or [],
            "architecture": row.get("Architecture"), "os": row.get("Os"),
            "scope": "local availability only; container execution and grading untested"}


def _check(rows: list[dict[str, Any]], binding: str, operation: Callable[[], Any]) -> Any:
    try:
        value = operation()
        rows.append({"binding": binding, "status": "passed", "evidence": to_jsonable(value)})
        return value
    except Exception as exc:
        rows.append({"binding": binding, "status": "failed", "error_type": type(exc).__name__,
                     "error": str(exc)})
        return None


def _interface(task: TaskAdapter, record: Any, config: CampaignConfig) -> dict[str, Any]:
    from .tasks.prompts import prompt_for_task
    if not isinstance(task, TaskAdapter):
        raise TypeError("Loaded adapter does not implement TaskAdapter")
    # Bind the actual runner call shapes, without executing materialization,
    # grading or native stage progression.
    inspect.signature(task.materialize).bind(Path("workspace"), stage_id="stage")
    inspect.signature(task.grade).bind(Path("workspace"), stage_id="stage",
                                       private_output_dir=Path("private-grade"), runtime=None)
    inspect.signature(task.next_stage).bind("stage", termination_reason="submitted")
    if not record.stages:
        raise ValueError("Task has no initial stage")
    prompts = []
    for stage in record.stages:
        if task.stage(stage.stage_id) != stage:
            raise ValueError(f"Adapter stage lookup differs from description: {stage.stage_id}")
        prompts.append(prompt_for_task(task, stage).binding())
    initial = task.stage(config.mining.stage_id) if config.mining.stage_id else record.stages[0]
    return {"protocol": "TaskAdapter", "initial_stage_id": initial.stage_id,
            "stage_ids": [s.stage_id for s in record.stages],
            "call_signatures": ["materialize", "grade", "next_stage"],
            "worker_task_prompts": prompts,
            "behavior_executed": False}


def preflight_campaigns(*, registry: Path, config: Path | None = None,
                        task_id: str | None = None, plan: Path | None = None,
                        artifact_root: Path | None = None, store: Path | None = None) -> dict[str, Any]:
    """Return all independently testable blockers for one config or a BatchPlan.

    Paths have the same meaning as the run/batch CLI. Main freeze checks require
    an existing store; development/calibration checks do not create one.
    ``passed`` means only that the listed offline checks found no blocker.
    """
    checks: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": "campaign-preflight.v1", "checks": checks, "jobs": reports,
        "scope": "offline launch bindings only; not scientific readiness or live execution proof",
        "untested": [
            {"binding": "provider", "status": "untested",
             "reason": "No credentials, route metadata, API compatibility, usage or model responses checked"},
            {"binding": "extractor", "status": "untested",
             "reason": "No observer calls, extraction quality, visibility audits or token matching executed"},
            {"binding": "calibration", "status": "untested",
             "reason": "No calibration runs or adequacy assessment; main freeze evidence is checked separately"},
            {"binding": "runtime", "status": "untested",
             "reason": "No container start, worker tools, evaluator execution, restore or isolation replay"},
            {"binding": "campaign_resume", "status": "untested",
             "reason": "Existing campaign phases and source-natural-run reuse are checked by the campaign runner"},
        ]}
    jobs: tuple[CampaignJob, ...] = ()

    def parse_input() -> dict[str, Any]:
        nonlocal jobs
        if (config is None) == (plan is None):
            raise ValueError("Supply exactly one config or batch plan")
        if plan is not None and task_id is not None:
            raise ValueError("Batch plans declare their own task IDs")
        if config is not None and not task_id:
            raise ValueError("A single campaign requires a registry task ID")
        path = plan if plan is not None else config
        raw = path.read_bytes()
        data = json.loads(raw)
        if plan is not None:
            parsed = from_jsonable(BatchPlan, data)
            jobs = parsed.jobs
        else:
            parsed = from_jsonable(CampaignConfig, data)
            jobs = (CampaignJob(task_id=task_id, config=parsed),)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "schema": type(parsed).__name__, "campaign_count": len(jobs),
                "includes": ["deserialization", "known_arms", "campaign_invariants", "analysis_settings"]}

    _check(checks, "campaign_input", parse_input)
    entries = None

    def load_registry() -> dict[str, Any]:
        nonlocal entries
        entries = read_registry(registry, root=artifact_root)
        return {"path": str(registry), "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                "registered_task_ids": sorted(entries)}

    _check(checks, "registry", load_registry)
    # Adapter constructors receive a disposable scratch path, but no adapter
    # method that runs tools or grades work is invoked.
    with tempfile.TemporaryDirectory(prefix="campaign-preflight-") as scratch:
        for index, job in enumerate(jobs):
            cfg = job.config
            rows: list[dict[str, Any]] = []
            item = {"campaign_id": cfg.campaign_id, "task_id": job.task_id, "checks": rows,
                    "declared": {"purpose": cfg.mining.purpose, "mode": cfg.mode,
                        "worker_model": cfg.mining.model, "model_settings": cfg.mining.model_settings,
                        "observer_model": cfg.mining.model, "observer_settings": cfg.observer_settings,
                        "calibration_status": cfg.calibration_status, "arms": cfg.arms,
                        "recovery_plan": cfg.recovery_plan}}
            reports.append(item)

            def launch_config() -> dict[str, Any]:
                if cfg.fixture:
                    raise ValueError("Live factory refuses scripted fixture configurations")
                return {"fixture": False}

            _check(rows, "live_factory_config", launch_config)

            def detector() -> dict[str, Any]:
                if not cfg.mining.detector_settings:
                    raise ValueError("Declare detector settings; no implicit calibration threshold")
                return from_jsonable(DetectorConfig, cfg.mining.detector_settings).to_record()

            _check(rows, "detector_settings", detector)
            task = record = None

            def load_task() -> dict[str, Any]:
                nonlocal task, record
                if entries is None:
                    raise ValueError("Registry could not be loaded")
                if job.task_id not in entries:
                    raise ValueError(f"Unknown registry task ID: {job.task_id}")
                from .launch_sources import verify_declared_sources, verify_loaded_task
                verify_declared_sources(cfg, entries[job.task_id])
                task = entries[job.task_id].load(scratch_root=Path(scratch) / str(index),
                                                purpose=cfg.mining.purpose)
                record = task.describe()
                verify_loaded_task(cfg, record)
                return {"task_id": record.task_id, "family_id": record.family_id,
                        "task_version": record.task_version, "environment_version": record.environment_version,
                        "admission": record.admission, "purpose": cfg.mining.purpose,
                        "adapter": entries[job.task_id].entry["adapter"],
                        "verified_by": "RegisteredTask.load and adapter admission attestation"}

            loaded = _check(rows, "task_admission_and_purpose", load_task)
            if loaded is not None:
                _check(rows, "task_interface", lambda: _interface(task, record, cfg))
                _check(rows, "local_environment", lambda: inspect_local_image(record.environment_version))
                item["settings_digest"] = hashlib.sha256(canonical_json(settings_binding(cfg, record))).hexdigest()
            else:
                for name in ("task_interface", "local_environment"):
                    rows.append({"binding": name, "status": "skipped", "reason": "Task admission/load failed"})
            if cfg.mining.purpose != "main":
                rows.append({"binding": "main_freeze", "status": "not_applicable", "purpose": cfg.mining.purpose})
            elif loaded is None:
                rows.append({"binding": "main_freeze", "status": "skipped", "reason": "Task admission/load failed"})
            else:
                def freeze() -> Any:
                    if store is None:
                        raise ValueError("Main preflight requires --store for existing freeze evidence")
                    return validate_main_freeze(_ReadOnlyStore(store), cfg, record)
                _check(rows, "main_freeze", freeze)
            item["passed"] = not any(r["status"] == "failed" for r in rows)
    report["blockers"] = [dict(row) for row in checks if row["status"] == "failed"]
    report["blockers"].extend({"campaign_id": job["campaign_id"], "task_id": job["task_id"], **row}
                              for job in reports for row in job["checks"] if row["status"] == "failed")
    report["passed"] = not report["blockers"]
    return to_jsonable(report)
