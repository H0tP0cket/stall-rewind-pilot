"""Registry-driven independent campaign scheduling with durable job outcomes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Any
from threading import Event, RLock

from .campaign import Campaign, CampaignConfig, PhaseJournal, needs_fresh_mine, validate_live_observer
from .contracts import canonical_json, utc_now
from .registry import read_registry
from .store import ArtifactStore


@dataclass(frozen=True, kw_only=True)
class CampaignJob:
    task_id: str
    config: CampaignConfig


OBSERVER_CIRCUIT_POLICIES = ("block_new_mines_on_any_circuit", "block_new_mines_on_systemic_circuits")


@dataclass(frozen=True, kw_only=True)
class BatchPlan:
    batch_id: str
    jobs: tuple[CampaignJob, ...]
    max_parallel: int
    version: str = "campaign-batch.v1"
    # Historical default: any run's latched observer circuit blocks every new
    # mine in the batch. The systemic policy blocks only on instrument failures
    # (consumer exceptions) or once this many distinct runs have latched, so one
    # run's provider timeout does not idle unrelated campaigns.
    observer_circuit_policy: str = "block_new_mines_on_any_circuit"
    observer_circuit_batch_threshold: int = 3

    def __post_init__(self):
        if (self.version != "campaign-batch.v1" or not isinstance(self.batch_id, str) or not self.batch_id.strip()
                or type(self.max_parallel) is not int or self.max_parallel < 1):
            raise ValueError("Invalid batch version, identity or parallelism")
        if self.observer_circuit_policy not in OBSERVER_CIRCUIT_POLICIES:
            raise ValueError("Unknown observer circuit policy")
        if type(self.observer_circuit_batch_threshold) is not int or self.observer_circuit_batch_threshold < 1:
            raise ValueError("observer_circuit_batch_threshold must be a positive integer")
        ids = [j.config.campaign_id for j in self.jobs]
        run_ids = [j.config.mining.run_id for j in self.jobs]
        if not ids or len(set(ids)) != len(ids) or len(set(run_ids)) != len(run_ids):
            raise ValueError("Batch requires unique campaign and natural-run identities")


class BatchCancellation:
    """Thread-safe stop scheduling request; active work is drained by its owner."""
    def __init__(self):
        self._requested, self._lock = False, RLock()
        self.reason: str | None = None

    def request(self, reason: str = "operator_requested") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Cancellation requires an explicit nonempty reason")
        with self._lock:
            if not self._requested:
                self.reason = reason
                self._requested = True

    def is_set(self) -> bool:
        return self._requested


def path_key(identity: str) -> str:
    """User identities remain in records, never interpreted as filesystem paths."""
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


NATIVE_GUARDED_KEY = "native_v4_guarded_calibration_v1"
NATIVE_GUARDED_VERSION = "native-v4-guarded-calibration/v1"


def _native_guarded_binding(config, *, task_id=None):
    """Select response protection without replacing native scoring/frontier."""
    if NATIVE_GUARDED_KEY not in config.mining.metadata:
        return None
    from .contracts import from_jsonable
    from .detector.threshold_policy_v1 import DetectorThresholdProfile, validate_detector_assignments
    value = config.mining.metadata[NATIVE_GUARDED_KEY]
    if (not isinstance(value, dict) or set(value) != {"version", "threshold_profile", "assignments"}
            or value["version"] != NATIVE_GUARDED_VERSION):
        raise ValueError("Declare the exact native-v4 guarded calibration activation object")
    if "certified_detector_v1" in config.mining.metadata:
        raise ValueError("Native-v4 activation cannot also select certificate rewriting")
    if (config.mode != "mine" or config.source_natural_run_id is not None
            or config.mining.purpose not in {"development", "calibration"}):
        raise ValueError("Native-v4 guarded activation is fresh development/calibration mining only")
    profile = from_jsonable(DetectorThresholdProfile, value["threshold_profile"])
    report = validate_detector_assignments(profile, assignments=value["assignments"])
    if task_id is not None:
        rows = [row for row in report["assignments"]
                if (row["task_id"], row["model"]) == (task_id, config.mining.model)]
        if len(rows) != 1 or rows[0]["detector_settings"] != dict(config.mining.detector_settings):
            raise ValueError("Executed native task/model/repeat differs from its threshold assignment")
    return profile, report


def validate_native_v4_guarded_batch(plan):
    """Bind every emitted repeat before any registry load or worker scheduling."""
    flags = [NATIVE_GUARDED_KEY in job.config.mining.metadata for job in plan.jobs]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError("A native-v4 guarded batch cannot mix selected and legacy instruments")
    binding, jobs = None, []
    for job in plan.jobs:
        _, report = _native_guarded_binding(job.config, task_id=job.task_id)
        if binding is None:
            binding = report
        elif report != binding:
            raise ValueError("Native-v4 threshold profile/assignments differ between jobs")
        jobs.append({"task_id": job.task_id, "model": job.config.mining.model,
            "campaign_id": job.config.campaign_id, "run_id": job.config.mining.run_id,
            "repeat_id": job.config.mining.repeat_id,
            "detector_settings": dict(job.config.mining.detector_settings)})
    return {"version": NATIVE_GUARDED_VERSION, "threshold_binding": binding, "executed_jobs": jobs,
            "frontier_policy": "native_task_adapter_unchanged"}


def _native_guarded_model(config):
    from .execution.received_response_guard_v1 import ReceivedResponseGuardV1
    policy = config.worker_request_policy
    return ReceivedResponseGuardV1(model_name=config.model, model_settings=dict(config.model_settings),
        max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
        timeout_seconds=policy.timeout_seconds,
        usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)


def live_campaign(store: ArtifactStore, task: Any, config: CampaignConfig, *, scratch: Path) -> Campaign:
    validate_live_observer(config, require_live=(config.mining.purpose in {"calibration", "main"}
                                                 and needs_fresh_mine(store, config)))
    from .detector.core import DetectorConfig
    from .execution.model import openrouter_model_factory
    from .mining.runner import MiningRunner
    from .providers import OpenRouterObserverClient, capture_route_metadata
    if config.fixture:
        raise ValueError("Live factory refuses scripted fixture configurations")
    if not config.mining.detector_settings:
        raise ValueError("Declare detector settings; no implicit calibration threshold")
    from .detector.activation_v1 import selected, build_runner
    native_binding = (_native_guarded_binding(config, task_id=task.describe().task_id)
                      if NATIVE_GUARDED_KEY in config.mining.metadata else None)
    if native_binding is not None:
        from .contracts import to_jsonable
        from .execution.received_response_guard_v1 import RULE
        profile, binding = native_binding
        for ref in profile.evidence_refs:
            store.get_bytes(ref)
        runner = MiningRunner(model_factory=_native_guarded_model,
            worker_image=task.describe().environment_version,
            detector_config=DetectorConfig(**config.mining.detector_settings),
            scratch_root=scratch / "runs", network="none")
        store.put_record("native_v4_guarded_activation_v1", config.campaign_id,
            {"version": NATIVE_GUARDED_VERSION, "task": to_jsonable(task.describe()),
             "threshold_binding": binding, "received_response_guard": RULE,
             "frontier_policy": "native_task_adapter_unchanged"})
    elif selected(config):
        # Task-native certificates and every numerical binding fail closed
        # before even route metadata lookup; no status-to-proof inference.
        runner = build_runner(store, task, config, scratch=scratch / "runs")
    else:
        runner = MiningRunner(model_factory=openrouter_model_factory(),
            worker_image=task.describe().environment_version,
            detector_config=DetectorConfig(**config.mining.detector_settings),
            scratch_root=scratch / "runs", network="none")
    PhaseJournal(store, config.campaign_id).execute("model_route_metadata", {"model": config.mining.model},
        lambda: capture_route_metadata(store, config.mining.model, campaign_id=config.campaign_id))
    return Campaign(store, task, runner, OpenRouterObserverClient(store))


def _verify_declared_sources(job: CampaignJob, registered: Any) -> None:
    from .launch_sources import verify_declared_sources
    verify_declared_sources(job.config, registered)


def run_batch(plan: BatchPlan, *, registry: Path, store: ArtifactStore, scratch: Path,
              output: Path, artifact_root: Path | None = None,
              campaign_factory: Callable[..., Campaign] = live_campaign,
              cancellation: BatchCancellation | None = None) -> dict[str, Any]:
    """Independent jobs may overlap; grading inside each job remains synchronous.

    Every failed job receives its own receipt. A failed or uncertain phase is
    never silently retried on restart; use an explicit new campaign after review.
    No success/stall outcome is used to admit jobs or change their allowance.
    """
    from .detector.activation_v1 import validate_batch
    native_activation = validate_native_v4_guarded_batch(plan)
    detector_activation = validate_batch(plan)
    tasks = read_registry(registry, root=artifact_root)
    missing = sorted({j.task_id for j in plan.jobs} - tasks.keys())
    if missing:
        raise ValueError(f"Unknown registry task IDs: {missing}")
    store.put_record("campaign_batch", plan.batch_id, plan)
    if native_activation is not None:
        store.put_record("native_v4_guarded_batch_activation_v1", plan.batch_id, native_activation)
    if detector_activation is not None:
        store.put_record("certified_detector_batch_activation_v1", plan.batch_id, detector_activation)
    registry_ref = store.put_bytes(registry.read_bytes(), media_type="application/json")
    store.put_record("campaign_batch_registry", plan.batch_id, {"registry_ref": registry_ref})
    journal = PhaseJournal(store, f"batch:{plan.batch_id}")
    cancellation = cancellation or BatchCancellation()
    output.mkdir(parents=True, exist_ok=True)
    cancellation_recorded = False
    from .observers.live import circuit_is_systemic, observer_circuits
    observer_breaker = Event()
    observer_run_ids = {j.config.source_natural_run_id or j.config.mining.run_id for j in plan.jobs}
    systemic_only = plan.observer_circuit_policy == "block_new_mines_on_systemic_circuits"

    class ObserverCircuitOpen(RuntimeError):
        pass

    def circuits_block(circuits: list[dict[str, Any]]) -> bool:
        if not systemic_only:
            return bool(circuits)
        return (any(circuit_is_systemic(c) for c in circuits)
                or len({c["run_id"] for c in circuits}) >= plan.observer_circuit_batch_threshold)

    def circuit_open() -> bool:
        if observer_breaker.is_set() or circuits_block(observer_circuits(store, run_ids=observer_run_ids)):
            observer_breaker.set()
        return observer_breaker.is_set()

    def on_observer_circuit_open(receipt: dict[str, Any]) -> None:
        # Service already durably retained its trigger; decide batch scope here.
        if not systemic_only or circuit_is_systemic(receipt):
            observer_breaker.set()
        else:
            circuit_open()

    def before_fresh_mine() -> None:
        if circuit_open():
            raise ObserverCircuitOpen("Live observer circuit blocks a new mine; active workers drain unchanged")

    def observer_blocked(job: CampaignJob) -> dict[str, Any]:
        return {"campaign_id": job.config.campaign_id, "task_id": job.task_id,
                "job_status": "observer_circuit_blocked_not_started",
                "reason": "Live observer circuit is latched; no new mine launched"}

    def record_cancellation(active: list[str]) -> None:
        nonlocal cancellation_recorded
        if cancellation.is_set() and not cancellation_recorded:
            receipt = {"batch_id": plan.batch_id, "recorded_at": utc_now(), "reason": cancellation.reason,
                "mode": "active_campaigns_drain", "active_campaigns": active,
                "cost_policy": "All active work remains recorded in its original CostLedger; active requests may continue."}
            identity = plan.batch_id + ":" + hashlib.sha256(canonical_json(receipt)).hexdigest()
            store.put_record("batch_cancellation", identity, receipt)
            cancellation_recorded = True

    def cancelled(job: CampaignJob) -> dict[str, Any]:
        return {"campaign_id": job.config.campaign_id, "task_id": job.task_id,
                "job_status": "cancelled_not_started", "cancellation_reason": cancellation.reason}

    def execute(job: CampaignJob) -> dict[str, Any]:
        if cancellation.is_set():
            return cancelled(job)
        if circuit_open() and needs_fresh_mine(store, job.config):
            return observer_blocked(job)
        key = path_key(job.config.campaign_id)
        def operation():
            validate_live_observer(job.config, require_live=(job.config.mining.purpose in {"calibration", "main"}
                                                            and needs_fresh_mine(store, job.config)))
            _verify_declared_sources(job, tasks[job.task_id])
            task = tasks[job.task_id].load(scratch_root=scratch / key / "grader", purpose=job.config.mining.purpose)
            from .launch_sources import verify_loaded_task
            if job.config.mining.metadata.get("source_binding") is not None:
                verify_loaded_task(job.config, task.describe())
            campaign = campaign_factory(store, task, job.config, scratch=scratch / key)
            campaign.on_observer_circuit_open = on_observer_circuit_open
            campaign.before_fresh_mine = before_fresh_mine
            result = campaign.run(job.config, export_dir=output / key)
            return {"campaign_id": job.config.campaign_id, "task_id": job.task_id,
                    "pipeline_status": result["pipeline_status"], "output": str(output / key),
                    "selection": result["selection"]}
        try:
            result = journal.execute(f"job:{job.config.campaign_id}", job, operation)
            return {**result, "job_status": "completed"}
        except Exception as exc:
            if isinstance(exc, ObserverCircuitOpen):
                return {**observer_blocked(job), "error_type": type(exc).__name__, "error": str(exc)}
            return {"campaign_id": job.config.campaign_id, "task_id": job.task_id,
                    "job_status": "failed_or_uncertain", "error_type": type(exc).__name__, "error": str(exc)}

    outcomes = {}
    def retain(job: CampaignJob, outcome: dict[str, Any]) -> None:
        outcome = {**outcome, "phase_id": f"batch:{plan.batch_id}:job:{job.config.campaign_id}"}
        identity = f"{plan.batch_id}:{job.config.campaign_id}:" + hashlib.sha256(canonical_json(outcome)).hexdigest()
        store.put_record("batch_job_outcome", identity, {"batch_id": plan.batch_id, **outcome})
        outcomes[job.config.campaign_id] = outcome
        (output / "batch-progress.json").write_bytes(canonical_json({
            "batch_id": plan.batch_id, "updated_at": utc_now(), "cancellation_requested": cancellation.is_set(),
            "jobs": [outcomes[j.config.campaign_id] for j in plan.jobs if j.config.campaign_id in outcomes]}) + b"\n")

    next_job = 0
    with ThreadPoolExecutor(max_workers=plan.max_parallel) as pool:
        pending = {}
        while pending or next_job < len(plan.jobs):
            try:
                while not cancellation.is_set() and len(pending) < plan.max_parallel and next_job < len(plan.jobs):
                    job = plan.jobs[next_job]
                    next_job += 1
                    if circuit_open() and needs_fresh_mine(store, job.config):
                        retain(job, observer_blocked(job))
                        continue
                    pending[pool.submit(execute, job)] = job
                if cancellation.is_set():
                    record_cancellation([j.config.campaign_id for j in pending.values()])
                    for job in plan.jobs[next_job:]:
                        retain(job, cancelled(job))
                    next_job = len(plan.jobs)
                if not pending:
                    continue
                done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    job = pending[future]
                    try:
                        outcome = future.result()
                    except BaseException as exc:
                        # A worker interruption leaves its started phase uncertain;
                        # never discard another job's result or schedule a replay.
                        cancellation.request(f"active_job_{type(exc).__name__}")
                        outcome = {"campaign_id": job.config.campaign_id, "task_id": job.task_id,
                            "job_status": "failed_or_uncertain", "error_type": type(exc).__name__, "error": str(exc)}
                    retain(job, outcome)
                    del pending[future]
            except KeyboardInterrupt:
                cancellation.request("KeyboardInterrupt")
    ordered = [outcomes[j.config.campaign_id] for j in plan.jobs]
    record_cancellation([])
    report = {"batch_id": plan.batch_id, "jobs": ordered,
              "passed": all(o.get("pipeline_status") == "outcomes_recorded" for o in ordered),
              "cancellation_requested": cancellation.is_set(), "cancellation_reason": cancellation.reason,
              "cancellation_policy": "active_campaigns_drain",
              "observer_circuit_open": circuit_open(),
              "observer_circuit_policy": f"{plan.observer_circuit_policy}; threshold={plan.observer_circuit_batch_threshold}; "
                                         "active_workers_drain; saved_completed_work_reusable",
              "observer_circuits": [{"run_id": c["run_id"], "reason": c.get("reason"), "systemic": circuit_is_systemic(c)}
                                    for c in observer_circuits(store, run_ids=observer_run_ids)],
              "registry_ref": registry_ref}
    # A rerun may report an already failed phase differently; preserve each
    # inspection by content identity instead of overwriting the first result.
    report_id = plan.batch_id + ":" + hashlib.sha256(canonical_json(report)).hexdigest()
    store.put_record("batch_result", report_id, report)
    (output / "batch-result.json").write_bytes(canonical_json(report) + b"\n")
    return report
