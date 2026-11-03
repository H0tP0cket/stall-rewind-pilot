"""Reusable campaign coordination; every phase uses the shared worker Runner.

Immutable phase receipts provide restart safety. A completed phase is reused.
A crashed or failed phase is preserved and requires a new explicit attempt,
not a silent rerun of potentially charged calls. The first development campaign
uses serial branches; concurrency is independent of scientific treatment rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import fcntl
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from .accounting import CostLedger
from .contracts import (AdmissionStatus, Annotation, ArtifactRef, Checkpoint, RunConfig, RunResult,
                        Runner, TaskAdapter, canonical_json, from_jsonable, to_jsonable, utc_now,
                        worker_operational_settings)
from .store import ArtifactStore
from .observers.tokens import TokenMatchConfig
from .observers.context import ContextPolicy
from .observers.live import LiveObserverPolicy


class PhaseUncertain(RuntimeError):
    pass


class PhaseFailed(RuntimeError):
    pass


class PhaseJournal:
    """Local process-safe claims over immutable store records."""
    def __init__(self, store: ArtifactStore, campaign_id: str):
        self.store, self.campaign_id = store, campaign_id
        self.lock_root = store.root / "locks"
        self.lock_root.mkdir(exist_ok=True, mode=0o700)

    def execute(self, name: str, inputs: Any, operation: Callable[[], Any]) -> Any:
        phase_id = f"{self.campaign_id}:{name}"
        digest = hashlib.sha256(canonical_json({"phase": phase_id, "inputs": inputs})).hexdigest()
        self.store.put_record("phase_spec", phase_id, {"inputs": inputs, "digest": digest})
        lockfile = self.lock_root / hashlib.sha256(phase_id.encode()).hexdigest()
        with lockfile.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                finished = self.store.get_record("phase_finished", phase_id)
            except KeyError:
                finished = None
            if finished is not None:
                if finished["status"] == "ok":
                    return finished["result"]
                raise PhaseFailed(f"Phase {phase_id} previously failed; receipt preserved")
            try:
                self.store.get_record("phase_started", phase_id)
            except KeyError:
                pass
            else:
                raise PhaseUncertain(f"Phase {phase_id} started without a final receipt; inspect before a new attempt")
            self.store.put_record("phase_started", phase_id, {"started_at": utc_now(), "digest": digest})
            try:
                result = to_jsonable(operation())
            except Exception as exc:
                self.store.put_record("phase_finished", phase_id, {
                    "status": "error", "finished_at": utc_now(),
                    "error_type": type(exc).__name__, "error": str(exc),
                })
                raise
            self.store.put_record("phase_finished", phase_id, {
                "status": "ok", "finished_at": utc_now(), "result": result,
            })
            return result


@dataclass(frozen=True, kw_only=True)
class ObserverAuditConfig:
    checkpoint_ordinals: tuple[int, ...] = ()
    selected_checkpoint: bool = False
    full_prefix_repeats: int = 1

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in self.checkpoint_ordinals):
            raise ValueError("Audit checkpoint ordinals must be nonnegative integers")
        if len(set(self.checkpoint_ordinals)) != len(self.checkpoint_ordinals):
            raise ValueError("Audit checkpoint ordinals must be unique")
        if type(self.selected_checkpoint) is not bool:
            raise ValueError("selected_checkpoint must be a boolean")
        if type(self.full_prefix_repeats) is not int or self.full_prefix_repeats < 1:
            raise ValueError("Audit full-prefix repeat count must be a positive integer")

    @property
    def configured(self) -> bool:
        return bool(self.checkpoint_ordinals or self.selected_checkpoint)


@dataclass(frozen=True, kw_only=True)
class PacketSettings:
    # None preserves the legacy byte proxy, explicitly labelled undeclared.
    token_counter: str | None = None
    encoding_name: str | None = None
    matching: TokenMatchConfig = field(default_factory=TokenMatchConfig)
    filters: tuple[str, ...] = ("facts_only", "facts_lessons")

    def __post_init__(self) -> None:
        from .observers.views import FILTERS
        if self.token_counter not in {None, "byte_proxy", "tiktoken"}:
            raise ValueError("Packet token_counter must be byte_proxy or tiktoken")
        if self.token_counter == "tiktoken":
            if not isinstance(self.encoding_name, str) or not self.encoding_name.strip():
                raise ValueError("Local tiktoken requires an explicit encoding_name")
        elif self.encoding_name is not None:
            raise ValueError("encoding_name applies only to the tiktoken counter")
        if not isinstance(self.matching, TokenMatchConfig):
            raise TypeError("Packet matching must be a TokenMatchConfig")
        if len(set(self.filters)) != len(self.filters) or any(f not in FILTERS for f in self.filters):
            raise ValueError("Packet filters must be unique registered filter names")

    def build_counter(self, model: str):
        from .observers.tokens import byte_proxy_counter, tiktoken_counter
        return (tiktoken_counter(model, self.encoding_name) if self.token_counter == "tiktoken"
                else byte_proxy_counter(model))


@dataclass(frozen=True, kw_only=True)
class CampaignConfig:
    campaign_id: str
    mining: RunConfig
    # Deferred recovery is represented by missing choices, never placeholder
    # allowances, repeat IDs or a default treatment allocation. Existing saved
    # campaigns retain the explicit values in their immutable wire records.
    recovery_generated_tokens: int | None = None
    horizons: tuple[int, ...] = ()
    repeats: tuple[str, ...] = ()
    engineering_fallback: bool = False
    observer_settings: Mapping[str, Any] = field(default_factory=dict)
    observer_source_policy: str = "legacy-full-trajectory/v1"
    observer_include_returned_reasoning: bool | None = None
    observer_context_policy: ContextPolicy | None = None
    observer_context_counter: Mapping[str, Any] | None = None
    live_observer: LiveObserverPolicy | None = None
    observation_mask_policy: str | None = None
    observer_audit: ObserverAuditConfig = field(default_factory=ObserverAuditConfig)
    packet_settings: PacketSettings = field(default_factory=PacketSettings)
    arms: tuple[str, ...] = ()
    calibration_status: str = "unfrozen_development"
    fixture: bool = False
    source_natural_run_id: str | None = None
    mode: str = "full"  # mine / observe / full; all use the same natural Runner
    schedule_seed: int | None = None
    freeze_ref: ArtifactRef | None = None
    analysis_settings: Mapping[str, Any] = field(default_factory=dict)
    near_budget_tokens: int | None = None
    resource_horizons: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    dollar_horizons: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        from .experiments.treatments import resolve_arms
        from .analysis.effects import AnalysisSpec
        resolve_arms(self.arms)
        AnalysisSpec(horizons=self.horizons, **self.analysis_settings)
        if not isinstance(self.observer_audit, ObserverAuditConfig) or not isinstance(self.packet_settings, PacketSettings):
            raise TypeError("Campaign observer_audit and packet_settings must use their declared schemas")
        if self.mining.campaign_id != self.campaign_id:
            raise ValueError("Mining run must belong to this campaign")
        if self.mode not in {"mine", "observe", "full"}:
            raise ValueError("Campaign mode must be mine, observe, or full")
        if self.observer_source_policy not in {"legacy-full-trajectory/v1", "native-session-primary/v1"}:
            raise ValueError("Unknown observer source policy")
        from .experiments.observation_mask import (PRIMARY_WINDOW_POLICY, SECONDARY_ALL_HISTORY_POLICY,
                                                  ENGINEERING_ALL_HISTORY_POLICY)
        if self.observation_mask_policy not in {None, PRIMARY_WINDOW_POLICY, SECONDARY_ALL_HISTORY_POLICY,
                                                ENGINEERING_ALL_HISTORY_POLICY}:
            raise ValueError("Unknown observation mask policy")
        if self.observer_include_returned_reasoning is not None and type(self.observer_include_returned_reasoning) is not bool:
            raise ValueError("Returned-reasoning source access must be an explicit boolean or pending")
        if self.observer_context_policy is not None and not isinstance(self.observer_context_policy, ContextPolicy):
            raise TypeError("Observer context policy must use its declared schema")
        if self.live_observer is not None and not isinstance(self.live_observer, LiveObserverPolicy):
            raise TypeError("Live observer policy must use its declared schema")
        if self.observer_context_counter is not None:
            raw = self.observer_context_counter
            if not isinstance(raw, Mapping) or not set(raw) <= {"token_counter", "encoding_name"} or raw.get("token_counter") is None:
                raise ValueError("Observer request counter requires a declared token_counter and optional encoding_name")
            PacketSettings(**raw)
            if self.observer_context_policy is None or self.observer_context_policy.max_input_tokens is None:
                raise ValueError("Observer request counter requires a declared input-token context policy")
        if (self.mode != "mine" or self.live_observer is not None) and self.observer_source_policy == "native-session-primary/v1" and self.observer_include_returned_reasoning is None:
            raise ValueError("Declare returned-reasoning source access before extraction; the scientific choice remains pending")
        if self.near_budget_tokens is not None and self.near_budget_tokens < 0:
            raise ValueError("Near-budget threshold must be nonnegative when declared")
        if self.mining.purpose not in {"development", "calibration", "main"}:
            raise ValueError("Unknown campaign purpose")
        if self.recovery_generated_tokens is not None and (
                type(self.recovery_generated_tokens) is not int or self.recovery_generated_tokens <= 0):
            raise ValueError("recovery_generated_tokens must be a positive integer when declared")
        if any(h <= 0 or (self.recovery_generated_tokens is not None and h > self.recovery_generated_tokens)
               for h in self.horizons):
            raise ValueError("Declared horizons must be positive and within the recovery allowance when declared")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Horizons must be strictly increasing")
        if any(not isinstance(r, str) or not r.strip() for r in self.repeats) or len(set(self.repeats)) != len(self.repeats):
            raise ValueError("Declared repeat identities must be nonempty strings and unique")
        if self.mode == "full":
            missing = [field for field in self.recovery_plan["pending_fields"]
                       if field != "arms" or not self.fixture]
            if missing:
                raise ValueError("Full mode requires explicit recovery settings: " + ", ".join(missing)
                                 + "; use mode=mine or mode=observe to defer recovery")
        if self.mining.model.startswith(("scripted/", "fixture/", "dev/")) and not self.fixture:
            raise ValueError("Scripted workers must be explicitly marked fixture=True")
        if self.mining.purpose == "main":
            if self.calibration_status != "frozen" or self.engineering_fallback or self.fixture or self.freeze_ref is None:
                raise ValueError("Main campaigns require frozen settings and natural primary states")

    @property
    def recovery_plan(self) -> dict[str, Any]:
        return {"execution": "enabled" if self.mode == "full" else "deferred",
                "settings_status": "pending" if self._pending_recovery_fields else "declared",
                "pending_fields": self._pending_recovery_fields}

    @property
    def _pending_recovery_fields(self) -> list[str]:
        return [name for name in ("recovery_generated_tokens", "horizons", "repeats", "arms")
                if not getattr(self, name)]


def select_checkpoint(store: ArtifactStore, result: RunResult, *, engineering_fallback: bool) -> tuple[Checkpoint | None, str]:
    from .experiments.selection import first_stall_annotation
    cps = store.list_checkpoints(result.run_id)
    annotations = [store.get_record("annotation", r["record_id"])
                   for r in store.list_records("annotation", run_id=result.run_id)]
    first_stall = first_stall_annotation(annotations)
    if first_stall is not None:
        # The first prospectively selected warning is never swapped for a more
        # convenient later outcome. An unbranchable first S remains a failure.
        cp = store.get_checkpoint(first_stall["checkpoint_id"])
        return cp, "primary_s1"
    if not engineering_fallback:
        return None, "no_natural_stall"
    candidates = [c for c in cps if c.branchable and c.evaluation_ref and c.ordinal > 0]
    candidates = [c for c in candidates if store.get_json(c.evaluation_ref).get("valid")]
    # Explicit instrument validation only: select the earliest usable mutation
    # checkpoint with a prior same-target donor; not based on later recovery.
    for cp in candidates:
        if any(p.ordinal < cp.ordinal and p.scope.target_id == cp.scope.target_id and p.branchable for p in cps):
            return cp, "engineering_checkpoint"
    return None, "no_branchable_engineering_checkpoint"


def analysis_stop_reason(store: ArtifactStore, result: RunResult) -> str:
    """Normalize a runner exit for censoring while retaining its raw receipt."""
    if result.status in {"error", "failed", "invalid", "technical_failure"}:
        return "technical_failure"
    termination = result.details.get("termination", {})
    if termination.get("policy_version") == "termination.v1":
        # This selects an analysis category, not eligibility for absorption.
        # The store reader independently validates the terminal evidence links.
        return {"voluntary_finish": "voluntary_finish", "verified_complete": "fixed_target_complete",
                "budget_exhausted": "token_boundary", "call_limit": "max_calls",
                "accounting_interruption": "accounting_interruption", "worker_error": "worker_error",
                "idle_termination": "idle", "technical_failure": "technical_failure",
                "interrupted": "interrupted"}.get(termination.get("normalized_reason"), "unknown")
    reason = result.termination_reason.lower()
    if reason == "fixed_target_complete":
        return "fixed_target_complete"
    if "budget" in reason or "token_boundary" in reason:
        return "token_boundary"
    if "call" in reason and "limit" in reason:
        return "max_calls"
    if "formaterror" in reason:
        return "worker_error"
    if "idle" in reason:
        return "idle"
    if "submitted" in reason or reason == "legitimate_stop":
        return "legitimate_stop"
    return "unknown"


def _run_observer_audits(store: ArtifactStore, run_id: str, selected: Checkpoint | None,
                        config: CampaignConfig, extractor: Any, client: Any,
                        phases: PhaseJournal, chain_namespace: str | None, scope: Any = None,
                        context_counter: Any = None) -> dict[str, Any]:
    from .observers.artifacts import load_version
    from .observers.pipeline import audit_checkpoint
    settings = config.observer_audit
    report = {"settings": to_jsonable(settings), "configured": settings.configured,
              "status": "not_configured", "results": [], "unavailable": [],
              "scientific_quality_gate": "evidence_required; independent extraction is not ground truth"}
    if not settings.configured:
        return report
    if config.mode == "mine" and config.live_observer is None:
        report["status"] = "deferred_mine_only"
        return report
    checkpoints = {cp.ordinal: cp for cp in store.list_checkpoints(run_id)}
    requested = {checkpoints[n].checkpoint_id: checkpoints[n] for n in settings.checkpoint_ordinals if n in checkpoints}
    report["unavailable"] = [{"ordinal": n, "reason": "Checkpoint not produced by this natural trajectory"}
                             for n in settings.checkpoint_ordinals if n not in checkpoints]
    if settings.selected_checkpoint:
        if selected is None:
            report["unavailable"].append({"selection": "selected_checkpoint", "reason": "No selected checkpoint"})
        else:
            requested[selected.checkpoint_id] = selected
    for cp in sorted(requested.values(), key=lambda cp: cp.ordinal):
        def operation():
            result = audit_checkpoint(store, run_id, cp.checkpoint_id, client, extractor,
                repeats=settings.full_prefix_repeats, chain_namespace=chain_namespace, scope=scope,
                context_counter=context_counter)
            primary = load_version(store, result["primary_version_id"])
            references = [load_version(store, vid) for vid in result["reference_version_ids"]]
            statuses = [v.status for v in references]
            return {**result, "primary_status": primary.status, "reference_statuses": statuses,
                    "call_record_ids": [cid for v in references for cid in v.call_record_ids],
                    "status": "failed" if any(s in {"failed", "missing"} for s in [primary.status, *statuses])
                    else "completed"}
        try:
            result = phases.execute(f"observer_audit:{cp.checkpoint_id}", {
                "run_id": run_id, "checkpoint_id": cp.checkpoint_id, "extractor": extractor,
                "repeats": settings.full_prefix_repeats, "chain_namespace": chain_namespace,
                "source_scope": scope.as_record() if scope else None}, operation)
        except Exception as exc:
            # The phase journal keeps failed/uncertain attempts. Continue to
            # accounting/reporting; never reissue an uncertain paid audit.
            result = {"status": "failed_or_uncertain", "error_type": type(exc).__name__, "error": str(exc)}
        report["results"].append({"checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal, **result})
    failed = any(r["status"] in {"failed", "failed_or_uncertain"} for r in report["results"])
    report["status"] = "completed_with_failures" if failed else "completed"
    if not report["results"]:
        report["status"] = "unavailable"
    elif not failed and report["unavailable"]:
        report["status"] = "completed_with_unavailable"
    # Unproduced checkpoints remain explicit missing coverage, not an invented
    # failure threshold for naturally shorter trajectories.
    return report


def reuse_completed_campaign(store: ArtifactStore, config: CampaignConfig, *, export_dir: Path) -> dict[str, Any] | None:
    """Re-emit immutable completed outcomes without loading current execution sources."""
    # Reopening completed work under newer code must not change its source
    # binding or rerun potentially charged phases. New analysis/repair gets
    # a new campaign identity referencing the old natural run explicitly.
    try:
        prior_config = from_jsonable(CampaignConfig, store.get_record("campaign", config.campaign_id))
    except KeyError:
        prior_config = None
    if prior_config is not None:
        if prior_config != config:
            raise ValueError("Campaign identity is already bound to different settings")
        try:
            previous = store.get_record("campaign_result", config.campaign_id)
        except KeyError:
            pass
        else:
            try:
                saved_exports = store.get_record("campaign_exports", config.campaign_id)
            except KeyError:
                return {**previous, "reused_completed_campaign": True,
                        "export_note": "Legacy completed campaign; original export paths were not journaled"}
            outputs = {}
            export_dir.mkdir(parents=True, exist_ok=True)
            for key, item in saved_exports.items():
                path = export_dir / item["filename"]
                data = store.get_bytes(from_jsonable(ArtifactRef, item["ref"]))
                if path.exists() and path.read_bytes() != data:
                    raise ValueError(f"Refusing to overwrite changed completed export: {path}")
                path.write_bytes(data)
                outputs[key] = str(path)
            return {**previous, "exports": outputs}
    return None


def needs_fresh_mine(store: ArtifactStore, config: CampaignConfig) -> bool:
    """Whether this request could launch a worker, rather than reuse its phase."""
    if config.source_natural_run_id:
        return False
    for kind, identity in (("campaign_result", config.campaign_id),
                           ("phase_started", f"{config.campaign_id}:mine"),
                           ("phase_finished", f"{config.campaign_id}:mine")):
        try:
            store.get_record(kind, identity)
        except KeyError:
            continue
        return False
    return True


def validate_live_observer(config: CampaignConfig, *, require_live: bool = False) -> None:
    """Validate prospective extraction without changing legacy readback schemas."""
    if config.live_observer is None:
        if require_live:
            raise ValueError("Fresh calibration/main mines require an enabled live observer")
        return
    if config.observer_source_policy != "native-session-primary/v1":
        raise ValueError("Live extraction requires native-session-primary/v1 source policy")
    if type(config.observer_include_returned_reasoning) is not bool:
        raise ValueError("Live extraction requires explicit returned-reasoning source access")
    if not isinstance(config.observer_settings, Mapping) or not config.observer_settings:
        raise ValueError("Live extraction requires explicit nonempty observer settings")
    if config.observer_context_policy is None:
        raise ValueError("Live extraction requires an explicit bounded observer context policy")
    if config.observer_context_policy.max_input_tokens is not None and config.observer_context_counter is None:
        raise ValueError("Live extraction requires the declared request token counter")


def _observer_counter(config: CampaignConfig):
    if config.observer_context_counter is None:
        return None
    from .observers.context import request_counter_from_packet_counter
    counter = request_counter_from_packet_counter(
        PacketSettings(**config.observer_context_counter).build_counter(config.mining.model))
    if (counter.counter_id, counter.revision) != (
            config.observer_context_policy.token_counter_id, config.observer_context_policy.token_counter_revision):
        raise ValueError("Observer request counter differs from the declared context policy")
    return counter


def resume_live_observer(store: ArtifactStore, config: CampaignConfig, client: Any, *,
                         export_dir: Path | None = None) -> dict[str, Any]:
    """Recover only the durable observer queue, preserving original campaign results.

    No task, worker runner, route factory or campaign phase is constructed. The
    live service owns charged-attempt reuse and checkpoint reconciliation.
    """
    validate_live_observer(config, require_live=True)
    if client is None:
        raise ValueError("Live observer resume requires an explicit client")
    original = from_jsonable(CampaignConfig, store.get_record("campaign", config.campaign_id))
    if original != config:
        raise ValueError("Observer resume must use the exact recorded campaign settings")
    from .observers.extractor import ExtractorConfig
    from .observers.live import LiveObserverService
    from .provenance import implementation_binding, same_execution_binding
    run_id = config.source_natural_run_id or config.mining.run_id
    producer = store.get_record("campaign_implementation", config.campaign_id)
    current = implementation_binding(store)
    source_matches = same_execution_binding(current, producer)
    if config.mining.purpose == "main" and not source_matches:
        raise ValueError("Main observer resume implementation differs from the original frozen producer")
    implementation_id = f"{config.campaign_id}:" + hashlib.sha256(canonical_json(current)).hexdigest()
    implementation_ref = store.put_record("campaign_live_observer_resume_implementation_v1", implementation_id,
        current, run_id=run_id)
    service = LiveObserverService(store, run_id, client,
        ExtractorConfig(model=config.mining.model, settings=config.observer_settings,
                        context_policy=config.observer_context_policy),
        namespace=config.campaign_id, include_returned_reasoning=config.observer_include_returned_reasoning,
        policy=config.live_observer, context_counter=_observer_counter(config))
    service.start()
    if not service.ready():
        service.stop()
        raise RuntimeError("Live observer service is not ready")
    coverage = service.finish()
    receipt = {"campaign_id": config.campaign_id, "run_id": run_id, "coverage": coverage,
        "original_campaign_outcomes": "unchanged", "worker_execution": "none",
        "implementation_ref": to_jsonable(implementation_ref),
        "implementation_fingerprint_sha256": current["fingerprint_sha256"],
        "original_producer_fingerprint_sha256": producer["fingerprint_sha256"],
        "matches_original_producer_implementation": source_matches,
        "source_policy": "main_requires_original_frozen_source; calibration/development_resume_records_current_source"}
    identity = f"{config.campaign_id}:" + hashlib.sha256(canonical_json(receipt)).hexdigest()
    store.put_record("campaign_live_observer_resume_v1", identity, receipt, run_id=run_id)
    report = {**coverage, "resume_record_id": identity,
        "implementation_ref": to_jsonable(implementation_ref),
        "implementation_fingerprint_sha256": current["fingerprint_sha256"],
        "original_producer_fingerprint_sha256": producer["fingerprint_sha256"],
        "matches_original_producer_implementation": source_matches}
    if export_dir is not None:
        from .analysis.export import write_json
        filename = "live_observer_resume_" + hashlib.sha256(canonical_json(report)).hexdigest() + ".json"
        write_json(Path(export_dir) / filename, report)
    return report


class Campaign:
    def __init__(self, store: ArtifactStore, task: TaskAdapter, runner: Runner, observer_client: Any, *,
                 on_observer_circuit_open: Callable[..., None] | None = None,
                 before_fresh_mine: Callable[[], None] | None = None):
        self.store, self.task, self.runner, self.observer_client = store, task, runner, observer_client
        self.on_observer_circuit_open = on_observer_circuit_open
        self.before_fresh_mine = before_fresh_mine

    def run(self, config: CampaignConfig, *, export_dir: Path) -> dict[str, Any]:
        from .analysis.continuations import from_run_records
        from .analysis.export import export_bundle, export_trace_tables
        from .experiments.treatments import (BranchSpec, RecoveryTarget, build_all,
                                            select_donor, validate_manifest, resolve_arms)
        from .observers.extractor import ExtractorConfig
        from .observers.pipeline import observe_run, observe_scoped_run, freeze_packets, FrozenPackets
        from .observers.prefix import SourceScope, SESSION_LINEAGE_REVISION

        previous = reuse_completed_campaign(self.store, config, export_dir=export_dir)
        if previous is not None:
            return previous
        from .campaign_launch_contract_v1 import requested as contract_requested, inspect_contract
        launch_contract_enabled = contract_requested(config)
        launch_contract_sources = None
        if launch_contract_enabled:
            validate_live_observer(config, require_live=True)
            from .observers.readiness_v1 import source_binding
            launch_contract_sources = source_binding()
        task_record = self.task.describe()
        from .registry import validate_task_purpose
        from .freeze import validate_main_freeze
        validate_task_purpose(task_record, config.mining.purpose)
        freeze = validate_main_freeze(self.store, config, task_record)
        validate_live_observer(config, require_live=(config.mining.purpose in {"calibration", "main"}
                                                     and needs_fresh_mine(self.store, config)))
        if config.live_observer is not None:
            if self.observer_client is None:
                raise ValueError("Live observer requires an explicit client")
            if needs_fresh_mine(self.store, config):
                import inspect
                parameters = inspect.signature(self.runner.run).parameters
                if "checkpoint_published" not in parameters and not any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                    raise ValueError("Live mining runner must support the checkpoint_published hook")
        if ((config.mode != "mine" or config.live_observer is not None) and config.mining.purpose in {"calibration", "main"}
                and config.observer_source_policy != "native-session-primary/v1"):
            raise ValueError("Pilot/main extraction requires the ratified native-session primary source policy")
        if (config.mode != "mine" or config.live_observer is not None) and config.mining.purpose in {"calibration", "main"}:
            if config.observer_context_policy is None:
                raise ValueError("Pilot/main extraction requires an explicit bounded observer context policy")
            if config.mode == "full" and (config.packet_settings.token_counter is None
                    or not config.packet_settings.matching.configured):
                raise ValueError("Pilot/main recovery requires an explicit packet counter and summary matching tolerance")
        # Resolve packet dependencies only for the phase that uses them. Mining
        # and extraction can precede the choice of a recovery tokenizer/matcher.
        # Full mode still fails before any calls if a declared counter is absent.
        counter = None
        if config.mode == "full":
            counter = config.packet_settings.build_counter(config.mining.model)
            counter.measure("")
        context_counter = None
        if config.mode != "mine" or config.live_observer is not None:
            context_counter = _observer_counter(config)
        primary_scope = (SourceScope(session_scope="current_session",
            include_returned_reasoning=config.observer_include_returned_reasoning,
            lineage_revision=SESSION_LINEAGE_REVISION)
            if config.observer_source_policy == "native-session-primary/v1" and
               config.observer_include_returned_reasoning is not None else None)
        observer_protocol = {"audit": to_jsonable(config.observer_audit),
            "live_observer": to_jsonable(config.live_observer),
            "execution": "live_during_mining" if config.live_observer is not None else "legacy_campaign_phase",
            "source_policy": config.observer_source_policy,
            "include_returned_reasoning": config.observer_include_returned_reasoning,
            "context_policy": config.observer_context_policy.as_record() if config.observer_context_policy else None,
            "context_counter": context_counter.as_record() if context_counter else None,
            "primary_scope": primary_scope.as_record() if primary_scope else None,
            "audit_status": "configured" if config.observer_audit.configured else "not_configured",
            "packet_settings": to_jsonable(config.packet_settings),
            "counter": counter.as_record() if counter is not None else None,
            "counter_selection": ("deferred_until_recovery" if counter is None else
                "declared" if config.packet_settings.token_counter else "unconfigured_legacy_byte_proxy"),
            "matching": config.packet_settings.matching.as_record(), "scientifically_calibrated": False}
        self.store.put_record("campaign_observer_protocol", config.campaign_id, observer_protocol)
        self.store.put_record("campaign", config.campaign_id, config)
        self.store.put_record("campaign_task", config.campaign_id, task_record)
        from .provenance import implementation_binding, same_execution_binding
        frozen_implementation = (self.store.get_json(from_jsonable(ArtifactRef, freeze["implementation_ref"]))
                                 if freeze is not None else None)
        try:
            prior_implementation = self.store.get_record("campaign_implementation", config.campaign_id)
        except KeyError:
            prior_implementation = None
        if freeze is not None and prior_implementation is not None:
            if not same_execution_binding(prior_implementation, frozen_implementation):
                raise ValueError("Incomplete main campaign has a different recorded implementation")
        else:
            implementation = implementation_binding(self.store)
            if freeze is not None and not same_execution_binding(implementation, frozen_implementation):
                raise ValueError("Main implementation changed after freeze validation")
            self.store.put_record("campaign_implementation", config.campaign_id, implementation)
        if freeze is not None:
            self.store.put_record("campaign_freeze", config.campaign_id, freeze)
        phases = PhaseJournal(self.store, config.campaign_id)
        extractor = ExtractorConfig(model=config.mining.model, settings=config.observer_settings,
                                    context_policy=config.observer_context_policy)
        live_service, live_coverage, live_started = None, None, False
        if config.live_observer is not None:
            from .observers.live import LiveObserverService
            live_service = LiveObserverService(self.store, config.source_natural_run_id or config.mining.run_id,
                self.observer_client, extractor, namespace=config.campaign_id,
                include_returned_reasoning=config.observer_include_returned_reasoning,
                policy=config.live_observer, context_counter=context_counter,
                on_circuit_open=self.on_observer_circuit_open)

        def start_live():
            nonlocal live_started
            if live_service is not None and not live_started:
                live_service.start()
                live_started = True
                if not live_service.ready():
                    live_service.stop()
                    raise RuntimeError("Live observer service is not ready")

        def finish_live():
            nonlocal live_coverage
            if live_started:
                try:
                    live_coverage = live_service.finish()
                except Exception as exc:
                    # Observer supervision must not replace a worker failure or
                    # conceal completed work. Durable queue records remain the
                    # authority for observer-only recovery.
                    from .observers.live import live_observer_status
                    live_coverage = {**live_observer_status(self.store,
                        config.source_natural_run_id or config.mining.run_id, namespace=config.campaign_id),
                        "complete": False, "supervision_error": {"type": type(exc).__name__, "error": str(exc)}}
                    identity = f"{config.campaign_id}:" + hashlib.sha256(canonical_json(live_coverage)).hexdigest()
                    self.store.put_record("campaign_live_observer_error_v1", identity, live_coverage,
                        run_id=config.source_natural_run_id or config.mining.run_id)
                    live_service.stop()

        if config.source_natural_run_id:
            source_events = list(self.store.iter_events(config.source_natural_run_id))
            start = next(e.payload for e in source_events if e.kind == "lifecycle" and e.payload.get("transition") == "run_started")
            if start.get("resume_from") or start.get("treatment"):
                raise ValueError("A recovery branch cannot be reused as a natural mine")
            if config.mining.purpose == "main" and start["config"].get("purpose") != "main":
                raise ValueError("Development/calibration mines cannot become main trajectories")
            if freeze is not None:
                source_campaign = start["config"].get("campaign_id")
                try:
                    source_implementation = self.store.get_record("campaign_implementation", source_campaign)
                except KeyError:
                    raise ValueError("Reused main mine lacks its own recorded implementation binding") from None
                if not same_execution_binding(source_implementation, frozen_implementation):
                    raise ValueError("Reused main mine implementation differs from the frozen conditions")
            # Reuse changes storage identity, never the scientific execution
            # binding. Compare all fields, including future schema additions,
            # instead of a permissive hand-maintained list of model settings.
            declared = to_jsonable(config.mining)
            original = start["config"]
            for key in (set(original) | set(declared)) - {"run_id", "campaign_id"}:
                if original.get(key) != declared.get(key):
                    raise ValueError(f"Reused mine differs from the declared worker configuration: {key}")
            if start["task"] != to_jsonable(task_record):
                raise ValueError("Reused mine has a different task binding")
            self.store.put_record("campaign_source_execution", config.campaign_id, start)
            natural = from_jsonable(RunResult, phases.execute("mine", {"source_run_id": config.source_natural_run_id},
                lambda: self.store.get_record("run_result", config.source_natural_run_id)))
        else:
            def mine():
                if self.before_fresh_mine is not None:
                    self.before_fresh_mine()
                start_live()
                kwargs = {"checkpoint_published": live_service.checkpoint_published} if live_service is not None else {}
                return self.runner.run(self.task, config.mining, store=self.store, **kwargs)
            try:
                natural = from_jsonable(RunResult, phases.execute("mine", config.mining, mine))
            finally:
                finish_live()
        if live_service is not None and not live_started:
            # A saved or phase-completed mine only resumes the external queue.
            start_live()
            finish_live()
        chain_namespace = config.campaign_id if config.source_natural_run_id else None
        observed = {"version_ids": [], "statuses": [], "deferred": config.mode == "mine"}
        scoped_observation = None
        if live_coverage is not None:
            observed = {"run_id": natural.run_id, "version_ids": live_coverage["version_ids"],
                "statuses": live_coverage["statuses"], "deferred": False, "execution": "live_during_mining",
                "complete": live_coverage["complete"]}
            chain_namespace = live_coverage["namespace"]
        elif config.mode != "mine":
            if config.observer_source_policy == "native-session-primary/v1":
                scoped_observation = phases.execute("observe", {"run_id": natural.run_id, "config": extractor,
                    "namespace": config.campaign_id, "source_policy": config.observer_source_policy,
                    "include_returned_reasoning": config.observer_include_returned_reasoning},
                    lambda: observe_scoped_run(self.store, natural.run_id, self.observer_client, extractor,
                        namespace=config.campaign_id, include_returned_reasoning=config.observer_include_returned_reasoning,
                        context_counter=context_counter))
                observed = scoped_observation["primary"]
                chain_namespace = scoped_observation["primary_namespace"]
            else:
                observed = phases.execute("observe", {"run_id": natural.run_id, "config": extractor, "chain_namespace": chain_namespace},
                    lambda: observe_run(self.store, natural.run_id, self.observer_client, extractor,
                        chain_namespace=chain_namespace, context_counter=context_counter))
        selected, selection_kind = select_checkpoint(self.store, natural,
                                                     engineering_fallback=config.engineering_fallback)
        fixture_warning = config.fixture and selection_kind == "primary_s1"
        if fixture_warning:
            selection_kind = "engineering_checkpoint"
        selection = {"checkpoint_id": selected.checkpoint_id if selected else None,
                     "selection_kind": selection_kind,
                     "natural_stall_present": not config.fixture and natural.first_stall_id is not None,
                     "detector_warning_present": natural.first_stall_id is not None,
                     "fixture_detector_warning": fixture_warning,
                     "engineering_only": selection_kind == "engineering_checkpoint"}
        self.store.put_record("campaign_selection", config.campaign_id, selection)
        launch_contract = None
        if launch_contract_enabled:
            launch_contract = inspect_contract(self.store, config, natural, extractor=extractor,
                context_counter=context_counter, expected_sources=launch_contract_sources)
            identity = config.campaign_id + ":" + hashlib.sha256(canonical_json(launch_contract)).hexdigest()
            self.store.put_record("campaign_launch_contract_v1", identity, launch_contract,
                                  run_id=natural.run_id)
        observer_audit = _run_observer_audits(self.store, natural.run_id, selected, config,
            extractor, self.observer_client, phases, chain_namespace, primary_scope, context_counter)
        manifests, skipped, branch_results, records = [], [], [], []
        packet_report = {"status": "not_frozen", "settings": observer_protocol["packet_settings"],
                         "counter": observer_protocol["counter"], "matching": observer_protocol["matching"],
                         "reason": "No branchable selected checkpoint in full mode"}
        if config.mode != "full":
            packet_report.update(status="deferred", reason="Recovery deferred by campaign mode")
        recovery_source_ready = launch_contract is None or launch_contract["recovery_source_ready"]
        if config.mode == "full" and not recovery_source_ready:
            packet_report.update(status="blocked_by_live_t", reason="Exact recorded live-T coverage is incomplete")
            skipped.append({"phase": "recovery", "reason": packet_report["reason"],
                            "contract_version": launch_contract["version"]})
        if config.mode == "full" and selected is not None and selected.branchable and recovery_source_ready:
            stage = self.task.stage(selected.scope.stage_id)
            target = RecoveryTarget(target_id=stage.target_id, stage_id=stage.stage_id,
                requirement_version=stage.requirement_version, evaluator_version=stage.evaluator_version,
                prerequisites=stage.prerequisites,
                instructions_ref=self.store.put_bytes(stage.instructions.encode(), media_type="text/plain"))
            packets = None
            try:
                packets = from_jsonable(FrozenPackets, phases.execute("packets", {
                    "run_id": natural.run_id, "checkpoint_id": selected.checkpoint_id, "config": extractor,
                    "chain_namespace": chain_namespace, "packet_settings": config.packet_settings,
                    "counter": counter.as_record(), "source_scope": primary_scope.as_record() if primary_scope else None},
                    lambda: freeze_packets(self.store, natural.run_id, selected.checkpoint_id,
                        client=self.observer_client, config=extractor, chain_namespace=chain_namespace,
                        scope=primary_scope,
                        context_counter=context_counter,
                        token_counter=counter, matching=config.packet_settings.matching,
                        filters=config.packet_settings.filters)))
                packet_report = {"status": "frozen", "result": to_jsonable(packets)}
                if packets.summary_id:
                    summary = self.store.get_record("summary", packets.summary_id)
                    packet_report.update(summary_status=summary["status"],
                                         summary_call_record_ids=summary["call_record_ids"])
            except Exception as exc:
                packets = None
                packet_report.update(status="failed_or_uncertain", error_type=type(exc).__name__, reason=str(exc))
                skipped.append({"phase": "packets", "reason": str(exc), "error_type": type(exc).__name__})
            from .experiments.observation_mask import build_observation_mask
            mask_spec = build_observation_mask(self.store, selected, selection_kind=selection_kind,
                                                policy=config.observation_mask_policy)
            spec = BranchSpec(run_id=natural.run_id, s1=selected,
                donor=select_donor(self.store, natural.run_id, selected), target=target,
                worker_model=config.mining.model, model_settings=config.mining.model_settings,
                generated_token_limit=config.recovery_generated_tokens, selection_kind=selection_kind,
                recovery_settings=worker_operational_settings(config.mining),
                runtime_policy="irrelevant" if selected.state.runtime_class.value == "R_IRRELEVANT" else "restore_captured",
                packets=packets, mask_spec=mask_spec)
            arms = resolve_arms(config.arms)
            manifests, arm_skips = build_all(spec, config.repeats, arms=arms)
            skipped.extend(arm_skips)
            if config.schedule_seed is not None:
                import random
                random.Random(config.schedule_seed).shuffle(manifests)
            self.store.put_record("campaign_schedule", config.campaign_id, {
                "selection": selection, "treatments": manifests, "skipped": skipped,
                "horizons": config.horizons, "order": "seeded_random" if config.schedule_seed is not None else "serial_manifest_order_development",
                "schedule_seed": config.schedule_seed,
                "calibration_status": config.calibration_status,
            })
            for treatment in manifests:
                errors = validate_manifest(treatment, self.store, natural.run_id)
                if errors:
                    skipped.append({"treatment_id": treatment.treatment_id, "reason": errors})
                    continue
                branch_id = f"{config.campaign_id}:branch:{treatment.treatment_id}"
                run_config = replace(config.mining, run_id=branch_id, stage_id=treatment.stage_id,
                    generated_token_limit=config.recovery_generated_tokens, repeat_id=treatment.repeat_id,
                    metadata={**config.mining.metadata, "treatment_id": treatment.treatment_id,
                              "selection_kind": selection_kind, "execution_role": "recovery"})
                self.store.put_record("treatment", treatment.treatment_id, treatment, run_id=branch_id)
                result = from_jsonable(RunResult, phases.execute(f"branch:{treatment.treatment_id}",
                    {"config": run_config, "treatment": treatment},
                    lambda t=treatment, c=run_config: self.runner.run(self.task, c, store=self.store,
                                                                    resume=selected, treatment=t)))
                branch_results.append(result)
                continuation = from_run_records(self.store, treatment, result.run_id,
                    stop_reason=analysis_stop_reason(self.store, result),
                    technical_failure=result.status in {"error", "failed", "invalid", "technical_failure"})
                records.append(replace(continuation, lineage={**continuation.lineage,
                    "raw_termination_reason": result.termination_reason, "run_status": result.status}))
        elif config.mode == "full" and selected is not None and not selected.branchable:
            skipped.append({"phase": "branch", "reason": "First selected checkpoint is not branchable"})
        # Observer repair/audit campaigns can share a saved mine. A completed
        # campaign keeps its original accounting cut instead of claiming later
        # observer charges on a restart or mutating its immutable result.
        try:
            accounting = self.store.get_record("campaign_result", config.campaign_id)["accounting_refs"]
        except KeyError:
            accounting = {run_id: to_jsonable(CostLedger(self.store).snapshot(run_id))
                          for run_id in [natural.run_id, *(r.run_id for r in branch_results)]}
        report = {"campaign_id": config.campaign_id, "fixture": config.fixture,
                  "mode": config.mode, "recovery_deferred": config.mode != "full",
                  "recovery_plan": config.recovery_plan,
                  "purpose": config.mining.purpose, "calibration_status": config.calibration_status,
                  "source_natural_run_id": config.source_natural_run_id, "T_chain_namespace": chain_namespace,
                  "task_admission": task_record.admission.value, "selection": selection,
                  "natural": to_jsonable(natural), "observer": observed,
                  "live_observer": live_coverage,
                  "observer_scoped": scoped_observation,
                  "observer_protocol": observer_protocol, "observer_audit": observer_audit,
                  "packets": packet_report,
                  "branch_results": to_jsonable(branch_results), "skipped": skipped,
                  "accounting_refs": accounting,
                  "limitations": ["One task is an instrument check, not a research effect estimate.",
                                   "Development settings are not a calibrated main-study protocol."]}
        if launch_contract is not None:
            report["launch_contract"] = launch_contract
        failures = natural.status != "completed" or any(r.status != "completed" for r in branch_results)
        failures = failures or (launch_contract is not None and not launch_contract["recovery_source_ready"])
        failures = failures or (live_coverage is not None and live_coverage["complete"] is not True)
        failures = failures or any(s in {"failed", "missing"} for s in observed.get("statuses", []))
        failures = failures or bool(scoped_observation and any(s in {"failed", "missing"}
            for s in scoped_observation["archive"]["statuses"]))
        failures = failures or observer_audit["status"] == "completed_with_failures"
        failures = failures or packet_report.get("summary_status") in {"failed", "unmatched"}
        report["pipeline_status"] = "partial_failure" if failures or skipped else "outcomes_recorded"
        from .analysis.effects import AnalysisSpec
        from .analysis.natural import natural_followup_report
        from .analysis.costs import policy_cost_report
        from .analysis.export import write_json, write_csv
        from .observers.audit import audit_recorded_chain
        analysis_spec = AnalysisSpec(horizons=config.horizons, **config.analysis_settings)
        outputs = export_bundle(export_dir, manifests=manifests, records=records,
                                horizons=config.horizons, analysis=report, analysis_spec=analysis_spec,
                                family_by_source={m.source_checkpoint_id: task_record.family_id for m in manifests})
        natural_analysis = natural_followup_report(self.store, [natural.run_id], horizons=config.horizons,
            near_budget_tokens=config.near_budget_tokens, resource_horizons=config.resource_horizons)
        outputs["natural_followup"] = str(write_json(export_dir / "natural_followup.json", natural_analysis))
        operation_keys = []
        for run_id, ref in accounting.items():
            snapshot = self.store.get_json(from_jsonable(ArtifactRef, ref))
            operation_keys.extend((run_id, operation_id) for operation_id in snapshot["operation_ids"])
        costs = policy_cost_report(self.store, manifests=manifests, records=records,
            actual_operation_keys=operation_keys, dollar_horizons=config.dollar_horizons)
        outputs["policy_costs"] = str(write_json(export_dir / "policy_costs.json", costs))
        import json
        outputs["cost_quality_points"] = str(write_csv(export_dir / "cost_quality_points.csv", [
            {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in costs["cost_quality_points"]]))
        outputs["observer_source_audit"] = str(write_json(export_dir / "observer_source_audit.json",
            audit_recorded_chain(self.store, observed["version_ids"])))
        if live_coverage is not None:
            outputs["live_observer_coverage"] = str(write_json(export_dir / "live_observer_coverage.json", live_coverage))
            pairs = [self.store.get_record("observer_scope_pair_v1", item["record_id"])
                     for item in self.store.list_records("observer_scope_pair_v1", run_id=natural.run_id)]
            outputs["observer_scope_pairs"] = str(write_json(export_dir / "observer_scope_pairs.json", {
                "execution": "live_during_mining", "coverage": live_coverage,
                "pairs": [pair for pair in pairs if pair["namespace"] == config.campaign_id],
                "archive_access": "secondary reference; never primary extractor input"}))
        if scoped_observation is not None:
            outputs["observer_scope_pairs"] = str(write_json(export_dir / "observer_scope_pairs.json", {
                **scoped_observation, "pairs": [self.store.get_record("observer_scope_pair_v1", key)
                    for key in scoped_observation["pair_ids"]],
                "archive_source_audit": audit_recorded_chain(self.store, scoped_observation["archive"]["version_ids"])}))
        outputs.update(export_trace_tables(export_dir, store=self.store,
            run_ids=[natural.run_id, *(r.run_id for r in branch_results)], version_ids=observed["version_ids"]))
        from .status import write_status
        outputs["status"] = str(write_status(self.store, export_dir / "status.html"))
        self.store.put_record("campaign_exports", config.campaign_id, {
            key: {"filename": Path(path).name, "ref": self.store.put_bytes(Path(path).read_bytes())}
            for key, path in outputs.items()})
        self.store.put_record("campaign_result", config.campaign_id, report)
        return {**report, "exports": outputs}
