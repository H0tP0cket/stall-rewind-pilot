"""Offline mining-first plans; unresolved scientific choices remain null.

Schema completeness is not admission, route qualification or permission to
spend. The existing launch preflight and adapter admission checks still apply.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

from .campaign import CampaignConfig, ObserverAuditConfig, PacketSettings
from .contracts import ArtifactStoreProtocol, RunConfig, WorkerRequestPolicy, canonical_json, from_jsonable, to_jsonable
from .detector.core import DetectorConfig
from .factory import BatchPlan, CampaignJob

VERSION = "mining-launch-plan.v1"
LIVE_VERSION = "mining-launch-plan.v2"
DEFERRED = ("recovery_generated_tokens", "recovery_horizons", "recovery_repeat_ids", "selected_arms",
            "observer_settings", "observer_audit", "packet_tokenizer_and_matching")
LIVE_DEFERRED = tuple(name for name in DEFERRED if name not in {"observer_settings", "observer_audit"})
LIVE_WORKER_FIELDS = {"observer_settings", "observer_context_policy", "observer_context_counter"}


def load_mining_launch_plan(path: Path) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate launch-plan JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite launch-plan JSON constant: " + value)
    value = json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Launch plan must be a JSON object")
    return value


def _mining_launch_template_v1(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Copy declared membership/models, supplying no scientific numeric values."""
    cohort = json.loads(canonical_json(cohort))
    task_ids = [t["task_id"] for t in cohort.get("primary_tasks", [])]
    return {"version": VERSION, "plan_id": None, "mode": "mine", "purpose": None, "fixture": None,
        "cohort": cohort, "admission": {task_id: None for task_id in task_ids},
        "workers": [{"model": model, "provider": None, "reasoning": None, "model_settings": None,
            "reply_max_tokens": None, "max_model_calls": None, "max_consecutive_format_errors": None,
            "tool_timeout_seconds": None,
            "request_policy": {f.name: None for f in fields(WorkerRequestPolicy)},
            "task_allowances": {task_id: None for task_id in task_ids}} for model in cohort.get("model_ids", [])],
        "detector_settings": {f.name: None for f in fields(DetectorConfig)},
        "mining_repeat_ids": None, "mining_horizons": {task_id: None for task_id in task_ids},
        "near_budget_tokens": {task_id: None for task_id in task_ids},
        "order": {"policy": None, "seed": None}, "max_parallel": None,
        "recovery": None, "observer": None, "packet_settings": None}


def mining_launch_template(cohort: Mapping[str, Any], *, version: str = LIVE_VERSION) -> dict[str, Any]:
    """Offer live mining v2 with pending choices; v1 remains explicitly available."""
    from .observers.context import ContextPolicy
    from .observers.live import LiveObserverPolicy, REVISION
    if version not in {VERSION, LIVE_VERSION}:
        raise ValueError("Unsupported mining launch plan version: " + str(version))
    plan = _mining_launch_template_v1(cohort)
    if version == VERSION:
        return plan
    plan["version"] = LIVE_VERSION
    plan["detector_settings"] = {task_id: {f.name: None for f in fields(DetectorConfig)}
                                 for task_id in plan["admission"]}
    plan["observer"] = {"version": REVISION, "source_policy": None, "include_returned_reasoning": None,
        "live_policy": {f.name: None for f in fields(LiveObserverPolicy)},
        "audit": {f.name: None for f in fields(ObserverAuditConfig)}}
    for worker in plan["workers"]:
        worker.update(observer_settings=None,
            observer_context_policy={f.name: None for f in fields(ContextPolicy)}, observer_context_counter=None)
    return plan


def _validate_mining_launch_plan_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Collect actionable missing/invalid fields without constructing paid clients."""
    pending, errors = [], []
    def check(value: Any, path: str, predicate, message: str) -> bool:
        if value is None:
            pending.append({"field": path, "reason": "Explicit choice required: " + message})
            return False
        if not predicate(value):
            errors.append({"field": path, "reason": message})
            return False
        return True
    def integer(value: Any, minimum: int = 1) -> bool:
        return type(value) is int and value >= minimum
    def label(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())
    def mapping(value: Any) -> bool:
        return isinstance(value, Mapping)
    def keys(value: Any, expected: set[str], path: str) -> bool:
        return check(value, path, lambda v: mapping(v) and set(v) == expected, "Expected fields: " + ", ".join(sorted(expected)))
    if not isinstance(plan, Mapping):
        return {"version": VERSION, "plan_id": None, "launchable": False, "pending": [],
                "errors": [{"field": "plan", "reason": "Plan must be an object"}], "deferred": list(DEFERRED)}
    expected = {"version", "plan_id", "mode", "purpose", "fixture", "cohort", "admission", "workers",
                "detector_settings", "mining_repeat_ids", "mining_horizons", "near_budget_tokens", "order",
                "max_parallel", "recovery", "observer", "packet_settings"}
    keys(plan, expected, "plan")
    check(plan.get("version"), "version", lambda v: v == VERSION, "Supported mining launch plan version required")
    check(plan.get("plan_id"), "plan_id", label, "Nonempty new plan identity")
    check(plan.get("mode"), "mode", lambda v: v == "mine", "This schema executes mine-only campaigns")
    check(plan.get("purpose"), "purpose", lambda v: v in ("development", "calibration"), "development or calibration; main requires a separate freeze")
    check(plan.get("fixture"), "fixture", lambda v: type(v) is bool, "Explicit fixture boolean; fixture values are never calibrated defaults")
    check(plan.get("max_parallel"), "max_parallel", integer, "Positive integer concurrency")
    for deferred in ("recovery", "observer", "packet_settings"):
        if plan.get(deferred) is not None:
            errors.append({"field": deferred, "reason": "Mine-only execution must leave this explicitly deferred as null"})
    cohort = plan.get("cohort")
    task_ids, models = [], []
    if check(cohort, "cohort", mapping, "Declared cohort object with source identities"):
        check(cohort.get("version"), "cohort.version", lambda v: v == "calibration-cohort.v1", "Supported declared cohort version")
        check(cohort.get("cohort_id"), "cohort.cohort_id", label, "Declared cohort identity")
        models = cohort.get("model_ids") or []
        if not check(models, "cohort.model_ids", lambda v: isinstance(v, list) and len(v) == 2 and all(label(x) for x in v) and len(set(v)) == 2,
                     "Exactly both distinct declared worker model IDs"):
            models = []
        tasks = cohort.get("primary_tasks")
        if check(tasks, "cohort.primary_tasks", lambda v: isinstance(v, list) and bool(v) and all(mapping(x) for x in v), "Nonempty declared primary cohort"):
            for i, task in enumerate(tasks):
                for name in ("task_id", "family_id", "adapter", "environment_version"):
                    check(task.get(name), f"cohort.primary_tasks[{i}].{name}", label, "Explicit source identity")
                if label(task.get("task_id")):
                    task_ids.append(task["task_id"])
                for name in ("task_record", "package_manifest", "attestation"):
                    ref = task.get(name)
                    if keys(ref, {"path", "sha256"}, f"cohort.primary_tasks[{i}].{name}"):
                        check(ref["path"], f"cohort.primary_tasks[{i}].{name}.path",
                              lambda v: label(v) and not Path(v).is_absolute() and ".." not in Path(v).parts, "Recorded relative source path within artifact root")
                        check(ref["sha256"], f"cohort.primary_tasks[{i}].{name}.sha256",
                              lambda v: isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v), "Exact source SHA-256")
            if len(set(task_ids)) != len(task_ids):
                errors.append({"field": "cohort.primary_tasks", "reason": "Duplicate primary task identity"})
    admission = plan.get("admission")
    if keys(admission, set(task_ids), "admission"):
        for task_id, status in admission.items():
            check(status, "admission." + task_id, lambda v: v == "ADMITTED", "ADMITTED required; HOLD/pending task is nonlaunchable and cannot be silently replaced")
    repeats = plan.get("mining_repeat_ids")
    check(repeats, "mining_repeat_ids", lambda v: isinstance(v, list) and bool(v) and all(label(x) for x in v) and len(set(v)) == len(v), "Explicit unique natural-trajectory repeat IDs")
    order = plan.get("order")
    if keys(order, {"policy", "seed"}, "order"):
        check(order["policy"], "order.policy", lambda v: v in ("declared", "seeded_shuffle"), "declared or seeded_shuffle")
        check(order["seed"], "order.seed", lambda v: integer(v, 0), "Explicit nonnegative integer schedule seed")
    detector = plan.get("detector_settings")
    if keys(detector, {f.name for f in fields(DetectorConfig)}, "detector_settings"):
        for name, value in detector.items():
            check(value, "detector_settings." + name, lambda v: True, "Explicit detector rule/settings; no implicit thresholds")
        if all(v is not None for v in detector.values()):
            try:
                DetectorConfig(**detector)
                for name in ("stagnation_token_threshold", "persistent_failure_grades", "weak_region_grades"):
                    if type(detector[name]) is not int:
                        raise ValueError(name + " must be an integer")
            except (TypeError, ValueError) as exc:
                errors.append({"field": "detector_settings", "reason": str(exc)})
    horizons, near = plan.get("mining_horizons"), plan.get("near_budget_tokens")
    for name, value in (("mining_horizons", horizons), ("near_budget_tokens", near)):
        if keys(value, set(task_ids), name):
            for task_id, setting in value.items():
                check(setting, name + "." + task_id,
                    (lambda v: isinstance(v, list) and bool(v) and all(integer(x) for x in v) and v == sorted(set(v)))
                    if name == "mining_horizons" else (lambda v: integer(v, 0)), "Explicit increasing positive horizons" if name == "mining_horizons" else "Explicit nonnegative near-budget threshold")
    workers = plan.get("workers")
    if check(workers, "workers", lambda v: isinstance(v, list) and len(v) == 2 and all(mapping(w) for w in v), "Both worker configurations required"):
        if sorted(str(w.get("model")) for w in workers) != sorted(models):
            errors.append({"field": "workers", "reason": "Worker models must match both declared cohort models exactly"})
        for i, worker in enumerate(workers):
            prefix = f"workers[{i}]"
            if isinstance(worker.get("model"), str) and worker["model"].startswith(("scripted/", "fixture/", "dev/")) and plan.get("fixture") is not True:
                errors.append({"field": prefix + ".model", "reason": "Scripted model requires explicit fixture=True"})
            keys(worker, {"model", "provider", "reasoning", "model_settings", "reply_max_tokens", "max_model_calls",
                          "max_consecutive_format_errors", "tool_timeout_seconds", "request_policy", "task_allowances"}, prefix)
            for name in ("provider", "reasoning"):
                check(worker.get(name), prefix + "." + name, lambda v: mapping(v) and bool(v), "Explicit route/reasoning object (including disabled choices)")
            check(worker.get("model_settings"), prefix + ".model_settings", lambda v: mapping(v) and not {"provider", "reasoning", "max_tokens", "max_completion_tokens"}.intersection(v), "Explicit additional settings; provider/reasoning/reply cap use their named fields")
            for name in ("reply_max_tokens", "max_model_calls", "max_consecutive_format_errors"):
                check(worker.get(name), prefix + "." + name, lambda v, n=name: integer(v, 0 if n == "max_consecutive_format_errors" else 1), "Explicit integer operational guard")
            check(worker.get("tool_timeout_seconds"), prefix + ".tool_timeout_seconds", lambda v: type(v) in (int, float) and math.isfinite(v) and v > 0, "Positive finite tool timeout")
            request = worker.get("request_policy")
            if keys(request, {f.name for f in fields(WorkerRequestPolicy)}, prefix + ".request_policy"):
                for name, value in request.items():
                    check(value, prefix + ".request_policy." + name, lambda v: True, "Explicit retry/reply timeout policy")
                if all(v is not None for v in request.values()):
                    try:
                        WorkerRequestPolicy(**request)
                    except (TypeError, ValueError) as exc:
                        errors.append({"field": prefix + ".request_policy", "reason": str(exc)})
            allowances = worker.get("task_allowances")
            if keys(allowances, set(task_ids), prefix + ".task_allowances"):
                for task_id, allowance in allowances.items():
                    if check(allowance, prefix + ".task_allowances." + task_id, integer, "Positive whole-task generated-token allowance"):
                        hs = horizons.get(task_id) if mapping(horizons) else None
                        if isinstance(hs, list) and all(integer(h) for h in hs) and any(h > allowance for h in hs):
                            errors.append({"field": "mining_horizons." + task_id, "reason": "Horizon exceeds a declared worker allowance"})
    try:
        canonical_json(plan)
    except (TypeError, ValueError) as exc:
        errors.append({"field": "plan", "reason": "Finite JSON values required: " + str(exc)})
    return {"version": VERSION, "plan_id": plan.get("plan_id"), "launchable": not pending and not errors,
        "pending": pending, "errors": errors, "deferred": list(DEFERRED), "mode": "mine",
        "qualification": "Offline schema completeness only; admission/source/image/route preflight and launch authorization remain separate.",
        "fixture": plan.get("fixture")}


def validate_mining_launch_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate v1 unchanged, or v2's explicit live-mining construction path."""
    if not isinstance(plan, Mapping) or plan.get("version") != LIVE_VERSION:
        return _validate_mining_launch_plan_v1(plan)
    from .campaign import validate_live_observer
    from .observers.context import ContextPolicy
    from .observers.live import LiveObserverPolicy, REVISION

    # Reuse the exact historical cohort/worker/allowance checks, excluding the
    # three fields whose shapes v2 explicitly replaces. This is validation only;
    # no projected value is ever used to construct a campaign.
    probe = dict(plan, version=VERSION, observer=None, detector_settings=None)
    if isinstance(plan.get("workers"), list):
        probe["workers"] = [{k: v for k, v in worker.items() if k not in LIVE_WORKER_FIELDS}
                            if isinstance(worker, Mapping) else worker for worker in plan["workers"]]
    report = _validate_mining_launch_plan_v1(probe)
    pending = [row for row in report["pending"] if not row["field"].startswith("detector_settings")]
    errors = [row for row in report["errors"] if not row["field"].startswith("detector_settings")]

    def check(value, path, predicate, message):
        if value is None:
            pending.append({"field": path, "reason": "Explicit choice required: " + message})
            return False
        if not predicate(value):
            errors.append({"field": path, "reason": message})
            return False
        return True

    def keys(value, expected, path):
        if not check(value, path, lambda v: isinstance(v, Mapping), "Explicit object required"):
            return False
        missing, extra = set(expected) - set(value), set(value) - set(expected)
        pending.extend({"field": path + "." + name, "reason": "Explicit field required"} for name in sorted(missing))
        if extra:
            errors.append({"field": path, "reason": "Unknown fields: " + ", ".join(sorted(extra))})
        return not missing and not extra

    def construct(cls, value, path):
        try:
            return from_jsonable(cls, value)
        except (TypeError, ValueError) as exc:
            errors.append({"field": path, "reason": str(exc)})
            return None

    def required_record(cls, value, path):
        if keys(value, {f.name for f in fields(cls)}, path):
            for name, setting in value.items():
                check(setting, path + "." + name, lambda v: True, "Explicit setting; no implicit defaults")
            if all(setting is not None for setting in value.values()):
                return construct(cls, value, path)
        return None

    tasks = (plan.get("cohort") or {}).get("primary_tasks", []) if isinstance(plan.get("cohort"), Mapping) else []
    task_ids = [t["task_id"] for t in tasks if isinstance(t, Mapping) and isinstance(t.get("task_id"), str)] if isinstance(tasks, list) else []
    detectors = plan.get("detector_settings")
    if keys(detectors, task_ids, "detector_settings"):
        for task_id, settings in detectors.items():
            required_record(DetectorConfig, settings, "detector_settings." + task_id)
    observer = plan.get("observer")
    if keys(observer, {"version", "source_policy", "include_returned_reasoning", "live_policy", "audit"}, "observer"):
        check(observer["version"], "observer.version", lambda v: v == REVISION, "Supported live-checkpoint observer schema version")
        check(observer["source_policy"], "observer.source_policy", lambda v: v == "native-session-primary/v1",
              "Live extraction requires native-session-primary/v1")
        check(observer["include_returned_reasoning"], "observer.include_returned_reasoning", lambda v: type(v) is bool,
              "Explicit returned-reasoning source access")
        required_record(LiveObserverPolicy, observer["live_policy"], "observer.live_policy")
        required_record(ObserverAuditConfig, observer["audit"], "observer.audit")
    workers = plan.get("workers")
    if isinstance(workers, list):
        for i, worker in enumerate(workers):
            if not isinstance(worker, Mapping):
                continue
            prefix = f"workers[{i}]"
            settings = worker.get("observer_settings")
            if check(settings, prefix + ".observer_settings", lambda v: isinstance(v, Mapping), "Explicit observer request settings"):
                for name in ("provider", "reasoning"):
                    check(settings.get(name), prefix + ".observer_settings." + name,
                          lambda v: isinstance(v, Mapping) and bool(v), "Explicit observer route/reasoning object")
                caps = set(settings) & {"max_tokens", "max_completion_tokens"}
                if not caps:
                    pending.append({"field": prefix + ".observer_settings.reply_cap", "reason": "One explicit positive reply token cap required"})
                elif len(caps) != 1:
                    errors.append({"field": prefix + ".observer_settings", "reason": "Specify exactly one reply token cap"})
                else:
                    name = next(iter(caps))
                    check(settings[name], prefix + ".observer_settings." + name,
                          lambda v: type(v) is int and v > 0, "Positive integer reply token cap")
                if set(settings) & {"model", "messages", "tools", "stream"}:
                    errors.append({"field": prefix + ".observer_settings", "reason": "Observer settings cannot override model/messages/tools/stream"})
            context_path = prefix + ".observer_context_policy"
            context = worker.get("observer_context_policy")
            if not keys(context, {f.name for f in fields(ContextPolicy)}, context_path):
                continue
            ready = check(context["allow_proxy"], context_path + ".allow_proxy", lambda v: type(v) is bool,
                          "Explicit acceptance or rejection of approximate counters")
            if context["max_request_bytes"] is None and context["max_input_tokens"] is None:
                pending.append({"field": context_path, "reason": "At least one explicit request byte or input-token limit required"})
                ready = False
            for name in ("max_request_bytes", "max_input_tokens"):
                if context[name] is not None:
                    ready = check(context[name], context_path + "." + name, lambda v: type(v) is int and v > 0,
                                  "Positive integer context limit") and ready
            if context["max_input_tokens"] is not None:
                for name in ("token_counter_id", "token_counter_revision"):
                    ready = check(context[name], context_path + "." + name,
                                  lambda v: isinstance(v, str) and bool(v.strip()), "Explicit request-counter identity/version") and ready
            elif context["token_counter_id"] is not None or context["token_counter_revision"] is not None:
                errors.append({"field": context_path, "reason": "Counter identity applies only to an input-token limit"})
                ready = False
            policy = construct(ContextPolicy, context, context_path) if ready else None
            counter_path = prefix + ".observer_context_counter"
            counter = worker.get("observer_context_counter")
            if "observer_context_counter" not in worker:
                pending.append({"field": counter_path, "reason": "Declare a request counter or explicit null for byte-only limits"})
            if policy is not None and policy.max_input_tokens is not None:
                if keys(counter, {"token_counter", "encoding_name"}, counter_path):
                    if check(counter["token_counter"], counter_path + ".token_counter",
                             lambda v: v in ("byte_proxy", "tiktoken"), "Explicit local request counter"):
                        construct(PacketSettings, counter, counter_path)
                        if counter["token_counter"] == "byte_proxy" and not policy.allow_proxy:
                            errors.append({"field": context_path + ".allow_proxy", "reason": "Byte proxy requires explicit acceptance"})
            elif policy is not None and counter is not None:
                errors.append({"field": counter_path, "reason": "Request counter applies only to an input-token limit"})
    try:
        canonical_json(plan)
    except (TypeError, ValueError) as exc:
        errors.append({"field": "plan", "reason": "Finite JSON values required: " + str(exc)})
    if not pending and not errors:
        try:
            for job in _mining_batch(json.loads(canonical_json(plan))).jobs:
                validate_live_observer(job.config, require_live=True)
        except (TypeError, ValueError, KeyError) as exc:
            errors.append({"field": "campaign_config", "reason": str(exc)})
    return {**report, "version": LIVE_VERSION, "launchable": not pending and not errors,
        "pending": pending, "errors": errors, "deferred": list(LIVE_DEFERRED),
        "qualification": "Offline schema and CampaignConfig/live-observer construction only; admission/source/image/route qualification, actual counter resolution and launch authorization remain separate."}


def build_mining_batch(plan: Mapping[str, Any]) -> BatchPlan:
    """Generate existing CampaignConfig/BatchPlan records without executing work."""
    report = validate_mining_launch_plan(plan)
    if not report["launchable"]:
        raise ValueError("Mining plan is nonlaunchable: " + json.dumps({"pending": report["pending"], "errors": report["errors"]}, sort_keys=True))
    return _mining_batch(json.loads(canonical_json(plan)))


def _mining_batch(plan: Mapping[str, Any]) -> BatchPlan:
    """Construct the validated version without upgrading historical choices."""
    from .observers.context import ContextPolicy
    from .observers.live import LiveObserverPolicy
    live = plan["version"] == LIVE_VERSION
    digest = hashlib.sha256(canonical_json(plan)).hexdigest()
    jobs = []
    for task in plan["cohort"]["primary_tasks"]:
        for worker in plan["workers"]:
            for repeat in plan["mining_repeat_ids"]:
                suffix = hashlib.sha256(canonical_json([task["task_id"], worker["model"], repeat])).hexdigest()[:24]
                identity = plan["plan_id"] + ":" + suffix
                mining = RunConfig(run_id=identity + ":natural", campaign_id=identity, purpose=plan["purpose"],
                    model=worker["model"], model_settings={**worker["model_settings"], "provider": worker["provider"],
                        "reasoning": worker["reasoning"], "max_tokens": worker["reply_max_tokens"]},
                    generated_token_limit=worker["task_allowances"][task["task_id"]],
                    max_model_calls=worker["max_model_calls"], tool_timeout_seconds=worker["tool_timeout_seconds"],
                    max_consecutive_format_errors=worker["max_consecutive_format_errors"],
                    worker_request_policy=from_jsonable(WorkerRequestPolicy, worker["request_policy"]),
                    detector_settings=plan["detector_settings"][task["task_id"]] if live else plan["detector_settings"], repeat_id=repeat,
                    metadata={"launch_plan_version": plan["version"], "launch_plan_id": plan["plan_id"], "launch_plan_sha256": digest,
                        "cohort_id": plan["cohort"]["cohort_id"], "source_binding": task,
                        "mining_horizons": plan["mining_horizons"][task["task_id"]]})
                observer_fields = {} if not live else {
                    "live_observer": from_jsonable(LiveObserverPolicy, plan["observer"]["live_policy"]),
                    "observer_source_policy": plan["observer"]["source_policy"],
                    "observer_include_returned_reasoning": plan["observer"]["include_returned_reasoning"],
                    "observer_settings": worker["observer_settings"],
                    "observer_context_policy": from_jsonable(ContextPolicy, worker["observer_context_policy"]),
                    "observer_context_counter": worker["observer_context_counter"],
                    "observer_audit": from_jsonable(ObserverAuditConfig, plan["observer"]["audit"])}
                jobs.append(CampaignJob(task_id=task["task_id"], config=CampaignConfig(campaign_id=identity,
                    mining=mining, mode="mine", fixture=plan["fixture"], recovery_generated_tokens=None,
                    horizons=(), repeats=(), arms=(), schedule_seed=plan["order"]["seed"],
                    near_budget_tokens=plan["near_budget_tokens"][task["task_id"]],
                    resource_horizons={"generated_tokens": tuple(plan["mining_horizons"][task["task_id"]])}, **observer_fields)))
    if plan["order"]["policy"] == "seeded_shuffle":
        random.Random(plan["order"]["seed"]).shuffle(jobs)
    return BatchPlan(batch_id=plan["plan_id"], jobs=tuple(jobs), max_parallel=plan["max_parallel"])


RECOVERY_VERSION = "saved-mine-recovery-launch-plan.v1"


def load_recovery_launch_plan(path: Path) -> dict[str, Any]:
    """Apply the same duplicate-key/nonfinite JSON checks as mining plans."""
    return load_mining_launch_plan(path)


def recovery_launch_template(cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Declare saved-source cells without selecting any recovery parameters.

    A caller supplies the versioned, qualified successor cohort declaration;
    this function copies its full membership, including reserve annotations.
    It neither requalifies sources nor relabels an old seal as current.
    """
    from .analysis.effects import AnalysisSpec
    from .observers.context import ContextPolicy
    from .observers.tokens import TokenMatchConfig
    cohort = json.loads(canonical_json(cohort))
    task_ids = [task["task_id"] for task in cohort.get("primary_tasks", [])]
    models = cohort.get("model_ids", [])
    return {"version": RECOVERY_VERSION, "plan_id": None, "mode": "full", "purpose": None, "fixture": None,
        "cohort": cohort, "admission": {task: None for task in task_ids},
        "saved_mines": {task: {model: None for model in models} for task in task_ids},
        "recovery": {"horizons": {task: None for task in task_ids}, "repeats": None, "arms": None,
                     "observation_mask_policy": None},
        "observer": {"source_policy": None, "include_returned_reasoning": None,
                     "audit": {f.name: None for f in fields(ObserverAuditConfig)}},
        "workers": [{"model": model, "recovery_task_allowances": {task: None for task in task_ids},
            "observer_settings": None, "observer_context_policy": {f.name: None for f in fields(ContextPolicy)},
            "observer_context_counter": None,
            "packet_settings": {"token_counter": None, "encoding_name": None,
                "matching": {f.name: None for f in fields(TokenMatchConfig)}, "filters": None}} for model in models],
        "analysis_settings": {f.name: None for f in fields(AnalysisSpec) if f.name != "horizons"},
        "near_budget_tokens": {task: None for task in task_ids}, "resource_horizons": None,
        "dollar_horizons": None, "order": {"policy": None, "seed": None}, "max_parallel": None}


def saved_mine_source(store: ArtifactStoreProtocol, run_id: str) -> dict[str, Any]:
    """Read a retained natural execution; keep failed/no-warning runs eligible as sources.

    Eligibility here means an identified retained natural run, not proof of a
    branchable S or a successful experiment. Campaign preserves those outcomes.
    """
    starts = [event.payload for event in store.iter_events(run_id)
              if event.kind == "lifecycle" and event.payload.get("transition") == "run_started"]
    if len(starts) != 1:
        raise ValueError("Saved natural run requires exactly one retained run_started record")
    start = starts[0]
    if start.get("resume_from") or start.get("treatment"):
        raise ValueError("Recovery/resumed executions cannot be supplied as natural mines")
    config = start.get("config")
    if not isinstance(config, Mapping) or config.get("run_id") != run_id:
        raise ValueError("Saved run_started does not bind the requested natural run identity")
    result = store.get_record("run_result", run_id)
    if result.get("run_id") != run_id:
        raise ValueError("Saved natural result does not bind its requested run identity")
    # Copy, never return aliases into retained records for plan editing.
    config = json.loads(canonical_json(config))
    return {"run_id": run_id, "original_mine_config": config,
            "original_mine_config_sha256": hashlib.sha256(canonical_json(config)).hexdigest()}


def validate_recovery_launch_plan(plan: Mapping[str, Any], *, store: ArtifactStoreProtocol | None = None) -> dict[str, Any]:
    """Validate explicit choices and optionally bind them to read-only saved runs.

    No route/client/tokenizer construction, registry mutation or launch occurs.
    Schema readiness and saved-source verification are separate from current
    adapter/source/image/route qualification and authorization to spend.
    """
    from .analysis.effects import AnalysisSpec
    from .experiments.observation_mask import PRIMARY_WINDOW_POLICY, SECONDARY_ALL_HISTORY_POLICY
    from .experiments.treatments import resolve_arms
    from .observers.context import ContextPolicy
    from .observers.tokens import TokenMatchConfig
    pending, errors, source_errors = [], [], []
    def check(value, path, predicate, reason):
        if value is None:
            pending.append({"field": path, "reason": "Explicit choice required: " + reason})
        else:
            try:
                valid = predicate(value)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if valid:
                return True
            errors.append({"field": path, "reason": reason})
        return False
    def label(value):
        return isinstance(value, str) and bool(value.strip())
    def integer(value, minimum=1):
        return type(value) is int and value >= minimum
    def keys(value, expected, path):
        return check(value, path, lambda v: isinstance(v, Mapping) and set(v) == set(expected),
                     "Expected fields: " + ", ".join(sorted(expected)))
    def construct(cls, value, path):
        try:
            return from_jsonable(cls, value)
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            errors.append({"field": path, "reason": str(error)})
            return None
    def names(value):
        return isinstance(value, list) and bool(value) and all(label(x) for x in value) and len(set(value)) == len(value)
    def increasing(value, minimum=1):
        return isinstance(value, list) and bool(value) and all(integer(x, minimum) for x in value) and value == sorted(set(value))
    if not isinstance(plan, Mapping):
        return {"version": RECOVERY_VERSION, "schema_ready": False, "saved_sources_verified": False,
                "launchable": False, "launch_authorized": False, "pending": [],
                "errors": [{"field": "plan", "reason": "Plan must be an object"}]}
    try:
        encoded = canonical_json(plan)
    except (TypeError, ValueError) as error:
        return {"version": RECOVERY_VERSION, "schema_ready": False, "saved_sources_verified": False,
                "launchable": False, "launch_authorized": False, "pending": [],
                "errors": [{"field": "plan", "reason": "Finite JSON required: " + str(error)}]}
    plan = json.loads(encoded)
    expected = {"version", "plan_id", "mode", "purpose", "fixture", "cohort", "admission", "saved_mines",
                "recovery", "observer", "workers", "analysis_settings", "near_budget_tokens", "resource_horizons",
                "dollar_horizons", "order", "max_parallel"}
    keys(plan, expected, "plan")
    check(plan.get("version"), "version", lambda v: v == RECOVERY_VERSION, "Supported saved-mine recovery plan version")
    check(plan.get("plan_id"), "plan_id", label, "New recovery plan identity")
    check(plan.get("mode"), "mode", lambda v: v == "full", "Full recovery over explicitly saved natural mines")
    check(plan.get("purpose"), "purpose", lambda v: v in {"development", "calibration"}, "development or calibration; main requires a separate freeze")
    check(plan.get("fixture"), "fixture", lambda v: type(v) is bool, "Explicit fixture boolean")
    check(plan.get("max_parallel"), "max_parallel", lambda v: integer(v) and v <= 2, "Explicit concurrency within the two-job maximum")
    cohort = plan.get("cohort")
    tasks, models = [], []
    if check(cohort, "cohort", lambda v: isinstance(v, Mapping), "Versioned declared cohort"):
        raw_tasks, raw_models = cohort.get("primary_tasks"), cohort.get("model_ids")
        if (isinstance(raw_tasks, list) and all(isinstance(t, Mapping) and label(t.get("task_id")) for t in raw_tasks)
                and isinstance(raw_models, list) and all(label(m) for m in raw_models)):
            # Reuse cohort/source/admission checks without supplying fake worker
            # or scientific values to the mining template's other fields.
            probe = mining_launch_template(cohort, version=VERSION)
            probe["admission"] = plan.get("admission")
            report = validate_mining_launch_plan(probe)
            for key, target in (("pending", pending), ("errors", errors)):
                target.extend(row for row in report[key] if row["field"].startswith(("cohort", "admission")))
            tasks, models = raw_tasks, raw_models
        else:
            errors.append({"field": "cohort", "reason": "Explicit primary task and worker model identities required"})
    task_ids = [task["task_id"] for task in tasks]
    order = plan.get("order")
    if keys(order, {"policy", "seed"}, "order"):
        check(order["policy"], "order.policy", lambda v: v in {"declared", "seeded_shuffle"}, "Explicit scheduling policy")
        check(order["seed"], "order.seed", lambda v: integer(v, 0), "Explicit nonnegative schedule seed")
    recovery = plan.get("recovery")
    horizons = {}
    if keys(recovery, {"horizons", "repeats", "arms", "observation_mask_policy"}, "recovery"):
        horizons = recovery["horizons"] if isinstance(recovery["horizons"], Mapping) else {}
        if keys(recovery["horizons"], task_ids, "recovery.horizons"):
            for task, values in horizons.items():
                check(values, "recovery.horizons." + task, increasing, "Increasing positive recovery horizons")
        check(recovery["repeats"], "recovery.repeats", names, "Explicit unique recovery repeat IDs")
        if check(recovery["arms"], "recovery.arms", names, "Explicit nonempty treatment subset"):
            try:
                resolve_arms(recovery["arms"])
            except ValueError as error:
                errors.append({"field": "recovery.arms", "reason": str(error)})
        check(recovery["observation_mask_policy"], "recovery.observation_mask_policy",
              lambda v: v in {PRIMARY_WINDOW_POLICY, SECONDARY_ALL_HISTORY_POLICY}, "Explicit primary detector window or named secondary all-history policy; no fabricated engineering S")
    observer = plan.get("observer")
    if keys(observer, {"source_policy", "include_returned_reasoning", "audit"}, "observer"):
        check(observer["source_policy"], "observer.source_policy", lambda v: v == "native-session-primary/v1", "Explicit ratified native-session primary source policy")
        check(observer["include_returned_reasoning"], "observer.include_returned_reasoning", lambda v: type(v) is bool, "Q14 returned-reasoning source choice remains required")
        audit = observer["audit"]
        if keys(audit, {f.name for f in fields(ObserverAuditConfig)}, "observer.audit"):
            for name, value in audit.items():
                check(value, "observer.audit." + name, lambda v: True, "Explicit audit scope/repeats, including disabled scope")
            if all(v is not None for v in audit.values()):
                construct(ObserverAuditConfig, audit, "observer.audit")
    analysis = plan.get("analysis_settings")
    if keys(analysis, {f.name for f in fields(AnalysisSpec) if f.name != "horizons"}, "analysis_settings"):
        for name, value in analysis.items():
            check(value, "analysis_settings." + name, lambda v: True, "Explicit descriptive/inference settings")
        if all(v is not None for v in analysis.values()):
            construct(AnalysisSpec, {"horizons": [], **analysis}, "analysis_settings")
    near = plan.get("near_budget_tokens")
    if keys(near, task_ids, "near_budget_tokens"):
        for task, value in near.items():
            check(value, "near_budget_tokens." + task, lambda v: integer(v, 0), "Explicit natural-run near-budget threshold")
    resources = plan.get("resource_horizons")
    if check(resources, "resource_horizons", lambda v: isinstance(v, Mapping), "Explicit resource horizon mapping, including empty"):
        for name, values in resources.items():
            check(values, "resource_horizons." + name, lambda v: isinstance(v, list) and bool(v)
                and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in v)
                and v == sorted(set(v)), "Increasing finite nonnegative resource horizons")
    check(plan.get("dollar_horizons"), "dollar_horizons", lambda v: isinstance(v, list)
          and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in v) and v == sorted(set(v)),
          "Explicit increasing dollar horizons, including empty")
    workers = plan.get("workers")
    if check(workers, "workers", lambda v: isinstance(v, list) and all(isinstance(w, Mapping) for w in v), "Both declared worker configurations"):
        if sorted(str(w.get("model")) for w in workers) != sorted(models):
            errors.append({"field": "workers", "reason": "Must preserve both declared worker models exactly"})
        for i, worker in enumerate(workers):
            path = f"workers[{i}]"
            if not keys(worker, {"model", "recovery_task_allowances", "observer_settings", "observer_context_policy", "observer_context_counter", "packet_settings"}, path):
                continue
            if str(worker["model"]).startswith(("scripted/", "fixture/", "dev/")) and plan.get("fixture") is not True:
                errors.append({"field": path + ".model", "reason": "Scripted model requires fixture=True"})
            allowances = worker["recovery_task_allowances"]
            if keys(allowances, task_ids, path + ".recovery_task_allowances"):
                for task, allowance in allowances.items():
                    if check(allowance, path + ".recovery_task_allowances." + task, integer, "Positive explicit recovery allowance"):
                        if increasing(horizons.get(task)) and any(h > allowance for h in horizons[task]):
                            errors.append({"field": "recovery.horizons." + task, "reason": "Horizon exceeds a declared recovery worker allowance"})
            settings = worker["observer_settings"]
            if check(settings, path + ".observer_settings", lambda v: isinstance(v, Mapping), "Explicit observer route/reasoning/reply settings"):
                for key in ("provider", "reasoning"):
                    check(settings.get(key), path + ".observer_settings." + key, lambda v: isinstance(v, Mapping) and bool(v), "Explicit route/reasoning choice")
                cap_keys = set(settings) & {"max_tokens", "max_completion_tokens"}
                if len(cap_keys) != 1 or not integer(settings[next(iter(cap_keys))]):
                    errors.append({"field": path + ".observer_settings", "reason": "One explicit positive reply token cap required"})
                if set(settings) & {"model", "messages", "tools", "stream"}:
                    errors.append({"field": path + ".observer_settings", "reason": "Observer settings cannot override request content/model"})
            context = worker["observer_context_policy"]
            if keys(context, {f.name for f in fields(ContextPolicy)}, path + ".observer_context_policy"):
                if all(v is None for v in context.values()):
                    pending.append({"field": path + ".observer_context_policy", "reason": "Explicit request limits and accepted measurement fidelity required"})
                else:
                    policy = construct(ContextPolicy, context, path + ".observer_context_policy")
                    counter = worker["observer_context_counter"]
                    if policy is not None and policy.max_input_tokens is not None:
                        if keys(counter, {"token_counter", "encoding_name"}, path + ".observer_context_counter"):
                            if check(counter["token_counter"], path + ".observer_context_counter.token_counter", lambda v: v in {"byte_proxy", "tiktoken"}, "Explicit local request counter"):
                                construct(PacketSettings, counter, path + ".observer_context_counter")
                                if counter["token_counter"] == "byte_proxy" and not policy.allow_proxy:
                                    errors.append({"field": path + ".observer_context_policy.allow_proxy", "reason": "Byte proxy requires explicit acceptance"})
                    elif counter is not None:
                        errors.append({"field": path + ".observer_context_counter", "reason": "Request counter applies only to an explicit input-token context limit"})
            packet = worker["packet_settings"]
            if keys(packet, {f.name for f in fields(PacketSettings)}, path + ".packet_settings"):
                check(packet["token_counter"], path + ".packet_settings.token_counter", lambda v: v in {"byte_proxy", "tiktoken"}, "Explicit packet-text tokenizer")
                check(packet["filters"], path + ".packet_settings.filters", lambda v: isinstance(v, list), "Explicit filtered packets, including empty")
                matching = packet["matching"]
                if keys(matching, {f.name for f in fields(TokenMatchConfig)}, path + ".packet_settings.matching"):
                    if matching["tolerance_tokens"] is None and matching["tolerance_fraction"] is None:
                        pending.append({"field": path + ".packet_settings.matching", "reason": "Explicit token-matching tolerance required"})
                    check(matching["max_attempts"], path + ".packet_settings.matching.max_attempts", integer, "Explicit bounded summary attempts")
                    check(matching["max_words"], path + ".packet_settings.matching.max_words", integer, "Explicit summary word guard")
                    if packet["token_counter"] is not None and packet["filters"] is not None and matching["max_attempts"] is not None:
                        construct(PacketSettings, packet, path + ".packet_settings")
    sources = plan.get("saved_mines")
    seen, source_count = set(), 0
    if keys(sources, task_ids, "saved_mines"):
        for task in tasks:
            task_id = task["task_id"]
            cells = sources[task_id]
            if not keys(cells, models, "saved_mines." + task_id):
                continue
            for model, runs in cells.items():
                path = "saved_mines." + task_id + "." + model
                if not check(runs, path, lambda v: isinstance(v, list) and bool(v), "Explicit nonempty saved natural-run list; no outcome-based replacement"):
                    continue
                for i, row in enumerate(runs):
                    source_count += 1
                    cell_path = path + f"[{i}]"
                    if not keys(row, {"run_id", "original_mine_config", "original_mine_config_sha256"}, cell_path):
                        continue
                    if not check(row["run_id"], cell_path + ".run_id", label, "Explicit saved natural run ID"):
                        continue
                    if row["run_id"] in seen:
                        errors.append({"field": cell_path, "reason": "Saved natural run appears more than once"})
                    seen.add(row["run_id"])
                    original = row["original_mine_config"]
                    if not keys(original, {f.name for f in fields(RunConfig)}, cell_path + ".original_mine_config"):
                        continue
                    cfg = construct(RunConfig, original, cell_path + ".original_mine_config")
                    if cfg is not None:
                        if original != to_jsonable(cfg):
                            errors.append({"field": cell_path, "reason": "Original mine config must roundtrip without defaults or ignored fields"})
                        if (cfg.run_id != row["run_id"] or cfg.model != model or cfg.purpose != plan.get("purpose")):
                            errors.append({"field": cell_path, "reason": "Saved run/model/purpose differs from its declared cell"})
                        if not isinstance(cfg.metadata, Mapping) or cfg.metadata.get("source_binding") != task:
                            errors.append({"field": cell_path, "reason": "Original mine does not bind this exact qualified successor task declaration"})
                    if row["original_mine_config_sha256"] != hashlib.sha256(canonical_json(original)).hexdigest():
                        errors.append({"field": cell_path, "reason": "Original mine configuration SHA-256 differs"})
                    if store is not None:
                        try:
                            retained = saved_mine_source(store, row["run_id"])
                            if retained != row:
                                raise ValueError("Declared original mine configuration differs from retained execution")
                            start = next(event.payload for event in store.iter_events(row["run_id"])
                                if event.kind == "lifecycle" and event.payload.get("transition") == "run_started")
                            if any(start.get("task", {}).get(name) != task[name] for name in ("task_id", "family_id", "environment_version")):
                                raise ValueError("Saved natural execution has a different task source binding")
                        except (KeyError, ValueError, TypeError) as error:
                            source_errors.append({"field": cell_path, "reason": str(error)})
    try:
        canonical_json(plan)
    except (TypeError, ValueError) as error:
        errors.append({"field": "plan", "reason": "Finite JSON required: " + str(error)})
    if not pending and not errors:
        try:
            _recovery_batch(plan)
        except (TypeError, ValueError, KeyError) as error:
            errors.append({"field": "campaign_config", "reason": str(error)})
    schema_ready = not pending and not errors
    verified = schema_ready and store is not None and not source_errors and source_count > 0
    if store is None:
        pending.append({"field": "saved_source_verification", "reason": "Read-only artifact store required to bind saved natural executions"})
    return {"version": RECOVERY_VERSION, "plan_id": plan.get("plan_id"), "mode": "full",
        "schema_ready": schema_ready, "saved_sources_verified": verified, "saved_source_count": source_count,
        "launchable": schema_ready and verified, "launch_authorized": False,
        "pending": pending, "errors": [*errors, *source_errors],
        "qualification": "Offline schema/saved-source checks only; current admission/source/image/route qualification, counter resolution and launch authorization remain separate."}


def _recovery_batch(plan: Mapping[str, Any]) -> BatchPlan:
    """Construct only after schema checks; preserve all original worker fields."""
    from .observers.context import ContextPolicy
    digest = hashlib.sha256(canonical_json(plan)).hexdigest()
    jobs = []
    for task in plan["cohort"]["primary_tasks"]:
        for worker in plan["workers"]:
            for source in plan["saved_mines"][task["task_id"]][worker["model"]]:
                suffix = hashlib.sha256(canonical_json([digest, source["run_id"]])).hexdigest()[:24]
                identity = plan["plan_id"] + ":" + suffix
                original = from_jsonable(RunConfig, source["original_mine_config"])
                mining = replace(original, run_id=identity + ":natural-reference", campaign_id=identity)
                jobs.append(CampaignJob(task_id=task["task_id"], config=CampaignConfig(campaign_id=identity,
                    mining=mining, mode="full", source_natural_run_id=source["run_id"],
                    fixture=plan["fixture"], engineering_fallback=False,
                    recovery_generated_tokens=worker["recovery_task_allowances"][task["task_id"]],
                    horizons=tuple(plan["recovery"]["horizons"][task["task_id"]]), repeats=tuple(plan["recovery"]["repeats"]),
                    arms=tuple(plan["recovery"]["arms"]), observation_mask_policy=plan["recovery"]["observation_mask_policy"],
                    observer_source_policy=plan["observer"]["source_policy"],
                    observer_include_returned_reasoning=plan["observer"]["include_returned_reasoning"],
                    observer_settings=worker["observer_settings"],
                    observer_context_policy=from_jsonable(ContextPolicy, worker["observer_context_policy"]),
                    observer_context_counter=worker["observer_context_counter"],
                    observer_audit=from_jsonable(ObserverAuditConfig, plan["observer"]["audit"]),
                    packet_settings=from_jsonable(PacketSettings, worker["packet_settings"]),
                    analysis_settings=plan["analysis_settings"], near_budget_tokens=plan["near_budget_tokens"][task["task_id"]],
                    resource_horizons={name: tuple(values) for name, values in plan["resource_horizons"].items()},
                    dollar_horizons=tuple(plan["dollar_horizons"]), schedule_seed=plan["order"]["seed"])))
    if plan["order"]["policy"] == "seeded_shuffle":
        random.Random(plan["order"]["seed"]).shuffle(jobs)
    return BatchPlan(batch_id=plan["plan_id"], jobs=tuple(jobs), max_parallel=plan["max_parallel"])


def build_recovery_batch(plan: Mapping[str, Any], *, store: ArtifactStoreProtocol | None) -> BatchPlan:
    """Build existing full CampaignConfig jobs; never mine again or launch work."""
    report = validate_recovery_launch_plan(plan, store=store)
    if not report["launchable"]:
        raise ValueError("Recovery plan is nonlaunchable: " + json.dumps({"pending": report["pending"], "errors": report["errors"]}, sort_keys=True))
    return _recovery_batch(json.loads(canonical_json(plan)))
