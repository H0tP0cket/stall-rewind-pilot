"""Opt-in, read-only live-T handoff; never starts or retries an extraction.

Completion means every recorded checkpoint has both validated T scopes and its
durable queue/pair receipts. Empty valid T is complete; a failed or missing T is
not. This module is intentionally not imported by the sealed campaign runner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..contracts import to_jsonable
from . import artifacts
from ._canonical import content_id
from .extractor import CheckpointInfo, validate_lineage_step
from .live import REVISION, live_observer_status
from .prefix import SESSION_LINEAGE_REVISION, SourceScope, view_for_checkpoint
from .update import validate_version

VERSION = "live-t-readiness-handoff.v1"
_FILES = ("live.py", "live_step.py", "context.py", "extractor.py", "prefix.py",
          "prompts.py", "update.py", "tstore.py", "artifacts.py", "client.py",
          "readiness_v1.py")


def source_binding():
    root = Path(__file__).parent
    files = {"observers/" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in _FILES}
    for name in ("providers.py", "accounting.py", "contracts.py", "persistence.py", "store.py"):
        files[name] = hashlib.sha256((root.parent / name).read_bytes()).hexdigest()
    return {"kind": "measured_source_digests_not_archived_artifact_refs", "files": files,
            "digest": content_id(files)}


def _get(store, kind, identity):
    try:
        return store.get_record(kind, identity)
    except KeyError:
        return None


def inspect_live_t_queue(store, run_id, *, namespace, config, include_returned_reasoning,
                         context_counter=None, expected_sources=None):
    """Return an exact-prefix coverage report without writing or calling models.

    Recovery directions are descriptive, not authorization to repeat a charged
    call. The existing live service/step owns locking and retained retry policy.
    """
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("Explicit live observer namespace required")
    if type(include_returned_reasoning) is not bool:
        raise ValueError("Explicit returned-reasoning choice required")
    sources = source_binding()
    errors = []
    if expected_sources is not None and expected_sources != sources:
        errors.append("current observer source binding differs from expected source")
    if config.context_policy is None:
        errors.append("observer context policy is not bounded")
    elif config.context_policy.max_input_tokens is not None:
        policy = config.context_policy
        if context_counter is None or (context_counter.model, context_counter.counter_id, context_counter.revision) != (
                config.model, policy.token_counter_id, policy.token_counter_revision):
            errors.append("context token counter does not bind the declared model and policy")
    queue_id = content_id({"revision": REVISION, "run_id": run_id, "namespace": namespace})
    binding = _get(store, "live_observer_binding_v1", queue_id)
    if binding is None:
        errors.append("durable live observer queue binding missing")
    else:
        expected = {"run_id": run_id, "namespace": namespace, "extractor": to_jsonable(config),
                    "include_returned_reasoning": include_returned_reasoning,
                    "context_counter": context_counter.as_record() if context_counter else None}
        for key, value in expected.items():
            if binding.get(key) != value:
                errors.append("durable queue binding differs: " + key)
    checkpoints = sorted(store.list_checkpoints(run_id), key=lambda cp: cp.ordinal)
    report = live_observer_status(store, run_id, namespace)
    states = {row["checkpoint_id"]: row for row in report["checkpoints"]}
    scopes = {name: SourceScope(session_scope=scope, include_returned_reasoning=include_returned_reasoning,
              lineage_revision=SESSION_LINEAGE_REVISION)
              for name, scope in (("primary", "current_session"), ("archive", "full_trajectory"))}
    namespaces = {"primary": namespace + ":primary-session-v1", "archive": namespace + ":archive-global-v1"}
    parents, prior, rows, blocked = {"primary": None, "archive": None}, None, [], None
    for index, cp in enumerate(checkpoints):
        problems = []
        if cp.ordinal != index or (prior and (cp.parent_checkpoint_id != prior.checkpoint_id
                                             or cp.event_sequence < prior.event_sequence)):
            problems.append("checkpoint is not in contiguous source order")
        job_id = content_id({"queue_id": queue_id, "checkpoint_id": cp.checkpoint_id})
        job = _get(store, "live_observer_job_v1", job_id)
        expected_job = {"revision": REVISION, "queue_id": queue_id, "run_id": run_id,
            "checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
            "checkpoint_sha256": content_id(to_jsonable(cp)), "event_cut": cp.event_sequence,
            "native_session_id": cp.scope.session_id,
            "predecessor_checkpoint_id": prior.checkpoint_id if prior else None}
        if job != expected_job:
            problems.append("durable checkpoint job missing or source binding differs")
        versions = {}
        for chain, scope in scopes.items():
            parent, previous = parents[chain], prior
            if chain == "primary" and prior and prior.scope.session_id != cp.scope.session_id:
                parent, previous = None, None
            version = artifacts.load_primary(store, run_id, cp.checkpoint_id, chain_namespace=namespaces[chain])
            versions[chain] = version
            if version is None or version.status not in {"valid", "valid_with_rejections"}:
                problems.append(chain + " T missing or invalid")
            else:
                info = lambda item: CheckpointInfo(item.checkpoint_id, item.event_sequence,
                                                    item.scope.session_id, item.scope.requirement_version)
                problems.extend(chain + ": " + problem for problem in validate_lineage_step(version,
                    run_id=run_id, parent=parent, checkpoint=info(cp), prior_checkpoint=info(previous) if previous else None,
                    config=config, scope=scope))
                view = view_for_checkpoint(store, run_id, through_seq=cp.event_sequence,
                                          target_session_id=cp.scope.session_id, scope=scope)
                problems.extend(chain + ": " + problem for problem in validate_version(version, view))
            parents[chain] = version
        pair_id = content_id({"revision": "observer-scope-pair/v1", "namespace": namespace,
                              "run_id": run_id, "checkpoint_id": cp.checkpoint_id})
        pair = _get(store, "observer_scope_pair_v1", pair_id)
        if pair is None or any(pair.get(chain + "_version_id") != (v.version_id() if v else None)
                               for chain, v in versions.items()):
            problems.append("durable primary/archive completion pair missing or mismatched")
        state = states.get(cp.checkpoint_id, {})
        if state.get("status") != "complete":
            problems.append("durable queue completion state missing")
        if blocked is not None:
            problems.append("earlier checkpoint is incomplete: " + blocked)
        if problems and blocked is None:
            blocked = cp.checkpoint_id
        rows.append({"checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
                     "event_cut": cp.event_sequence, "native_session_id": cp.scope.session_id,
                     "complete": not problems, "queue_status": state.get("status", "queued"),
                     "attempt_index": state.get("attempt_index", 0), "problems": problems,
                     "pair_id": pair_id if pair else None,
                     "version_ids": {key: value.version_id() if value else None for key, value in versions.items()}})
        prior = cp
    if source_binding() != sources:
        errors.append("observer sources changed during inspection")
    if [to_jsonable(cp) for cp in store.list_checkpoints(run_id)] != [to_jsonable(cp) for cp in checkpoints]:
        errors.append("checkpoint inventory changed during inspection; inspect again")
    first = next((row for row in rows if not row["complete"]), None)
    action = ("none" if first is None else {
        "uncertain": "reconcile_same_attempt_from_durable_receipts; do_not_increment_or_skip",
        "retrying": "existing_service_applies_retained_retry_policy_and_finite_attempt_limit",
        "failed": "blocked; inspect_failure_and_create_explicit_versioned_repair_without_skipping",
        "running": "wait_or_recover_original_attempt; do_not_start_second_consumer",
    }.get(first["queue_status"], "reconcile_durable_outbox_and_process_earliest_incomplete_checkpoint"))
    complete = bool(rows) and not errors and first is None and report["observer_receipt_coverage_complete"]
    return {"revision": VERSION, "run_id": run_id, "namespace": namespace,
            "complete_at_recorded_cut": complete, "checkpoint_count": len(rows),
            "complete_checkpoint_count": sum(row["complete"] for row in rows), "checkpoints": rows,
            "next_checkpoint_id": first["checkpoint_id"] if first else None, "recovery_direction": action,
            "binding_errors": errors, "source_binding": sources,
            "receipt_coverage_complete": report["observer_receipt_coverage_complete"],
            "cost_usage_missing_call_ids": report["observer_usage_missing_call_ids"],
            "scope": "recorded checkpoint coverage only; not semantic T quality or future worker completion",
            "provider_calls": 0, "store_writes": 0, "worker_modified": False}
