"""One ordered live T checkpoint; immutable failed attempts never advance T.

The caller schedules/retries work and owns retry limits. This primitive never
falls back to full-prefix extraction or launches a descendant past a gap.
Only observer work takes the per-run lock; natural workers do not use it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3

from ..contracts import ArtifactRef, canonical_json, from_jsonable, to_jsonable
from ..persistence import persist
from . import artifacts
from ._canonical import content_id
from .client import CallReceipt, NullClient, ScriptedClient
from .extractor import CheckpointInfo, assert_lineage_step, update_t
from .prefix import SESSION_LINEAGE_REVISION, SourceScope, view_for_checkpoint
from .prompts import render_events
from .tstore import TVersion
from .update import validate_version

VERSION = "live-t-checkpoint-step.v1"
VALID = {"valid", "valid_with_rejections"}


class _Uncertain(RuntimeError):
    pass


def _get(store, kind, identity):
    try:
        return store.get_record(kind, identity)
    except KeyError:
        return None


def _put(store, kind, identity, body, run_id):
    return persist(lambda: store.put_record(kind, identity, body, run_id=run_id), evidence=body)


def _info(cp):
    return CheckpointInfo(cp.checkpoint_id, cp.event_sequence, cp.scope.session_id, cp.scope.requirement_version)


@contextmanager
def _lock(store, namespace, run_id):
    path = Path(store.root) / "locks" / ("live-t-" + hashlib.sha256(canonical_json([namespace, run_id])).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate(store, version, cp, parent, prior, config, scope):
    assert_lineage_step(version, run_id=cp.run_id, parent=parent, checkpoint=_info(cp),
                        prior_checkpoint=_info(prior) if prior else None, config=config, scope=scope)
    view = view_for_checkpoint(store, cp.run_id, through_seq=cp.event_sequence,
                              target_session_id=cp.scope.session_id, scope=scope)
    problems = validate_version(version, view)
    if problems:
        raise ValueError("Live T source validation failed: " + "; ".join(problems))


def _known_retry(store, receipts):
    """Consume provider policy; successful malformed ops have a separate retry.

    Known usage alone does not authorize repeating a provider error. Decisions
    must belong to the exact retained receipt, and missing policy fails closed.
    """
    retryable, uncertain, delays = bool(receipts), False, []
    # True only when every uncertainty is a durably finished provider receipt
    # whose usage is unknown (timeout/dropped connection). Such an attempt may be
    # repeated under an explicit declared policy; its unknown usage stays unknown.
    repeatable_unknown_usage = bool(receipts)
    for receipt in receipts:
        if receipt.status == "ok":
            continue
        finished = _get(store, "provider_call_finished", receipt.call_record_id)
        decision = finished.get("retry_decision") if finished is not None else None
        if (finished is None or finished.get("receipt") != receipt.as_record()
                or not isinstance(decision, dict) or type(decision.get("retry")) is not bool):
            retryable, uncertain, repeatable_unknown_usage = False, True, False
            continue
        delay = decision.get("delay_seconds")
        valid_delay = type(delay) in (int, float) and math.isfinite(delay) and delay >= 0
        if valid_delay:
            delays.append(delay)
        if receipt.usage is None or receipt.usage.generated_tokens is None:
            retryable, uncertain = False, True
            if receipt.status not in {"timeout", "error"}:
                repeatable_unknown_usage = False
        elif not decision["retry"]:
            retryable, repeatable_unknown_usage = False, False
        elif not valid_delay:
            retryable, uncertain, repeatable_unknown_usage = False, True, False
    return {"retryable": retryable, "retry_after_seconds": max(delays) if delays else None,
            "status": "uncertain" if uncertain else "failed",
            "repeatable_unknown_usage": uncertain and repeatable_unknown_usage}


def _supports_call_id(client):
    try:
        parameter = inspect.signature(client.complete).parameters.get("call_id")
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class _JournalClient:
    def __init__(self, store, client, identity, run_id, *, recovery_only=False):
        self.store, self.client, self.identity, self.run_id = store, client, identity, run_id
        self.recovery_only = recovery_only

    def _received(self, request, raw, receipt, *, reconciled_from=None):
        if (receipt.request_hash != request.request_hash() or receipt.requested_model != request.model
                or receipt.cost_role != request.cost_role):
            raise _Uncertain("Returned receipt does not bind this exact observer request")
        body = {"request": to_jsonable(request), "raw_response": raw,
                "receipt": receipt.as_record(), "reconciled_from": reconciled_from}
        # A process death after this write never requires another provider call.
        _put(self.store, "live_t_call_received_v1", self.identity, body, self.run_id)
        return body

    def _reconcile(self, request, started):
        matches = []
        expected = canonical_json({"model": request.model, "messages": request.messages,
                                   **request.settings, "stream": False})
        prior = set(started["prior_provider_call_ids"])
        for row in self.store.list_records("provider_call_started", run_id=self.run_id):
            call_id = row["record_id"]
            if call_id in prior:
                continue
            if started.get("provider_call_id") is not None and call_id != started["provider_call_id"]:
                continue
            source = self.store.get_record("provider_call_started", call_id)
            if any(source.get(k) != v for k, v in {
                "run_id": request.run_id, "target_checkpoint_id": request.target_checkpoint_id,
                "request_hash": request.request_hash(), "cost_role": request.cost_role,
                "purpose": request.purpose}.items()):
                continue
            if self.store.get_bytes(from_jsonable(ArtifactRef, source["request_ref"])) != expected:
                continue
            matches.append((call_id, source))
        if len(matches) != 1:
            raise _Uncertain("Started observer call lacks one unique newly-started exact provider receipt")
        call_id, source = matches[0]
        finished = _get(self.store, "provider_call_finished", call_id)
        if finished is None:
            raise _Uncertain("Exact provider call started without a finished receipt")
        receipt = from_jsonable(CallReceipt, finished["receipt"])
        if receipt.call_record_id != call_id or finished["request_ref"] != source["request_ref"]:
            raise _Uncertain("Provider receipt identity or request binding differs")
        raw = None
        if finished.get("response_ref") is not None:
            data = self.store.get_bytes(from_jsonable(ArtifactRef, finished["response_ref"]))
            if hashlib.sha256(data).hexdigest() != receipt.response_hash:
                raise _Uncertain("Provider raw response hash differs from receipt")
            if receipt.status == "ok":
                response = json.loads(data)
                raw = response["choices"][0]["message"]["content"]
                if not isinstance(raw, str):
                    raise _Uncertain("Completed provider response lacks exact text")
        elif receipt.status == "ok":
            raise _Uncertain("Completed provider response bytes are unavailable")
        return self._received(request, raw, receipt, reconciled_from=call_id)

    def complete(self, request):
        received = _get(self.store, "live_t_call_received_v1", self.identity)
        started = _get(self.store, "live_t_call_started_v1", self.identity)
        if received is None:
            if started is not None:
                if started["request"] != to_jsonable(request):
                    raise ValueError("Live observer request changed across restart")
                stable_id = started.get("provider_call_id")
                if stable_id is not None and _supports_call_id(self.client):
                    # The explicit stable-ID API owns request-bound replay,
                    # unresolved-start refusal, and safe first start if this
                    # process died before the provider recorded any start.
                    raw, receipt = self.client.complete(request, call_id=stable_id)
                    received = self._received(request, raw, receipt)
                else:
                    received = self._reconcile(request, started)
            else:
                if self.recovery_only:
                    raise _Uncertain("Earlier attempt has no durable completion; do not skip it")
                stable_id = "live-observer:" + self.identity if _supports_call_id(self.client) else None
                started = {"request": to_jsonable(request), "provider_call_id": stable_id, "prior_provider_call_ids": [
                    row["record_id"] for row in self.store.list_records("provider_call_started", run_id=self.run_id)]}
                _put(self.store, "live_t_call_started_v1", self.identity, started, self.run_id)
                raw, receipt = (self.client.complete(request, call_id=stable_id) if stable_id is not None
                                else self.client.complete(request))
                # Development clients restart their local counters. Give only
                # these explicitly scripted clients a durable attempt identity;
                # never relabel a real provider receipt or cost operation.
                if isinstance(self.client, (ScriptedClient, NullClient)):
                    receipt = replace(receipt, call_record_id="live-scripted:" + self.identity)
                received = self._received(request, raw, receipt)
        if received["request"] != to_jsonable(request):
            raise ValueError("Received observer request changed across restart")
        receipt = from_jsonable(CallReceipt, received["receipt"])
        artifacts.record_calls(self.store, self.run_id, (receipt,), received["raw_response"])
        return received["raw_response"], receipt


def _chain(store, cp, parent, prior, cuts, client, config, scope, chain_namespace,
           attempt_index, context_counter, reusable=None, allow_uncertain_repeat=False):
    binding = {"version": VERSION, "checkpoint": to_jsonable(cp), "chain_namespace": chain_namespace,
               "parent_version_id": parent.version_id() if parent else None,
               "prior_checkpoint_id": prior.checkpoint_id if prior else None,
               "config_id": config.config_id(), "scope": scope.as_record(),
               "context_counter": context_counter.as_record() if context_counter else None}
    view = view_for_checkpoint(store, cp.run_id, through_seq=cp.event_sequence,
        after_seq=prior.event_sequence if prior else -1, target_session_id=cp.scope.session_id, scope=scope)
    binding["source_text_hash"] = content_id(render_events(view.events()))
    identity = content_id({"version": VERSION, "run_id": cp.run_id,
                           "checkpoint_id": cp.checkpoint_id, "chain_namespace": chain_namespace})
    _put(store, "live_t_step_spec_v1", identity, binding, cp.run_id)
    existing = artifacts.load_primary(store, cp.run_id, cp.checkpoint_id, chain_namespace=chain_namespace)
    if existing is not None:
        _validate(store, existing, cp, parent, prior, config, scope)
        if existing.status not in VALID:
            return {"status": "failed", "retryable": False, "error": "Existing immutable primary pointer is failed/missing"}, existing
        return {"status": "complete", "reused": True}, existing
    # A new attempt is legal only after every earlier started attempt has a
    # known failed result. Unfinished earlier work is recovered at its old ID.
    # An earlier attempt whose only uncertainty is unknown usage on a finished
    # timeout/error receipt may be repeated when the caller declares that
    # policy; its receipts keep their unknown usage in the ledger.
    repeated_uncertain = []
    for index in range(attempt_index):
        old_id = content_id({"step_id": identity, "attempt_index": index})
        old = _get(store, "live_t_attempt_started_v1", old_id)
        if old is None:
            continue  # e.g. primary's first call follows an archive retry
        outcome = _get(store, "live_t_extraction_result_v1", old_id)
        if outcome is None:
            return _chain(store, cp, parent, prior, cuts, client, config, scope, chain_namespace,
                          index, context_counter, reusable, allow_uncertain_repeat)
        if outcome["version"]["status"] in VALID:
            return _chain(store, cp, parent, prior, cuts, client, config, scope, chain_namespace,
                          index, context_counter, reusable, allow_uncertain_repeat)
        policy = _known_retry(store, tuple(from_jsonable(CallReceipt, row) for row in outcome["receipts"]))
        if not policy["retryable"]:
            if allow_uncertain_repeat and policy["repeatable_unknown_usage"]:
                repeated_uncertain.append(old_id)
                continue
            return {**policy,
                    "error": "Earlier failed attempt is not safe to automatically repeat", "attempt_index": index}, TVersion.from_record(outcome["version"])
    attempt_id = content_id({"step_id": identity, "attempt_index": attempt_index})
    started = {"step_id": identity, "binding": binding, "attempt_index": attempt_index}
    if repeated_uncertain:
        started["repeats_uncertain_attempts"] = repeated_uncertain
        started["earlier_usage_policy"] = "Earlier receipts retain unknown usage in CostLedger; never zeroed or reattributed"
    _put(store, "live_t_attempt_started_v1", attempt_id, started, cp.run_id)
    result = _get(store, "live_t_extraction_result_v1", attempt_id)
    if result is None:
        journal = _JournalClient(store, client, attempt_id, cp.run_id)
        try:
            out = update_t(store=store, run_id=cp.run_id, parent=parent, checkpoint=_info(cp),
                prior_checkpoint=_info(prior) if prior else None, checkpoint_cuts=cuts,
                client=journal, config=config, scope=scope, context_counter=context_counter,
                reusable_call=reusable)
        except Exception as exc:
            # Never retry a client operation inside a persistence retry. Error
            # text can contain client secrets; retain type plus safe guidance.
            error = {"step_id": identity, "attempt_index": attempt_index, "status": "uncertain",
                     "retryable": False, "error_type": type(exc).__name__,
                     "error": "Observer completion interrupted; recover this same attempt from durable receipts."}
            _put(store, "live_t_attempt_error_v1", content_id({"attempt_id": attempt_id, **error}), error, cp.run_id)
            return error, None
        _validate(store, out.version, cp, parent, prior, config, scope)
        policy = _known_retry(store, out.receipts)
        result = {"binding": binding, "version": out.version.as_record(),
                  "advisor_ops": [dict(op) for op in out.advisor_ops],
                  "receipts": [receipt.as_record() for receipt in out.receipts], "raw_response": out.raw_response,
                  **policy, "status": "complete" if out.version.status in VALID else policy["status"],
                  "retryable": out.version.status not in VALID and policy["retryable"], "attempt_index": attempt_index}
        _put(store, "live_t_extraction_result_v1", attempt_id, result, cp.run_id)
    if result["binding"] != binding:
        raise ValueError("Received extraction binding changed")
    version = TVersion.from_record(result["version"])
    _validate(store, version, cp, parent, prior, config, scope)
    artifacts.save_version(store, version, advisor_ops=result["advisor_ops"],
        receipts=tuple(from_jsonable(CallReceipt, receipt) for receipt in result["receipts"]), raw_response=result["raw_response"])
    if version.status in VALID:
        artifacts.save_version(store, version, primary=True, chain_namespace=chain_namespace)
    policy = (_known_retry(store, tuple(from_jsonable(CallReceipt, row) for row in result["receipts"]))
              if version.status not in VALID else {"status": "complete", "retryable": False, "retry_after_seconds": None})
    return {**policy,
            "error": version.failure, "attempt_index": result["attempt_index"]}, version


def advance_checkpoint(store, run_id, checkpoint_id, client, config, *, namespace,
                       include_returned_reasoning, context_counter=None, attempt_index=0,
                       allow_uncertain_repeat=False):
    """Advance exactly one immutable checkpoint, or report its blocking gap.

    Retry the same attempt after uncertainty/local persistence failure. Increment
    only for a returned ``failed, retryable=True``, or for an ``uncertain`` result
    reporting ``repeatable_unknown_usage`` when the caller passes
    ``allow_uncertain_repeat`` under its declared policy; caller owns finite retries.
    Valid-with-rejections is structurally valid T, with rejected ops retained.
    """
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("Live T requires an explicit namespace")
    if type(include_returned_reasoning) is not bool:
        raise ValueError("Live T requires an explicit returned-reasoning source choice")
    if type(attempt_index) is not int or attempt_index < 0:
        raise ValueError("Live T attempt index must be a nonnegative integer")
    base = {"version": VERSION, "run_id": run_id, "checkpoint_id": checkpoint_id,
            "namespace": namespace, "attempt_index": attempt_index,
            "primary_namespace": namespace + ":primary-session-v1", "archive_namespace": namespace + ":archive-global-v1",
            "primary_version_id": None, "archive_version_id": None}
    with _lock(store, namespace, run_id):
        try:
            cps = sorted(store.list_checkpoints(run_id), key=lambda cp: cp.ordinal)
            target = next((i for i, cp in enumerate(cps) if cp.checkpoint_id == checkpoint_id), None)
            if target is None:
                raise ValueError("Checkpoint is not recorded in this run")
            cps = cps[:target + 1]
            sessions, last_session = set(), None
            for index, cp in enumerate(cps):
                if cp.run_id != run_id or not isinstance(cp.scope.session_id, str) or not cp.scope.session_id:
                    raise ValueError("Live T requires immutable run and native session identities")
                if index and (cp.ordinal <= cps[index - 1].ordinal or cp.event_sequence < cps[index - 1].event_sequence):
                    raise ValueError("Checkpoint order/cuts are invalid")
                if cp.scope.session_id != last_session:
                    if cp.scope.session_id in sessions:
                        raise ValueError("Native session identity reappears after reset")
                    sessions.add(cp.scope.session_id)
                    last_session = cp.scope.session_id
            common = {"include_returned_reasoning": include_returned_reasoning, "lineage_revision": SESSION_LINEAGE_REVISION}
            scopes = {"primary": SourceScope(session_scope="current_session", **common),
                      "archive": SourceScope(session_scope="full_trajectory", **common)}
            parents, prior = {"primary": None, "archive": None}, None
            # Validate immediate predecessors and their ordered ancestry. This
            # permits reload of valid legacy-compatible namespaced pointers but
            # never inherits the batch observer's failed-parent skipping rule.
            for earlier in cps[:-1]:
                for chain in ("archive", "primary"):
                    parent, previous = parents[chain], prior
                    if chain == "primary" and prior and prior.scope.session_id != earlier.scope.session_id:
                        parent, previous = None, None
                    version = artifacts.load_primary(store, run_id, earlier.checkpoint_id,
                        chain_namespace=base[chain + "_namespace"])
                    if version is None or version.status not in VALID:
                        return {**base, "status": "retrying", "retryable": False,
                                "blocked_on_checkpoint_id": earlier.checkpoint_id, "blocked_chain": chain,
                                "error": "Previous checkpoint requires valid primary and archive T"}
                    _validate(store, version, earlier, parent, previous, config, scopes[chain])
                    parents[chain] = version
                prior = earlier
            cp, cuts = cps[-1], {item.checkpoint_id: item.event_sequence for item in cps}
            versions = {}
            for chain in ("archive", "primary"):
                parent, previous = parents[chain], prior
                if chain == "primary" and prior and prior.scope.session_id != cp.scope.session_id:
                    parent, previous = None, None
                reusable = artifacts.reusable_call(store, versions["archive"]) if chain == "primary" else None
                result, version = _chain(store, cp, parent, previous, cuts, client, config, scopes[chain],
                    base[chain + "_namespace"], attempt_index, context_counter, reusable, allow_uncertain_repeat)
                if version is not None:
                    base[chain + "_version_id"] = version.version_id()
                    versions[chain] = version
                if result["status"] != "complete":
                    return {**base, **result, "failed_chain": chain}
            primary, archive = versions["primary"], versions["archive"]
            pair_id = content_id({"revision": "observer-scope-pair/v1", "namespace": namespace,
                                 "run_id": run_id, "checkpoint_id": checkpoint_id})
            pair = {"revision": "observer-scope-pair/v1", "namespace": namespace, "run_id": run_id,
                "checkpoint_id": checkpoint_id, "primary_version_id": primary.version_id(), "archive_version_id": archive.version_id(),
                "primary_namespace": base["primary_namespace"], "archive_namespace": base["archive_namespace"],
                "native_session_id": cp.scope.session_id, "primary_scope": scopes["primary"].as_record(),
                "archive_scope": scopes["archive"].as_record(),
                "archive_access": "retained_secondary_reference_not_primary_extractor_input",
                "primary_entry_access": [{"entry_id": e.entry_id, "source_session_id": e.session_id,
                    "source_seqs": [s.seq for s in e.spans], "access": "observed_in_current_native_session",
                    "applicability": "unknown_unless_separately_supported"} for e in primary.entries],
                "shared_call_record_ids": sorted(set(primary.call_record_ids) & set(archive.call_record_ids))}
            _put(store, "observer_scope_pair_v1", pair_id, pair, run_id)
            return {**base, "status": "complete", "retryable": False, "pair_id": pair_id, "error": None}
        except Exception as exc:
            return {**base, "status": "uncertain" if isinstance(exc, (OSError, sqlite3.OperationalError, _Uncertain)) else "failed",
                    "retryable": False, "error_type": type(exc).__name__, "error": str(exc)}
