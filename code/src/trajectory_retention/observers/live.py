"""Durable, worker-blind ordered observation of immutable checkpoints.

The checkpoint table is the durable outbox. Notifications are merely wakeups:
missed callbacks or a process restart cannot lose an official checkpoint. Each
consumer holds a process lock for its immutable run/namespace binding. No worker
thread calls an extractor, waits for its result, or receives its output.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import math
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

from ..contracts import Checkpoint, canonical_json, to_jsonable, utc_now
from ..persistence import persist
from ._canonical import content_id
from .artifacts import load_primary
from .extractor import ExtractorConfig

REVISION = "live-checkpoint-observer.v1"
VALID = {"valid", "valid_with_rejections"}


@dataclass(frozen=True)
class LiveObserverPolicy:
    max_attempts: int = 3
    retry_delay_seconds: float = 1.0
    failure_threshold: int = 3
    poll_interval_seconds: float = .05
    drain_timeout_seconds: float = 30.0
    # Declared number of additional attempts after an "uncertain" outcome whose
    # only uncertainty is unknown provider usage (timeout / dropped connection
    # with a finished receipt). The earlier receipts stay recorded with unknown
    # usage; a repeat never rewrites or zeroes them. 0 keeps the historical
    # "one uncertain result ends this queue" behavior.
    uncertain_retries: int = 0

    def __post_init__(self):
        for name in ("max_attempts", "failure_threshold"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.uncertain_retries) is not int or self.uncertain_retries < 0:
            raise ValueError("uncertain_retries must be a nonnegative integer")
        for name in ("retry_delay_seconds", "poll_interval_seconds", "drain_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if not self.poll_interval_seconds:
            raise ValueError("poll_interval_seconds must be positive")


def _identity(run_id, namespace):
    return content_id({"revision": REVISION, "run_id": run_id, "namespace": namespace})


def _records(store, kind, run_id=None):
    return [store.get_record(kind, item["record_id"]) for item in store.list_records(kind, run_id=run_id)]


def observer_circuits(store, run_ids=None):
    """Durable circuit receipts, optionally restricted to the given natural runs."""
    selected = None if run_ids is None else set(run_ids)
    return [row for row in _records(store, "live_observer_circuit_v1")
            if selected is None or row["run_id"] in selected]


def observer_circuit_open(store, run_ids=None):
    return bool(observer_circuits(store, run_ids))


def circuit_is_systemic(receipt) -> bool:
    """A consumer exception is an instrument failure; provider/queue outcomes are per-run."""
    reason = str(receipt.get("reason", ""))
    return reason.startswith("observer_consumer_exception")


def live_observer_status(store, run_id, namespace=None):
    bindings = _records(store, "live_observer_binding_v1", run_id)
    if namespace is None:
        if len(bindings) != 1:
            return {"run_id": run_id, "status": "not_configured" if not bindings else "namespace_required",
                    "complete": False, "missing_checkpoint_ids": [cp.checkpoint_id for cp in store.list_checkpoints(run_id)]}
        namespace = bindings[0]["namespace"]
    # Accept the consumer's queue namespace, never silently strip an arbitrary
    # suffix that could name a different source/configuration chain.
    identity = _identity(run_id, namespace)
    states = [row for row in _records(store, "live_observer_state_v1", run_id) if row["queue_id"] == identity]
    latest = {}
    for row in states:
        if row["checkpoint_id"] not in latest or row["sequence"] > latest[row["checkpoint_id"]]["sequence"]:
            latest[row["checkpoint_id"]] = row
    cps = store.list_checkpoints(run_id)
    primary_ns, archive_ns = f"{namespace}:primary-session-v1", f"{namespace}:archive-global-v1"
    rows, versions, counts = [], [], {key: 0 for key in ("queued", "running", "retrying", "failed", "uncertain", "complete")}
    for cp in cps:
        current = latest.get(cp.checkpoint_id, {})
        status = current.get("status", "queued")
        primary = load_primary(store, run_id, cp.checkpoint_id, chain_namespace=primary_ns)
        archive = load_primary(store, run_id, cp.checkpoint_id, chain_namespace=archive_ns)
        valid_primary = primary is not None and primary.status in VALID
        valid_archive = archive is not None and archive.status in VALID
        if valid_primary:
            versions.append(primary.version_id())
        # A pointer is useful coverage, but full job completion also certifies
        # both scopes, ordered lineage, and the durable step completion receipt.
        if status == "complete" and not (valid_primary and valid_archive):
            status = "uncertain"
        counts[status] = counts.get(status, 0) + 1
        rows.append({"checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
                     "event_cut": cp.event_sequence, "native_session_id": cp.scope.session_id,
                     "status": status, "attempt_index": current.get("attempt_index", 0),
                     "primary_version_id": primary.version_id() if valid_primary else None,
                     "archive_version_id": archive.version_id() if valid_archive else None,
                     "primary_empty": not primary.entries if valid_primary else None,
                     "archive_empty": not archive.entries if valid_archive else None,
                     "blocked_on_checkpoint_id": current.get("blocked_on_checkpoint_id"),
                     "retry_after_seconds": current.get("retry_after_seconds"),
                     "repeat_uncertain": bool(current.get("repeat_uncertain")),
                     "error_type": current.get("error_type"), "reason": current.get("reason")})
    circuits = [r for r in _records(store, "live_observer_circuit_v1", run_id) if r["queue_id"] == identity]
    costs = [store.get_record("observer_call", r["record_id"]) for r in store.list_records("observer_call", run_id=run_id)]
    received = {row["record_id"]: store.get_record("live_t_call_received_v1", row["record_id"])
                for row in store.list_records("live_t_call_received_v1", run_id=run_id)}
    unfinished = [row["record_id"] for row in store.list_records("live_t_call_started_v1", run_id=run_id)
                  if row["record_id"] not in received]
    allocated = {row["call_record_id"] for row in costs}
    unallocated = [row["receipt"]["call_record_id"] for row in received.values()
                   if row["receipt"]["call_record_id"] not in allocated]
    return {"revision": REVISION, "run_id": run_id, "queue_namespace": namespace, "namespace": primary_ns,
            "primary_namespace": primary_ns, "archive_namespace": archive_ns,
            "checkpoint_count": len(cps), "complete": bool(cps) and counts["complete"] == len(cps),
            "status": "complete" if cps and counts["complete"] == len(cps) else "incomplete",
            "counts": counts, **{key: value for key, value in counts.items() if key != "complete"},
            "complete_count": counts["complete"], "checkpoints": rows,
            "version_ids": versions, "statuses": [r["status"] for r in rows],
            "missing_checkpoint_ids": [r["checkpoint_id"] for r in rows if r["status"] != "complete"],
            "circuit_open": bool(circuits), "circuits": circuits,
            "observer_call_record_ids": [r["call_record_id"] for r in costs],
            "observer_unfinished_attempt_ids": unfinished,
            "observer_unallocated_receipt_ids": unallocated,
            "observer_cost_usd_missing_call_ids": [r["call_record_id"] for r in costs
                if r.get("usage") is None or r["usage"].get("cost_usd") is None],
            "observer_receipt_coverage_complete": not unfinished and not unallocated,
            "observer_usage_missing_call_ids": [r["call_record_id"] for r in costs
                if r.get("usage") is None or r["usage"].get("generated_tokens") is None],
            "cost_scope": "All observer roles in this run; exact once-charged allocation is in CostLedger. Unknown usage is not zero.",
            "worker_waits_for_t": False, "worker_receives_t": False}


class LiveObserverService:
    def __init__(self, store, run_id: str, client, extractor: ExtractorConfig, *, namespace: str,
                 include_returned_reasoning: bool, policy: LiveObserverPolicy,
                 context_counter=None, on_circuit_open: Callable | None = None):
        if not isinstance(namespace, str) or not namespace.strip() or type(include_returned_reasoning) is not bool:
            raise ValueError("Live T requires an explicit namespace and returned-reasoning source choice")
        self.store, self.run_id, self.client, self.extractor = store, run_id, client, extractor
        self.namespace, self.policy = namespace, policy
        self.include_returned_reasoning, self.context_counter = include_returned_reasoning, context_counter
        self.on_circuit_open = on_circuit_open
        self.queue_id = _identity(run_id, namespace)
        self._wake, self._stop, self._finished = Event(), Event(), Event()
        self._thread, self._lock_file = None, None
        self._failure, self._uncertain_checked = None, set()

    def ready(self):
        return self._thread is not None and self._lock_file is not None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Service already started; create an explicit resume consumer")
        # Lock before creating any execution claim. flock is released by process
        # death, unlike a stale lease which could overlap a still-paid request.
        lock_dir = Path(self.store.root) / "live-observer-locks"
        lock_dir.mkdir(exist_ok=True, mode=0o700)
        handle = (lock_dir / self.queue_id).open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise RuntimeError("Live observer queue already has an active consumer") from None
        self._lock_file = handle
        try:
            binding = {"revision": REVISION, "queue_id": self.queue_id, "run_id": self.run_id,
                       "namespace": self.namespace, "extractor": to_jsonable(self.extractor),
                       "include_returned_reasoning": self.include_returned_reasoning,
                       "policy": to_jsonable(self.policy),
                       "context_counter": self.context_counter.as_record() if self.context_counter else None}
            try:
                existing = self.store.get_record("live_observer_binding_v1", self.queue_id)
            except KeyError:
                persist(lambda: self.store.put_record("live_observer_binding_v1", self.queue_id, binding, run_id=self.run_id), evidence=binding)
            else:
                # Old direct callers could persist 60 while JSON dataclass
                # loading produced 60.0. Normalize only declared duration
                # fields; all model/source/identity bytes remain exact. Keep
                # the original immutable binding, never rewrite its receipt.
                normalized = {**existing, "policy": dict(existing["policy"])}
                # Queues bound before the declared uncertain-retry field carried
                # the historical default; the retained receipt is not rewritten.
                normalized["policy"].setdefault("uncertain_retries", 0)
                for key in ("retry_delay_seconds", "poll_interval_seconds", "drain_timeout_seconds"):
                    value = normalized["policy"][key]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("Invalid retained observer duration binding")
                    normalized["policy"][key] = float(value)
                if canonical_json(normalized) != canonical_json(binding):
                    raise ValueError("Live observer queue configuration changed; use an explicit new namespace")
            self._reconcile()
            self._thread = Thread(target=self._work, name="live-observer-" + self.queue_id[:12], daemon=True)
            self._thread.start()
        except BaseException:
            self._release()
            raise
        return self

    def checkpoint_published(self, checkpoint: Checkpoint):
        # Invoked from the worker thread: NO store reads/writes, provider calls,
        # queue backpressure or result waits. The checkpoint itself is durable.
        self._wake.set()

    def _release(self):
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def _state(self, cp, status, attempt_index=0, **details):
        states = [r for r in _records(self.store, "live_observer_state_v1", self.run_id)
                  if r["queue_id"] == self.queue_id]
        value = {"queue_id": self.queue_id, "run_id": self.run_id, "checkpoint_id": cp.checkpoint_id,
                 "sequence": 1 + max((r["sequence"] for r in states), default=-1),
                 "status": status, "attempt_index": attempt_index, "recorded_at": utc_now(), **details}
        persist(lambda: self.store.put_record("live_observer_state_v1", content_id(value), value, run_id=self.run_id), evidence=value)
        return value

    def _uncertain_repeats(self, cp):
        """Declared uncertain repeats already scheduled for this checkpoint on this queue."""
        return sum(r.get("status") == "retrying" and r.get("repeat_uncertain") is True
                   and r["checkpoint_id"] == cp.checkpoint_id
                   for r in _records(self.store, "live_observer_state_v1", self.run_id) if r["queue_id"] == self.queue_id)

    def _reconcile(self):
        cps = self.store.list_checkpoints(self.run_id)
        for index, cp in enumerate(cps):
            if cp.ordinal != index or (index and cp.parent_checkpoint_id != cps[index - 1].checkpoint_id):
                raise ValueError("Live observer requires the contiguous immutable official checkpoint chain")
            value = {"revision": REVISION, "queue_id": self.queue_id, "run_id": self.run_id,
                     "checkpoint_id": cp.checkpoint_id, "ordinal": cp.ordinal,
                     "checkpoint_sha256": content_id(to_jsonable(cp)), "event_cut": cp.event_sequence,
                     "native_session_id": cp.scope.session_id,
                     "predecessor_checkpoint_id": cps[index - 1].checkpoint_id if index else None}
            job_id = content_id({"queue_id": self.queue_id, "checkpoint_id": cp.checkpoint_id})
            try:
                existing = self.store.get_record("live_observer_job_v1", job_id)
            except KeyError:
                persist(lambda: self.store.put_record("live_observer_job_v1", job_id, value, run_id=self.run_id), evidence=value)
            else:
                if existing != value:
                    raise ValueError("Observer job source binding changed")
        return cps

    def _circuit(self, reason):
        value = {"revision": REVISION, "queue_id": self.queue_id, "run_id": self.run_id,
                 "namespace": self.namespace, "reason": reason,
                 "policy": "stop_new_mines_only; active_workers_and_ordered_observation_continue"}
        # Latched once per queue, with immutable triggering reason.
        try:
            existing = self.store.get_record("live_observer_circuit_v1", self.queue_id)
        except KeyError:
            persist(lambda: self.store.put_record("live_observer_circuit_v1", self.queue_id, value, run_id=self.run_id), evidence=value)
        else:
            value = existing
        if self.on_circuit_open is not None:
            self.on_circuit_open(value)

    def _work(self):
        from .live_step import advance_checkpoint
        try:
            while not self._stop.is_set():
                cps = self._reconcile()
                report = self.status()
                pending = next((row for row in report["checkpoints"] if row["status"] != "complete"), None)
                if pending is None:
                    if self._finished.is_set():
                        break
                    self._wake.wait(self.policy.poll_interval_seconds); self._wake.clear()
                    continue
                cp = next((cp for cp in cps if cp.checkpoint_id == pending["checkpoint_id"]), None)
                if cp is None:
                    # The worker published after the outbox scan but before
                    # the status read. Reconcile that new durable job first.
                    continue
                attempt = pending["attempt_index"]
                if pending["status"] == "failed" or (pending["status"] == "uncertain" and cp.checkpoint_id in self._uncertain_checked):
                    if self._finished.is_set():
                        break
                    self._wake.wait(self.policy.poll_interval_seconds); self._wake.clear()
                    continue
                repeat_uncertain = False
                if pending["status"] == "retrying":
                    retry_after = pending.get("retry_after_seconds") or 0
                    if self._stop.wait(max(self.policy.retry_delay_seconds, retry_after)):
                        break
                    attempt += 1
                    repeat_uncertain = bool(pending.get("repeat_uncertain"))
                self._state(cp, "running", attempt, repeat_uncertain=repeat_uncertain)
                result = advance_checkpoint(self.store, self.run_id, cp.checkpoint_id, self.client, self.extractor,
                    namespace=self.namespace, include_returned_reasoning=self.include_returned_reasoning,
                    context_counter=self.context_counter, attempt_index=attempt,
                    allow_uncertain_repeat=repeat_uncertain)
                status = result["status"]
                if status == "failed" and result.get("retryable"):
                    status = "retrying"
                if (status == "uncertain" and result.get("repeatable_unknown_usage")
                        and self._uncertain_repeats(cp) < self.policy.uncertain_retries
                        and attempt + 1 < self.policy.max_attempts):
                    # Declared repeat: the finished receipt has unknown usage,
                    # which stays recorded as unknown. Nothing is reissued for
                    # attempts whose provider outcome is not durably known.
                    status, repeat_uncertain = "retrying", True
                else:
                    repeat_uncertain = False
                if status == "complete":
                    self._state(cp, status, attempt, primary_version_id=result.get("primary_version_id"),
                                archive_version_id=result.get("archive_version_id"))
                elif result.get("blocked_on_checkpoint_id"):
                    self._state(cp, "queued", attempt, blocked_on_checkpoint_id=result["blocked_on_checkpoint_id"])
                    self._circuit("invalid_predecessor_binding")
                    break
                else:
                    if status not in {"retrying", "failed", "uncertain"}:
                        raise ValueError("Unexpected observer step status")
                    if status == "retrying" and attempt + 1 >= self.policy.max_attempts:
                        status = "failed"
                    self._state(cp, status, attempt, reason=result.get("error", result.get("reason")),
                                error_type=result.get("error_type"), retry_after_seconds=result.get("retry_after_seconds"),
                                repeat_uncertain=repeat_uncertain)
                    states = [r for r in _records(self.store, "live_observer_state_v1", self.run_id) if r["queue_id"] == self.queue_id]
                    last_complete = max((r["sequence"] for r in states if r["status"] == "complete"), default=-1)
                    failures = sum(r["sequence"] > last_complete and r["status"] in {"retrying", "failed", "uncertain"} for r in states)
                    if failures >= self.policy.failure_threshold or status in {"failed", "uncertain"}:
                        self._circuit("observer_" + status)
                    if status == "uncertain":
                        self._uncertain_checked.add(cp.checkpoint_id)
        except BaseException as exc:
            self._failure = type(exc).__name__
            try:
                self._circuit("observer_consumer_exception:" + type(exc).__name__)
            except BaseException:
                pass  # Existing start/received records remain the restart boundary.
        finally:
            self._release()

    def status(self):
        report = live_observer_status(self.store, self.run_id, self.namespace)
        if self._failure:
            report = {**report, "consumer_error_type": self._failure, "complete": False, "status": "incomplete"}
        return report

    def finish(self):
        self._finished.set(); self._wake.set()
        if self._thread is not None:
            self._thread.join(self.policy.drain_timeout_seconds)
            if self._thread.is_alive():
                self._circuit("observer_drain_deadline_incomplete")
                self._stop.set(); self._wake.set()
        report = self.status()
        # An in-flight provider request retains the lock and its durable start;
        # do not pretend thread timeout cancels its charge or manufacture a retry.
        report["consumer_running"] = bool(self._thread and self._thread.is_alive())
        record = {**report, "recorded_at": utc_now()}
        persist(lambda: self.store.put_record("live_observer_coverage_v1", content_id(record), record, run_id=self.run_id), evidence=record)
        return report

    def stop(self):
        self._stop.set(); self._wake.set()
        if self._thread is not None:
            self._thread.join(self.policy.drain_timeout_seconds)
        return self.status()
