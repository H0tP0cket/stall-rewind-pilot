"""Recording mini-SWE-agent worker.

``RecordingAgent`` subclasses the pinned ``DefaultAgent`` (mini-swe-agent 2.4.6)
and preserves its query/action/observation semantics and exceptions. Its shared
prompt specifies interface mechanics; task adapters supply benchmark instructions.
It adds:

* exhaustive event emission at the model and tool boundaries,
* per-attempt accounting through the shared cost ledger,
* an execution cursor (message count + pending actions) so a checkpoint records
  exactly where execution stands and a restore never repeats completed actions,
* construction from restored history/counters instead of a fresh task.

The wrapper records intermediate tool mutations; these are not branch points.
The runner drives ``step()`` and performs grading, checkpointing and detection
between complete steps.
"""

from __future__ import annotations

import json
from copy import deepcopy
import sys
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded

from ..accounting import CostLedger
from ..contracts import Usage, Visibility, new_id, to_jsonable
from ..persistence import preserve_failure
from .context_limit import EXIT_STATUS as CONTEXT_EXIT_STATUS, MASK_KEY, ContextLimit, fit as fit_context_limit
from .environment import CaptureError, DockerWorkerEnvironment, ExecResult, ToolTransportFailure, comparable
from .model import ModelAbort, _BaseModel
from .recorder import Recorder

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n"
# Shared tool interface only. Benchmark instructions are built by task adapters.
# Exact rendered messages and this version are retained in every fresh history.
INSTANCE_TEMPLATE = """{{task}}

Tool and submission rules:
- Use the bash tool to run shell commands and edit files. Each response must include at least one bash tool call.
- Each command runs in a fresh subshell. Files persist; shell directory changes and environment variables do not persist between calls.
- To submit the current task or stage, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone. Submission ends work on the current task or stage.
"""
VERIFICATION_INSTANCE_TEMPLATE = INSTANCE_TEMPLATE + """
Before submitting, compare your implementation against every stated requirement of the current task or stage, including required existing behavior. Use the tests, examples and other evidence available to you to check correctness and edge cases. A small smoke test does not establish that the whole task is complete. If you find a concrete unmet requirement, continue working on it within your remaining allowance. Submit when you believe the complete task is satisfied, or when you decide you cannot make further progress; do not perform repetitive actions merely to spend the allowance.
"""
VERIFICATION_PROMPT_VERSION = "requirements-verification/v1"
HARNESS_VERSION = "mini-swe-agent-2.4.6/recording-agent/v2-family-prompts"
# Per-tool state recording. "full" (default, unchanged): a pre-tool scan and a full W/R capture after every action.
# "per_action_manifest" (v2 natural runs): a W fidelity + R manifest scan after every executed action except the last
# of its batch; the runner's step-end capture closes the last action, so a create/delete or edit/revert across calls
# stays visible. "none" (v2 branches): no per-tool state; the event log is otherwise unchanged.
TOOL_STATE_POLICIES = ("full", "per_action_manifest", "none")
PER_ACTION_MANIFEST_RULE = ("per-action-manifest/v1: after each executed action except the last of its batch, the W "
                            "fidelity manifest and the R manifest (docker diff entries, deleted list, mounted state); "
                            "the step-end checkpoint capture is the last action's after-state")


class WorkerBudgetExhausted(LimitsExceeded):
    pass


class WorkerAccountingInterrupted(LimitsExceeded):
    pass


class ContextWindowExhausted(LimitsExceeded):
    """The next request does not fit the context window even with every tool output hidden (context_limit.py)."""


class RecordingAgent(DefaultAgent):
    def __init__(self, model: _BaseModel, env: DockerWorkerEnvironment, *, recorder: Recorder, ledger: CostLedger,
                 generated_token_limit: int, max_model_calls: int | None = None,
                 system_template: str = SYSTEM_TEMPLATE, instance_template: str = INSTANCE_TEMPLATE,
                 max_consecutive_format_errors: int = 3) -> None:
        super().__init__(model, env, config_class=AgentConfig, system_template=system_template,
                         instance_template=instance_template, step_limit=max_model_calls or 0, cost_limit=0.0,
                         max_consecutive_format_errors=max_consecutive_format_errors)
        self.recorder = recorder
        self.ledger = ledger
        self.generated_token_limit = generated_token_limit
        self.known_generated_tokens = 0
        self.unknown_usage_calls = 0
        self.generated_token_offset = 0
        self.unknown_usage_offset = 0
        self._accounting_incident_recorded = False
        self.call_offset = 0  # historical calls restored from a checkpoint; not charged to this run's call cap
        self.pending_actions: list[dict[str, Any]] = []
        self.completed_in_batch: list[dict[str, Any]] = []
        self.last_usage: Usage | None = None
        self.last_response_event_sequence: int | None = None
        self.usage_log: list[tuple[int, Usage]] = []
        self.observation_refs: list[Any] = []
        self.batch_mutated = False
        self.batch_state_events = []
        self.batch_lap_observations = []
        self.tool_state_policy = "full"
        self.batch_action_states: list[tuple[int, dict, Any]] = []
        # Provider-reported cost of every finished attempt of this agent (failed attempts included), and the count
        # of finished attempts whose cost is unknown. Never restored from a checkpoint: it is this run's spend.
        self.attempt_cost_known = 0.0
        self.attempt_cost_unknown = 0
        # Charged-retry rule (provider_policy.worst_case_charge): unresolved attempts that were retried. Their usage
        # stays unknown in the ledger; the guards count their worst-case bound instead of stopping the run (Q8).
        self.attempt_cost_worst_case = 0.0  # this run's spend, never restored, like attempt_cost_known
        self.charged_unknown_attempts = 0
        self.worst_case_generated_tokens = 0  # bound charged against the token allowance; restored like known tokens
        self.tool_execution_uncertain = False
        # v2 branches: a one-shot intervention turn ("review" / "note") marks its model events with payload mode;
        # the continuation agent carries the turn's model calls so the branch step guard counts them.
        self.turn_mode: str | None = None
        self.intervention_calls = 0
        self.last_request_ref = None
        self._open_attempts: dict[tuple[str, int], tuple[str, dict[str, Any]]] = {}
        self._received_attempts: dict[tuple[str, int], dict[str, Any]] = {}
        self.control_handler = None  # Explicit recovery baseline only; natural tool interface stays unchanged.
        # The context-limit rule (execution/context_limit.py), bound by the runner for v2 runs; None leaves every
        # request unchanged. The ratio is the provider-to-local prompt-token ratio of the latest request of this
        # context with a provider prompt count (the anchor); it only sizes requests and never changes accounting.
        self.context_limit: ContextLimit | None = None
        self.context_ratio: float | None = None
        self.context_anchor: dict[str, Any] | None = None
        self._sent_local_tokens: int | None = None

    # ------------------------------------------------------------- lifecycle
    def harness_binding(self) -> dict[str, Any]:
        return {"harness_version": HARNESS_VERSION, "agent": self.config.model_dump(mode="json"),
                "model": self.model.config, "environment": self.env.binding(),
                "generated_token_limit": self.generated_token_limit}

    def counters(self) -> dict[str, Any]:
        return {"n_calls": self.n_calls, "calls_this_run": self.n_calls - self.call_offset, "cost": self.cost,
                "cost_basis": "SDK known accepted-response sum only; complete costs and missingness are in CostLedger",
                "known_generated_tokens": self.known_generated_tokens,
                "generated_token_offset": self.generated_token_offset,
                "known_generated_tokens_this_run": self.known_generated_tokens - self.generated_token_offset,
                "unknown_usage_offset": self.unknown_usage_offset,
                "unknown_usage_calls_this_run": self.unknown_usage_calls - self.unknown_usage_offset,
                "unknown_usage_calls": self.unknown_usage_calls, "n_consecutive_format_errors": self.n_consecutive_format_errors,
                "charged_unknown_attempts": self.charged_unknown_attempts,
                "worst_case_generated_tokens": self.worst_case_generated_tokens,
                "elapsed_seconds": int(time.time() - self._start_time)}

    def cursor(self) -> dict[str, Any]:
        return {"message_count": len(self.messages), "pending_actions": list(self.pending_actions),
                "completed_in_batch": list(self.completed_in_batch)}

    def start(self, task: str, **template_vars: Any) -> None:
        """Fresh conversation: system + instance messages, both recorded as disclosures."""
        self.extra_template_vars |= {"task": task, **template_vars}
        self.messages = []
        system = self.model.format_message(role="system", content=self._render_template(self.config.system_template))
        user = self.model.format_message(role="user", content=self._render_template(self.config.instance_template))
        self.add_messages(system, user)
        for i, m in enumerate(self.messages):
            self.recorder.emit("task_disclosure", payload={"role": m["role"], "message_index": i},
                               delivered_ref=self.recorder.text_ref(m["content"], Visibility.WORKER_VISIBLE))

    def append_user_message(self, content: str, *, kind: str = "task_disclosure", payload: dict[str, Any] | None = None) -> None:
        msg = self.model.format_message(role="user", content=content)
        self.add_messages(msg)
        self.recorder.emit(kind, payload={"role": "user", "message_index": len(self.messages) - 1, **(payload or {})},
                           delivered_ref=self.recorder.text_ref(content, Visibility.WORKER_VISIBLE))

    def restore(self, messages: list[dict[str, Any]], counters: dict[str, Any], cursor: dict[str, Any], *, task: str) -> None:
        """Adopt recorded history and counters; refuse a mid-batch cursor (not yet supported)."""
        if cursor.get("pending_actions"):
            raise ValueError("restoring a mid-batch execution position is not supported; checkpoint is not resumable here")
        if cursor.get("message_count") != len(messages):
            raise ValueError("cursor/message count mismatch")
        self.extra_template_vars |= {"task": task}
        self.messages = [dict(m) for m in messages]
        self.pending_actions = []
        self.completed_in_batch = deepcopy(cursor.get("completed_in_batch", []))
        # O is the addressable tool-content substream of the actual restored
        # H. Reconstruct from those bytes (including mask placeholders), not
        # a new empty session or an unverified legacy observation manifest.
        self.observation_refs = [self.recorder.put_bytes(m["content"].encode("utf-8"),
            media_type="text/plain; charset=utf-8", visibility=Visibility.WORKER_VISIBLE)
            for m in self.messages if m.get("role") == "tool" and isinstance(m.get("content"), str)]
        self.n_calls = int(counters.get("n_calls", 0))
        self.call_offset = self.n_calls
        self.cost = float(counters.get("cost", 0.0))
        self.known_generated_tokens = int(counters.get("known_generated_tokens", 0))
        self.unknown_usage_calls = int(counters.get("unknown_usage_calls", 0))
        self.worst_case_generated_tokens = int(counters.get("worst_case_generated_tokens", 0))
        self.generated_token_offset = self.known_generated_tokens
        self.unknown_usage_offset = self.unknown_usage_calls
        self.n_consecutive_format_errors = int(counters.get("n_consecutive_format_errors", 0))

    # ------------------------------------------------------------------ query
    def budget_exhausted(self) -> bool:
        return self.known_generated_tokens + self.worst_case_generated_tokens >= self.generated_token_limit

    def accounting_interrupted(self) -> bool:
        return self.unknown_usage_calls > self.unknown_usage_offset

    def _remaining_allowance(self) -> int:
        if self.accounting_interrupted():
            if not self._accounting_incident_recorded:
                self.recorder.incident("worker_accounting_interruption",
                    unknown_usage_calls_this_run=self.unknown_usage_calls - self.unknown_usage_offset,
                    known_generated_tokens_this_run=self.known_generated_tokens - self.generated_token_offset,
                    further_requests_permitted=False)
                self._accounting_incident_recorded = True
            raise WorkerAccountingInterrupted({"role": "exit", "content": "WorkerAccountingInterrupted",
                "extra": {"exit_status": "WorkerAccountingInterrupted", "submission": ""}})
        if self.budget_exhausted():
            raise WorkerBudgetExhausted({"role": "exit", "content": "WorkerBudgetExhausted",
                "extra": {"exit_status": "WorkerBudgetExhausted", "submission": ""}})
        return self.generated_token_limit - self.known_generated_tokens - self.worst_case_generated_tokens

    def query(self) -> dict:
        self.batch_mutated = False
        self.batch_state_events = []
        self.batch_lap_observations = []
        self.batch_action_states = []
        if self.tool_execution_uncertain:
            raise RuntimeError("Uncertain tool execution prevents further worker requests")
        if self.recorder.capture_failure is not None:
            raise RuntimeError("Unresolved capture failure prevents further worker requests")
        if self.recorder.persistence_failure is not None or self._received_attempts:
            raise RuntimeError("Unresolved persistence failure prevents further worker requests")
        self._remaining_allowance()
        if 0 < self.config.step_limit <= self.n_calls - self.call_offset:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded", "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        fitted = self.fit_context()  # raises ContextWindowExhausted before any call is counted or sent
        self.n_calls += 1
        call_id = new_id("call")
        mode = {"mode": self.turn_mode} if self.turn_mode else {}
        self.model.allowance_remaining = self._remaining_allowance
        try:
            prepared = self.model.prepare_request(self.messages)
            limit_record = {}
            if fitted is not None:
                # The rule is checked on the actual serialized request: the body sent is the body that was fitted.
                if prepared.body != fitted["body"]:
                    raise RuntimeError("The serialized request differs from the request the context-limit rule fitted")
                self._sent_local_tokens = fitted["after"]["local_tokens"]
                limit_record = {"context_limit": {
                    "window_tokens": fitted["window_tokens"], "ratio": fitted["ratio"], "size": fitted["size_after"],
                    "fits": fitted["fits"], "local_tokens": fitted["after"]["local_tokens"],
                    "response_cap": fitted["after"]["response_cap"], "outputs_hidden_now": len(fitted["masked"]),
                    "outputs_hidden_total": sum(1 for m in self.messages if MASK_KEY in (m.get("extra") or {}))}}
            request_ref = self.recorder.put_bytes(prepared.body, media_type="application/json",
                                                    visibility=Visibility.WORKER_VISIBLE)
            self.last_request_ref = request_ref
            req_event = self.recorder.emit("model_request", call_id=call_id, input_ref=request_ref, actor="worker",
                                       payload={"message_count": len(self.messages), "model": self.model.model_name,
                                                "settings": self.model.model_settings, **mode, **limit_record})
        except BaseException:
            self.model.allowance_remaining = None
            raise
        self.model.attempt_sink = lambda phase, rec: self._record_attempt(call_id, req_event.event_id, phase, rec)
        try:
            message = self.model.query(self.messages, prepared_request=prepared)
        except FormatError as e:
            extra = e.messages[0]["extra"]
            usage = extra["usage"]
            resp = self.recorder.emit("model_response", call_id=call_id, parent_event_id=req_event.event_id, actor="worker", usage=usage,
                                      output_ref=self.recorder.put_json(extra["normalized"], visibility=Visibility.WORKER_VISIBLE),
                                      payload={"accepted": False, "format_error": True,
                                               "note": "response was not appended to history; only the format error message was delivered",
                                               **mode})
            self.recorder.emit("format_error", call_id=call_id, parent_event_id=req_event.event_id,
                               delivered_ref=self.recorder.text_ref(e.messages[0]["content"], Visibility.WORKER_VISIBLE),
                               payload={"rejected_message_ref": self.recorder.put_json(extra["rejected_message"]).sha256,
                                        **mode})
            self.cost += extra.get("cost") or 0.0
            self._anchor(usage, call_id, resp.sequence)
            # History keeps only the delivered error message plus slim provenance; raw bytes live in the store.
            e.messages[0]["extra"] = {"interrupt_type": "FormatError", "call_id": call_id, "cost": extra.get("cost"),
                                      "response_event_sequence": resp.sequence}
            raise
        except ModelAbort as e:
            try:
                self.recorder.incident("model_call_failed", call_id=call_id, error=str(e), attempts=len(e.attempts))
            except BaseException as cleanup:
                e.add_note(f"Failure incident persistence also failed: {type(cleanup).__name__}")
            if self.accounting_interrupted():
                self._remaining_allowance()
            raise
        finally:
            # Interruptions can happen after a request leaves the process but
            # before a response/usage receipt arrives. Preserve uncertainty;
            # never turn such a request into complete zero spend.
            primary = sys.exception()
            try:
                for (pending_call, number), (parent, receipt) in list(self._open_attempts.items()):
                    if pending_call == call_id and (pending_call, number) not in self._received_attempts:
                        try:
                            self._record_attempt(pending_call, parent, "finished", {
                                **receipt, "attempt": number, "status": "interrupted",
                                "error": "Attempt started without a completed provider receipt; usage unknown"})
                        except BaseException as cleanup:
                            if primary is None:
                                raise
                            primary.add_note(f"Attempt cleanup also failed: {type(cleanup).__name__}")
            finally:
                self.model.attempt_sink = None
                self.model.allowance_remaining = None
        extra = message["extra"]
        usage: Usage = extra["usage"]
        self.last_usage = usage
        resp = self.recorder.emit("model_response", call_id=call_id, parent_event_id=req_event.event_id, actor="worker", usage=usage,
                                  output_ref=self.recorder.put_json(extra["normalized"], visibility=Visibility.WORKER_VISIBLE),
                                  payload={"accepted": True, "format_error": False, "n_actions": len(extra["actions"]),
                                           "finish_reason": extra["normalized"]["finish_reason"], **mode})
        self.last_response_event_sequence = resp.sequence
        self.cost += extra.get("cost") or 0.0
        self._anchor(usage, call_id, resp.sequence)
        # History keeps the provider message verbatim; the recorder keeps the raw bytes separately.
        stored = dict(message)
        stored["extra"] = {"actions": extra["actions"], "cost": extra["cost"], "timestamp": extra["timestamp"],
                           "call_id": call_id, "response_event_sequence": resp.sequence}
        self.add_messages(stored)
        self.pending_actions = list(extra["actions"])
        self.completed_in_batch = []
        return stored

    # --------------------------------------------------------- context limit
    def _anchor(self, usage: Usage | None, call_id: str, sequence: int) -> None:
        """A provider prompt count for the request just sent becomes the size anchor of the next requests."""
        local, self._sent_local_tokens = self._sent_local_tokens, None
        if self.context_limit is None or local is None or local <= 0 or usage is None or usage.input_tokens is None:
            return
        self.context_ratio = usage.input_tokens / local
        self.context_anchor = {"source": "this_run", "call_id": call_id, "response_event_sequence": sequence,
                               "provider_prompt_tokens": usage.input_tokens, "local_tokens": local}

    def fit_context(self) -> dict[str, Any] | None:
        """Apply the context-limit rule to the next request: hide the oldest tool outputs until it fits, and log it.

        Returns the fit (None when no rule is bound). Hidden outputs replace their messages in the context, so every
        later checkpoint carries them, and O follows the context. Raises ContextWindowExhausted when the request does
        not fit with every tool output hidden; then nothing is changed or sent."""
        if self.context_limit is None:
            return None
        from ..experiments.context_edit import steps_of
        saved = self.model.allowance_remaining
        self.model.allowance_remaining = self._remaining_allowance  # serialized exactly as the call will send it
        put = lambda text: to_jsonable(self.recorder.put_bytes(text.encode("utf-8"),
            media_type="text/plain; charset=utf-8", visibility=Visibility.WORKER_VISIBLE))
        try:
            result = fit_context_limit(self.messages, self.context_limit,
                                       lambda ms: self.model.prepare_request(ms).body,
                                       ratio=self.context_ratio if self.context_ratio is not None else 1.0,
                                       put_original=put, step=self.n_calls + 1, context_steps=steps_of)
        finally:
            self.model.allowance_remaining = saved
        summary = {"rule": self.context_limit.binding()["rule"], "step": self.n_calls + 1,
                   "context_point": len(steps_of(self.messages)) - 1,  # for a turn, its source point
                   "window_tokens": result["window_tokens"], "ratio": result["ratio"], "anchor": self.context_anchor,
                   "tokens_before": {"local": result["before"]["local_tokens"], "size": result["size_before"],
                                     "response_cap": result["before"]["response_cap"]},
                   "tokens_after": {"local": result["after"]["local_tokens"], "size": result["size_after"],
                                    "response_cap": result["after"]["response_cap"]},
                   "fits": result["fits"], **({"mode": self.turn_mode} if self.turn_mode else {})}
        if not result["fits"]:
            self.recorder.emit("lifecycle", payload={"transition": "context_window_exhausted", **summary,
                "tool_outputs_hideable": len(result["masked"]),
                "note": "does not fit with every tool output hidden; nothing was changed or sent"})
            raise ContextWindowExhausted({"role": "exit", "content": CONTEXT_EXIT_STATUS,
                                          "extra": {"exit_status": CONTEXT_EXIT_STATUS, "submission": ""}})
        if result["masked"]:
            tool_ordinal = [i for i, m in enumerate(self.messages)
                            if m.get("role") == "tool" and isinstance(m.get("content"), str)]
            aligned = len(tool_ordinal) == len(self.observation_refs)
            if self.observation_refs and not aligned:
                raise RuntimeError("Tool-output references are not aligned with the context")
            self.messages[:] = result["messages"]
            if aligned:  # a one-shot turn's ephemeral context carries no O
                for item in result["masked"]:
                    self.observation_refs[tool_ordinal.index(item["message_index"])] = self.recorder.put_bytes(
                        self.context_limit.placeholder.encode("utf-8"), media_type="text/plain; charset=utf-8",
                        visibility=Visibility.WORKER_VISIBLE)
            self.recorder.emit("lifecycle", payload={"transition": "context_limit_masked", **summary,
                "outputs": result["masked"], "outputs_hidden_total": result["outputs_hidden_total"],
                "placeholder": self.context_limit.placeholder, "message_count": len(self.messages)})
        return result

    def _record_attempt(self, call_id: str, parent: str, phase: str, a: dict[str, Any]) -> None:
        """Durably journal one attempt as it starts and as it finishes; ledger on finish only."""
        if phase == "retry_wait":
            operation_id = f"{call_id}:attempt{a['attempt']}:retry_wait"
            usage = (Usage(elapsed_seconds=a["elapsed_seconds"], measurement_basis="measured_provider_retry_wait")
                     if a["phase"] == "finished" else None)
            try:
                event = self.recorder.emit("operation", call_id=call_id, attempt=a["attempt"],
                    parent_event_id=parent, usage=usage,
                    payload={"name": "worker_retry_wait", "operation_id": operation_id, **a})
                if usage is not None:
                    self.ledger.record(run_id=self.recorder.run_id, operation_id=operation_id,
                        role="infrastructure", kind="infrastructure", status=a["status"], usage=usage,
                        source_event_id=event.event_id, source_event_sequence=event.sequence)
            except BaseException as exc:
                preserve_failure(exc, a)
                self.recorder.persistence_failure = exc
                raise
            return
        if phase == "started":
            if a["attempt"] > 1:
                self._remaining_allowance()
            self.recorder.emit("model_attempt", call_id=call_id, attempt=a["attempt"], parent_event_id=parent,
                               input_ref=self.recorder.put_bytes(a["request_bytes"], media_type="application/json",
                                   visibility=Visibility.WORKER_VISIBLE) if a.get("request_bytes") is not None else None,
                               payload={"phase": "started", "started_at": a.get("started_at"),
                                        "request_url": a.get("request_url"), "request_method": a.get("request_method"),
                                        "request_headers": a.get("request_headers")})
            self._open_attempts[(call_id, a["attempt"])] = (parent, dict(a))
            return
        usage = a.get("usage")
        # Keep the actual received receipt before any write. Cleanup must never
        # replace it with an invented no-response interruption.
        self._received_attempts[(call_id, a["attempt"])] = dict(a)
        try:
            self._persist_received_attempt(call_id, parent, a, usage)
        except BaseException as exc:
            preserve_failure(exc, self._received_attempts[(call_id, a["attempt"])])
            self.recorder.persistence_failure = exc
            raise
        self._received_attempts.pop((call_id, a["attempt"]), None)

    def _persist_received_attempt(self, call_id: str, parent: str, a: dict, usage: Usage | None) -> None:
        output_ref = self.recorder.put_bytes(a["response_bytes"]) if a.get("response_bytes") is not None else None
        lookup = a.get("usage_recovery")
        recovery_ref = None
        if lookup is not None:
            recovery_ref = self.recorder.put_bytes(lookup["response_bytes"]) if lookup.get("response_bytes") is not None else None
            recovery_usage = Usage(elapsed_seconds=lookup["elapsed_seconds"], measurement_basis="measured_usage_recovery_seconds")
            recovery = self.recorder.emit("operation", call_id=call_id, attempt=a["attempt"], parent_event_id=parent,
                output_ref=recovery_ref, usage=recovery_usage,
                payload={"name": "worker_usage_recovery", **{k: v for k, v in lookup.items() if k != "response_bytes"}})
            self.ledger.record(run_id=self.recorder.run_id, operation_id=f"{call_id}:attempt{a['attempt']}:usage_recovery",
                role="infrastructure", kind="infrastructure", status=lookup["status"], usage=recovery_usage,
                source_event_id=recovery.event_id, source_event_sequence=recovery.sequence)
        raw_usage = None
        if a.get("response_bytes"):
            try:
                parsed = json.loads(a["response_bytes"])
                raw_usage = parsed.get("usage") if isinstance(parsed, dict) else None
            except ValueError:
                raw_usage = None
        ev = self.recorder.emit("model_attempt", call_id=call_id, attempt=a["attempt"], parent_event_id=parent,
                                output_ref=output_ref, usage=usage,
                                input_ref=self.recorder.put_bytes(a["request_bytes"], media_type="application/json",
                                    visibility=Visibility.WORKER_VISIBLE) if a.get("request_bytes") is not None else None,
                                payload={"phase": "finished", "status": a["status"], "error": a.get("error"),
                                         "http_status": a.get("http_status"), "latency_seconds": a.get("latency_seconds"),
                                         "started_at": a.get("started_at"), "response_headers": a.get("response_headers"),
                                         "response_identity": a.get("response_identity"),
                                         "response_parse_error": a.get("response_parse_error"),
                                         "usage_recovery_ref": to_jsonable(recovery_ref),
                                         "usage_recovery_unavailable": a.get("usage_recovery_unavailable"),
                                         "original_usage": to_jsonable(a.get("original_usage")),
                                         "provider_accounting": a.get("provider_accounting"),
                                         "retry_decision": a.get("retry_decision"),
                                         "generation_id_source": a.get("generation_id_source"),
                                         "raw_usage": raw_usage, "usage_known": usage is not None and usage.generated_tokens is not None,
                                         **({"worst_case_charge": a["worst_case_charge"]} if a.get("worst_case_charge") else {}),
                                         **({"not_sent": a["not_sent"]} if a.get("not_sent") else {})})
        measured = usage or Usage(elapsed_seconds=a.get("latency_seconds"), measurement_basis="attempt_without_provider_usage")
        self.ledger.record(run_id=self.recorder.run_id, operation_id=f"{call_id}:attempt{a['attempt']}", role="worker",
                           usage=measured,
                           kind="model", status=a["status"], source_event_id=ev.event_id, source_event_sequence=ev.sequence)
        if measured.cost_usd is None:
            self.attempt_cost_unknown += 1
        else:
            self.attempt_cost_known += measured.cost_usd
        charge = a.get("worst_case_charge")
        if charge and measured.generated_tokens is None:
            self.charged_unknown_attempts += 1
            self.worst_case_generated_tokens += charge["completion_token_bound"]
            if measured.cost_usd is None:
                self.attempt_cost_worst_case += charge["usd"]
            self.usage_log.append((ev.sequence, measured))
        else:
            self._account(measured, ev.sequence)
        self._open_attempts.pop((call_id, a["attempt"]), None)

    def _account(self, usage: Usage, response_event_sequence: int) -> None:
        if usage.generated_tokens is None:
            self.unknown_usage_calls += 1
        else:
            self.known_generated_tokens += usage.generated_tokens
        self.usage_log.append((response_event_sequence, usage))

    # ---------------------------------------------------------------- actions
    def _measure_before_tool_state(self, *, call_id: str, index: int) -> dict:
        started, status = time.monotonic(), "failed"
        try:
            state = {"workspace": self.env.workspace_fidelity_manifest(), "runtime": self.env.runtime_manifest()}
            status = "captured"
            return state
        finally:
            elapsed = time.monotonic() - started
            primary = sys.exception()
            try:
                event = self.recorder.emit("operation", call_id=call_id, payload={
                    "name": "pre_tool_state_measurement", "status": status, "tool_index": index})
                self.ledger.record(run_id=self.recorder.run_id, operation_id=event.event_id,
                    role="infrastructure", kind="local", status=status,
                    usage=Usage(elapsed_seconds=elapsed, measurement_basis="measured_pre_tool_state_recording"),
                    source_event_id=event.event_id, source_event_sequence=event.sequence)
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"Pre-tool timing persistence also failed: {type(cleanup).__name__}")

    def _record_tool_state(self, *, call_id: str, parent: str, index: int, before: dict) -> None:
        """Retain intermediate mutations without declaring partial batches branchable."""
        started = time.monotonic()
        try:
            w_bytes, w_manifest = self.env.capture_workspace()
            runtime = self.env.capture_runtime()
            r_archive = runtime.pop("archive", None)
            runtime["archive_ref"] = to_jsonable(self.recorder.put_bytes(r_archive, media_type="application/x-tar")) if r_archive else None
            after = {"workspace": w_manifest, "runtime": runtime}
            before_w, after_w = comparable(before["workspace"]), comparable(w_manifest)
            before_r, after_r = comparable(before["runtime"]["entries"]), comparable(runtime["entries"])
            self.batch_mutated |= (before_w != after_w or before_r != after_r
                                   or before["runtime"]["deleted"] != runtime["deleted"]
                                   or before["runtime"].get("mounts") != runtime.get("mounts"))
            event = self.recorder.emit("tool_state", parent_event_id=parent, call_id=call_id,
                input_ref=self.recorder.put_json(before), output_ref=self.recorder.put_json(after), payload={
                    "status": "captured", "tool_index": index, "branchable_checkpoint": False,
                    "workspace_archive_ref": to_jsonable(self.recorder.put_bytes(w_bytes, media_type="application/x-tar")),
                    "workspace_changed": sorted(p for p in set(before_w) | set(after_w) if before_w.get(p) != after_w.get(p)),
                    "runtime_changed": sorted(p for p in set(before_r) | set(after_r) if before_r.get(p) != after_r.get(p)),
                    "runtime_deleted_before": before["runtime"]["deleted"], "runtime_deleted_after": runtime["deleted"],
                    "mounted_state_changed": before["runtime"].get("mounts") != runtime.get("mounts"),
                    "runtime_class": runtime["runtime_class"], "coverage": runtime["coverage"]})
            self.batch_state_events.append(event)
        except Exception as exc:
            evidence = {"before": before, "workspace_bytes": locals().get("w_bytes"),
                "workspace_manifest": locals().get("w_manifest"), "runtime": locals().get("runtime"),
                "runtime_bytes": locals().get("r_archive"), "tool_result_event_id": parent}
            input_ref = diagnostic_ref = None
            diagnostic = None
            if isinstance(exc, CaptureError):
                self.recorder.capture_failure = exc
                capture = dict(getattr(exc, "capture_evidence", {}))
                exc.capture_evidence = {**evidence, "capture": capture}
                diagnostic = {"schema": "failed-capture-diagnostic.v1", "restorable": False,
                    "branchable": False, "validated_snapshot": False,
                    "capture": {k: v for k, v in capture.items() if k != "archive_bytes"},
                    "workspace_manifest": evidence["workspace_manifest"],
                    "before_ref": None, "failed_archive_ref": None, "workspace_diagnostic_archive_ref": None}
                try:
                    input_ref = self.recorder.put_json(before)
                    diagnostic["before_ref"] = to_jsonable(input_ref)
                    if capture.get("archive_bytes") is not None:
                        diagnostic["failed_archive_ref"] = to_jsonable(self.recorder.put_bytes(
                            capture["archive_bytes"], media_type="application/x-tar"))
                    if evidence["workspace_bytes"] is not None:
                        diagnostic["workspace_diagnostic_archive_ref"] = to_jsonable(self.recorder.put_bytes(
                            evidence["workspace_bytes"], media_type="application/x-tar"))
                    diagnostic_ref = self.recorder.put_json(diagnostic)
                except BaseException as cleanup:
                    exc.capture_diagnostic_persistence_failure = cleanup
                    exc.add_note(f"Capture diagnostic persistence also failed: {type(cleanup).__name__}")
            else:
                preserve_failure(exc, evidence)
                self.recorder.persistence_failure = exc
            try:
                event = self.recorder.emit("tool_state", parent_event_id=parent, call_id=call_id,
                    input_ref=input_ref, output_ref=diagnostic_ref,
                    payload={"status": "failed", "tool_index": index, "branchable_checkpoint": False,
                             "error": str(exc), "failure_kind": "capture_failure" if isinstance(exc, CaptureError) else "persistence_failure",
                             "diagnostic_evidence": diagnostic, "diagnostic_complete": diagnostic_ref is not None})
            except BaseException as cleanup:
                if isinstance(exc, CaptureError):
                    exc.capture_diagnostic_persistence_failure = cleanup
                exc.add_note(f"Tool-state failure receipt also failed: {type(cleanup).__name__}")
            raise
        finally:
            elapsed = time.monotonic() - started
            primary = sys.exception()
            if isinstance(primary, CaptureError):
                primary.capture_evidence["tool_state_duration_seconds"] = elapsed
            if "event" in locals():
                try:
                    self.ledger.record(run_id=self.recorder.run_id, operation_id=event.event_id,
                        role="infrastructure", kind="local", status=event.payload["status"],
                        usage=Usage(elapsed_seconds=elapsed, measurement_basis="measured_tool_state_recording"),
                        source_event_id=event.event_id, source_event_sequence=event.sequence)
                except BaseException as cleanup:
                    if primary is None:
                        self.recorder.persistence_failure = cleanup
                        raise
                    if isinstance(primary, CaptureError):
                        primary.capture_diagnostic_persistence_failure = cleanup
                        self.recorder.persistence_failure = cleanup
                    primary.add_note(f"Tool-state cost persistence also failed: {type(cleanup).__name__}")

    def _record_action_manifest(self, *, call_id: str, parent: str, index: int) -> None:
        """Per-action change evidence without archives: the after-state manifests of one non-final action."""
        started, event, status = time.monotonic(), None, "failed"
        try:
            state = {"workspace": self.env.workspace_fidelity_manifest(), "runtime": self.env.runtime_manifest()}
            status = "manifest"
            event = self.recorder.emit("tool_state", parent_event_id=parent, call_id=call_id,
                output_ref=self.recorder.put_json(state), payload={
                    "status": status, "rule": PER_ACTION_MANIFEST_RULE, "tool_index": index,
                    "branchable_checkpoint": False})
            self.batch_action_states.append((index, state, event))
        except BaseException as exc:
            # Change evidence for this action is lost: stop further worker calls, never infer "unchanged".
            self.recorder.capture_failure = exc
            try:
                event = self.recorder.emit("tool_state", parent_event_id=parent, call_id=call_id, payload={
                    "status": "failed", "rule": PER_ACTION_MANIFEST_RULE, "tool_index": index,
                    "branchable_checkpoint": False, "error": f"{type(exc).__name__}: {exc}"})
            except BaseException as cleanup:
                exc.add_note(f"Per-action manifest failure receipt also failed: {type(cleanup).__name__}")
            raise
        finally:
            if event is not None:
                elapsed = time.monotonic() - started
                self.ledger.record(run_id=self.recorder.run_id, operation_id=event.event_id, role="infrastructure",
                    kind="local", status=status, source_event_id=event.event_id, source_event_sequence=event.sequence,
                    usage=Usage(elapsed_seconds=elapsed, measurement_basis="measured_per_action_manifest"))

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        call_id = message.get("extra", {}).get("call_id")
        outputs = []
        # Executed outputs of the current batch stay addressable when a
        # submission interrupts it (native submission-feedback reopen policy).
        self.batch_outputs = (message, outputs)
        executed = []
        lap_sources = {}
        full_states = self.tool_state_policy == "full"
        for index, action in enumerate(actions):
            before_state = self._measure_before_tool_state(call_id=call_id, index=index) if full_states else None
            tool_name = action.get("tool_name", "bash")
            operation_id = f"{call_id}:tool{index}"
            original_call = next((c for c in message.get("tool_calls") or []
                                  if c.get("id") == action.get("tool_call_id")), None)
            original_arguments = (original_call.get("function") or {}).get("arguments") if original_call else None
            arguments = {"command": action.get("command")} if tool_name == "bash" else action.get("arguments")
            ev = self.recorder.emit("tool_call", call_id=action.get("tool_call_id"), actor="worker",
                                    input_ref=self.recorder.text_ref(original_arguments, Visibility.WORKER_VISIBLE)
                                        if isinstance(original_arguments, str) else None,
                                    payload={"command": action.get("command"), "tool_name": tool_name,
                                             "arguments": arguments, "model_call_id": call_id,
                                             "tool_index": index, "operation_id": operation_id})
            started = time.monotonic()
            try:
                if tool_name == "bash":
                    result = self.env.execute_recorded(action)
                else:
                    if self.control_handler is None:
                        raise RuntimeError("Recovery tool unavailable in this worker")
                    content = self.control_handler(action, self.last_response_event_sequence)
                    result = ExecResult(json.dumps(content, ensure_ascii=False), 0, "", time.monotonic()-started)
            except BaseException as exc:
                # Cancellation and recovery control transfer do not deliver an
                # observation. Persist measured attempt duration before billing,
                # so a process exit between these writes has a read-only view.
                elapsed = time.monotonic() - started
                status = ("control_transfer" if type(exc).__name__ == "RewindRequested" else
                          "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error")
                usage = Usage(elapsed_seconds=elapsed, measurement_basis="measured_interrupted_tool")
                try:
                    event = self.recorder.emit("tool_result", call_id=action.get("tool_call_id"),
                        parent_event_id=ev.event_id, actor="harness", visibility=Visibility.RESEARCH_PRIVATE,
                        usage=usage, payload={"tool_name": tool_name, "operation_id": operation_id,
                            "status": status, "duration_seconds": elapsed,
                            "delivery_status": "control_transfer_not_appended" if status == "control_transfer" else "interrupted_not_appended",
                            "control_transfer": type(exc).__name__ if status == "control_transfer" else None,
                            "exception_type": type(exc).__name__, "exception_info": str(exc)})
                    self.ledger.record(run_id=self.recorder.run_id, operation_id=operation_id,
                        role="worker", kind="tool", status=status, usage=usage,
                        source_event_id=event.event_id, source_event_sequence=event.sequence)
                except Exception:
                    # The durable start still gives an uncertain view if final
                    # recording fails. Do not replace the original interruption.
                    pass
                raise
            # Execution evidence must survive failures in observation formatting
            # or template/environment queries. This event says nothing about
            # rendering or delivery, which have not happened yet.
            raw_bytes = result.raw_output if result.raw_output is not None else result.output.encode("utf-8")
            uncertain = result.status in {"transport_error", "transport_timeout"}
            self.tool_execution_uncertain |= uncertain
            try:
                transport = {key: to_jsonable(self.recorder.put_bytes(value, media_type="application/octet-stream"))
                    for key, value in (("stdout_ref", result.transport_stdout), ("stderr_ref", result.transport_stderr))
                    if value is not None}
                result_event = self.recorder.emit("tool_result", call_id=action.get("tool_call_id"), parent_event_id=ev.event_id, actor="tool",
                                   output_ref=self.recorder.put_bytes(raw_bytes, media_type="text/plain; charset=utf-8"),
                                   visibility=Visibility.RESEARCH_PRIVATE,
                                   payload={"returncode": result.returncode, "duration_seconds": result.duration_seconds,
                                            "operation_id": operation_id,
                                            "status": result.status, "execution_details": result.extra,
                                            "transport": transport,
                                            "delivery_status": "execution_uncertain_not_rendered" if uncertain else "execution_finished_not_rendered",
                                            "exception_info": result.exception_info, "raw_chars": len(result.output)})
                executed.append((result_event, raw_bytes))
                lap_sources[result_event.event_id] = {"attempt_id": ev.event_id,
                    "tool_name": tool_name, "command": action.get("command", ""),
                    "returncode": result.returncode, "status": result.status,
                    "source_event_ids": {"tool_call": ev.event_id, "tool_result": result_event.event_id}}
                # Provider tool-call ids are not unique across sessions; key the ledger on our call id.
                self.ledger.record(run_id=self.recorder.run_id, operation_id=operation_id, role="worker",
                                   usage=Usage(elapsed_seconds=result.duration_seconds, measurement_basis="measured_wall_clock"),
                                   kind="tool", status=result.status,
                                   source_event_id=result_event.event_id, source_event_sequence=result_event.sequence)
            except BaseException as exc:
                preserve_failure(exc, {"result": result, "raw_output": raw_bytes, "operation_id": operation_id,
                                       "tool_call_event_id": ev.event_id})
                self.recorder.persistence_failure = exc
                raise
            if uncertain:
                raise ToolTransportFailure(result)
            output = result.as_output()
            outputs.append(output)
            rendered = self.model.format_observation_messages(message, [output], self.get_template_vars())[0]
            rendered_bytes = rendered["content"].encode("utf-8")
            rendered_ref = self.recorder.put_bytes(rendered_bytes, media_type="text/plain; charset=utf-8")
            self.recorder.emit("tool_render", call_id=action.get("tool_call_id"), parent_event_id=result_event.event_id,
                actor="harness", visibility=Visibility.RESEARCH_PRIVATE,
                output_ref=rendered_ref,
                payload={"operation_id": operation_id, "delivery_status": "rendered_not_appended",
                         "rendered_ref": to_jsonable(rendered_ref),
                         "raw_equals_rendered": raw_bytes == rendered_bytes})
            self.completed_in_batch.append(action)
            self.pending_actions = self.pending_actions[1:]
            if full_states:
                self._record_tool_state(call_id=call_id, parent=result_event.event_id, index=index, before=before_state)
            # Submission detection mirrors the pinned environment; remaining actions are not executed.
            self.env._check_finished(output)
            if self.tool_state_policy == "per_action_manifest" and index < len(actions) - 1:
                self._record_action_manifest(call_id=call_id, parent=result_event.event_id, index=index)
        observation_messages = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        self.pending_actions = []
        added = self.add_messages(*observation_messages)
        for (execution, raw_bytes), observation in zip(executed, observation_messages, strict=True):
            delivered_bytes = observation["content"].encode("utf-8")
            delivered_ref = self.recorder.put_bytes(delivered_bytes, media_type="text/plain; charset=utf-8",
                                                         visibility=Visibility.WORKER_VISIBLE)
            self.observation_refs.append(delivered_ref)
            delivery = self.recorder.emit("observation_delivery", call_id=execution.call_id,
                parent_event_id=execution.event_id, actor="harness", delivered_ref=delivered_ref,
                visibility=Visibility.WORKER_VISIBLE, payload={"role": "tool", "delivery_status": "appended_to_history",
                    "raw_equals_delivered": raw_bytes == delivered_bytes})
            source = lap_sources[execution.event_id]
            self.batch_lap_observations.append({**source, "delivered": observation["content"],
                "source_event_ids": {**source["source_event_ids"], "delivery": delivery.event_id},
                "event_sequence": delivery.sequence})
        return added

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())
