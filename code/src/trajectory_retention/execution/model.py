"""Instrumented model boundary for mini-SWE-agent.

Both models implement the pinned mini-SWE-agent (2.4.6) ``Model`` protocol so
``DefaultAgent`` logic is reused unchanged. Unlike the upstream models, every
HTTP attempt (including failures and retries) is returned to the caller in
``message["extra"]["attempts"]`` with exact request/response bytes, status,
error, latency and provider usage. The agent persists them; nothing is hidden
inside a retry wrapper.

Usage is normalized by the shared accounting module; missing counters stay
``None``.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable

import requests

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import (BASH_TOOL, format_toolcall_observation_messages,
                                                        parse_toolcall_actions)

from ..accounting import openrouter_usage
from ..contracts import Usage, WorkerRequestPolicy, to_jsonable
from ..persistence import preserve_failure
from ..provider_policy import (NOT_SENT_RULE, REJECTION_RULE, RETRY_RULE, generation_identity, never_sent,
                              pinned_provider, provider_identity_matches, rejection_usage, retry_plan,
                              charged_retry_plan, worst_case_charge)


def history_replay_issue(messages: list[dict]) -> str | None:
    """A native continuation needs a complete provider conversation, not an exit marker."""
    pending: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "exit":
            return "terminal_exit_requires_explicit_reopen_policy"
        if role == "tool":
            call = message.get("tool_call_id")
            if call not in pending:
                return "orphan_tool_result"
            pending.remove(call)
            continue
        if pending:
            return "missing_tool_result_before_next_turn"
        if role not in {"system", "developer", "user", "assistant"}:
            return "unsupported_message_role"
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            if not call_id or call_id in pending:
                return "invalid_pending_tool_call_identity"
            pending.add(call_id)
    return "pending_tool_results" if pending else None

DEFAULT_OBSERVATION_TEMPLATE = (
    "{%- if output.output | length < 10000 -%}\n"
    "{\n  \"returncode\": {{ output.returncode }},\n  \"output\": {{ output.output | tojson }}"
    "{%- if output.exception_info %}, \"exception_info\": {{ output.exception_info | tojson }}{% endif %}\n}\n"
    "{%- else -%}\n"
    "{\n  \"returncode\": {{ output.returncode }},\n  \"output_head\": {{ output.output[:5000] | tojson }},\n"
    "  \"output_tail\": {{ output.output[-5000:] | tojson }},\n  \"elided_chars\": {{ output.output | length - 10000 }},\n"
    "  \"warning\": \"Output too long.\""
    "{%- if output.exception_info %}, \"exception_info\": {{ output.exception_info | tojson }}{% endif %}\n}\n"
    "{%- endif -%}"
)
DEFAULT_FORMAT_ERROR_TEMPLATE = (
    "{% if finish_reason is defined and (finish_reason == \"length\" or (finish_reason == \"tool_calls\" and not has_tool_calls)) -%}\n"
    "Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a tool call, "
    "so it was cut off. Respond more concisely and finish with exactly one bash tool call. If you need to think more, do so briefly.\n"
    "{%- else -%}\nTool call error:\n\n<error>\n{{error}}\n</error>\n\nHere is general guidance on how to submit correct toolcalls:\n\n"
    "Every response needs to use the 'bash' tool at least once to execute commands.\n\nCall the bash tool with your command as the argument:\n"
    "- Tool: bash\n- Arguments: {\"command\": \"your_command_here\"}\n\nIf you want to end the task, please issue the following command: "
    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command.\n{%- endif %}"
)


class ModelAbort(RuntimeError):
    """Raised when a model call cannot be completed; attempts are attached."""

    def __init__(self, message: str, attempts: list[dict[str, Any]], request: dict[str, Any]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.request = request


class _ToolCall:
    def __init__(self, d: dict[str, Any]) -> None:
        self.id = d.get("id")
        fn = d.get("function") or {}
        self.function = type("Fn", (), {"name": fn.get("name"), "arguments": fn.get("arguments")})()


def normalize_message(choice_message: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
    """Worker-visible normalized response view (content, tool calls, returned reasoning)."""
    calls = []
    for tc in choice_message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": fn.get("arguments")})
    return {"content": choice_message.get("content"), "tool_calls": calls,
            "reasoning_content": choice_message.get("reasoning"), "finish_reason": finish_reason}


AttemptSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PreparedRequest:
    """One immutable serialization shared by the archive and HTTP transport."""
    body: bytes

    @property
    def request(self) -> dict[str, Any]:
        return json.loads(self.body)


class _BaseModel:
    """Shared message formatting; mirrors OpenRouterModel in mini-SWE-agent 2.4.6.

    ``attempt_sink(phase, record)`` is called synchronously with phase "started"
    before each HTTP attempt and "finished" as soon as it returns, so a crash or
    retry can never lose an earlier receipt.
    """

    def __init__(self, *, model_name: str, model_settings: dict[str, Any] | None = None,
                 observation_template: str = DEFAULT_OBSERVATION_TEMPLATE,
                 format_error_template: str = DEFAULT_FORMAT_ERROR_TEMPLATE,
                 request_policy: WorkerRequestPolicy | None = None) -> None:
        self.attempt_sink: AttemptSink | None = None
        self.allowance_remaining: Callable[[], int] | None = None
        # v2: the charged-retry rule for unresolved attempts (provider_policy.unresolved_retry_rule), bound by the
        # runner from the run's metadata; None keeps the Q8 stop.
        self.unresolved_retry: dict[str, Any] | None = None
        self.model_name = model_name
        self.request_policy = request_policy or WorkerRequestPolicy()
        self.expected_request_policy: WorkerRequestPolicy | None = None
        self.expected_model_binding: tuple[str, dict[str, Any]] | None = None
        self.model_settings = dict(model_settings or {})
        if set(self.model_settings) & {"model", "messages", "tools", "stream"}:
            raise ValueError("Worker settings cannot override model, history, tools or response mode")
        caps = [self.model_settings[k] for k in ("max_tokens", "max_completion_tokens")
                if self.model_settings.get(k) is not None]
        if len(caps) > 1 or any(type(cap) is not int or cap <= 0 for cap in caps):
            raise ValueError("Declare at most one positive integer ordinary response cap")
        self.observation_template = observation_template
        self.format_error_template = format_error_template
        self.action_parser = parse_toolcall_actions
        self.config = {"model_name": model_name, "model_settings": self.model_settings,
                       "observation_template": observation_template, "format_error_template": format_error_template,
                       "tools": [BASH_TOOL], "allowance_policy": "remaining-and-ordinary-response-cap/v1",
                       "worker_request_policy": to_jsonable(self.request_policy)}

    @staticmethod
    def prepare_messages(messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in m.items() if k != "extra"} for m in messages]

    def build_request(self, messages: list[dict]) -> dict[str, Any]:
        issue = history_replay_issue(messages)
        if issue:
            raise ValueError(f"History is not ready for a native model continuation: {issue}")
        # A one-shot turn without tools (v2 arm 5 note) sends no tools key at all, never an empty list.
        tools = {"tools": self.config["tools"]} if self.config["tools"] else {}
        return {"model": self.model_name, "messages": self.prepare_messages(messages), **tools,
                "usage": {"include": True}, **self.model_settings}

    def prepare_request(self, messages: list[dict]) -> PreparedRequest:
        if self.expected_request_policy is not None and self.effective_request_policy() != self.expected_request_policy:
            raise ValueError("Worker request policy changed after its execution binding")
        if self.expected_model_binding is not None and (self.model_name, self.model_settings) != self.expected_model_binding:
            raise ValueError("Worker model/settings changed after their execution binding")
        request = self.build_request(messages)
        if self.allowance_remaining is not None:
            remaining = self.allowance_remaining()  # Recomputed before every physical attempt.
            if type(remaining) is not int or remaining <= 0:
                raise ValueError("A request requires a positive known remaining allowance")
            key = "max_tokens" if self.model_settings.get("max_tokens") is not None else "max_completion_tokens"
            ordinary = self.model_settings.get(key)
            request.pop("max_tokens", None)
            request.pop("max_completion_tokens", None)
            request[key] = min(remaining, ordinary) if ordinary is not None else remaining
        return PreparedRequest(json.dumps(request).encode())

    def effective_request_policy(self) -> WorkerRequestPolicy:
        return self.request_policy

    def bind_execution(self, request_policy: WorkerRequestPolicy) -> None:
        self.expected_request_policy = request_policy
        self.expected_model_binding = (self.model_name, deepcopy(self.model_settings))

    def _charged_retry(self, record: dict[str, Any], request: dict[str, Any], body: bytes, attempt: int,
                       decision: dict) -> dict:
        """Under the charged-retry rule, an unresolved attempt is retried and carries its worst-case charge."""
        if self.unresolved_retry is None or decision.get("reason") != "unresolved_generation_Q8_stop":
            return decision
        policy = self.effective_request_policy()
        charged = charged_retry_plan(decision, attempt=attempt, max_attempts=policy.max_attempts,
            backoff_seconds=policy.backoff_seconds, max_wait_seconds=policy.timeout_seconds)
        if charged["retry"]:
            record["worst_case_charge"] = worst_case_charge(self.unresolved_retry, request, body)
        return charged

    def _emit(self, phase: str, record: dict[str, Any]) -> None:
        if self.attempt_sink is not None:
            self.attempt_sink(phase, record)

    def _finish(self, request: dict[str, Any], response: dict[str, Any], attempts: list[dict[str, Any]], usage: Usage) -> dict:
        choice = response["choices"][0]
        message = dict(choice["message"])
        finish_reason = choice.get("finish_reason")
        normalized = normalize_message(message, finish_reason)
        extra = {"request": request, "response": response, "attempts": attempts, "usage": usage,
                 "normalized": normalized, "cost": usage.cost_usd, "timestamp": time.time()}
        try:
            actions = self.action_parser([_ToolCall(tc) for tc in message.get("tool_calls") or []],
                                             format_error_template=self.format_error_template,
                                             template_kwargs={"finish_reason": finish_reason})
        except FormatError as e:
            e.messages[0]["extra"].update(extra)
            e.messages[0]["extra"]["rejected_message"] = message
            raise
        message["extra"] = {"actions": actions, **extra}
        return message

    def format_message(self, **kwargs: Any) -> dict:
        return kwargs

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        return format_toolcall_observation_messages(actions=message.get("extra", {}).get("actions", []), outputs=outputs,
                                                    observation_template=self.observation_template, template_vars=template_vars)

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.config)

    def serialize(self) -> dict:
        return {"info": {"config": {"model": self.config, "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}"}}}


class OpenRouterChatModel(_BaseModel):
    """OpenRouter chat completions with explicit, fully recorded retry attempts."""

    NON_RETRYABLE = {400, 401, 403, 404, 413, 422}

    def __init__(self, *, model_name: str, model_settings: dict[str, Any] | None = None, api_key_env: str = "OPENROUTER_API_KEY",
                 max_attempts: int = 5, backoff_seconds: float = 4.0, timeout_seconds: float = 600.0,
                 url: str = "https://openrouter.ai/api/v1/chat/completions",
                 usage_recovery_timeout_seconds: float = 30.0, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, model_settings=model_settings,
            request_policy=WorkerRequestPolicy(max_attempts=max_attempts, backoff_seconds=backoff_seconds,
                timeout_seconds=timeout_seconds, usage_recovery_timeout_seconds=usage_recovery_timeout_seconds), **kwargs)
        self.api_key_env = api_key_env
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self.url = url
        self.usage_recovery_url = "https://openrouter.ai/api/v1/generation" if url == "https://openrouter.ai/api/v1/chat/completions" else None
        self.usage_recovery_timeout_seconds = usage_recovery_timeout_seconds
        self.config.update({"provider": "openrouter", "url": url, "max_attempts": max_attempts,
                            "backoff_seconds": backoff_seconds, "timeout_seconds": timeout_seconds, "api_key_env": api_key_env,
                            "usage_recovery": {"url": self.usage_recovery_url, "max_attempts": 1,
                                "timeout_seconds": usage_recovery_timeout_seconds, "rule": "bound-final-native-completion/v1"},
                            "rejection_accounting_rule": REJECTION_RULE, "retry_rule": RETRY_RULE,
                            "max_retry_wait_seconds": timeout_seconds})
        if not os.environ.get(api_key_env):
            raise ModelAbort(f"{api_key_env} is not set; live calls refused", [], {})

    @staticmethod
    def _receive(record: dict[str, Any], response: requests.Response) -> dict | None:
        # Decode only the captured bytes, not a separately transformed JSON view.
        record.update(http_status=response.status_code, response_bytes=response.content,
                      response_headers=dict(response.headers))
        try:
            parsed = json.loads(record["response_bytes"])
            if not isinstance(parsed, dict):
                raise ValueError("response JSON is not an object")
        except (ValueError, UnicodeError) as error:
            record["response_parse_error"] = str(error)
            parsed = None
        record["response_identity"] = {k: parsed[k] for k in (
            "id", "model", "provider", "system_fingerprint", "openrouter_metadata")
            if parsed is not None and k in parsed}
        record["generation_id_source"] = generation_identity(parsed, record["response_headers"])[1]
        record["usage"] = openrouter_usage(parsed.get("usage") or {} if parsed is not None else {},
            elapsed_seconds=record["latency_seconds"])
        return parsed

    def effective_request_policy(self) -> WorkerRequestPolicy:
        return WorkerRequestPolicy(max_attempts=self.max_attempts, backoff_seconds=self.backoff_seconds,
            timeout_seconds=self.timeout_seconds, usage_recovery_timeout_seconds=self.usage_recovery_timeout_seconds)

    @staticmethod
    def _response_issue(parsed: dict | None) -> str | None:
        if parsed is None:
            return "invalid JSON response"
        if "error" in parsed:
            return f"provider error: {parsed['error']}"
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return "malformed response choices"
        if choices[0].get("error") or choices[0].get("finish_reason") == "error":
            return "provider interrupted generation; partial response retained"
        message = choices[0].get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return "malformed assistant message"
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                return "malformed tool_calls"
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    return "malformed tool call"
                function = call["function"]
                if not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                    return "malformed tool function"
        return None

    def _recover_usage(self, record: dict[str, Any], headers: dict[str, str]) -> None:
        if record["usage"].generated_tokens is not None:
            return
        identity = record.get("response_identity") or {}
        generation_id, source = generation_identity(identity, record.get("response_headers", {}))
        record["generation_id_source"] = source
        if not self.usage_recovery_url or not isinstance(generation_id, str) or not generation_id:
            record["usage_recovery_unavailable"] = "No supported endpoint and bound generation ID"
            return
        lookup = {"url": self.usage_recovery_url, "generation_id": generation_id, "status": "unavailable"}
        record["usage_recovery"] = lookup
        start = time.monotonic()
        try:
            response = requests.get(self.usage_recovery_url, headers=headers, params={"id": generation_id},
                                    timeout=self.usage_recovery_timeout_seconds)
            lookup.update(http_status=response.status_code, response_bytes=response.content,
                          response_headers=dict(response.headers))
            if response.status_code != 200:
                lookup["error"] = f"HTTP {response.status_code}"
                return
            parsed = json.loads(response.content)
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if (not isinstance(data, dict) or data.get("id") != generation_id
                    or data.get("model") != identity.get("model", self.model_name)
                    or not isinstance(data.get("finish_reason"), str) or not data["finish_reason"]):
                lookup["error"] = "Generation receipt is not final and bound to the returned generation/model"
                return
            pin = pinned_provider({"provider": self.model_settings.get("provider")})
            if pin and data.get("provider_name") is not None and (
                    not isinstance(data["provider_name"], str) or not provider_identity_matches(pin, data["provider_name"])):
                lookup["error"] = "Generation receipt provider differs from the pinned route"
                return
            recovered = openrouter_usage({"completion_tokens": data.get("native_tokens_completion"),
                "prompt_tokens": data.get("native_tokens_prompt"), "cost": data.get("total_cost"),
                "completion_tokens_details": {"reasoning_tokens": data.get("native_tokens_reasoning")},
                "prompt_tokens_details": {"cached_tokens": data.get("native_tokens_cached")}},
                elapsed_seconds=record["latency_seconds"])
            if recovered.generated_tokens is None or (recovered.reasoning_tokens is not None
                    and recovered.reasoning_tokens > recovered.generated_tokens):
                lookup["error"] = "Native completion count absent, invalid or inconsistent; normalized token counts are not substituted"
                return
            original = record["usage"]
            if original.reasoning_tokens is not None and original.reasoning_tokens > recovered.generated_tokens:
                lookup["error"] = "Generation receipt conflicts with already reported reasoning usage"
                return
            record["original_usage"] = original
            updates = {key: getattr(recovered, key) for key in ("input_tokens", "output_tokens", "reasoning_tokens",
                "cached_input_tokens", "generated_tokens", "cost_usd") if getattr(original, key) is None}
            record["usage"] = replace(original, **updates,
                measurement_basis="provider_reported_openrouter_generation_receipt")
            lookup["status"] = "recovered"
        except (requests.RequestException, ValueError, UnicodeError) as error:
            lookup["error"] = f"{type(error).__name__}: {error}"
        except BaseException as error:
            # The worker response predates this optional lookup. Cancellation
            # must not replace it with a fabricated no-response interruption.
            lookup.update(status="interrupted", error=f"{type(error).__name__}: {error}",
                          elapsed_seconds=time.monotonic() - start)
            record.update(status="interrupted", error="Interrupted during usage recovery")
            try:
                self._emit("finished", record)
            except BaseException as cleanup:
                preserve_failure(error, record)
                error.add_note(f"Received response persistence also failed: {type(cleanup).__name__}")
            raise
        finally:
            lookup.setdefault("elapsed_seconds", time.monotonic() - start)

    def _retry_wait(self, attempt: int, decision: dict) -> None:
        delay = decision["delay_seconds"]
        if delay <= 0:
            return
        record = {"attempt": attempt, "phase": "started", "started_at": time.time(),
                  "planned_seconds": delay, "rule": RETRY_RULE}
        self._emit("retry_wait", record)
        start = time.monotonic()
        try:
            time.sleep(delay)
        except BaseException as error:
            record = {**record, "phase": "finished", "status": "interrupted",
                      "elapsed_seconds": time.monotonic() - start}
            try:
                self._emit("retry_wait", record)
            except BaseException as cleanup:
                preserve_failure(error, record)
                error.add_note(f"Retry wait persistence also failed: {type(cleanup).__name__}")
            raise
        self._emit("retry_wait", {**record, "phase": "finished", "status": "ok",
                                  "elapsed_seconds": time.monotonic() - start})

    def query(self, messages: list[dict], *, prepared_request: PreparedRequest | None = None, **kwargs: Any) -> dict:
        prepared = prepared_request or self.prepare_request(messages)
        request = prepared.request
        headers = {"Authorization": f"Bearer {os.environ[self.api_key_env]}", "Content-Type": "application/json"}
        attempts: list[dict[str, Any]] = []
        for n in range(1, self.max_attempts + 1):
            if n > 1:
                prepared = self.prepare_request(messages)
                request = prepared.request
            record: dict[str, Any] = {"attempt": n, "started_at": time.time(), "request_bytes": prepared.body,
                                     "request_url": self.url, "request_method": "POST",
                                     "request_headers": {"Content-Type": "application/json"}}
            self._emit("started", record)
            t0, parsed = time.monotonic(), None
            # Only the external transport belongs in this exception boundary.
            # Receipt persistence failures must never become another paid call.
            try:
                resp = requests.post(self.url, headers=headers, data=prepared.body, timeout=self.timeout_seconds)
            except requests.RequestException as error:
                record.update(latency_seconds=time.monotonic() - t0,
                              status="timeout" if isinstance(error, requests.Timeout) else "error",
                              error=f"{type(error).__name__}: {error}")
                resp = error.response
                cause = never_sent(error) if resp is None else None
                if cause is not None:
                    record["usage"] = Usage(input_tokens=0, output_tokens=0, total_tokens=0, generated_tokens=0,
                                            cost_usd=0.0, elapsed_seconds=record["latency_seconds"],
                                            measurement_basis=NOT_SENT_RULE)
                    record["not_sent"] = {"rule": NOT_SENT_RULE, "cause": cause}
            else:
                record["latency_seconds"] = time.monotonic() - t0
            if resp is not None:
                parsed = self._receive(record, resp)
                self._recover_usage(record, headers)
                original = record["usage"]
                measured, accounting = rejection_usage(url=self.url, request=request,
                    http_status=resp.status_code, parsed=parsed, usage=original,
                    headers=record.get("response_headers", {}), recovery=record.get("usage_recovery"))
                record.update(usage=measured, provider_accounting=accounting)
                if measured != original:
                    record.setdefault("original_usage", original)
            usage = record.get("usage")
            issue = self._response_issue(parsed) if resp is not None and resp.status_code == 200 else None
            identity_issue = None
            if parsed is not None and resp.status_code == 200 and issue is None:
                if parsed.get("model") != self.model_name:
                    identity_issue = "Returned model identity mismatch; worker continuation refused"
                pin = pinned_provider(request)
                if pin and (not isinstance(parsed.get("provider"), str)
                            or not provider_identity_matches(pin, parsed["provider"])):
                    identity_issue = "Returned provider identity differs from the pinned route"
            if resp is not None and resp.status_code == 200 and issue is None and identity_issue is None and "error" not in record:
                record.update(status="ok", usage=usage)
                attempts.append(record)
                self._emit("finished", record)
                return self._finish(request, parsed, attempts, usage)
            record.setdefault("status", "error")
            record.setdefault("error", identity_issue or issue or f"HTTP {getattr(resp, 'status_code', None)}")
            decision = retry_plan(attempt=n, max_attempts=self.max_attempts, backoff_seconds=self.backoff_seconds,
                max_wait_seconds=self.timeout_seconds, headers=record.get("response_headers", {}), parsed=parsed,
                now=time.time(), usage=usage, retryable=identity_issue is None
                and record.get("http_status") not in self.NON_RETRYABLE)
            decision = self._charged_retry(record, request, prepared.body, n, decision)
            record["retry_decision"] = decision
            attempts.append(record)
            self._emit("finished", record)
            if not decision["retry"]:
                raise ModelAbort(f"Model request stopped: {decision['reason']}; {record['error']}", attempts, request)
            # Charge and check the old attempt before waiting or sending again.
            if self.allowance_remaining is not None:
                self.allowance_remaining()
            self._retry_wait(n, decision)
        raise ModelAbort(f"model call failed after {len(attempts)} attempts", attempts, request)


class ScriptedModel(_BaseModel):
    """Deterministic fixture model producing OpenRouter-shaped responses.

    Each script entry: {"content": str|None, "commands": [str], "usage": {...openrouter usage...},
    optional "reasoning": str, "fail_before": int (transient failures before success),
    "abort": bool (unrecoverable failure), "failure_usage"/"abort_usage": optional explicit receipts,
    "finish_reason": str, "raw_tool_calls": [...] (override)}.
    Entries are consumed in order; running past the script raises ModelAbort.
    """

    def __init__(self, script: list[dict[str, Any]], *, model_name: str = "scripted/fixture", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.script = list(script)
        self.index = 0
        self.config.update({"provider": "scripted", "script_length": len(script)})

    def query(self, messages: list[dict], *, prepared_request: PreparedRequest | None = None, **kwargs: Any) -> dict:
        prepared = prepared_request or self.prepare_request(messages)
        request, body = prepared.request, prepared.body
        if self.index >= len(self.script):
            rec = {"attempt": 1, "status": "error", "error": "script exhausted", "request_bytes": body,
                   "latency_seconds": 0.0, "started_at": time.time()}
            self._emit("started", rec)
            self._emit("finished", rec)
            raise ModelAbort("script exhausted", [rec], request)
        entry = self.script[self.index]
        self.index += 1
        attempts: list[dict[str, Any]] = []
        for n in range(1, min(entry.get("fail_before", 0), self.request_policy.max_attempts) + 1):
            if n > 1:
                prepared = self.prepare_request(messages)
                request, body = prepared.request, prepared.body
            rec = {"attempt": n, "started_at": time.time(), "request_bytes": body}
            self._emit("started", rec)
            rec.update({"status": "error", "error": "simulated transient failure", "http_status": 502,
                        "response_bytes": json.dumps({"error": {"message": "simulated"},
                            "usage": entry.get("failure_usage")}).encode(), "latency_seconds": 0.01})
            if entry.get("failure_usage") is not None:
                rec["usage"] = openrouter_usage(entry["failure_usage"], elapsed_seconds=0.01)
            elif self.unresolved_retry is not None:
                # The charged-retry rule, exactly as OpenRouterChatModel applies it (no sleep in fixtures).
                rec["retry_decision"] = self._charged_retry(rec, request, body, n, {
                    "rule": RETRY_RULE, "retry": False, "reason": "unresolved_generation_Q8_stop"})
                if not rec["retry_decision"]["retry"]:
                    attempts.append(rec)
                    self._emit("finished", rec)
                    raise ModelAbort(f"Model request stopped: {rec['retry_decision']['reason']}", attempts, request)
            attempts.append(rec)
            self._emit("finished", rec)
        if len(attempts) >= self.request_policy.max_attempts:
            raise ModelAbort(f"model call failed after {len(attempts)} attempts", attempts, request)
        if attempts:
            prepared = self.prepare_request(messages)
            request, body = prepared.request, prepared.body
        if entry.get("abort"):
            rec = {"attempt": len(attempts) + 1, "started_at": time.time(), "request_bytes": body}
            self._emit("started", rec)
            rec.update({"status": "error", "error": "simulated unrecoverable failure", "http_status": 401,
                        "response_bytes": b"", "latency_seconds": 0.01})
            if entry.get("abort_usage") is not None:
                rec["response_bytes"] = json.dumps({"error": "simulated unrecoverable failure", "usage": entry["abort_usage"]}).encode()
                rec["usage"] = openrouter_usage(entry["abort_usage"], elapsed_seconds=0.01)
            attempts.append(rec)
            self._emit("finished", rec)
            raise ModelAbort("simulated unrecoverable failure", attempts, request)
        calls = entry.get("raw_tool_calls")
        if calls is None:
            calls = [{"id": f"call_{self.index}_{i}", "type": "function",
                      "function": {"name": "bash", "arguments": json.dumps({"command": c})}}
                     for i, c in enumerate(entry.get("commands", []))]
        message: dict[str, Any] = {"role": "assistant", "content": entry.get("content"), "tool_calls": calls or None}
        if entry.get("reasoning") is not None:
            message["reasoning"] = entry["reasoning"]
        response = {"id": f"scripted-{self.index}", "model": self.model_name, "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": entry.get("finish_reason", "tool_calls")}],
                    "usage": entry.get("usage")}
        raw = json.dumps(response).encode()
        latency = 0.001
        usage = openrouter_usage(response["usage"] or {}, elapsed_seconds=latency) if response["usage"] is not None \
            else Usage(elapsed_seconds=latency, measurement_basis="provider_usage_absent")
        rec = {"attempt": len(attempts) + 1, "started_at": time.time(), "request_bytes": body}
        self._emit("started", rec)
        rec.update({"status": "ok", "http_status": 200, "response_bytes": raw, "latency_seconds": latency, "usage": usage})
        attempts.append(rec)
        self._emit("finished", rec)
        return self._finish(request, json.loads(raw), attempts, usage)


def openrouter_model_factory(*, api_key_env: str = "OPENROUTER_API_KEY") -> Callable[[Any], OpenRouterChatModel]:
    """Execute the declared route and safeguards; each attempt applies its remaining allowance cap."""
    def factory(config: Any) -> OpenRouterChatModel:
        policy = config.worker_request_policy
        return OpenRouterChatModel(model_name=config.model, model_settings=dict(config.model_settings),
            api_key_env=api_key_env, max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
            timeout_seconds=policy.timeout_seconds, usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)
    return factory
