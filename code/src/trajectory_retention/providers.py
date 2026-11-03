"""Private observer transport with durable request/response and cost receipts.

No automatic retries: a timeout can already have incurred provider charges.
Started calls without a finished receipt are uncertain, never zero-cost calls.
Transport injection supports deterministic failure tests without paid requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping

from .accounting import CostLedger, openrouter_usage
from .contracts import ArtifactRef, Usage, canonical_json, from_jsonable, new_id, to_jsonable, utc_now
from .observers.client import CallReceipt, ModelRequest
from .store import ArtifactStore
from .persistence import persist, preserve_failure
from .provider_policy import (REJECTION_RULE, RETRY_RULE, generation_identity, pinned_provider,
                              provider_identity_matches, rejection_usage, retry_plan)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def capture_route_metadata(store: ArtifactStore, model: str, *, campaign_id: str) -> dict:
    """Save the public model/endpoint description before a campaign's first call.

    A catalog record is provenance, not proof that an alias denotes immutable
    weights. Actual returned route/version evidence must still be recorded.
    """
    from urllib.parse import quote
    url = f"https://openrouter.ai/api/v1/models/{quote(model, safe='/')}/endpoints"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    ref = store.put_bytes(raw, media_type="application/json")
    parsed = json.loads(raw)
    if not isinstance(parsed.get("data"), dict):
        raise ValueError("OpenRouter returned no route metadata")
    record = {"requested_model": model, "url": url, "fetched_at": utc_now(), "raw_metadata_ref": ref,
              "limitation": "Alias metadata does not establish immutable weights or guarantee per-call backend identity."}
    store.put_record("campaign_model_metadata", campaign_id, record)
    return to_jsonable(record)


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes
    headers: Mapping[str, str]


def _post(url: str, body: bytes, key: str, timeout: float) -> HttpResult:
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return HttpResult(response.status, response.read(), dict(response.headers))
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), dict(exc.headers))


def _get(url: str, key: str, timeout: float) -> HttpResult:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return HttpResult(response.status, response.read(), dict(response.headers))
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read(), dict(exc.headers))


def _safe_headers(headers):
    # Preserve retry/identity/phase evidence, never credentials echoed by a
    # transport. Original response body remains a separate private artifact.
    return {k: v for k, v in headers.items() if k.lower() not in {
        "authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"}}


@contextmanager
def _call_lock(store, call_id):
    path = store.root / "locks" / ("provider-" + hashlib.sha256(call_id.encode()).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class OpenRouterObserverClient:
    def __init__(self, store: ArtifactStore, *, api_key_env: str = "OPENROUTER_API_KEY",
                 timeout_seconds: float = 180, transport: Callable = _post,
                 usage_transport: Callable = _get, usage_recovery_timeout_seconds: float = 30):
        self.store = store
        self.api_key_env = api_key_env
        self.timeout = timeout_seconds
        self.transport = transport
        self.usage_transport = usage_transport
        self.usage_recovery_timeout = usage_recovery_timeout_seconds

    def complete(self, request: ModelRequest, *, call_id: str | None = None) -> tuple[str | None, CallReceipt]:
        if set(request.settings) & {"messages", "model", "stream", "tools"}:
            raise ValueError("Observer settings cannot override source messages/model or enable tools/streaming")
        body = canonical_json({"model": request.model, "messages": request.messages,
                               **request.settings, "stream": False})
        call_id = new_id("observer_call") if call_id is None else call_id
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Provider call ID must be a nonempty string")
        with _call_lock(self.store, call_id):
            return self._complete(request, body, call_id)

    def _complete(self, request, body, call_id):
        binding = {"run_id": request.run_id, "target_checkpoint_id": request.target_checkpoint_id,
                   "purpose": request.purpose, "cost_role": request.cost_role,
                   "request_hash": request.request_hash(), "url": OPENROUTER_URL,
                   "timeout_seconds": self.timeout, "usage_recovery_timeout_seconds": self.usage_recovery_timeout,
                   "rejection_accounting_rule": REJECTION_RULE, "retry_rule": RETRY_RULE}
        try:
            saved = self.store.get_record("provider_call_started", call_id)
        except KeyError:
            saved = None
        if saved is not None:
            if (any(saved.get(k) != v for k, v in binding.items())
                    or self.store.get_bytes(from_jsonable(ArtifactRef, saved["request_ref"])) != body):
                raise ValueError("Provider call ID is already bound to a different request or policy")
            try:
                finished = self.store.get_record("provider_call_finished", call_id)
            except KeyError:
                raise RuntimeError("Provider call started without durable completion; uncertain, do not resend") from None
            receipt = from_jsonable(CallReceipt, finished["receipt"])
            if (finished["request_ref"] != saved["request_ref"] or receipt.call_record_id != call_id
                    or receipt.request_hash != request.request_hash() or receipt.requested_model != request.model
                    or receipt.cost_role != request.cost_role):
                raise ValueError("Completed provider receipt does not match its exact request")
            text = None
            if finished.get("response_ref") is not None:
                raw = self.store.get_bytes(from_jsonable(ArtifactRef, finished["response_ref"]))
                if hashlib.sha256(raw).hexdigest() != receipt.response_hash:
                    raise ValueError("Completed provider response hash differs from receipt")
                if receipt.status == "ok":
                    text = json.loads(raw)["choices"][0]["message"]["content"]
                    if not isinstance(text, str):
                        raise ValueError("Completed provider receipt has no text")
            elif receipt.status == "ok":
                raise ValueError("Completed provider receipt has no response artifact")
            CostLedger(self.store).record(run_id=request.run_id, operation_id=call_id,
                role=request.cost_role, usage=receipt.usage, status=receipt.status)
            return text, receipt
        request_ref = persist(lambda: self.store.put_bytes(body, media_type="application/json"), evidence={"request_bytes": body})
        start_record = {
            **binding, "call_id": call_id, "started_at": utc_now(), "request_ref": request_ref,
        }
        persist(lambda: self.store.put_record("provider_call_started", call_id, start_record, run_id=request.run_id),
                evidence=start_record)
        started = time.monotonic()
        text = error = resolved_model = route = fingerprint = response_hash = response_ref = None
        status, http_status, response_headers = "error", None, {}
        usage = Usage(measurement_basis="provider_response_absent")
        result = parsed = primary = lookup = original_usage = None
        generation_id_source = None
        provider_accounting = {"rule": REJECTION_RULE, "classification": "unresolved_generation"}
        key = os.environ.get(self.api_key_env)
        try:
            if not key:
                raise ValueError(f"Credential unavailable: {self.api_key_env}")
            result = self.transport(OPENROUTER_URL, body, key, self.timeout)
            elapsed = time.monotonic() - started
            # Interpret in memory first. A local write failure must not be
            # converted to a provider error or erase already received usage.
            response_hash = hashlib.sha256(result.body).hexdigest()
            http_status = result.status
            response_headers = _safe_headers(result.headers)
            parsed = json.loads(result.body)
            if not isinstance(parsed, dict):
                raise ValueError("Provider response is not an object")
            raw_usage = parsed.get("usage")
            if isinstance(raw_usage, dict):
                usage = openrouter_usage(raw_usage, elapsed_seconds=elapsed)
            else:
                usage = replace(usage, elapsed_seconds=elapsed)
            resolved_model, route = parsed.get("model"), parsed.get("provider")
            fingerprint = parsed.get("system_fingerprint")
            generation_id, generation_id_source = generation_identity(parsed, response_headers)
            original_usage = usage
            if usage.generated_tokens is None and generation_id is not None:
                usage, lookup, primary = self._lookup(request, call_id, generation_id, usage, key)
            usage, provider_accounting = rejection_usage(url=OPENROUTER_URL, request=json.loads(body),
                http_status=http_status, parsed=parsed, usage=usage, headers=response_headers, recovery=lookup)
            if primary is not None:
                raise primary
            if result.status != 200 or "error" in parsed:
                raise ValueError(f"Provider returned HTTP {result.status}; see response artifact")
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("Provider returned no choices")
            if not isinstance(choices[0], dict) or choices[0].get("error") or choices[0].get("finish_reason") == "error":
                raise ValueError("Provider interrupted generation; partial response retained")
            text = choices[0].get("message", {}).get("content")
            if not isinstance(text, str):
                raise ValueError("Provider returned no text content")
            if choices[0].get("finish_reason") == "length":
                raise ValueError("Provider output was truncated; response retained")
            # An explicitly returned conflicting identity is not silently accepted.
            if resolved_model and resolved_model != request.model:
                raise ValueError("Returned model differs from requested model; inspect recorded identities")
            pin = pinned_provider(json.loads(body))
            if pin and (not isinstance(route, str) or not provider_identity_matches(pin, route)):
                raise ValueError("Returned provider differs from the pinned route")
            status = "ok"
        except BaseException as exc:
            if not isinstance(exc, Exception) or hasattr(exc, "persistence_evidence"):
                primary = exc
            status = "interrupted" if primary is not None else "timeout" if isinstance(exc, TimeoutError) else "error"
            # Do not persist request objects/authorization headers from exceptions.
            error = f"{type(exc).__name__}: {str(exc)}"
            if key:
                error = error.replace(key, "[CREDENTIAL REDACTED]")
            text = None
        elapsed = time.monotonic() - started
        if usage.elapsed_seconds is None:
            usage = Usage(**{**to_jsonable(usage), "elapsed_seconds": elapsed})
        pin = pinned_provider(json.loads(body))
        decision = retry_plan(attempt=1, max_attempts=2, backoff_seconds=0,
            max_wait_seconds=self.timeout, headers=response_headers, parsed=parsed, now=time.time(), usage=usage,
            retryable=status not in {"ok", "interrupted"} and http_status not in {400, 401, 402, 403, 404, 413, 422}
                and not (resolved_model is not None and resolved_model != request.model)
                and not (http_status == 200 and pin and (not isinstance(route, str)
                    or not provider_identity_matches(pin, route))))
        decision["owner"] = "external_observer_queue; one physical attempt in this receipt"
        receipt = CallReceipt(call_record_id=call_id, cost_role=request.cost_role,
                              requested_model=request.model, resolved_model=resolved_model,
                              route=route, status=status, usage=usage,
                              request_hash=request.request_hash(), response_hash=response_hash,
                              error=error, fingerprint=fingerprint)
        evidence = {"receipt": receipt.as_record(), "response_bytes": result.body if result else None,
                    "http_status": http_status, "response_headers": response_headers, "request_ref": request_ref}
        try:
            if result is not None:
                response_ref = persist(lambda: self.store.put_bytes(result.body), evidence=evidence)
            finish_record = {
                "receipt": receipt.as_record(), "request_ref": request_ref, "response_ref": response_ref,
                "http_status": http_status, "response_headers": response_headers,
                "provider_accounting": provider_accounting, "retry_decision": decision,
                "generation_id_source": generation_id_source, "usage_recovery": lookup,
                "original_usage": to_jsonable(original_usage) if original_usage != usage else None,
                "finished_at": utc_now(),
            }
            persist(lambda: self.store.put_record("provider_call_finished", call_id, finish_record, run_id=request.run_id),
                    evidence=evidence)
            # Reuse the immutable allocation when observer artifacts are saved.
            CostLedger(self.store).record(run_id=request.run_id, operation_id=call_id,
                                         role=request.cost_role, usage=usage, status=status)
        except BaseException as exc:
            preserve_failure(exc, evidence)
            if primary is not None:
                preserve_failure(primary, evidence)
                primary.add_note(f"Received observer receipt persistence also failed: {type(exc).__name__}")
                raise primary
            raise
        if primary is not None:
            raise primary
        return text, receipt

    def _lookup(self, request, call_id, generation_id, usage, key):
        """One metadata read, never a generation retry or proof from a 404."""
        from urllib.parse import urlencode
        operation_id = call_id + ":usage_lookup"
        lookup = {"operation_id": operation_id, "run_id": request.run_id,
                  "generation_id": generation_id, "status": "unavailable",
                  "url": "https://openrouter.ai/api/v1/generation?" + urlencode({"id": generation_id}),
                  "timeout_seconds": self.usage_recovery_timeout, "started_at": utc_now()}
        persist(lambda: self.store.put_record("provider_usage_lookup_started", operation_id, lookup,
                                             run_id=request.run_id), evidence=lookup)
        start = time.monotonic()
        primary = response = None
        recovered = usage
        try:
            response = self.usage_transport(lookup["url"], key, self.usage_recovery_timeout)
            lookup.update(http_status=response.status, response_headers=_safe_headers(response.headers))
            if response.status != 200:
                raise ValueError(f"Generation metadata HTTP {response.status}; usage remains unresolved")
            parsed = json.loads(response.body)
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if (not isinstance(data, dict) or data.get("id") != generation_id or data.get("model") != request.model
                    or not isinstance(data.get("finish_reason"), str) or not data["finish_reason"]):
                raise ValueError("Generation metadata is not final and bound to request/model")
            pin = pinned_provider({"provider": request.settings.get("provider")})
            if pin and data.get("provider_name") is not None and (not isinstance(data["provider_name"], str)
                    or not provider_identity_matches(pin, data["provider_name"])):
                raise ValueError("Generation metadata provider differs from pin")
            native = openrouter_usage({"completion_tokens": data.get("native_tokens_completion"),
                "prompt_tokens": data.get("native_tokens_prompt"), "cost": data.get("total_cost"),
                "completion_tokens_details": {"reasoning_tokens": data.get("native_tokens_reasoning")},
                "prompt_tokens_details": {"cached_tokens": data.get("native_tokens_cached")}},
                elapsed_seconds=usage.elapsed_seconds)
            if (native.generated_tokens is None or (native.reasoning_tokens is not None
                    and native.reasoning_tokens > native.generated_tokens)
                    or (usage.reasoning_tokens is not None and usage.reasoning_tokens > native.generated_tokens)):
                raise ValueError("Native generation counters absent or inconsistent")
            updates = {k: getattr(native, k) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                "cached_input_tokens", "generated_tokens", "cost_usd") if getattr(usage, k) is None}
            recovered = replace(usage, **updates, measurement_basis="provider_reported_openrouter_generation_receipt")
            lookup["status"] = "recovered"
        except BaseException as exc:
            if not isinstance(exc, Exception):
                primary = exc
                lookup["status"] = "interrupted"
            lookup["error"] = f"{type(exc).__name__}: {str(exc)}".replace(key, "[CREDENTIAL REDACTED]")
        lookup["elapsed_seconds"] = time.monotonic() - start
        evidence = {**lookup, "response_bytes": response.body if response is not None else None,
                    "usage": to_jsonable(recovered)}
        try:
            if response is not None:
                lookup["response_ref"] = persist(lambda: self.store.put_bytes(response.body), evidence=evidence)
            persist(lambda: self.store.put_record("provider_usage_lookup_finished", operation_id, lookup,
                                                 run_id=request.run_id), evidence=evidence)
            CostLedger(self.store).record(run_id=request.run_id, operation_id=operation_id,
                role="infrastructure", kind="infrastructure", status=lookup["status"],
                usage=Usage(elapsed_seconds=lookup["elapsed_seconds"], measurement_basis="measured_usage_recovery"))
        except BaseException as exc:
            if primary is not None:
                preserve_failure(primary, evidence)
                primary.add_note(f"Usage lookup persistence also failed: {type(exc).__name__}")
            else:
                preserve_failure(exc, evidence)
                primary = exc
        return recovered, lookup, primary
