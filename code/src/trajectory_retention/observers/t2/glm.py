"""GLM-5.3 Flash client for T2.2 checks: OpenRouter, pinned to GMICloud fp8, with receipts and a budget guard.

Used only by analysis-side checks (the recovery probe). It never sees grades,
detector output or reference material; callers pass it packet text only.

Rules:

* Every request goes to OpenRouter chat completions for ``z-ai/glm-5.3-flash``
  with the provider pinned to the single endpoint ``gmicloud/fp8``
  (``only`` = ``order`` = [that slug], ``allow_fallbacks: false``,
  ``quantizations: ["fp8"]``, ``require_parameters: true``) and
  ``reasoning.effort = "max"``. The pin is part of the request body, hence of
  the cached request hash.
* Every receipt records the provider OpenRouter reports as having served the
  call. A response from any other provider is stored as a failure and raises
  ``ProviderMismatch``; it is never returned as an answer.
* Receipts live in the output ArtifactStore keyed by the canonical request hash
  plus an explicit repeat index, so identical completed calls are reused and
  repeats are deliberate, separately recorded calls.
* Cost is ``usage.cost`` as reported. When it is missing it stays missing on
  the receipt, and the pre-call reservation stays charged against the budget.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...contracts import canonical_json, utc_now
from ...provider_policy import provider_identity_matches
from .jev import BudgetExceeded, _jsonable_ref, _ref

GLM_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
GLM_MODEL = "z-ai/glm-5.3-flash"
PROVIDER_SLUG = "gmicloud/fp8"
EXPECTED_PROVIDER = "GMICloud"
PROVIDER_PIN = {"only": [PROVIDER_SLUG], "order": [PROVIDER_SLUG], "allow_fallbacks": False,
                "quantizations": ["fp8"], "require_parameters": True}
REASONING = {"effort": "max"}
MAX_TOKENS = 32768  # development setting; effort "max" lets reasoning use most of it
API_KEY_ENV = "OPENROUTER_API_KEY"
# GMICloud list prices for this model in the OpenRouter catalog (2026-09-26). Used only to reserve budget
# before a call; the charged cost is the provider-reported usage.cost.
USD_PER_PROMPT_TOKEN = 0.09 / 1_000_000
USD_PER_COMPLETION_TOKEN = 0.30 / 1_000_000
MAX_ATTEMPTS = 4


class ProviderMismatch(RuntimeError):
    """OpenRouter reported a provider other than the pinned one; the answer is not used."""


@dataclass
class GlmAnswer:
    text: str
    reasoning: str | None
    provider: str | None
    model: str | None
    finish_reason: str | None
    usage: Mapping[str, Any]
    cost_usd: float | None
    cached: bool
    call_record_id: str


class GlmClient:
    def __init__(self, store, *, budget_usd: float, max_tokens: int = MAX_TOKENS, timeout: float = 600.0,
                 reasoning: Mapping[str, Any] | None = None, model: str = GLM_MODEL,
                 provider_slug: str | None = None, expected_provider: str = EXPECTED_PROVIDER,
                 usd_per_token: tuple[float, float] = (USD_PER_PROMPT_TOKEN, USD_PER_COMPLETION_TOKEN)):
        """Pinned OpenRouter chat client. Defaults are the ratified GLM-5.3 Flash on GMICloud fp8; other pinned
        models (e.g. for the linker bake-off) pass ``model``, ``provider_slug`` and ``expected_provider``."""
        self.reasoning = dict(reasoning or REASONING)  # part of the request, so of the receipt key
        self.slug = provider_slug or PROVIDER_SLUG
        self.pin = dict(PROVIDER_PIN) if provider_slug is None else {
            "only": [provider_slug], "order": [provider_slug], "allow_fallbacks": False, "require_parameters": True,
            **({"quantizations": [provider_slug.split("/")[-1]]} if "/" in provider_slug else {})}
        self.expected_provider = expected_provider
        self.usd_in, self.usd_out = usd_per_token
        self.store = store
        self.budget = budget_usd
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model = model
        self.key = os.environ.get(API_KEY_ENV)
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self.spent = self._prior_spend()
        self.calls = 0
        self.cache_hits = 0

    def _prior_spend(self) -> float:
        total = 0.0
        for kind in ("glm_call", "glm_call_failed"):
            for row in self.store.list_records(kind):
                record = self.store.get_json(_ref(row["ref"]))
                total += record.get("cost_usd") or 0.0
        return total

    def request_body(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int | None = None) -> dict[str, Any]:
        return {"model": self.model, "messages": [dict(m) for m in messages], "provider": dict(self.pin),
                "reasoning": dict(self.reasoning), "max_tokens": max_tokens or self.max_tokens,
                "usage": {"include": True}, "stream": False}

    def complete(self, messages: Sequence[Mapping[str, str]], *, purpose: str, repeat: int = 0,
                 max_tokens: int | None = None) -> GlmAnswer:
        body = self.request_body(messages, max_tokens=max_tokens)
        raw = canonical_json(body)
        digest = hashlib.sha256(raw).hexdigest()
        record_id = f"{digest}:r{repeat}"
        with self._lock:
            gate = self._inflight.setdefault(record_id, threading.Lock())
        with gate:
            return self._complete(raw, body, digest, record_id, purpose, repeat)

    def _complete(self, raw: bytes, body: dict, digest: str, record_id: str, purpose: str, repeat: int) -> GlmAnswer:
        try:
            saved = self.store.get_record("glm_call", record_id)
        except KeyError:
            saved = None
        if saved is not None and saved.get("status") == "ok":
            parsed = self.store.get_json(_ref(saved["response_ref"]))
            with self._lock:
                self.cache_hits += 1
            return _answer(parsed, saved, True, record_id)
        if not self.key:
            raise RuntimeError(f"{API_KEY_ENV} unavailable")
        estimate = len(raw) / 3.2 * self.usd_in + body["max_tokens"] * self.usd_out
        with self._lock:
            if self.spent + estimate > self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.4f} + estimate ${estimate:.4f} > budget ${self.budget:.2f}")
            self.spent += estimate  # reserve; corrected after the response
        request_ref = self.store.put_bytes(raw, media_type="application/json")
        attempts: list[dict[str, Any]] = []
        response = error_body = None
        started = time.monotonic()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status, payload, headers = _post(GLM_URL, raw, self.key, self.timeout)
            attempts.append({"attempt": attempt, "http_status": status, "at": utc_now(),
                             **({"transport_error": headers[TRANSPORT_ERROR]} if TRANSPORT_ERROR in headers else {})})
            if status == 200 and payload:
                response = payload
                break
            error_body = (payload or b"")[:600].decode("utf-8", "replace")
            if status in (408, 409, 425, 429) or status >= 500 or status == 0:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(30, 2 ** attempt)
                time.sleep(delay)
                continue
            break
        parsed = None
        if response is not None:
            try:
                parsed = json.loads(response)
            except ValueError:
                error_body = response[:600].decode("utf-8", "replace")
        usage = (parsed or {}).get("usage") or {}
        billed = usage.get("cost")
        cost = float(billed) if isinstance(billed, (int, float)) and not isinstance(billed, bool) else None
        choice = ((parsed or {}).get("choices") or [{}])[0]
        served = (parsed or {}).get("provider")
        matches = isinstance(served, str) and provider_identity_matches(self.slug, served)
        has_choice = bool((parsed or {}).get("choices")) and "error" not in (parsed or {})
        status = "ok" if has_choice and matches else "provider_mismatch" if has_choice else "error"
        record = {
            "purpose": purpose, "repeat": repeat, "request_sha256": digest, "request_ref": _jsonable_ref(request_ref),
            "response_ref": _jsonable_ref(self.store.put_bytes(response, media_type="application/json")) if response else None,
            "status": status, "attempts": attempts, "elapsed_seconds": time.monotonic() - started,
            "model_requested": self.model, "model": (parsed or {}).get("model"), "generation_id": (parsed or {}).get("id"),
            "provider_pin": dict(self.pin), "served_provider": served, "provider_matches_pin": matches,
            "reasoning_requested": dict(self.reasoning), "max_tokens": body["max_tokens"],
            "finish_reason": choice.get("finish_reason"), "native_finish_reason": choice.get("native_finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost_usd": cost, "cost_basis": "provider-reported usage.cost" if cost is not None else "missing",
            "url": GLM_URL, "recorded_at": utc_now(),
            "error": error_body or (parsed or {}).get("error"),
        }
        with self._lock:
            self.spent += (cost if cost is not None else estimate) - estimate
            self.calls += 1
        if status == "ok":
            self.store.put_record("glm_call", record_id, record)
            return _answer(parsed, record, False, record_id)
        self.store.put_record("glm_call_failed", f"{record_id}:{utc_now()}", record)
        if status == "provider_mismatch":
            raise ProviderMismatch(f"served by {served!r}, pinned {self.slug}: {purpose}")
        raise RuntimeError(f"GLM call failed after {len(attempts)} attempts: {attempts[-1]}")

    def lookup_generation(self, generation_id: str, *, tries: int = 5) -> dict[str, Any] | None:
        """OpenRouter's own generation record (provider name, cost); None when not yet available."""
        url = f"{GENERATION_URL}?id={generation_id}"
        for n in range(tries):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    return (json.loads(response.read()) or {}).get("data")
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
                time.sleep(2 + 2 * n)
        return None

    def verify_pin(self, *, repeat: int = 0) -> dict[str, Any]:
        """One cheap call; reports which provider served it (response body and OpenRouter's generation record).

        The generation record lists every upstream attempt; one GMICloud attempt means no fallback was tried.
        FP8 is bound by the request's endpoint slug and quantization filter; no response field attests it."""
        answer = self.complete([{"role": "user", "content": "Reply with the single word: ok"}],
                               purpose="t2.2-glm-pin-check", repeat=repeat, max_tokens=2048)
        receipt = self.store.get_record("glm_call", answer.call_record_id)
        lookup = self.lookup_generation(receipt["generation_id"]) if receipt.get("generation_id") and self.key else None
        return {"served_provider": answer.provider, "provider_matches_pin": receipt["provider_matches_pin"],
                "generation_record_provider": (lookup or {}).get("provider_name"),
                "generation_record_model": (lookup or {}).get("model"),
                "generation_record_cost": (lookup or {}).get("total_cost"),
                "generation_record_attempts": [{k: r.get(k) for k in ("provider_name", "status", "endpoint_id")}
                                               for r in (lookup or {}).get("provider_responses") or []],
                "model": answer.model, "finish_reason": answer.finish_reason, "cost_usd": answer.cost_usd,
                "cached": answer.cached, "call_record_id": answer.call_record_id, "pin": dict(self.pin),
                "reasoning": dict(self.reasoning)}


def _answer(parsed: Mapping[str, Any], record: Mapping[str, Any], cached: bool, record_id: str) -> GlmAnswer:
    message = ((parsed.get("choices") or [{}])[0].get("message") or {})
    return GlmAnswer(text=message.get("content") or "", reasoning=message.get("reasoning"),
                     provider=parsed.get("provider"), model=parsed.get("model"),
                     finish_reason=record.get("finish_reason"), usage=dict(parsed.get("usage") or {}),
                     cost_usd=record.get("cost_usd"), cached=cached, call_record_id=record_id)


TRANSPORT_ERROR = "x-local-transport-error"  # set only on status 0; never a real response header


def _post(url: str, body: bytes, key: str, timeout: float) -> tuple[int, bytes | None, Mapping[str, str]]:
    """POST once. A response that never arrives or is cut off (timeout, connection error, or a body cut short, such as
    ``http.client.IncompleteRead`` or ``RemoteDisconnected``) returns status 0, which the caller retries like a
    timeout; the error's type is reported under ``TRANSPORT_ERROR``."""
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except (http.client.HTTPException, OSError):
            payload = None
        return exc.code, payload, dict(exc.headers or {})
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
        return 0, None, {TRANSPORT_ERROR: type(exc).__name__}
