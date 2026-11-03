"""TypeSafe Jev (System One classifier) client with receipts, caching and a budget guard.

Jev returns typed answers with calibrated probabilities and charges only input
tokens. Every request/response is stored in the output ArtifactStore under a
record keyed by the canonical request hash and an explicit repeat index, so
reruns reuse identical completed calls and reproducibility repeats are
deliberate, separately recorded calls. Missing usage stays missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from ...contracts import canonical_json, utc_now

# Routes: OpenRouter's decisions endpoint (default; one credential with the workers) or TypeSafe
# direct (only to reuse receipts recorded before 2026-09-25). Same request/response shape; the
# model string is part of the cached request hash, so the routes never share receipts.
ROUTES = {
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13-20260917", "OPENROUTER_API_KEY"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-1.13.0", "TYPESAFE_API_KEY"),
}
JEV_URL, JEV_MODEL, _ = ROUTES["openrouter"]
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
MAX_ATTEMPTS = 6


class BudgetExceeded(RuntimeError):
    pass


class RequestBlocked(RuntimeError):
    """The provider's edge firewall refused the payload (HTTP 403); nothing was evaluated."""


@dataclass
class JevAnswer:
    answers: Mapping[str, Any]
    model: str | None
    input_tokens: int | None
    cost_usd: float | None
    cached: bool
    call_record_id: str


class JevClient:
    def __init__(self, store, *, budget_usd: float, route: str = "openrouter", timeout: float = 120.0):
        self.store = store
        self.budget = budget_usd
        self.route = route
        self.url, self.model, api_key_env = ROUTES[route]
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.key = os.environ.get(api_key_env)
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self.spent = self._prior_spend()
        self.calls = 0
        self.cache_hits = 0

    def _prior_spend(self) -> float:
        total = 0.0
        for row in self.store.list_records("jev_call"):
            record = self.store.get_json(_ref(row["ref"]))
            total += record.get("cost_usd") or 0.0
        return total

    def ask(self, state: Any, questions: Mapping[str, Any], *, purpose: str, repeat: int = 0) -> JevAnswer:
        body = {"model": self.model, "state": state, "questions": dict(questions)}
        raw = canonical_json(body)
        digest = hashlib.sha256(raw).hexdigest()
        record_id = f"{digest}:r{repeat}"
        with self._lock:
            gate = self._inflight.setdefault(record_id, threading.Lock())
        with gate:  # identical concurrent requests: the second reuses the first's receipt
            return self._ask(raw, digest, record_id, purpose, repeat)

    def _ask(self, raw: bytes, digest: str, record_id: str, purpose: str, repeat: int) -> JevAnswer:
        try:
            saved = self.store.get_record("jev_call", record_id)
        except KeyError:
            saved = None
        if saved is not None and saved.get("status") == "ok":
            response = self.store.get_json(_ref(saved["response_ref"]))
            with self._lock:
                self.cache_hits += 1
            return JevAnswer(response["answers"], response.get("model"), saved.get("input_tokens"),
                             saved.get("cost_usd"), True, record_id)
        if not self.key:
            raise RuntimeError(f"{self.api_key_env} unavailable")
        estimate = len(raw) / 3.2 * USD_PER_INPUT_TOKEN
        with self._lock:
            if self.spent + estimate > self.budget:
                raise BudgetExceeded(f"spent ${self.spent:.4f} + estimate ${estimate:.4f} > budget ${self.budget:.2f}")
            self.spent += estimate  # reserve; corrected after the response
        request_ref = self.store.put_bytes(raw, media_type="application/json")
        attempts: list[dict[str, Any]] = []
        response = None
        error_body = None
        started = time.monotonic()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            status, payload, headers = _post(self.url, raw, self.key, self.timeout)
            attempts.append({"attempt": attempt, "http_status": status, "at": utc_now()})
            if status == 200:
                response = payload
                break
            error_body = (payload or b"")[:600].decode("utf-8", "replace")
            if status in (408, 409, 425, 429) or status >= 500 or status == 0:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(30, 2 ** attempt)
                time.sleep(delay)
                continue
            break
        elapsed = time.monotonic() - started
        parsed = None
        if response is not None:
            parsed = json.loads(response)
        usage = (parsed or {}).get("usage") or {}
        input_tokens = usage.get("input_tokens") if isinstance(usage.get("input_tokens"), int) else None
        billed = usage.get("cost")
        if isinstance(billed, (int, float)) and not isinstance(billed, bool):
            cost, cost_basis = float(billed), "provider-reported usage.cost"
        else:
            cost = input_tokens * USD_PER_INPUT_TOKEN if input_tokens is not None else None
            cost_basis = "input_tokens * $0.042/Mtok (published price); output free"
        record = {
            "purpose": purpose, "repeat": repeat, "request_sha256": digest, "request_ref": _jsonable_ref(request_ref),
            "response_ref": _jsonable_ref(self.store.put_bytes(response, media_type="application/json")) if response else None,
            "status": "ok" if parsed is not None and "answers" in parsed else "error",
            "attempts": attempts, "elapsed_seconds": elapsed, "model": (parsed or {}).get("model"),
            "input_tokens": input_tokens, "output_tokens": usage.get("output_tokens"),
            "cost_usd": cost, "cost_basis": cost_basis, "route": self.route, "url": self.url,
            "recorded_at": utc_now(), "error_body_prefix": error_body if response is None else None,
        }
        with self._lock:
            self.spent += (cost if cost is not None else estimate) - estimate
            self.calls += 1
        if record["status"] == "ok":
            self.store.put_record("jev_call", record_id, record)
            return JevAnswer(parsed["answers"], parsed.get("model"), input_tokens, cost, False, record_id)
        self.store.put_record("jev_call_failed", f"{record_id}:{utc_now()}", record)
        if attempts and attempts[-1]["http_status"] == 403:
            raise RequestBlocked(f"Jev request blocked (403): {purpose}")
        raise RuntimeError(f"Jev call failed after {len(attempts)} attempts: {attempts[-1]}")


def _post(url: str, body: bytes, key: str, timeout: float) -> tuple[int, bytes | None, Mapping[str, str]]:
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers or {})
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, None, {}


def _jsonable_ref(ref) -> dict[str, Any]:
    return {"sha256": ref.sha256, "size_bytes": ref.size_bytes, "media_type": ref.media_type,
            "visibility": str(ref.visibility)}


def _ref(raw):
    from ...contracts import ArtifactRef, from_jsonable
    return from_jsonable(ArtifactRef, raw)
