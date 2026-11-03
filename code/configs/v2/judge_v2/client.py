"""Judge calls for stall judge v2: settings binding, retries then no_answer (user Q8), receipts, spend guard, and a
scripted transport for engine checks (no request leaves the machine).

A call is one vote (or the one type call). Each attempt goes through a transport; an attempt that raises, returns an
error body or returns no choices is retried, up to `attempts` in all (settings; 3), then the vote is `no_answer`.
An answer that arrives is final: the parsed verdict, `refused` (refusal finish reason or refusal field) or
`unparseable`. Nothing but `stall` is a stall vote. Receipts are content-addressed by the caller and reused on rerun
(a `no_answer` receipt is re-asked only with retry_no_answer). Saved: the request minus its messages, the sha256 of
system and user text, every attempt's response or error, usage and cost; the API key is never saved or printed.

Parsing (declared reading): the answer is the LAST JSON object in the content that holds the answer key (so an
answer after a continued transcript still counts); the verdict must be one of the allowed values. onset_step: an
integer, or a string that holds exactly one integer; null or absent = missing; anything else = unreadable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REFUSALS = {"refusal", "content_filter"}
FINAL = {"unparseable", "refused", "no_answer"}
URL = "https://openrouter.ai/api/v1/chat/completions"

# A socket timeout alone is not an elapsed deadline: periodic HTTP data can keep
# a response alive indefinitely. A child process gives each attempt a hard bound.
TRANSPORT_CODE = r'''
import json, os, sys, urllib.request, urllib.error
body = sys.stdin.buffer.read()
req = urllib.request.Request(sys.argv[1], data=body, headers={
    "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
    "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=float(sys.argv[2])) as response:
        sys.stdout.write(response.read().decode())
except urllib.error.HTTPError as error:
    sys.stdout.write(json.dumps({"transport_error": str(error),
                                "error_body": error.read().decode("utf-8", "replace")[:4000]}))
'''


@dataclass(frozen=True)
class Settings:
    path: str
    version: str
    model: str
    provider: dict
    provider_name: str
    reasoning: dict
    max_tokens: int
    attempts: int
    price_in: float
    price_out: float
    transport: str = "openrouter"

    @classmethod
    def load(cls, path: Path) -> "Settings":
        s = json.loads(Path(path).read_text())
        return cls(str(path), s["version"], s["model"], s["provider"], s["provider_name"], s["reasoning"],
                   int(s["max_tokens"]), int(s.get("attempts", 3)), float(s["guard_price_per_token"]["in"]),
                   float(s["guard_price_per_token"]["out"]), s.get("transport", "openrouter"))

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9.]+", "-", self.model.lower()).strip("-")

    def body(self, system: str, user: str) -> dict:
        return {"model": self.model, "messages": [{"role": "system", "content": system},
                                                   {"role": "user", "content": user}],
                "provider": self.provider, "reasoning": self.reasoning, "max_tokens": self.max_tokens,
                "usage": {"include": True}}

    def worst_case_usd(self, chars: int) -> float:
        """The guard's reservation for one call: input at 2.5 chars per token (conservative) plus a full response."""
        return chars / 2.5 * self.price_in + self.max_tokens * self.price_out


class TransportError(RuntimeError):
    def __init__(self, message: str, body: str):
        super().__init__(message)
        self.body = body

    def read(self) -> bytes:
        return self.body.encode()


class OpenRouter:
    """The real transport. Only constructed for a paid run; reads OPENROUTER_API_KEY from the environment."""
    scripted = False

    def __init__(self, timeout: int = 900):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is not set (source the key file into the environment first)")
        self.timeout = timeout

    def post(self, body: dict, ctx: dict) -> dict:
        result = subprocess.run([sys.executable, "-c", TRANSPORT_CODE, URL, str(self.timeout)],
                                input=json.dumps(body), capture_output=True, text=True, timeout=self.timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        response = json.loads(result.stdout)
        if response.get("transport_error"):
            raise TransportError(response["transport_error"], response.get("error_body", ""))
        return response

    def backoff(self, attempt: int) -> None:
        time.sleep(5 * (attempt + 1))


class Scripted:
    """Fixed answers instead of a model. Spec (JSON):
      {"default": {"stall": [a1, a2, a3], "type": a},
       "points": {"<run-id substring>@<point>": {"stall": [...], "type": a}}}
    An answer a is a string (the response content), or {"content": s, "finish_reason": f, "refusal": r}, or
    {"error": "<text>"} (the attempt raises), or {"error_body": {...}} (the attempt returns an error body), or a list
    of such outcomes, one per attempt (the last one repeats). Every request body sent is appended to `log` (JSONL)
    so a check can read exactly what the judge was sent."""
    scripted = True

    def __init__(self, spec: dict, log: Path, provider_name: str):
        self.spec, self.log, self.provider_name = spec, log, provider_name
        self.lock = threading.Lock()
        log.parent.mkdir(parents=True, exist_ok=True)

    def _answer(self, ctx: dict):
        for key, v in self.spec.get("points", {}).items():
            sub, _, point = key.rpartition("@")
            if sub in ctx["run_id"] and int(point) == ctx["point"]:
                break
        else:
            v = self.spec["default"]
        if ctx["call"] == "type":
            a = v.get("type", self.spec["default"]["type"])
        else:
            a = v.get("stall", self.spec["default"]["stall"])[ctx["vote"] - 1]
        if isinstance(a, list):
            a = a[min(ctx["attempt"], len(a) - 1)]
        return a

    def post(self, body: dict, ctx: dict) -> dict:
        with self.lock, self.log.open("a") as f:
            f.write(json.dumps({"ctx": ctx, "body": body}) + "\n")
        a = self._answer(ctx)
        if isinstance(a, dict) and "error" in a:
            raise RuntimeError(f"scripted transport error: {a['error']}")
        if isinstance(a, dict) and "error_body" in a:
            return {"error": a["error_body"]}
        a = {"content": a} if isinstance(a, str) else a
        n_in = sum(len(m["content"]) for m in body["messages"]) // 4
        return {"provider": self.provider_name, "model": body["model"], "scripted": True,
                "choices": [{"finish_reason": a.get("finish_reason", "stop"),
                             "message": {"role": "assistant", "content": a.get("content"),
                                         "refusal": a.get("refusal")}}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": 50, "cost": 0.0}}

    def backoff(self, attempt: int) -> None:
        return None


class Budget:
    """A hard spend guard across one command: a call is not made if its worst case could pass the cap."""

    def __init__(self, cap_usd: float, spent: float = 0.0):
        self.cap, self.spent, self.reserved = cap_usd, spent, 0.0
        self.lock = threading.Lock()

    def reserve(self, est: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + est > self.cap:
                return False
            self.reserved += est
            return True

    def settle(self, est: float, actual: float) -> None:
        with self.lock:
            self.reserved -= est
            self.spent += actual


def receipts_cost(root: Path) -> float:
    tot = 0.0
    for f in root.rglob("*.json") if root.exists() else ():
        try:
            tot += float(json.loads(f.read_text()).get("cost_usd") or 0.0)
        except (ValueError, AttributeError):
            continue
    return tot


def _last_object(content: str, key: str) -> dict | None:
    dec = json.JSONDecoder()
    found = None
    for m in re.finditer(r"\{", content):
        try:
            obj, _ = dec.raw_decode(content, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and key in obj:
            found = obj
    return found


def parse_onset(v: Any) -> tuple[int | None, str]:
    if v is None:
        return None, "missing"
    if isinstance(v, bool):
        return None, "unreadable"
    if isinstance(v, int):
        return v, "given"
    if isinstance(v, float) and v.is_integer():
        return int(v), "given"
    if isinstance(v, str):
        nums = re.findall(r"-?\d+", v)
        if len(nums) == 1:
            return int(nums[0]), "given"
        if not v.strip() or v.strip().lower() in ("null", "none"):
            return None, "missing"
    return None, "unreadable"


def parse(content: str | None, key: str, allowed: set[str]) -> dict:
    if not content:
        return {"value": None}
    obj = _last_object(content, key)
    if obj is None:
        return {"value": None}
    v = obj.get(key)
    out = {"value": v if v in allowed else None, "evidence": obj.get("evidence")}
    if key == "verdict":
        out["onset_raw"] = obj.get("onset_step")
        out["onset"], out["onset_status"] = parse_onset(obj.get("onset_step"))
    return out


def ask(settings: Settings, transport, budget: Budget, path: Path, system: str, user: str, key: str,
        allowed: set[str], ctx: dict, *, retry_no_answer: bool = False) -> dict:
    """One vote (or the type call). Returns the receipt; `final` is the vote."""
    previous = None
    if path.exists():
        rec = json.loads(path.read_text())
        if not (retry_no_answer and rec.get("final") == "no_answer"):
            return rec
        previous = rec
    est = settings.worst_case_usd(len(system) + len(user))
    if not budget.reserve(est):
        return {"final": "skipped_spend_guard", "reserve_usd": est}
    body = settings.body(system, user)
    rec = {"version": "judge-v2-receipt.v1", "settings": settings.path, "settings_version": settings.version,
           "model": settings.model, "provider_pin": settings.provider, "scripted": transport.scripted,
           "request_body_without_messages": {k: v for k, v in body.items() if k != "messages"},
           "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
           "user_sha256": hashlib.sha256(user.encode()).hexdigest(), "request_chars": len(system) + len(user),
           "ctx": ctx, "attempts": []}
    if previous is not None:
        rec["previous_rounds"] = previous.get("previous_rounds", []) + [
            {k: v for k, v in previous.items() if k != "previous_rounds"}]
    cost = 0.0
    try:
        for attempt in range(settings.attempts):
            t0 = time.time()
            try:
                resp = transport.post(body, dict(ctx, attempt=attempt))
            except Exception as e:  # noqa: BLE001 - recorded, then retried
                err = getattr(e, "read", lambda: b"")()
                rec["attempts"].append({"started": t0, "elapsed": time.time() - t0, "error": repr(e)[:2000],
                                        "error_body": err.decode("utf-8", "replace")[:4000] if err else None})
                transport.backoff(attempt)
                continue
            rec["attempts"].append({"started": t0, "elapsed": time.time() - t0, "response": resp})
            cost += float(((resp.get("usage") or {}).get("cost")) or 0.0)
            if resp.get("error") or not resp.get("choices"):
                transport.backoff(attempt)
                continue
            if resp.get("provider") != settings.provider_name:
                rec["provider_violation"] = resp.get("provider")
            choice = resp["choices"][0]
            msg = choice.get("message") or {}
            p = parse(msg.get("content"), key, allowed)
            refused = p["value"] is None and (choice.get("finish_reason") in REFUSALS
                                              or choice.get("native_finish_reason") in REFUSALS
                                              or bool(msg.get("refusal")))
            u = resp.get("usage") or {}
            rec.update(parsed=p["value"], evidence=p.get("evidence"),
                       final=p["value"] or ("refused" if refused else "unparseable"),
                       finish_reason=choice.get("finish_reason"), usage=u,
                       **({k: p.get(k) for k in ("onset_raw", "onset", "onset_status")} if key == "verdict" else {}))
            break
        else:
            rec["final"] = "no_answer"
    finally:
        budget.settle(est, cost)
        rec["cost_usd"] = cost + float((previous or {}).get("cost_usd") or 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1))
    return rec


def make_transport(settings: Settings, scripted: Path | None, log: Path) -> Any:
    if scripted is not None:
        return Scripted(json.loads(Path(scripted).read_text()), log, settings.provider_name)
    if settings.transport == "codex-subscription":
        from trajectory_retention.observers.codex_subscription import CodexSubscription
        return CodexSubscription(log.parent / "codex-sessions")
    if settings.transport != "openrouter":
        raise ValueError(f"unknown judge transport: {settings.transport}")
    return OpenRouter()


Transport = Callable[[dict, dict], dict]
