"""Narrow OpenRouter rejection evidence and bounded retry timing.

An empty response, an error body, or a missing generation lookup is never
zero-generation evidence. The rule below additionally requires the documented
pre-generation HTTP phase and an explicit rate-limit category on a pinned,
non-streaming OpenRouter request. Monetary usage is never inferred here.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
from typing import Any, Mapping

from .contracts import Usage

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
REJECTION_RULE = "openrouter-pre-generation-rate-limit.v2"
RETRY_RULE = "known-usage-pinned-retry-after.v1"
ERROR_CONTRACT = "https://openrouter.ai/docs/api_reference/errors-and-debugging"
LIMIT_CONTRACT = "https://openrouter.ai/docs/api_reference/limits"


def pinned_provider(request: Mapping[str, Any]) -> str | None:
    if request.get("models"):
        return None
    route = request.get("provider")
    if not isinstance(route, dict) or route.get("allow_fallbacks") is not False:
        return None
    only = route.get("only")
    if not isinstance(only, list) or len(only) != 1 or not isinstance(only[0], str) or not only[0]:
        return None
    if route.get("order", only) != only:
        return None
    return only[0]


def provider_identity_matches(pin: str, returned: str) -> bool:
    # OpenRouter reports display names ("GMICloud"), while routing uses slugs
    # ("gmicloud/fp8"). The endpoint/quantization remains bound by the request;
    # a display-name match alone is not a claim of immutable weights.
    def name(value):
        return "".join(c for c in value.split("/", 1)[0].lower() if c.isalnum())
    return name(pin) == name(returned)


def generation_identity(parsed: dict | None, headers: Mapping[str, str]) -> tuple[str | None, str | None]:
    body_id = parsed.get("id") if isinstance(parsed, dict) else None
    header_ids = {v for k, v in headers.items() if k.lower() == "x-generation-id" and isinstance(v, str) and v}
    if isinstance(body_id, str) and body_id:
        if header_ids and header_ids != {body_id}:
            return None, "conflicting_body_and_header_generation_ids"
        return body_id, "response_body"
    if len(header_ids) == 1:
        return next(iter(header_ids)), "x-generation-id_header"
    return None, "no_unambiguous_generation_id"


def rejection_usage(*, url: str, request: dict, http_status: int | None,
                    parsed: dict | None, usage: Usage, headers: Mapping[str, str] | None = None,
                    recovery: dict | None = None) -> tuple[Usage, dict]:
    evidence = {"rule": REJECTION_RULE, "classification": "unresolved_generation",
                "contracts": [ERROR_CONTRACT, LIMIT_CONTRACT]}
    if (usage.generated_tokens == 0 and usage.reasoning_tokens not in (None, 0)):
        return replace(usage, generated_tokens=None, measurement_basis="conflicting_provider_generation_counters"), {
            **evidence, "reason": "Zero completion conflicts with reported reasoning"}
    if usage.generated_tokens is not None:
        return usage, {**evidence, "classification": "provider_reported_generation"}
    pin = pinned_provider(request)
    if (generation_identity(parsed, headers or {})[1] == "conflicting_body_and_header_generation_ids"
            or (recovery is not None and recovery.get("http_status") == 200
                and recovery.get("status") != "recovered")):
        return usage, {**evidence, "reason": "Conflicting or incomplete generation evidence"}
    error = parsed.get("error") if isinstance(parsed, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if (url != OPENROUTER_CHAT_URL or request.get("stream", False) is not False
            or pin is None or http_status != 429 or not isinstance(error, dict)
            or error.get("code") != 429 or not isinstance(metadata, dict)):
        return usage, {**evidence, "reason": "No bound pinned pre-generation rate-limit envelope"}
    # Canonical error_type is documented. The legacy envelope retained from
    # OpenRouter's shared-pool rejection additionally names its admission source.
    canonical = metadata.get("error_type") == "rate_limit_exceeded"
    shared_pool = (metadata.get("provider_error_code") == "rate_limit_exceeded"
                   and metadata.get("limit_source") == "upstream_provider_shared_pool")
    # OpenRouter's retained shared-pool envelope can omit all typed codes.
    # Accept only its explicit, model-bound rejection sentence plus the exact
    # admission source and an identified provider (checked against the pin
    # below). Text alone, generic 429s and conflicting typed codes do not qualify.
    model, raw_error = request.get("model"), metadata.get("raw")
    rejection_sentence = f"{model} is temporarily rate-limited upstream."
    shared_pool_text = (
        metadata.get("limit_source") == "upstream_provider_shared_pool"
        and isinstance(metadata.get("provider_name"), str) and bool(metadata["provider_name"])
        and isinstance(model, str) and bool(model) and isinstance(raw_error, str)
        and (raw_error == rejection_sentence or raw_error.startswith(rejection_sentence + " "))
        and not any(key in metadata for key in ("error_type", "provider_error_code", "provider_code")))
    if not (canonical or shared_pool or shared_pool_text):
        return usage, {**evidence, "reason": "Rate-limit category is not explicit"}
    if (parsed.get("choices") or parsed.get("output") or parsed.get("content")
            or (parsed.get("model") is not None and parsed["model"] != request.get("model"))
            or any(metadata.get(k) is not None and not isinstance(metadata[k], str)
                   for k in ("provider_name",))):
        return usage, {**evidence, "reason": "Response contradicts a bound pre-generation rejection"}
    for provider in (metadata.get("provider_name"), parsed.get("provider")):
        if provider is not None and (not isinstance(provider, str) or not provider_identity_matches(pin, provider)):
            return usage, {**evidence, "reason": "Returned provider differs from the pin"}
    raw = parsed.get("usage")
    if raw is not None:
        if not isinstance(raw, dict):
            return usage, {**evidence, "reason": "Malformed usage cannot establish rejection accounting"}
        details = raw.get("completion_tokens_details")
        if details is not None and not isinstance(details, dict):
            return usage, {**evidence, "reason": "Malformed completion details"}
        for value in ([raw["completion_tokens"]] if "completion_tokens" in raw else []) + (
                [details["reasoning_tokens"]] if isinstance(details, dict) and "reasoning_tokens" in details else []):
            if type(value) is not int or value != 0:
                return usage, {**evidence, "reason": "Reported generation is nonzero or malformed"}
    if usage.output_tokens not in (None, 0) or usage.reasoning_tokens not in (None, 0):
        return usage, {**evidence, "reason": "Existing usage contradicts zero generation"}
    return replace(usage, output_tokens=0, generated_tokens=0, reasoning_tokens=0,
                   reasoning_accounting="included_in_output", measurement_basis=REJECTION_RULE), {
        **evidence, "classification": "documented_pre_generation_rejection",
        "http_status": http_status, "pinned_provider": pin,
        "category_source": ("error_type" if canonical else "shared_pool_provider_error_code" if shared_pool
                            else "shared_pool_model_bound_rejection_text"),
        "generation_evidence": "HTTP rejection before provider response headers under the OpenRouter contract",
        "monetary_evidence": "unchanged; no dollar amount inferred"}


def retry_plan(*, attempt: int, max_attempts: int, backoff_seconds: float,
               max_wait_seconds: float, headers: Mapping[str, str], parsed: dict | None,
               now: float, usage: Usage | None, retryable: bool) -> dict:
    result = {"rule": RETRY_RULE, "retry": False, "attempt": attempt,
              "max_attempts": max_attempts, "max_wait_seconds": max_wait_seconds}
    if not retryable:
        return {**result, "reason": "non_retryable_result"}
    if usage is None or usage.generated_tokens is None:
        return {**result, "reason": "unresolved_generation_Q8_stop"}
    if attempt >= max_attempts:
        return {**result, "reason": "attempt_limit"}
    hint_values = [("response_header", v) for k, v in headers.items() if k.lower() == "retry-after"]
    error = parsed.get("error") if isinstance(parsed, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if isinstance(metadata, dict):
        if "retry_after_seconds" in metadata:
            hint_values.append(("error_metadata_seconds", metadata["retry_after_seconds"]))
        nested = metadata.get("headers")
        if isinstance(nested, dict):
            hint_values += [("error_metadata_header", v) for k, v in nested.items() if k.lower() == "retry-after"]
    hints, invalid = [], []
    for source, value in hint_values:
        seconds = None
        if type(value) in (int, float):
            seconds = float(value)
        elif isinstance(value, str):
            try:
                seconds = float(value) if value.strip().isdigit() else None
                if seconds is None:
                    date = parsedate_to_datetime(value)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    seconds = max(0.0, date.timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                pass
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            invalid.append(source)
        else:
            hints.append({"source": source, "seconds": seconds})
    # Existing request timeout is also the declared upper bound on one wait.
    # A longer provider minimum stops retries; it is never shortened silently.
    try:
        exponential = math.ldexp(backoff_seconds, min(attempt - 1, 1024))
    except OverflowError:
        exponential = math.inf
    delay = max([exponential, *(h["seconds"] for h in hints)])
    result.update(retry_after_hints=hints, invalid_retry_after_sources=invalid,
                  delay_seconds=delay if math.isfinite(delay) else None)
    if not math.isfinite(delay) or delay > max_wait_seconds:
        return {**result, "reason": "retry_wait_exceeds_declared_timeout"}
    return {**result, "retry": True, "reason": "known_generation_within_attempt_and_wait_limits"}


UNRESOLVED_RETRY_RULE = "worst-case-charged-retry.v1"

# User 2026-11-02, after a DNS outage charged five never-sent retries $1.23 each: a request whose connection never
# opened (name resolution failed or the connection was refused) sent no byte to the provider, so it cannot have been
# generated or billed. It is a known $0 attempt and is retried like any known-usage failure; no worst-case charge.
# Timeouts after connecting and every HTTP-level failure stay under the charged rule.
NOT_SENT_RULE = "not-sent-zero-cost-retry.v1"
NOT_SENT_MARKERS = ("NameResolutionError", "NewConnectionError")


def never_sent(error: BaseException | None) -> str | None:
    """The urllib3 class name if the request's connection was never established, else None."""
    from urllib3.exceptions import MaxRetryError, NewConnectionError
    stack, seen = [error], set()
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, NewConnectionError):
            return type(e).__name__
        if isinstance(e, MaxRetryError):
            stack.append(e.reason)
        stack += [e.__cause__, e.__context__] + [a for a in getattr(e, "args", ()) if isinstance(a, BaseException)]
    return None


def recorded_never_sent(payload: Mapping[str, Any]) -> bool:
    """A recorded finished attempt that failed before any connection (retroactive form of never_sent)."""
    return (payload.get("http_status") is None and payload.get("status") == "error"
            and any(m in str(payload.get("error") or "") for m in NOT_SENT_MARKERS))


def unresolved_retry_rule(value: Any) -> dict | None:
    """The declared charged-retry rule (configs/v2/settings.json unresolved_retry), validated; None leaves Q8 stops."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("rule") != UNRESOLVED_RETRY_RULE:
        raise ValueError(f"unresolved_retry must declare rule {UNRESOLVED_RETRY_RULE}")
    prices = value.get("worst_case_usd_per_million_tokens")
    if not isinstance(prices, Mapping) or set(prices) != {"prompt", "completion"} or any(
            type(prices[k]) not in (int, float) or not math.isfinite(prices[k]) or prices[k] <= 0 for k in prices):
        raise ValueError("unresolved_retry needs positive worst-case prompt and completion prices per million tokens")
    return dict(value)


def worst_case_charge(rule: Mapping[str, Any], request: Mapping[str, Any], request_bytes: bytes) -> dict:
    """The most one unresolved attempt could have cost: every request byte as an uncached prompt token (a token
    encodes at least one byte of the serialized request) plus the full response cap as generated tokens, both at the
    declared worst-case prices. A bound used only by the run guards; the attempt's own usage stays unknown."""
    cap = request.get("max_tokens", request.get("max_completion_tokens"))
    if type(cap) is not int or cap <= 0:
        raise ValueError("A worst-case charge needs the attempt's positive response cap")
    prices = rule["worst_case_usd_per_million_tokens"]
    prompt_bound = len(request_bytes)
    usd = prompt_bound * prices["prompt"] / 1e6 + cap * prices["completion"] / 1e6
    return {"rule": UNRESOLVED_RETRY_RULE, "usd": usd, "prompt_token_bound": prompt_bound,
            "prompt_token_bound_basis": "serialized request bytes", "completion_token_bound": cap,
            "completion_token_bound_basis": "the attempt's response cap",
            "worst_case_usd_per_million_tokens": dict(prices),
            "basis": "upper bound for the run guards only; the attempt's usage and cost stay unknown in the ledger"}


def charged_retry_plan(decision: Mapping[str, Any], *, attempt: int, max_attempts: int,
                       backoff_seconds: float, max_wait_seconds: float) -> dict:
    """Retry an unresolved attempt (the Q8 stop of retry_plan) under the charged rule: same request, exponential
    backoff, within the attempt and wait limits. The caller records the worst-case charge on the attempt."""
    result = {**decision, "rule": UNRESOLVED_RETRY_RULE, "replaces": decision.get("rule"), "retry": False}
    if decision.get("reason") != "unresolved_generation_Q8_stop":
        return dict(decision)
    if attempt >= max_attempts:
        return {**result, "reason": "attempt_limit_unresolved"}
    try:
        delay = math.ldexp(backoff_seconds, min(attempt - 1, 1024))
    except OverflowError:
        delay = math.inf
    if not math.isfinite(delay) or delay > max_wait_seconds:
        return {**result, "reason": "retry_wait_exceeds_declared_timeout", "delay_seconds": None}
    return {**result, "retry": True, "delay_seconds": delay, "retry_after_hints": [],
            "reason": "unresolved_generation_charged_worst_case"}
