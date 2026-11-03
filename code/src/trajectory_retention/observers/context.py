"""Explicit local limits on complete observer requests, without truncation.

Byte counts describe the retained canonical request projection, not a model's
token window. Token counts use a caller-supplied, versioned local measurement;
neither measure asserts provider-exact context fit or scientific calibration.
This module performs no provider calls and creates no usage/cost receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contracts import ArtifactStoreProtocol, canonical_json, to_jsonable
from ..persistence import persist
from ._canonical import content_id
from .client import ModelRequest
from .tokens import TokenCounter

CONTEXT_REVISION = "observer-request-context-preflight/v1"
REQUEST_SCOPE = "complete_model_messages_settings; canonical_json_with_stream_false"


@dataclass(frozen=True)
class ContextPolicy:
    max_request_bytes: int | None = None
    max_input_tokens: int | None = None
    token_counter_id: str | None = None
    token_counter_revision: str | None = None
    allow_proxy: bool = False

    def __post_init__(self) -> None:
        if self.max_request_bytes is None and self.max_input_tokens is None:
            raise ValueError("context policy requires at least one explicit limit")
        for value in (self.max_request_bytes, self.max_input_tokens):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("context limits must be positive integers")
        if self.max_input_tokens is not None and not (self.token_counter_id and self.token_counter_revision):
            raise ValueError("input-token limits require an explicit counter identity and revision")
        if not isinstance(self.allow_proxy, bool):
            raise ValueError("allow_proxy must be an explicit boolean")

    def as_record(self) -> dict[str, Any]:
        return {"revision": CONTEXT_REVISION, "max_request_bytes": self.max_request_bytes,
                "max_input_tokens": self.max_input_tokens, "token_counter_id": self.token_counter_id,
                "token_counter_revision": self.token_counter_revision, "allow_proxy": self.allow_proxy,
                "over_limit_action": "incomplete_without_client_call", "truncation": "none",
                "calibration_status": "explicit_operational_limits_not_scientifically_calibrated"}


@dataclass(frozen=True)
class RequestTokenCounter:
    """A local counter for the complete request, distinct from packet tokens.

    The revision must bind implementation, tokenizer/settings and chat-wrapper
    assumptions. ``count`` must be local and may return None when unavailable.
    """
    model: str
    counter_id: str
    revision: str
    fidelity: str
    count: Callable[[ModelRequest], int | None] = field(repr=False, compare=False)
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.model, self.counter_id, self.revision)):
            raise ValueError("request counter requires model, identity and revision")
        if self.fidelity not in {"local_tokenizer", "approximate_proxy"}:
            raise ValueError("request counters cannot assert provider-exact fidelity")
        if not callable(self.count):
            raise ValueError("request counter must supply a local callable")

    def as_record(self) -> dict[str, Any]:
        return {"model": self.model, "counter_id": self.counter_id, "revision": self.revision,
                "fidelity": self.fidelity, "settings": dict(self.settings), "scope": REQUEST_SCOPE,
                "provider_exact": False}


def _request_body(request: ModelRequest) -> bytes:
    if set(request.settings) & {"model", "messages", "stream", "tools"}:
        raise ValueError("observer settings must not override model/messages/stream/tools")
    return canonical_json({"model": request.model, "messages": request.messages,
                           **request.settings, "stream": False})


def request_counter_from_packet_counter(counter: TokenCounter) -> RequestTokenCounter:
    """Reuse a selected local encoding on the complete canonical JSON text.

    This is a new measurement scope and identity; it is not the packet-text
    count or provider chat tokenization. No encoding or numerical cap is chosen.
    """
    binding = {"revision": "packet-counter-on-complete-request-projection/v1",
               "request_scope": REQUEST_SCOPE, "counter_implementation": counter.as_record(),
               "provider_chat_framing": "unverified_canonical_json_text_only"}
    return RequestTokenCounter(model=counter.model, counter_id="request-projection:" + counter.tokenizer_id,
                               revision=content_id(binding), fidelity=counter.fidelity,
                               count=lambda request: counter.count(_request_body(request).decode("utf-8")),
                               settings=binding)


def context_preflight(
    store: ArtifactStoreProtocol, request: ModelRequest, policy: ContextPolicy, *,
    token_counter: RequestTokenCounter | None = None,
) -> dict[str, Any]:
    """Persist complete request and decision before a prospective client call.

    Unknown required token measurements fail closed. An unchanged counter
    revision producing a different decision conflicts with the retained record
    before a client call, rather than silently replacing prior evidence.
    """
    body = _request_body(request)
    request_ref = persist(lambda: store.put_bytes(body, media_type="application/json"), evidence=body)
    counter_record = None if token_counter is None else token_counter.as_record()
    binding = {"revision": CONTEXT_REVISION, "request_hash": request.request_hash(),
               "model": request.model, "cost_role": request.cost_role, "purpose": request.purpose,
               "run_id": request.run_id, "target_checkpoint_id": request.target_checkpoint_id,
               "request_ref": to_jsonable(request_ref), "request_scope": REQUEST_SCOPE,
               "policy": policy.as_record(), "counter": counter_record}
    input_tokens = {"value": None, "status": "not_required", "measurement": counter_record}
    status, reason = "within_configured_limits", None
    if policy.max_request_bytes is not None and len(body) > policy.max_request_bytes:
        status, reason = "request_bytes_exceeded", "complete request exceeds the explicit byte limit"
        input_tokens["status"] = "not_measured_byte_limit_exceeded"
    elif policy.max_input_tokens is not None:
        unknown = None
        if token_counter is None:
            unknown = "configured request token counter is unavailable"
        elif (token_counter.model != request.model or token_counter.counter_id != policy.token_counter_id
              or token_counter.revision != policy.token_counter_revision):
            unknown = "request token counter does not bind the configured model/identity/revision"
        elif token_counter.fidelity == "approximate_proxy" and not policy.allow_proxy:
            unknown = "approximate proxy is not explicitly accepted by this context policy"
        else:
            try:
                value = token_counter.count(request)
            except Exception as error:
                unknown = f"local request token measurement failed: {type(error).__name__}"
            else:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    unknown = "local request token measurement is unavailable or invalid"
                else:
                    input_tokens.update(value=value, status="measured")
        if unknown is not None:
            status, reason = "input_token_measurement_unknown", unknown
            input_tokens["status"] = "unknown"
        elif input_tokens["value"] > policy.max_input_tokens:
            status, reason = "input_tokens_exceeded", "complete request exceeds the explicit measured input-token limit"
    record = {**binding, "preflight_id": content_id(binding),
              "request_bytes": {"value": len(body), "fidelity": "exact_canonical_request_bytes"},
              "input_tokens": input_tokens, "status": status, "reason": reason,
              "client_call_allowed": status == "within_configured_limits", "provider_context_fit": None}
    persist(lambda: store.put_record("observer_context_preflight_v1", record["preflight_id"], record,
                                     run_id=request.run_id), evidence=record)
    return record
