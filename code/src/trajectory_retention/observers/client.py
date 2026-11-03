"""Observer model-call boundary.

Observers never call a provider directly. They call an ObserverClient, which
returns the raw response text plus a CallReceipt (usage, model/route identity,
status). The live client is supplied by the integration/accounting session
and must record exact request/response artifacts under the declared cost
role. Two development clients are provided here:

* ScriptedClient — deterministic, keyed by request hash; used for fixtures and
  tests so the pipeline runs end to end without paid calls;
* NullClient — always fails; verifies failure/missingness paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..contracts import Usage, to_jsonable
from ._canonical import content_id

COST_ROLES = ("extraction", "audit", "summary", "advisor", "render")


@dataclass(frozen=True)
class CallReceipt:
    call_record_id: str
    cost_role: str
    requested_model: str
    resolved_model: str | None
    route: str | None
    status: str  # ok / error / timeout
    usage: Usage | None
    request_hash: str
    response_hash: str | None
    error: str | None = None
    fingerprint: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "call_record_id": self.call_record_id,
            "cost_role": self.cost_role,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "route": self.route,
            "status": self.status,
            "usage": None if self.usage is None else to_jsonable(self.usage),
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "error": self.error,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[Mapping[str, str], ...]
    settings: Mapping[str, Any]
    cost_role: str
    purpose: str  # e.g. "t_update", "t_audit", "summary"
    run_id: str
    target_checkpoint_id: str

    def __post_init__(self) -> None:
        if self.cost_role not in COST_ROLES:
            raise ValueError(f"unknown cost role {self.cost_role!r}")

    def request_hash(self) -> str:
        return content_id({
            "model": self.model,
            "messages": [dict(m) for m in self.messages],
            "settings": dict(self.settings),
        })


class ObserverClient(Protocol):
    def complete(self, request: ModelRequest) -> tuple[str | None, CallReceipt]: ...


@dataclass
class ScriptedClient:
    """Deterministic client: responses keyed by request hash or supplied by a function."""

    responses: dict[str, str] = field(default_factory=dict)
    fallback: Callable[[ModelRequest], str | None] | None = None
    model_name: str = "scripted/dev"
    calls: list[tuple[ModelRequest, CallReceipt]] = field(default_factory=list)
    counter: int = 0

    def complete(self, request: ModelRequest) -> tuple[str | None, CallReceipt]:
        self.counter += 1
        rh = request.request_hash()
        text = self.responses.get(rh)
        if text is None and self.fallback is not None:
            text = self.fallback(request)
        status = "ok" if text is not None else "error"
        receipt = CallReceipt(
            call_record_id=f"call:{request.run_id}:{request.cost_role}:{self.counter}",
            cost_role=request.cost_role,
            requested_model=request.model,
            resolved_model=self.model_name,
            route="scripted",
            status=status,
            usage=Usage(measurement_basis="scripted_no_usage"),
            request_hash=rh,
            response_hash=content_id(text) if text is not None else None,
            error=None if text is not None else "no scripted response",
        )
        self.calls.append((request, receipt))
        return text, receipt


@dataclass
class NullClient:
    calls: int = 0

    def complete(self, request: ModelRequest) -> tuple[str | None, CallReceipt]:
        self.calls += 1
        return None, CallReceipt(
            call_record_id=f"call:{request.run_id}:{request.cost_role}:null{self.calls}",
            cost_role=request.cost_role,
            requested_model=request.model,
            resolved_model=None,
            route=None,
            status="error",
            usage=None,
            request_hash=request.request_hash(),
            response_hash=None,
            error="null client",
        )


def receipts_record(receipts: Sequence[CallReceipt]) -> list[dict[str, Any]]:
    return [r.as_record() for r in receipts]
