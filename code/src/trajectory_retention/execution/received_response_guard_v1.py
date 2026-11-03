"""Opt-in preservation of received worker responses during local failures.

This is an unactivated model subclass, not a mutation of the sealed provider
instrument. Selecting it requires an explicit new instrument/config binding.
It does not retry HTTP or turn a failed local processor into a usable reply.
"""
from __future__ import annotations

from typing import Any

from ..persistence import preserve_failure
from .model import OpenRouterChatModel


RULE = "received-response-guard/v1"


class ReceivedResponseGuardV1(OpenRouterChatModel):
    """Retain the original response if processing fails before final journaling."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config["received_response_guard"] = RULE
        self._guard_received: dict[str, Any] | None = None
        self._guard_finished_entered = False

    def _emit(self, phase: str, record: dict[str, Any]) -> None:
        if phase == "started":
            self._guard_received = None
            self._guard_finished_entered = False
        elif phase == "finished":
            # The recording agent owns bounded persistence retries. Once this
            # starts, a failure must never cause a second final receipt attempt.
            self._guard_finished_entered = True
        super()._emit(phase, record)

    def _receive(self, record: dict[str, Any], response: Any) -> dict | None:
        self._guard_received = record
        # Capture wire evidence before delegating parsing/accounting. The base
        # method writes the same fields; this also protects an early parse error.
        record.update(http_status=response.status_code, response_bytes=response.content,
                      response_headers=dict(response.headers))
        return super()._receive(record, response)

    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        self._guard_received = None
        self._guard_finished_entered = False
        try:
            return super().query(messages, **kwargs)
        except BaseException as error:
            record = self._guard_received
            if record is not None and not self._guard_finished_entered:
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                record.update(status="interrupted" if interrupted else "error",
                    error=f"{RULE}: local_response_processing_"
                          f"{'interrupted' if interrupted else 'failed'}: {type(error).__name__}: {error}")
                try:
                    self._emit("finished", record)
                except BaseException as persistence_error:
                    preserve_failure(error, record)
                    error.add_note("Received response persistence also failed: "
                                   f"{type(persistence_error).__name__}")
            raise
