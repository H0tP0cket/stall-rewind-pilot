"""Which recorded runs may be the source of a manual recovery branch.

A natural run may. A natural resample may too (exp48, user 2026-09-28: resampled stalls are part of the experiment):
the runner resumed a natural run's checkpoint with no treatment, the source run's worker settings and a declared
``resample_origin`` whose checkpoint is the ``resume_from`` checkpoint. Each of its checkpoints captures the complete
H (the source prefix it resumed from included), W and R, so a branch restores from it exactly as from a natural
checkpoint. A treatment branch is never a source, and neither is a resample of anything other than a natural run.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def is_natural(start: Mapping[str, Any]) -> bool:
    return not start.get("resume_from") and not start.get("treatment")


def is_natural_resample(start: Mapping[str, Any], origin_start: Callable[[str], Mapping[str, Any] | None]) -> bool:
    """``start`` is the run's run_started payload; ``origin_start(run_id)`` returns another run's run_started payload."""
    if start.get("treatment") or not start.get("resume_from"):
        return False
    meta = (start.get("config") or {}).get("metadata") or {}
    origin = meta.get("resample_origin")
    if meta.get("execution_role") != "resample" or not isinstance(origin, Mapping):
        return False
    if origin.get("origin_checkpoint_id") != start["resume_from"]:
        return False
    source = origin.get("source_run_id")
    if not source or not str(start["resume_from"]).startswith(f"{source}/"):
        return False
    source_start = origin_start(source)
    return source_start is not None and is_natural(source_start)


def recovery_source_ok(start: Mapping[str, Any], origin_start: Callable[[str], Mapping[str, Any] | None]) -> bool:
    return is_natural(start) or is_natural_resample(start, origin_start)


def store_start_lookup(store: Any) -> Callable[[str], Mapping[str, Any] | None]:
    def lookup(run_id: str) -> Mapping[str, Any] | None:
        for event in store.iter_events(run_id):
            if event.kind == "lifecycle" and event.payload.get("transition") == "run_started":
                return event.payload
        return None
    return lookup
