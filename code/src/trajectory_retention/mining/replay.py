"""Offline detector replay from recorded inputs; must reproduce live outputs exactly."""

from __future__ import annotations

from typing import Any

from ..contracts import ArtifactStoreProtocol
from ..detector.core import DetectorConfig, records_digest, replay


def recorded_detector_io(store: ArtifactStoreProtocol, run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs, outputs = [], []
    for ev in store.iter_events(run_id):
        if ev.kind != "detector":
            continue
        if ev.payload.get("role") == "input":
            inputs.append(dict(ev.payload["input"]))
        elif ev.payload.get("role") == "output":
            outputs.append(dict(ev.payload["output"]))
    return inputs, outputs


def replay_run(store: ArtifactStoreProtocol, run_id: str, config: DetectorConfig) -> dict[str, Any]:
    inputs, live = recorded_detector_io(store, run_id)
    offline = replay(inputs, config, run_id=run_id)
    return {"inputs": len(inputs), "live_outputs": len(live), "offline_outputs": len(offline),
            "live_digest": records_digest(live), "offline_digest": records_digest(offline),
            "agrees": live == offline}
