"""Points of a run: point k is the state after step k and before the model call for step k+1 (point 0 = start).

Step k is the k-th model call of the run's history (a format-error response is a step), together with its delivered
tool outputs and any submission feedback delivered right after it. A checkpoint records ``step_index = k`` in its
validation (natural-v2 runs checkpoint every point). Several occurrences may share one k, for example a submission
checkpoint that ends in the exit marker and the post-feedback checkpoint; the checkpoint for point k is the last
branchable one with that k.
"""
from __future__ import annotations

from typing import Any


def checkpoint_step_index(store: Any, checkpoint: Any) -> int:
    """Recorded step_index, or the model-call count of the checkpoint's own history for earlier runs."""
    k = (checkpoint.validation or {}).get("step_index")
    if type(k) is int:
        return k
    return int(store.get_json(checkpoint.state.history)["counters"].get("n_calls", 0))


def point_checkpoints(store: Any, run_id: str) -> dict[int, str]:
    """{k: checkpoint_id} of the branchable checkpoint for every recorded point of one run."""
    points: dict[int, str] = {}
    for checkpoint in sorted(store.list_checkpoints(run_id), key=lambda c: c.ordinal):
        if checkpoint.branchable:
            points[checkpoint_step_index(store, checkpoint)] = checkpoint.checkpoint_id
    return dict(sorted(points.items()))
