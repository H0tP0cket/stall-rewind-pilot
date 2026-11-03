"""Matched experiment scheduling specification.

Produces a campaign-schedule record the integration scheduler can execute:
branches grouped per S1 into matched blocks, repeats interleaved, with the
common resource horizons every branch must report at. Nothing here launches
work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..contracts import TreatmentManifest
from ..observers._canonical import content_id

SCHEDULE_REVISION = "schedule/dev-2026-09-15"


@dataclass(frozen=True)
class HorizonSpec:
    """Common generated-token horizons (fractions of the branch allowance) for curves."""

    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)

    def horizons(self, allowance: int) -> list[int]:
        return sorted({max(1, int(round(allowance * f))) for f in self.fractions})


def build_schedule(manifests: Sequence[TreatmentManifest], *, campaign_id: str, horizon: HorizonSpec = HorizonSpec(),
                   skipped: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    by_s1: dict[str, list[TreatmentManifest]] = {}
    for m in manifests:
        by_s1.setdefault(m.source_checkpoint_id, []).append(m)
    blocks = []
    for s1, ms in by_s1.items():
        allowances = {m.generated_token_limit for m in ms}
        if len(allowances) != 1:
            raise ValueError(f"matched block {s1} has unequal allowances {allowances}")
        allowance = allowances.pop()
        # interleave repeats: all arms of repeat 1, then all arms of repeat 2, ... so that
        # provider drift over time is spread across arms rather than confounded with one arm.
        repeats = sorted({m.repeat_id for m in ms})
        order = [m.treatment_id for rep in repeats for m in ms if m.repeat_id == rep]
        blocks.append({
            "source_checkpoint_id": s1, "allowance": allowance, "horizons": horizon.horizons(allowance),
            "arms": sorted({m.metadata.get("arm_id", "?") for m in ms}), "repeats": repeats,
            "execution_order": order, "isolation": "each branch in a fresh isolated environment",
            "worker_model": sorted({m.worker_model for m in ms}),
        })
    record = {
        "revision": SCHEDULE_REVISION, "campaign_id": campaign_id, "blocks": blocks,
        "treatment_ids": [m.treatment_id for m in manifests],
        "skipped": list(skipped),
        "reporting_requirements": [
            "graded checkpoints with fixed-target score and check vector at every official grade",
            "cumulative generated tokens, cost and elapsed time at each grade",
            "state at each common horizon (or explicit censoring when the branch stopped earlier)",
            "stop reason; technical failure as a separate status, never a zero score",
        ],
    }
    record["schedule_id"] = content_id(record)
    return record
