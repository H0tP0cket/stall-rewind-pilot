"""Implemented-decision windows on steps (the rule of ``observers/t2/units22._implemented_decision_windows``,
re-keyed from sentences to steps).

* Commitment: a step that changed the workspace or runtime, or submitted.
* Anchor: a step with a decision tag followed by a commitment at that step or later, before the next step with a
  decision tag.
* Window 1 starts at step 1. Window m > 1 starts at the earliest of its anchor step and the steps the anchor's decisions
  explicitly refer back to (one hop, no transitive backfill), but never before the step after the previous window's
  last commitment. A window ends where the next one starts; the last window runs to the stall.
* With no anchor there is one window (``no_anchor``); with no steps there is none.

Difference from the sentence version: windows stay contiguous runs of steps. The sentence-level override (a sentence
before the last anchor that refers back to an earlier anchor moved to that window) is recorded as ``revisits`` but
moves nothing, so every approach is a contiguous step range and a rewind point is well defined.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

WINDOWS_VERSION = "ledger-windows/v1"


def implemented_decision_windows(n_steps: int, decision_steps: Iterable[int], commitments: Iterable[int],
                                 refers: Mapping[int, Iterable[int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions = sorted(set(decision_steps))
    commits = sorted(set(commitments))
    anchors = []
    for m, d in enumerate(decisions):
        nxt = decisions[m + 1] if m + 1 < len(decisions) else n_steps + 1
        if any(d <= c < nxt for c in commits):
            anchors.append(d)
    meta: dict[str, Any] = {"version": WINDOWS_VERSION, "n_steps": n_steps, "decision_steps": decisions,
                            "commitment_steps": commits, "anchors": anchors, "no_anchor": not anchors}
    if n_steps <= 0:
        return [], meta
    if not anchors:
        return [{"id": "W1", "first_step": 1, "last_step": n_steps, "anchor_step": None, "backfill": 0}], meta
    starts, backfill = [1], [0]
    for m in range(1, len(anchors)):
        a = anchors[m]
        lb = max(c for c in commits if anchors[m - 1] <= c < a) + 1
        hop = [t for t in refers.get(a, ()) if lb <= t < a]
        starts.append(min([a] + hop))
        backfill.append(a - min(hop) if hop else 0)
    bounds = starts + [n_steps + 1]
    windows = [{"id": f"W{m + 1}", "first_step": bounds[m], "last_step": bounds[m + 1] - 1, "anchor_step": anchors[m],
                "backfill": backfill[m]} for m in range(len(anchors))]
    win_of = {k: m for m, w in enumerate(windows) for k in range(w["first_step"], w["last_step"] + 1)}
    anchor_of = {w["anchor_step"]: m for m, w in enumerate(windows)}
    meta["revisits"] = [{"step": s, "window": windows[win_of[s]]["id"], "refers_to_anchor_of": windows[anchor_of[t]]["id"]}
                        for s, ts in sorted(refers.items()) if s < anchors[-1] and s not in anchor_of and s in win_of
                        for t in ts if t in anchor_of and anchor_of[t] < win_of[s]]
    return windows, meta
