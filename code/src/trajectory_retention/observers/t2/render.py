"""Worker-visible T2 packets and deterministic removal views.

The research record keeps IDs, spans, probabilities and links. The packet shown
to a recovering worker contains only verbatim content grouped by role, plus
status notes derived from the worker's own stated changes. No identifiers,
hashes or provenance boilerplate are rendered. Every view is a deterministic
filter of one frozen checkpoint T; nothing is regenerated.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .derive import CheckpointT, kept
from .schema import OBSERVATION_SOURCES, WORKER_SOURCES, Item

RENDER_VERSION = "t2-render/v4"
# Development setting: worker packet token budget for the selected full-T set.
# Views remove items from that fixed selection and never refill freed space.
PACKET_TOKEN_BUDGET = 3000
NEAR_DUPLICATE_JACCARD = 0.7
HEADER = ("Notes recorded from earlier work on this task. They are verbatim excerpts of what was observed, "
          "said or tried. They may describe code or files that have since changed.")
SECTIONS = (
    ("approach", "Explanations and approaches considered"),
    ("question", "Open questions"),
    ("takeaway", "Conclusions drawn"),
    ("intended_action", "Planned next steps"),
    ("working_note", "Working notes"),
    ("unknown", "Other notes"),
)
STATUS_NOTE = {"withdrawn": " (later dropped by the agent)", "replaced": " (later replaced by a different approach)",
               "confirmed": " (later confirmed by the agent)"}
ACTION_CHARS = 160


@dataclass(frozen=True)
class Packet:
    text: str
    item_ids: tuple[str, ...]
    view: str


def _one_line(text: str, limit: int) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


_WORD = re.compile(r"[a-z0-9_]+")


def _shingles(text: str) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < 5:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def near_duplicates(items: Sequence[Item]) -> set[str]:
    """IDs of kept prose items repeated later in near-identical words (the latest is kept)."""
    prose = [i for i in items if kept(i) and i.source_kind in WORKER_SOURCES]
    prose.sort(key=lambda i: (i.available_seq, i.seq, i.order))
    shingles = [_shingles(i.text) for i in prose]
    index: dict[str, list[int]] = {}
    dropped: set[str] = set()
    for pos in range(len(prose) - 1, -1, -1):
        mine = shingles[pos]
        if not mine:
            continue
        counts: dict[int, int] = {}
        for sh in mine:
            for later in index.get(sh, ()):
                counts[later] = counts.get(later, 0) + 1
        if any(c / len(mine | shingles[later]) >= NEAR_DUPLICATE_JACCARD for later, c in counts.items()):
            dropped.add(prose[pos].item_id)
            continue
        for sh in mine:
            index.setdefault(sh, []).append(pos)
    return dropped


def _keep_probability(item: Item) -> float:
    label = item.labels.get("keep")
    if label is None or not label.probabilities:
        return 0.0
    return float(label.probabilities.get("yes", 0.0))


KEEP_BANDS = (0.9, 0.75, 0.6)


def _band(p: float) -> int:
    return sum(p >= b for b in KEEP_BANDS)


def select_for_packet(view: CheckpointT, budget: int | None, count_tokens: Callable[[str], int]) -> set[str]:
    """Deterministic selection of the full-T packet: kept, de-duplicated, then budgeted by calibrated
    keep probability band, then recency. ``budget=None`` keeps everything."""
    dup = near_duplicates(view.items)
    pool = [i for i in view.items if kept(i) and i.item_id not in dup and i.source_kind != "worker_action"]
    if budget is None:
        return {i.item_id for i in pool}
    # Calibrated keep probability in coarse bands, then recency: most items sit near the top band,
    # where exact probabilities differ only by classifier jitter.
    pool.sort(key=lambda i: (-_band(_keep_probability(i)), -i.available_seq, -i.seq, i.order))
    chosen, used = set(), 0
    for item in pool:
        cost = count_tokens(item.text) + 4
        if used + cost > budget:
            continue
        chosen.add(item.item_id)
        used += cost
    return chosen


def render(view: CheckpointT, include: Callable[[Item], bool] | None = None, *, name: str = "full",
           selected: set[str] | None = None) -> Packet:
    by_id = view.by_id
    chosen = [i for i in view.items if kept(i) and (selected is None or i.item_id in selected)
              and (include is None or include(i))]
    chosen_ids = {i.item_id for i in chosen}
    result_of = {l.source_id: l.target_id for l in view.links if l.relation == "result_of"}

    lines: list[str] = [HEADER, ""]
    used: list[str] = []

    # Evidence: prose evidence plus observed results grouped under the action that produced them.
    evidence_lines: list[str] = []
    seen_text: set[str] = set()
    grouped: dict[str, list[Item]] = {}
    order: list[tuple[int, int, str]] = []
    for item in chosen:
        if item.source_kind in OBSERVATION_SOURCES:
            action_id = result_of.get(item.item_id, f"none:{item.seq}")
            if action_id not in grouped:
                grouped[action_id] = []
                order.append((item.seq, item.order, action_id))
            grouped[action_id].append(item)
        elif item.label("role") == "evidence":
            order.append((item.seq, item.order, item.item_id))
    for _, _, key in sorted(order):
        if key in grouped:
            obs = [o for o in grouped[key] if o.text not in seen_text]
            if not obs:
                continue
            action = by_id.get(key)
            head = f"- After `{_one_line(action.text, ACTION_CHARS)}`:" if action is not None else "- Observed:"
            evidence_lines.append(head)
            for o in obs:
                seen_text.add(o.text)
                evidence_lines.append("    " + _one_line(o.text, 400))
                used.append(o.item_id)
            if action is not None:
                used.append(action.item_id)
        else:
            item = by_id[key]
            if item.text in seen_text:
                continue
            seen_text.add(item.text)
            evidence_lines.append(f"- {item.text}")
            used.append(item.item_id)
    observed = [ln for ln in evidence_lines if not ln.startswith("- ") or ln.startswith("- After `") or ln.startswith("- Observed:")]
    stated = [ln for ln in evidence_lines if ln.startswith("- ") and not (ln.startswith("- After `") or ln.startswith("- Observed:"))]
    if observed:
        lines += ["Observed results:"] + observed + [""]
    if stated:
        lines += ["Facts stated by the agent:"] + stated + [""]

    for role, title in SECTIONS:
        section = []
        for item in chosen:
            if item.source_kind not in WORKER_SOURCES or item.label("role") != role or item.text in seen_text:
                continue
            seen_text.add(item.text)
            note = STATUS_NOTE.get(view.status.get(item.item_id, ("unknown", None))[0], "")
            section.append(f"- {item.text}{note}")
            used.append(item.item_id)
        if section:
            lines += [f"{title}:"] + section + [""]
    return Packet("\n".join(lines).rstrip() + "\n", tuple(used), name)


# ---- removal views (deterministic filters of the same frozen checkpoint T) ----

def drop_roles(roles: Iterable[str]) -> Callable[[Item], bool]:
    roles = set(roles)
    return lambda i: i.source_kind in OBSERVATION_SOURCES or i.label("role") not in roles


def drop_status(statuses: Iterable[str], view: CheckpointT) -> Callable[[Item], bool]:
    statuses = set(statuses)
    return lambda i: view.status.get(i.item_id, ("unknown", None))[0] not in statuses


def drop_pursuit(view: CheckpointT, root_id: str) -> Callable[[Item], bool]:
    """Withhold one pursuit's explanations/plans/questions; keep all evidence and observations."""
    members = set(view.pursuits.get(root_id, ()))
    return lambda i: not (i.item_id in members and i.source_kind in WORKER_SOURCES and i.label("role") != "evidence")


def random_matched(view: CheckpointT, removed_ids: set[str], seed: int,
                   selected: set[str] | None = None) -> Callable[[Item], bool]:
    """Remove the same number of randomly chosen selected prose items as a targeted view removed."""
    pool = sorted(i.item_id for i in view.items if kept(i) and i.source_kind in WORKER_SOURCES
                  and (selected is None or i.item_id in selected)
                  and i.label("role") != "evidence" and i.item_id not in removed_ids)
    rng = random.Random(seed)
    drop = set(rng.sample(pool, min(len(removed_ids), len(pool))))
    return lambda i: i.item_id not in drop


def removed_by(view: CheckpointT, include: Callable[[Item], bool], selected: set[str] | None = None) -> set[str]:
    return {i.item_id for i in view.items if kept(i) and (selected is None or i.item_id in selected) and not include(i)}
