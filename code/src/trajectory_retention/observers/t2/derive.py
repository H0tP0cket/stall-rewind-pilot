"""Deterministic derivations over labeled T2 items: checkpoint views, status, pursuits.

Nothing here calls a model. A checkpoint's T is exactly the items and links
available at its event cut in its native session (primary scope). Pursuit
status is derived only from explicit worker-stated change links; everything
else is ``unknown`` or, when a later item builds on it, ``continued`` with the
observer-inferred tier recorded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schema import OBSERVATION_SOURCES, WORKER_SOURCES, Item, Link
from .sources import CheckpointInfo, RunSource

PURSUIT_ROLES = {"approach", "intended_action", "question", "takeaway"}


def kept(item: Item) -> bool:
    """Selected for T: classifier keep=yes, not a task copy, not narration.

    Items the classifier judges to only restate the task specification are
    recorded but not selected, like exact task copies.
    A kept item whose role stayed below the confidence gate remains in T under
    role ``unknown``; role-based removal views never touch it.
    """
    return (item.label("keep") == "yes" and not item.task_copy and item.label("role") != "narration"
            and item.label("spec_statement") != "yes")


@dataclass
class CheckpointT:
    checkpoint: CheckpointInfo
    items: list[Item]
    links: list[Link]
    status: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # item_id -> (status, tier)
    pursuits: dict[str, list[str]] = field(default_factory=dict)  # root approach id -> member ids

    @property
    def by_id(self) -> dict[str, Item]:
        return {i.item_id: i for i in self.items}


def at_checkpoint(source: RunSource, checkpoint: CheckpointInfo, links: Sequence[Link], *,
                  session_scoped: bool = True) -> CheckpointT:
    cut = checkpoint.event_sequence
    items = [i for i in source.items if i.available_seq <= cut
             and (not session_scoped or i.session_id == checkpoint.session_id)]
    ids = {i.item_id for i in items}
    live = [l for l in links if l.evidence_seq <= cut and l.source_id in ids and l.target_id in ids]
    view = CheckpointT(checkpoint, items, live)
    view.status = derive_status(items, live)
    view.pursuits = derive_pursuits(items, live)
    return view


def derive_status(items: Sequence[Item], links: Sequence[Link]) -> dict[str, tuple[str, str | None]]:
    status: dict[str, tuple[str, str | None]] = {}
    targets = {i.item_id for i in items if i.label("role") in PURSUIT_ROLES}
    for link in sorted(links, key=lambda l: l.evidence_seq):
        if link.target_id not in targets:
            continue
        if link.tier == "worker_stated" and link.relation in {"abandons", "replaces", "confirms"}:
            status[link.target_id] = ({"abandons": "withdrawn", "replaces": "replaced", "confirms": "confirmed"}[link.relation],
                                      "worker_stated")
        elif link.relation == "refers_to" and link.target_id not in status:
            status[link.target_id] = ("continued", "observer_inferred")
    for item_id in targets:
        status.setdefault(item_id, ("unknown", None))
    return status


def derive_pursuits(items: Sequence[Item], links: Sequence[Link]) -> dict[str, list[str]]:
    """Group kept items under the earliest approach reached by following refers_to links.

    Membership uses observer-inferred links, so pursuits are a selection device for
    interventions and analysis, never content shown to a worker as its own reasoning.
    """
    by_id = {i.item_id: i for i in items}
    parent: dict[str, str] = {}
    for link in links:
        if link.relation == "refers_to" and link.source_id in by_id and link.target_id in by_id:
            parent.setdefault(link.source_id, link.target_id)

    def root(item_id: str) -> str | None:
        seen, found, cur = set(), None, item_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            item = by_id.get(cur)
            if item is not None and item.label("role") == "approach" and kept(item):
                found = cur
            cur = parent.get(cur)
        return found

    pursuits: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item.source_kind in WORKER_SOURCES and kept(item):
            r = root(item.item_id)
            if r is not None:
                pursuits[r].append(item.item_id)
    # Structural attachments: same-turn actions of members and kept results of those actions.
    same_turn: dict[str, list[str]] = defaultdict(list)
    results: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if link.relation == "same_turn":
            same_turn[link.source_id].append(link.target_id)
        elif link.relation == "result_of":
            results[link.target_id].append(link.source_id)
    for r, members in pursuits.items():
        extra: list[str] = []
        for m in members:
            for action in same_turn.get(m, ()):
                extra.append(action)
                extra.extend(o for o in results.get(action, ()) if kept(by_id[o]))
        pursuits[r] = list(dict.fromkeys(members + extra))
    return {k: v for k, v in pursuits.items() if len(v) >= 2}


def pursuit_summary(view: CheckpointT, selected: set[str] | None = None) -> list[dict[str, Any]]:
    """Pursuits ordered by size; with ``selected``, only members present in that packet selection count."""
    by_id = view.by_id
    out = []
    for root_id, all_members in view.pursuits.items():
        members = [m for m in all_members if selected is None or m in selected or by_id[m].source_kind == "worker_action"]
        if sum(by_id[m].source_kind in WORKER_SOURCES for m in members) < 1:
            continue
        seqs = [by_id[m].seq for m in members]
        roles = defaultdict(int)
        for m in members:
            roles[by_id[m].label("role") if by_id[m].source_kind in WORKER_SOURCES else by_id[m].source_kind] += 1
        out.append({"root": root_id, "root_text": by_id[root_id].text, "members": len(members),
                    "prose_members_selected": sum(by_id[m].source_kind in WORKER_SOURCES for m in members),
                    "first_seq": min(seqs), "last_seq": max(seqs), "roles": dict(roles),
                    "status": view.status.get(root_id, ("unknown", None))[0]})
    return sorted(out, key=lambda p: (-p["prose_members_selected"], -p["members"]))
