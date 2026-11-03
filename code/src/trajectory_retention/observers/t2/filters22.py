"""T2.2 removal views: role filters over one frozen packet, and matched random controls.

A filter removes worker-prose items whose main role is in its role set (``None``
means any role) and, when given, for which ``unit_predicate(item, approach)``
holds, where ``approach`` is the packed approach the item belongs to or
``None``. The scope is the current approach's prose in the packet
(``current_approach``, the default for per-approach questions) or all packet
prose (``global``). Tool outputs are evidence and are never removed.

Removals must be real. Every view also removes:
  * reference dependencies: packet prose with an explicit ``restates`` link to a
    removed item, followed transitively, whatever its scope;
  * paraphrases: when ``paraphrase_fn`` is given, kept prose it judges to
    restate removed items (value ``restates_removed``).
and records the removed ids by cause, their displayed tokens and share of the
packet, the rendered sizes before and after, a borderline report, the commands
issued with the removed statements (input for the workspace-leak check), and
empty slots for the workspace-leak and recovery-probe results. Views never
refill: kept ids are always a subset of the frozen packet.

Borderline: a sentence is borderline for a filter when its top two role
probabilities straddle the filter's role set with top1 < ROLE_THRESHOLD and
top1 + top2 >= BORDERLINE_TOP2_SUM, or when its main role is unknown and a
filter role is in its top two. It is reported for kept and removed items.

The candidate families below are definitions for validation only; which arms
are run is not ratified.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Collection, Sequence

from .packet22 import Count, Packet22, item_tokens, render22
from .schema import UNKNOWN
from .units21 import PROSE, View21
from .units22 import Approach
from .vocab22 import (APPROACH_MEMBER_ROLES, EVIDENCE_ROLES, FILTER_SCOPES, LINK_TIER_EXPLICIT, RANDOM_TOLERANCE,
                      ROLE_GROUPS, ROLE_OPTIONS, ROLE_THRESHOLD)

FILTERS22_VERSION = "t2.2-filters/v1"
BORDERLINE_TOP2_SUM = 0.8
RANDOM_SEEDS = (1, 2, 3)
DEAD_END_STATUSES = frozenset({"withdrawn", "replaced", "refuted"})
PARAPHRASE_VALUE = "restates_removed"
Predicate = Callable[[Any, "Approach | None"], bool]
ParaphraseFn = Callable[[list, list], dict]


@dataclass(frozen=True)
class FilterSpec:
    name: str
    roles: frozenset[str] | None
    scope: str = "current_approach"
    unit_predicate: Predicate | None = None

    def __post_init__(self):
        if self.scope not in FILTER_SCOPES:
            raise ValueError(f"scope must be one of {FILTER_SCOPES}, got {self.scope!r}")
        if self.roles is not None:
            object.__setattr__(self, "roles", frozenset(self.roles))

    def as_record(self) -> dict[str, Any]:
        return {"name": self.name, "roles": None if self.roles is None else sorted(self.roles), "scope": self.scope,
                "unit_predicate": getattr(self.unit_predicate, "__name__", None) if self.unit_predicate else None}


@dataclass
class FilterView:
    name: str
    spec: dict[str, Any]
    status: str  # applied | nothing_to_remove | no_current_approach_in_packet
    current: str | None
    item_ids: list[str]
    removed_primary: list[str]
    removed_dependencies: list[str]
    removed_paraphrase: list[str]
    removed_tokens: int
    packet_tokens: int
    removed_share: float
    rendered_tokens_full: int
    rendered_tokens_kept: int
    paraphrase: dict[str, Any]
    borderline: dict[str, Any]
    removed_actions: list[dict]
    workspace_leak: dict | None = None   # filled by the workspace-leak check
    recovery_probe: dict | None = None   # filled by the recovery probe
    rollback_checkpoint: dict | None = None  # W x T ablation: workspace rollback point (filled by the pipeline)
    version: str = FILTERS22_VERSION
    random_controls: list[dict] = field(default_factory=list)

    @property
    def removed_ids(self) -> list[str]:
        return self.removed_primary + self.removed_dependencies + self.removed_paraphrase

    def as_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---- candidate families (definitions only; arms not ratified) ----

def dead_end(item, approach: Approach | None) -> bool:
    return approach is not None and approach.status in DEAD_END_STATUSES


def noise(item, approach: Approach | None) -> bool:
    return item.label("role") == "narration" or item.label("restatement") == "restates_requirement"


_ALL_ROLES = frozenset(ROLE_OPTIONS) | {UNKNOWN}
CANDIDATE_FAMILIES: dict[str, FilterSpec] = {
    "forget_theories": FilterSpec("forget_theories", frozenset(ROLE_GROUPS["theories"])),
    "forget_plans": FilterSpec("forget_plans", frozenset(ROLE_GROUPS["plans"])),
    "forget_current_approach": FilterSpec("forget_current_approach", None),
    "forget_done_claims": FilterSpec("forget_done_claims", frozenset(ROLE_GROUPS["claims"])),
    "facts_only": FilterSpec("facts_only", _ALL_ROLES - ROLE_GROUPS["facts"], "global"),
    "forget_dead_ends": FilterSpec("forget_dead_ends", None, "global", dead_end),
    "denoise": FilterSpec("denoise", None, "global", noise),
}


def candidate_family(name: str, scope: str | None = None) -> FilterSpec:
    spec = CANDIDATE_FAMILIES[name]
    return spec if scope is None else replace(spec, scope=scope)


# ---- applying a filter ----

def _owners(packet: Packet22) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for u, o in packet.approach_items.items():
        for i in o["prose"]:
            out.setdefault(i, []).append(u)
    return out


def _predicate_holds(spec: FilterSpec, item, owners: list[str], by_unit: dict[str, Approach]) -> bool:
    if spec.unit_predicate is None:
        return True
    if not owners:
        return bool(spec.unit_predicate(item, None))
    return any(spec.unit_predicate(item, by_unit.get(u)) for u in owners)


def _scope_prose(packet: Packet22, view: View21, scope: str, current_id: str | None) -> list[str] | None:
    by_id = view.by_id
    if scope == "current_approach":
        if current_id is None or current_id not in packet.approach_items:
            return None
        return list(packet.approach_items[current_id]["prose"])
    return [i for i in packet.item_ids if by_id[i].source_kind in PROSE]


def _top_two(item) -> list[tuple[str, float]]:
    lab = item.labels.get("role")
    probs = (lab.probabilities or {}) if lab is not None else {}
    return sorted(((k, float(v)) for k, v in probs.items()), key=lambda kv: (-kv[1], kv[0]))[:2]


def borderline_reason(item, roles: frozenset[str]) -> str | None:
    top = _top_two(item)
    if len(top) == 2 and ((top[0][0] in roles) != (top[1][0] in roles)) and top[0][1] < ROLE_THRESHOLD \
            and top[0][1] + top[1][1] >= BORDERLINE_TOP2_SUM:
        return "straddles_role_set"
    if item.label("role") == UNKNOWN and any(r in roles for r, _ in top):
        return "unknown_with_filter_role_in_top_two"
    return None


def _borderline(view: View21, candidates: Sequence[str], removed: set[str], roles: frozenset[str] | None,
                count: Count) -> dict[str, Any]:
    if roles is None:
        return {"applicable": False, "reason": "filter has no role set"}
    by_id = view.by_id
    out: dict[str, Any] = {"applicable": True, "top2_sum": BORDERLINE_TOP2_SUM, "top1_below": ROLE_THRESHOLD}
    for side in ("kept", "removed"):
        out[side] = {"count": 0, "tokens": 0, "ids": [], "by_reason": {}}
    for iid in candidates:
        reason = borderline_reason(by_id[iid], roles)
        if reason is None:
            continue
        side = out["removed" if iid in removed else "kept"]
        side["count"] += 1
        side["tokens"] += item_tokens(by_id[iid], count)
        side["ids"].append(iid)
        side["by_reason"][reason] = side["by_reason"].get(reason, 0) + 1
    return out


def _restating(view: View21, packet_prose: set[str], removed: set[str]) -> list[str]:
    """Packet prose that explicitly restates a removed item, transitively."""
    restates = [l for l in view.links if l.relation == "restates" and l.tier == LINK_TIER_EXPLICIT]
    found: list[str] = []
    frontier = set(removed)
    while frontier:
        new = {l.source_id for l in restates if l.target_id in frontier and l.source_id in packet_prose}
        new -= removed | set(found)
        found.extend(sorted(new))
        frontier = new
    return found


def apply_filter(packet: Packet22, view: View21, approaches: Sequence[Approach], current: Approach | str | None,
                 spec: FilterSpec, count: Count, paraphrase_fn: ParaphraseFn | None = None) -> FilterView:
    by_id = view.by_id
    pos = {i.item_id: (i.available_seq, i.seq, i.order) for i in view.items}
    current_id = current.unit_id if isinstance(current, Approach) else current if current is not None else packet.current
    by_unit = {a.unit_id: a for a in approaches}
    owners = _owners(packet)
    packet_prose = {i for i in packet.item_ids if by_id[i].source_kind in PROSE}
    full_text = render22(view, packet.item_ids, approaches=approaches)
    scope_ids = _scope_prose(packet, view, spec.scope, current_id)
    status = "applied"
    if scope_ids is None:
        status, scope_ids = "no_current_approach_in_packet", []
    pool = [i for i in scope_ids if _predicate_holds(spec, by_id[i], owners.get(i, []), by_unit)]
    primary = [i for i in pool if spec.roles is None or by_id[i].label("role") in spec.roles]
    removed = set(primary)
    deps = _restating(view, packet_prose, removed)
    removed.update(deps)
    para_ids: list[str] = []
    paraphrase: dict[str, Any] = {"availability": "not_run"}
    if paraphrase_fn is not None and removed:
        kept_items = [by_id[i] for i in sorted(packet_prose - removed, key=pos.get)]
        removed_items = [by_id[i] for i in sorted(removed, key=pos.get)]
        results = paraphrase_fn(kept_items, removed_items) if kept_items else {}
        para_ids = sorted((k for k, r in results.items() if r.get("value") == PARAPHRASE_VALUE and k in packet_prose
                           and k not in removed), key=pos.get)
        paraphrase = {"availability": "run", "asked": len(kept_items), "against": len(removed_items),
                      "removed": len(para_ids), "results": results}
        removed.update(para_ids)
        later_deps = _restating(view, packet_prose, removed)
        deps += later_deps
        removed.update(later_deps)
    if status == "applied" and not removed:
        status = "nothing_to_remove"
    kept = [i for i in packet.item_ids if i not in removed]
    removed_tokens = sum(item_tokens(by_id[i], count) for i in removed)
    packet_tokens = sum(item_tokens(by_id[i], count) for i in packet.item_ids)
    same_turn = {}
    for l in view.links:
        if l.relation == "same_turn" and l.source_id in removed and l.target_id in by_id:
            same_turn[l.target_id] = by_id[l.target_id].text
    return FilterView(
        name=spec.name, spec=spec.as_record(), status=status, current=current_id, item_ids=kept,
        removed_primary=sorted(primary, key=pos.get), removed_dependencies=sorted(deps, key=pos.get),
        removed_paraphrase=para_ids, removed_tokens=removed_tokens, packet_tokens=packet_tokens,
        removed_share=removed_tokens / packet_tokens if packet_tokens else 0.0,
        rendered_tokens_full=count(full_text), rendered_tokens_kept=count(render22(view, kept, approaches=approaches)),
        paraphrase=paraphrase,
        borderline=_borderline(view, sorted(set(pool) | removed, key=pos.get), removed, spec.roles, count),
        removed_actions=[{"action_id": a, "text": t} for a, t in sorted(same_turn.items(), key=lambda kv: pos[kv[0]])])


# ---- matched random controls ----

def matched_random(packet: Packet22, view: View21, removed_tokens: int, seed: int, scope: str, count: Count, *,
                   exclude: Collection[str] = ()) -> dict[str, Any]:
    """Delete complete non-evidence statements from the scope matching ``removed_tokens`` within ±RANDOM_TOLERANCE
    (approach-member roles first, then any non-evidence statement), as in ``packet21.random_unit_deletion``.
    ``exclude`` keeps ids out of the pool (normally the targeted removal). Never refills."""
    if scope not in FILTER_SCOPES:
        raise ValueError(f"scope must be one of {FILTER_SCOPES}, got {scope!r}")
    if removed_tokens <= 0:
        return {"status": "no_target", "seed": seed, "scope": scope, "item_ids": list(packet.item_ids)}
    by_id = view.by_id
    scope_ids = _scope_prose(packet, view, scope, packet.current) or []
    skip = set(exclude)
    pool = [by_id[i] for i in scope_ids if i not in skip and by_id[i].label("role") not in EVIDENCE_ROLES]
    tiers = ((([i for i in pool if i.label("role") in APPROACH_MEMBER_ROLES]), "approach_member_roles"),
             (pool, "any_non_evidence_statement"))
    rng = random.Random(seed)
    low, high = removed_tokens * (1 - RANDOM_TOLERANCE), removed_tokens * (1 + RANDOM_TOLERANCE)
    for candidates, composition in tiers:
        if not candidates:
            continue
        for _ in range(300):
            order = candidates[:]
            rng.shuffle(order)
            chosen, mass = [], 0
            for item in order:
                c = item_tokens(item, count)
                if mass + c <= high:
                    chosen.append(item.item_id)
                    mass += c
                if mass >= low:
                    drop = set(chosen)
                    return {"status": "matched", "seed": seed, "scope": scope, "composition": composition,
                            "target_tokens": removed_tokens, "deleted_tokens": mass,
                            "mismatch": (mass - removed_tokens) / removed_tokens, "deleted_ids": sorted(drop),
                            "item_ids": [i for i in packet.item_ids if i not in drop]}
    return {"status": "no_matched_random_control", "seed": seed, "scope": scope, "target_tokens": removed_tokens,
            "pool_size": len(pool), "item_ids": list(packet.item_ids)}


def matched_randoms(packet: Packet22, view: View21, fview: FilterView, count: Count, *, scope: str | None = None,
                    seeds: Sequence[int] = RANDOM_SEEDS) -> list[dict[str, Any]]:
    """The same-size random controls for one filter view: one per seed, excluding the view's removed items."""
    scope = scope or fview.spec["scope"]
    return [matched_random(packet, view, fview.removed_tokens, s, scope, count, exclude=fview.removed_ids)
            for s in seeds]
