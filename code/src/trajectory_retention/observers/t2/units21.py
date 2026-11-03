"""T2.1 checkpoint representation, pursuit units and the frozen selector (guide §8–§10).

A pursuit groups approach-advocating worker statements connected by
structural or expressed (classifier-judged explicit) edges. It records, and
keeps separate:

* approach_members      - explanations/hypotheses/strategies/intended actions/questions of the unit
* associated_evidence   - raw tool-output results of the unit's actions and worker observation
                          reports explicitly tied to it (never removed by the targeted omission)
* execution_events      - the actions issued in the same responses as members
* reference_dependencies- later statements that only restate a member (removed with it)
* supporting_edges      - edges that justified membership
* support_class         - "expressed" (structural + expressed edges only)

Observer-inferred (below-threshold) edges never define primary units; they
build a separate exploratory grouping. Status is taken only from explicit
withdraws/replaces/confirms edges; silence, reverts and later inactivity stay
``unknown``. Last activity is stored separately from status.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import Link
from .sources import CheckpointInfo

APPROACH_ROLES = {"explanation_or_hypothesis", "strategy_or_design_choice", "intended_action", "unresolved_question"}
EVIDENCE_ROLES = {"observation_report", "factual_assertion"}
MEMBER_RELATIONS = {"continues_or_builds_on", "tests_or_checks", "restates"}
STATUS_RELATIONS = {"withdraws": "withdrawn", "replaces": "replaced", "confirms": "confirmed"}
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
SELECTOR_VERSION = "t2.1-selector/v1"


@dataclass
class View21:
    checkpoint: CheckpointInfo
    items: list
    links: list[Link]

    @property
    def by_id(self) -> dict[str, Any]:
        return {i.item_id: i for i in self.items}


def at_checkpoint21(ledger, checkpoint: CheckpointInfo, links: Sequence[Link], *, session_scoped: bool = True) -> View21:
    """Prefix-only frozen view: items available at the cut (same native session) and links known by then."""
    cut = checkpoint.event_sequence
    items = [i for i in ledger.items if i.available_seq <= cut
             and (not session_scoped or i.session_id == checkpoint.session_id)]
    ids = {i.item_id for i in items}
    return View21(checkpoint, items, [l for l in links if l.evidence_seq <= cut and l.source_id in ids and l.target_id in ids])


@dataclass
class Pursuit:
    unit_id: str
    approach_members: list[str]
    associated_evidence: list[str]
    execution_events: list[str]
    reference_dependencies: list[str]
    supporting_edges: list[dict]
    support_class: str
    status: str
    status_edge: dict | None
    last_activity_seq: int
    first_seq: int
    checkpoint_span: int
    attempts: int
    entangled_mixed_members: list[str] = field(default_factory=list)
    valid_removal: bool = True
    invalid_reasons: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_pursuits(view: View21, *, checkpoint_cuts: Sequence[int], tier: str = "expressed_classified") -> list[Pursuit]:
    by_id = view.by_id
    role = lambda i: i.label("role")
    approach = {i.item_id for i in view.items if i.source_kind in PROSE and role(i) in APPROACH_ROLES | {"mixed"}}
    uf = _UF()
    support: dict[str, list[dict]] = defaultdict(list)
    for iid in approach:
        uf.find(iid)
    edges = [l for l in view.links if l.tier == tier]
    for l in edges:
        if l.relation in MEMBER_RELATIONS and l.source_id in approach and l.target_id in approach:
            uf.union(l.source_id, l.target_id)
            support[l.source_id].append({"from": l.source_id, "to": l.target_id, "relation": l.relation,
                                         "tier": l.tier, "p": l.probability})
    groups: dict[str, list[str]] = defaultdict(list)
    for iid in approach:
        groups[uf.find(iid)].append(iid)

    same_turn: dict[str, list[str]] = defaultdict(list)
    results: dict[str, list[str]] = defaultdict(list)
    for l in view.links:
        if l.relation == "same_turn":
            same_turn[l.source_id].append(l.target_id)
        elif l.relation == "result_of":
            results[l.target_id].append(l.source_id)
    incoming: dict[str, list[Link]] = defaultdict(list)
    for l in edges:
        incoming[l.target_id].append(l)

    def interval(seq: int) -> int:
        return sum(seq > c for c in checkpoint_cuts)

    out = []
    for root, members in groups.items():
        members = sorted(set(members), key=lambda m: (by_id[m].available_seq, by_id[m].seq, by_id[m].order))
        mset = set(members)
        actions = sorted({a for m in members for a in same_turn.get(m, ())})
        evidence = sorted({o for a in actions for o in results.get(a, ())})
        deps, extra_evidence, status, status_edge = [], [], "unknown", None
        for m in members:
            for l in incoming.get(m, ()):
                src = by_id.get(l.source_id)
                if src is None or l.source_id in mset:
                    continue
                if l.relation in STATUS_RELATIONS:
                    if status_edge is None or l.evidence_seq >= status_edge["evidence_seq"]:
                        status, status_edge = STATUS_RELATIONS[l.relation], l.as_record()
                elif l.relation == "restates" and role(src) not in EVIDENCE_ROLES:
                    deps.append(l.source_id)
                elif role(src) in EVIDENCE_ROLES:
                    extra_evidence.append(l.source_id)
        evidence = sorted(set(evidence) | set(extra_evidence))
        seqs = [by_id[m].available_seq for m in members] + [by_id[a].seq for a in actions]
        mixed = [m for m in members if role(by_id[m]) == "mixed"]
        p = Pursuit(unit_id=root, approach_members=members, associated_evidence=evidence, execution_events=actions,
                    reference_dependencies=sorted(set(deps) - mset), supporting_edges=[e for m in members for e in support[m]],
                    support_class="expressed" if tier == "expressed_classified" else "observer_inferred",
                    status=status, status_edge=status_edge, last_activity_seq=max(seqs), first_seq=min(seqs),
                    checkpoint_span=len({interval(s) for s in seqs}), attempts=len(actions), entangled_mixed_members=mixed)
        if set(p.associated_evidence) & (mset | set(p.reference_dependencies)):
            p.valid_removal, p.invalid_reasons = False, p.invalid_reasons + ["evidence_overlaps_removal_set"]
        if mixed:
            p.valid_removal, p.invalid_reasons = False, p.invalid_reasons + ["mixed_member_would_remove_evidence"]
        out.append(p)
    return out


def select_unit(pursuits: Sequence[Pursuit]) -> dict[str, Any]:
    """Frozen prefix-only selector (guide §10). Never inspects outcomes or asks which approach is wrong.

    Eligible: expressed support, >=2 approach members, >=2 distinct linked worker attempts (actions),
    valid evidence-preserving removal. Cross-checkpoint status is recorded, not required.
    Rank: latest linked activity (descending), then earliest source order, then unit id.
    """
    eligible = [p for p in pursuits if p.support_class == "expressed" and len(p.approach_members) >= 2
                and p.attempts >= 2 and p.valid_removal]
    ranked = sorted(eligible, key=lambda p: (-p.last_activity_seq, p.first_seq, p.unit_id))
    return {"selector_version": SELECTOR_VERSION,
            "n_pursuits": len(pursuits),
            "n_multi_member": sum(len(p.approach_members) >= 2 for p in pursuits),
            "n_eligible": len(eligible),
            "n_cross_checkpoint_eligible": sum(p.checkpoint_span >= 2 for p in eligible),
            "case": "zero" if not eligible else "one" if len(eligible) == 1 else "multiple",
            "selected": ranked[0].unit_id if ranked else None,
            "ineligible_reasons": _reasons(pursuits)}


def _reasons(pursuits: Sequence[Pursuit]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for p in pursuits:
        if len(p.approach_members) < 2:
            c["single_member"] += 1
            continue
        if p.attempts < 2:
            c["fewer_than_two_attempts"] += 1
        for r in p.invalid_reasons:
            c[r] += 1
    return dict(c)
