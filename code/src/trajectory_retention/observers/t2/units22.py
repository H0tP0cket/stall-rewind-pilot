"""T2.2 approach units and the frozen selector.

Joining uses only explicit-wording links (tier ``explicit_wording``) and the
structural links from the event log (``same_turn``, ``result_of``). A shared
named thing (tier ``same_thing_named``) and below-threshold links never join
anything.

Worker-prose statements whose main role is in APPROACH_MEMBER_ROLES are joined
over explicit links whose relation is in MEMBER_RELATIONS. A group is an
approach only if it contains a root (``proposal`` or ``commitment``); other
groups are recorded as clusters. Each approach records, separately:

* core_members          - the joined statements
* actions               - worker actions issued in the same response as a core member (``same_turn``)
* outputs               - tool outputs ``result_of`` those actions; evidence, kept on removal
* readings              - statements with an explicit reads_output_* link to one of those outputs (members)
* revisions, result_claims - members with those main roles, including revision/result_claim statements
                          attached by any explicit link from them to a core member
* linked_evidence       - observation/fact statements with a non-restating member-relation link to a
                          member; kept on removal (T2.1 associated evidence)
* reference_dependencies- other statements that explicitly restate a member (removed with it)

``members`` = core members + readings + attached statements. Only statements
outside every approach's core are attached, each to one approach: the one
whose linked item is latest. Status comes only from explicit links in
STATUS_RELATIONS that point at a member (reads_output_* at an output): the
latest edge wins and the default is ``open``; silence, reverts and inactivity
never set it. ``action_carried_out`` records, for each commitment and
intended_action member, whether a worker action in the same or the next
response names one of its artifacts (``None`` when it names none, or when the
next response is not in the view). ``valid_removal`` is as in T2.1.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .schema import UNKNOWN, Link
from .units21 import PROSE, SELECTOR_VERSION as T21_SELECTOR_VERSION, View21, _UF, at_checkpoint21
from .vocab22 import (APPROACH_MEMBER_ROLES, APPROACH_ROOT_ROLES, EVIDENCE_ROLES, LINK_TIER_BUILTIN,
                      LINK_TIER_CANDIDATE, LINK_TIER_EXPLICIT, MEMBER_RELATIONS, OUTPUT_RELATIONS,
                      STATUS_RELATIONS)

__all__ = ["Approach", "Cluster", "Structure", "View21", "at_checkpoint21", "build_approaches", "build_structure",
           "select_current", "UNITS22_VERSION", "SELECTOR22_VERSION"]

UNITS22_VERSION = "t2.2-units/v1"
SELECTOR22_VERSION = "t2.2-selector/v1"
ATTACHED_ROLES = {"revision", "result_claim"}
CARRIED_OUT_ROLES = {"commitment", "intended_action"}
Named = Callable[[str], frozenset]


def default_named() -> tuple[Named, str]:
    """Named-artifact extractor: ``artifacts22.named_artifacts`` when present, else T2.1 path/test tags."""
    try:
        from .artifacts22 import named_artifacts  # type: ignore[import-not-found]
        return named_artifacts, "artifacts22.named_artifacts"
    except ImportError:
        from .segment import tags_for
        return (lambda text: frozenset(tags_for(text))), "segment.tags_for"


def _edge(link: Link) -> dict[str, Any]:
    return {"from": link.source_id, "to": link.target_id, "relation": link.relation, "tier": link.tier,
            "p": link.probability}


@dataclass
class Approach:
    unit_id: str
    roots: list[str]
    members: list[str]
    core_members: list[str]
    readings: list[str]
    revisions: list[str]
    result_claims: list[str]
    attachment_edges: list[dict]
    actions: list[str]
    outputs: list[str]
    linked_evidence: list[str]
    reference_dependencies: list[str]
    supporting_edges: list[dict]
    status: str
    status_edge: dict | None
    action_carried_out: dict[str, dict]
    repeat_counts: dict[str, Any]
    first_seq: int
    last_activity_seq: int
    checkpoint_span: int
    attempts: int
    entangled_mixed_members: list[str] = field(default_factory=list)
    valid_removal: bool = True
    invalid_reasons: list[str] = field(default_factory=list)

    @property
    def evidence(self) -> list[str]:
        return self.outputs + self.linked_evidence

    def as_record(self) -> dict[str, Any]:
        return {"kind": "approach", **self.__dict__}


@dataclass
class Cluster:
    unit_id: str
    members: list[str]
    supporting_edges: list[dict]
    roles: dict[str, int]

    def as_record(self) -> dict[str, Any]:
        return {"kind": "cluster", **self.__dict__}


@dataclass
class Structure:
    approaches: list[Approach]
    clusters: list[Cluster]
    stats: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {"version": UNITS22_VERSION, "approaches": [a.as_record() for a in self.approaches],
                "clusters": [c.as_record() for c in self.clusters], "stats": self.stats}


INFERRED_TIER = "inferred_same_approach"   # Jev same-approach judgement; joins only when join_inferred=True
INFERRED_RELATION = "same_approach"


def build_approaches(view: View21, checkpoint_cuts: Sequence[int], *, named: Named | None = None,
                     join_inferred: bool = False) -> list[Approach]:
    return build_structure(view, checkpoint_cuts, named=named, join_inferred=join_inferred).approaches


DECISION_CLOSERS = {"replaces": "replaced", "withdraws": "withdrawn"}


def _decision_bounded(order: list[str], role, links_from: dict[str, list], roots_set: set[str]) -> tuple[dict, list, dict]:
    """Group statements into approaches bounded by the agent's decisions (user rule, 2026-09-27).

    An approach is ONE proposal/commitment (its root) plus statements directly linked into it while it is open; it
    closes when the agent replaces or withdraws it, or commits to something new that does not link to it. Old
    approaches stay separate objects with statuses. Failure modes this guards against (written before the code):
      1. mega-approach from chaining a whole reasoning burst -> chaining only happens inside one open decision and
         stops at every closure; the largest approach and its share of prose are reported;
      2. fragmentation (every "maybe X" its own approach) -> a root that links to an open approach refines it
         (proposal -> commitment adoption) instead of starting a new one;
      3. wrong closure by a sub-step commitment ("I'll edit file X") -> only an UNLINKED commitment closes;
      4. lost lead-up (the question the decision answers) -> the root's direct link targets join, one hop only;
      5. revisits much later -> a statement linking into a closed approach reopens it (recorded);
      6. a statement linking into two approaches -> the most recently active one wins (counted);
      7. unknown-role statements breaking chains -> they may join when explicitly linked (never as roots);
      8. prefix safety -> only links already in the view are used; outputs are never members.
    Returns (rooted: unit -> members, unrooted groups, meta: unit -> {closed_by, reopened}).
    """
    owner: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    open_: dict[str, bool] = {}
    meta: dict[str, dict] = {}
    last_active: dict[str, int] = {}
    loose: list[str] = []
    current: str | None = None
    for n, s in enumerate(order):
        tgts = links_from.get(s, [])
        for t, rel in tgts:
            u = owner.get(t)
            if u is not None and rel in DECISION_CLOSERS and open_[u]:
                open_[u] = False
                meta[u]["closed_by"] = DECISION_CLOSERS[rel]
        joined = [owner[t] for t, rel in tgts if t in owner and rel in MEMBER_RELATIONS]
        target = max(joined, key=lambda u: last_active[u]) if joined else None
        if joined and len(set(joined)) > 1:
            meta[target]["contested"] = meta[target].get("contested", 0) + 1
        if s in roots_set and target is None:
            if role(s) == "commitment" and current is not None and open_[current]:
                open_[current] = False
                meta[current]["closed_by"] = "superseded"
            members[s], owner[s], open_[s], meta[s], last_active[s] = [s], s, True, {"closed_by": None}, n
            for t, rel in tgts:  # lead-up: what the decision directly answers or builds on (one hop back)
                if t not in owner and t in role.nodes and rel in MEMBER_RELATIONS:
                    members[s].append(t)
                    owner[t] = s
            current = s
        elif target is not None:
            if not open_[target]:
                open_[target] = True
                meta[target]["reopened"] = meta[target].get("reopened", 0) + 1
                meta[target]["closed_by"] = None
            members[target].append(s)
            owner[s] = target
            last_active[target] = n
            current = target
        else:
            loose.append(s)
    return members, [[x] for x in loose if x not in owner], meta


def _implemented_decision_windows(view, prose: set[str], explicit: list, commitments: frozenset[str], pos):
    """Approaches as contiguous time windows anchored on CARRIED-OUT decisions (user 2026-09-27).

    - Timeline: every item of the view in time order.
    - Anchor: a sentence labelled action=decision followed by a commitment (a relevant workspace/runtime mutation or a
      native grader submission) before the next decision sentence.
    - Window 0 starts at the start of the run. Window m>0 starts at the earliest of (a) the first item of the anchor's
      own response and (b) the sentences the anchor explicitly links back to (ONE hop, no transitive backfill), but
      never before the previous window's last commitment and that commitment's outputs.
    - A window ends where the next one starts; the last window runs to the stall.
    - Override: a sentence BEFORE the last anchor that explicitly links to an earlier window's anchor moves to that
      window. Sentences after the last anchor always stay in the current (last) window.
    - Actions belong to the window they fall in (one owner). With no anchor there is one window (flagged)."""
    items = sorted(view.items, key=lambda i: (i.available_seq, i.seq, i.order))
    idx = {i.item_id: k for k, i in enumerate(items)}
    n = len(items)
    decisions = [k for k, i in enumerate(items) if i.item_id in prose and i.label("action") == "decision"]
    commits = sorted(idx[a] for a in commitments if a in idx)
    anchors = []
    for m, k in enumerate(decisions):
        nxt = decisions[m + 1] if m + 1 < len(decisions) else n
        if any(k < c < nxt for c in commits):
            anchors.append(k)
    out_of: dict[str, list[int]] = defaultdict(list)  # action -> indexes of its outputs
    links_from: dict[str, list[str]] = defaultdict(list)
    for l in view.links:
        if l.relation == "result_of" and l.source_id in idx and l.target_id in idx:
            out_of[l.target_id].append(idx[l.source_id])
    for l in explicit:
        links_from[l.source_id].append(l.target_id)
    meta: dict[str, Any] = {"n_items": n, "n_decisions": len(decisions), "n_commitments": len(commits),
                            "anchors": [items[k].item_id for k in anchors], "no_anchor": not anchors}
    if not anchors:
        u = items[0].item_id if items else "empty"
        members = [i.item_id for i in items if i.item_id in prose]
        if not members:  # nothing the agent wrote yet (e.g. C0): no approach
            meta["windows"] = []
            return {}, [], {}, meta, {}
        acts = [i.item_id for i in items if i.source_kind == "worker_action"]
        meta["windows"] = [{"unit": u, "anchor": None, "start": 0, "end": n, "backfill": None}]
        return {u: members}, [], {}, meta, {u: acts}
    starts, backfill = [0], [None]
    for m in range(1, len(anchors)):
        a_k = anchors[m]
        prev_commits = [c for c in commits if anchors[m - 1] < c < a_k]
        lb = 0
        if prev_commits:
            c = prev_commits[-1]
            lb = max([c] + out_of.get(items[c].item_id, [])) + 1
        resp = [k for k in range(n) if items[k].seq == items[a_k].seq and items[k].session_id == items[a_k].session_id]
        cand = [max(lb, min(resp))] if resp else [a_k]
        hop = [idx[t] for t in links_from.get(items[a_k].item_id, ()) if t in idx and lb <= idx[t] < a_k]
        starts.append(min(cand + hop))
        backfill.append((a_k - min(hop)) if hop else 0)
    bounds = starts + [n]
    win_of = [0] * n
    for m in range(len(anchors)):
        for k in range(bounds[m], bounds[m + 1]):
            win_of[k] = m
    last_anchor = anchors[-1]
    anchor_win = {items[k].item_id: m for m, k in enumerate(anchors)}
    moved = 0
    for k in range(n):
        iid = items[k].item_id
        if iid not in prose or k >= last_anchor or k in anchors:
            continue
        back = [anchor_win[t] for t in links_from.get(iid, ()) if t in anchor_win and anchor_win[t] < win_of[k]]
        if back:
            win_of[k] = min(back)
            moved += 1
    units = [items[k].item_id for k in anchors]
    rooted = {u: [] for u in units}
    acts: dict[str, list[str]] = {u: [] for u in units}
    for k in range(n):
        u = units[win_of[k]]
        if items[k].item_id in prose:
            rooted[u].append(items[k].item_id)
        elif items[k].source_kind == "worker_action":
            acts[u].append(items[k].item_id)
    closure = {u: {"closed_by": "superseded"} for u in units[:-1]}
    meta["override_moved"] = moved
    meta["windows"] = [{"unit": units[m], "anchor": units[m], "start": bounds[m], "end": bounds[m + 1],
                        "start_item": items[bounds[m]].item_id, "backfill": backfill[m]} for m in range(len(anchors))]
    return rooted, [], closure, meta, acts


def build_structure(view: View21, checkpoint_cuts: Sequence[int], *, named: Named | None = None,
                    join_inferred: bool = False, grouping: str = "decision_bounded",
                    extra_roots: frozenset[str] = frozenset(), commitments: frozenset[str] = frozenset()) -> Structure:
    """``grouping``: "decision_bounded" (default; see ``_decision_bounded``) or "components" (the earlier
    connected-component grouping, kept for comparison).

    ``join_inferred`` applies the ratified contingency tier (Jev same-approach, for when explicit-only approaches
    stay tiny) WITHOUT chaining: after the explicit groups are formed, a statement outside every group's core joins
    an approach only through an inferred link directly to one of that approach's roots (proposal/commitment); the
    latest such root wins. Inferred links between non-root statements never join anything, so consecutive
    investigation steps cannot chain into one run-long "approach". The default builds explicit-only units, so every
    result can be recomputed on explicit links alone."""
    by_id = view.by_id
    extractor = "caller"
    if named is None:
        named, extractor = default_named()
    role = lambda iid: by_id[iid].label("role")
    pos = lambda iid: (by_id[iid].available_seq, by_id[iid].seq, by_id[iid].order)
    prose = {i.item_id for i in view.items if i.source_kind in PROSE}
    explicit = [l for l in view.links if l.tier == LINK_TIER_EXPLICIT and l.source_id in by_id and l.target_id in by_id]

    # 1. groups
    nodes = {i for i in prose if role(i) in APPROACH_MEMBER_ROLES}
    support = defaultdict(list)
    closure_meta: dict[str, dict] = {}
    windows_meta: dict[str, Any] | None = None
    if grouping == "components":
        uf = _UF()
        for n in nodes:
            uf.find(n)
        for l in explicit:
            if l.relation in MEMBER_RELATIONS and l.source_id in nodes and l.target_id in nodes:
                uf.union(l.source_id, l.target_id)
                support[l.source_id].append(_edge(l))
        groups: dict[str, list[str]] = defaultdict(list)
        for n in sorted(nodes, key=pos):
            groups[uf.find(n)].append(n)
        rooted, unrooted = {}, []
        for g in groups.values():
            roots_g = [m for m in g if role(m) in APPROACH_ROOT_ROLES]
            if roots_g:
                rooted[roots_g[0]] = g  # keyed by the earliest root
            else:
                unrooted.append(g)
    elif grouping == "implemented_decision":
        rooted, unrooted, closure_meta, windows_meta, win_actions = _implemented_decision_windows(
            view, prose, explicit, commitments, pos)
        groups = dict(rooted)
    else:
        linkable = nodes | {i for i in prose if role(i) == UNKNOWN}  # failure mode 7
        links_from: dict[str, list] = defaultdict(list)
        for l in sorted(explicit, key=lambda l: pos(l.target_id)):
            if l.source_id in linkable and l.target_id in linkable:
                links_from[l.source_id].append((l.target_id, l.relation))
                if l.relation in MEMBER_RELATIONS or l.relation in DECISION_CLOSERS:
                    support[l.source_id].append(_edge(l))

        class _Role:
            def __call__(self, iid):
                return role(iid)
        r = _Role()
        r.nodes = linkable
        # roots: proposals/commitments, hypotheses, and (user A5) the sentence that goes with the first
        # distinctive state-changing action of a response when no plan was written (``extra_roots``)
        roots = {i for i in nodes if role(i) in APPROACH_ROOT_ROLES | {"explanation_or_hypothesis"}}
        roots |= {i for i in extra_roots if i in linkable}
        rooted, unrooted, closure_meta = _decision_bounded(sorted(linkable, key=pos), r, links_from, roots)
        rooted = {u: sorted(m, key=pos) for u, m in rooted.items()}
        groups = {**rooted, **{g[0]: g for g in unrooted}}
    core_owner = {m: u for u, g in rooted.items() for m in g}
    inferred_joined = 0
    if join_inferred:
        root_owner = {m: u for u, g in rooted.items() for m in g if role(m) in APPROACH_ROOT_ROLES}
        choice: dict[str, tuple] = {}
        for l in view.links:
            if l.tier != INFERRED_TIER or l.relation != INFERRED_RELATION:
                continue
            for n, r in ((l.source_id, l.target_id), (l.target_id, l.source_id)):
                if n in nodes and n not in core_owner and r in root_owner and (n not in choice or pos(r) > choice[n][0]):
                    choice[n] = (pos(r), root_owner[r], l)
        for n, (_, u, l) in choice.items():
            rooted[u].append(n)
            core_owner[n] = u
            support[n].append(_edge(l))
            inferred_joined += 1
        for g in rooted.values():
            g.sort(key=pos)
        joined_ids = set(choice)
        unrooted = [[m for m in g if m not in joined_ids] for g in unrooted]
        unrooted = [g for g in unrooted if g]

    # 2. actions and outputs from structural links
    same_turn, results = defaultdict(set), defaultdict(set)
    for l in view.links:
        if l.tier != LINK_TIER_BUILTIN or l.source_id not in by_id or l.target_id not in by_id:
            continue
        if l.relation == "same_turn" and by_id[l.target_id].source_kind == "worker_action":
            same_turn[l.source_id].add(l.target_id)
        elif l.relation == "result_of":
            results[l.target_id].add(l.source_id)
    actions = {u: sorted({a for m in g for a in same_turn.get(m, ())}, key=pos) for u, g in rooted.items()}
    if grouping == "implemented_decision":  # time containment: every action has exactly one owner, its window
        actions = {u: sorted(win_actions.get(u, ()), key=pos) for u in rooted}
    outputs = {u: sorted({o for a in acts for o in results.get(a, ())}, key=pos) for u, acts in actions.items()}
    output_owners: dict[str, list[str]] = defaultdict(list)
    for u, outs in outputs.items():
        for o in outs:
            output_owners[o].append(u)

    # 3. attach readings and revision/result_claim statements that are outside every core
    options: dict[str, list[tuple]] = defaultdict(list)
    for l in explicit:
        s = l.source_id
        if s not in prose or s in core_owner:
            continue
        if l.relation in OUTPUT_RELATIONS:
            options[s] += [(pos(l.target_id), u, l, "reading") for u in output_owners.get(l.target_id, ())]
        elif role(s) in ATTACHED_ROLES and l.target_id in core_owner:
            options[s].append((pos(l.target_id), core_owner[l.target_id], l, "attached"))
    attached: dict[str, list[tuple[str, Link, str]]] = defaultdict(list)
    contested = 0
    for s, opts in options.items():
        if not opts:  # a reading of an output no approach owns
            continue
        contested += len({o[1] for o in opts}) > 1
        best = sorted(sorted(opts, key=lambda o: o[1]), key=lambda o: o[0], reverse=True)[0]
        attached[best[1]].append((s, best[2], best[3]))
    member_owner = dict(core_owner)
    for u, rows in attached.items():
        for s, _, _ in rows:
            member_owner[s] = u

    # 4. per-approach records
    worker_seqs: dict[Any, list[int]] = defaultdict(list)
    actions_at: dict[tuple, list[str]] = defaultdict(list)
    for i in view.items:
        if i.source_kind in PROSE or i.source_kind == "worker_action":
            worker_seqs[i.session_id].append(i.seq)
        if i.source_kind == "worker_action":
            actions_at[(i.session_id, i.seq)].append(i.item_id)
    worker_seqs = {k: sorted(set(v)) for k, v in worker_seqs.items()}
    names: dict[str, frozenset] = {}

    def art(iid: str) -> frozenset:
        if iid not in names:
            names[iid] = frozenset(named(by_id[iid].text))
        return names[iid]

    def carried_out(m: str) -> dict[str, Any]:
        item = by_id[m]
        arts = art(m)
        if not arts:
            return {"value": None, "reason": "names_no_artifact", "artifacts": [], "matched_action": None,
                    "extractor": extractor}
        later = [s for s in worker_seqs.get(item.session_id, []) if s > item.seq]
        nxt = later[0] if later else None
        for where, seq in (("same_response", item.seq), ("next_response", nxt)):
            for a in actions_at.get((item.session_id, seq), []) if seq is not None else []:
                if art(a) & arts:
                    return {"value": True, "reason": f"named_in_{where}", "artifacts": sorted(arts),
                            "matched_action": a, "extractor": extractor}
        return {"value": None if nxt is None else False,
                "reason": "next_response_not_in_view" if nxt is None else "not_named",
                "artifacts": sorted(arts), "matched_action": None, "extractor": extractor}

    def interval(seq: int) -> int:
        return sum(seq > c for c in checkpoint_cuts)

    approaches = []
    for u, core in rooted.items():
        rows = attached.get(u, [])
        members = sorted(set(core) | {s for s, _, _ in rows}, key=pos)
        mset, oset = set(members), set(outputs[u])
        readings = sorted((s for s, _, kind in rows if kind == "reading"), key=pos)
        deps, linked, restates, status_edges = set(), set(), 0, []
        for n, l in enumerate(explicit):
            src = l.source_id
            if l.relation in STATUS_RELATIONS and (l.target_id in (oset if l.relation in OUTPUT_RELATIONS else mset)):
                status_edges.append((l.evidence_seq, pos(src), n, l))
            if l.target_id not in mset:
                continue
            if l.relation == "restates":
                restates += 1
                if src in prose and src not in mset and src not in member_owner:
                    deps.add(src)
            elif (l.relation in MEMBER_RELATIONS and src in prose and src not in mset
                  and role(src) in EVIDENCE_ROLES):
                linked.add(src)
        # agent-claimed status: only the agent's own words (explicit status links, or a decision closure);
        # never a judgement of correctness
        status, status_edge = "open", None
        if status_edges:
            latest = max(status_edges, key=lambda e: e[:3])[3]
            status, status_edge = STATUS_RELATIONS[latest.relation], latest.as_record()
        closed_by = (closure_meta.get(u) or {}).get("closed_by")
        if closed_by and status in ("open", "confirmed"):
            status, status_edge = closed_by, {"closure": closed_by}
        commands = Counter(" ".join(by_id[a].text.split()) for a in actions[u])
        seqs = [by_id[m].available_seq for m in members] + [by_id[a].seq for a in actions[u]]
        mixed = [m for m in members if role(m) == "mixed"]
        a = Approach(
            unit_id=u, roots=[m for m in members if role(m) in APPROACH_ROOT_ROLES], members=members,
            core_members=list(core), readings=readings,
            revisions=[m for m in members if role(m) == "revision"],
            result_claims=[m for m in members if role(m) == "result_claim"],
            attachment_edges=[{**_edge(l), "attached_as": kind} for _, l, kind in rows],
            actions=actions[u], outputs=outputs[u], linked_evidence=sorted(linked - mset, key=pos),
            reference_dependencies=sorted(deps, key=pos),
            supporting_edges=[e for m in core for e in support[m]], status=status, status_edge=status_edge,
            action_carried_out={m: carried_out(m) for m in members if role(m) in CARRIED_OUT_ROLES},
            repeat_counts={"commands": {c: k for c, k in commands.items() if k >= 2},
                           "repeated_commands": sum(k - 1 for k in commands.values()),
                           "distinct_commands": len(commands), "restates_links": restates},
            first_seq=min(seqs), last_activity_seq=max(seqs), checkpoint_span=len({interval(s) for s in seqs}),
            attempts=len(actions[u]), entangled_mixed_members=mixed)
        removal = mset | set(a.reference_dependencies)
        if set(a.evidence) & removal:
            a.valid_removal, a.invalid_reasons = False, a.invalid_reasons + ["evidence_overlaps_removal_set"]
        if mixed:
            a.valid_removal, a.invalid_reasons = False, a.invalid_reasons + ["mixed_member_would_remove_evidence"]
        approaches.append(a)
    approaches.sort(key=lambda a: (a.first_seq, a.unit_id))

    clusters = []
    for g in unrooted:
        rest = [m for m in g if m not in member_owner]
        if rest:
            clusters.append(Cluster(rest[0], rest, [e for m in rest for e in support[m]],
                                    dict(Counter(role(m) for m in rest))))
    sizes = Counter(len(a.members) for a in approaches)
    stats = {"version": UNITS22_VERSION, "items": len(view.items), "member_role_statements": len(nodes),
             "groups": len(groups), "approaches": len(approaches), "clusters": len(clusters),
             "approach_sizes": dict(sorted(sizes.items())),
             "attached": dict(Counter(kind for rows in attached.values() for _, _, kind in rows)),
             "contested_attachments": contested,
             "statuses": dict(Counter(a.status for a in approaches)),
             "links_by_tier": dict(Counter(l.tier for l in view.links)),
             "candidate_links_ignored": sum(l.tier == LINK_TIER_CANDIDATE for l in view.links),
             "artifact_extractor": extractor, "join_inferred": join_inferred,
             "inferred_joined": inferred_joined, "grouping": grouping}
    if grouping == "implemented_decision":
        stats["windows"] = windows_meta
    return Structure(approaches, clusters, stats)


def select_current(approaches: Sequence[Approach]) -> dict[str, Any]:
    """The T2.1 frozen prefix-only rule over T2.2 approaches; never inspects outcomes.

    Eligible: >=2 members, >=2 linked actions (attempts), valid evidence-preserving removal. Rank: latest
    activity (descending), then earliest first activity, then unit id. Records zero/one/multiple.
    """
    eligible = [a for a in approaches if len(a.members) >= 2 and a.attempts >= 2 and a.valid_removal]
    ranked = sorted(eligible, key=lambda a: (-a.last_activity_seq, a.first_seq, a.unit_id))
    reasons: dict[str, int] = defaultdict(int)
    for a in approaches:
        if len(a.members) < 2:
            reasons["single_member"] += 1
            continue
        if a.attempts < 2:
            reasons["fewer_than_two_attempts"] += 1
        for r in a.invalid_reasons:
            reasons[r] += 1
    return {"selector_version": SELECTOR22_VERSION, "rule_version": T21_SELECTOR_VERSION,
            "n_approaches": len(approaches), "n_multi_member": sum(len(a.members) >= 2 for a in approaches),
            "n_eligible": len(eligible), "n_cross_checkpoint_eligible": sum(a.checkpoint_span >= 2 for a in eligible),
            "case": "zero" if not eligible else "one" if len(eligible) == 1 else "multiple",
            "selected": ranked[0].unit_id if ranked else None, "ineligible_reasons": dict(reasons)}
