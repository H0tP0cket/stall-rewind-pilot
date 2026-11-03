"""T2.2 classifier passes: main role and descriptive labels (A), explicit relations (B), paraphrase check.

Pass A is ``label21.pass_a`` with registry ``t2.2-questions/v1``: one request
per source event, field and kind with at most ``BATCH`` items, the same retry of
each item with minimal context after a firewall block (method suffix
``:minimal-context``) and the same Jev receipts. Actions, harness messages and
tool outputs keep their deterministic roles.

Pass B asks whether a later worker sentence's own wording explicitly connects
it to an earlier item. ``pair_candidates22`` is a logged, deterministic cost
bound, not evidence of a relation:

* later items: worker prose, not generic, not a task copy, main role in
  APPROACH_MEMBER_ROLES | EVIDENCE_ROLES | {retrospective_takeaway}. No cue-word gate.
* earlier pool: same session, ``seq < later.seq`` and ``available_seq <=
  later.available_seq``; worker prose with an approach-member role (not
  generic, not a task copy) and tool-output items.
* candidates: pool items sharing >= 1 named artifact (``artifacts22``; a tool
  output also carries its producing command's names), ranked by number shared
  then recency, capped at MAX_ARTIFACT_CANDIDATES; plus the
  RECENT_ALWAYS most recent approach-member prose items; plus the output items
  of the most recent response with actions before the later item (the last
  result the agent saw; all its parallel tool calls).

Every candidate that shares a named artifact also gives a ``same_thing_named``
link: data only, it never joins items. A relation answer at or above
RELATION_THRESHOLD gives an ``explicit_wording`` link, with ``reads_output_*``
stored only to a tool output and every other relation only to worker prose.
Every asked pair is kept with its full distribution and call receipt id.

Pass C is the ratified contingency for sparse explicit links. It asks whether
two theory/plan sentences (main role in SAME_APPROACH_ROLES on both sides) are
part of the same approach even though neither mentions the other. Retrieval is
pass B's shared-name ranking plus the most recent, without tool outputs or last
results; pairs that already have an explicit-wording link are skipped. Answers
at or above SAME_APPROACH_THRESHOLD give ``inferred_same_approach`` links, a
separate tier, so every result can be recomputed on explicit links only.

Pass B and pass C only retrieve items from earlier events, so a sentence is
never asked about the sentences just before it in its own response. The
same-response passes (``pass_b22_same_response``, ``pass_c22_same_response``)
ask the same questions, in separate requests, about the up to
SAME_RESPONSE_PREVIOUS nearest preceding usable sentences of the same event and
field, skipping pairs already asked or explicitly linked.

``paraphrase_check`` asks, for each kept sentence of a removal view, whether it
says the same thing as any removed item (removed texts as a numbered list,
chunked when long).

Prefix safety: pass-A inputs are the item's own event (tool output also gets
its producing command); pass-B pairs are same-session items available at the
later item's cut. No input includes grades, detector output, annotations,
hidden tests, reference material or future events.
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .artifacts22 import ARTIFACTS22_VERSION, artifacts_for
from .jev import BudgetExceeded, JevClient
from .label21 import BATCH, CONTEXT_CHARS, _ask_with_fallback, _choice_label, _context
from .questions22 import QUESTIONS
from .schema import UNKNOWN, Label, Link
from .vocab22 import (APPROACH_MEMBER_ROLES, EVIDENCE_ROLES, LINK_TIER_CANDIDATE, LINK_TIER_EXPLICIT,
                      LINK_TIER_INFERRED, MAX_ARTIFACT_CANDIDATES, OUTPUT_RELATIONS, RECENT_ALWAYS, RELATION_THRESHOLD,
                      SAME_APPROACH_RELATION, SAME_APPROACH_ROLES, SAME_APPROACH_THRESHOLD)

LABEL22_VERSION = "t2.2-label/v1"
PROSE = frozenset({"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"})
OUTPUT = "observation"
LATER_ROLES = APPROACH_MEMBER_ROLES | EVIDENCE_ROLES | {"retrospective_takeaway"}
NULL_RELATIONS = {"no_explicit_relation", "ambiguous", UNKNOWN}
CANDIDATE_RELATION = "shares_named_artifact"
SOURCES = ("shared_artifact", "recent_member", "last_result")
SAME_RESPONSE_SOURCE = "same_response_previous"
SAME_RESPONSE_PREVIOUS = 3  # nearest preceding sentences of the same response asked about (same-response passes)
PARAPHRASE_CHUNK_CHARS = 6000
PASS_A_FIELDS = ("role", "certainty", "specificity", "restatement", "output_kind")

_recency = lambda i: (i.available_seq, i.seq, i.order)


def _run_jobs(client: JevClient, jobs: list[tuple[Any, Callable, list, str]], *, workers: int, repeat: int,
              stats: dict, on_error: Callable[[str], None], label: str) -> list[tuple[Any, list, Any, bool]]:
    """Ask every (key, build, batch, purpose) job with the label21 fallback; results sorted by job key."""
    lock = threading.Lock()
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ask_with_fallback, client, build, batch, purpose, repeat, stats, lock): key
                   for key, build, batch, purpose in jobs}
        for future in as_completed(futures):
            try:
                out.extend((futures[future], sub, answer, minimal) for sub, answer, minimal in future.result())
            except BudgetExceeded:
                raise
            except Exception as exc:
                stats["failed"] += 1
                on_error(f"{label} failure: {exc}")
    for _, _, answer, minimal in out:
        stats["requests"] += 1
        stats["cache_hits"] += int(answer.cached)
        stats["minimal_context_batches"] += int(minimal)
    return sorted(out, key=lambda r: r[0])


def _new_stats(**extra) -> dict[str, Any]:
    return {"requests": 0, "cache_hits": 0, "failed": 0, "blocked_items": 0, "minimal_context_batches": 0, **extra}


# ---- pass A ----

def pass_a22(items: Sequence, decoded: Mapping[tuple[int, str], str], commands: Mapping[str, str], client: JevClient, *,
             workers: int = 12, repeat: int = 0, on_error: Callable[[str], None] = print,
             label_prefix: str = "") -> dict[str, Any]:
    """Label items in place (``item.labels[label_prefix + name]``); returns request statistics."""
    base_method = f"jev:{client.model}:{LABEL22_VERSION}"
    groups: dict[tuple[int, str, str], list] = {}
    for item in items:
        if item.source_kind == "worker_action":
            item.labels[label_prefix + "role"] = Label("action", None, "deterministic:source_kind")
            continue
        if item.source_kind == "format_error":
            item.labels[label_prefix + "role"] = Label("harness_message", None, "deterministic:source_kind")
            continue
        if item.source_kind == OUTPUT:
            item.labels[label_prefix + "role"] = Label("tool_output", None, "deterministic:source_kind")
        kind = "obs" if item.source_kind == OUTPUT else "prose"
        groups.setdefault((item.seq, item.span.field, kind), []).append(item)

    def build(kind):
        names = ("output_kind", "specificity") if kind == "obs" else ("role", "certainty", "specificity", "restatement")

        def _build(minimal: bool, batch: list):
            state: dict[str, Any] = {"source_text_excerpt": "" if minimal else _context(decoded, batch),
                                     "items": {f"I{n}": it.text for n, it in enumerate(batch)}}
            if kind == "obs" and not minimal:
                state = {"command": commands.get(batch[0].tool_call_id or "", "unknown"), **state}
            questions = {f"I{n}.{name}": {"type": "choice", "criteria": QUESTIONS[name]["options"],
                                          "instructions": QUESTIONS[name]["instruction"].format(ref=f"items.I{n}")}
                         for n in range(len(batch)) for name in names}
            return state, questions
        return _build

    jobs = [((seq, f, kind, k), build(kind), g[k:k + BATCH], f"t2.2-passA:{kind}")
            for (seq, f, kind), g in groups.items() for k in range(0, len(g), BATCH)]
    stats = _new_stats()
    for _, batch, answer, minimal in _run_jobs(client, jobs, workers=workers, repeat=repeat, stats=stats,
                                               on_error=on_error, label="passA22"):
        method = base_method + (":minimal-context" if minimal else "")
        for n, it in enumerate(batch):
            for name in PASS_A_FIELDS:
                a = answer.answers.get(f"I{n}.{name}")
                if a is not None:
                    it.labels[label_prefix + name] = _choice_label(a, QUESTIONS[name]["threshold"], method)
    return stats


# ---- pass B retrieval ----

@dataclass
class Candidate22:
    earlier: Any
    sources: list[str] = field(default_factory=list)
    shared: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {"earlier_id": self.earlier.item_id, "earlier_kind": _kind(self.earlier), "sources": list(self.sources),
                "shared_artifacts": list(self.shared)}


def _kind(item) -> str:
    return "tool_output" if item.source_kind == OUTPUT else "worker_prose"


def pair_candidates22(items: Sequence, artifacts_by_id: Mapping[str, frozenset[str]], *,
                      label_prefix: str = "") -> tuple[list[tuple[Any, list[Candidate22]]], list[Link], dict[str, Any]]:
    """Deterministic, prefix-safe (later, [Candidate22]) pairs, ``same_thing_named`` candidate links and stats."""
    out, candidate_links, stats = _retrieve(items, artifacts_by_id, later_roles=LATER_ROLES,
                                            member_roles=APPROACH_MEMBER_ROLES, with_outputs=True,
                                            label_prefix=label_prefix)
    stats["retrieval_rule"] = (f"no cue gate; same session, earlier seq < later seq and available at the later cut; "
                               f"up to {MAX_ARTIFACT_CANDIDATES} approach-member prose or tool-output items sharing a "
                               f"named artifact (most shared, then most recent), plus the {RECENT_ALWAYS} most recent "
                               f"approach-member prose items, plus the output items of the most recent response with "
                               f"actions before the later item")
    return out, candidate_links, stats


def _retrieve(items: Sequence, artifacts_by_id: Mapping[str, frozenset[str]], *, later_roles: set[str],
              member_roles: set[str], with_outputs: bool,
              label_prefix: str = "") -> tuple[list[tuple[Any, list[Candidate22]]], list[Link], dict[str, Any]]:
    lab = lambda i, n: i.label(label_prefix + n)
    arts = lambda i: artifacts_by_id.get(i.item_id, frozenset())
    usable = lambda i: i.source_kind in PROSE and lab(i, "specificity") != "generic" and not i.task_copy
    later_items = sorted((i for i in items if usable(i) and lab(i, "role") in later_roles), key=_recency)
    members = [i for i in items if usable(i) and lab(i, "role") in member_roles]
    outputs = [i for i in items if i.source_kind == OUTPUT] if with_outputs else []
    by_session: dict[Any, dict[str, Any]] = defaultdict(lambda: {"members": [], "member_keys": [], "actions": [],
                                                                 "action_seqs": [], "index": defaultdict(list)})
    for i in sorted(members, key=_recency):
        by_session[i.session_id]["members"].append(i)
        by_session[i.session_id]["member_keys"].append(_recency(i))
    for i in sorted(members + outputs, key=_recency):
        for a in arts(i):
            by_session[i.session_id]["index"][a].append(i)
    for i in sorted((i for i in items if with_outputs and i.source_kind == "worker_action"),
                    key=lambda i: (i.seq, i.order)):
        by_session[i.session_id]["actions"].append(i)
        by_session[i.session_id]["action_seqs"].append(i.seq)
    outputs_by_call: dict[str, list] = defaultdict(list)
    for o in sorted(outputs, key=_recency):
        if o.tool_call_id:
            outputs_by_call[o.tool_call_id].append(o)

    counts = {s: 0 for s in SOURCES}
    stats: dict[str, Any] = {"later_items_eligible": len(later_items), "later_items_considered": 0,
                             "later_items_without_candidates": 0, "later_items_with_artifacts": 0, "pairs": 0,
                             "candidates_by_source": counts, "candidates_from_several_sources": 0,
                             "pairs_by_earlier_kind": {"worker_prose": 0, "tool_output": 0}, "candidate_links": 0,
                             "last_action_without_output_items": 0}
    out: list[tuple[Any, list[Candidate22]]] = []
    candidate_links: list[Link] = []
    for later in later_items:
        session = by_session.get(later.session_id)
        before = lambda e: e.seq < later.seq and e.available_seq <= later.available_seq
        chosen: dict[str, Candidate22] = {}
        la = arts(later)

        def add(e, source: str) -> None:
            chosen.setdefault(e.item_id, Candidate22(e, [], tuple(sorted(la & arts(e))))).sources.append(source)

        stats["later_items_with_artifacts"] += bool(la)
        if session is not None:
            shared: dict[str, tuple[Any, set]] = {}
            for a in la:
                for e in session["index"].get(a, ()):
                    if before(e):
                        shared.setdefault(e.item_id, (e, set()))[1].add(a)
            ranked = sorted(shared.values(), key=lambda es: (-len(es[1]), tuple(-x for x in _recency(es[0]))))
            for e, _ in ranked[:MAX_ARTIFACT_CANDIDATES]:
                add(e, "shared_artifact")
            recent, k = [], bisect.bisect_right(session["member_keys"], _recency(later))
            while k and len(recent) < RECENT_ALWAYS:
                k -= 1
                if before(session["members"][k]):
                    recent.append(session["members"][k])
            for e in recent:
                add(e, "recent_member")
            seqs, k = session["action_seqs"], bisect.bisect_left(session["action_seqs"], later.seq)
            if k:  # every action of the latest response before the later item (parallel tool calls included)
                last = session["actions"][bisect.bisect_left(seqs, seqs[k - 1]):k]
                results = [o for a in last if a.tool_call_id for o in outputs_by_call.get(a.tool_call_id, ())
                           if before(o)]
                stats["last_action_without_output_items"] += not results
                for o in results:
                    add(o, "last_result")
        if not chosen:
            stats["later_items_without_candidates"] += 1
            continue
        cands = list(chosen.values())
        out.append((later, cands))
        for c in cands:
            for s in c.sources:
                counts[s] += 1
            stats["candidates_from_several_sources"] += len(c.sources) > 1
            stats["pairs_by_earlier_kind"][_kind(c.earlier)] += 1
            if c.shared:
                candidate_links.append(Link(later.item_id, c.earlier.item_id, CANDIDATE_RELATION, LINK_TIER_CANDIDATE,
                                            None, later.available_seq, f"deterministic:{ARTIFACTS22_VERSION}"))
    stats["later_items_considered"] = len(out)
    stats["pairs"] = sum(len(c) for _, c in out)
    stats["candidate_links"] = len(candidate_links)
    return out, candidate_links, stats


# ---- pass B ----

@dataclass
class RelationPass:
    links: list[Link]              # pass B: explicit_wording links; pass C: inferred_same_approach links
    candidate_links: list[Link]    # same_thing_named links (data only, never join)
    pairs: list[dict[str, Any]]    # every asked pair: full distribution, method and call receipt id
    stats: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {"version": LABEL22_VERSION, "links": [l.as_record() for l in self.links],
                "candidate_links": [l.as_record() for l in self.candidate_links], "pairs": self.pairs,
                "stats": self.stats}

    def asked_pairs(self) -> set[tuple[str, str]]:
        """(later_id, earlier_id) of every answered pair."""
        return {(r["later_id"], r["earlier_id"]) for r in self.pairs}


def _earlier_state(item, commands: Mapping[str, str], minimal: bool) -> dict[str, Any]:
    if item.source_kind != OUTPUT:
        return {"kind": "worker statement", "text": item.text}
    command = {} if minimal else {"command": commands.get(item.tool_call_id or "", "unknown")}
    return {"kind": "tool output", **command, "text": item.text[:CONTEXT_CHARS]}


def _ask_pairs(pairs: list[tuple[Any, list[Candidate22]]], decoded: Mapping[tuple[int, str], str], client: JevClient,
               *, question: str, key: str, purpose: str, earlier_state: Callable[[Any, bool], Any], workers: int,
               repeat: int, on_error: Callable[[str], None], stats: dict) -> list[tuple[Any, Candidate22, Label, dict]]:
    """Ask ``question`` about every (later, candidate) pair, one request per later item (at most BATCH pairs).

    Returns (later, candidate, label, pair record) for every answered pair; the record keeps the full
    distribution, method and call receipt id.
    """
    base_method = f"jev:{client.model}:{LABEL22_VERSION}"
    q = QUESTIONS[question]

    def build(later):
        def _build(minimal: bool, batch: list):
            state = {"later": later.text, "later_context": "" if minimal else _context(decoded, [later]),
                     "earlier": {f"E{n}": earlier_state(c.earlier, minimal) for n, c in enumerate(batch)}}
            questions = {f"E{n}.{key}": {"type": "choice", "instructions": q["instruction"].format(ref=f"E{n}"),
                                         "criteria": q["options"]} for n in range(len(batch))}
            return state, questions
        return _build

    jobs = [((n, k), build(later), cands[k:k + BATCH], purpose)
            for n, (later, cands) in enumerate(pairs) for k in range(0, len(cands), BATCH)]
    out = []
    for (n, _), batch, answer, minimal in _run_jobs(client, jobs, workers=workers, repeat=repeat, stats=stats,
                                                    on_error=on_error, label=purpose):
        later = pairs[n][0]
        method = base_method + (":minimal-context" if minimal else "")
        for m, c in enumerate(batch):
            a = answer.answers.get(f"E{m}.{key}")
            if a is None:
                stats["missing_answers"] += 1
                continue
            stats["pairs_asked"] += 1
            lab = _choice_label(a, q["threshold"], method)
            probs = dict(lab.probabilities or {})
            top = max(probs.items(), key=lambda kv: kv[1]) if probs else (None, None)
            record = {"later_id": later.item_id, "evidence_seq": later.available_seq, **c.as_record(),
                      "value": lab.value, "p": probs.get(lab.value) if lab.value != UNKNOWN else None,
                      "top": top[0], "top_p": top[1], "probabilities": probs,
                      "confidence": lab.confidence, "method": method, "call_record_id": answer.call_record_id,
                      "cached": answer.cached, "link": False}
            out.append((later, c, lab, record))
    return out


def pass_b22(items: Sequence, decoded: Mapping[tuple[int, str], str], client: JevClient, *,
             artifacts_by_id: Mapping[str, frozenset[str]] | None = None, workers: int = 12, repeat: int = 0,
             on_error: Callable[[str], None] = print, label_prefix: str = "") -> RelationPass:
    artifacts_by_id = artifacts_for(items) if artifacts_by_id is None else artifacts_by_id
    pairs, candidate_links, retrieval = pair_candidates22(items, artifacts_by_id, label_prefix=label_prefix)
    return _relation_pass(items, pairs, decoded, client, purpose="t2.2-passB:relation", retrieval=retrieval,
                          candidate_links=candidate_links, workers=workers, repeat=repeat, on_error=on_error)


def _relation_pass(items: Sequence, pairs: list[tuple[Any, list[Candidate22]]], decoded: Mapping[tuple[int, str], str],
                   client: JevClient, *, purpose: str, retrieval: dict[str, Any], candidate_links: list[Link],
                   workers: int, repeat: int, on_error: Callable[[str], None]) -> RelationPass:
    """Relation question over ``pairs``; explicit-wording links at RELATION_THRESHOLD (kind-compatible only)."""
    commands = {i.tool_call_id: i.text for i in items if i.source_kind == "worker_action" and i.tool_call_id}
    stats = _new_stats(pairs_asked=0, missing_answers=0, incompatible_relations=0, **retrieval)
    links: list[Link] = []
    records: list[dict[str, Any]] = []
    for later, c, lab, record in _ask_pairs(pairs, decoded, client, question="relation", key="rel",
                                            purpose=purpose, workers=workers, repeat=repeat,
                                            earlier_state=lambda e, minimal: _earlier_state(e, commands, minimal),
                                            on_error=on_error, stats=stats):
        if lab.value not in NULL_RELATIONS:
            if (lab.value in OUTPUT_RELATIONS) == (c.earlier.source_kind == OUTPUT):
                links.append(Link(later.item_id, c.earlier.item_id, lab.value, LINK_TIER_EXPLICIT,
                                  record["p"], later.available_seq, record["method"]))
                record["link"] = True
            else:
                record["link_rejected"] = "relation_incompatible_with_earlier_kind"
                stats["incompatible_relations"] += 1
        records.append(record)
    stats["explicit_links"] = len(links)
    stats["threshold"] = RELATION_THRESHOLD
    return RelationPass(links, candidate_links, records, stats)


# ---- pass C (contingency tier): inferred same-approach links ----

def same_approach_candidates22(items: Sequence, artifacts_by_id: Mapping[str, frozenset[str]], *,
                               explicit_links: Sequence[Link] = (),
                               label_prefix: str = "") -> tuple[list[tuple[Any, list[Candidate22]]], dict[str, Any]]:
    """Pass-B retrieval restricted to theory/plan sentences (SAME_APPROACH_ROLES on both sides).

    Shared-name ranking capped at MAX_ARTIFACT_CANDIDATES plus the RECENT_ALWAYS most recent; no tool outputs
    and no last-result candidates. Pairs that already have an explicit-wording link (either direction) are
    skipped after retrieval, not replaced. Shared names give no candidate links here (pass B records those).
    """
    out, _, stats = _retrieve(items, artifacts_by_id, later_roles=SAME_APPROACH_ROLES,
                              member_roles=SAME_APPROACH_ROLES, with_outputs=False, label_prefix=label_prefix)
    linked = {(l.source_id, l.target_id) for l in explicit_links if l.tier == LINK_TIER_EXPLICIT}
    linked |= {(b, a) for a, b in linked}
    kept, skipped, counts = [], 0, {s: 0 for s in SOURCES[:2]}
    for later, cands in out:
        ask = [c for c in cands if (later.item_id, c.earlier.item_id) not in linked]
        skipped += len(cands) - len(ask)
        for c in ask:
            for src in c.sources:
                counts[src] += 1
        if ask:
            kept.append((later, ask))
    stats = {"later_items_eligible": stats["later_items_eligible"], "later_items_considered": len(kept),
             "later_items_with_artifacts": stats["later_items_with_artifacts"],
             "pairs": sum(len(c) for _, c in kept), "candidates_by_source": counts,
             "skipped_explicit_pairs": skipped,
             "retrieval_rule": f"theory/plan sentences only ({', '.join(sorted(SAME_APPROACH_ROLES))}) on both sides; "
                               f"same session, earlier seq < later seq and available at the later cut; up to "
                               f"{MAX_ARTIFACT_CANDIDATES} sharing a named artifact (most shared, then most recent) plus "
                               f"the {RECENT_ALWAYS} most recent; pairs with an explicit-wording link skipped"}
    return kept, stats


def pass_c22(items: Sequence, decoded: Mapping[tuple[int, str], str], client: JevClient, *,
             artifacts_by_id: Mapping[str, frozenset[str]] | None = None, explicit_links: Sequence[Link] = (),
             workers: int = 12, repeat: int = 0, on_error: Callable[[str], None] = print,
             label_prefix: str = "") -> RelationPass:
    """Ask the same-approach question; answers at or above SAME_APPROACH_THRESHOLD give LINK_TIER_INFERRED links."""
    artifacts_by_id = artifacts_for(items) if artifacts_by_id is None else artifacts_by_id
    pairs, retrieval = same_approach_candidates22(items, artifacts_by_id, explicit_links=explicit_links,
                                                  label_prefix=label_prefix)
    return _same_approach_pass(pairs, decoded, client, purpose="t2.2-passC:same_approach", retrieval=retrieval,
                               workers=workers, repeat=repeat, on_error=on_error)


def _same_approach_pass(pairs: list[tuple[Any, list[Candidate22]]], decoded: Mapping[tuple[int, str], str],
                        client: JevClient, *, purpose: str, retrieval: dict[str, Any], workers: int, repeat: int,
                        on_error: Callable[[str], None]) -> RelationPass:
    stats = _new_stats(pairs_asked=0, missing_answers=0, **retrieval)
    links: list[Link] = []
    records: list[dict[str, Any]] = []
    for later, c, lab, record in _ask_pairs(pairs, decoded, client, question="same_approach", key="same",
                                            purpose=purpose, workers=workers, repeat=repeat,
                                            earlier_state=lambda e, minimal: e.text, on_error=on_error, stats=stats):
        if lab.value == SAME_APPROACH_RELATION:
            links.append(Link(later.item_id, c.earlier.item_id, SAME_APPROACH_RELATION, LINK_TIER_INFERRED,
                              record["p"], later.available_seq, record["method"]))
            record["link"] = True
        records.append(record)
    stats["inferred_links"] = len(links)
    stats["threshold"] = SAME_APPROACH_THRESHOLD
    return RelationPass(links, [], records, stats)


# ---- same-response passes: earlier sentences of the later sentence's own response ----

def same_response_candidates22(items: Sequence, *, later_roles: set[str], earlier_roles: set[str] | None = None,
                               skip_pairs: set[tuple[str, str]] = frozenset(),
                               label_prefix: str = "") -> tuple[list[tuple[Any, list[Candidate22]]], dict[str, Any]]:
    """(later, [Candidate22]) with the nearest SAME_RESPONSE_PREVIOUS preceding prose items of the same source event.

    Later items are usable prose (not generic, not a task copy) with a main role in ``later_roles``. Candidates are
    usable prose items of the same run, event (``seq``) and field with a lower ``order`` (and, when given, a main
    role in ``earlier_roles``), nearest first by order, returned in source order. Pairs in ``skip_pairs`` are
    dropped after selection, not replaced. Same event and lower order make these prefix-safe by construction.
    """
    lab = lambda i, n: i.label(label_prefix + n)
    usable = lambda i: i.source_kind in PROSE and lab(i, "specificity") != "generic" and not i.task_copy
    events: dict[tuple, list] = defaultdict(list)
    for i in items:
        if usable(i):
            events[(i.run_id, i.session_id, i.seq, i.span.field)].append(i)
    out: list[tuple[Any, list[Candidate22]]] = []
    stats: dict[str, Any] = {"later_items_eligible": 0, "later_items_considered": 0, "pairs": 0,
                             "candidates_by_source": {SAME_RESPONSE_SOURCE: 0}, "skipped_pairs": 0,
                             "distance_in_order": {}}
    distance: dict[int, int] = defaultdict(int)
    for group in events.values():
        group.sort(key=lambda i: i.order)
        for n, later in enumerate(group):
            if lab(later, "role") not in later_roles:
                continue
            stats["later_items_eligible"] += 1
            prev = [e for e in group[:n] if (earlier_roles is None or lab(e, "role") in earlier_roles)
                    and e.available_seq <= later.available_seq][-SAME_RESPONSE_PREVIOUS:]
            ask = [e for e in prev if (later.item_id, e.item_id) not in skip_pairs]
            stats["skipped_pairs"] += len(prev) - len(ask)
            if not ask:
                continue
            for e in ask:
                distance[later.order - e.order] += 1
            out.append((later, [Candidate22(e, [SAME_RESPONSE_SOURCE]) for e in ask]))
    out.sort(key=lambda lc: _recency(lc[0]))
    stats["later_items_considered"] = len(out)
    stats["pairs"] = stats["candidates_by_source"][SAME_RESPONSE_SOURCE] = sum(len(c) for _, c in out)
    stats["distance_in_order"] = {str(k): v for k, v in sorted(distance.items())}  # record keys must be strings
    stats["retrieval_rule"] = (f"same run, event and field; up to {SAME_RESPONSE_PREVIOUS} nearest preceding "
                               f"usable prose items (lower order); pairs already asked or linked skipped")
    return out, stats


def pass_b22_same_response(items: Sequence, decoded: Mapping[tuple[int, str], str], client: JevClient, *,
                           asked_pairs: set[tuple[str, str]] = frozenset(), workers: int = 12, repeat: int = 0,
                           on_error: Callable[[str], None] = print, label_prefix: str = "") -> RelationPass:
    """Relation question for same-response pairs that pass B cannot retrieve (it requires an earlier event).

    Later items follow pass B's rule; earlier items are any-role usable prose from the same response. Pairs in
    ``asked_pairs`` (pass B's ``RelationPass.asked_pairs()``) are skipped. Separate requests and purpose, so pass B's
    requests and receipts are unchanged; same question, threshold and tier as pass B.
    """
    pairs, retrieval = same_response_candidates22(items, later_roles=LATER_ROLES, skip_pairs=set(asked_pairs),
                                                  label_prefix=label_prefix)
    return _relation_pass(items, pairs, decoded, client, purpose="t2.2-passB:relation-same-response",
                          retrieval=retrieval, candidate_links=[], workers=workers, repeat=repeat, on_error=on_error)


def pass_c22_same_response(items: Sequence, decoded: Mapping[tuple[int, str], str], client: JevClient, *,
                           explicit_links: Sequence[Link] = (), asked_pairs: set[tuple[str, str]] = frozenset(),
                           workers: int = 12, repeat: int = 0, on_error: Callable[[str], None] = print,
                           label_prefix: str = "") -> RelationPass:
    """Same-approach question for same-response pairs, both sentences in SAME_APPROACH_ROLES.

    Skips pairs with an explicit-wording link in either direction (pass B and its same-response pass) and pairs in
    ``asked_pairs`` (pass C's ``RelationPass.asked_pairs()``). Links are LINK_TIER_INFERRED, as in pass C.
    """
    linked = {(l.source_id, l.target_id) for l in explicit_links if l.tier == LINK_TIER_EXPLICIT}
    skip = set(asked_pairs) | linked | {(b, a) for a, b in linked}
    pairs, retrieval = same_response_candidates22(items, later_roles=SAME_APPROACH_ROLES,
                                                  earlier_roles=SAME_APPROACH_ROLES, skip_pairs=skip,
                                                  label_prefix=label_prefix)
    return _same_approach_pass(pairs, decoded, client, purpose="t2.2-passC:same_approach-same-response",
                               retrieval=retrieval, workers=workers, repeat=repeat, on_error=on_error)


# ---- paraphrase check (removal views) ----

def _chunks(removed: Sequence, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    size = 0
    for n, item in enumerate(removed, start=1):
        if not out or size + len(item.text) > limit:
            out.append({})
            size = 0
        out[-1][str(n)] = item.text
        size += len(item.text)
    return out


def paraphrase_check(client: JevClient, kept_items: Sequence, removed_items: Sequence, *, workers: int = 12,
                     repeat: int = 0, on_error: Callable[[str], None] = print,
                     chunk_chars: int = PARAPHRASE_CHUNK_CHARS, stats: dict | None = None) -> dict[str, dict[str, Any]]:
    """For each kept item: does it say the same thing as any removed item?

    Returns ``{kept_id: {"p", "value", ...}}``. ``p`` is the highest probability
    of ``restates_removed`` over the removed-list chunks (None when no chunk was
    answered). ``value`` is ``restates_removed`` if any chunk reaches the
    threshold, ``adds_different_content`` only if every chunk was answered so at
    or above it, otherwise ``ambiguous`` or ``unknown``. Empty when nothing was removed.
    """
    stats = {} if stats is None else stats
    stats.update(_new_stats())
    if not kept_items or not removed_items:
        return {}
    q = QUESTIONS["paraphrase"]
    base_method = f"jev:{client.model}:{LABEL22_VERSION}"
    chunks = _chunks(removed_items, chunk_chars)

    def build(removed: dict[str, str]):
        def _build(minimal: bool, batch: list):  # after a block: one kept item per request, same removed list
            state = {"removed": removed, "kept": {f"K{n}": it.text for n, it in enumerate(batch)}}
            questions = {f"K{n}.paraphrase": {"type": "choice", "criteria": q["options"],
                                              "instructions": q["instruction"].format(ref=f"kept.K{n}")}
                         for n in range(len(batch))}
            return state, questions
        return _build

    jobs = [((c, k), build(removed), list(kept_items[k:k + BATCH]), "t2.2-paraphrase")
            for c, removed in enumerate(chunks) for k in range(0, len(kept_items), BATCH)]
    answers: dict[str, dict[int, tuple[Label, str]]] = defaultdict(dict)
    for (c, _), batch, answer, minimal in _run_jobs(client, jobs, workers=workers, repeat=repeat, stats=stats,
                                                    on_error=on_error, label="paraphrase22"):
        method = base_method + (":minimal-context" if minimal else "")
        for n, it in enumerate(batch):
            a = answer.answers.get(f"K{n}.paraphrase")
            if a is not None:
                answers[it.item_id][c] = (_choice_label(a, q["threshold"], method), answer.call_record_id)
    result: dict[str, dict[str, Any]] = {}
    for it in kept_items:
        got = answers.get(it.item_id, {})
        p_by_chunk = {c: float((lab.probabilities or {}).get("restates_removed", 0.0)) for c, (lab, _) in got.items()}
        values = [lab.value for lab, _ in got.values()]
        best = max(p_by_chunk, key=p_by_chunk.get) if p_by_chunk else None
        if "restates_removed" in values:
            value = "restates_removed"
        elif values and len(got) == len(chunks) and all(v == "adds_different_content" for v in values):
            value = "adds_different_content"
        elif "ambiguous" in values:
            value = "ambiguous"
        else:
            value = UNKNOWN
        result[it.item_id] = {
            "p": p_by_chunk[best] if best is not None else None, "value": value,
            "probabilities": dict(got[best][0].probabilities or {}) if best is not None else None,
            "matched_chunk": best if value == "restates_removed" else None,
            "chunks": len(chunks), "chunks_answered": len(got),
            "methods": sorted({lab.method for lab, _ in got.values()}),
            "call_record_ids": [got[c][1] for c in sorted(got)]}
    return result
