"""Ledger v2: the ratified rebuild of T (user decisions 2026-09-27; memory final-ratification-2026-09-26).

What changes from T2.2 (everything else is reused unchanged):
* labels  - one 13-way role becomes independent axes asked of every agent sentence: epistemic (observation /
            inference / hypothesis / question / none), action (proposal / decision / next_step / retrospective /
            none), self-assessment (success_claim / failure_claim / correction / none) and content (concrete /
            task_restatement / discourse_only). A sentence can carry several axes at once.
* outputs - tool-output labels are deterministic, never a model: state-changing vs read-only (the run's recorded
            file mutations), pass / fail / unclear (exit code and test summaries), display / listing (command).
* links   - the stretch linker (``linkllm22``): an LLM reads each response with its recent context and every
            earlier plan/hypothesis sentence, and lists explicit back-references. The backup "same idea" tier is gone.
* units   - decision-bounded approaches (``units22.build_structure``): roots are plans, hypotheses, or the sentence
            of a response whose action changed state; statuses are agent-claimed only.
* current - the current approach is the one owning most of the last X state-changing actions (X calibrated).
* packets - time order; budgets full / 6k / 3k / 1.5k.
* views   - the user's starting Ledger arms, matched controls, collateral loss, rollback point, removed files.
* claims  - a success claim that names a test is checked against that test's own result at that moment.

Everything is prefix-only: labels read an item's own response, links point backwards, units and views are rebuilt
at each checkpoint from what existed by then. Offline incremental replay therefore reproduces what a live build
would have recorded at every checkpoint (up to model sampling), and every model call is cached.
"""

from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from .artifacts22 import artifacts_for
from .filters22 import FilterSpec, apply_filter
from .label21 import _choice_label
from .label22 import BATCH, _context, _run_jobs, _new_stats
from .linkllm22 import link_stretch
from .packet22 import build_packet22, item_tokens
from .schema import Label
from .units21 import at_checkpoint21
from .units22 import build_structure

LEDGER2_VERSION = "ledger/v2-2026-09-27"
PROSE = {"worker_text", "worker_reasoning", "provider_summary", "worker_code_comment"}
OUTPUT = "observation"
AXIS_THRESHOLD = 0.6

AXES = {
    "epistemic": {
        "instruction": "What kind of knowing does the sentence `{ref}` express, read in its context? Judge the wording, "
                       "not whether it is true.",
        "options": {"observation": "Reports something the agent saw or measured (a result, error, output, behaviour).",
                    "inference": "States a conclusion or fact the agent treats as settled (about code, environment, "
                                 "requirements).",
                    "hypothesis": "Offers a tentative explanation or guess about why something happens or what is wrong.",
                    "question": "Raises a question or unknown the agent has not answered.",
                    "none": "None of these."}},
    "action": {
        "instruction": "What does the sentence `{ref}` say about acting? Judge the wording only.",
        "options": {"proposal": "Considers an approach, fix or design without adopting it (maybe, could, one option).",
                    "decision": "Adopts or commits to an approach, fix or design it will now pursue.",
                    "next_step": "States a specific action it is about to take on a named target (inspect, edit, run).",
                    "retrospective": "Describes what it already did.",
                    "none": "None of these."}},
    "self_assessment": {
        "instruction": "Does the sentence `{ref}` assess the agent's own work? Judge the wording, not whether it is true.",
        "options": {"success_claim": "Claims its own work succeeded: a fix works, tests pass, done.",
                    "failure_claim": "Says its own work failed or is still broken.",
                    "correction": "Says an earlier idea, belief, plan or result of its own was wrong or is changed.",
                    "none": "No self-assessment."}},
    "content": {
        "instruction": "What does the sentence `{ref}` contribute?",
        "options": {"concrete": "Concrete content about this work (a value, file, function, test, behaviour, idea).",
                    "task_restatement": "Only repeats what the task or instructions say, adding nothing of its own.",
                    "discourse_only": "Process talk with no content ('let me look', 'now I will implement it')."}},
}

# ---- labels ----

def pass_axes(items: Sequence, decoded: Mapping[tuple[int, str], str], client, *, workers: int = 8,
              on_error: Callable[[str], None] = print) -> dict[str, Any]:
    """Jev asks the four axis questions of every agent sentence (tool outputs get deterministic labels only)."""
    groups: dict[tuple[int, str], list] = {}
    for item in items:
        if item.source_kind in PROSE:
            groups.setdefault((item.seq, item.span.field), []).append(item)

    def build(minimal: bool, batch: list):
        state = {"source_text_excerpt": "" if minimal else _context(decoded, batch),
                 "items": {f"I{n}": it.text for n, it in enumerate(batch)}}
        qs = {f"I{n}.{ax}": {"type": "choice", "criteria": spec["options"],
                             "instructions": spec["instruction"].format(ref=f"items.I{n}")}
              for n in range(len(batch)) for ax, spec in AXES.items()}
        return state, qs

    jobs = [((seq, f, k), build, g[k:k + BATCH], "ledger2-axes") for (seq, f), g in groups.items()
            for k in range(0, len(g), BATCH)]
    stats = _new_stats()
    method = f"jev:{client.model}:{LEDGER2_VERSION}"
    for _, batch, answer, minimal in _run_jobs(client, jobs, workers=workers, repeat=0, stats=stats,
                                               on_error=on_error, label="ledger2-axes"):
        for n, it in enumerate(batch):
            for ax in AXES:
                a = answer.answers.get(f"I{n}.{ax}")
                if a is not None:
                    it.labels[ax] = _choice_label(a, AXIS_THRESHOLD, method + (":minimal-context" if minimal else ""))
    return stats


_DISPLAY = re.compile(r"^\s*(?:cd [^;&]+(?:&&|;)\s*)?(?:cat|head|tail|sed -n|nl|less|more|bat)\b")
_LISTING = re.compile(r"^\s*(?:cd [^;&]+(?:&&|;)\s*)?(?:ls|find|tree|du|grep -[a-zA-Z]*l)\b")
_FAIL = re.compile(r"returncode:\s*[1-9]|Traceback|\bFAILED\b|\b\d+ failed\b|\bError\b|error:|No such file", re.I)
_PASS = re.compile(r"\b\d+ passed\b(?!.*\bfailed\b)|\bOK\b\s*$|returncode:\s*0\b|\ball tests passed\b", re.I)


def output_labels(item, command: str) -> dict[str, Label]:
    """Deterministic tool-output labels (no model): outcome and output kind."""
    text = item.text
    outcome = "fail" if _FAIL.search(text) else "pass" if _PASS.search(text) else "unclear"
    kind = ("displayed_file_or_code" if _DISPLAY.search(command or "")
            else "listing_or_progress" if _LISTING.search(command or "") else "execution_result")
    det = "deterministic:ledger2-output"
    return {"outcome": Label(outcome, {outcome: 1.0}, det), "output_kind": Label(kind, {kind: 1.0}, det)}


def state_changing_actions(source_store, run_id: str, items: Sequence) -> dict[str, list[str]]:
    """worker_action item -> files its response batch changed (recorded mutation events; empty = read-only)."""
    changed: dict[int, list[str]] = {}
    for e in source_store.iter_events(run_id):
        if e.kind != "mutation":
            continue
        p = e.payload or {}
        if p.get("relevant"):
            files = list(p.get("workspace_fidelity_paths_changed") or p.get("files") or []) + \
                    ["/" + str(x).lstrip("/") for x in p.get("runtime_paths_changed") or ()]
            changed[int(p.get("response_event_sequence") or -1)] = files or ["<unlisted>"]
    return {i.item_id: changed[i.seq] for i in items if i.source_kind == "worker_action" and i.seq in changed}


def submission_actions(source_store, run_id: str, items: Sequence) -> frozenset[str]:
    """worker_action items that submitted to the native grader (the last action before each native feedback event);
    a submission is a commitment even when it changes no file (audit fix, user 2026-09-27)."""
    acts = sorted((i for i in items if i.source_kind == "worker_action"), key=lambda i: i.seq)
    out = set()
    for e in source_store.iter_events(run_id):
        if e.kind == "native_submission_feedback":
            prev = [a for a in acts if a.seq < e.sequence]
            if prev:
                out.add(prev[-1].item_id)
    return frozenset(out)


def derived_role(item) -> str:
    """One role per sentence for the structure code only (grouping, packet exclusions); filters use the axes."""
    v = lambda ax: item.label(ax)
    if v("action") == "decision":
        return "commitment"
    if v("action") == "proposal":
        return "proposal"
    if v("self_assessment") == "success_claim":
        return "result_claim"
    if v("self_assessment") in ("correction", "failure_claim"):
        return "revision"
    if v("epistemic") == "hypothesis":
        return "explanation_or_hypothesis"
    if v("epistemic") == "question":
        return "unresolved_question"
    if v("action") == "next_step":
        return "intended_action"
    if v("epistemic") == "observation":
        return "observation_report"
    if v("epistemic") == "inference":
        return "factual_assertion"
    if v("content") == "discourse_only":
        return "narration"
    return "unknown"


def apply_derived(items: Sequence, commands: Mapping[str, str]) -> None:
    """Write the compatibility labels the reused T2.2 structure/packet code reads (role, specificity,
    restatement, output_kind). Decisions, proposals, hypotheses and questions are never marked generic, so short
    decisive sentences ("Let's defer") are no longer dropped as filler."""
    det = "derived:ledger2"
    for it in items:
        if it.source_kind in PROSE:
            role = derived_role(it)
            it.labels["role"] = Label(role, {role: 1.0}, det)
            content = it.labels.get("content")
            p = dict(content.probabilities or {}) if content else {}
            protected = role in ("commitment", "proposal", "explanation_or_hypothesis", "unresolved_question")
            gen = 0.0 if protected else float(p.get("discourse_only") or 0.0)
            it.labels["specificity"] = Label("generic" if gen >= 0.6 else "specific", {"generic": gen}, det)
            rest = float(p.get("task_restatement") or 0.0)
            it.labels["restatement"] = Label("restates_requirement" if rest >= 0.6 else "adds_agent_content",
                                             {"restates_requirement": rest}, det)
        elif it.source_kind == OUTPUT:
            it.labels.update(output_labels(it, commands.get(it.tool_call_id or "", "")))


# ---- links (stretch linker over a whole run) ----

ROOTISH = {"commitment", "proposal", "explanation_or_hypothesis"}


def link_run(llm, items: Sequence, commands: Mapping[str, str], *, context: int = 30, chunk: int = 40,
             max_roots: int = 40, workers: int = 8,
             on_error: Callable[[str], None] = print) -> tuple[list, dict[str, Any]]:
    """One call per chunk of a response's sentences; the model also sees the previous ``context`` items and the
    ``max_roots`` most recent earlier plan/hypothesis sentences, so long-range revisits can be linked."""
    pos = lambda i: (i.available_seq, i.seq, i.order)
    ordered = sorted((i for i in items if i.source_kind in PROSE | {OUTPUT}), key=pos)
    idx = {i.item_id: n for n, i in enumerate(ordered)}
    by_resp: dict[tuple, list] = defaultdict(list)
    for i in ordered:
        if i.source_kind in PROSE:
            by_resp[(i.session_id, i.seq)].append(i)
    jobs = []
    for (sess, _), sents in sorted(by_resp.items(), key=lambda kv: pos(kv[1][0])):
        for k in range(0, len(sents), chunk):
            part = sents[k:k + chunk]
            start = idx[part[0].item_id]
            prior = [i for i in ordered[:start] if i.session_id == sess]
            recent = prior[-context:]
            recent_ids = {i.item_id for i in recent}
            roots = [i for i in prior if i.item_id not in recent_ids and i.label("role") in ROOTISH][-max_roots:]
            jobs.append((part, roots + recent + part))

    def one(job):
        part, stretch = job
        try:
            return link_stretch(llm, stretch, commands, later_ids={i.item_id for i in part}, purpose="ledger2-links"), None
        except Exception as exc:  # recorded, never silently dropped
            return None, f"link chunk {part[0].item_id}: {type(exc).__name__}: {exc}"

    links, stats = [], Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r, err in pool.map(one, jobs):
            if err:
                stats["failed_chunks"] += 1
                on_error(err)
                continue
            links += r["links"]
            stats["chunks"] += 1
            stats["rejected"] += len(r["rejected"])
            stats["empty"] += r["empty_answer"]
    stats["links"] = len(links)
    return links, dict(stats)


# ---- per-checkpoint snapshot ----

def current_approach(approaches, state_changing: Mapping[str, list[str]], by_id, cut: int, x: int):
    """The approach owning most of the last ``x`` state-changing actions up to ``cut`` (ties: most recent).
    Fallback: the latest decision-rooted approach, then the most recently active approach."""
    acts = sorted((a for a in state_changing if by_id.get(a) is not None and by_id[a].seq <= cut),
                  key=lambda a: by_id[a].seq)[-x:]
    owner = {a: ap.unit_id for ap in approaches for a in ap.actions}
    counts = Counter(owner[a] for a in acts if a in owner)
    if counts:
        last = {u: max(by_id[a].seq for a in acts if owner.get(a) == u) for u in counts}
        return max(counts, key=lambda u: (counts[u], last[u])), "last_x_state_changing"
    decided = [ap for ap in approaches if any(by_id[r].label("role") == "commitment" for r in ap.roots if r in by_id)]
    if decided:
        return max(decided, key=lambda ap: ap.last_activity_seq).unit_id, "fallback_latest_decision"
    if approaches:
        return max(approaches, key=lambda ap: ap.last_activity_seq).unit_id, "fallback_latest_active"
    return None, "none"


ROOT_ROLES = {"proposal", "commitment", "hypothesis"}
OWNER_MODES = ("single", "share", "legacy")


def action_owners(approaches, view, state_changing: Mapping[str, list[str]], by_id, mode: str = "single"):
    """Owner weights {action_id: {unit_id: weight}} for the current-approach rule only (user 2026-09-27, fix 1-6).

    The structural ``same_turn`` links give every sentence of a response every action of that response, so an
    approach's ``actions`` (a union, still used for outputs/readings/removal) has ~4 claimants per action and the
    legacy dict silently kept the last one in list order. Here each action gets owners explicitly:
    - single: one owner. Among the prose sentences of the action's response (same seq and session), preferred in
      order: (1) the last sentence naming one of the action's files (basename or path); (2) the last sentence whose
      role is a root role (proposal/commitment/hypothesis); (3) the last sentence. The owner is the approach holding
      that sentence (root > core member > member; ties by latest first_seq); if the chosen sentence is in no approach
      the next candidate in the same preference order is tried.
    - share: each union claimant gets 1/k (attribution-free check).
    - legacy: the old overwrite (last claimant in list order), kept only for comparison.
    """
    out: dict[str, dict[str, float]] = {}
    if mode == "legacy":
        for ap in approaches:
            for a in ap.actions:
                out[a] = {ap.unit_id: 1.0}
        return out
    claim: dict[str, list[str]] = defaultdict(list)
    for ap in approaches:
        for a in ap.actions:
            claim[a].append(ap.unit_id)
    if mode == "share":
        return {a: {u: 1.0 / len(us) for u in us} for a, us in claim.items()}
    home: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for ap in approaches:
        roots, core = set(ap.roots), set(getattr(ap, "core_members", ()) or ())
        for m in ap.members:
            home[m].append((0 if m in roots else 1 if m in core else 2, -(ap.first_seq or 0), ap.unit_id))
    for a in state_changing:
        act = by_id.get(a)
        if act is None:
            continue
        turn = sorted((i for i in view.items if i.source_kind in PROSE and i.seq == act.seq
                       and i.session_id == act.session_id), key=lambda i: (i.available_seq, i.order))
        if not turn:
            continue
        names = {n for f in state_changing.get(a) or () for n in (f, f.rsplit("/", 1)[-1]) if n and len(n) > 2}
        named_s = [i for i in turn if any(n in i.text for n in names)]
        rooted_s = [i for i in turn if i.label("role") in ROOT_ROLES]
        for cand in list(reversed(named_s)) + list(reversed(rooted_s)) + list(reversed(turn)):
            if home.get(cand.item_id):
                out[a] = {min(home[cand.item_id])[2]: 1.0}
                break
    return out


def current_approach_owned(approaches, owners, state_changing: Mapping[str, list[str]], by_id, cut: int, x: int):
    """Current approach = most owner weight over the last ``x`` state-changing actions up to ``cut``; ties by the
    latest owned action, then latest activity. Same fallbacks as ``current_approach``."""
    acts = sorted((a for a in state_changing if by_id.get(a) is not None and by_id[a].seq <= cut),
                  key=lambda a: by_id[a].seq)[-x:]
    score: Counter = Counter()
    last: dict[str, int] = {}
    for a in acts:
        for u, w in owners.get(a, {}).items():
            score[u] += w
            last[u] = max(last.get(u, -1), by_id[a].seq)
    if score:
        act_of = {ap.unit_id: ap.last_activity_seq for ap in approaches}
        return max(score, key=lambda u: (round(score[u], 9), last[u], act_of.get(u, -1))), "last_x_state_changing"
    return current_approach(approaches, {}, by_id, cut, x)


def _has(ax_values: Mapping[str, set[str]]):
    """Predicate over axes: true when any listed axis carries one of the listed values."""
    return lambda item, approach=None: any(item.label(ax) in vals for ax, vals in ax_values.items())


PLAN = {"action": {"proposal", "decision", "next_step"}}
HYP = {"epistemic": {"hypothesis"}}
SUCCESS = {"self_assessment": {"success_claim"}}
BELIEF = {"epistemic": {"hypothesis", "inference"}}
REASONING_CHANNEL = {"worker_reasoning", "provider_summary"}
NONFACT = {"action": {"proposal", "decision", "next_step"}, "self_assessment": {"success_claim", "failure_claim",
           "correction"}, "epistemic": {"hypothesis", "question"}}


def view_specs(scope: str) -> dict[str, FilterSpec]:
    """The user's starting Ledger arms (2026-09-27) as removal views over one frozen packet."""
    return {
        f"minus_plans@{scope}": FilterSpec("minus_plans", None, scope, _has(PLAN)),
        f"minus_hypotheses@{scope}": FilterSpec("minus_hypotheses", None, scope, _has(HYP)),
        f"minus_success_claims@{scope}": FilterSpec("minus_success_claims", None, scope, _has(SUCCESS)),
    }


def matched_control(packet, view, removed: set[str], count, *, seed: int, exclude: set[str]) -> dict[str, Any]:
    """Random deletion matched to the target on tokens (+-10%) and, per item, on recency tertile, source kind and
    concreteness; reports residual differences in link count and shared mentions (user A7)."""
    by_id = view.by_id
    prose = [i for i in packet.item_ids if by_id[i].source_kind in PROSE]
    if not removed or not prose:
        return {"status": "no_target"}
    rank = {iid: n / max(1, len(prose) - 1) for n, iid in enumerate(sorted(prose, key=lambda i: by_id[i].available_seq))}
    stratum = lambda i: (min(2, int(rank.get(i, 0) * 3)), by_id[i].source_kind, by_id[i].label("specificity"))
    pool = defaultdict(list)
    for i in prose:
        if i not in removed and i not in exclude:
            pool[stratum(i)].append(i)
    rng = random.Random(seed)
    for v in pool.values():
        rng.shuffle(v)
    target = sum(item_tokens(by_id[i], count) for i in removed)
    chosen, got = [], 0
    for i in sorted(removed, key=lambda i: rng.random()):
        if pool[stratum(i)]:
            j = pool[stratum(i)].pop()
            chosen.append(j)
            got += item_tokens(by_id[j], count)
    rest = [j for v in pool.values() for j in v]
    while got < 0.9 * target and rest:
        j = rest.pop()
        chosen.append(j)
        got += item_tokens(by_id[j], count)
    ok = 0.9 * target <= got <= 1.1 * target
    return {"status": "matched" if ok else "approximate", "seed": seed, "target_tokens": target, "deleted_tokens": got,
            "deleted_ids": chosen, "item_ids": [i for i in packet.item_ids if i not in set(chosen)]}


def collateral_loss(packet, view, removed: set[str], named: Mapping[str, frozenset]) -> dict[str, Any]:
    """Removed sentences that were the ONLY carrier (in the packet, outputs included) of a named test, file or error."""
    kept = [i for i in packet.item_ids if i not in removed]
    kept_names = set().union(*(named.get(i, frozenset()) for i in kept)) if kept else set()
    lost = {i: sorted(n for n in named.get(i, frozenset()) if n not in kept_names) for i in removed}
    lost = {i: v for i, v in lost.items() if v}
    return {"removed_sentences": len(removed), "sentences_with_lost_names": len(lost),
            "lost_names": sorted({n for v in lost.values() for n in v})}


def claims_per_test(items: Sequence, source_store, run_id: str) -> list[dict[str, Any]]:
    """Success claims that name a test, checked against that test's own result at the latest grade before them."""
    grades = []
    for cp in source_store.list_checkpoints(run_id):
        if getattr(cp, "evaluation_ref", None) is None:
            continue
        try:
            ev = source_store.get_json(cp.evaluation_ref)
        except Exception:
            continue
        checks = {c.get("check_id"): c.get("status") or c.get("result") or c.get("outcome") for c in ev.get("checks") or ()}
        grades.append((cp.event_sequence, cp.checkpoint_id, checks, ev.get("full_solve")))
    grades.sort()
    out = []
    for it in items:
        if it.source_kind not in PROSE or it.label("self_assessment") != "success_claim":
            continue
        tests = [t.split(":", 1)[1] for t in artifacts_for([it]).get(it.item_id, ()) if t.startswith("test:")]
        g = [x for x in grades if x[0] <= it.available_seq]
        row = {"item_id": it.item_id, "text": it.text[:200], "named_tests": tests,
               "grade_checkpoint": g[-1][1] if g else None, "full_solve": g[-1][3] if g else None}
        if g and tests:
            hits = {cid: st for cid, st in g[-1][2].items() if cid and any(t in cid for t in tests)}
            row["test_results"] = dict(Counter(str(v) for v in hits.values()))
            row["verdict"] = ("no_matching_check" if not hits else
                              "all_pass" if all(str(v).lower() in ("pass", "passed", "true") for v in hits.values())
                              else "some_fail")
        else:
            row["verdict"] = "no_grade" if not g else "no_named_test"
        out.append(row)
    return out


def _dangling(view, packet: set[str], removed: set[str]) -> int:
    """Kept packet items with an explicit link whose target was removed (a reference left without its referent)."""
    return len({l.source_id for l in view.links if l.tier == "explicit_wording" and l.source_id in packet
                and l.source_id not in removed and l.target_id in removed})


def _window_views(snap, views, record, pk, view, cur_ap, wmeta, L, cp, branchable, count) -> None:
    """Views for the implemented-decision windows (user 2026-09-27; audit fixes).
    - minus_current_approach = everything in the current window, i.e. everything since the start of the reasoning for
      the last carried-out decision (prose members and the outputs of the window's commands).
    - minus_current_approach_tombstone = the same, keeping word-for-word the anchor decision and the window's last
      failing output (else its last output), so the record that the approach was tried and failed survives.
    - minus_shifted_cut_earlier / _later = the same packet-tail cut placed at a turn boundary that is not a window
      start, 25-50% of the window's size earlier or later (does the anchor boundary matter?).
    - minus_matched_noncurrent_slice = a contiguous same-size slice from the non-current region, closest in prose
      share (placebo that is available whenever that region is large enough)."""
    by_id = view.by_id
    items = sorted(view.items, key=lambda i: (i.available_seq, i.seq, i.order))
    tix = {i.item_id: k for k, i in enumerate(items)}
    order_pk = sorted(pk.item_ids, key=lambda i: tix.get(i, -1))
    wcur = wmeta["windows"][-1]
    start = wcur["start"]
    mem = set(cur_ap.members) | set(cur_ap.outputs)
    win = {i for i in order_pk if i in mem or (by_id[i].source_kind not in PROSE and tix.get(i, -1) >= start)}
    rem = record("minus_current_approach", win)
    views["minus_current_approach"]["definition"] = "everything since the start of the reasoning for the last carried-out decision"
    keep = {wcur["anchor"]} & win if wcur.get("anchor") else set()
    outs = [i for i in order_pk if i in win and by_id[i].source_kind == OUTPUT]
    fails = [i for i in outs if by_id[i].label("outcome") == "fail"]
    if outs:
        keep.add((fails or outs)[-1])
    record("minus_current_approach_tombstone", win - keep)
    views["minus_current_approach_tombstone"]["kept"] = sorted(keep)
    views["current_approach_only"] = {"item_ids": [i for i in order_pk if i in win], "removed_ids": []}
    # shifted tail cuts
    p = next((j for j, i in enumerate(order_pk) if i in win), len(order_pk))
    size = len(order_pk) - p
    turn_starts = [j for j in range(1, len(order_pk)) if by_id[order_pk[j]].seq != by_id[order_pk[j - 1]].seq]
    win_starts = set()
    for w in wmeta["windows"]:
        j = next((j for j, i in enumerate(order_pk) if tix.get(i, -1) >= w["start"]), None)
        if j is not None:
            win_starts.add(j)
    # the cut goes to the turn boundary (not a window start) whose distance from the anchor cut is closest to 37.5% of
    # the window's packet size (the middle of the 25-50% band), searched within 10-100% of the size; if no turn
    # boundary exists there, the item boundary at exactly 37.5% (deterministic, no sampling)
    for name, sign in (("minus_shifted_cut_earlier", -1), ("minus_shifted_cut_later", 1)):
        want = p + sign * max(1, round(0.375 * size))
        lo, hi = sorted((p + sign * max(1, size // 10), p + sign * size))
        cands = [j for j in turn_starts if lo <= j <= hi and j not in win_starts and 0 < j < len(order_pk)]
        j = min(cands, key=lambda j: (abs(j - want), j)) if cands else (want if 0 < want < len(order_pk) else None)
        record(name, set(order_pk[j:]) if j is not None and size >= 4 else set())
        views[name]["cut"] = {"index": j, "anchor_index": p, "offset_frac": round((j - p) / size, 3) if j is not None and size else None,
                              "boundary": "turn" if cands else "item"}
    # contiguous non-current placebo
    region = order_pk[:p]
    prose_share = lambda ids: sum(by_id[i].source_kind in PROSE for i in ids) / max(len(ids), 1)
    target = prose_share(order_pk[p:])
    starts = [0] + [j for j in turn_starts if j + size <= len(region)]
    best = min((s for s in starts if s + size <= len(region)), default=None,
               key=lambda s: (abs(prose_share(region[s:s + size]) - target), -s))
    record("minus_matched_noncurrent_slice", set(region[best:best + size]) if best is not None and size else set())
    # current-window facts
    first = items[start].available_seq if start < len(items) else None
    before = [c for c in L.checkpoints if first is not None and c.event_sequence < first and c.ordinal <= cp.ordinal
              and (branchable is None or c.checkpoint_id in branchable)]
    toks = views["minus_current_approach"].get("removed_tokens", 0)
    snap["current_window"] = {"unit": wcur["unit"], "anchor": wcur.get("anchor"), "start_item": wcur.get("start_item"),
                              "backfill": wcur.get("backfill"), "no_anchor": wmeta["no_anchor"],
                              "override_moved": wmeta.get("override_moved"), "packet_items": len(win),
                              "packet_share_items": round(len(win) / max(len(order_pk), 1), 3),
                              "packet_share_tokens": round(toks / max(pk.tokens, 1), 3),
                              "rollback_checkpoint": before[-1].checkpoint_id if before else None}


def snapshot(L, links, cp, *, state_changing, named, x: int, budgets, count, s_views: bool,
             branchable: set[str] | None = None, owner_mode: str = "single", grouping: str = "decision_bounded",
             submissions: frozenset[str] = frozenset(), current_share: float | None = None) -> dict[str, Any]:
    """Everything the Ledger holds at one checkpoint; removal views only at stall (S) checkpoints."""
    view = at_checkpoint21(L, cp, links)
    by_id = view.by_id
    extra = set()
    for a, files in state_changing.items():
        act = by_id.get(a)
        if act is None:
            continue
        same = [i for i in view.items if i.source_kind in PROSE and i.seq == act.seq and i.session_id == act.session_id]
        if same:
            extra.add(same[-1].item_id)
    st = build_structure(view, [c.event_sequence for c in L.checkpoints], extra_roots=frozenset(extra),
                         grouping=grouping, commitments=frozenset(state_changing) | frozenset(submissions))
    aps = st.approaches
    owners = {m: action_owners(aps, view, state_changing, by_id, m) for m in OWNER_MODES}
    by_mode_x = {m: {str(k): current_approach_owned(aps, owners[m], state_changing, by_id, cp.event_sequence, k)[0]
                     for k in (1, 2, 3, 5, 8)} for m in OWNER_MODES}
    cur, cur_rule = current_approach_owned(aps, owners[owner_mode], state_changing, by_id, cp.event_sequence, x)
    wmeta = st.stats.get("windows") if grouping == "implemented_decision" else None
    win_by_unit = {w["unit"]: w for w in (wmeta or {}).get("windows", [])}
    if wmeta is not None:  # the window containing the stall; mechanical (user 2026-09-27)
        cur, cur_rule = ((wmeta["windows"][-1]["unit"], "no_anchor" if wmeta["no_anchor"] else "stall_window")
                         if wmeta["windows"] else (None, "none"))
    snap: dict[str, Any] = {
        "checkpoint_id": cp.checkpoint_id, "annotations": list(cp.annotations), "version": LEDGER2_VERSION,
        "approaches": [{"unit_id": a.unit_id, "size": len(a.members), "status": a.status, "roots": a.roots,
                        "first_seq": a.first_seq, "last_seq": a.last_activity_seq, "actions": len(a.actions),
                        # single-owner map (fix 2026-09-27): state-changing actions this approach owns, and their files
                        "owned": sorted(x for x, w in owners["single"].items() if a.unit_id in w),
                        "files": sorted({f for x, w in owners["single"].items() if a.unit_id in w
                                         for f in state_changing.get(x) or ()}),
                        "window": win_by_unit.get(a.unit_id)} for a in aps],
        "current": cur, "current_rule": cur_rule, "x": x, "owner_mode": owner_mode, "stats": st.stats, "packets": {},
        # X calibration (user A4; ownership fix 2026-09-27): current approach per owner mode and candidate X
        "current_by_x": by_mode_x[owner_mode], "current_by_mode_x": by_mode_x, "current_share": current_share}
    packets = {}
    for b in budgets:
        pk = build_packet22(view, aps, cur, b, count, current_share=current_share if wmeta is not None else None)
        packets[b] = pk
        snap["packets"][str(b or "full")] = {"item_ids": pk.item_ids, "tokens": pk.tokens}
    if not s_views:
        return snap
    pk = packets[3000]
    views: dict[str, Any] = {}
    cur_ap = next((a for a in aps if a.unit_id == cur), None)

    def record(name, removed: set[str]):
        removed = {i for i in removed if i in set(pk.item_ids)}
        first = min((by_id[i].available_seq for i in removed), default=None)
        # the latest BRANCHABLE checkpoint before the earliest removed item (a submission checkpoint or a failed
        # capture cannot be restored, so it is never a rollback point)
        before = [c for c in L.checkpoints if first is not None and c.event_sequence < first and c.ordinal <= cp.ordinal
                  and (branchable is None or c.checkpoint_id in branchable)]
        acts = {a for ap in aps for a in ap.actions if set(ap.members) & removed}
        views[name] = {"item_ids": [i for i in pk.item_ids if i not in removed], "removed_ids": sorted(removed),
                       "removed_tokens": sum(item_tokens(by_id[i], count) for i in removed),
                       "rollback_checkpoint": before[-1].checkpoint_id if before else None,
                       "removed_actions": sorted(acts),
                       "removed_files": sorted({f for a in acts for f in state_changing.get(a, [])}),
                       "collateral": collateral_loss(pk, view, removed, named)}
        views[name]["dangling"] = _dangling(view, set(pk.item_ids), removed)
        if removed:
            mc = matched_control(pk, view, removed, count, seed=1, exclude=set())
            views[name]["matched_control"] = mc
            deleted = set(mc.get("deleted_ids") or ())
            mc["overlap_with_treatment"] = round(len(deleted & removed) / len(removed), 3)
            mc["dangling"] = _dangling(view, set(pk.item_ids), deleted)
        return removed

    views["full"] = {"item_ids": list(pk.item_ids), "removed_ids": []}
    prose_in = [i for i in pk.item_ids if by_id[i].source_kind in PROSE]
    for scope in ("global", "current_approach"):
        for name, spec in view_specs(scope).items():
            fv = apply_filter(pk, view, aps, cur, spec, count)
            record(name, set(fv.removed_ids))
    record("facts_only", {i for i in prose_in if _has(NONFACT)(by_id[i])})
    # user 2026-09-27 ablations: beliefs = hypotheses and inferences; observed_only keeps only prose that reports an
    # observation (and nothing else) plus tool outputs; minus_reasoning_channel keeps what the worker wrote visibly
    # plus tool outputs (the injected-reasoning confound: DeepSeek's reasoning is not re-read natively)
    record("minus_beliefs@global", {i for i in prose_in if _has(BELIEF)(by_id[i])})
    # the plans x beliefs 2x2's "both removed" cell: exactly the union of the two single removals
    record("minus_plans_and_beliefs@global", set(views["minus_plans@global"]["removed_ids"])
           | set(views["minus_beliefs@global"]["removed_ids"]))
    record("observed_only", {i for i in prose_in
                             if by_id[i].label("epistemic") != "observation" or _has(NONFACT)(by_id[i])})
    record("minus_reasoning_channel", {i for i in prose_in if by_id[i].source_kind in REASONING_CHANNEL})
    if cur_ap is not None and wmeta is not None:
        _window_views(snap, views, record, pk, view, cur_ap, wmeta, L, cp, branchable, count)
        # size of each window in the unbudgeted Ledger (tokens as displayed), to judge packing at 3k
        full = packets.get(None)
        if full is not None:
            tix = {i.item_id: k for k, i in enumerate(sorted(view.items, key=lambda i: (i.available_seq, i.seq, i.order)))}
            snap["current_window"]["window_tokens_unbudgeted"] = [
                sum(item_tokens(by_id[i], count) for i in full.item_ids if w["start"] <= tix.get(i, -1) < w["end"])
                for w in wmeta["windows"]]
            snap["current_window"]["ledger_tokens_unbudgeted"] = full.tokens
    elif cur_ap is not None:
        cur_ids = set(cur_ap.members) | set(cur_ap.reference_dependencies)
        rem = record("minus_current_approach", cur_ids)
        views["current_approach_only"] = {"item_ids": [i for i in pk.item_ids if i in cur_ids or by_id[i].source_kind == OUTPUT
                                                       and i in set(cur_ap.outputs)], "removed_ids": []}
        # placebo: another approach of similar size and recency that is not the current one
        others = [a for a in aps if a.unit_id != cur and set(a.members) & set(pk.item_ids)]
        if others and rem:
            size = len(rem)
            best = min(others, key=lambda a: (abs(len(set(a.members) & set(pk.item_ids)) - size),
                                              abs(a.last_activity_seq - cur_ap.last_activity_seq)))
            record("minus_matched_noncurrent_approach", set(best.members) | set(best.reference_dependencies))
    snap["views"] = views
    return snap
