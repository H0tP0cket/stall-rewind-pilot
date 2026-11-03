"""Ledger v2 offline incremental rebuild: one snapshot per checkpoint, views + checks at stall checkpoints.

  python -m trajectory_retention.observers.t2.ledger2_offline --source STORE [...] --out OUT --iteration N \
      --jev-budget-usd B --llm-budget-usd B [--runs SUBSTR ...] [--x 3] [--linker MODEL] [--no-probe]

Outputs (E2E artifact, re-runnable from cached calls):
  OUT/store                              receipts and records (ledger2_run, ledger2_snapshot)
  OUT/iteration-N/manifests/<run>/<cp>.json   per-checkpoint snapshot (the interface for the experiment driver)
  OUT/iteration-N/packets/<run>/<cp>/<view>.txt   rendered packets (time order) at stall checkpoints
  OUT/iteration-N/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import tiktoken

from ...store import ArtifactStore
from .artifacts22 import artifacts_for
from .glm import GlmClient
from .jev import JevClient
from .ledger2 import (LEDGER2_VERSION, apply_derived, claims_per_test, link_run, pass_axes, snapshot,
                      state_changing_actions, submission_actions)
from .offline import load_env
from .packet21 import HEADER
from .packet22 import _block_lines, _pos, _result_of
from .probe22 import distractor_pool, probe_family_choice
from .schema import UNKNOWN, Label, Link
from .sources21 import RunLedger, load_ledger
from .store_ro import ReadOnlyStore
from .units21 import at_checkpoint21
from .units22 import build_structure

ENC = tiktoken.get_encoding("o200k_base")
_TOK: dict[str, int] = {}
BUDGETS = (None, 6000, 3000, 1500)
LINKERS = {
    "glm-5.3-flash/low": dict(model="z-ai/glm-5.3-flash", reasoning={"effort": "low"}),
    "glm-5.3-flash/medium": dict(model="z-ai/glm-5.3-flash", reasoning={"effort": "medium"}),
    "mimo-v2.6-pro/low": dict(model="xiaomi/mimo-v2.6-pro", provider_slug="xiaomi/fp8", expected_provider="Xiaomi",
                              usd_per_token=(0.435e-6, 0.87e-6), reasoning={"effort": "low"}),
}
PROBED = ("minus_plans@global", "minus_hypotheses@global", "minus_success_claims@global", "minus_current_approach",
          "facts_only")


def resume_origin(src, run_id: str) -> str | None:
    """The checkpoint a resumed run (a resample) started from, from its first lifecycle event."""
    for e in src.iter_events(run_id):
        p = e.payload if isinstance(e.payload, dict) else {}
        if p.get("resume_from"):
            return p["resume_from"]
        if e.sequence >= 20:
            return None
    return None


def load_joined(src, run_id: str):
    """A resumed run records only what happened after its resume point; the history before it is the source run's,
    restored as one block. Join the source run's items up to the resume checkpoint with the resumed run's items,
    shifted to follow it, so the Ledger holds everything the worker had written. Returns the joined ledger, each
    item's (run, sequence offset), and the source checkpoints (kept for rollback, never snapshotted here)."""
    L = load_ledger(src, run_id)
    origin = resume_origin(src, run_id)
    if origin is None:
        return L, {i.item_id: (run_id, 0) for i in L.items}, [], None
    source_run = origin.rsplit("/", 1)[0]
    P = load_ledger(src, source_run)
    ocp = next(c for c in P.checkpoints if c.checkpoint_id == origin)
    cut = ocp.event_sequence
    pre = [i for i in P.items if i.available_seq <= cut]
    keep = {i.item_id for i in pre}
    post = [replace(i, seq=i.seq + cut, available_seq=i.available_seq + cut, span=replace(i.span, seq=i.span.seq + cut))
            for i in L.items]
    links = [l for l in P.links if l.source_id in keep and l.target_id in keep and l.evidence_seq <= cut] + \
            [replace(l, evidence_seq=l.evidence_seq + cut) for l in L.links]
    pre_cps = [c for c in P.checkpoints if c.event_sequence <= cut]
    cps = pre_cps + [replace(c, ordinal=c.ordinal + ocp.ordinal + 1, event_sequence=c.event_sequence + cut)
                     for c in L.checkpoints]
    decoded = {k: v for k, v in P.decoded_fields.items() if k[0] <= cut} | \
              {(sq + cut, f): v for (sq, f), v in L.decoded_fields.items()}
    J = RunLedger(run_id, L.model or P.model, pre + post, links, P.discards + L.discards, cps,
                  P.task_text_by_session | L.task_text_by_session, decoded, L.worker_cost_usd, L.old_observer_cost_usd,
                  [x for x in P.unreplayed_reasoning_events if x <= cut] + [x + cut for x in L.unreplayed_reasoning_events])
    part = {i.item_id: (source_run, 0) for i in pre} | {i.item_id: (run_id, cut) for i in L.items}
    return J, part, pre_cps, {"source_run": source_run, "resume_checkpoint": origin, "cut": cut,
                              "prefix_items": len(pre), "prefix_items_available_after_cut": sum(
                                  1 for i in P.items if i.seq <= cut < i.available_seq)}


def per_part(fn, src, items, part):
    """Run a per-run store reader on each joined part with the items' own sequence numbers."""
    by_run: dict[str, list] = {}
    for i in items:
        r, off = part[i.item_id]
        by_run.setdefault(r, []).append(replace(i, seq=i.seq - off, available_seq=i.available_seq - off,
                                                span=replace(i.span, seq=i.span.seq - off)) if off else i)
    return [fn(r, its) for r, its in by_run.items()]


def tokens(text: str) -> int:
    n = _TOK.get(text)
    if n is None:
        n = _TOK[text] = len(ENC.encode(text, disallowed_special=()))
    return n


def strkeys(x):
    """Records require string keys (histograms use ints)."""
    if isinstance(x, dict):
        return {str(k): strkeys(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strkeys(v) for v in x]
    return x


def render_time(view, ids) -> str:
    """Word-for-word notes in original time order (outputs under their command)."""
    pos = _pos(view)
    ids = sorted({i for i in ids if i in view.by_id}, key=pos.get)
    return "\n".join([HEADER, ""] + _block_lines(view, ids, _result_of(view))).rstrip() + "\n"


def robustness(L, links, cp, extra_frac: float = 0.2, seed: int = 7) -> dict[str, Any]:
    """Link-robustness (user A11): drop / add 20% of links, rebuild approaches, compare memberships."""
    view = at_checkpoint21(L, cp, links)
    base = build_structure(view, [c.event_sequence for c in L.checkpoints]).approaches
    owner = {m: a.unit_id for a in base for m in a.members}
    rng = random.Random(seed)
    explicit = [l for l in view.links if l.tier == "explicit_wording"]
    out = {}
    for name in ("drop", "add"):
        if name == "drop":
            keep = [l for l in explicit if rng.random() >= extra_frac]
            new_links = [l for l in view.links if l.tier != "explicit_wording"] + keep
        else:
            prose = [i for i in view.items if i.source_kind in ("worker_text", "worker_reasoning", "provider_summary",
                                                                "worker_code_comment")]
            add = []
            for _ in range(int(len(explicit) * extra_frac)):
                if len(prose) < 2:
                    break
                a, b = sorted(rng.sample(range(len(prose)), 2))
                add.append(Link(prose[b].item_id, prose[a].item_id, "continues_or_builds_on", "explicit_wording",
                                None, prose[b].available_seq, "robustness:random-add"))
            new_links = list(view.links) + add
        v2 = at_checkpoint21(L, cp, new_links)
        aps = build_structure(v2, [c.event_sequence for c in L.checkpoints]).approaches
        own2 = {m: a.unit_id for a in aps for m in a.members}
        members = [m for m in owner if m in own2]
        same = sum(1 for i, m in enumerate(members) for n in members[i + 1:i + 6]
                   if (owner[m] == owner[n]) == (own2[m] == own2[n]))
        total = sum(1 for i, m in enumerate(members) for n in members[i + 1:i + 6])
        out[name] = {"pairwise_agreement": same / total if total else None, "approaches_before": len(base),
                     "approaches_after": len(aps)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", required=True)
    ap.add_argument("--jev-budget-usd", type=float, required=True)
    ap.add_argument("--llm-budget-usd", type=float, required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--x", type=int, default=3)
    ap.add_argument("--owner-mode", default="single", choices=["single", "share", "legacy"])
    ap.add_argument("--grouping", default="decision_bounded", choices=["decision_bounded", "implemented_decision"])
    ap.add_argument("--budgets-are-new", action="store_true",
                    help="the two budgets cap new spend in this invocation, on top of what the out store already spent")
    ap.add_argument("--through-states", action="store_true",
                    help="with --states: label and link only items up to each run's last state checkpoint")
    ap.add_argument("--current-share", type=float, default=None,
                    help="implemented_decision packing: minimum share of each budget for the current window (1.0 = newest-first)")
    ap.add_argument("--action-threshold", type=float, default=None,
                    help="sensitivity only: re-derive the action label at this probability threshold (default: as labelled, 0.6)")
    ap.add_argument("--linker", default="glm-5.3-flash/low", choices=sorted(LINKERS))
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--context", type=int, default=30, help="recent items shown to the linker per chunk")
    ap.add_argument("--max-roots", type=int, default=40, help="earlier plan/hypothesis sentences shown per chunk")
    ap.add_argument("--states", default=None, help="postrun states.json: also build views at exactly these checkpoints; "
                    "without --runs, only these runs are built")
    a = ap.parse_args(argv)
    load_env(Path("~/.config/trajectory-retention/openrouter.env").expanduser())
    out = Path(a.out).expanduser().resolve()
    rd = out / f"iteration-{a.iteration}"
    rd.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(out / "store")
    jev = JevClient(store, budget_usd=a.jev_budget_usd)
    linker = GlmClient(store, budget_usd=a.llm_budget_usd, max_tokens=16384, timeout=180.0, **LINKERS[a.linker])
    probe = GlmClient(store, budget_usd=a.llm_budget_usd, max_tokens=4096, timeout=180.0, reasoning={"effort": "low"})
    if a.budgets_are_new:  # the clients count everything the out store has ever spent; cap only this run's new spend
        jev.budget = jev.spent + a.jev_budget_usd
        linker.budget = linker.spent + a.llm_budget_usd
        probe.budget = probe.spent + a.llm_budget_usd
    errors: list[str] = []
    log = lambda m: (errors.append(m), print(m, file=sys.stderr))
    report: dict[str, Any] = {"version": LEDGER2_VERSION, "linker": a.linker, "x": a.x, "owner_mode": a.owner_mode, "grouping": a.grouping, "action_threshold": a.action_threshold, "current_share": a.current_share, "context": a.context,
                              "max_roots": a.max_roots, "runs": []}
    state_cps: set[str] = set()
    if a.states:
        sj = json.loads(Path(a.states).read_text())
        state_cps = {s["checkpoint_id"] for s in (sj["states"] if isinstance(sj, dict) else sj)}
        report["states"] = {"path": str(Path(a.states).resolve()), "checkpoints": sorted(state_cps)}
    for sp in a.source:
        src = ReadOnlyStore(sp)
        runs = src.run_ids() if not a.runs else [r for r in src.run_ids() if any(s in r for s in a.runs)]
        if a.states and not a.runs:
            runs = [r for r in runs if any(c.startswith(r + "/") for c in state_cps)]
        done = {r["record_id"] for r in store.list_records("ledger2_run")}
        for run_id in runs:
            if f"{a.iteration}:{run_id}" in done:  # already built in this out store (e.g. by a priority run)
                print(f"{run_id.split(':')[1][:8]} skipped: already built in iteration {a.iteration}", file=sys.stderr)
                continue
            t0, j0, g0 = time.time(), jev.spent, linker.spent + probe.spent
            L, part, pre_cps, joined = load_joined(src, run_id)
            if joined:
                print(f"{run_id} joined: {joined}", file=sys.stderr)
            if a.through_states and state_cps:  # nothing after the run's last stall state can enter its views
                last = max((c.event_sequence for c in L.checkpoints if c.checkpoint_id in state_cps), default=None)
                if last is not None:
                    L.items = [i for i in L.items if i.available_seq <= last]
                    kept = {i.item_id for i in L.items}
                    L.links = [l for l in L.links if l.source_id in kept and l.target_id in kept]
                    L.checkpoints = [c for c in L.checkpoints if c.event_sequence <= last]
            commands = {i.tool_call_id: i.text for i in L.items if i.source_kind == "worker_action" and i.tool_call_id}
            sa = pass_axes(L.items, L.decoded_fields, jev, workers=a.workers, on_error=log)
            apply_derived(L.items, commands)
            if a.action_threshold is not None:  # sensitivity rebuild: re-derive action labels from stored probabilities
                for it in L.items:
                    lab = it.labels.get("action")
                    if lab is not None and lab.probabilities:
                        top, p = max(lab.probabilities.items(), key=lambda kv: kv[1])
                        it.labels["action"] = Label(top if p >= a.action_threshold else UNKNOWN, lab.probabilities,
                                                    f"{lab.method}:threshold={a.action_threshold}", lab.confidence)
            links, ls = link_run(linker, L.items, commands, context=a.context, max_roots=a.max_roots, on_error=log)
            all_links = L.links + links
            changing = {k: v for d in per_part(lambda r, its: state_changing_actions(src, r, its), src, L.items, part)
                        for k, v in d.items()}
            submits = frozenset().union(*per_part(lambda r, its: submission_actions(src, r, its), src, L.items, part)) \
                if a.grouping == "implemented_decision" else frozenset()
            branchable = {c.checkpoint_id for r in {r for r, _ in part.values()} | {run_id}
                          for c in src.list_checkpoints(r) if str(getattr(c, "branchable", "")) == "True"}
            named = artifacts_for(L.items, L.links)
            claims = [c for cl in per_part(lambda r, its: claims_per_test(its, src, r), src, L.items, part) for c in cl]
            store.put_record("ledger2_run", f"{a.iteration}:{run_id}", {
                "version": LEDGER2_VERSION, "run_id": run_id, "model": L.model,
                "items": [i.as_record() for i in L.items], "links": [l.as_record() for l in all_links],
                "state_changing": changing, "claims_per_test": claims, "link_stats": ls, "joined": joined}, run_id=run_id)
            parts = run_id.split(":")  # same key as the experiment machinery's arms.run_key(run_id, 12)
            hx = next((k for k, p in enumerate(parts) if len(p) == 12 and all(c in "0123456789abcdef" for c in p)), 1)
            stem = "-".join([parts[hx][:12]] + parts[hx + 1:]) if run_id.startswith("resample-") else parts[1][:12]
            skip = {c.checkpoint_id for c in pre_cps}
            (rd / "manifests" / stem).mkdir(parents=True, exist_ok=True)
            rows = []
            for cp in L.checkpoints:
                if cp.checkpoint_id in skip:  # the source run's own checkpoints: rollback targets only
                    continue
                is_s = "S" in cp.annotations or cp.checkpoint_id in state_cps
                try:
                    snap = snapshot(L, all_links, cp, state_changing=changing, named=named, x=a.x,
                                    budgets=BUDGETS, count=tokens, s_views=is_s, branchable=branchable,
                                    owner_mode=a.owner_mode, grouping=a.grouping, submissions=submits,
                                    current_share=a.current_share)
                except Exception as exc:  # recorded, never silently dropped
                    log(f"{run_id} {cp.checkpoint_id}: {type(exc).__name__}: {exc}")
                    continue
                cpn = cp.checkpoint_id.split("/")[-1]
                view = at_checkpoint21(L, cp, all_links)
                if is_s:
                    pdir = rd / "packets" / stem / cpn
                    pdir.mkdir(parents=True, exist_ok=True)
                    full_text = render_time(view, snap["packets"]["3000"]["item_ids"])
                    for b, p in snap["packets"].items():
                        (pdir / f"budget_{b}.txt").write_text(render_time(view, p["item_ids"]))
                    for vn, v in snap["views"].items():
                        text = render_time(view, v["item_ids"])
                        (pdir / f"{vn}.txt").write_text(text)
                        v["tokens"] = tokens(text)
                        mc = v.get("matched_control") or {}
                        if mc.get("item_ids"):
                            (pdir / f"{vn}__matched_control.txt").write_text(render_time(view, mc["item_ids"]))
                        if not a.no_probe and vn in PROBED and v["removed_ids"]:
                            rem = sorted(v["removed_ids"], key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:36]
                            rv = {"matched": (render_time(view, mc["item_ids"]), mc.get("deleted_ids") or [])} \
                                if mc.get("item_ids") else None
                            try:
                                v["probe"] = probe_family_choice(
                                    probe, family=vn, full_text=full_text, view_text=text,
                                    removed_items=[view.by_id[i] for i in rem], count=tokens, random_views=rv,
                                    distractor_pool=distractor_pool(L.items, cp.event_sequence, all_links))
                                v["probe"] = {k: v["probe"].get(k) for k in ("recovery_index", "recovery_probe_rate",
                                                                            "ceiling", "floor", "random", "n_paired")}
                            except Exception as exc:
                                v["probe"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    snap["robustness"] = robustness(L, all_links, cp)
                store.put_record("ledger2_snapshot", f"{a.iteration}:{cp.checkpoint_id}", strkeys(snap), run_id=run_id)
                (rd / "manifests" / stem / f"{cpn}.json").write_text(json.dumps(snap, default=str))
                rows.append({"checkpoint_id": cp.checkpoint_id, "S": is_s, "approaches": len(snap["approaches"]),
                             "sizes": dict(Counter(min(x["size"], 10) for x in snap["approaches"])),
                             "current": snap["current"], "current_rule": snap["current_rule"],
                             "views": {vn: {k: v.get(k) for k in ("removed_tokens", "tokens", "rollback_checkpoint",
                                                                  "collateral", "probe")} | {
                                 "control": (v.get("matched_control") or {}).get("status")}
                                       for vn, v in (snap.get("views") or {}).items()},
                             "robustness": snap.get("robustness")})
            report["runs"].append({"run_id": run_id, "model": L.model, "items": len(L.items), "axes": sa,
                                   "links": ls, "state_changing_actions": len(changing),
                                   "claims_per_test": dict(Counter(c["verdict"] for c in claims)),
                                   "checkpoints": rows, "cost": {"jev": jev.spent - j0,
                                                                 "llm": linker.spent + probe.spent - g0},
                                   "seconds": time.time() - t0})
            print(f"{stem[:8]} {L.model} items={len(L.items)} links={ls.get('links')} jev=${jev.spent - j0:.3f} "
                  f"llm=${linker.spent + probe.spent - g0:.3f} {time.time() - t0:.0f}s", flush=True)
            (rd / "report.json").write_text(json.dumps(report, indent=1, default=str))
    report["errors"] = errors[:50]
    report["totals"] = {"jev": jev.spent, "llm": linker.spent + probe.spent}
    (rd / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["totals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
