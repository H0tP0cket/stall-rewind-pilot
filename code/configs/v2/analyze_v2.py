"""Analysis of the v2 main arms (track S). Read-only over the study's spec sets, results and stall dirs; no model calls.

Paired by stall: each arm's outcome is averaged over its repeats within a stall, then compared stall by stall;
intervals resample whole tasks; p-values flip the signs of whole tasks' differences (two-sided). Headline: arm 2 vs
arm 3 on solve rate and on share of hidden tests passed at the best submission (final state also reported), with
Holm across those two. Key secondaries: 2 vs 1, 6 vs 4, 6 vs 5. Everything else is descriptive with intervals.
Only purpose "main" spec sets are read; tuning stalls and screening branches never enter results.

  python configs/v2/analyze_v2.py --study R [--screen-json F] [--out R/analysis]
writes analysis.json, branches.jsonl and report.md.
Rules marked DRAFT below (covariate list, long-context split, re-stall definition, re-estimate targets) are not
ratified; they are pre-registration items.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from collections import defaultdict
from pathlib import Path

from study_common import (ARMS, MAIN_REPEATS, REVIEW_VIEWS, iter_results, load_specset, read_json, runs_dir,
                          write_json)
from trajectory_retention.analysis import v2_stats as S
from trajectory_retention.analysis.submissions_v2 import VERSION as SUBMISSION_BEHAVIOUR
from trajectory_retention.analysis.submissions_v2 import summary as submission_summary

HEADLINE = ("2_ledger_review", "3_context_review")
SECONDARIES = (("2_ledger_review", "1_continue"), ("6_rewind_ledger_recipe", "4_forced_rewind"),
               ("6_rewind_ledger_recipe", "5_rewind_own_summary"))
VS_CONTINUE = tuple((a, "1_continue") for a in ARMS[1:])
OUTCOMES = {"solve": lambda r: 1.0 if r.get("solved") else 0.0,
            "best_share": lambda r: r.get("best_share_passed"),
            "final_share": lambda r: r.get("final_share_passed")}
HOLM_OUTCOMES = ("solve", "best_share")
SCREEN_COVARIATES = {"name": "screen_covariates", "status": "DRAFT (pre-registration item)",
                     "covariates": ["screen_continue_solve_rate", "screen_rewind_solve_rate"]}
LONG_CONTEXT_RULE = {"name": "long_context_split", "status": "DRAFT (pre-registration item)",
                     "rule": "context_tokens_at_stall above the median of the analysed stalls"}
RESTALL_RULE = {"name": "branch_restall", "status": "DRAFT (contract: Open)", "consecutive_identical": 2,
                "rule": "a branch re-stalls at the first submission whose failing set (hash, failing count > 0) "
                        "equals the previous consecutive_identical - 1 submissions' sets"}
REESTIMATE = {"name": "blinded_reestimate", "status": "DRAFT targets (the method uses pooled variance only)",
              "after_stalls": 10, "planned_stalls": 40, "max_stalls": 60, "alpha_two_sided": 0.025, "power": 0.8,
              "target_difference": {"solve": 0.15, "best_share": 0.10}}
BOOT = 4000


def restall_at(subs: list[dict]) -> int | None:
    k = RESTALL_RULE["consecutive_identical"]
    for i in range(k - 1, len(subs)):
        win = subs[i - k + 1:i + 1]
        if win[-1]["failing_count"] > 0 and len({s["failing_hash"] for s in win}) == 1:
            return subs[i]["step"]
    return None


def stall_meta(block: dict, screen: dict) -> dict:
    d = Path(block["stall_dir"])
    st = read_json(d / "stall.json", {})
    led = read_json(d / "ledger.json", None)
    b = read_json(d / "boundary.json", {})
    sc = screen.get(block["unit_id"], {})
    return {"stall_id": block["unit_id"], "task_id": block["task_id"], "stratum": sc.get("stratum"),
            "covariates": sc.get("covariates") or {}, "tuning": sc.get("tuning"), "admitted": sc.get("admitted"),
            "first_approach": b.get("rewind_point") == 0 if b else None, "later_solved": st.get("later_solved"),
            "n_approaches": len(led["approaches"]) if led and isinstance(led.get("approaches"), list) else None,
            "context_tokens": st.get("context_tokens_at_stall"), "stall_step": st.get("step_index"),
            "rewind_point": b.get("rewind_point")}


def branch_row(meta: dict, job: dict, r: dict) -> dict:
    ch = r.get("choice") or {}
    dec = ch.get("decision") or {}
    rw = dec.get("rewind") if "rewind" in dec else ({"coupled": dec.get("rewind_to")} if "rewind_to" in dec else None)
    points = {k: v for k, v in (rw or {}).items() if v is not None} if isinstance(rw, dict) else {}
    subs = r.get("submissions") or []
    return {"stall_id": meta["stall_id"], "task_id": meta["task_id"], "arm": job["arm"], "repeat": job["repeat"],
            "branch_id": job["branch_id"], "solved": bool(r.get("solved")), "best_share": r.get("best_share_passed"),
            "final_share": r.get("final_share_passed"), "end_reason": r.get("end_reason"),
            "submissions_used": len(subs), "new_tokens": r.get("new_tokens"), "review_tokens": r.get("review_tokens"),
            "note_tokens": r.get("note_tokens"), "handback_tokens": r.get("handback_tokens"), "steps": r.get("steps"),
            "cost_usd": r.get("cost_usd"), "cost_usd_uncached": r.get("cost_usd_uncached"),
            "restore_seconds": r.get("restore_seconds"), "wall_seconds": r.get("wall_seconds"),
            "flags": r.get("flags") or {}, "restall_step": restall_at(subs),
            "submission_behaviour": submission_summary(r["store"], r["run_id"]) if r.get("store") and r.get("run_id")
            else None,
            "choice": None if not ch else {"mode": ch.get("mode"), "calls": ch.get("calls"), "valid": ch.get("valid"),
                                           "change_nothing": dec.get("change_nothing"), "rewind_points": points,
                                           "drops": len(dec.get("drop") or []), "keeps": len(dec.get("keep") or [])},
            "stratum": meta["stratum"], "first_approach": meta["first_approach"],
            "later_solved": meta["later_solved"], "n_approaches": meta["n_approaches"],
            "context_tokens": meta["context_tokens"], "stall_step": meta["stall_step"]}


def per_stall(rows: list[dict], outcome: str) -> dict:
    """{stall: {arm: mean over repeats}} for one outcome (missing values dropped, never filled)."""
    f = OUTCOMES[outcome]
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = f({"solved": r["solved"], "best_share_passed": r["best_share"], "final_share_passed": r["final_share"]})
        if v is not None:
            acc[r["stall_id"]][r["arm"]].append(v)
    return {s: {a: statistics.fmean(v) for a, v in arms.items()} for s, arms in acc.items()}


def contrast(rows: list[dict], meta: dict, a: str, b: str, outcome: str, *, covariates: list[str] | None = None) -> dict:
    ps = per_stall(rows, outcome)
    diffs = [(meta[s]["task_id"], v[a] - v[b], s) for s, v in ps.items() if a in v and b in v]
    out = {"contrast": f"{a} - {b}", "outcome": outcome, "stalls": len(diffs),
           "tasks": len({t for t, _, _ in diffs}), "mean_a": S.mean(ps[s][a] for _, _, s in diffs),
           "mean_b": S.mean(ps[s][b] for _, _, s in diffs)}
    out |= S.cluster_bootstrap([(t, d) for t, d, _ in diffs], n=BOOT)
    out["p_two_sided"] = S.sign_flip_p([(t, d) for t, d, _ in diffs])
    if covariates:
        cov_rows = [(t, d, [meta[s]["covariates"].get(c) for c in covariates]) for t, d, s in diffs]
        cov_rows = [r for r in cov_rows if all(x is not None for x in r[2])]
        out["covariate_adjusted"] = S.adjusted_bootstrap(cov_rows) | {"covariates": covariates,
                                                                      "stalls_with_covariates": len(cov_rows)}
    return out


def choice_table(rows: list[dict]) -> dict:
    out = {}
    for arm in REVIEW_VIEWS:
        rs = [r for r in rows if r["arm"] == arm and r["choice"]]
        if not rs:
            continue
        rw = [r for r in rs if r["choice"]["rewind_points"]]
        dist = [r["stall_step"] - min(r["choice"]["rewind_points"].values()) for r in rw if r["stall_step"] is not None]
        chans = defaultdict(int)
        for r in rw:
            for k in r["choice"]["rewind_points"]:
                chans[k] += 1
        out[arm] = {"branches": len(rs), "no_change_share": S.mean(1.0 if r["choice"]["change_nothing"] else 0.0 for r in rs),
                    "invalid_call_share": S.mean(1.0 if r["flags"].get("invalid_call") else 0.0 for r in rs),
                    "mean_calls": S.mean(r["choice"]["calls"] for r in rs),
                    "rewind_share": len(rw) / len(rs), "rewind_channel_counts": dict(chans),
                    "median_rewind_depth_steps": statistics.median(dist) if dist else None,
                    "mean_drops": S.mean(r["choice"]["drops"] for r in rs),
                    "mean_keeps": S.mean(r["choice"]["keeps"] for r in rs),
                    "rescues_items_share": S.mean(1.0 if r["choice"]["keeps"] else 0.0 for r in rs),
                    "mismatch_share": S.mean(1.0 if r["flags"].get("mismatch") else 0.0 for r in rs),
                    "failed_command_share": S.mean(1.0 if r["flags"].get("failed_command") else 0.0 for r in rs),
                    "mean_review_tokens": S.mean(r["review_tokens"] for r in rs),
                    "mean_handback_tokens": S.mean(r["handback_tokens"] for r in rs)}
    return out


def cost_table(study: Path) -> dict:
    out = {}
    for m_path in sorted((study / "specsets").glob("*.json")):
        m = load_specset(m_path)
        by = defaultdict(lambda: {"branches": 0, "cost_usd": 0.0, "cost_usd_uncached": 0.0, "uncached_missing": 0,
                                  "new_tokens": 0, "review_tokens": 0, "note_tokens": 0, "error_attempts": 0,
                                  "error_cost_usd": 0.0})
        for _, job, res, attempts in iter_results(m, runs_dir(study, m["prefix"])):
            t = by[job["arm"]]
            for x in attempts:
                if x["status"] == "error":
                    t["error_attempts"] += 1
                    t["error_cost_usd"] += float(x["result"].get("cost_usd") or 0.0)
            if res is None:
                continue
            t["branches"] += 1
            t["cost_usd"] += float(res.get("cost_usd") or 0.0)
            if res.get("cost_usd_uncached") is None:
                t["uncached_missing"] += 1
            else:
                t["cost_usd_uncached"] += float(res["cost_usd_uncached"])
            for k in ("new_tokens", "review_tokens", "note_tokens"):
                t[k] += int(res.get(k) or 0)
        out[m["prefix"]] = {"purpose": m["purpose"], "arms": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                                                  for kk, vv in v.items()} for k, v in by.items()}}
    return out


def restall_table(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if rs:
            out[arm] = {"branches": len(rs), "restall_share": S.mean(1.0 if r["restall_step"] else 0.0 for r in rs),
                        "identical_failures_end_share": S.mean(1.0 if r["end_reason"] == "identical_failures" else 0.0
                                                               for r in rs),
                        "end_reasons": dict(sorted(_count(r["end_reason"] for r in rs).items()))}
    return out


def submission_table(rows: list[dict]) -> dict:
    """Per arm, means over branches of each branch's submission-behaviour summary (descriptive; None values dropped)."""
    keys = ("submissions", "median_steps_between", "share_resubmitted_with_no_change",
            "share_resubmitted_with_workspace_unchanged", "share_with_own_check_before",
            "share_same_failing_set_as_previous")
    out = {"version": SUBMISSION_BEHAVIOUR}
    for arm in ARMS:
        bs = [r["submission_behaviour"] for r in rows if r["arm"] == arm and r.get("submission_behaviour")]
        if bs:
            out[arm] = {"branches_read": len(bs), **{k: S.mean(b[k] for b in bs if b.get(k) is not None) for k in keys}}
    return out


def _count(xs) -> dict:
    c: dict = defaultdict(int)
    for x in xs:
        c[x] += 1
    return c


def opus_table(study: Path, rows: list[dict]) -> dict | None:
    recs = [read_json(p) for p in sorted((study / "opus-review" / "records").glob("*.json"))]
    if not recs:
        return None
    out = []
    for rec in recs:
        dec = rec.get("decision") or {}
        worker = [r for r in rows if r["stall_id"] == rec.get("stall_id") and r["arm"] == rec.get("arm") and r["choice"]]
        out.append({"stall_id": rec.get("stall_id"), "arm": rec.get("arm"), "valid": rec.get("valid"),
                    "opus_change_nothing": dec.get("change_nothing"), "opus_drops": len(dec.get("drop") or []),
                    "opus_keeps": len(dec.get("keep") or []),
                    "worker_change_nothing_share": S.mean(1.0 if r["choice"]["change_nothing"] else 0.0 for r in worker),
                    "worker_branches": len(worker)})
    return {"records": len(out), "rows": out, "role": "record-only; never executed and never fed back"}


def reentry_table(study: Path, rows: list[dict], meta: dict) -> dict | None:
    key_path = study / "reentry" / "key.json"
    recs = {p.stem: read_json(p) for p in sorted((study / "reentry" / "records").glob("*.json"))}
    if not recs or not key_path.exists():
        return None
    key = read_json(key_path)
    by_branch = {v["branch_id"]: recs[k] for k, v in key["items"].items() if k in recs}
    out = {}
    for arm in ARMS:
        vals = [(r["task_id"], r["stall_id"], 1.0 if by_branch[r["branch_id"]]["reentry"] == "yes" else 0.0)
                for r in rows if r["arm"] == arm and r["branch_id"] in by_branch
                and by_branch[r["branch_id"]].get("reentry") in ("yes", "no")]
        if not vals:
            continue
        st = defaultdict(list)
        for t, s, v in vals:
            st[(t, s)].append(v)
        out[arm] = {"judged": len(vals), "unclear_or_unparseable": sum(1 for r in rows if r["arm"] == arm and r["branch_id"]
                                                                       in by_branch and by_branch[r["branch_id"]].get("reentry")
                                                                       not in ("yes", "no")),
                    **S.cluster_bootstrap([(t, statistics.fmean(v)) for (t, _), v in st.items()], n=BOOT)}
    return {"by_arm": out, "judge": key.get("judge"), "prompt": key.get("prompt")}


def reestimate(rows: list[dict], order: list[str]) -> dict:
    """The blinded re-estimate: outcomes of the first stalls pooled over arms; arm labels are dropped here."""
    used = {r["stall_id"] for r in rows}
    first = [s for s in order if s in used][:REESTIMATE["after_stalls"]]
    blind = [(r["stall_id"], r["solved"], r["best_share"], r["final_share"]) for r in rows if r["stall_id"] in first]
    out = {"stalls_used": first, "note": "arm labels never reach this computation; extending beyond the planned "
                                         "stalls needs the user's OK"}
    for o in HOLM_OUTCOMES:
        vals = [(s, OUTCOMES[o]({"solved": sv, "best_share_passed": b, "final_share_passed": f})) for s, sv, b, f in blind]
        out[o] = S.blinded_sample_size([(s, v) for s, v in vals if v is not None], repeats_per_arm=MAIN_REPEATS,
                                       target_difference=REESTIMATE["target_difference"][o],
                                       alpha=REESTIMATE["alpha_two_sided"], power=REESTIMATE["power"],
                                       planned_stalls=REESTIMATE["planned_stalls"], max_stalls=REESTIMATE["max_stalls"])
    return out


def load_rows(study: Path, screen_json: Path | None) -> tuple[list[dict], dict, dict, list[str], list[str]]:
    screen = {s["stall_id"]: s for s in (read_json(screen_json)["stalls"] if screen_json and screen_json.exists() else [])}
    rows, meta, missing, sets, order = [], {}, defaultdict(lambda: defaultdict(int)), [], []
    for m_path in sorted((study / "specsets").glob("*.json")):
        m = load_specset(m_path)
        if m["purpose"] != "main":
            continue
        sets.append(m["prefix"])
        for block in m["blocks"]:
            order.append(block["unit_id"])
            meta[block["unit_id"]] = stall_meta(block, screen)
            if meta[block["unit_id"]]["tuning"]:
                raise SystemExit(f"tuning stall {block['unit_id']} found in a main spec set")
        for block, job, res, attempts in iter_results(m, runs_dir(study, m["prefix"])):
            missing[job["arm"]]["planned"] += 1
            if res is None:
                missing[job["arm"]]["errors_only" if attempts else "not_run"] += 1
                continue
            rows.append(branch_row(meta[block["unit_id"]], job, res))
    return rows, meta, missing, sets, order


def analyze(study: Path, screen_json: Path | None, out: Path) -> dict:
    rows, meta, missing, sets, order = load_rows(study, screen_json)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "branches.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    used = {r["stall_id"] for r in rows}
    meta = {k: v for k, v in meta.items() if k in used}
    res: dict = {"version": "analysis.v2", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "synthetic": (study / "SYNTHETIC").exists(), "spec_sets": sets, "branches": len(rows),
                 "stalls": len(meta), "tasks": len({m["task_id"] for m in meta.values()}),
                 "rules": {"covariates": SCREEN_COVARIATES, "long_context": LONG_CONTEXT_RULE, "restall": RESTALL_RULE,
                           "reestimate": REESTIMATE},
                 "missingness": {a: dict(v) for a, v in missing.items()}}
    cov = SCREEN_COVARIATES["covariates"]
    head = {o: contrast(rows, meta, *HEADLINE, o, covariates=cov) for o in OUTCOMES}
    holm = S.holm({o: head[o]["p_two_sided"] for o in HOLM_OUTCOMES})
    for o in HOLM_OUTCOMES:
        head[o]["holm"] = holm[o]
    res["headline"] = head
    res["key_secondaries"] = {f"{a} vs {b}": {o: contrast(rows, meta, a, b, o) for o in OUTCOMES} for a, b in SECONDARIES}
    res["vs_continue"] = {f"{a} vs {b}": {o: contrast(rows, meta, a, b, o) for o in OUTCOMES} for a, b in VS_CONTINUE}
    ctx = [m["context_tokens"] for m in meta.values() if m["context_tokens"] is not None]
    med = statistics.median(ctx) if ctx else None
    splits = {
        "approaches": lambda m: None if m["n_approaches"] is None else ("2+" if m["n_approaches"] >= 2 else "1"),
        "long_context": lambda m: None if med is None or m["context_tokens"] is None else (
            "long" if m["context_tokens"] > med else "not_long"),
        "stratum": lambda m: m["stratum"],
        "first_approach": lambda m: None if m["first_approach"] is None else (
            "first_approach" if m["first_approach"] else "later_approach")}
    res["splits"] = {}
    for name, fn in splits.items():
        levels = defaultdict(set)
        for s, m in meta.items():
            levels[fn(m)].add(s)
        res["splits"][name] = {}
        for lev, ss in sorted(levels.items(), key=lambda kv: str(kv[0])):
            sub = [r for r in rows if r["stall_id"] in ss]
            pairs = (HEADLINE, *SECONDARIES, *VS_CONTINUE) if name != "stratum" else (HEADLINE, *VS_CONTINUE)
            res["splits"][name][str(lev)] = {"stalls": len(ss), "contrasts": {
                f"{a} vs {b}": {o: contrast(sub, meta, a, b, o) for o in HOLM_OUTCOMES} for a, b in dict.fromkeys(pairs)}}
    res["splits"]["long_context_median_tokens"] = med
    res["blinded_reestimate"] = reestimate(rows, order)
    res["choices"] = choice_table(rows)
    recipes = {}
    for m_path in sorted((study / "specsets").glob("*.json")):
        m = load_specset(m_path)
        if m["purpose"] == "main":
            for b in m["blocks"]:
                rc = read_json(Path(b["stall_dir"]) / "recipe.json", None) if b["unit_id"] in used else None
                if rc is not None:
                    recipes[b["unit_id"]] = {"cap_tokens": rc.get("cap_tokens"), "cuts": len(rc.get("cut_log") or [])}
    res["note_and_recipe"] = {"arm5_note_tokens": sorted(r["note_tokens"] for r in rows if r["arm"] == "5_rewind_own_summary"
                                                         and r["note_tokens"] is not None),
                              "arm6_handback_tokens": sorted(r["handback_tokens"] for r in rows
                                                             if r["arm"] == "6_rewind_ledger_recipe"
                                                             and r["handback_tokens"] is not None),
                              "arm6_recipes": recipes}
    res["restalls"] = restall_table(rows)
    res["submission_behaviour"] = submission_table(rows)
    res["costs"] = cost_table(study)
    res["opus_record_only"] = opus_table(study, rows)
    res["reentry"] = reentry_table(study, rows, meta)
    write_json(out / "analysis.json", res)
    (out / "report.md").write_text(report(res))
    return res


def _fmt(c: dict) -> str:
    if not c or c.get("estimate") is None:
        return "n/a"
    ci = c.get("ci")
    p = c.get("p_two_sided")
    return (f"{c['estimate']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "") +
            (f", p={p:.3f}" if p is not None else "") +
            f" ({c.get('stalls', c.get('units'))} stalls / {c.get('tasks', c.get('clusters'))} tasks)")


def report(r: dict) -> str:
    L = ["# v2 analysis", ""]
    if r["synthetic"]:
        L += ["**SYNTHETIC DATA: mechanics test of the analysis, not evidence about the study.**", ""]
    L += [f"Spec sets: {', '.join(r['spec_sets'])}. Branches: {r['branches']}; stalls: {r['stalls']}; tasks: {r['tasks']}.",
          "Paired by stall; 95% intervals resample whole tasks; p-values are two-sided task sign-flip tests.", "",
          "## Headline: arm 2 (Ledger review) vs arm 3 (context review)", "",
          "| outcome | arm 2 | arm 3 | difference [95% CI], p | Holm |", "|---|---|---|---|---|"]
    for o, c in r["headline"].items():
        h = c.get("holm")
        L.append(f"| {o} | {c['mean_a'] if c['mean_a'] is None else round(c['mean_a'], 3)} | "
                 f"{c['mean_b'] if c['mean_b'] is None else round(c['mean_b'], 3)} | {_fmt(c)} | "
                 + (f"p_holm={h['p_holm']:.3f}, reject={h['reject']}" if h and h.get("p_holm") is not None else
                    "not in Holm (final state, reported)" if o == "final_share" else "n/a") + " |")
    adj = r["headline"]["solve"].get("covariate_adjusted") or {}
    L += ["", f"Covariate-adjusted (screen covariates {adj.get('covariates')}, DRAFT): solve "
          f"{_fmt(adj | {'stalls': adj.get('stalls_with_covariates'), 'tasks': adj.get('clusters')}) if adj.get('estimate') is not None else 'n/a'}.",
          "", "## Key secondaries", "", "| contrast | solve | best share | final share |", "|---|---|---|---|"]
    for k, v in r["key_secondaries"].items():
        L.append(f"| {k} | {_fmt(v['solve'])} | {_fmt(v['best_share'])} | {_fmt(v['final_share'])} |")
    L += ["", "## Every arm vs continue (descriptive)", "", "| contrast | solve | best share |", "|---|---|---|"]
    for k, v in r["vs_continue"].items():
        L.append(f"| {k} | {_fmt(v['solve'])} | {_fmt(v['best_share'])} |")
    L += ["", "## Pre-registered splits", ""]
    for name, levels in r["splits"].items():
        if not isinstance(levels, dict):
            continue
        L.append(f"**{name}**")
        for lev, d in levels.items():
            hc = d["contrasts"].get(f"{HEADLINE[0]} vs {HEADLINE[1]}", {})
            vc = d["contrasts"].get("2_ledger_review vs 1_continue", {})
            L.append(f"- {lev} ({d['stalls']} stalls): headline solve {_fmt(hc.get('solve'))}; "
                     f"arm 2 vs continue solve {_fmt(vc.get('solve'))}")
        L.append("")
    br = r["blinded_reestimate"]
    L += ["## Blinded variance re-estimate (pooled variance only; never arm labels)", "",
          f"Stalls used: {len(br['stalls_used'])}. {br['note']}."]
    for o in HOLM_OUTCOMES:
        b = br[o]
        if "no_interaction" in b:
            L.append(f"- {o}: within-stall pooled variance {b['within_stall_pooled_variance']:.4f}; stalls needed "
                     f"{b['no_interaction']['stalls_needed']} (no interaction) to {b['worst_case_interaction']['stalls_needed']} "
                     f"(worst case) for a difference of {b['target_difference']} (DRAFT target); recommended "
                     f"{b['no_interaction']['recommended']}-{b['worst_case_interaction']['recommended']} (max {b['max_stalls']}).")
        else:
            L.append(f"- {o}: {b.get('note')}")
    L += ["", "## Review choices (arms 2 and 3)", "",
          "| arm | branches | no change | invalid call | rewind | median depth | drops | keeps | rescues items | mismatch |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
    for arm, c in r["choices"].items():
        L.append(f"| {arm} | {c['branches']} | {f2(c['no_change_share'])} | {f2(c['invalid_call_share'])} | "
                 f"{f2(c['rewind_share'])} | {c['median_rewind_depth_steps']} | {f2(c['mean_drops'])} | "
                 f"{f2(c['mean_keeps'])} | {f2(c['rescues_items_share'])} | {f2(c['mismatch_share'])} |")
    L += ["", "## Re-stalls inside branches (DRAFT rule; logged only)", "", "| arm | re-stall share | ended on 3 identical failures |",
          "|---|---|---|"]
    for arm, v in r["restalls"].items():
        L.append(f"| {arm} | {f2(v['restall_share'])} | {f2(v['identical_failures_end_share'])} |")
    sb = {k: v for k, v in r.get("submission_behaviour", {}).items() if k != "version"}
    if sb:
        L += ["", "## Submission behaviour (descriptive; means over branches)", "",
              "| arm | branches read | submissions | steps between | resubmitted with no change | own check before | "
              "same failing set |", "|---|---|---|---|---|---|---|"]
        for arm, v in sb.items():
            L.append(f"| {arm} | {v['branches_read']} | {f2(v['submissions'])} | {f2(v['median_steps_between'])} | "
                     f"{f2(v['share_resubmitted_with_no_change'])} | {f2(v['share_with_own_check_before'])} | "
                     f"{f2(v['share_same_failing_set_as_previous'])} |")
    L += ["", "## Costs (with and without caching)", "", "| set | arm | branches | $ cached | $ uncached | error attempts |",
          "|---|---|---|---|---|---|"]
    for prefix, s in r["costs"].items():
        for arm, v in s["arms"].items():
            L.append(f"| {prefix} | {arm} | {v['branches']} | {v['cost_usd']:.4f} | {v['cost_usd_uncached']:.4f}"
                     + (f" ({v['uncached_missing']} missing)" if v["uncached_missing"] else "") + f" | {v['error_attempts']} |")
    if r.get("reentry"):
        L += ["", "## Re-entry into the failed approach (Opus 5.5, blind to arm)", "", "| arm | judged | share re-entering |",
              "|---|---|---|"]
        for arm, v in r["reentry"]["by_arm"].items():
            ci = v.get("ci")
            L.append(f"| {arm} | {v['judged']} | {v['estimate']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                     + f" ({v['units']} stalls / {v['clusters']} tasks) |")
    if r.get("opus_record_only"):
        L += ["", f"Opus record-only review choices recorded: {r['opus_record_only']['records']} (never executed)."]
    L += ["", "## Missing branches", "", "| arm | planned | errors only | not run |", "|---|---|---|---|"]
    for arm, v in r["missingness"].items():
        L.append(f"| {arm} | {v.get('planned', 0)} | {v.get('errors_only', 0)} | {v.get('not_run', 0)} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> dict:
    global ARMS, SECONDARIES, VS_CONTINUE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--arm-set", choices=("v2", "v3"), default="v2")
    ap.add_argument("--screen-json")
    ap.add_argument("--out")
    ap.add_argument("--blinded-reestimate-only", action="store_true",
                    help="after the first 10 stalls: compute and print only the blinded re-estimate (no arm contrasts)")
    a = ap.parse_args(argv)
    if a.arm_set == "v3":
        from study_common import ARMS_V3
        ARMS = ARMS_V3
        SECONDARIES = (("2_ledger_review", "5_ledger_read_only"), ("3_context_review", "6_context_read_only"),
                       ("4_forced_rewind_to_p", "1_continue"), ("2_ledger_review", "4_forced_rewind_to_p"),
                       ("5_ledger_read_only", "6_context_read_only"))
        VS_CONTINUE = tuple((arm, "1_continue") for arm in ARMS[1:])
    study = Path(a.study)
    screen_json = Path(a.screen_json) if a.screen_json else study / "screen" / "screen.json"
    out = Path(a.out) if a.out else study / "analysis"
    if a.blinded_reestimate_only:
        rows, _, _, _, order = load_rows(study, screen_json)
        r = reestimate(rows, order) | {"rule": REESTIMATE}
        write_json(out / "blinded-reestimate.json", r)
        print(json.dumps(r, indent=1))
        return r
    r = analyze(study, screen_json, out)
    print(json.dumps({k: r[k] for k in ("branches", "stalls", "tasks")} |
                     {"headline": {o: {k: r["headline"][o].get(k) for k in ("estimate", "ci", "p_two_sided")}
                                   for o in r["headline"]}}, indent=1))
    return r


if __name__ == "__main__":
    main()
