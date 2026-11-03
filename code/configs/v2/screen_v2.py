"""Screen aggregation, admission and strata (track S). No model calls.

Per candidate stall the screen ran 5 continue (arm 1) and 5 forced-rewind (arm 4) branches on fresh seeds.
Admission (ratified): a stall is excluded when continue solves 5 of 5 (counted), otherwise admitted when either set
solves at least once. Admission uses the first 5 repeats of each set only; extra resamples refine strata, never
admission. Strata (rewind better / similar / continue better) use every screen branch of the stall.

The strata rule and the near-cutoff rule are not decided: they are DRAFT constants below. Nothing built on them may
run arms until the user ratifies them (arms_v2 and driver_v2 refuse DRAFT rules without --allow-draft).

  python configs/v2/screen_v2.py --study R --stalls-root D [--tuning-stall ID ...] [--out R/screen]
writes screen.json (per stall: counts, admission, stratum, near-cutoff flag, covariates for the analysis, the frozen
boundary hash) and screen-report.md, including "how much undoing the current approach helps" with intervals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from study_common import (ROOT, SCREEN_ARMS, SCREEN_REPEATS, draft_rules, iter_results, load_specset, load_stall,
                          read_json, runs_dir, stall_dirs, write_json)
from trajectory_retention.analysis import v2_stats as S

ADMISSION_RULE = {"name": "screen_admission", "status": "ratified (contract: Screen and strata)",
                  "rule": "exclude if continue solves 5/5; else admit if continue or forced rewind solves >= 1 of 5"}
STRATA_RULE = {"name": "strata_rule", "status": "DRAFT, needs the user's ratification before any arm runs",
               "solve_margin": 0.4, "pass_margin": 0.15,
               "rule": "d = rewind solve rate - continue solve rate over all screen branches; dp = mean best share "
                       "passed, rewind - continue. rewind_better if d >= solve_margin, continue_better if "
                       "d <= -solve_margin; otherwise rewind_better if dp >= pass_margin, continue_better if "
                       "dp <= -pass_margin, else similar"}
NEAR_CUTOFF_RULE = {"name": "near_cutoff_rule", "status": "DRAFT, needs the user's ratification before any arm runs",
                    "pass_band": 0.05, "extra_resamples": 5, "max_extra_rounds": 1,
                    "rule": "near a cutoff if flipping one screen branch's solve outcome (either set) changes the "
                            "stratum, or the solve rule does not decide and dp is within pass_band of +/-pass_margin; "
                            "such stalls get extra_resamples more branches per set, at most max_extra_rounds rounds"}
ARM_NAME = {"1_continue": "continue", "4_forced_rewind": "rewind"}


def stratum(d: float, dp: float | None) -> str:
    m, pm = STRATA_RULE["solve_margin"], STRATA_RULE["pass_margin"]
    if d >= m - 1e-9:
        return "rewind_better"
    if d <= -m + 1e-9:
        return "continue_better"
    if dp is not None and dp >= pm - 1e-9:
        return "rewind_better"
    if dp is not None and dp <= -pm + 1e-9:
        return "continue_better"
    return "similar"


def near_cutoff(cs: int, cn: int, rs: int, rn: int, dp: float | None) -> list[str]:
    base = stratum(rs / rn - cs / cn, dp)
    why = []
    for name, (a, b, c, d) in {"continue +1 solve": (cs + 1, cn, rs, rn), "continue -1 solve": (cs - 1, cn, rs, rn),
                               "rewind +1 solve": (cs, cn, rs + 1, rn), "rewind -1 solve": (cs, cn, rs - 1, rn)}.items():
        if 0 <= a <= b and 0 <= c <= d and stratum(c / d - a / b, dp) != base:
            why.append(f"{name} -> {stratum(c / d - a / b, dp)}")
    d = rs / rn - cs / cn
    pm, band = STRATA_RULE["pass_margin"], NEAR_CUTOFF_RULE["pass_band"]
    if abs(d) < STRATA_RULE["solve_margin"] - 1e-9 and dp is not None and min(abs(dp - pm), abs(dp + pm)) < band:
        why.append(f"pass-rate difference {dp:+.3f} within {band} of +/-{pm}")
    return why


def collect(study: Path) -> tuple[dict, list[str]]:
    """{stall_id: {arm: [(repeat, result)]}} over every screen spec set, plus the sets read."""
    by: dict = {}
    sets = []
    for m_path in sorted((study / "specsets").glob("screen*.json"), key=lambda p: (p.stem != "screen", p.stem)):
        m = load_specset(m_path)
        if m["purpose"] != "screen":
            continue
        sets.append(m["prefix"])
        for block, job, res, attempts in iter_results(m, runs_dir(study, m["prefix"])):
            row = by.setdefault(block["unit_id"], {"task_id": block["task_id"], "boundary_hashes": set(),
                                                   "arms": {a: [] for a in SCREEN_ARMS}, "errors": 0})
            row["boundary_hashes"].add(block["inputs"].get("boundary.json"))
            if res is None:
                row["errors"] += any(x["status"] in ("error", "runner_failed") for x in attempts)
                continue
            row["arms"][job["arm"]].append((job["repeat"], res))
    return by, sets


def arm_stats(rows: list[tuple[int, dict]]) -> dict:
    return {"n": len(rows), "solves": sum(bool(r["solved"]) for _, r in rows),
            "best_share_mean": S.mean(r.get("best_share_passed") for _, r in rows),
            "final_share_mean": S.mean(r.get("final_share_passed") for _, r in rows),
            "no_submission": sum(1 for _, r in rows if not r.get("submissions"))}


def run(study: Path, stalls_root: Path, tuning: list[str], out: Path) -> dict:
    by, sets = collect(study)
    meta = {s["stall_id"]: s for s in (load_stall(d) for d in stall_dirs(stalls_root))}
    stalls, pairs_adm, pairs_all, dp_adm, dp_all = [], [], [], [], []
    for sid, row in by.items():
        st = meta.get(sid, {})
        primary = {a: [x for x in row["arms"][a] if x[0] < SCREEN_REPEATS] for a in SCREEN_ARMS}
        allr = row["arms"]
        complete = all(len(primary[a]) == SCREEN_REPEATS for a in SCREEN_ARMS)
        p = {ARM_NAME[a]: arm_stats(primary[a]) for a in SCREEN_ARMS}
        al = {ARM_NAME[a]: arm_stats(allr[a]) for a in SCREEN_ARMS}
        changed = len(row["boundary_hashes"]) > 1
        rec = {"stall_id": sid, "task_id": row["task_id"], "tuning": sid in tuning,
               "boundary_sha256": None if changed else next(iter(row["boundary_hashes"])),
               "boundary_changed_during_screen": changed, "first_approach": None, "later_solved": st.get("later_solved"),
               "context_tokens_at_stall": st.get("context_tokens_at_stall"), "primary": p, "all": al,
               "status": "complete" if complete else "incomplete", "errors_without_result": row["errors"]}
        if st.get("has_boundary"):
            rec["first_approach"] = read_json(Path(st["dir"]) / "boundary.json").get("rewind_point") == 0
        if complete:
            excl = p["continue"]["solves"] == SCREEN_REPEATS
            adm = not excl and (p["continue"]["solves"] > 0 or p["rewind"]["solves"] > 0)
            c, r = al["continue"], al["rewind"]
            d = r["solves"] / r["n"] - c["solves"] / c["n"]
            dp = (None if r["best_share_mean"] is None or c["best_share_mean"] is None
                  else r["best_share_mean"] - c["best_share_mean"])
            near = near_cutoff(c["solves"], c["n"], r["solves"], r["n"], dp)
            rounds_done = max(0, (max(c["n"], r["n"]) - SCREEN_REPEATS) // NEAR_CUTOFF_RULE["extra_resamples"])
            rec |= {"admitted": adm, "excluded_continue_5of5": excl, "no_solve": not excl and not adm,
                    "stratum": stratum(d, dp), "stratum_inputs": {"d_solve_rate": d, "d_best_share": dp},
                    "near_cutoff": bool(near) and adm, "near_reasons": near,
                    "extra_resamples": (NEAR_CUTOFF_RULE["extra_resamples"] if near and adm and
                                        rounds_done < NEAR_CUTOFF_RULE["max_extra_rounds"] else 0),
                    "covariates": {"screen_continue_solve_rate": c["solves"] / c["n"],
                                   "screen_rewind_solve_rate": r["solves"] / r["n"],
                                   "screen_continue_best_share": c["best_share_mean"],
                                   "screen_rewind_best_share": r["best_share_mean"]}}
            pair = (c["solves"], c["n"], r["solves"], r["n"])
            pairs_all.append(pair)
            if dp is not None:
                dp_all.append((row["task_id"], dp))
            if adm and not rec["tuning"]:
                pairs_adm.append(pair)
                if dp is not None:
                    dp_adm.append((row["task_id"], dp))
        stalls.append(rec)
    done = [s for s in stalls if s["status"] == "complete"]
    helps = {}
    for name, pairs, dps in (("admitted_main_stalls", pairs_adm, dp_adm), ("all_complete_screened", pairs_all, dp_all)):
        helps[name] = {"solve_rate_difference_beta": S.beta_difference(pairs),
                       "solve_rate_difference_hierarchical": S.hierarchical_difference(pairs),
                       "best_share_difference_task_bootstrap": S.cluster_bootstrap(dps)}
    res = {"version": "screen.v2", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "spec_sets": sets, "admission_rule": ADMISSION_RULE, "strata_rule": STRATA_RULE,
           "near_cutoff_rule": NEAR_CUTOFF_RULE, "draft_rules_used": draft_rules(STRATA_RULE, NEAR_CUTOFF_RULE),
           "tuning_stalls": tuning,
           "counts": {"screened": len(stalls), "complete": len(done), "incomplete": len(stalls) - len(done),
                      "admitted": sum(bool(s.get("admitted")) for s in done),
                      "admitted_main": sum(bool(s.get("admitted")) and not s["tuning"] for s in done),
                      "excluded_continue_5of5": sum(bool(s.get("excluded_continue_5of5")) for s in done),
                      "no_solve_in_either_set": sum(bool(s.get("no_solve")) for s in done),
                      "near_cutoff": sum(bool(s.get("near_cutoff")) for s in done),
                      "strata": {k: sum(s.get("stratum") == k and bool(s.get("admitted")) for s in done)
                                 for k in ("rewind_better", "similar", "continue_better")}},
           "undoing_the_current_approach_helps": helps, "stalls": stalls}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "screen.json", res)
    (out / "screen-report.md").write_text(report(res, synthetic=(study / "SYNTHETIC").exists()))
    return res


def _ci(x: dict) -> str:
    if not x or x.get("estimate") is None:
        return "n/a"
    ci = x.get("ci")
    return f"{x['estimate']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else " (no interval)")


def report(r: dict, synthetic: bool) -> str:
    L = ["# Screen report", ""]
    if synthetic:
        L += ["**SYNTHETIC DATA: mechanics test of the study machinery, not evidence.**", ""]
    c = r["counts"]
    L += [f"Spec sets: {', '.join(r['spec_sets'])}. Stalls screened: {c['screened']} ({c['complete']} complete).",
          f"Admitted: {c['admitted']} (main-eligible, not tuning: {c['admitted_main']}). Excluded because continue "
          f"solved 5/5: {c['excluded_continue_5of5']}. No solve in either set: {c['no_solve_in_either_set']}. "
          f"Near a strata cutoff: {c['near_cutoff']}.", "",
          f"Strata rule ({r['strata_rule']['status']}): {r['strata_rule']['rule']}.", "",
          "| stall | task | continue solves | rewind solves | best share c / r | admitted | stratum | near cutoff |",
          "|---|---|---|---|---|---|---|---|"]
    for s in r["stalls"]:
        if s["status"] != "complete":
            L.append(f"| {s['stall_id']} | {s['task_id']} | incomplete | | | | | |")
            continue
        a = s["all"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        L.append(f"| {s['stall_id']}{' (tuning)' if s['tuning'] else ''} | {s['task_id']} | "
                 f"{a['continue']['solves']}/{a['continue']['n']} | {a['rewind']['solves']}/{a['rewind']['n']} | "
                 f"{fmt(a['continue']['best_share_mean'])} / {fmt(a['rewind']['best_share_mean'])} | "
                 f"{'yes' if s['admitted'] else 'excluded (continue 5/5)' if s['excluded_continue_5of5'] else 'no'} | "
                 f"{s['stratum']} | {'; '.join(s['near_reasons']) or 'no'} |")
    L += ["", "## How much undoing the current approach helps", "",
          "Rewind (arm 4) minus continue (arm 1) on the screen branches. Admission leans toward stalls where "
          "rewinding works, so the all-screened row is the less selected one.", "",
          "| population | solve rate, Beta per stall | solve rate, hierarchical | best share passed (task bootstrap) |",
          "|---|---|---|---|"]
    for name, h in r["undoing_the_current_approach_helps"].items():
        L.append(f"| {name} | {_ci(h['solve_rate_difference_beta'])} | {_ci(h['solve_rate_difference_hierarchical'])} "
                 f"| {_ci(h['best_share_difference_task_bootstrap'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> dict:
    global SCREEN_ARMS, ARM_NAME
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--arm-set", choices=("v2", "v3"), default="v2")
    ap.add_argument("--stalls-root", required=True)
    ap.add_argument("--tuning-stall", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.arm_set == "v3":
        SCREEN_ARMS = ("1_continue", "4_forced_rewind_to_p")
        ARM_NAME = dict(zip(SCREEN_ARMS, ("continue", "rewind")))
        STRATA_RULE["status"] = "ratified: Q32-Q37, two solves and 15 percentage points"
        NEAR_CUTOFF_RULE["status"] = "ratified: Q32-Q37, one five-per-arm top-up near cutoff"
    study = Path(a.study)
    r = run(study, Path(a.stalls_root), a.tuning_stall, Path(a.out) if a.out else study / "screen")
    print(json.dumps(r["counts"], indent=1))
    return r


if __name__ == "__main__":
    main()
