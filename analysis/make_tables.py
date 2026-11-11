"""Tables for the README and paper, from data/ only. Writes analysis/out/tables.md, analysis/out/summary.json and
paper/tables/*.tex. Standard library only.

Run: python analysis/make_tables.py
"""
import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
TEX = ROOT / "paper/tables"
ROWS = list(csv.DictReader((ROOT / "data/branches.csv").open()))
SCREEN = list(csv.DictReader((ROOT / "data/screen.csv").open()))
DECISIONS = json.loads((ROOT / "data/review_decisions.json").read_text())
OBSERVER = list(csv.DictReader((ROOT / "data/observer_choices.csv").open()))

ARMS = ["1_continue", "2_ledger_review", "3_context_review", "4_forced_rewind_to_p", "5_ledger_read_only",
        "6_context_read_only"]
ARM_NAME = {"1_continue": "Continue", "2_ledger_review": "Ledger review", "3_context_review": "Context review",
            "4_forced_rewind_to_p": "Forced rewind", "5_ledger_read_only": "Ledger read-only",
            "6_context_read_only": "Context read-only"}
STALLS = [("production-planning", "stuck"), ("atrx-vep-crispr", "under-commit"), ("heat-pump-warranty", "over-commit")]
STALL_ID = {"production-planning": "terminalbench-v4-production-planning--5f6830d3e8--p20",
            "atrx-vep-crispr": "terminalbench-v4-atrx-vep-crispr--6bd677acf1--p100",
            "heat-pump-warranty": "terminalbench-v4-pilot-heatpump-heat-pump-warranty--46457a59cd--p70"}
CONDITIONS = ["inj_continue", "inj_small_neutral", "inj_large_neutral", "inj_stale_output", "inj_false_success_claim",
              "inj_failed_approach_promising", "inj_failed_attempt_failure_shown", "inj_workspace_debris"]
COND_NAME = {"inj_continue": "None (control)", "inj_small_neutral": "Small neutral filler",
             "inj_large_neutral": "Large neutral filler", "inj_stale_output": "Stale output",
             "inj_false_success_claim": "False success claim",
             "inj_failed_approach_promising": "Failed approach, before failure",
             "inj_failed_attempt_failure_shown": "Failed approach, failure shown",
             "inj_workspace_debris": "Leftover files (debris)"}
HEALTHY = [("production-planning", "production p10"), ("atrx-vep-crispr", "ATRX p60"), ("mvcc-lsm-compaction", "mvcc p20")]
CHOOSERS = [("worker", "The worker itself (its actual choice)"),
            ("deepseek_resample", "Worker model, asked again with the same request"),
            ("deepseek_outside_observer", "Worker model, asked as an outside observer"),
            ("opus", "Opus 5.5, asked as an outside observer")]
CHOICE_KINDS = ["changed nothing", "trimmed context only", "rewound outputs only",
                "rewound context, files or runtime", "no usable answer"]


def fisher_two_sided(k1, n1, k2, n2):
    """Two-sided Fisher exact p for k1/n1 against k2/n2 (sum of all tables no more likely than the observed one)."""
    k, n = k1 + k2, n1 + n2

    def prob(x):
        return comb(n1, x) * comb(n2, k - x) / comb(n, k)
    p0 = prob(k1)
    return sum(prob(x) for x in range(max(0, k - n2), min(k, n1) + 1) if prob(x) <= p0 * (1 + 1e-9))


def choice_kind(r):
    """One of CHOICE_KINDS for a row of observer_choices.csv."""
    if r["usable_answer"] != "True":
        return "no usable answer"
    if "True" in (r["rewound_context"], r["rewound_files"], r["rewound_runtime"]):
        return "rewound context, files or runtime"
    if r["rewound_outputs_only"] == "True":
        return "rewound outputs only"
    if int(r["n_drops"]) + int(r["n_keeps"]) > 0:
        return "trimmed context only"
    return "changed nothing"


def kn(rows):
    return sum(r["solved"] == "True" for r in rows), len(rows)


def cell(rows):
    k, n = kn(rows)
    return f"{k}/{n}" if n else "--"


HEADS = {'main.tex': ('\\begin{tabular}{@{}lcc|cccccc@{}}\n\\toprule\n& \\multicolumn{2}{c|}{Screen} & & Ledger & Context & Forced & Ledger & Context \\\\\nStall (type) & cont. & rewind & Continue & review & review & rewind & read-only & read-only \\\\\n\\midrule', '\\bottomrule\n\\end{tabular}'), 'followup.tex': ('\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nStall & Ledger review & Ledger, told stalled & Context review & Context, told stalled \\\\\n\\midrule', '\\bottomrule\n\\end{tabular}'), 'injection.tex': ('\\begin{tabular}{@{}lccc@{}}\n\\toprule\nInserted content & production p10 & ATRX p60 & mvcc p20 \\\\\n\\midrule', '\\bottomrule\n\\end{tabular}')}


def write_tex(name, rows):
    head, tail = HEADS.get(name, ("", ""))
    (TEX / name).write_text("\n".join([head, *rows, tail]).strip() + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    md, summary = [], {}
    by = defaultdict(list)
    for r in ROWS:
        by[(r["set"], r["task"], r["arm"])].append(r)

    # Table 1: main arms per stall, with the screen
    scr = {(s["stall_id"], s["set"]): s for s in SCREEN}
    md.append("### Main test: solved / branches, per stall and arm\n")
    extra = [("followup_anytime_review", "7_ledger_review_anytime", "Any-time review (arm 7, follow-up)"),
             ("followup_opus_decides", "2_ledger_review_opus_decides", "Ledger review, Opus decides (follow-up)")]
    md.append("| Stall (type) | Screen: continue | Screen: rewind | " + " | ".join(ARM_NAME[a] for a in ARMS) + " | "
              + " | ".join(e[2] for e in extra) + " |")
    md.append("|---" * (3 + len(ARMS) + len(extra)) + "|")
    tex1 = []
    summary["main"] = {}
    summary["followup_2026_11_04"] = {}
    for task, kind in STALLS:
        c, w = scr[(STALL_ID[task], "continue")], scr[(STALL_ID[task], "rewind")]
        cells = [cell(by[("pilot_main", task, a)]) for a in ARMS]
        ecells = [cell(by[(s_, task, a)]) for s_, a, _ in extra]
        md.append(f"| {task} ({kind}) | {c['solves']}/{c['n']} | {w['solves']}/{w['n']} | " + " | ".join(cells + ecells)
                  + " |")
        tex1.append(f"{task} ({kind}) & {c['solves']}/{c['n']} & {w['solves']}/{w['n']} & " + " & ".join(cells) + r" \\")
        summary["main"][task] = {a: kn(by[("pilot_main", task, a)]) for a in ARMS}
        summary["followup_2026_11_04"][task] = {a: kn(by[(s_, task, a)]) for s_, a, _ in extra}
    tot = [cell([r for t, _ in STALLS for r in by[("pilot_main", t, a)]]) for a in ARMS]
    etot = [cell([r for t, _ in STALLS for r in by[(s_, t, a)]]) for s_, a, _ in extra]
    md.append("| **All three** | | | " + " | ".join(tot + etot) + " |")
    tex1.append(r"\midrule All three & & & " + " & ".join(tot) + r" \\")
    write_tex("main.tex", tex1)

    # Table 1b: continue vs forced rewind, the screen and the arms pooled
    md.append("\n### Continue vs forced rewind, screen and arms pooled\n")
    count_scr = [t for t, _ in STALLS if scr[(STALL_ID[t], "continue")]["feedback"].startswith("count line")]
    neutral_scr = [t for t, _ in STALLS if scr[(STALL_ID[t], "continue")]["feedback"].startswith("neutral line")]
    md.append("The arms all used the neutral feedback line (\"The task is not complete yet. Continue working.\"). "
              f"The screens of {' and '.join(count_scr)} used the count line (\"Hidden tests still failing: N.\"), "
              f"so pooling mixes the two lines for those stalls; the screen of {' and '.join(neutral_scr)} used the "
              "neutral line.\n")
    md.append("| Stall (type) | Stall point (step) | Forced rewind restores (step) | Steps undone | "
              "Screen feedback line | Continue: screen | Continue: arms (neutral line) | Continue pooled | "
              "Forced rewind: screen | Forced rewind: arms (neutral line) | Forced rewind pooled | "
              "Fisher exact p, two-sided |")
    md.append("|---" * 12 + "|")
    summary["pooled_continue_vs_rewind"] = {}
    for task, kind in STALLS:
        c, w = scr[(STALL_ID[task], "continue")], scr[(STALL_ID[task], "rewind")]
        ck, cn = kn(by[("pilot_main", task, "1_continue")])
        wk, wn = kn(by[("pilot_main", task, "4_forced_rewind_to_p")])
        pc = (int(c["solves"]) + ck, int(c["n"]) + cn)
        pw = (int(w["solves"]) + wk, int(w["n"]) + wn)
        pval = fisher_two_sided(*pc, *pw)
        rw = by[("pilot_main", task, "4_forced_rewind_to_p")][0]
        point, back = int(rw["branch_point"]), int(rw["forced_rewind_to"])
        line = c["feedback"].split(" (")[0]
        assert line == w["feedback"].split(" (")[0]
        md.append(f"| {task} ({kind}) | {point} | {back} | {point - back} | {line} | {c['solves']}/{c['n']} | {ck}/{cn} | "
                  f"{pc[0]}/{pc[1]} | {w['solves']}/{w['n']} | {wk}/{wn} | {pw[0]}/{pw[1]} | {pval:.3f} |")
        summary["pooled_continue_vs_rewind"][task] = {"stall_point": point, "rewind_restores": back,
                                                      "screen_feedback": line,
                                                      "continue": pc, "forced_rewind": pw, "fisher_p": round(pval, 4)}

    # Table 2: follow-up (stalled-told reviews)
    md.append("\n### Follow-up: reviews told \"This run looks stalled.\"\n")
    md.append("| Stall | Ledger review (pilot) | Ledger review (told stalled) | Context review (pilot) | "
              "Context review (told stalled) |")
    md.append("|---|---|---|---|---|")
    tex2 = []
    for task, kind in STALLS:
        cs = [cell(by[("pilot_main", task, "2_ledger_review")]), cell(by[("followup_stalled_told", task, "2_ledger_review")]),
              cell(by[("pilot_main", task, "3_context_review")]), cell(by[("followup_stalled_told", task, "3_context_review")])]
        md.append(f"| {task} | " + " | ".join(cs) + " |")
        tex2.append(f"{task} ({kind}) & " + " & ".join(cs) + r" \\")
    write_tex("followup.tex", tex2)

    # Table 3: review decisions
    md.append("\n### What the review turn chose\n")
    md.append("| Where | Reviews | Changed nothing | Dropped context only | Rewound a channel | Rewound context or files |")
    md.append("|---|---|---|---|---|---|")
    tex3 = []
    summary["decisions"] = {}
    for label, sel in (("Stalls, neutral prompt", lambda d: d["set"] == "pilot_main"),
                       ("Stalls, told stalled", lambda d: d["set"] == "followup_stalled_told"),
                       ("Healthy runs (harm test)", lambda d: d["set"] == "pilot_harm")):
        ds = [d for d in DECISIONS if sel(d)]
        nothing = sum(bool(d["decision"].get("change_nothing")) for d in ds)
        rw = [d for d in ds if any(v is not None for v in (d["decision"].get("rewind") or {}).values())]
        rw_cf = [d for d in rw if any((d["decision"]["rewind"] or {}).get(ch) is not None
                                      for ch in ("context", "workspace", "runtime"))]
        drop_only = sum(1 for d in ds if not d["decision"].get("change_nothing") and d not in rw)
        md.append(f"| {label} | {len(ds)} | {nothing} | {drop_only} | {len(rw)} | {len(rw_cf)} |")
        tex3.append(f"{label} & {len(ds)} & {nothing} & {drop_only} & {len(rw)} & {len(rw_cf)} " + r"\\")
        summary["decisions"][label] = {"n": len(ds), "change_nothing": nothing, "drop_only": drop_only,
                                       "rewound_any": len(rw), "rewound_context_or_files": len(rw_cf)}
    write_tex("decisions.tex", tex3)
    stall_d = [summary["decisions"][k] for k in ("Stalls, neutral prompt", "Stalls, told stalled")]
    n_rev = sum(d["n"] for d in stall_d)
    md.append(f"\nAcross the {n_rev} one-shot reviews at stalls ({stall_d[0]['n']} with the neutral prompt, "
              f"{stall_d[1]['n']} told the run looks stalled): {sum(d['change_nothing'] for d in stall_d)} of {n_rev} "
              f"changed nothing, {sum(d['drop_only'] for d in stall_d)} dropped context only, "
              f"{sum(d['rewound_any'] - d['rewound_context_or_files'] for d in stall_d)} rewound the outputs channel "
              f"only, and {sum(d['rewound_context_or_files'] for d in stall_d)} of {n_rev} rewound context, files or "
              "runtime.")

    # Table 3b: the any-time review (arm 7), from the worker rows of observer_choices.csv
    md.append("\n### Follow-up: any-time review (arm 7), every opening of the review\n")
    md.append("| Stall | Branches solved | Openings | " + " | ".join(k.capitalize() for k in CHOICE_KINDS[:4]) + " |")
    md.append("|---" * 7 + "|")
    summary["anytime_openings"] = {}
    opens = [r for r in OBSERVER if r["chooser"] == "worker" and r["arm"] == "7_ledger_review_anytime"]
    for task, _ in STALLS + [("all", "")]:
        rs = [r for r in opens if task in ("all", r["task"])]
        counts = [sum(choice_kind(r) == k for r in rs) for k in CHOICE_KINDS[:4]]
        solved = cell([r for t, _ in STALLS if task in ("all", t)
                       for r in by[("followup_anytime_review", t, "7_ledger_review_anytime")]])
        name = "**All three**" if task == "all" else task
        md.append(f"| {name} | {solved} | {len(rs)} | " + " | ".join(map(str, counts)) + " |")
        summary["anytime_openings"][task] = {"openings": len(rs)} | dict(zip(CHOICE_KINDS[:4], counts))
    rew = [r for r in opens if choice_kind(r) == "rewound context, files or runtime"]
    md.append(f"\n{len(rew)} of {len(opens)} openings rewound context, files or runtime ("
              + ", ".join(sorted({r["task"] for r in rew})) + ").")

    # Table 3c: who rewinds, given the same review request
    md.append("\n### Who rewinds: every answer to the same review requests\n")
    reqs = {r["request_id"]: r["unit_kind"] for r in OBSERVER}
    n_stall, n_healthy = sum(v == "stall" for v in reqs.values()), sum(v == "healthy" for v in reqs.values())
    md.append(f"{len(reqs)} review requests ({n_stall} at stalls, {n_healthy} at healthy checkpoints): the "
              f"{sum(r['chooser'] == 'worker' and r['arm'] != '7_ledger_review_anytime' for r in OBSERVER)} one-shot "
              f"review turns share {len({r['request_id'] for r in OBSERVER if r['arm'] != '7_ledger_review_anytime'})} "
              f"requests, and each of the {len(opens)} any-time openings is its own request.\n")
    summary["who_rewinds"] = {}
    for scope, sel in (("All requests", lambda r: True), ("Requests at stalls", lambda r: r["unit_kind"] == "stall"),
                       ("Requests at healthy checkpoints", lambda r: r["unit_kind"] == "healthy")):
        md.append(f"**{scope}**\n")
        md.append("| Who answers | Answers | " + " | ".join(k.capitalize() for k in CHOICE_KINDS)
                  + " | Share that rewound | Answers with a call error |")
        md.append("|---" * 9 + "|")
        summary["who_rewinds"][scope] = {}
        for key, label in CHOOSERS:
            rs = [r for r in OBSERVER if r["chooser"] == key and sel(r)]
            counts = [sum(choice_kind(r) == k for r in rs) for k in CHOICE_KINDS]
            errs = sum(r["valid"] != "True" for r in rs)
            share = f"{100 * counts[3] / len(rs):.0f}% ({counts[3]}/{len(rs)})" if rs else "--"
            md.append(f"| {label} | {len(rs)} | " + " | ".join(map(str, counts)) + f" | {share} | {errs}/{len(rs)} |")
            summary["who_rewinds"][scope][key] = {"answers": len(rs), "call_errors": errs} | dict(zip(CHOICE_KINDS, counts))
        md.append("")
    opus = [r for r in OBSERVER if r["chooser"] == "opus"]
    changed = sum(choice_kind(r) not in ("changed nothing", "no usable answer") for r in opus)
    md.append(f"Opus changed something at {changed} of {len(opus)} requests. "
              f"{sum(r['input_variant'] == 'notes' for r in opus)} of its {len(opus)} answers were made from a copy of "
              "the request in which the word \"reasoning\" was replaced by \"notes\" (column input_variant).")
    ds_cost = sum(float(r["cost_usd"] or 0) for r in OBSERVER)
    md.append(f"The {sum(r['chooser'].startswith('deepseek') for r in OBSERVER)} extra worker-model answers cost "
              f"${ds_cost:.2f}; none of these answers was executed.")
    summary["observer_cost_usd"] = round(ds_cost, 2)

    # Table 4: injection
    md.append("\n### Injection test: solved / branches on healthy checkpoints\n")
    md.append("| Injected content | " + " | ".join(h for _, h in HEALTHY) + " |")
    md.append("|---" * (1 + len(HEALTHY)) + "|")
    tex4 = []
    summary["injection"] = {}
    for c in CONDITIONS:
        cs = [cell(by[("pilot_injection", t, c)]) for t, _ in HEALTHY]
        md.append(f"| {COND_NAME[c]} | " + " | ".join(cs) + " |")
        tex4.append(f"{COND_NAME[c]} & " + " & ".join(cs) + r" \\")
        summary["injection"][c] = {h: kn(by[("pilot_injection", t, c)]) for t, h in HEALTHY}
    write_tex("injection.tex", tex4)
    fk, fn = kn(by[("pilot_injection", "atrx-vep-crispr", "inj_failed_attempt_failure_shown")])
    ck_, cn_ = kn(by[("pilot_injection", "atrx-vep-crispr", "inj_continue")])
    p_inj = fisher_two_sided(fk, fn, ck_, cn_)
    md.append(f"\nATRX p60, failed approach with its failure shown against the control: {fk}/{fn} vs {ck_}/{cn_}, "
              f"Fisher exact p = {p_inj:.2f} (two-sided).")
    summary["injection_failure_shown_vs_control_atrx"] = {"failure_shown": (fk, fn), "control": (ck_, cn_),
                                                          "fisher_p": round(p_inj, 4)}

    # Table 5: harm
    md.append("\n### Harm test: Ledger review on healthy checkpoints\n")
    md.append("| Healthy checkpoint | Ledger review solved | No-injection control solved |")
    md.append("|---|---|---|")
    tex5 = []
    for t, h in HEALTHY:
        cs = [cell(by[("pilot_harm", t, "2_ledger_review")]), cell(by[("pilot_injection", t, "inj_continue")])]
        md.append(f"| {h} | " + " | ".join(cs) + " |")
        tex5.append(f"{h} & " + " & ".join(cs) + r" \\")
    write_tex("harm.tex", tex5)

    # cost
    cost = defaultdict(float)
    for r in ROWS:
        cost[r["set"]] += float(r["cost_usd"])
    summary["cost_usd"] = {k: round(v, 2) for k, v in cost.items()} | {"total": round(sum(cost.values()), 2),
                                                                         "branches": len(ROWS)}
    md.append(f"\nCost: ${summary['cost_usd']['total']} for {len(ROWS)} branches (" +
              ", ".join(f"{k} ${v:.2f}" for k, v in sorted(cost.items())) + ").")
    n_set = defaultdict(int)
    for r in ROWS:
        n_set[r["set"]] += 1
    later = sum(v for k, v in n_set.items() if k in ("followup_anytime_review", "followup_opus_decides"))
    md.append("\nBranches per set: " + ", ".join(f"{k} {v}" for k, v in sorted(n_set.items()))
              + f". {len(ROWS) - later} branches ran on 2026-11-03 and {later} follow-up branches on 2026-11-04.")
    summary["branches_per_set"] = dict(n_set)
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
