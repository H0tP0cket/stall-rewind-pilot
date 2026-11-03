"""Tables for the README and paper, from data/ only. Writes analysis/out/tables.md, analysis/out/summary.json and
paper/tables/*.tex. Standard library only.

Run: python analysis/make_tables.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
TEX = ROOT / "paper/tables"
ROWS = list(csv.DictReader((ROOT / "data/branches.csv").open()))
SCREEN = list(csv.DictReader((ROOT / "data/screen.csv").open()))
DECISIONS = json.loads((ROOT / "data/review_decisions.json").read_text())

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
    md.append("| Stall (type) | Screen: continue | Screen: rewind | " + " | ".join(ARM_NAME[a] for a in ARMS) + " |")
    md.append("|---" * (3 + len(ARMS)) + "|")
    tex1 = []
    summary["main"] = {}
    for task, kind in STALLS:
        c, w = scr[(STALL_ID[task], "continue")], scr[(STALL_ID[task], "rewind")]
        cells = [cell(by[("pilot_main", task, a)]) for a in ARMS]
        md.append(f"| {task} ({kind}) | {c['solves']}/{c['n']} | {w['solves']}/{w['n']} | " + " | ".join(cells) + " |")
        tex1.append(f"{task} ({kind}) & {c['solves']}/{c['n']} & {w['solves']}/{w['n']} & " + " & ".join(cells) + r" \\")
        summary["main"][task] = {a: kn(by[("pilot_main", task, a)]) for a in ARMS}
    tot = [cell([r for t, _ in STALLS for r in by[("pilot_main", t, a)]]) for a in ARMS]
    md.append("| **All three** | | | " + " | ".join(tot) + " |")
    tex1.append(r"\midrule All three & & & " + " & ".join(tot) + r" \\")
    write_tex("main.tex", tex1)

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
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
