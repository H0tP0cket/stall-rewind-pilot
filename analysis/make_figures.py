"""Figures for the README and paper, from data/ only. Writes figures/*.pdf and figures/*.png.

Run: python analysis/make_figures.py   (needs matplotlib)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
ROWS = list(csv.DictReader((ROOT / "data/branches.csv").open()))
DECISIONS = json.loads((ROOT / "data/review_decisions.json").read_text())
SCREEN = {(r["stall_id"], r["set"]): r for r in csv.DictReader((ROOT / "data/screen.csv").open())}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL, NEUTRAL_LIGHT = "#9b9a94", "#cfcec8"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2, validated (light)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})

ARMS = [("1_continue", "Continue"), ("2_ledger_review", "Ledger review"), ("3_context_review", "Context review"),
        ("5_ledger_read_only", "Ledger read-only"), ("6_context_read_only", "Context read-only"),
        ("4_forced_rewind_to_p", "Forced rewind to onset")]
STALLS = [("production-planning", "Production planning — stuck", "terminalbench-v4-production-planning--5f6830d3e8--p20"),
          ("atrx-vep-crispr", "ATRX VEP/CRISPR — under-commit", "terminalbench-v4-atrx-vep-crispr--6bd677acf1--p100"),
          ("heat-pump-warranty", "Heat-pump warranty — over-commit",
           "terminalbench-v4-pilot-heatpump-heat-pump-warranty--46457a59cd--p70")]


def solved(rows):
    return sum(r["solved"] == "True" for r in rows), len(rows)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_main():
    by = defaultdict(list)
    for r in ROWS:
        if r["set"] == "pilot_main":
            by[(r["task"], r["arm"])].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
    for ax, (task, title, sid) in zip(axes, STALLS):
        ys = list(range(len(ARMS)))[::-1]
        for y, (arm, label) in zip(ys, ARMS):
            k, n = solved(by[(task, arm)])
            color = BLUE if arm == "4_forced_rewind_to_p" else NEUTRAL
            ax.barh(y, n, height=0.62, color=NEUTRAL_LIGHT if n else SURFACE, linewidth=0)
            if k:
                ax.barh(y, k, height=0.62, color=color, linewidth=0)
            ax.text(n + 0.08, y, f"{k}/{n}", va="center", fontsize=8.5, color=INK)
        c, w = SCREEN[(sid, "continue")], SCREEN[(sid, "rewind")]
        ax.set_title(f"{title}\nscreen: continue {c['solves']}/{c['n']}, rewind {w['solves']}/{w['n']}",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlim(0, 2.6)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks(ys)
        ax.set_yticklabels([a[1] for a in ARMS])
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("branches solved (dark) of branches run (light)")
    save(fig, "fig1_main_arms")


def fig_decisions():
    groups = [("Stalls, neutral review prompt", "pilot_main"), ("Stalls, told “This run looks stalled.”",
                                                                 "followup_stalled_told"),
              ("Healthy runs (harm test)", "pilot_harm")]
    cats = [("Changed nothing", NEUTRAL), ("Dropped context only", BLUE), ("Rewound outputs channel", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.4, 1.9))
    for i, (label, s) in enumerate(groups[::-1]):
        ds = [d["decision"] for d in DECISIONS if d["set"] == s]
        rw = sum(1 for d in ds if any(v is not None for v in (d.get("rewind") or {}).values()))
        nothing = sum(bool(d.get("change_nothing")) for d in ds)
        counts = [nothing, len(ds) - nothing - rw, rw]
        left = 0
        for (cname, color), v in zip(cats, counts):
            if v:
                ax.barh(i, v - 0.06, left=left + 0.03, height=0.6, color=color, linewidth=0)
                if cname != "Changed nothing" or True:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=8,
                            color="white" if color != NEUTRAL_LIGHT else INK)
            left += v
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups[::-1]])
    ax.set_xlabel("review decisions (no review rewound context, files or runtime)")
    ax.tick_params(length=0)
    ax.set_xlim(0, 12.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cats]
    ax.legend(handles, [c[0] for c in cats], loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3, frameon=False,
              fontsize=8)
    save(fig, "fig2_review_decisions")


def fig_injection():
    conds = [("inj_continue", "None (control)"), ("inj_small_neutral", "Small neutral filler"),
             ("inj_stale_output", "Stale output"), ("inj_false_success_claim", "False success claim"),
             ("inj_failed_approach_promising", "Failed approach, before failure"),
             ("inj_workspace_debris", "Leftover files (debris)"),
             ("inj_failed_attempt_failure_shown", "Failed approach, failure shown")]
    rows = [r for r in ROWS if r["set"] == "pilot_injection" and r["task"] == "atrx-vep-crispr"]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ys = list(range(len(conds)))[::-1]
    for y, (c, label) in zip(ys, conds):
        k, n = solved([r for r in rows if r["arm"] == c])
        ax.barh(y, n, height=0.6, color=NEUTRAL_LIGHT, linewidth=0)
        if k:
            ax.barh(y, k, height=0.6, color=NEUTRAL if c != "inj_continue" else INK2, linewidth=0)
        ax.text(n + 0.06, y, f"{k}/{n}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[1] for c in conds])
    ax.set_xlim(0, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Healthy ATRX checkpoint (step 60): solved with injected history", fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("branches solved (dark) of branches run (light)")
    save(fig, "fig3_injection_atrx")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_main()
    fig_decisions()
    fig_injection()
    print("figures written to", FIG)
