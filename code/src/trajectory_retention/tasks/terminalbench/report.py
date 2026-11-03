"""Generate the README result tables from the committed lane evidence (no Docker, no model calls)."""
from __future__ import annotations

import json
from pathlib import Path

from .catalog import STALL_RATIONALE
from .package import ADMISSION, PACKAGES, WORKTREE

README = WORKTREE / "docs/research/terminalbench-admission/README.md"
BEGIN, END = "<!-- BEGIN GENERATED RESULTS -->", "<!-- END GENERATED RESULTS -->"


def _fmt(x, nd=2):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def render() -> str:
    ledger = json.loads((ADMISSION / "ledger.json").read_text())["tasks"]
    registry = json.loads((ADMISSION / "registry.json").read_text())["tasks"]
    admitted = {t["upstream_task"]: t for t in registry}
    lines = [BEGIN, "", "## Result", "",
             f"- **Admitted (calibration split): {len(admitted)} tasks in "
             f"{len({t['family_id'] for t in registry})} families**; members of "
             "`configs/calibration/cohort-terminalbench-v1.json`.",
             f"- Held (calibration split, recorded reason): {sum(r['status'] == 'HOLD' for r in ledger)}.", "",
             "| Task | Family | Checks | Start | Partial (control) | Reference | Grade s (start / ref) | "
             "Solve s | W start | python3 added | Native metric at partial |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for task in sorted(admitted):
        entry = admitted[task]
        v = json.loads((ADMISSION / "tasks" / entry["task_id"] / "validation.json").read_text())
        m = json.loads((PACKAGES / entry["task_id"] / "package.json").read_text())
        g = v["grades"]
        partial = g.get("partial")
        metric = {k: val for k, val in ((partial or {}).get("native_metrics") or {}).items() if val is not None}
        lines.append("| {t} | {f} | {c} | {s} | {p} | {r} | {gs} / {gr} | {solve} | {w} | {py} | {met} |".format(
            t=task, f=entry["family_id"].split(":", 1)[1], c=len(v["grades"]["start_1"]["statuses"]),
            s=_fmt(g["start_1"]["score"]), p=_fmt(partial["score"]) if partial else "binary-only",
            r=_fmt(g["reference_1"]["score"]), gs=_fmt(g["start_1"]["grade_wall_seconds"], 1),
            gr=_fmt(g["reference_1"]["grade_wall_seconds"], 1), solve=_fmt(v["reference_run"]["seconds"], 2),
            w=_size(m["native_start"]["regular_bytes"]), py=m["environment"]["worker"]["python3_added"],
            met=", ".join(f"{k}={val:.4g}" for k, val in metric.items())))
    lines += ["", "Every admitted task passed every gate (start ×2 identical, reference ×2 full and identical, "
              "partial strictly between where declared, tamper = start, no lane-introduced leak, runner restore "
              "regrade identical, grade ≤ 600 s). Grade times are wall seconds of one grading container on this "
              "Mac (Docker Desktop 6 CPUs / 16 GB, shared with other lanes).", "",
              "### Holds and not-packaged tasks", "", "| Task | Split | Status | Reason (measured) |", "|---|---|---|---|"]
    for row in sorted(ledger, key=lambda r: (r["status"], r["task"])):
        if row["status"] == "VALIDATED":
            continue
        lines.append(f"| {row['task']} | {row['split']} | {row['status']} | {row.get('reason') or ''} |")
    lines += ["", "### Stall rationale (published evidence; see `catalog.STALL_RATIONALE`)", ""]
    for task in sorted(admitted):
        r = STALL_RATIONALE.get(task) or {}
        pub = "; ".join(f"{k} {v}" for k, v in (r.get("published") or {}).items())
        lines.append(f"- **{task}** — expected: {r.get('expected_stall', 'not declared')}. Self-visible objective: "
                     f"{r.get('self_visible_objective', '')}. Published: {pub}."
                     + (f" Own history: {r['own_history']}." if r.get("own_history") else "")
                     + (f" Caveat: {r['caveat']}." if r.get("caveat") else ""))
    lines += ["", END]
    return "\n".join(lines)


def write() -> None:
    text = README.read_text()
    block = render()
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.replace("<!-- RESULT-TABLE -->", block)
    README.write_text(text)


if __name__ == "__main__":
    write()
