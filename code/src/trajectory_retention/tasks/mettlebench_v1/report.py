"""Render the measured candidate table for docs/research/mettlebench-admission/README.md."""
from __future__ import annotations

import json
from pathlib import Path

from . import source as S

REPO = Path(__file__).resolve().parents[4]
README = REPO / "docs/research/mettlebench-admission/README.md"
BEGIN, END = "<!-- BEGIN GENERATED TABLE -->", "<!-- END GENERATED TABLE -->"

# Concrete holds that do not depend on witness authoring.
STRUCTURAL_HOLDS = {
    "programbench_ripgrep_config_handoff": (
        "Workspace has case-colliding paths (eval/test_resources/...) and a 170 MB tree; the runner extracts W on the "
        "case-insensitive macOS host for grading, so W cannot be represented faithfully here."),
    "terminal_overfull_editorial_handoff": (
        "Evaluator reads tasks/<id>/internal/initial_state_manifest.yaml (trigger lexicon, word families, protected "
        "paths, proof sha256/size), which is not in the pinned repository; the private data cannot be reconstructed."),
}


def rows() -> list[dict]:
    manifest = S.verify_source()["manifest"]
    ids = [r["task_id"] for r in manifest]
    split = S.family_split(ids)
    admission = REPO / "benchmark_tasks/admission"
    registry_path = next((p for p in (admission / "registry-mettlebench-v2.json", admission / "registry-mettlebench-v1.json")
                          if p.exists()), None)
    registry = {e["upstream_task_id"]: e for e in json.loads(registry_path.read_text())["tasks"]} if registry_path else {}
    out = []
    for row in sorted(manifest, key=lambda r: r["task_id"]):
        tid = row["task_id"]
        task_id = S.package_task_id(tid)
        family = S.family_for(tid)
        q_path = REPO / "benchmark_tasks/admission" / task_id / "qualification.json"
        meta_path = REPO / "benchmark_tasks/mettlebench-references" / tid / "meta.json"
        q = json.loads(q_path.read_text()) if q_path.exists() else None
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        checklist = json.loads((S.DEFAULT_CACHE / "src/tasks" / tid / "verifier/verifier_checklist.json").read_text())
        r = {"task": tid, "benchmark": row["source_benchmark"], "family": family, "split": split[family],
             "criteria": len(checklist["points"]), "status": "HOLD", "reason": "", "start": None, "partial": None,
             "reference": None, "grade_s": None, "commands": []}
        if tid in registry and registry[tid]["admission"] == "ADMITTED":
            r["status"] = "ADMITTED"
        if q is not None:
            s = q["summary"]
            r.update(start=s["start"], partial=(s["partial"], s["partial_steps"]), reference=s["reference"],
                     grade_s=s["grade_seconds"], commands=q["visible_feedback"]["prompt_named_commands"])
            if not q["passed"]:
                r["reason"] = "qualification gates failed: " + ", ".join(s["gates_failed"])
        if r["status"] != "ADMITTED" and not r["reason"]:
            if tid in STRUCTURAL_HOLDS:
                r["reason"] = STRUCTURAL_HOLDS[tid]
            elif meta is not None and meta.get("status") == "BLOCKED":
                r["reason"] = "witness blocked: " + str(meta.get("notes", ""))[:200]
            elif meta is None:
                draft = (REPO / "benchmark_tasks/mettlebench-references" / tid / "solve.sh").exists()
                r["reason"] = ("PENDING: " + ("witness draft incomplete (interrupted, no meta.json)" if draft
                               else "no private witness yet") + "; next: author witness, then `qualify`, `admit`"
                               + (" (held-out: sealed but not loadable for dev/cal)" if r["split"] == "held_out" else ""))
            elif q is None:
                r["reason"] = ("PENDING: witness DONE (dev-graded 1.0 on image v3); next: `qualify --task " + tid
                               + "` on image v4, then `admit`")
            else:
                r["reason"] = "PENDING: qualified; next: `admit --task " + tid + "`"
        out.append(r)
    return out


def fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def table(data: list[dict]) -> str:
    lines = ["| Upstream task | Source | Family | Split | Criteria | Status | Start | Partial (steps) | Witness | Grade s | Worker-runnable checks named in prompt | Hold reason |",
             "|---|---|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    for r in data:
        partial = "" if not r["partial"] or r["partial"][0] is None else f"{r['partial'][0]:.2f} ({r['partial'][1]})"
        # Prompt fragments that span lines are regex artifacts, not commands; keep table rows on one line.
        commands = "; ".join(f"`{c}`" for c in [c for c in r["commands"] if "\n" not in c][:2]).replace("|", "\\|")
        lines.append(f"| {r['task']} | {r['benchmark']} | {r['family']} | {r['split']} | {r['criteria']} | {r['status']} | "
                     f"{fmt(r['start'])} | {partial} | {fmt(r['reference'])} | {fmt(r['grade_s'], 1)} | {commands} | "
                     f"{r['reason'].replace('|', '/')} |")
    return "\n".join(lines)


def counts(data: list[dict]) -> dict:
    admitted = [r for r in data if r["status"] == "ADMITTED"]
    return {"candidates": len(data), "admitted": len(admitted),
            "admitted_calibration": sum(r["split"] == "calibration" for r in admitted),
            "admitted_held_out": sum(r["split"] == "held_out" for r in admitted),
            "hold": len(data) - len(admitted),
            "admitted_families": len({r["family"] for r in admitted}),
            "grade_seconds_median": sorted(r["grade_s"] for r in admitted)[len(admitted) // 2] if admitted else None,
            "grade_seconds_max": max((r["grade_s"] for r in admitted), default=None)}


def render() -> dict:
    data = rows()
    text = README.read_text()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    README.write_text(text[:start] + "\n" + table(data) + "\n" + text[end:])
    return counts(data)


if __name__ == "__main__":
    print(json.dumps(render(), indent=2))
