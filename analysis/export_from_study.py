"""Export the pilot's data from the study folders into data/ (run inside the original research repo; not needed to
reproduce the tables and figures, which read only data/).

Writes:
  data/branches.csv          one row per branch: pilot main / injection / harm, plus the stalled-told follow-up
  data/screen.csv            the stall screen (5 continue + 5 forced-rewind branches per candidate)
  data/review_decisions.json every review-turn decision with an excerpt of the agent's reasoning
Usage: python analysis/export_from_study.py /path/to/llm-trajectories
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))
from trajectory_retention.experiments.branch_v2 import BranchStore, ReadOnlyArtifactStore  # noqa: E402

PILOT = REPO / "artifacts/v2/pilot-20261027-terra"
RUNS, SPECS = PILOT / "study/runs", PILOT / "study/specsets"
OUT = Path(__file__).resolve().parents[1] / "data"

UNITS = {  # unit id -> (task, kind, stall type from the type judge, screen stratum)
    "terminalbench-v4-production-planning--5f6830d3e8--p20": ("production-planning", "stall", "stuck", "rewind_better"),
    "terminalbench-v4-atrx-vep-crispr--6bd677acf1--p100": ("atrx-vep-crispr", "stall", "under-commit", "continue_better"),
    "terminalbench-v4-pilot-heatpump-heat-pump-warranty--46457a59cd--p70": ("heat-pump-warranty", "stall", "over-commit",
                                                                             "similar"),
    "healthy-cd83723d94a8-p10": ("production-planning", "healthy", "", ""),
    "healthy-2255bd597d7a-p60": ("atrx-vep-crispr", "healthy", "", ""),
    "healthy-af1c6eafaae6-p20": ("mvcc-lsm-compaction", "healthy", "", ""),
}
SETS = {"main": "pilot_main", "injection": "pilot_injection", "harm": "pilot_harm",
        "main-stalled-a": "followup_stalled_told", "main-stalled-b": "followup_stalled_told"}


def review_turn(spec, res, attempt_dir):
    """The review turn's decision and reasoning, read from the branch store."""
    st = BranchStore(attempt_dir / "store", ReadOnlyArtifactStore(spec["source"]["store"]))
    for e in st.iter_events(res["run_id"]):
        if e.kind == "model_response" and e.output_ref is not None:
            o = st.get_json(e.output_ref)
            calls = [c for c in (o.get("tool_calls") or []) if c.get("name") == "review_decision"]
            if calls:
                return json.loads(calls[0]["arguments"]), str(o.get("reasoning") or o.get("reasoning_content") or "")
    return None, ""


def main():
    rows, decisions = [], []
    for prefix, label in SETS.items():
        for spec_path in sorted((SPECS / prefix).glob("*/*__r*.json")):
            spec = json.loads(spec_path.read_text())
            unit, arm, rep = spec["branch_id"].rsplit("/", 2)
            attempt_dir = RUNS / prefix / "branches" / spec["branch_id"].replace("/", "__").replace(":", "~") / "a0"
            if not (attempt_dir / "result.json").exists():
                attempt_dir = next((RUNS / prefix / "branches").glob(f"{unit}__{arm.replace(':', '~')}__{rep}/a*"))
            res = json.loads((attempt_dir / "result.json").read_text())
            task, kind, stall_type, stratum = UNITS[unit]
            point = int(unit.rsplit("-p", 1)[1])
            dec, reasoning = review_turn(spec, res, attempt_dir) if spec.get("review") else (None, "")
            pts = ((res.get("choice") or {}).get("compiled") or {}).get("points") or {}
            rewound = ({ch: pts[ch] for ch in ("workspace", "context", "runtime", "outputs")
                        if pts.get(ch) is not None and pts[ch] < point} if dec else {})
            rows.append({
                "set": label, "spec_set": prefix, "unit": unit, "task": task, "unit_kind": kind,
                "stall_type": stall_type, "screen_stratum": stratum, "branch_point": point,
                "arm": arm.replace("injection:", "inj_"), "repeat": int(rep),
                "review_prompt": (spec.get("review") or {}).get("prompt_version", ""),
                "forced_rewind_to": (spec.get("stall_onset_point") if arm == "4_forced_rewind_to_p" else ""),
                "solved": bool(res.get("solved")), "best_share_passed": res.get("best_share_passed"),
                "final_share_passed": res.get("final_share_passed"),
                "submissions": len(res.get("submissions") or []), "end_reason": res.get("end_reason"),
                "cost_usd": round(res.get("cost_usd") or 0, 5), "new_tokens": res.get("new_tokens"),
                "steps": res.get("steps"),
                "review_change_nothing": "" if dec is None else bool(dec.get("change_nothing")),
                "review_rewound_channels": "" if dec is None else ";".join(f"{k}->{v}" for k, v in rewound.items()),
                "review_drops": "" if dec is None else len(dec.get("drop") or []),
                "review_keeps": "" if dec is None else len(dec.get("keep") or []),
                "error": res.get("error") or "",
            })
            if dec is not None:
                decisions.append({"set": label, "branch_id": spec["branch_id"], "task": task, "unit_kind": kind,
                                  "stall_type": stall_type, "arm": arm, "prompt": rows[-1]["review_prompt"],
                                  "decision": dec, "solved": rows[-1]["solved"],
                                  "reasoning_excerpt": reasoning[:1500]})
    OUT.mkdir(exist_ok=True)
    with (OUT / "branches.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "review_decisions.json").write_text(json.dumps(decisions, indent=1) + "\n")

    screen = json.loads((PILOT / "study/screen/screen.json").read_text())
    srows = []
    for s in screen["stalls"]:
        for set_name in ("continue", "rewind"):
            p = (s.get("primary") or {}).get(set_name) or {}
            srows.append({"stall_id": s["stall_id"], "task": s["task_id"].replace("terminalbench-v4-", ""),
                          "admitted": s.get("admitted"), "stratum": s.get("stratum"), "set": set_name,
                          "n": p.get("n"), "solves": p.get("solves"), "best_share_mean": p.get("best_share_mean"),
                          "feedback": "count line (Hidden tests still failing: N.)"})
    with (OUT / "screen.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    print(len(rows), "branches;", len(decisions), "review decisions;", len(srows), "screen rows")


if __name__ == "__main__":
    main()
