"""Export the pilot's data from the study folders into data/ (run inside the original research repo; not needed to
reproduce the tables and figures, which read only data/).

Writes:
  data/branches.csv          one row per branch: pilot main / injection / harm, the stalled-told follow-up, and the
                             2026-11-04 follow-ups (any-time review, arm 7; Opus's recorded choice executed)
  data/screen.csv            the stall screen (5 continue + 5 forced-rewind branches per candidate)
  data/review_decisions.json every one-shot review-turn decision by the worker, with an excerpt of its reasoning
  data/observer_choices.csv  one row per review request x chooser x answer: what the worker chose, what the worker
                             model chose when asked again, what it chose as an outside observer, and what Opus chose
                             (decisions only, no text)
Usage: python analysis/export_from_study.py /path/to/llm-trajectories

Review columns of branches.csv: for the one-shot review arms they describe the single review turn. For
`2_ledger_review_opus_decides` they describe Opus's recorded decision, which the branch executed. For
`7_ledger_review_anytime` they sum over every opening of the review in the branch (change_nothing is True only if
every opening changed nothing; rewound channels are listed per opening, with step numbers as the context was numbered
when that opening happened).

The source stores are opened read-only and immutable.
"""
import csv
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))
from trajectory_retention.mining.branch_store import ReadOnlyArtifactStore  # noqa: E402

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
        "main-stalled-a": "followup_stalled_told", "main-stalled-b": "followup_stalled_told",
        "anytime": "followup_anytime_review", "opus-decider": "followup_opus_decides"}
CHANNELS = ("workspace", "context", "runtime", "outputs")
FEEDBACK_LINE = {None: "count line (Hidden tests still failing: N.)",
                 "neutral-submission-feedback/v2": "neutral line (The task is not complete yet. Continue working.)"}


class FrozenStore(ReadOnlyArtifactStore):
    """A finished store opened as immutable, so reading it creates no journal side files beside the source data."""

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=30)
        try:
            yield db
        finally:
            db.close()


def review_turn(spec, res, attempt_dir):
    """The review turn's decision and reasoning, read from the branch store."""
    st, src = FrozenStore(attempt_dir / "store"), FrozenStore(spec["source"]["store"])
    for e in st.iter_events(res["run_id"]):
        if e.kind == "model_response" and e.output_ref is not None:
            try:
                o = st.get_json(e.output_ref)
            except FileNotFoundError:
                o = src.get_json(e.output_ref)
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
            decider = (spec.get("review") or {}).get("decider")
            openings = (res.get("anytime") or {}).get("openings") if spec.get("anytime") else None
            if decider:  # Opus's recorded decision, executed in place of the worker's review turn
                dec, reasoning = decider["decision"], ""
            elif spec.get("review"):
                dec, reasoning = review_turn(spec, res, attempt_dir)
            else:
                dec, reasoning = None, ""
            pts = ((res.get("choice") or {}).get("compiled") or {}).get("points") or {}
            rewound = ({ch: pts[ch] for ch in CHANNELS if pts.get(ch) is not None and pts[ch] < point} if dec else {})
            review_cols = {
                "review_change_nothing": "" if dec is None else bool(dec.get("change_nothing")),
                "review_rewound_channels": "" if dec is None else ";".join(f"{k}->{v}" for k, v in rewound.items()),
                "review_drops": "" if dec is None else len(dec.get("drop") or []),
                "review_keeps": "" if dec is None else len(dec.get("keep") or []),
            }
            if openings is not None:  # arm 7: sum over every opening of the review in this branch
                ods = [o["decision"] for o in openings]
                review_cols = {
                    "review_change_nothing": all(bool(d.get("change_nothing")) for d in ods) if ods else "",
                    "review_rewound_channels": ";".join(
                        f"opening{o['opening']}:{ch}->{(o['decision'].get('rewind') or {})[ch]}"
                        for o in openings for ch in CHANNELS
                        if (o["decision"].get("rewind") or {}).get(ch) is not None),
                    "review_drops": sum(len(d.get("drop") or []) for d in ods),
                    "review_keeps": sum(len(d.get("keep") or []) for d in ods),
                }
            arm_name = arm.replace("injection:", "inj_").replace("~opus", "_opus_decides")
            rows.append({
                "set": label, "spec_set": prefix, "unit": unit, "task": task, "unit_kind": kind,
                "stall_type": stall_type, "screen_stratum": stratum, "branch_point": point,
                "arm": arm_name, "repeat": int(rep),
                "review_prompt": ((spec.get("review") or {}).get("prompt_version", "")
                                  or (spec.get("anytime") or {}).get("prompt_version", "")),
                "forced_rewind_to": (spec.get("stall_onset_point") if arm == "4_forced_rewind_to_p" else ""),
                "solved": bool(res.get("solved")), "best_share_passed": res.get("best_share_passed"),
                "final_share_passed": res.get("final_share_passed"),
                "submissions": len(res.get("submissions") or []), "end_reason": res.get("end_reason"),
                "cost_usd": round(res.get("cost_usd") or 0, 5), "new_tokens": res.get("new_tokens"),
                "steps": res.get("steps"),
                **review_cols,
                "error": res.get("error") or "",
            })
            if dec is not None and not decider:
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
        # the feedback line a screen branch delivered: the rule its spec binds, else the source run's count line
        rules = {(json.loads(f.read_text()).get("submission_feedback") or {}).get("rule")
                 for f in SPECS.glob(f"screen-*/{s['stall_id']}/*__r*.json")}
        assert len(rules) == 1, (s["stall_id"], rules)
        feedback = FEEDBACK_LINE[rules.pop()]
        for set_name in ("continue", "rewind"):
            p = (s.get("primary") or {}).get(set_name) or {}
            srows.append({"stall_id": s["stall_id"], "task": s["task_id"].replace("terminalbench-v4-", ""),
                          "admitted": s.get("admitted"), "stratum": s.get("stratum"), "set": set_name,
                          "n": p.get("n"), "solves": p.get("solves"), "best_share_mean": p.get("best_share_mean"),
                          "feedback": feedback})
    with (OUT / "screen.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0]))
        w.writeheader()
        w.writerows(srows)
    orows = observer_rows()
    with (OUT / "observer_choices.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(orows[0]))
        w.writeheader()
        w.writerows(orows)
    print(len(rows), "branches;", len(decisions), "review decisions;", len(srows), "screen rows;", len(orows),
          "observer-choice rows")


def observer_rows():
    """Every answer to each review request: the worker's own, the worker model asked again with the same request
    (resample), the worker model asked as an outside observer, and Opus as an outside observer. Decisions only."""
    summary = json.loads((PILOT / "study/observer-choices/summary.json").read_text())
    out = []
    for r in summary["rows"]:
        task, kind, _, _ = UNITS[r["unit"]]
        answers = ([("worker", i, w.get("turn", ""), w) for i, w in enumerate(r["worker"])]
                   + [("deepseek_resample", a["sample"], "", a) for a in r["ds-resample"]]
                   + [("deepseek_outside_observer", a["sample"], "", a) for a in r["ds-observer"]]
                   + [("opus", 0, "", r["opus"])])
        for chooser, sample, turn, a in answers:
            dec = a.get("decision") or {}
            usable = "change_nothing" in dec and a.get("kind") not in ("empty", "none")
            rw = (dec.get("rewind") or {}) if usable else {}
            flag = {ch: rw.get(ch) is not None for ch in CHANNELS}
            out.append({
                "request_id": r["request_id"], "task": task, "unit_kind": kind, "arm": r["arm"],
                "chooser": chooser, "sample": sample, "worker_turn": turn,
                "decision": a.get("kind", ""),
                "usable_answer": usable,
                "changed_nothing": bool(dec.get("change_nothing")) if usable else "",
                "rewound_context": flag["context"] if usable else "",
                "rewound_files": flag["workspace"] if usable else "",
                "rewound_runtime": flag["runtime"] if usable else "",
                "rewound_outputs_only": (flag["outputs"] and not (flag["context"] or flag["workspace"]
                                                                  or flag["runtime"])) if usable else "",
                "n_drops": len(dec.get("drop") or []) if usable else "",
                "n_keeps": len(dec.get("keep") or []) if usable else "",
                "valid": usable and not a.get("errors") and a.get("ok", True) is not False,
                "input_variant": a.get("input_variant", ""),
                "cost_usd": round(a["cost"], 5) if "cost" in a else "",
            })
    return out


if __name__ == "__main__":
    main()
