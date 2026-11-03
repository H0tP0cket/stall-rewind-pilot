"""Fixed-unit labels -> review views -> six Docker arms. Scripted models, no paid calls.

Builds a fresh m6-style fixture store. Writes a timestamped store tree and a check receipt.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys
from common import ROOT, SCRATCH, ARTIFACTS, step, write, events, requests
sys.path.insert(0, str(ROOT / "configs/v2"))
import arms_v2
from judge_v2 import record as R
from m6_arms_v3 import review_call, reply
import m6_arms_v3 as fixture
from trajectory_retention.experiments.branch_v2 import run_branch, ReadOnlyArtifactStore, BranchStore, BranchPlan, run_started
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.ledger.units import build_units, scripted_units_labeller
from trajectory_retention.ledger.arm_views import build_views
from trajectory_retention.store import ArtifactStore
from trajectory_retention.mining.points import point_checkpoints


def main():
    out = SCRATCH / "m9" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    fixture.OUT = out / "source"
    fixture.LIMIT = dict(fixture.LIMIT, window_tokens=65536, source="m9 fixture window")
    _, source, rid, _ = fixture.run_natural()
    src = ReadOnlyArtifactStore(source)
    cp = point_checkpoints(src, rid)[8]
    d = out / "stall"
    doc = build_units(src, rid, cp, scripted_units_labeller(ArtifactStore(out / "labels")))
    views = build_views(doc, d)
    items = {k: v for k, v in views["ledger"]["units"].items() if v["kind"] == "item"}
    keep = next(k for k, v in items.items() if v["steps"][0] > 4)
    drop = next(k for k, v in items.items() if v["steps"][0] > 4 and k != keep)
    arms_v2.configure_arm_set("v3")
    stall = {"stall_id": "m9-p8", "store": str(source), "run_id": rid, "checkpoint_id": cp,
             "dir": str(d), "step_index": 8, "stall_onset_point": 4}
    checks = {"labels_complete": doc["status"] == "ok", "fake_labels": doc["fake_labels"]}
    results = {}
    for arm in arms_v2.ARMS:
        sp = arms_v2.arm_spec(stall, arm, 0, purpose="tuning", prefix="m9", caps=None)
        sp["budget"]["step_limit"] = 2 if sp["review"] else 1
        continuation = step("Probe.", ["cat /tmp/v6/marker; cat /app/v6_note.txt"], tokens=100)
        script = [continuation]
        if arm == "2_ledger_review":
            args = {"rewind": {"workspace": 4, "context": 4, "runtime": 4, "outputs": None},
                    "drop": ["O2"], "keep": [keep], "change_nothing": False}
            script.insert(0, review_call(args, reasoning="Keep one labelled item."))
        elif arm == "3_context_review":
            args = {"rewind": {"workspace": None, "context": 4, "runtime": None, "outputs": None},
                    "drop": ["O2"], "keep": ["S6"], "change_nothing": False}
            script.insert(0, review_call(args, reasoning="Keep a step."))
        elif sp["review"]:
            script.insert(0, reply("Review recorded.", reasoning="Reviewing the work."))
        model = ScriptedModel(script, model_settings={"max_tokens": 2000})
        result = run_branch(sp, out_dir=out / arm, model_factory=lambda config: model)
        results[arm] = result
        checks[arm + "_completed"] = result["end_reason"] == "step_guard"
        checks[arm + "_no_submission_scores_zero"] = result["best_share_passed"] == 0
        bs = BranchStore(Path(result["store"]), src)
        ev = events(bs, result["run_id"])
        req = requests(bs, ev)
        checks[arm + "_q56"] = any(e.kind == "lifecycle" and
            (e.payload.get("config", {}).get("metadata", {}).get("submission_stop") == sp["submission_stop"]) for e in ev)
        if arm == "2_ledger_review":
            checks["fixed_unit_kept_with_label"] = bool(req) and items[keep]["kept_text"] in req[-1]
            # Also verify exact fixed-unit drop compilation against the real serialized history.
            plan = BranchPlan(sp, bs, run_started(bs, rid))
            compiled = plan.decision_plan({"rewind": None, "drop": [drop], "keep": [], "change_nothing": False}, "per_channel")
            checks["fixed_unit_drop_exact_quote"] = compiled["edits"] == [
                {"op": "drop_quote", "step": items[drop]["steps"][0], "quote": items[drop]["quote"]}]
        print(arm, result["end_reason"], result.get("error"), flush=True)
    report = {"all_pass": all(checks.values()), "checks": checks, "provider_calls": 0,
              "fake_labels": True, "root": str(out), "results": results}
    write(ARTIFACTS / "m9-units-arms.json", report)
    print(json.dumps({"all_pass": report["all_pass"], "failed": [k for k, v in checks.items() if not v]}))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
