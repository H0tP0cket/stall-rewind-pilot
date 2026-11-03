"""Review records -> healthy selection -> eight injection compositions on real saved traces.

Selections and judgements are explicitly synthetic. No model calls or scientific outcomes.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from common import ROOT, SCRATCH, ARTIFACTS, write
sys.path.insert(0, str(ROOT / "configs/v2"))
import arms_v2
from pilot_existing import healthy
from reviewed_injections import compile_selection
from judge_v2 import record as R
from study_common import INJECTION_CONDITIONS, read_json
from trajectory_retention.experiments.branch_v2 import ReadOnlyArtifactStore, BranchStore, BranchPlan, run_started, check_spec


def main():
    manifest_path = ROOT / "artifacts/v2/pilot-existing-20261026/manifest.json"
    manifest = read_json(manifest_path)
    judged = Path.home() / ".cache/trajectory-retention/pilot-integration-20261026/judge-check/healthy-output/stall_records.json"
    out = SCRATCH / "m11" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    src = ReadOnlyArtifactStore(manifest["store"])
    receiver = next(r for r in manifest["runs"] if "production-planning.r02" in r["run_id"])
    donor = next(r for r in manifest["runs"] if "production-planning.r00" in r["run_id"])
    text = R.RunView(src, receiver["run_id"]).record(30).render()
    review = {"run_id": receiver["run_id"], "point": 30, "reviewer": "SCRIPTED test",
              "progress_reason": "SYNTHETIC: verifies input binding, not a real healthy selection",
              "evidence": [text.splitlines()[1]]}
    write(out / "review.json", review)
    selection = healthy(manifest_path, judged, out / "review.json", out / "healthy")
    hd = Path(selection["directory"])
    h = read_json(hd / "healthy.json")
    quote = "Hidden tests still failing: 3."
    conditions = {}
    for condition in INJECTION_CONDITIONS[1:]:
        conditions[condition] = {"run_id": donor["run_id"], "first": 45,
            "last": 49 if condition == "failed_approach_promising" else 50,
            "reason": "SYNTHETIC composition check, not a real semantic donor selection",
            "evidence": [{"point": 50, "quote": quote}]}
    review = {"healthy_id": h["healthy_id"], "reviewer": "SCRIPTED test", "conditions": conditions}
    write(out / "selection.json", review)
    payloads = compile_selection(manifest_path, hd, out / "selection.json")
    checks = {}
    arms_v2.configure_arm_set("v3")
    for condition in INJECTION_CONDITIONS:
        sp = arms_v2.spec(purpose="injection", prefix="m11", unit_id=h["healthy_id"],
                          arm=f"injection:{condition}", repeat=0, caps=None,
                          source={k: h[k] for k in ("store", "run_id", "checkpoint_id")})
        sp["injection"] = payloads["conditions"][condition]["payload"]
        check_spec(sp)
        bs = BranchStore(out / condition / "store", src)
        plan = BranchPlan(sp, bs, run_started(bs, h["run_id"]))
        result = plan.compile(None)
        checks[condition] = result["injection"]["record"]["condition"] == condition
    bad = json.loads(json.dumps(review))
    bad["conditions"]["false_success_claim"]["evidence"][0]["quote"] = "ABSENT_PRIVATE_OR_FUTURE_EVIDENCE"
    write(out / "bad.json", bad)
    try:
        compile_selection(manifest_path, hd, out / "bad.json")
        checks["unseen_evidence_rejected"] = False
    except ValueError:
        checks["unseen_evidence_rejected"] = True
    bad = json.loads(json.dumps(review))
    bad["conditions"]["failed_attempt_failure_shown"]["last"] = 60
    write(out / "bad-cut.json", bad)
    try:
        compile_selection(manifest_path, hd, out / "bad-cut.json")
        checks["post_stop_donor_rejected"] = False
    except ValueError:
        checks["post_stop_donor_rejected"] = True
    report = {"all_pass": all(checks.values()), "synthetic": True, "provider_calls": 0,
              "checks": checks, "root": str(out)}
    write(ARTIFACTS / "m11-reviewed-inputs.json", report)
    print(json.dumps(report, indent=1))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
