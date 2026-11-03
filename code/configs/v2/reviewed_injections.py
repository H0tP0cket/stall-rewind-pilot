"""Compile reviewed, real trace segments into injection payloads; no model calls.

User approved manual donor review for the development pilot, 2026-10-26.
Selection JSON: {healthy_id, reviewer, conditions: {condition: {run_id, first,
last, reason, evidence: [{point, quote}], ...}}}. Missing conditions are explicit
unavailable entries, never fabricated. Workspace debris uses before/after points.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import steps_v2
from healthy_v2 import _payload, _versions_match
from judge_v2 import record as R
from study_common import INJECTION_CONDITIONS, read_json, write_json
from trajectory_retention.experiments.branch_v2 import ReadOnlyArtifactStore


def compile_selection(manifest_path: Path, healthy_dir: Path, selection_path: Path) -> dict:
    manifest = read_json(manifest_path)
    healthy = read_json(healthy_dir / "healthy.json")
    selection = read_json(selection_path)
    if selection.get("healthy_id") != healthy["healthy_id"] or not selection.get("reviewer"):
        raise ValueError("selection needs the matching healthy_id and reviewer")
    if set(selection["conditions"]) != set(INJECTION_CONDITIONS) - {"continue"}:
        raise ValueError("record every non-control condition, including unavailable conditions")
    runs = {r["run_id"]: r for r in manifest["runs"]}
    source = ReadOnlyArtifactStore(manifest["store"])
    scope = steps_v2.run_scope(healthy["store"], healthy["run_id"])
    conditions = {"continue": {"payload": _payload("continue", healthy, scope, channel=None)}}
    for condition, item in selection["conditions"].items():
        if item.get("unavailable"):
            conditions[condition] = {"unavailable": item["unavailable"]}
            continue
        run = runs[item["run_id"]]
        first, last = item["first"], item["last"]
        if type(first) is not int or type(last) is not int or not 1 <= first <= last <= run["branch_before_step"]:
            raise ValueError(f"{condition}: invalid or post-stop segment")
        same = run["run_id"] == healthy["run_id"]
        if (same and last > healthy["step_index"]) or (not same and run["later_solved"]):
            raise ValueError(f"{condition}: donor must be a failed run or earlier in receiving run")
        donor_scope = steps_v2.run_scope(manifest["store"], run["run_id"])
        if run["task_id"] != healthy["task_id"] or not _versions_match(scope, donor_scope):
            raise ValueError(f"{condition}: task/version mismatch")
        if not item.get("reason") or not item.get("evidence"):
            raise ValueError(f"{condition}: reviewed rationale and visible evidence required")
        view = R.RunView(source, run["run_id"])
        for evidence in item["evidence"]:
            point = evidence["point"]
            if not 1 <= point <= run["branch_before_step"] or (same and point > healthy["step_index"]):
                raise ValueError(f"{condition}: evidence outside eligible donor prefix")
            if not evidence["quote"] or evidence["quote"] not in view.record(point).render():
                raise ValueError(f"{condition}: evidence quote not in worker-visible prefix")
        donor = {"store": manifest["store"], "run_id": run["run_id"], "task_id": run["task_id"],
                 "same_run": same, "steps": [first, last],
                 "cut": {"after_point": first - 1, "through_point": last},
                 "task_version": {k: donor_scope.get(k) for k in ("requirement_version", "evaluator_version")},
                 "selection": {"rule": "reviewed-segments/pilot-v1", "reviewer": selection["reviewer"],
                               "reason": item["reason"], "evidence": item["evidence"]}}
        debris = None
        if condition == "workspace_debris":
            debris = {"donor_point_before": first - 1, "donor_point_after": last,
                      "path_rule": "created/modified donor files absent from receiver; engine checks scope", "paths": None}
        measured = steps_v2.approx_tokens("\n".join(steps_v2.step_text(s) for s in
            steps_v2.run_steps(manifest["store"], run["run_id"])["steps"] if first <= s["k"] <= last))
        payload = _payload(condition, healthy, scope, channel="workspace" if debris else "context",
                           donor=donor, debris=debris,
                           size={"target_tokens": {"small_neutral": 500, "large_neutral": 6000}.get(condition),
                                 "measured_tokens": measured})
        payload["rules"] = "reviewed-segments/pilot-v1"
        conditions[condition] = {"payload": payload}
    pair = selection["conditions"]
    promising, failure = pair["failed_approach_promising"], pair["failed_attempt_failure_shown"]
    if not promising.get("unavailable") and not failure.get("unavailable"):
        if promising["run_id"] != failure["run_id"] or promising["last"] >= failure["last"]:
            raise ValueError("promising segment must precede the shown failure in the same donor run")
    result = {"version": "injection-donors.v1", "healthy_id": healthy["healthy_id"],
              "rules": {"name": "reviewed-segments/pilot-v1", "status": "user-approved development pilot"},
              "draft_rules_used": [], "review": str(selection_path.resolve()), "conditions": conditions}
    target = healthy_dir / "injection_donors.json"
    if target.exists() and read_json(target) != result:
        raise ValueError("refusing to overwrite different reviewed donors")
    write_json(target, result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--healthy-dir", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    a = ap.parse_args()
    r = compile_selection(a.manifest, a.healthy_dir, a.selection)
    print(json.dumps({k: "available" if "payload" in v else v["unavailable"] for k, v in r["conditions"].items()}, indent=1))
