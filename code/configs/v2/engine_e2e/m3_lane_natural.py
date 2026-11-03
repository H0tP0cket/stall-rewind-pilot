"""Engine milestone 3 end-to-end check: one scripted natural-v2 run per admitted terminalbench-v2 task, on the real
lane (its own submission protocol, grader container and private tests), through the driver's run_job.

m1 and m2 run on the v1 lane with a stand-in protocol; this is the check that the screen's natural runs work on the
lane they will use. llm-inference-batching-scheduler: look, submit the starter, apply the reference, submit (ends on a
passing submission). regex-chess: look, then submit the starter three times (ends on three identical failing sets).
Checks: end reasons; failing-count feedback is exactly the v2 text and is the next request's last message; the first
request says how to submit and nothing about the loop; no worker-visible request names a hidden test or the tests
directory; every point before the end is a branchable checkpoint; grades are tied to their submissions; submission behaviour is in
the status file. No provider calls.

Usage: python configs/v2/engine_e2e/m3_lane_natural.py
Writes artifacts/v2/engine/m3-lane-natural.json; stores under ~/.cache/trajectory-retention/v2-engine/m3.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path

from common import ARTIFACTS, ROOT, SCRATCH, SUBMIT, events, lifecycle, requests, step, write
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.experiments.v2_text import SUBMIT_RULE, SUBMISSION_FEEDBACK
from trajectory_retention.mining.points import point_checkpoints
from trajectory_retention.store import ArtifactStore
from trajectory_retention.tasks.terminalbench_v2.adapter import private_dir

OUT = SCRATCH / "m3"
REGISTRY = ROOT / "benchmark_tasks/admission/terminalbench-v2/registry.json"
DRIVER = ROOT / "configs/v2/natural.py"
_spec = importlib.util.spec_from_file_location("natural", DRIVER)
natural = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural)
LLM = "terminalbench-v2-llm-inference-batching-scheduler"
CHESS = "terminalbench-v2-regex-chess"


def scripts() -> dict[str, tuple[list[dict], str]]:
    solve = (private_dir(LLM) / "solution/solve.sh").read_text()
    return {
        LLM: ([step("Look at the task files.", ["ls /app/task_file"]), step("Submit the starter.", [SUBMIT]),
               step("Apply a solution.", [solve]), step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])],
              "sequence_complete:native_success"),
        CHESS: ([step("Look.", ["ls /app"]), step("Submit.", [SUBMIT]), step("Submit again.", [SUBMIT]),
                 step("Submit a third time.", [SUBMIT]), step("Never reached.", ["echo never"])],
                "sequence_complete:native_repeated_failure"),
    }


def plan_for(task: str) -> dict:
    d = OUT / task
    path = d / "plan.json"
    plan = {"version": natural.PLAN_VERSION, "plan_id": f"m3-{task.removeprefix('terminalbench-v2-')}",
            "registry": str(REGISTRY), "tasks": [task], "repeats": 1, "model": "scripted/lane-check", "settings": {},
            "store": str(d / "run"), "concurrency": 1, "detector": {"rule": "stall-clock/v8.1"},
            "budget": {"cost_limit_usd": 3.0, "step_limit": 500}, "purpose": "screen"}
    write(path, plan)
    # load_plan refuses scripted models outside fixture plans (fixtures wrap the v1 lane); this check builds the same
    # plan by hand so the real lane protocol runs with a scripted model, in a scratch store never read as data.
    return dict(plan, execution_purpose="development", tool_timeout_seconds=natural.DEFAULT_TOOL_TIMEOUT,
                _path=str(path.resolve()))


def check(task: str, script: list[dict], expected_end: str) -> dict:
    plan = plan_for(task)
    job = natural.jobs(plan)[0]
    t0 = time.monotonic()
    status = natural.run_job(plan, job, model_factory_override=lambda c: ScriptedModel(script, model_name=plan["model"]))
    store = ArtifactStore(natural.layout(plan)["store"])
    rid = job["run_id"]
    evs = events(store, rid)
    reqs = [json.loads(r) for r in requests(store, evs)]
    verdicts = [e.payload for e in lifecycle(evs, "native_submission_verdict")]
    grades = {e.event_id for e in evs if e.kind == "grade"}
    points = point_checkpoints(store, rid)
    steps = status.get("steps") or 0
    hidden = json.loads((private_dir(task) / "checks.json").read_text())
    names = [c if isinstance(c, str) else c.get("id") or c.get("name") for c in
             (hidden if isinstance(hidden, list) else hidden.get("checks") or hidden.get("tests") or [])]
    names = [n.split("::")[-1] for n in names if n]
    visible = "\n".join(json.dumps(r["messages"]) for r in reqs)
    feedback_ok = []
    for v in verdicts:
        if v["status"] == "pass":
            continue
        later = [r for r in reqs if any(m.get("role") == "user" and m.get("content") == SUBMISSION_FEEDBACK.format(
            n=v["failing_count"]) for m in r["messages"][-1:])]
        feedback_ok.append(bool(later))
    first = json.dumps(reqs[0]["messages"]) if reqs else ""
    checks = {
        "completed_with_expected_end": status.get("status") == "completed" and status.get("termination_reason") == expected_end,
        "failing_count_feedback_is_next_request_last_message": bool(feedback_ok) and all(feedback_ok),
        "first_request_says_how_to_submit": SUBMIT_RULE in first,
        "first_request_says_nothing_about_the_loop": not any(w in first for w in ("Hidden tests", "failing", "three times")),
        "no_hidden_test_named_in_worker_requests": bool(names) and not any(n in visible for n in names)
                                                     and "/tests/" not in visible,
        # the point after the run-ending submission is a terminal exit, not branchable (as in m1)
        "every_point_before_the_end_branchable": all(k in points for k in range(0, steps)),
        "every_grade_tied_to_its_submission": all(v.get("grade_event_id") in grades for v in verdicts),
        "submission_behaviour_in_status": len((status.get("submission_behaviour") or {}).get("submissions") or [])
                                          == len(verdicts),
    }
    return {"task": task, "run_id": rid, "checks": checks, "status": {k: status.get(k) for k in (
        "status", "termination_reason", "steps", "spend_usd_known", "solved_by_submission", "checkpoints")},
        "submissions": [{k: v.get(k) for k in ("ordinal", "status", "failing_count", "share_passed", "outcome",
                                               "step_index")} for v in verdicts],
        "hidden_tests": len(names), "wall_seconds": round(time.monotonic() - t0, 1)}


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = {"what": __doc__.split("\n\n")[0], "command": "python configs/v2/engine_e2e/m3_lane_natural.py",
              "registry": str(REGISTRY), "provider_calls": 0, "runs": {}}
    for task, (script, end) in scripts().items():
        report["runs"][task] = check(task, script, end)
    report["all_checks"] = {f"{t}.{k}": v for t, r in report["runs"].items() for k, v in r["checks"].items()}
    report["passed"] = all(report["all_checks"].values())
    write(ARTIFACTS / "m3-lane-natural.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in report["all_checks"].items() if not v],
                      "runs": {t: {"end": r["status"]["termination_reason"], "submissions": r["submissions"],
                                   "wall_seconds": r["wall_seconds"]} for t, r in report["runs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
