"""Engine end-to-end check for the terminalbench-v4 lane: scripted natural-v2 runs on the real lane (its own submission
protocol, Harbor-style verifier container and private tests) through the driver's run_job. No provider calls.

For every task the lane ledger lists as VALIDATED or ADMITTED (or the task ids given): look, submit the starter, apply
the task's own reference, submit (ends on a passing submission). For the first such task also: look, then submit the
starter three times (ends on three identical failing sets). The reference is the task's solution directory shipped in
one command as base64 and run like Harbor's oracle (``bash solve.sh > log 2>&1``); the worker cannot write /solution, so
the scaffold stages it under /tmp and rewrites "/solution/" to that path in the staged copy only (recorded).

Checks (as m3, plus the lane's own): end reasons; failing-count feedback is exactly the v2 text and is the next request's
last message; the first request says how to submit and nothing about the loop; the first request carries the
instruction with Harbor's canary stripped; no worker-visible request names a hidden test or the tests directory; every
point before the end is a branchable checkpoint; grades are tied to their submissions; submission behaviour is in the
status file; every grade ran in the verifier image with network none.

Sealed tasks load through the registry as the driver loads them. A task not yet sealed is not eligible for any purpose
under the registry rule, so it is loaded straight from its package, as the gate's restore check loads it; each run
records how it was loaded and the task's lane status. A task id given on the command line may be any packaged task,
including one on HOLD.

Usage: python configs/v2/engine_e2e/m4_lane_v4_natural.py [TASK_ID ...]
Writes artifacts/v2/engine/m4-lane-v4-natural.json; stores under ~/.cache/trajectory-retention/v2-engine/m4.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path

from common import ARTIFACTS, ROOT, SCRATCH, SUBMIT, events, lifecycle, requests, step, write
from trajectory_retention.contracts import ArtifactRef
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.experiments.v2_text import SUBMIT_RULE, SUBMISSION_FEEDBACK
from trajectory_retention.mining.points import point_checkpoints
from trajectory_retention.registry import read_registry, validate_task_purpose
from trajectory_retention.store import ArtifactStore
from trajectory_retention.tasks.terminalbench_v4.adapter import TerminalBenchV4TaskAdapter, private_dir, strip_canary
from trajectory_retention.tasks.terminalbench_v4.source import remove_tree

OUT = SCRATCH / "m4"
LANE = ROOT / "benchmark_tasks/admission/terminalbench-v4"
REGISTRY = LANE / "registry.json"
PACKAGES = ROOT / "benchmark_tasks/packages/terminalbench-v4"
DRIVER = ROOT / "configs/v2/natural.py"
STAGED = "/tmp/tbv4-solution"
# The verifier's hidden tests directory /tests/, not a worker path that ends in tests/ (mvcc-lsm-compaction's instruction
# names /app/tests/regression_test.cc).
HIDDEN_TESTS_DIR = re.compile(r"(?<![\w./-])/tests/")
_spec = importlib.util.spec_from_file_location("natural", DRIVER)
natural = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural)
LOADED: dict[str, str] = {}
_registry_load = natural.load_task


def load_task(plan: dict, task_id: str, scratch: Path):
    """The driver's registry load for sealed tasks; a direct development load of the package otherwise."""
    if task_id in read_registry(REGISTRY):
        LOADED[task_id] = "registry (sealed)"
        return _registry_load(plan, task_id, scratch)
    # Unsealed: the adapter marks it ineligible for every purpose, so the registry rule would refuse it. Like the gate's
    # restore check, this check loads the package directly; its store is scratch and never read as data.
    task = TerminalBenchV4TaskAdapter(PACKAGES / task_id, scratch_root=scratch)
    try:
        validate_task_purpose(task.describe(), plan["execution_purpose"])
        eligible = True
    except ValueError:
        eligible = False
    if getattr(task, "native_submission_protocol", None) is None:
        raise ValueError(f"{task_id} declares no submission protocol")
    LOADED[task_id] = ("package (unsealed), loaded directly as the gate's restore check does; registry eligibility "
                       f"for {plan['execution_purpose']}: {eligible}")
    return task


natural.load_task = load_task


def lane_status() -> dict[str, dict]:
    ledger = json.loads((LANE / "ledger.json").read_text())
    return {t["task_id"]: {"status": t["status"], "reasons": t["reasons"]} for t in ledger["tasks"]}


def lane_tasks() -> list[str]:
    return [t for t, row in lane_status().items() if row["status"] in ("VALIDATED", "ADMITTED")]


def reference_command(task_id: str) -> tuple[str, dict]:
    """The task's solution directory as one base64 tar.gz, staged under /tmp and run like Harbor's oracle."""
    root = private_dir(task_id) / "solution"
    buf, rewritten = io.BytesIO(), []
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            data = path.read_bytes()
            if b"/solution/" in data:
                data = data.replace(b"/solution/", (STAGED + "/").encode())
                rewritten.append(path.relative_to(root).as_posix())
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            info.size, info.mode = len(data), path.stat().st_mode & 0o777
            tf.addfile(info, io.BytesIO(data))
    blob = base64.b64encode(buf.getvalue()).decode()
    command = (f"rm -rf {STAGED} && mkdir -p {STAGED} && echo {blob} | base64 -d | tar -xzf - -C {STAGED} && "
               f"chmod +x {STAGED}/solve.sh && export DEBIAN_FRONTEND=noninteractive && "
               f"(bash {STAGED}/solve.sh) > /tmp/tbv4-oracle.txt 2>&1; echo reference exit $?")
    return command, {"staged_at": STAGED, "rewritten_files": rewritten, "archive_bytes": len(buf.getvalue())}


def scripts(tasks: list[str]) -> dict[str, tuple[str, list[dict], str, dict]]:
    out = {}
    for task in tasks:
        command, note = reference_command(task)
        out[f"{task}::pass"] = (task, [step("Look at the task files.", ["ls -la /app"]),
                                       step("Submit the starter.", [SUBMIT]), step("Apply a solution.", [command]),
                                       step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])],
                                "sequence_complete:native_success", note)
    if tasks:
        task = tasks[0]
        out[f"{task}::repeat"] = (task, [step("Look.", ["ls /app"]), step("Submit.", [SUBMIT]),
                                         step("Submit again.", [SUBMIT]), step("Submit a third time.", [SUBMIT]),
                                         step("Never reached.", ["echo never"])],
                                  "sequence_complete:native_repeated_failure", {})
    return out


def plan_for(key: str, task: str) -> dict:
    d = OUT / key.replace("::", "-")
    path = d / "plan.json"
    plan = {"version": natural.PLAN_VERSION, "plan_id": f"m4-{key.replace('::', '-').removeprefix('terminalbench-v4-')}",
            "registry": str(REGISTRY), "tasks": [task], "repeats": 1, "model": "scripted/lane-check", "settings": {},
            "store": str(d / "run"), "concurrency": 1, "detector": {"rule": "stall-clock/v8.1"},
            "budget": {"cost_limit_usd": 3.0, "step_limit": 500}, "purpose": "screen"}
    write(path, plan)
    # As in m3: load_plan refuses scripted models outside fixture plans, so the plan is built by hand and the real lane
    # protocol runs with a scripted model, in a scratch store never read as data.
    return dict(plan, execution_purpose="development", tool_timeout_seconds=natural.DEFAULT_TOOL_TIMEOUT,
                _path=str(path.resolve()))


def grade_networks(store, evs) -> list[dict]:
    """Per grade: the verifier image and the network of the verifier and source containers, from its private receipt."""
    rows = []
    for e in evs:
        if e.kind != "grade":
            continue
        evaluation = store.get_json(ArtifactRef(**e.payload["evaluation_ref"]))
        receipt_path = Path(evaluation["details"]["private_output_dir"]) / "receipt.json"
        receipt = json.loads(receipt_path.read_text()).get("receipt", {}) if receipt_path.is_file() else {}
        rows.append({"evaluation_id": e.payload["evaluation_id"],
                     "verifier_network": (receipt.get("verifier_binding") or {}).get("network"),
                     "source_network": (receipt.get("source_binding") or {}).get("network"),
                     "verifier_image_id": (receipt.get("verifier_binding") or {}).get("image_id")})
    return rows


def check(key: str, task: str, script: list[dict], expected_end: str, note: dict) -> dict:
    plan = plan_for(key, task)
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
    names = [n.split("::")[-1].split("[")[0] for n in json.loads((private_dir(task) / "checks.json").read_text())["checks"]]
    shown = [m for r in reqs for m in r["messages"] if m.get("role") != "assistant"]  # what the worker is shown
    visible = "\n".join(json.dumps(m) for m in shown)
    visible_text = "\n".join(m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
                             for m in shown)  # decoded, so an escaped newline is not taken for a path character
    feedback_ok = []
    for v in verdicts:
        if v["status"] == "pass":
            continue
        later = [r for r in reqs if any(m.get("role") == "user" and m.get("content") == SUBMISSION_FEEDBACK.format(
            n=v["failing_count"]) for m in r["messages"][-1:])]
        feedback_ok.append(bool(later))
    first = json.dumps(reqs[0]["messages"]) if reqs else ""
    instruction = strip_canary((PACKAGES / task / "public/instruction.md").read_text())
    grade_rows = grade_networks(store, evs)
    verifier_image = json.loads((PACKAGES / task / "package.json").read_text())["environment"]["verifier"]["image_id"]
    checks = {
        "completed_with_expected_end": status.get("status") == "completed" and status.get("termination_reason") == expected_end,
        "failing_count_feedback_is_next_request_last_message": all(feedback_ok) and (bool(feedback_ok) or expected_end
                                                                                       == "sequence_complete:native_success"
                                                                                       and len(verdicts) >= 1),
        "first_request_says_how_to_submit": SUBMIT_RULE in first,
        "first_request_says_nothing_about_the_loop": not any(w in first for w in ("Hidden tests", "failing", "three times")),
        "first_request_carries_the_stripped_instruction": bool(reqs) and any(
            instruction.strip() in (m.get("content") or "") for m in reqs[0]["messages"] if m.get("role") == "user"),
        "no_hidden_test_named_in_worker_requests": bool(names) and not any(n in visible for n in set(names))
                                                     and not HIDDEN_TESTS_DIR.search(visible_text),
        # the point after the run-ending submission is a terminal exit, not branchable (as in m1 and m3)
        "every_point_before_the_end_branchable": all(k in points for k in range(0, steps)),
        "every_grade_tied_to_its_submission": bool(verdicts) and all(v.get("grade_event_id") in grades for v in verdicts),
        "submission_behaviour_in_status": len((status.get("submission_behaviour") or {}).get("submissions") or [])
                                          == len(verdicts),
        "every_grade_in_the_verifier_image_with_network_none": bool(grade_rows) and all(
            g["verifier_network"] == "none" and g["source_network"] == "none" and g["verifier_image_id"] == verifier_image
            for g in grade_rows),
    }
    return {"key": key, "task": task, "loaded_via": LOADED.get(task), "run_id": rid, "checks": checks,
            "status": {k: status.get(k) for k in ("status", "termination_reason", "steps", "spend_usd_known",
                                                  "solved_by_submission", "checkpoints")},
            "submissions": [{k: v.get(k) for k in ("ordinal", "status", "failing_count", "share_passed", "outcome",
                                                   "step_index")} for v in verdicts],
            "grade_events": len(grade_rows), "hidden_tests": len(names), "reference_scaffold": note,
            "wall_seconds": round(time.monotonic() - t0, 1)}


def main() -> None:
    tasks = sys.argv[1:] or lane_tasks()
    if not tasks:
        raise SystemExit("no VALIDATED or ADMITTED terminalbench-v4 task to run")
    if OUT.exists():
        remove_tree(OUT)
    OUT.mkdir(parents=True)
    status = lane_status()
    report = {"what": __doc__.split("\n\n")[0], "command": "python configs/v2/engine_e2e/m4_lane_v4_natural.py "
              + " ".join(sys.argv[1:]), "registry": str(REGISTRY), "provider_calls": 0, "tasks": tasks,
              "lane_status": {t: status.get(t) for t in tasks}, "runs": {}}
    for key, (task, script, end, note) in scripts(tasks).items():
        report["runs"][key] = check(key, task, script, end, note)
    report["all_checks"] = {f"{k}.{c}": v for k, r in report["runs"].items() for c, v in r["checks"].items()}
    report["passed"] = all(report["all_checks"].values())
    write(ARTIFACTS / "m4-lane-v4-natural.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in report["all_checks"].items() if not v],
                      "runs": {k: {"end": r["status"]["termination_reason"], "loaded_via": r["loaded_via"],
                                   "submissions": r["submissions"], "wall_seconds": r["wall_seconds"]}
                               for k, r in report["runs"].items()}}, indent=1))


if __name__ == "__main__":
    main()
