"""Per-task restore check through the real MiningRunner with a ScriptedModel (no provider calls).

1. A natural scripted run on the package: step 1 changes W (a new directory and file under /app) and R (a marker
   under /tmp and an executable under /usr/local/bin); step 2 runs the probe command (a full W listing with modes,
   owners, sizes and mtimes, a hash of every W file, the R marker, its listing, the R executable, the worker identity);
   then it submits three times. Each submission is graded on the captured W and R, the worker gets only the
   failing-count feedback, and the run ends on the third identical failing set (the lane's submission protocol).
2. The checkpoint for point 1 (the last branchable checkpoint before the step-2 model call) is restored into a new
   container through the runner's resume path (W, R and the context), and the scripted model re-runs the probe.
3. The probe's raw output and its delivered observation must be byte-identical to step 2 of the source run. The
   branch-start grade of the restored state must equal the source checkpoint's grade when that checkpoint was graded.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from trajectory_retention.contracts import RunConfig, to_jsonable
from trajectory_retention.detector.core import DetectorConfig
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.mining.runner import MiningRunner
from trajectory_retention.store import ArtifactStore

from .adapter import TerminalBenchV2TaskAdapter

VERSION = "terminalbench-v2.restore-check.v1"
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SETUP = ("mkdir -p /app/.tbv2-restore/sub && printf 'W marker\\n' > /app/.tbv2-restore/sub/w.txt && "
         "mkdir -p /tmp/tbv2-restore && printf 'R marker\\n' > /tmp/tbv2-restore/marker && "
         "{ printf '#!/bin/sh\\necho r-tool-ok\\n' > /usr/local/bin/tbv2-tool && chmod 755 /usr/local/bin/tbv2-tool "
         "|| echo 'no root: R tool skipped'; }")
PROBE = ("cd /app && find . -xdev -printf '%M %u %g %s %TY-%Tm-%Td %TT %p -> %l\\n' | LC_ALL=C sort; "
         "echo '--- W content'; find . -xdev -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum; "
         "echo '--- R'; cat /tmp/tbv2-restore/marker; "
         "find /tmp/tbv2-restore -printf '%M %u %g %s %TY-%Tm-%Td %TT %p\\n' | LC_ALL=C sort; "
         "command -v tbv2-tool >/dev/null 2>&1 && tbv2-tool; echo '--- identity'; id; pwd -P")
DETECTOR = DetectorConfig(stagnation_token_threshold=10000, rule_version="stall-clock/v8.1")


def _step(content: str, command: str) -> dict:
    return {"content": content, "commands": [command],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0}}


def _signature(evaluation: dict) -> dict:
    return {"valid": evaluation["valid"], "score": evaluation["score"], "full_solve": evaluation["full_solve"],
            "statuses": {c["check_id"]: c["status"] for c in evaluation["checks"]}}


def _probe_outputs(store: ArtifactStore, run_id: str) -> list[dict]:
    events = list(store.iter_events(run_id))
    calls = {e.event_id: e for e in events if e.kind == "tool_call" and e.payload.get("command") == PROBE}
    results = {e.event_id: e for e in events if e.kind == "tool_result" and e.parent_event_id in calls and e.output_ref}
    delivered = {e.parent_event_id: e for e in events if e.kind == "observation_delivery" and e.parent_event_id in results}
    out = []
    for rid, result in results.items():
        delivery = delivered.get(rid)
        out.append({"sequence": result.sequence, "raw": store.get_bytes(result.output_ref).decode(errors="replace"),
                    "delivered": store.get_bytes(delivery.delivered_ref).decode(errors="replace") if delivery else None,
                    "returncode": result.payload.get("returncode")})
    return sorted(out, key=lambda r: r["sequence"])


def run_check(package: Path, output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    task = TerminalBenchV2TaskAdapter(package, scratch_root=output / "private-grading")
    record = task.describe()
    store = ArtifactStore(output / "store")
    summary = {"version": VERSION, "task_id": record.task_id, "provider_calls": 0, "detector_rule": DETECTOR.rule_version,
               "probe_command": PROBE, "setup_command": SETUP, "runs": [], "assertions": {}}

    def config(run_id: str, calls: int) -> RunConfig:
        return RunConfig(run_id=run_id, campaign_id="terminalbench-v2-restore-check", purpose="development",
                         model="scripted/fixture", model_settings={}, generated_token_limit=100000, max_model_calls=calls,
                         detector_settings=DETECTOR.to_record(), metadata={"fixture": True, "not_natural_mining": True})

    def runner(script: list[dict]) -> MiningRunner:
        model = ScriptedModel(script)
        return MiningRunner(model_factory=lambda cfg: model, worker_image=record.environment_version,
                            detector_config=DETECTOR, scratch_root=output / "scratch", network="none")

    try:
        source_script = [_step("Set up W and R state.", SETUP), _step("Probe the state.", PROBE),
                         _step("Submit.", SUBMIT), _step("Submit again.", SUBMIT), _step("Submit a third time.", SUBMIT)]
        source = runner(source_script).run(task, config("source", 8), store=store)
        events = list(store.iter_events("source"))
        verdicts = [e.payload for e in events if e.kind == "lifecycle"
                    and e.payload.get("transition") == "native_submission_verdict"]
        requests = [e for e in events if e.kind == "model_request"]
        checkpoints = store.list_checkpoints("source")
        summary["runs"].append({"run_id": "source", "status": source.status, "termination": source.termination_reason,
                                "checkpoints": len(checkpoints), "submissions": len(verdicts)})
        a = summary["assertions"]
        a["source_ended_on_three_identical_failing_sets"] = (source.status == "completed"
            and len(verdicts) == 3 and verdicts[-1].get("outcome") == "native_repeated_failure"
            and len({v.get("criterion_id") for v in verdicts}) == 1)
        graded = {c.checkpoint_id: c for c in checkpoints if c.evaluation_ref is not None}
        grades = [store.get_json(graded[v["checkpoint_id"]].evaluation_ref) if v["checkpoint_id"] in graded else {}
                  for v in verdicts]
        a["every_submission_graded_on_captured_w_and_r"] = bool(grades) and all(
            (g.get("details") or {}).get("runtime_export_consumed") is True for g in grades)
        template = task.native_submission_protocol["feedback_template"]
        expected = [template.replace("{n}", str(len(g["details"]["failing_tests"])))
                    for v, g in zip(verdicts, grades) if v.get("outcome") == "continue"]
        history = store.get_json(checkpoints[-1].state.history)["messages"]
        users = [m["content"] for m in history if m.get("role") == "user"]
        summary["feedback_shown"] = users[1:]
        a["feedback_is_count_only"] = len(expected) == 2 and users[1:] == expected
        summary["submission_grades"] = [{**{k: v.get(k) for k in ("ordinal", "status", "consecutive_same", "checkpoint_id",
                                                                   "evaluation_id", "outcome")},
                                         "share_passed": g.get("score"),
                                         "failing_count": len((g.get("details") or {}).get("failing_tests") or [])}
                                        for v, g in zip(verdicts, grades)]
        step2 = requests[1].sequence
        point1 = [c for c in checkpoints if c.branchable and c.event_sequence < step2]
        if not point1:
            raise RuntimeError("no branchable checkpoint before the step-2 model call")
        source_cp = point1[-1]
        summary["restored_checkpoint"] = {"checkpoint_id": source_cp.checkpoint_id, "reasons": list(source_cp.reasons),
                                          "graded": source_cp.evaluation_ref is not None}
        replay = runner([_step("Probe the restored state.", PROBE), _step("Submit.", SUBMIT), _step("Submit again.", SUBMIT),
                         _step("Submit a third time.", SUBMIT)]).run(task, config("replay", 6), store=store, resume=source_cp)
        summary["runs"].append({"run_id": "replay", "status": replay.status, "termination": replay.termination_reason,
                                "checkpoints": len(store.list_checkpoints("replay"))})
        original, again = _probe_outputs(store, "source"), _probe_outputs(store, "replay")
        a["probe_ran_in_both"] = len(original) == 1 and len(again) == 1
        if a["probe_ran_in_both"]:
            o, r = original[0], again[0]
            a["probe_raw_output_identical"] = o["raw"] == r["raw"]
            a["probe_delivered_output_identical"] = o["delivered"] == r["delivered"]
            a["probe_saw_w_and_r"] = ("W content" in o["raw"] and "R marker" in o["raw"] and ".tbv2-restore/sub/w.txt" in o["raw"])
            summary["probe_output_excerpt"] = o["raw"][-1500:]
            summary["probe_r_tool"] = "r-tool-ok" in o["raw"]
            if not a["probe_raw_output_identical"]:
                summary["probe_diff"] = {"source": o["raw"][-4000:], "replay": r["raw"][-4000:]}
        start = next((c for c in store.list_checkpoints("replay") if "branch_start" in c.reasons
                      or "initial" in c.reasons), None)
        if source_cp.evaluation_ref is not None and start is not None and start.evaluation_ref is not None:
            a["restored_grade_identical"] = (_signature(store.get_json(start.evaluation_ref))
                                             == _signature(store.get_json(source_cp.evaluation_ref)))
        ops = [e for e in store.iter_events("replay") if e.kind == "operation"
               and e.payload.get("name") in ("workspace_restore", "runtime_restore", "restored_session_prepare")]
        summary["restore_seconds"] = {e.payload["name"]: e.usage.elapsed_seconds for e in ops if e.usage}
        summary["passed"] = all(a.values())
    except Exception as exc:  # recorded, never raised past the gate
        summary["passed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")
    return summary
