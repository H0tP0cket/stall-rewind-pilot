"""Shared-runner fixture: graded checkpoints and W+R restore regrade through the actual MiningRunner.

A ScriptedModel (no provider calls) runs the natural start: it inspects /app (the
instruction's paths resolve through the /app -> /workspace symlink), then makes
an innocuous W edit plus a /tmp runtime marker (checkpoint A), then runs the
declared public-only partial control (checkpoint B, a different grade), then
submits. A branch restored from B (W+R, fresh H) is graded at branch start and
must reproduce the source checkpoint's evaluation exactly (per-check statuses,
score, validity) and show the runtime marker. No reference material enters the
scripted trajectory. The detector uses the legacy rule path.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from trajectory_retention.contracts import RunConfig, TreatmentManifest, to_jsonable
from trajectory_retention.detector.core import DetectorConfig
from trajectory_retention.execution.environment import comparable, fidelity_manifest_from_tar
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.mining.audit import audit_run
from trajectory_retention.mining.replay import replay_run
from trajectory_retention.mining.runner import MiningRunner
from trajectory_retention.store import ArtifactStore

from .adapter import TerminalBenchTaskAdapter
from .validate import PARTIALS

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MARKER = "/tmp/terminalbench-fixture/state.txt"


def _step(content: str, commands: list[str]) -> dict:
    return {"content": content, "commands": commands,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0}}


def _signature(evaluation: dict) -> dict:
    return {"valid": evaluation["valid"], "score": evaluation["score"], "full_solve": evaluation["full_solve"],
            "statuses": {c["check_id"]: c["status"] for c in evaluation["checks"]}}


def run_check(package: Path, output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    task = TerminalBenchTaskAdapter(package, scratch_root=output / "private-grading")
    record = task.describe()
    partial = PARTIALS.get(task.upstream_task)
    detector = DetectorConfig(stagnation_token_threshold=1000)
    store = ArtifactStore(output / "store")
    summary = {"task_id": record.task_id, "fixture": True, "provider_calls": 0, "natural_model_outcomes": 0,
               "detector_rule": detector.rule_version, "runs": [], "assertions": {}}

    def launch(run_id, script, treatment=None):
        model = ScriptedModel(script)
        runner = MiningRunner(model_factory=lambda cfg: model, worker_image=record.environment_version,
                              detector_config=detector, scratch_root=output / "scratch", network="none")
        config = RunConfig(run_id=run_id, campaign_id="terminalbench-admission-fixture", purpose="development",
                           model="scripted/fixture", model_settings={}, generated_token_limit=10000,
                           max_model_calls=len(script) + 1, metadata={"fixture": True, "not_natural_mining": True})
        result = runner.run(task, config, store=store, treatment=treatment)
        checkpoints = store.list_checkpoints(run_id)
        events = list(store.iter_events(run_id))
        audit, replay = audit_run(store, run_id, detector), replay_run(store, run_id, detector)
        summary["runs"].append({"run_id": run_id, "status": result.status, "termination": result.termination_reason,
                                "checkpoints": len(checkpoints), "events": len(events), "audit_ok": audit["ok"],
                                "replay_agrees": replay["agrees"]})
        return result, checkpoints, events, audit, replay

    try:
        edit_b = (partial["commands"] if partial else ["printf 'fixture B\\n' > /app/terminalbench_fixture_b.txt"])
        script = [
            _step("Inspect the task directory.", ["pwd; ls -la /app/ | head -n 40; readlink /app"]),
            _step("Fixture edit A.", ["printf 'fixture A\\n' > /app/terminalbench_fixture_a.txt",
                                      f"mkdir -p {Path(MARKER).parent} && printf 'A\\n' > {MARKER}"]),
            _step("Fixture edit B.", edit_b + [f"printf 'B\\n' > {MARKER}"]),
            _step("Submit.", [SUBMIT]),
        ]
        result, cps, events, audit, replay = launch("native", script)
        tool_outputs = [store.get_bytes(e.output_ref).decode(errors="replace") for e in events
                        if e.kind == "tool_result" and e.output_ref]
        graded = [c for c in cps if c.evaluation_ref is not None]
        evaluations = {c.checkpoint_id: store.get_json(c.evaluation_ref) for c in graded}
        mutated = [c for c in graded if "initial" not in c.reasons and "terminal" not in c.reasons]
        a = summary["assertions"]
        a["native_completed"] = result.status == "completed"
        a["native_audit_ok"] = audit["ok"]
        a["native_replay_agrees"] = replay["agrees"]
        a["all_grades_valid"] = all(e["valid"] for e in evaluations.values())
        first = tool_outputs[0] if tool_outputs else ""
        a["app_resolves_to_workspace"] = "/workspace" in first
        summary["visible_start_excerpt"] = first[-800:]
        a["two_mid_run_graded_checkpoints"] = len(mutated) >= 2
        requests = [e for e in events if e.kind == "model_request"]
        encoded = json.dumps([store.get_json(e.input_ref) for e in requests])
        a["private_grade_absent_from_worker_input"] = ("private-grading" not in encoded and "reports.jsonl" not in encoded
                                                       and "test_outputs.py::" not in encoded)
        if len(mutated) >= 2:
            source_a, source_b = mutated[-2], mutated[-1]
            sig_a, sig_b = _signature(evaluations[source_a.checkpoint_id]), _signature(evaluations[source_b.checkpoint_id])
            if partial:
                a["checkpoint_B_grade_differs_from_A"] = sig_a != sig_b
            summary["source_grades"] = {"A": {k: v for k, v in sig_a.items() if k != "statuses"},
                                        "B": {k: v for k, v in sig_b.items() if k != "statuses"}}
            stage = record.stages[0]
            treatment = TreatmentManifest(treatment_id="restore_B", source_checkpoint_id=source_b.checkpoint_id,
                target_id=stage.target_id, stage_id=stage.stage_id, selection_kind="engineering_checkpoint",
                component_donors={"workspace": source_b.checkpoint_id, "runtime": source_b.checkpoint_id,
                                  "history": source_b.checkpoint_id},
                history_policy="fresh", runtime_policy="restore_captured", worker_model="scripted/fixture",
                generated_token_limit=10000, repeat_id="fixture-1")
            bresult, bcps, bevents, baudit, breplay = launch(
                "restore_B", [_step("Inspect restored state.", [f"cat {MARKER}", SUBMIT])], treatment)
            start = next(c for c in bcps if "branch_start" in c.reasons)
            restored = _signature(store.get_json(start.evaluation_ref))
            outputs = [store.get_bytes(e.output_ref).decode(errors="replace") for e in bevents
                       if e.kind == "tool_result" and e.output_ref]
            a["restore_B_completed"] = bresult.status == "completed" and baudit["ok"] and breplay["agrees"]
            a["restore_B_identical_evaluation"] = restored == sig_b
            a["restore_B_workspace_fidelity_equal"] = comparable(fidelity_manifest_from_tar(
                store.get_bytes(start.state.workspace))) == comparable(fidelity_manifest_from_tar(
                store.get_bytes(source_b.state.workspace)))
            a["restore_B_runtime_marker"] = bool(outputs) and outputs[0].strip().endswith("B")
        summary["partial_control_used_for_B"] = bool(partial)
        summary["passed"] = all(a.values())
    except Exception as exc:
        summary["passed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")
    return summary
