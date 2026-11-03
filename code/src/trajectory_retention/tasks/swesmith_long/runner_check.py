"""Shared-runner fixture: visible feedback, graded checkpoints and W+R restore regrade.

A ScriptedModel (no provider calls) runs the task's natural start on the actual
MiningRunner: it runs the repository tests (agent-visible feedback), makes an
innocuous source edit plus a runtime marker (checkpoint A), then breaks the
edited module (checkpoint B, different grade), then submits. A branch restored
from B (W+R, fresh H) is graded at branch start and must reproduce the source
checkpoint's evaluation exactly (per-check statuses, score, validity). The first
tasks checked also restored from A.
No reference material enters the scripted trajectory. The detector uses the
legacy rule path, which needs no task-specific detector declarations.
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

from .adapter import SWESmithLongTaskAdapter

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MARKER = "/tmp/swesmith-long-fixture/state.txt"


def _step(content: str, commands: list[str]) -> dict:
    return {"content": content, "commands": commands,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "cost": 0}}


def _signature(evaluation: dict) -> dict:
    return {"valid": evaluation["valid"], "score": evaluation["score"], "full_solve": evaluation["full_solve"],
            "statuses": {c["check_id"]: c["status"] for c in evaluation["checks"]}}


def run_check(package: Path, output: Path, adapter_cls=SWESmithLongTaskAdapter) -> dict:
    output = Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    task = adapter_cls(package, scratch_root=output / "private-grading")
    withheld = [r["path"] for r in getattr(task, "withheld", {}).get("files", [])]
    record = task.describe()
    f2p_file = task.scorer.f2p[0].split("::", 1)[0]
    edited = task.manifest["bug_shape"]["files"][0]
    detector = DetectorConfig(stagnation_token_threshold=1000)
    store = ArtifactStore(output / "store")
    summary = {"task_id": record.task_id, "fixture": True, "provider_calls": 0, "natural_model_outcomes": 0,
               "detector_rule": detector.rule_version, "runs": [], "assertions": {}}

    def launch(run_id, script, treatment=None):
        model = ScriptedModel(script)
        runner = MiningRunner(model_factory=lambda cfg: model, worker_image=record.environment_version,
                              detector_config=detector, scratch_root=output / "scratch", network="none")
        config = RunConfig(run_id=run_id, campaign_id="swesmith-long-admission-fixture", purpose="development",
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
        if withheld:
            # Native SWE-smith start: the FAIL_TO_PASS files are absent; the worker's
            # visible feedback is the remaining repository tests.
            visible = sorted({c.split("::", 1)[0] for c in task.scorer.p2p} - set(withheld))
            probe = (f"pytest -q -p no:cacheprovider {visible[0]} 2>&1 | tail -n 12" if visible else
                     "pytest -q -p no:cacheprovider --co 2>&1 | tail -n 5")
            first = [f"for f in {' '.join(withheld)}; do test -e \"$f\" && echo WITHHELD_PRESENT \"$f\"; done; "
                     "echo WITHHELD_CHECK_DONE; " + probe]
        else:
            first = [f"pytest -q -p no:cacheprovider {f2p_file} 2>&1 | tail -n 12"]
        git_start = getattr(task, "git_start", None)
        if git_start:
            # v3: W is a Git repository. A read-only `git status` must neither dirty W
            # nor rewrite .git/index (otherwise it would count as a W mutation).
            first = ["echo GIT_INDEX_BEFORE=$(sha256sum .git/index | cut -c1-64); "
                     "echo GIT_STATUS_LINES=$(git status --porcelain=v1 --untracked-files=all | wc -l); "
                     "echo GIT_COMMITS=$(git rev-list --all --count) GIT_HEAD=$(git rev-parse HEAD); "
                     "echo GIT_INDEX_AFTER=$(sha256sum .git/index | cut -c1-64)"] + first
        git_status = ("git status --porcelain=v1 --untracked-files=all; "
                      "echo GIT_COMMITS=$(git rev-list --all --count) GIT_STASHES=$(git stash list | wc -l)")
        script = [
            _step("Run the repository tests.", first),
            _step("Fixture edit A.", [f"printf '\\n# swesmith-long runner fixture edit\\n' >> {edited}",
                                      f"mkdir -p {Path(MARKER).parent} && printf 'A\\n' > {MARKER}"]),
            _step("Fixture edit B.", [f"printf '\\nraise RuntimeError(\"swesmith-long runner fixture\")\\n' >> {edited}",
                                      f"printf 'B\\n' > {MARKER}"] + ([git_status] if git_start else [])),
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
        if git_start:
            import re
            probe_out = tool_outputs[0] if tool_outputs else ""
            before = re.search(r"GIT_INDEX_BEFORE=(\w+)", probe_out)
            after = re.search(r"GIT_INDEX_AFTER=(\w+)", probe_out)
            a["git_status_clean_at_start_in_worker"] = "GIT_STATUS_LINES=0" in probe_out
            a["git_one_commit_at_start_in_worker"] = f"GIT_COMMITS=1 GIT_HEAD={git_start['commit']}" in probe_out
            a["git_status_leaves_index_unchanged"] = bool(before and after and before.group(1) == after.group(1))
            summary["git_probe_excerpt"] = probe_out[-600:]
            tool_outputs = tool_outputs[1:]
        feedback = tool_outputs[0] if tool_outputs else ""
        if withheld:
            import re
            a["withheld_f2p_files_absent_in_worker"] = ("WITHHELD_CHECK_DONE" in feedback
                                                       and "WITHHELD_PRESENT" not in feedback)
            summary["visible_test_files"] = len(visible)
            # gpxpy-style repositories have a single test file; natively nothing visible remains.
            a["visible_repository_tests_run"] = bool(re.search(
                r"\d+ (passed|failed)|\d+ tests? collected|collected \d+" + ("" if visible else r"|no tests collected"),
                feedback))
        else:
            a["visible_test_feedback_shows_failures"] = "failed" in feedback.lower()
        summary["visible_feedback_excerpt"] = feedback[-800:]
        a["two_mid_run_graded_checkpoints"] = len(mutated) >= 2
        requests = [e for e in events if e.kind == "model_request"]
        encoded = json.dumps([store.get_json(e.input_ref) for e in requests])
        a["private_grade_absent_from_worker_input"] = ("private-grading" not in encoded and "pristine" not in encoded
                                                       and "reports.jsonl" not in encoded)
        if len(mutated) >= 2:
            source_a, source_b = mutated[-2], mutated[-1]  # after fixture edits A and B
            sig_a, sig_b = _signature(evaluations[source_a.checkpoint_id]), _signature(evaluations[source_b.checkpoint_id])
            a["checkpoint_B_grade_differs_from_A"] = sig_a != sig_b
            summary["source_grades"] = {"A": {k: v for k, v in sig_a.items() if k != "statuses"},
                                        "B": {k: v for k, v in sig_b.items() if k != "statuses"}}
            # One restored branch (from B, whose grade differs from the start) keeps
            # the fixture short on a loaded shared Docker host.
            for label, source, expected_marker in (("B", source_b, "B"),):
                stage = record.stages[0]
                treatment = TreatmentManifest(treatment_id="restore_" + label, source_checkpoint_id=source.checkpoint_id,
                    target_id=stage.target_id, stage_id=stage.stage_id, selection_kind="engineering_checkpoint",
                    component_donors={"workspace": source.checkpoint_id, "runtime": source.checkpoint_id,
                                      "history": source.checkpoint_id},
                    history_policy="fresh", runtime_policy="restore_captured", worker_model="scripted/fixture",
                    generated_token_limit=10000, repeat_id="fixture-1")
                inspect = [f"cat {MARKER}"]
                if git_start:
                    inspect.append(git_status)
                bresult, bcps, bevents, baudit, breplay = launch(
                    "restore_" + label, [_step("Inspect restored state.", inspect + [SUBMIT])], treatment)
                start = next(c for c in bcps if "branch_start" in c.reasons)
                restored = _signature(store.get_json(start.evaluation_ref))
                original = sig_a if label == "A" else sig_b
                outputs = [store.get_bytes(e.output_ref).decode(errors="replace") for e in bevents
                           if e.kind == "tool_result" and e.output_ref]
                a[f"restore_{label}_completed"] = bresult.status == "completed" and baudit["ok"] and breplay["agrees"]
                a[f"restore_{label}_identical_evaluation"] = restored == original
                a[f"restore_{label}_workspace_fidelity_equal"] = comparable(fidelity_manifest_from_tar(
                    store.get_bytes(start.state.workspace))) == comparable(fidelity_manifest_from_tar(
                    store.get_bytes(source.state.workspace)))
                a[f"restore_{label}_runtime_marker"] = bool(outputs) and outputs[0].strip().endswith(expected_marker)
                if git_start:
                    # Restored W is the same repository in the same state as the source at B.
                    at_source = next((o for o in tool_outputs if "GIT_STASHES=" in o), None)
                    at_branch = outputs[1] if len(outputs) > 1 else None
                    a[f"restore_{label}_git_status_equals_source"] = (at_source is not None and at_source == at_branch
                        and f" M {edited}" in at_source.splitlines() and "GIT_COMMITS=1 GIT_STASHES=0" in at_source)
                    summary[f"restore_{label}_git_status"] = {"source": at_source, "branch": at_branch}
        summary["passed"] = all(a.values())
    except Exception as exc:
        summary["passed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2) + "\n")
    return summary
