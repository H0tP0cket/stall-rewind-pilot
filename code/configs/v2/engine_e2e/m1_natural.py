"""Engine milestone 1 end-to-end check: natural-v2 runs on a real Terminal-Bench task in Docker (scripted model).

Every natural run goes through the driver (configs/v2/natural.py) on terminalbench-llm-inference-batching-scheduler-v1
(the terminalbench-v1 package, sealed sources untouched) wrapped in the count-feedback fixture protocol:
  A loop        a format error, a no-change step, a W+R change (the public partial control: 5 of 6 tests pass), a
                create/delete inside one step, a per-command timeout, high token use so detector v8.1 warns, and
                three identical failing submissions (the run ends there)
  B last turn   step limit 3 with the submission on call 3: verdict and feedback before the step guard
  C cost guard  $1.60 per call, the submission on the call that crosses $3: verdict before the cost guard
  D pass        the reference solution left unsubmitted (a private full solve, only logged), then a passing submission
Then: the driver's claims, STOP file, status and spend files; a v2 config with a wall-clock cap refused before any
evidence; restores from three recorded points through the existing branch path; postrun with a scripted judge on run
A's store (the durable stall_confirmation record); and the step overhead of the earlier path against natural-v2 on one
24-step script, plus natural-v2 overhead growth over a 120-step script.

Usage: python configs/v2/engine_e2e/m1_natural.py [--skip-overhead]
Writes artifacts/v2/engine/m1-natural.json in the main checkout; stores stay under ~/.cache/trajectory-retention/v2-engine/m1.
No provider calls.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import (ARTIFACTS, PACKAGE, PARTIAL, REGISTRY_V1, ROOT, SCRATCH, SUBMIT, TASK, events, lifecycle,
                    requests, stats, step, step_overheads, write)
from trajectory_retention.accounting import CostLedger
from trajectory_retention.contracts import ArtifactRef, RunConfig, TreatmentManifest, from_jsonable
from trajectory_retention.detector.core import DetectorConfig
from trajectory_retention.execution.environment import comparable, fidelity_manifest_from_tar
from trajectory_retention.execution.model import ScriptedModel, history_replay_issue
from trajectory_retention.experiments.v2_fixture import CountFeedbackFixture
from trajectory_retention.experiments.v2_text import SUBMIT_RULE, SUBMISSION_FEEDBACK
from trajectory_retention.mining.points import checkpoint_step_index, point_checkpoints
from trajectory_retention.mining.runner import MiningRunner
from trajectory_retention.registry import read_registry
from trajectory_retention.store import ArtifactStore

OUT = SCRATCH / "m1"
DRIVER = ROOT / "configs/v2/natural.py"
POSTRUN = ROOT / "configs/detector-judge-20260926/postrun.py"
_spec = importlib.util.spec_from_file_location("natural", DRIVER)
natural = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural)
FIXTURE = "count-feedback-fixture/v1"
TAG = time.strftime("%Y%m%dT%H%M%S")  # fresh identities for the direct (non-driver) runs of each invocation


def make_plan(name: str, script: list[dict], *, step_limit=500, cost=3.0, tool_timeout=15, repeats=1) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    write(d / "script.json", script)
    plan = {"version": "natural-plan.v2", "plan_id": f"m1-{name}", "registry": str(REGISTRY_V1), "tasks": [TASK],
            "repeats": repeats, "model": "scripted/fixture", "settings": {}, "store": str(d / "run"), "concurrency": 1,
            "detector": {"rule": "stall-clock/v8.1"}, "budget": {"cost_limit_usd": cost, "step_limit": step_limit},
            "purpose": "stall_supply", "tool_timeout_seconds": tool_timeout,
            "fixture": {"submission_protocol": FIXTURE, "scripts": {"*": str(d / "script.json")}}}
    write(d / "plan.json", plan)
    return d / "plan.json"


def drive(plan: Path, *extra: str) -> dict:
    proc = subprocess.run([sys.executable, str(DRIVER), "--plan", str(plan), *extra], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"driver failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def store_of(name: str) -> ArtifactStore:
    return ArtifactStore(OUT / name / "run" / "store")


def run_id(name: str) -> str:
    return job_of(name)["run_id"]


def job_of(name: str) -> dict:
    return natural.jobs(natural.load_plan(OUT / name / "plan.json"))[0]


def verdicts(evs):
    return [e for e in lifecycle(evs, "native_submission_verdict")]


def common_checks(store, rid, evs) -> dict:
    """Checks every natural-v2 run must pass."""
    points = point_checkpoints(store, rid)
    cps = sorted(store.list_checkpoints(rid), key=lambda c: c.ordinal)
    n_steps = sum(e.kind == "model_response" for e in evs)
    reqs = requests(store, evs)
    first_user = json.loads(reqs[0])["messages"][1]["content"]
    grades = {e.payload["checkpoint_id"]: e for e in evs if e.kind == "grade"}
    sub_ok = []
    for v in verdicts(evs):
        p = v.payload
        g = grades.get(p["checkpoint_id"])
        cp = store.get_checkpoint(p["checkpoint_id"])
        binding = store.get_json(from_jsonable(ArtifactRef, store.get_json(cp.evaluation_ref)["details"][
            "source_binding_ref"])) if cp.evaluation_ref else {}
        sub_ok.append(g is not None and g.event_id == p["grade_event_id"] and g.payload["runtime_export_consumed"] is True
                      and binding.get("capture_event_sequence") == cp.event_sequence and cp.state.runtime is not None)
    plain = CostLedger(store)
    snapshots_identical = all(store.get_json(c.accounting_ref) == store.get_json(
        plain.snapshot(rid, through_sequence=c.event_sequence)) for c in cps)
    history_k = {}
    for k, cid in points.items():
        cp = store.get_checkpoint(cid)
        h = store.get_json(cp.state.history)
        history_k[k] = (h["counters"]["n_calls"] == k and history_replay_issue(h["messages"]) is None
                        and cp.validation["step_index"] == k)
    return {
        "points": {str(k): v for k, v in points.items()},
        "checks": {
            "instance_template_v2": SUBMIT_RULE in first_user and "Submission ends work" not in first_user,
            "every_point_checkpointed_and_branchable": set(points) >= set(range(n_steps)),
            "point_k_history_has_k_calls": all(history_k.values()),
            "occurrences_never_merged": len({c.checkpoint_id for c in cps}) == len(cps)
                                        and [c.ordinal for c in cps] == list(range(len(cps))),
            "no_full_per_tool_captures": all(e.payload.get("status") == "manifest" for e in evs if e.kind == "tool_state"),
            "submissions_graded_on_captured_W_and_R": all(sub_ok),
            "no_wall_clock_cap": lifecycle(evs, "run_started")[0].payload["config"]["wall_clock_limit_seconds"] is None,
            "private_grades_absent_from_worker_requests": not any("test_outputs.py::" in r or "private-grading" in r
                                                                  for r in reqs),
            "accounting_snapshots_identical_to_plain_ledger": snapshots_identical,
        },
        "steps": n_steps, "checkpoints": len(cps), "graded_checkpoints": sum(c.evaluation_ref is not None for c in cps),
    }


def feedback_in_next_request(store, evs) -> list[bool]:
    reqs = [e for e in evs if e.kind == "model_request"]
    out = []
    for fb in [e for e in evs if e.kind == "native_submission_feedback"]:
        text = store.get_bytes(fb.delivered_ref).decode()
        nxt = next((r for r in reqs if r.sequence > fb.sequence), None)
        out.append(nxt is not None and json.loads(store.get_bytes(nxt.input_ref))["messages"][-1]["content"] == text)
    return out


def check_A(report: dict) -> None:
    store, rid = store_of("A"), run_id("A")
    evs = events(store, rid)
    base = common_checks(store, rid, evs)
    vs = [v.payload for v in verdicts(evs)]
    muts = [e.payload for e in evs if e.kind == "mutation"]
    within = [m for m in muts if m["within_batch_mutation"]]
    tool_results = [e for e in evs if e.kind == "tool_result"]
    timeouts = [e for e in tool_results if e.payload.get("status") == "timeout"]
    points = point_checkpoints(store, rid)
    post_feedback = {}
    for v in vs[:-1]:
        cp = store.get_checkpoint(points[v["step_index"]])
        last = store.get_json(cp.state.history)["messages"][-1]
        post_feedback[v["step_index"]] = ("post_feedback" in cp.reasons and last["role"] == "user"
                                          and last["content"] == SUBMISSION_FEEDBACK.format(n=v["failing_count"]))
    ungraded_noop = [store.get_checkpoint(points[k]).evaluation_ref is None for k in (1, 2)]
    annotations = [store.get_record("annotation", r["record_id"]) for r in store.list_records("annotation", run_id=rid)]
    warnings = [a for a in annotations if a.get("kind") == "S"]
    terminal = next(e.payload for e in reversed(evs) if e.kind == "terminal")
    checks = dict(base["checks"])
    checks.update({
        "format_error_is_step_1": points.get(1) is not None and any(e.kind == "format_error" for e in evs),
        "no_change_points_ungraded": all(ungraded_noop),
        "create_delete_in_one_step_kept_and_graded": bool(within) and all(
            any(a["changed"] for a in m["per_action"]) for m in within),
        "per_command_timeout_kept_run_continued": bool(timeouts) and base["steps"] > 5,
        "count_feedback_in_next_request": feedback_in_next_request(store, evs) == [True, True],
        "post_feedback_point_is_branchable_with_feedback_last": bool(post_feedback) and all(post_feedback.values()),
        "three_identical_failing_sets_end_run": [v["consecutive_same"] for v in vs] == [1, 2, 3]
                                                and terminal["reason"] == "sequence_complete:native_repeated_failure"
                                                and base["steps"] == 11,
        "detector_v81_warned_live": bool(warnings),
        "points_after_hard_links_branchable": all(k in points for k in range(7, base["steps"])),
    })
    report["A_loop"] = {**base, "checks": checks, "termination": terminal["reason"],
        "submissions": [{k: v[k] for k in ("step_index", "status", "failing_count", "share_passed", "consecutive_same",
                                           "checkpoint_id", "grade_event_id")} for v in vs],
        "within_batch_mutation": [{"per_action": m["per_action"], "endpoint_changed": bool(
            m["workspace_fidelity_paths_changed"] or m["runtime_paths_changed"])} for m in within],
        "timeout_result": timeouts[0].payload if timeouts else None,
        "warnings": [{"annotation_id": w["annotation_id"], "checkpoint_id": w["checkpoint_id"],
                      "step_index": checkpoint_step_index(store, store.get_checkpoint(w["checkpoint_id"])),
                      "trigger": w["evidence"].get("trigger")} for w in warnings],
        "feedback_texts": [store.get_bytes(e.delivered_ref).decode() for e in evs if e.kind == "native_submission_feedback"]}


def guard_check(name: str, guard: str, reason: str, n_requests: int) -> dict:
    store, rid = store_of(name), run_id(name)
    evs = events(store, rid)
    base = common_checks(store, rid, evs)
    v = verdicts(evs)
    g = [e for e in lifecycle(evs, "run_guard_boundary") if e.payload["guard"] == guard]
    fb = [e for e in evs if e.kind == "native_submission_feedback"]
    terminal = next(e.payload for e in reversed(evs) if e.kind == "terminal")
    checks = dict(base["checks"])
    checks.update({
        "submission_on_last_allowed_turn_graded": len(v) == 1 and v[0].payload["step_index"] == base["steps"],
        "verdict_and_feedback_before_guard": bool(g and fb) and v[0].sequence < fb[0].sequence < g[0].sequence,
        "guard_ends_run": terminal["reason"] == reason,
        "no_call_after_guard": sum(e.kind == "model_request" for e in evs) == n_requests,
        "post_feedback_point_exists": base["steps"] in point_checkpoints(store, rid),
    })
    return {**base, "checks": checks, "termination": terminal["reason"], "guard": g[0].payload if g else None,
            "verdict": {k: v[0].payload[k] for k in ("step_index", "status", "failing_count", "outcome")} if v else None}


def check_D(report: dict) -> None:
    store, rid = store_of("D"), run_id("D")
    evs = events(store, rid)
    base = common_checks(store, rid, evs)
    logged = lifecycle(evs, "private_full_solve_logged")
    v = verdicts(evs)
    terminal = next(e.payload for e in reversed(evs) if e.kind == "terminal")
    checks = dict(base["checks"])
    checks.update({
        "private_full_solve_only_logged": len(logged) >= 1 and logged[0].payload["step_index"] == 1 and base["steps"] == 3,
        "passing_submission_ends_run": len(v) == 1 and v[0].payload["status"] == "pass"
                                       and terminal["reason"] == "sequence_complete:native_success",
    })
    report["D_pass"] = {**base, "checks": checks, "termination": terminal["reason"],
                        "private_full_solve": logged[0].payload if logged else None}


def check_driver(report: dict) -> None:
    out = {}
    plan_a = OUT / "A" / "plan.json"
    before = len(events(store_of("A"), run_id("A")))
    again = drive(plan_a)
    out["rerun_starts_nothing"] = (len(events(store_of("A"), run_id("A"))) == before
                                   and again["by_state"] == {"finished": 1})
    dry = drive(plan_a, "--dry-run")
    out["dry_run_lists_jobs"] = dry["jobs"][0]["run_id"] == run_id("A") and dry["jobs"][0]["claimed"] is True
    stop_plan = make_plan("STOP", [step("x", ["ls"])], repeats=2)
    (OUT / "STOP" / "run").mkdir(parents=True, exist_ok=True)
    (OUT / "STOP" / "run" / "STOP").write_text("stop\n")
    stopped = drive(stop_plan)
    out["stop_file_starts_nothing"] = stopped["by_state"] == {"pending": 2} and not any((OUT / "STOP/run/claims").iterdir())
    spend = json.loads((OUT / "C" / "run" / "spend.json").read_text())
    status_c = json.loads((OUT / "C" / "run" / "status" / f"{job_of('C')['key']}.json").read_text())
    out["spend_file_totals"] = abs(spend["spend_usd_known"] - 3.2) < 1e-9 and status_c["state"] == "finished"
    out["status_file_has_submissions"] = len(status_c["submissions"]) == 1
    report["driver"] = {"checks": out, "spend_C": spend, "status_C": status_c, "dry_run": dry}


def check_wall_cap(report: dict) -> None:
    plan = natural.load_plan(OUT / "A" / "plan.json")
    job = dict(natural.jobs(plan)[0], run_id=f"m1-wallcap-{TAG}:x:natural")
    config = dataclasses.replace(natural.run_config(plan, job), wall_clock_limit_seconds=600.0)
    task = CountFeedbackFixture(read_registry(REGISTRY_V1)[TASK].load(scratch_root=OUT / "wallcap-task",
                                                                      purpose="development"))
    store = ArtifactStore(OUT / "wallcap-store")
    runner = MiningRunner(model_factory=lambda c: ScriptedModel([]), worker_image=task.describe().environment_version,
                          detector_config=DetectorConfig(**config.detector_settings), scratch_root=OUT / "wallcap-scratch",
                          network="none")
    try:
        runner.run(task, config, store=store)
        refused = None
    except ValueError as exc:
        refused = str(exc)
    report["wall_cap"] = {"checks": {"v2_wall_cap_refused_before_evidence": refused is not None
                                     and not events(store, config.run_id)}, "error": refused}


def check_restores(report: dict) -> None:
    """Branch from three recorded points of run A through the existing restore path (W, R and context from the point)."""
    store, rid = store_of("A"), run_id("A")
    plan = natural.load_plan(OUT / "A" / "plan.json")
    points = point_checkpoints(store, rid)
    source_cfg = natural.run_config(plan, natural.jobs(plan)[0])
    vs = [v.payload for v in verdicts(events(store, rid))]
    chosen = {"point_0": 0, "after_create_delete": 4, "after_first_feedback": vs[0]["step_index"]}
    task = CountFeedbackFixture(read_registry(REGISTRY_V1)[TASK].load(scratch_root=OUT / "restore-task",
                                                                      purpose="development"))
    stage = task.describe().stages[0]
    out = {}
    for label, k in chosen.items():
        source = store.get_checkpoint(points[k])
        brid = f"m1-restore-{label}-{TAG}"
        treatment = TreatmentManifest(treatment_id=brid, source_checkpoint_id=source.checkpoint_id,
            target_id=stage.target_id, stage_id=stage.stage_id, selection_kind="engineering_checkpoint",
            component_donors={c: source.checkpoint_id for c in ("workspace", "runtime", "history")},
            history_policy="native", runtime_policy="restore_captured", worker_model="scripted/fixture",
            generated_token_limit=natural.NO_TOKEN_CAP, repeat_id="m1")
        config = RunConfig(run_id=brid, campaign_id="m1-restore", purpose="development", model="scripted/fixture",
            model_settings={}, generated_token_limit=natural.NO_TOKEN_CAP, max_model_calls=1,
            tool_timeout_seconds=source_cfg.tool_timeout_seconds, detector_settings=source_cfg.detector_settings,
            metadata={"fixture": True, "engine_check": "restore from a natural-v2 point"})
        model = ScriptedModel([step("Check the restored state.", [
            "cat /tmp/v2e2e/marker 2>&1; ls /app/task_file/output_data 2>&1; ls /app/v2e2e_scratch.txt 2>&1; "
            "stat -c 'links %h %n' /tmp/v2e2e/marker.hl /app/v2e2e_h2.txt 2>&1"])],
            model_name="scripted/fixture")
        runner = MiningRunner(model_factory=lambda c: model, worker_image=task.describe().environment_version,
                              detector_config=DetectorConfig(**config.detector_settings),
                              scratch_root=OUT / "restore-scratch", network="none")
        t0 = time.monotonic()
        result = runner.run(task, config, store=store, treatment=treatment)
        bevs = events(store, brid)
        restored = next(e.payload for e in lifecycle(bevs, "history_restored"))
        start = next(c for c in store.list_checkpoints(brid) if "branch_start" in c.reasons)
        request = json.loads(requests(store, bevs)[0])
        source_messages = [{k: v for k, v in m.items() if k != "extra"}
                           for m in store.get_json(source.state.history)["messages"]]
        output = next(store.get_bytes(e.output_ref).decode() for e in bevs if e.kind == "tool_result")
        ops = {e.payload["name"]: e.usage.elapsed_seconds for e in bevs if e.kind == "operation" and e.usage}
        out[label] = {"k": k, "source_checkpoint_id": source.checkpoint_id, "status": result.status,
            "history_validation": restored["validation"],
            "workspace_equal": comparable(fidelity_manifest_from_tar(store.get_bytes(start.state.workspace)))
                               == comparable(fidelity_manifest_from_tar(store.get_bytes(source.state.workspace))),
            "next_request_messages_equal_source_context": request["messages"] == source_messages,
            "first_output": output[-400:],
            "restore_seconds": round(sum(ops.get(n, 0.0) for n in ("workspace_restore_input", "workspace_restore",
                "runtime_restore_input", "runtime_restore", "history_restore_input", "restored_session_prepare")), 3),
            "branch_wall_seconds": round(time.monotonic() - t0, 3)}
    marker = {"point_0": "No such file", "after_create_delete": "A", "after_first_feedback": "B"}
    checks = {f"{label}_restored": (r["status"] == "completed" and all(v for kk, v in r["history_validation"].items()
                                                                       if kk != "revision")
                                    and r["workspace_equal"] and r["next_request_messages_equal_source_context"]
                                    and marker[label] in r["first_output"].splitlines()[0])
              for label, r in out.items()}
    links = out["after_first_feedback"]["first_output"]
    checks["after_first_feedback_hard_links_restored"] = ("links 2 /tmp/v2e2e/marker.hl" in links
                                                          and "links 2 /app/v2e2e_h2.txt" in links)
    report["restores"] = {"checks": checks, "branches": out}


def overhead(report: dict) -> None:
    """The earlier path (per-tool full captures, checkpoints on change) against natural-v2 on one scripted run."""
    task = CountFeedbackFixture(read_registry(REGISTRY_V1)[TASK].load(scratch_root=OUT / "overhead-task",
                                                                      purpose="development"))
    store = ArtifactStore(OUT / "overhead-store")
    plan = natural.load_plan(OUT / "A" / "plan.json")
    script = []
    for i in range(24):
        if i % 3 == 0:
            script.append(step(f"note {i}", [f"echo line{i} >> /app/v2e2e_notes.txt"], tokens=300))
        elif i % 3 == 1:
            script.append(step(f"look {i}", ["ls /app/task_file > /dev/null && echo ok"], tokens=300))
        else:
            script.append(step(f"runtime {i}", [f"echo r{i} > /tmp/v2e2e_r.txt", "cat /tmp/v2e2e_r.txt"], tokens=300))
    long_script = [step(f"tick {i}", [f"echo tick{i}"], tokens=50) for i in range(120)]
    out = {}
    for label, entries, v2 in (("earlier_path", script, False), ("natural_v2", script, True),
                               ("natural_v2_120_steps", long_script, True)):
        job = dict(natural.jobs(plan)[0], run_id=f"m1-overhead-{label}-{TAG}")
        config = natural.run_config(plan, job)
        meta = dict(config.metadata) if v2 else {"fixture": True, "engine_check": "overhead, earlier path"}
        config = dataclasses.replace(config, max_model_calls=len(entries), metadata=meta,
                                     cost_limit_usd=config.cost_limit_usd if v2 else None)
        model = ScriptedModel(entries, model_name="scripted/fixture")
        runner = MiningRunner(model_factory=lambda c: model, worker_image=task.describe().environment_version,
                              detector_config=DetectorConfig(**config.detector_settings),
                              scratch_root=OUT / "overhead-scratch", network="none")
        t0 = time.monotonic()
        result = runner.run(task, config, store=store)
        wall = time.monotonic() - t0
        evs = events(store, config.run_id)
        per_step = step_overheads(evs)
        caps = [e for e in evs if e.kind == "capture" and e.payload.get("status") == "captured"]
        out[label] = {"termination": result.termination_reason, "steps": sum(e.kind == "model_response" for e in evs),
                      "wall_seconds": round(wall, 2), "step_overhead_seconds": stats(per_step),
                      "first_10": stats(per_step[:10]), "last_10": stats(per_step[-10:]),
                      "checkpoints": len(store.list_checkpoints(config.run_id)), "captures": len(caps),
                      "graded": sum(e.kind == "grade" for e in evs),
                      "tool_state_full_captures": sum(e.kind == "tool_state" and e.payload.get("status") != "manifest"
                                                      for e in evs),
                      "tool_state_manifests": sum(e.kind == "tool_state" and e.payload.get("status") == "manifest"
                                                  for e in evs),
                      "events": len(evs)}
    report["overhead"] = out


def postrun(report: dict) -> None:
    store, rid = store_of("A"), run_id("A")
    annotations = [store.get_record("annotation", r["record_id"]) for r in store.list_records("annotation", run_id=rid)]
    warnings = sorted((a for a in annotations if a.get("kind") == "S"), key=lambda a: a["evidence_cutoff"])
    short = [w["checkpoint_id"].split("/")[-1] for w in warnings]
    # first warning not confirmed (one stall vote of three), every later one confirmed: shows the primary rule
    spec = {"stall": {"*": ["stall", "stall", "unclear"], **({short[0]: ["not_stall", "stall", "unclear"]} if short else {})},
            "type": {"*": "stuck"}}
    write(OUT / "postrun-judge.json", spec)
    out_dir = OUT / "postrun"
    cmd = [sys.executable, str(POSTRUN), "--store", str(OUT / "A" / "run" / "store"), "--batch", "m1", "--out",
           str(out_dir), "--scripted-judge", str(OUT / "postrun-judge.json"), "--judge-all"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    second = subprocess.run(cmd, capture_output=True, text=True)  # idempotent: the record is written once
    record = store.get_record("stall_confirmation", rid)
    confirmations = json.loads((out_dir / "stall_confirmations.json").read_text())
    receipt = json.loads((out_dir / record["primary"]["judge_receipts"][0]).read_text()) if record["primary"] else {}
    type_receipt = json.loads((out_dir / record["primary"]["type_receipt"]).read_text()) if record["primary"] else {}
    keys = {"run_id", "task_id", "primary", "reserved", "later_solved", "detector_rule", "judge"}
    checks = {
        "postrun_ran": first.returncode == 0 and second.returncode == 0,
        "record_has_interface_fields": keys <= set(record) and {"warning_id", "checkpoint_id", "step_index", "judge_votes",
                                                                "judge_receipts"} <= set(record["primary"] or {}),
        "primary_is_first_confirmed_not_first_warning": bool(record["primary"]) and len(warnings) >= 2
            and record["primary"]["checkpoint_id"] == warnings[1]["checkpoint_id"],
        "primary_branchable_at_a_point": bool(record["primary"]) and record["primary"]["branchable"],
        "within_task_cap": (record.get("task_cap") or {}).get("within_cap") is True,
        "scripted_judge_no_spend": receipt.get("scripted") is True and receipt.get("attempts") == [],
        "type_call_saw_reference_solution": type_receipt.get("request_chars", 0) > len(
            (PACKAGE / "private/solution/solve.sh").read_text()),
        "rerun_reads_existing_record": confirmations[0].get("already_recorded") is True,
        "v2_judge_is_haiku_by_default": record["judge"]["model"] == json.loads(
            (ROOT / "configs/v2/judge.json").read_text())["model"] and receipt.get("model") == record["judge"]["model"],
    }
    report["postrun"] = {"checks": checks, "command": " ".join(cmd[1:]), "record": record,
                         "stdout_tail": first.stdout[-1500:], "stderr_tail": first.stderr[-1500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-overhead", action="store_true")
    ap.add_argument("--checks-only", action="store_true", help="keep the natural runs of the last invocation")
    a = ap.parse_args()
    if OUT.exists() and not a.checks_only:
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"what": __doc__.split("\n\n")[0], "task": TASK, "package": str(PACKAGE), "fixture": FIXTURE,
              "command": "python configs/v2/engine_e2e/m1_natural.py", "provider_calls": 0}
    t0 = time.monotonic()
    solve = (PACKAGE / "private/solution/solve.sh").read_text()
    plans = {
        "A": make_plan("A", [
            step("I will plan first.", []),  # no tool call: a format error, still step 1
            step("Look at the task files.", ["ls /app/task_file"]),
            # the idea sentence sits in both the reasoning and the visible text (a Ledger item for milestone 2's
            # drop check, which must remove every copy in the step)
            step("Run the baseline packer. Maybe the greedy packer could be tuned for the larger batches later.",
                 [PARTIAL, "mkdir -p /tmp/v2e2e && echo A > /tmp/v2e2e/marker"],
                 reasoning="Maybe the greedy packer could be tuned for the larger batches later. First I'll run it "
                           "as it is."),
            step("Scratch file made and removed in one step.", ["echo scratch > /app/v2e2e_scratch.txt",
                                                                "rm /app/v2e2e_scratch.txt"]),
            step("A long command.", ["sleep 40"]),
            step("Read the plan. The baseline plan output is correct for the first bucket.",
                 ["head -c 300 /app/task_file/output_data/plan_b1.jsonl"]),
            # hard links in R and W, as a package install makes them (unzip/zipinfo); every later point must
            # still capture and restore
            step("Change the runtime note.", ["echo B > /tmp/v2e2e/marker && ln /tmp/v2e2e/marker /tmp/v2e2e/marker.hl"
                                              " && echo h > /app/v2e2e_h1.txt && ln /app/v2e2e_h1.txt /app/v2e2e_h2.txt"]),
            step("List outputs.", ["ls /app/task_file/output_data"]),
            step("Submit.", [SUBMIT]), step("Submit again.", [SUBMIT]), step("Submit a third time.", [SUBMIT]),
            step("Never reached.", ["echo never"])]),
        "B": make_plan("B", [step("Run the baseline packer.", [PARTIAL]), step("Look.", ["ls /app/task_file"]),
                             step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])], step_limit=3),
        "C": make_plan("C", [step("Run the baseline packer.", [PARTIAL], cost=1.6), step("Submit.", [SUBMIT], cost=1.6),
                             step("Never reached.", ["echo never"], cost=1.6)]),
        "D": make_plan("D", [step("Apply a solution.", [solve]), step("Look.", ["ls /app/task_file/output_data"]),
                             step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])]),
    }
    report["driver_runs"] = {name: drive(path) for name, path in plans.items()}  # a rerun starts nothing (claims)
    check_A(report)
    report["B_last_turn"] = guard_check("B", "max_model_calls", "call_limit", 3)
    report["C_cost_guard"] = guard_check("C", "cost_limit_usd", "cost_limit", 2)
    check_D(report)
    check_driver(report)
    check_wall_cap(report)
    check_restores(report)
    postrun(report)
    if not a.skip_overhead:
        overhead(report)
    report["all_checks"] = {f"{section}.{name}": value for section, body in report.items()
                            if isinstance(body, dict) and isinstance(body.get("checks"), dict)
                            for name, value in body["checks"].items()}
    report["passed"] = all(report["all_checks"].values())
    report["wall_seconds"] = round(time.monotonic() - t0, 1)
    write(ARTIFACTS / "m1-natural.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in report["all_checks"].items() if not v],
                      "overhead": {k: {"wall": v["wall_seconds"], "step": v["step_overhead_seconds"]}
                                   for k, v in report.get("overhead", {}).items()}}, indent=1))


if __name__ == "__main__":
    main()
