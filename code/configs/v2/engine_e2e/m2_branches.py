"""Engine milestone 2 end-to-end check: v2 branches through run_branch, in Docker, with scripted models only.

Source: the confirmed primary stall of milestone 1's natural run A (terminalbench-llm-inference-batching-scheduler-v1,
count-feedback fixture; stall at point 10, after two identical failing submissions). The Ledger files for that stall
are built by track D's code with its scripted (fake) labeller; the branch specs by track S's arms_v2.arm_spec and the
injection payload by healthy_v2._payload, so every input is in the exact interface format. One branch per arm kind:

  1 continue        submission cap 2: the identical-failure counter starts fresh (the source already had two)
  1 continue (r1)   three answers without a tool call: end_reason worker_ended with error_kind worker_outcome
  2 Ledger review   per-channel rewind (W 2, R 3, context 8, outputs 5), item/output/step drops, a kept step; the
                    first call is invalid (keeps a unit the rewind does not undo), the retry is valid; the new-token
                    cap counts both review calls and stops the branch after one continuation step
  3 context review  coupled mode, both calls invalid -> change nothing (flag); the 500-step guard set to 3 counts
                    the two review calls and stops after one continuation step
  4 forced rewind   to the frozen boundary's rewind point, the reference solution, a passing submission
  5 own summary     the note turn (first call has a tool call -> one retry), the arm 4 undo, the note handed back,
                    then three identical failing submissions
  6 Ledger recipe   the arm 4 undo with the recipe handed back; $1.60 per call stops at the $3 guard
  injection         failed_attempt_failure_shown: run A's steps 8-9 (the first failing submission and its feedback)
                    spliced after point 2 of run D (which then submits and passes)
  injection         workspace_debris: the files run A's step 3 created enter run D's W at point 0 (context unchanged)

Then the per-branch overhead of the earlier branch path against branch-v2 on one six-step script.

Usage: python configs/v2/engine_e2e/m2_branches.py   (needs the milestone-1 stores: python configs/v2/engine_e2e/m1_natural.py)
Writes artifacts/v2/engine/m2-branches.json in the main checkout; scratch under ~/.cache/trajectory-retention/v2-engine/m2.
No provider calls.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from common import ARTIFACTS, PACKAGE, REGISTRY_V1, ROOT, SCRATCH, SUBMIT, TASK, events, lifecycle, stats, step, \
    step_overheads, write
from trajectory_retention.contracts import ArtifactRef, RunConfig, TreatmentManifest, from_jsonable
from trajectory_retention.detector.core import DetectorConfig
from trajectory_retention.execution.environment import comparable, manifest_digest
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.experiments import v2_text as T
from trajectory_retention.experiments.branch_v2 import (BranchStore, ReadOnlyArtifactStore, price_card, run_branch,
                                                        uncached_cost)
from trajectory_retention.experiments.context_edit import steps_of, visible_texts
from trajectory_retention.experiments.v2_fixture import CountFeedbackFixture
from trajectory_retention.ledger.__main__ import build_stall
from trajectory_retention.ledger.labeller import ScriptedLabeller
from trajectory_retention.mining.points import checkpoint_step_index, point_checkpoints
from trajectory_retention.mining.runner import MiningRunner
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore
from trajectory_retention.registry import read_registry
from trajectory_retention.store import ArtifactStore

CONFIGS = ROOT / "configs" / "v2"
sys.path.insert(0, str(CONFIGS))
import arms_v2  # noqa: E402  track S: spec builder
import healthy_v2  # noqa: E402  track S: injection payload
import steps_v2  # noqa: E402  track S: worker-visible steps reader

M1 = SCRATCH / "m1"
OUT = SCRATCH / "m2"
INTERFACE_KEYS = {"branch_id", "arm", "repeat", "end_reason", "solved", "submissions", "best_share_passed",
                  "final_share_passed", "choice", "review_tokens", "note_tokens", "handback_tokens", "new_tokens",
                  "steps", "cost_usd", "cost_usd_uncached", "restore_seconds", "wall_seconds", "flags", "store", "run_id"}


def m1_run(name: str) -> tuple[Path, str]:
    root = M1 / name / "run" / "store"
    if not (root / "journal.sqlite3").exists():
        raise SystemExit(f"milestone-1 store {root} missing: run python configs/v2/engine_e2e/m1_natural.py first")
    with ReadOnlyArtifactStore(root)._connect() as db:
        runs = [r[0] for r in db.execute("SELECT DISTINCT run_id FROM events").fetchall() if r[0].endswith(":natural")]
    return root, runs[0]


def review_call(args: dict, *, tokens: int, reasoning: str, content: str | None = None) -> dict:
    return {"content": content, "reasoning": reasoning,
            "raw_tool_calls": [{"id": f"rv-{tokens}", "type": "function",
                                "function": {"name": T.REVIEW_TOOL_NAME, "arguments": json.dumps(args)}}],
            "usage": {"prompt_tokens": 9000, "completion_tokens": tokens, "total_tokens": 9000 + tokens, "cost": 0.0}}


def branch(name: str, spec: dict, script: list[dict]) -> dict:
    out = OUT / "branches" / name / "a0"
    model = ScriptedModel(script, model_name="scripted/fixture")
    res = run_branch(spec, out_dir=out, model_factory=lambda config: model)
    return {"spec": spec, "result": res, "out": out, "script_used": model.index, "script_length": len(script)}


def evidence(b: dict, source_store: ArtifactStore) -> dict:
    """What the branch did, read back from its own store (and the source store for donor points)."""
    res, spec = b["result"], b["spec"]
    store = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(spec["source"]["store"]))
    evs = events(store, res["run_id"])
    started = (lifecycle(evs, "branch_started") or [None])[0]
    started = started.payload if started else {}
    compiled = (lifecycle(evs, "branch_v2_compiled") or [None])[0]
    compiled = compiled.payload if compiled else {}
    validated = (lifecycle(evs, "next_request_validated") or [None])[0]
    validated = validated.payload if validated else {}
    restored = (lifecycle(evs, "history_restored") or [None])[0]
    restored = restored.payload if restored else {}
    donor_points = {}
    for channel in ("workspace_donor", "runtime_donor", "history_donor"):
        cid = started.get(channel)
        if cid:
            cp = store.get_checkpoint(cid)
            donor_points[channel] = {"checkpoint_id": cid, "point": checkpoint_step_index(store, cp),
                                     "fidelity_digest": cp.validation.get("workspace_fidelity_digest")}
    request = json.loads(store.get_bytes(from_jsonable(ArtifactRef, validated["request_ref"]))) if validated else None
    first_request = next((json.loads(store.get_bytes(e.input_ref)) for e in evs
                          if e.kind == "model_request" and not e.payload.get("mode")), None)
    handback = next((store.get_bytes(e.delivered_ref).decode("utf-8") for e in evs if e.kind == "handback_delivery"),
                    None)
    grades = [e.payload["checkpoint_id"] for e in evs if e.kind == "grade"]
    grade_reasons = {cid: list(store.get_checkpoint(cid).reasons) for cid in grades}
    outputs = []
    first_response = next((e for e in evs if e.kind == "model_response" and not e.payload.get("mode")), None)
    if first_response is not None:
        calls = {e.event_id for e in evs if e.kind == "tool_call" and e.payload.get("model_call_id") == first_response.call_id}
        outputs = [store.get_bytes(e.output_ref).decode("utf-8", "replace")[-600:] for e in evs
                   if e.kind == "tool_result" and e.parent_event_id in calls]
    modes = sorted({e.payload["mode"] for e in evs if e.kind in ("model_request", "model_response")
                    and e.payload.get("mode")})
    turn_requests = []
    for e in evs:
        if e.kind == "model_request" and e.payload.get("mode"):
            body = json.loads(store.get_bytes(e.input_ref))
            turn_requests.append({"mode": e.payload["mode"], "has_tools_key": "tools" in body,
                                  "tools": [t["function"]["name"] for t in body.get("tools") or []],
                                  "messages": len(body["messages"]), "last_role": body["messages"][-1]["role"],
                                  "max_completion_tokens": body.get("max_completion_tokens")})
    reader = steps_v2.run_steps(res["store"], res["run_id"], handback_header=T.HANDBACK_HEADER)
    choice_record = None
    try:
        choice_record = store.get_record("choice", res["run_id"])
    except KeyError:
        pass
    return {"donor_points": donor_points, "workspace_restore": started.get("workspace_restore"),
            "runtime_restore": started.get("runtime_restore"), "history_policy": started.get("history_policy"),
            "history_restored": {k: restored.get(k) for k in ("history_policy", "message_count", "validation",
                                                              "context_edits")},
            "compiled": {k: compiled.get(k) for k in ("points", "effective_points", "context_edits", "skipped_drops",
                                                       "flags", "undone", "removed", "grader_record", "injection",
                                                       "workspace_composition")},
            "request_validation": {k: validated.get(k) for k in ("checks", "valid", "quotes", "request_messages")},
            "request": request, "first_request_equals_validated": first_request == request if request else None,
            "handback": handback, "grades": grade_reasons, "first_step_outputs": outputs, "turn_modes": modes,
            "turn_requests": turn_requests,
            "s_reader": {"steps": len(reader["steps"]), "excluded": [x["what"] for x in reader["excluded"]]},
            "choice_record": choice_record, "events": len(evs),
            "ready": (lifecycle(evs, "branch_v2_ready") or [None])[0].payload if lifecycle(evs, "branch_v2_ready") else None}


def strip(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "extra"} for m in messages]


def main() -> None:
    t_all = time.monotonic()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    a_root, a_run = m1_run("A")
    d_root, d_run = m1_run("D")
    a_store = ReadOnlyArtifactStore(a_root)
    confirmation = a_store.get_record("stall_confirmation", a_run)
    stall_cp = a_store.get_checkpoint(confirmation["primary"]["checkpoint_id"])
    s = checkpoint_step_index(a_store, stall_cp)
    source_history = a_store.get_json(stall_cp.state.history)

    # ---- the stall's Ledger files (track D, fake labels) and the specs (track S)
    stall_dir = OUT / "stall" / "m1-A-C12"
    t0 = time.monotonic()
    build_stall(ReadOnlyStore(a_root), a_run, stall_cp.checkpoint_id, stall_dir,
                ScriptedLabeller(ArtifactStore(stall_dir / "labeller")), recipe_cap=120, max_chars=300_000)
    ledger_seconds = time.monotonic() - t0
    boundary = json.loads((stall_dir / "boundary.json").read_text())
    rp = boundary["rewind_point"]
    view = json.loads((stall_dir / "review_view_ledger.json").read_text())
    units = view["units"]
    stall = {"stall_id": "m1-A-C12", "store": str(a_root), "run_id": a_run, "checkpoint_id": stall_cp.checkpoint_id,
             "dir": str(stall_dir), "has_boundary": True, "step_index": s, "has_recipe": True}
    spec_of = lambda arm, caps=None, **kw: arms_v2.arm_spec(stall, arm, 0, purpose="tuning", prefix="m2-engine",
                                                            caps=caps, **kw)
    item = next(u for u, v in sorted(units.items()) if v["kind"] == "item" and 2 <= v["steps"][0] <= 8
                and v["steps"][0] != 4)
    item_step = units[item]["steps"][0]
    solve = (PACKAGE / "private/solution/solve.sh").read_text()

    # ---- the branches
    per_channel = {"rewind": {"workspace": 2, "context": 8, "runtime": 3, "outputs": 5},
                   "drop": [item, "O2", "S4"], "keep": ["S9"], "change_nothing": False}
    invalid_keep = dict(per_channel, keep=["S3"])
    probe = "cat /tmp/v2e2e/marker 2>&1; ls /app/task_file/output_data 2>&1 | head -5"
    runs = {}
    runs["1_continue"] = branch("1_continue", spec_of("1_continue", {"new_token_cap": None, "submission_cap": 2}), [
        step("Check the state.", [probe]), step("Submit.", [SUBMIT]), step("Submit again.", [SUBMIT]),
        step("Never reached.", ["echo never"])])
    # a worker-made end outside the interface's reasons: three answers in a row without a tool call
    runs["1_continue_format_errors"] = branch("1_continue_format_errors", arms_v2.arm_spec(
        stall, "1_continue", 1, purpose="tuning", prefix="m2-engine", caps=None), [
        {"content": f"Thinking {i}.", "commands": [],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "cost": 0.0}}
        for i in range(3)] + [step("Never reached.", ["echo never"])])
    runs["2_ledger_review"] = branch("2_ledger_review", spec_of(
        "2_ledger_review", {"new_token_cap": 6000, "submission_cap": None}, opus_record=True), [
        review_call(invalid_keep, tokens=1500, reasoning="I will rewind and keep step 3 of the plan."),
        review_call(per_channel, tokens=2000, reasoning="Rewinding the workspace to point 2, keeping step 9."),
        step("Check the restored state.", [probe], tokens=2500), step("Never reached.", ["echo never"])])
    runs["3_context_review"] = branch("3_context_review", dict(spec_of(
        "3_context_review", None, rewind_mode="coupled"), budget={"new_token_cap": None, "submission_cap": None,
                                                                   "cost_limit_usd": 3.0, "step_limit": 3}), [
        {"content": "I would rather keep going without a call.", "commands": [],
         "usage": {"prompt_tokens": 9000, "completion_tokens": 700, "total_tokens": 9700, "cost": 0.0}},
        review_call({"rewind_to": 99, "drop": [], "keep": [], "change_nothing": False}, tokens=800,
                    reasoning="Rewind far back."),
        step("Continue.", [probe], tokens=1000), step("Never reached.", ["echo never"])])
    runs["4_forced_rewind"] = branch("4_forced_rewind", spec_of("4_forced_rewind"), [
        step("Check the restored state.", [probe]), step("Apply a solution.", [solve]), step("Submit.", [SUBMIT]),
        step("Never reached.", ["echo never"])])
    note = ("Submissions: two submissions, 1 hidden test failing each time.\nWhat was tried: ran the baseline packer "
            "and resubmitted.\nPlan: write a real packer; first commands: ls /app/task_file; cat the task spec.")
    runs["5_rewind_own_summary"] = branch("5_rewind_own_summary", spec_of("5_rewind_own_summary"), [
        {"content": "Let me check first.", "commands": ["ls /app"],
         "usage": {"prompt_tokens": 9000, "completion_tokens": 400, "total_tokens": 9400, "cost": 0.0}},
        {"content": note, "commands": [], "reasoning": "Summarise the turns for the note.",
         "usage": {"prompt_tokens": 9000, "completion_tokens": 1200, "total_tokens": 10200, "cost": 0.0}},
        step("Check the restored state.", [probe]), step("Submit.", [SUBMIT]), step("Submit.", [SUBMIT]),
        step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])])
    runs["6_rewind_ledger_recipe"] = branch("6_rewind_ledger_recipe", spec_of("6_rewind_ledger_recipe"), [
        step("Check the restored state.", [probe], cost=1.6), step("Look.", ["ls /app/task_file"], cost=1.6),
        step("Never reached.", ["echo never"], cost=1.6)])
    d_store = ReadOnlyArtifactStore(d_root)
    d_points = point_checkpoints(d_store, d_run)
    receiver = {"healthy_id": "m1-D-p2", "store": str(d_root), "run_id": d_run, "task_id": TASK,
                "checkpoint_id": d_points[2], "step_index": 2}
    scope = steps_v2.run_scope(str(d_root), d_run)
    failed = {"stall": {"store": str(a_root), "run_id": a_run, "task_id": TASK}, "dir": str(stall_dir),
              "scope": steps_v2.run_scope(str(a_root), a_run)}
    payload = healthy_v2._payload("failed_attempt_failure_shown", receiver, scope, channel="context",
                                  donor=healthy_v2._donor(failed, 8, 9, "engine e2e: run A steps 8-9", {}))
    payload["donor"]["task_version"] = {k: failed["scope"].get(k) for k in ("requirement_version", "evaluator_version")}
    inj_spec = arms_v2.spec(purpose="injection", prefix="m2-engine", unit_id="m1-D-p2",
                            arm="injection:failed_attempt_failure_shown", repeat=0,
                            source={"store": str(d_root), "run_id": d_run, "checkpoint_id": d_points[2]}, caps=None)
    inj_spec["injection"] = payload
    runs["injection"] = branch("injection", inj_spec, [step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])])
    # the workspace channel: files run A's step 3 created (the baseline packer's outputs) enter run D's W at point 0
    receiver0 = dict(receiver, healthy_id="m1-D-p0", checkpoint_id=d_points[0], step_index=0)
    debris = healthy_v2._payload("workspace_debris", receiver0, scope, channel="workspace",
                                 donor=healthy_v2._donor(failed, 3, 3, "engine e2e: run A step 3", {}),
                                 debris={"donor_point_before": 2, "donor_point_after": 3, "paths": None,
                                         "path_rule": "files created or modified in the donor between the two points, "
                                                      "excluding paths that exist in the receiver's workspace"})
    debris["donor"]["task_version"] = payload["donor"]["task_version"]
    deb_spec = arms_v2.spec(purpose="injection", prefix="m2-engine", unit_id="m1-D-p0", arm="injection:workspace_debris",
                            repeat=0, source={"store": str(d_root), "run_id": d_run, "checkpoint_id": d_points[0]},
                            caps={"new_token_cap": None, "submission_cap": 1})
    deb_spec["injection"] = debris
    runs["injection_debris"] = branch("injection_debris", deb_spec, [
        step("Look.", ["ls /app/task_file/output_data"]), step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])])

    # ---- evidence and checks per branch
    report = {"command": "python configs/v2/engine_e2e/m1_natural.py && python configs/v2/engine_e2e/m2_branches.py",
              "source": {"store": str(a_root), "run_id": a_run, "checkpoint_id": stall_cp.checkpoint_id, "point": s,
                         "stall_confirmation_primary": confirmation["primary"]["warning_id"]},
              "ledger_files": {"dir": str(stall_dir), "boundary": boundary, "seconds": round(ledger_seconds, 2),
                               "units": len(units), "fake_labels": True},
              "branches": {}, "checks": {}}
    source_messages = strip(source_history["messages"])
    a_points = point_checkpoints(a_store, a_run)
    for name, b in runs.items():
        ev = evidence(b, a_store)
        res = b["result"]
        report["branches"][name] = {"result": res, "evidence": {k: v for k, v in ev.items() if k != "request"},
                                    "request_messages": len(ev["request"]["messages"]) if ev["request"] else None,
                                    "script_used": b["script_used"]}
        b["evidence"] = ev
    c = report["checks"]
    R = {n: b["result"] for n, b in runs.items()}
    E = {n: b["evidence"] for n, b in runs.items()}
    c["every_result_has_interface_keys"] = all(INTERFACE_KEYS <= set(r) for r in R.values())
    c["no_technical_errors"] = all(r["details"].get("error_kind") != "technical" for r in R.values())
    c["end_reasons"] = {n: r["end_reason"] for n, r in R.items()}
    c["end_reasons_as_scripted"] = c["end_reasons"] == {
        "1_continue": "submission_cap", "2_ledger_review": "token_cap", "3_context_review": "step_guard",
        "4_forced_rewind": "passing_submission", "5_rewind_own_summary": "identical_failures",
        "6_rewind_ledger_recipe": "cost_guard", "injection": "passing_submission",
        "injection_debris": "submission_cap", "1_continue_format_errors": "worker_ended"}
    rf = R["1_continue_format_errors"]
    c["format_errors_are_a_worker_outcome"] = (rf["details"]["error_kind"] == "worker_outcome"
                                               and rf["details"]["end_detail"] == "worker_error:RepeatedFormatError"
                                               and rf["steps"] == 3 and rf["final_share_passed"] is not None)
    c["requests_validated_and_sent_as_validated"] = all(
        e["request_validation"]["valid"] and e["first_request_equals_validated"] for e in E.values())
    c["system_and_task_kept"] = all(e["request"]["messages"][:2] == source_messages[:2] for n, e in E.items()
                                    if n != "injection")
    c["grades_only_at_submissions_and_end"] = all(
        all(set(reasons) & {"native_submission", "terminal"} and not set(reasons) & {"branch_start", "point"}
            for reasons in e["grades"].values()) for e in E.values())
    c["grade_count_is_submissions_plus_end"] = {n: (len(E[n]["grades"]), len(R[n]["submissions"])) for n in R}
    c["ended_at_submission_has_no_second_grade"] = all(
        len(E[n]["grades"]) == len(R[n]["submissions"]) for n in R
        if R[n]["end_reason"] in ("submission_cap", "passing_submission", "identical_failures"))
    c["ended_at_guard_has_one_end_grade"] = all(
        len(E[n]["grades"]) == len(R[n]["submissions"]) + 1 for n in R
        if R[n]["end_reason"] in ("token_cap", "step_guard", "cost_guard"))
    c["turn_events_marked_and_excluded_by_S_reader"] = all(
        (("review" in E[n]["turn_modes"]) == (n in ("2_ledger_review", "3_context_review")))
        and (("note" in E[n]["turn_modes"]) == (n == "5_rewind_own_summary"))
        and E[n]["s_reader"]["steps"] == R[n]["steps"] for n in R)
    # arm 1
    e1 = E["1_continue"]
    c["arm1_continue_unchanged_context"] = (e1["request"]["messages"] == source_messages and e1["handback"] is None
                                           and e1["history_policy"] == "native"
                                           and {v["point"] for v in e1["donor_points"].values()} == {s})
    source_criteria = [e.payload.get("criterion_id") for e in a_store.iter_events(
        a_run, through_sequence=stall_cp.event_sequence) if e.kind == "lifecycle"
        and e.payload.get("transition") == "native_submission_verdict"]
    subs1 = R["1_continue"]["submissions"]
    # the source already had two identical failing submissions: a third would end it at identical_failures unless
    # the counter starts fresh at the branch, so two more identical ones end it at the cap of 2 instead
    c["arm1_identical_counter_reset"] = (len(source_criteria) == 2 and len(set(source_criteria)) == 1
                                         and len(subs1) == 2 and all(x["failing_hash"] == source_criteria[0]
                                                                     for x in subs1))
    # arm 2: per channel, edits, hand-back, budget, invalid then valid
    r2, e2 = R["2_ledger_review"], E["2_ledger_review"]
    pts = {k: v["point"] for k, v in e2["donor_points"].items()}
    c["arm2_per_channel_donors"] = pts == {"workspace_donor": 2, "runtime_donor": 3, "history_donor": 8}
    w2 = e2["workspace_restore"]
    # the restore raises unless the live W equals the donor checkpoint's fidelity manifest (directory mtimes aside)
    c["arm2_workspace_restored_from_point_2"] = w2["validated"] is True and pts.get("workspace_donor") == 2
    first_out = "\n".join(e2["first_step_outputs"])
    # R from point 3 (marker A, written at step 3, B only at step 7); W from point 2 (no output_data before step 3)
    c["arm2_live_state_matches_channels"] = (first_out.splitlines()[0] == "A"
                                             and "output_data': No such file or directory" in first_out)
    req2 = e2["request"]["messages"]
    raw8 = a_store.get_json(a_store.get_checkpoint(a_points[8]).state.history)["messages"]
    ctx8 = strip(raw8)
    steps8 = steps_of(raw8)  # FormatError user messages start a step: read with their extra
    quote = units[item]["quote"]
    src_step = [ctx8[i] for i in steps8[item_step] if ctx8[i].get("role") == "assistant"][0]
    c["arm2_item_quote_was_in_content_and_reasoning"] = quote in src_step["content"] and quote in src_step["reasoning"]
    c["arm2_item_quote_removed_everywhere_in_its_step"] = not any(quote in t for m in req2 for t in visible_texts(m))
    c["arm2_engine_quote_check_agrees"] = [(q["step"], q["left_in_step"]) for q in e2["request_validation"]["quotes"]] \
        == [(item_step, 0)]
    tool_msgs = {m["tool_call_id"]: m["content"] for m in req2 if m.get("role") == "tool"}
    step_calls = lambda k: [c_["id"] for i in steps8[k] for c_ in ctx8[i].get("tool_calls") or []]
    c["arm2_outputs_after_point_5_and_O2_replaced_commands_kept"] = all(
        tool_msgs.get(cid) == T.OUTPUT_PLACEHOLDER for k in (2, 6, 7, 8) for cid in step_calls(k)) and all(
        any(c_.get("id") == cid for m in req2 for c_ in m.get("tool_calls") or []) for k in (6, 7, 8)
        for cid in step_calls(k))
    c["arm2_step_4_dropped"] = not any(cid in tool_msgs for cid in step_calls(4))
    c["arm2_steps_after_context_point_absent"] = len([m for m in req2 if m.get("role") == "assistant"]) == 8 - 1 - 1
    hb2 = e2["handback"] or ""
    grader = [T.GRADER_RECORD_LINE.format(i=g["i"], k=g["k"], n=g["n"]) for g in e2["compiled"]["grader_record"]]
    c["arm2_handback_header_generic_grader_then_kept_step"] = (
        hb2.startswith(T.HANDBACK_HEADER + "\n" + T.GENERIC_LINE + "\n" + "\n".join(grader)) and len(grader) == 2
        and req2[-1]["content"] == hb2 and "Submit." in hb2.split("\n".join(grader), 1)[1])
    c["arm2_budget_counts_review_turn"] = (r2["review_tokens"] == 3500 and r2["new_tokens"] == 6000
                                          and r2["steps"] == 1 and e2["ready"]["intervention_calls"] == 2)
    c["arm2_invalid_then_valid_choice"] = (r2["choice"]["calls"] == 2 and r2["choice"]["valid"]
                                          and len(r2["choice"]["invalid_reasons"]) == 1
                                          and "keep may name only units from undone steps"
                                          in r2["choice"]["invalid_reasons"][0]
                                          and e2["choice_record"] == r2["choice"])
    c["arm2_mismatch_flag"] = r2["flags"]["mismatch"] is True
    opus = OUT / "branches" / "2_ledger_review" / "a0" / "opus-review-input.json"
    c["arm2_opus_input_is_the_review_request"] = opus.exists() and (lambda o: o["request"]["tools"][0]["function"][
        "name"] == T.REVIEW_TOOL_NAME and o["request"]["messages"][:-1] == source_messages
        and T.REVIEW_PROMPT in o["request"]["messages"][-1]["content"])(json.loads(opus.read_text()))
    c["arm2_review_turn_absent_from_next_request"] = (e2["request_validation"]["checks"]["intervention_turn_absent"]
                                                     and not any(T.REVIEW_PROMPT in t for m in req2
                                                                 for t in visible_texts(m)))
    # arm 3: invalid twice -> change nothing; step guard counts review calls
    r3, e3 = R["3_context_review"], E["3_context_review"]
    c["arm3_invalid_retry_change_nothing"] = (r3["flags"]["invalid_call"] and r3["flags"]["no_change"]
                                             and r3["choice"]["fallback_change_nothing"] and r3["choice"]["calls"] == 2
                                             and e3["request"]["messages"] == source_messages and e3["handback"] is None)
    c["arm3_step_guard_counts_review_calls"] = r3["steps"] == 1 and e3["ready"]["intervention_calls"] == 2
    # arm 4: forced rewind, pass
    r4, e4 = R["4_forced_rewind"], E["4_forced_rewind"]
    c["arm4_all_channels_at_rewind_point"] = ({v["point"] for v in e4["donor_points"].values()} == {rp}
                                             and len(e4["request"]["messages"]) == len(
        strip(a_store.get_json(a_store.get_checkpoint(a_points[rp]).state.history)["messages"])) + 1)
    c["arm4_handback_generic_and_grader_only"] = e4["handback"] == "\n".join(
        [T.HANDBACK_HEADER, T.GENERIC_LINE] + grader)
    c["arm4_solved"] = r4["solved"] and r4["final_share_passed"] == 1.0 and r4["best_share_passed"] == 1.0
    # arm 5: note
    r5, e5 = R["5_rewind_own_summary"], E["5_rewind_own_summary"]
    c["arm5_note_retry_then_handed_back"] = (e5["handback"] == "\n".join([T.HANDBACK_HEADER, T.GENERIC_LINE] + grader)
                                            + "\n\n" + note and not r5["flags"]["invalid_call"]
                                            and r5["note_tokens"] > 0)
    c["arm5_note_turn_absent_except_handback"] = (e5["request_validation"]["checks"]["intervention_turn_absent"]
                                                 and sum(note in t for m in e5["request"]["messages"]
                                                         for t in visible_texts(m)) == 1)
    c["arm5_note_requests_have_no_tools_key"] = [(t["mode"], t["has_tools_key"]) for t in e5["turn_requests"]] == [
        ("note", False), ("note", False)]
    c["arm2_review_requests_offer_only_review_decision"] = [(t["mode"], t["tools"]) for t in e2["turn_requests"]] == [
        ("review", [T.REVIEW_TOOL_NAME])] * 2
    # arm 6: recipe, cost guard
    r6, e6 = R["6_rewind_ledger_recipe"], E["6_rewind_ledger_recipe"]
    recipe = json.loads((stall_dir / "recipe.json").read_text())["text"]
    c["arm6_recipe_handed_back"] = e6["handback"] == "\n".join([T.HANDBACK_HEADER, T.GENERIC_LINE] + grader) + \
        "\n\n" + recipe
    c["arm6_cost_guard"] = r6["steps"] == 2 and r6["cost_usd"] >= 3.0
    # injection
    ri, ei = R["injection"], E["injection"]
    d_ctx = strip(d_store.get_json(d_store.get_checkpoint(d_points[2]).state.history)["messages"])
    a_ctx = a_store.get_json(a_store.get_checkpoint(a_points[9]).state.history)["messages"]
    a_steps = steps_of(a_ctx)
    donor_msgs = [a_ctx[i] for k in (8, 9) for i in a_steps[k]]
    spliced = ei["request"]["messages"][len(d_ctx):]
    c["injection_receiver_context_kept_then_spliced"] = (ei["request"]["messages"][:len(d_ctx)] == d_ctx
                                                         and len(spliced) == len(donor_msgs)
                                                         and [m.get("content") for m in spliced]
                                                         == [m.get("content") for m in donor_msgs])
    c["injection_tool_call_ids_renamed"] = all(
        (c_["id"].startswith(payload["splice"]["id_namespace"] + "-")) for m in spliced
        for c_ in m.get("tool_calls") or []) and all(m["tool_call_id"].startswith(payload["splice"]["id_namespace"])
                                                     for m in spliced if m.get("role") == "tool")
    c["injection_failure_feedback_shown"] = [m["content"] for m in spliced if m.get("role") == "user"] == [
        T.SUBMISSION_FEEDBACK.format(n=1)]
    c["injection_no_handback_and_solved"] = ei["handback"] is None and ri["solved"]
    rd, ed = R["injection_debris"], E["injection_debris"]
    comp = ed["compiled"]["workspace_composition"] or {}
    d0 = strip(d_store.get_json(d_store.get_checkpoint(d_points[0]).state.history)["messages"])
    listed = set(ed["first_step_outputs"][0].split()) if ed["first_step_outputs"] else set()
    # step 3 ran the baseline packer: its two plan files (and Python's bytecode cache) are the debris
    plans = {"task_file/output_data/plan_b1.jsonl", "task_file/output_data/plan_b2.jsonl"}
    c["debris_paths_from_donor_step"] = plans <= set(comp.get("paths") or []) and comp.get(
        "base_checkpoint_id") == d_points[0] and not comp.get("candidates_in_receiver")
    c["debris_in_live_workspace"] = {x.rsplit("/", 1)[-1] for x in plans} <= listed
    c["debris_context_unchanged"] = ed["request"]["messages"] == d0 and ed["handback"] is None
    c["debris_workspace_restore_validated"] = ed["workspace_restore"]["validated"] is True
    # restore time and the price card
    card = price_card()["models"]["deepseek/deepseek-v4.1-flash"]
    b2 = runs["2_ledger_review"]
    from trajectory_retention.accounting import CostLedger
    st2 = BranchStore(Path(b2["result"]["store"]), ReadOnlyArtifactStore(str(a_root)))
    priced, unknown = uncached_cost(CostLedger(st2).records(b2["result"]["run_id"]), card)
    report["cost_usd_uncached_at_card_prices_arm2"] = {"usd": round(priced, 6), "unknown_attempts": unknown,
        "expected": round((9000 * 2 + 1000) * card["input"] / 1e6 + (1500 + 2000 + 2500) * card["output"] / 1e6, 6),
        "note": "result.json keeps cost_usd_uncached null for the scripted model (no price card entry); this applies "
                "the DeepSeek card to the same recorded usage"}
    c["uncached_cost_from_card"] = abs(priced - report["cost_usd_uncached_at_card_prices_arm2"]["expected"]) < 1e-9
    report["restore_seconds"] = {n: r["restore_seconds"] for n, r in R.items()}
    report["branch_wall_seconds"] = {n: r["wall_seconds"] for n, r in R.items()}
    overhead(report, a_root, a_run, a_points)
    c["all_true"] = all(v for k, v in c.items() if isinstance(v, bool))
    report["passed"] = c["all_true"]
    report["seconds"] = round(time.monotonic() - t_all, 1)
    write(ARTIFACTS / "m2-branches.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in c.items() if v is False],
                      "end_reasons": c["end_reasons"], "seconds": report["seconds"],
                      "overhead": report.get("overhead", {}).get("summary")}, indent=1))


def overhead(report: dict, a_root: Path, a_run: str, a_points: dict) -> None:
    """Per-branch overhead: the earlier branch path against branch-v2 on one six-step script from point 8."""
    script = []
    for i in range(6):
        cmd = (f"echo line{i} >> /app/v2e2e_m2.txt" if i % 2 == 0 else f"echo r{i} > /tmp/v2e2e_m2 && cat /tmp/v2e2e_m2")
        script.append(step(f"work {i}", [cmd], tokens=300))
    copy = OUT / "overhead-source-store"  # the earlier path writes its branch into the store it reads: use a copy
    shutil.copytree(a_root, copy)
    store = ArtifactStore(copy)
    source = store.get_checkpoint(a_points[8])
    out = {}
    # branch-v2: the arm 1 spec shape with the 6-call guard, from point 8
    spec = arms_v2.spec(purpose="tuning", prefix="m2-overhead", unit_id="m1-A-p8", arm="1_continue", repeat=0,
                        source={"store": str(a_root), "run_id": a_run, "checkpoint_id": source.checkpoint_id},
                        caps=None)
    spec["budget"]["step_limit"] = 6
    b = branch("overhead-v2", spec, script)
    st = BranchStore(Path(b["result"]["store"]), ReadOnlyArtifactStore(str(a_root)))
    evs = events(st, b["result"]["run_id"])
    out["branch_v2"] = {"wall_seconds": b["result"]["wall_seconds"], "restore_seconds": b["result"]["restore_seconds"],
                        "end_reason": b["result"]["end_reason"], "steps": b["result"]["steps"],
                        "grades": sum(e.kind == "grade" for e in evs), "checkpoints": b["result"]["details"]["checkpoints"],
                        "step_overhead": stats(step_overheads(evs)),
                        "setup_seconds": b["result"]["details"]["setup_seconds"]}
    # the earlier path: a native treatment through the same runner without a v2 policy (writes into the source store,
    # as the milestone-1 restore checks did)
    task = CountFeedbackFixture(read_registry(REGISTRY_V1)[TASK].load(scratch_root=OUT / "overhead-task",
                                                                      purpose="development"))
    start = next(e.payload for e in store.iter_events(a_run) if e.kind == "lifecycle"
                 and e.payload.get("transition") == "run_started")
    cfg = start["config"]
    stage = task.describe().stages[0]
    brid = f"m2-earlier-path-{time.strftime('%Y%m%dT%H%M%S')}"
    treatment = TreatmentManifest(treatment_id=brid, source_checkpoint_id=source.checkpoint_id,
        target_id=stage.target_id, stage_id=stage.stage_id, selection_kind="engineering_checkpoint",
        component_donors={k: source.checkpoint_id for k in ("workspace", "runtime", "history")},
        history_policy="native", runtime_policy="restore_captured", worker_model="scripted/fixture",
        generated_token_limit=10**9, repeat_id="m2")
    config = RunConfig(run_id=brid, campaign_id="m2-overhead", purpose="development", model="scripted/fixture",
        model_settings={}, generated_token_limit=10**9, max_model_calls=6,
        tool_timeout_seconds=cfg["tool_timeout_seconds"], detector_settings=cfg["detector_settings"],
        metadata={"fixture": True, "engine_check": "earlier branch path overhead"})
    model = ScriptedModel(script, model_name="scripted/fixture")
    runner = MiningRunner(model_factory=lambda cfg_: model, worker_image=task.describe().environment_version,
                          detector_config=DetectorConfig(**config.detector_settings),
                          scratch_root=OUT / "overhead-scratch", network="none")
    t0 = time.monotonic()
    result = runner.run(task, config, store=store, treatment=treatment)
    wall = time.monotonic() - t0
    evs = events(store, brid)
    ops = {}
    for e in evs:
        if e.kind == "operation" and e.usage:
            ops[e.payload["name"]] = ops.get(e.payload["name"], 0.0) + (e.usage.elapsed_seconds or 0.0)
    out["earlier_path"] = {"wall_seconds": round(wall, 3), "status": result.status,
                           "restore_seconds": round(sum(ops.get(n, 0.0) for n in (
                               "environment_create", "workspace_restore_input", "workspace_restore",
                               "runtime_restore_input", "runtime_restore", "history_restore_input",
                               "restored_session_prepare")), 3),
                           "end": result.termination_reason, "grades": sum(e.kind == "grade" for e in evs),
                           "checkpoints": len(result.checkpoint_ids), "step_overhead": stats(step_overheads(evs))}
    v2, old = out["branch_v2"], out["earlier_path"]
    out["summary"] = {"wall_seconds": {"earlier_path": old["wall_seconds"], "branch_v2": v2["wall_seconds"]},
                      "restore_seconds": {"earlier_path": old["restore_seconds"], "branch_v2": v2["restore_seconds"]},
                      "grades": {"earlier_path": old["grades"], "branch_v2": v2["grades"]},
                      "mean_step_overhead": {"earlier_path": old["step_overhead"].get("mean"),
                                             "branch_v2": v2["step_overhead"].get("mean")}}
    report["overhead"] = out
    report["checks"]["overhead_branch_v2_grades_only_at_end"] = v2["grades"] == 1 and v2["end_reason"] == "step_guard"


if __name__ == "__main__":
    main()
