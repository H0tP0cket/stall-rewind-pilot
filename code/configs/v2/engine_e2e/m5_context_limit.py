"""Engine end-to-end check of the context-limit rule (user, 2026-10-16), in Docker, with scripted models only.

The rule (src/trajectory_retention/execution/context_limit.py): when the next request won't fit, hide the oldest tool
outputs, the same way in every arm, and log it; review views are built from the full record. Here the worker's window
is an artificial WINDOW tokens with a RESPONSE_CAP response cap, so a short run on the real Terminal-Bench task
(terminalbench-llm-inference-batching-scheduler-v1 with the count-feedback fixture, as in m1/m2) needs it. The scripted
model plays a provider whose prompt count is RATIO times the local measure of the actual request, so every request is
also checked in the "provider's" own count.

  N  natural run through the driver: outputs pile up until outputs are hidden, a failing submission, more hiding,
     the reference solution, a passing submission
  X  natural run whose own assistant text alone overflows the window: ends with context_window_exhausted
  B1 arm 1 from a point after hiding: the restored context is the masked view; hiding continues; passes
  B2 arm 1 from the point just before the first hiding: its first request equals the natural run's request at that
     step byte for byte (same rule, same hidden outputs)
  B3 arm 2 at an early point: the review request (masked context + the Ledger view, built from the full record) fits
     only once outputs are hidden; the hiding stays in the turn, and the continuation gets the source context
  B4 arm 2 at a point after hiding: the review request cannot fit (the view holds every output): the branch ends with
     context_window_exhausted before any continuation call
  B5 arm 1 from a point after hiding, whose own long answer overflows: context_window_exhausted

Usage: python configs/v2/engine_e2e/m5_context_limit.py
Writes artifacts/v2/engine/m5-context-limit.json in the main checkout; stores stay under
~/.cache/trajectory-retention/v2-engine/m5. No provider calls.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

from common import ARTIFACTS, PACKAGE, PARTIAL, REGISTRY_V1, ROOT, SCRATCH, SUBMIT, TASK, events, lifecycle, step, write
from m1_natural import common_checks
from trajectory_retention.contracts import ArtifactRef, from_jsonable
from trajectory_retention.execution.context_limit import MASK_KEY, RULE_VERSION, ContextLimit
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.experiments import v2_text as T
from trajectory_retention.experiments.branch_v2 import BranchStore, ReadOnlyArtifactStore, run_branch
from trajectory_retention.ledger.__main__ import build_stall
from trajectory_retention.ledger.checks import check_quotes
from trajectory_retention.ledger.labeller import ScriptedLabeller
from trajectory_retention.ledger.steps import full_messages, load_context
from trajectory_retention.mining.points import point_checkpoints
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore
from trajectory_retention.store import ArtifactStore

OUT = SCRATCH / "m5"
DRIVER = ROOT / "configs/v2/natural.py"
_spec = importlib.util.spec_from_file_location("natural", DRIVER)
natural = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural)
sys.path.insert(0, str(ROOT / "configs" / "v2"))
import arms_v2  # noqa: E402  track S: spec builder

FIXTURE = "count-feedback-fixture/v1"
WINDOW = 13500        # artificial window, tokens (run N first hides outputs at step 10)
EARLY = 5             # B3's review point: before any hiding, where the review request needs it
RESPONSE_CAP = 2000   # the plan's max_tokens: every request reserves it
RATIO = 1.17          # the scripted "provider" counts this many prompt tokens per local token
MEASURE = ContextLimit(window_tokens=WINDOW, placeholder=T.OUTPUT_PLACEHOLDER)
LIMIT = {"rule": RULE_VERSION, "window_tokens": WINDOW, "source": "m5 e2e (artificial window)"}


class ProviderCountingModel(ScriptedModel):
    """Scripted responses; usage reports prompt tokens = ceil(RATIO x local measure of the request it received)."""

    def query(self, messages, *, prepared_request=None, **kwargs):
        prepared = prepared_request or self.prepare_request(messages)
        if self.index < len(self.script):
            entry = dict(self.script[self.index])
            prompt = math.ceil(RATIO * MEASURE.measure(json.loads(prepared.body))["local_tokens"])
            usage = dict(entry.get("usage") or {})
            usage.update(prompt_tokens=prompt, total_tokens=prompt + (usage.get("completion_tokens") or 0))
            entry["usage"] = usage
            self.script[self.index] = entry
        return super().query(messages, prepared_request=prepared, **kwargs)


def seq(a: int, b: int) -> str:
    return f"seq {a} {b}"


def long_text(n_words: int) -> str:
    return " ".join(f"thought{i % 97}" for i in range(n_words))


# ------------------------------------------------------------------------------------------------ natural runs
def make_plan(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    write(d / "script.json", [])
    plan = {"version": "natural-plan.v2", "plan_id": f"m5-{name}", "registry": str(REGISTRY_V1), "tasks": [TASK],
            "repeats": 1, "model": "scripted/fixture", "settings": {"max_tokens": RESPONSE_CAP}, "store": str(d / "run"),
            "concurrency": 1, "detector": {"rule": "stall-clock/v8.1"},
            "budget": {"cost_limit_usd": 3.0, "step_limit": 500}, "purpose": "stall_supply",
            "tool_timeout_seconds": 60, "context_limit": LIMIT,
            "fixture": {"submission_protocol": FIXTURE, "scripts": {"*": str(d / "script.json")}}}
    write(d / "plan.json", plan)
    return d / "plan.json"


def run_natural(name: str, script: list[dict]) -> tuple[ArtifactStore, str, dict]:
    plan = natural.load_plan(make_plan(name))
    job = natural.jobs(plan)[0]
    model = ProviderCountingModel(script, model_name="scripted/fixture", model_settings={"max_tokens": RESPONSE_CAP})
    status = natural.run_job(plan, job, model_factory_override=lambda config: model)
    return ArtifactStore(OUT / name / "run" / "store"), job["run_id"], status


def strip(messages):
    return [{k: v for k, v in m.items() if k != "extra"} for m in messages]


def sent_requests(store, evs, *, mode=None):
    """(request event, body, provider prompt tokens of its accepted/rejected response or None) per model request."""
    responses = {e.parent_event_id: e for e in evs if e.kind == "model_response"}
    out = []
    for e in evs:
        if e.kind == "model_request" and (e.payload or {}).get("mode") == mode:
            r = responses.get(e.event_id)
            out.append((e, json.loads(store.get_bytes(e.input_ref)),
                        r.usage.input_tokens if r is not None and r.usage is not None else None))
    return out


def request_checks(store, evs, full: list[dict], *, mode=None) -> dict:
    """Every request of one run (or one turn mode) against the full record."""
    reqs = sent_requests(store, evs, mode=mode)
    full_sent = strip(full)
    first = reqs[0][1]["messages"][:2] if reqs else None
    fits_local, fits_provider, only_tools, prefix_order, head_kept = [], [], [], [], []
    for e, body, provider in reqs:
        cap = body.get("max_tokens", body.get("max_completion_tokens"))
        lim = e.payload.get("context_limit") or {}
        fits_local.append(lim.get("size") is not None and lim["size"] <= WINDOW and cap == RESPONSE_CAP)
        if provider is not None:
            fits_provider.append(provider + cap <= WINDOW)
        msgs = body["messages"]
        head_kept.append(msgs[:2] == first)
        if mode is None:  # continuation requests are prefixes of the run's full record, outputs hidden or not
            diff = [i for i, m in enumerate(msgs) if i >= len(full_sent) or m != full_sent[i]]
            only_tools.append(all(i < len(full_sent) and msgs[i].get("role") == "tool"
                                  and msgs[i]["content"] == T.OUTPUT_PLACEHOLDER
                                  and dict(msgs[i], content=full_sent[i]["content"]) == full_sent[i] for i in diff))
        tools = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
        hidden = [i for i in tools if msgs[i]["content"] == T.OUTPUT_PLACEHOLDER]
        prefix_order.append(hidden == tools[:len(hidden)])
    return {"requests": len(reqs), "every_request_fits_local_size": all(fits_local),
            "every_request_fits_provider_count": bool(fits_provider) and all(fits_provider),
            "provider_counts_checked": len(fits_provider),
            "system_and_task_messages_intact": all(head_kept),
            "only_tool_outputs_changed_to_placeholder": all(only_tools) if mode is None else None,
            "hidden_outputs_are_the_oldest": all(prefix_order)}


def masking_checks(store, evs, *, mode=None) -> dict:
    """Each logged hiding: its fields, its decision, and minimality (one output fewer would not fit)."""
    masked = [e for e in lifecycle(evs, "context_limit_masked") if e.payload.get("mode") == mode]
    reqs = [e for e in evs if e.kind == "model_request" and (e.payload or {}).get("mode") == mode]
    logged, minimal, decided = [], [], []
    for m in masked:
        p = m.payload
        logged.append(all(k in p for k in ("step", "outputs", "tokens_before", "tokens_after", "window_tokens", "ratio"))
                      and all({"message_index", "tool_call_id", "context_step", "original_sha256", "original_ref"}
                              <= set(o) for o in p["outputs"]) and p["outputs"])
        decided.append(p["tokens_before"]["size"] > WINDOW >= p["tokens_after"]["size"])
        nxt = next((r for r in reqs if r.sequence > m.sequence), None)
        if nxt is None:  # the hiding before a branch's first call is validated at delivery; the call follows
            minimal.append(False)
            continue
        body = json.loads(store.get_bytes(nxt.input_ref))
        last = p["outputs"][-1]
        original = store.get_bytes(from_jsonable(ArtifactRef, last["original_ref"])).decode("utf-8")
        undo = dict(body, messages=list(body["messages"]))
        undo["messages"][last["message_index"]] = dict(body["messages"][last["message_index"]], content=original)
        measured = MEASURE.measure(undo)
        minimal.append(MEASURE.size(measured["local_tokens"], measured["response_cap"], p["ratio"]) > WINDOW)
    return {"events": len(masked), "outputs_hidden": sum(len(m.payload["outputs"]) for m in masked),
            "steps": [m.payload["step"] for m in masked],
            "logged_with_step_outputs_and_tokens": bool(logged) and all(logged),
            "decision_matches_window": bool(decided) and all(decided),
            "one_fewer_would_not_fit": bool(minimal) and all(minimal)}


def check_N(report: dict) -> dict:
    store, rid, status = run_natural("N", [
        step("Look at the task.", ["ls /app/task_file"]),
        step("Numbers.", [seq(1, 400)]),
        step("Run the baseline packer.", [PARTIAL]),
        step("More numbers.", [seq(401, 800)]),
        step("More numbers.", [seq(801, 1200)]),
        step("Submit.", [SUBMIT]),
        step("More numbers.", [seq(1201, 1600)]),
        step("More numbers.", [seq(1601, 2000)]),
        step("More numbers.", [seq(2001, 2600)]),
        step("More numbers.", [seq(2601, 3000)]),
        step("Apply a solution.", [(PACKAGE / "private/solution/solve.sh").read_text()]),
        step("More numbers.", [seq(3001, 3400)]),
        step("Submit.", [SUBMIT]),
        step("Never reached.", ["echo never"])])
    evs = events(store, rid)
    points = point_checkpoints(store, rid)
    last_cp = max(store.list_checkpoints(rid), key=lambda c: c.ordinal)
    full, _ = full_messages(store, store.get_json(last_cp.state.history))
    masked = lifecycle(evs, "context_limit_masked")
    first_step = masked[0].payload["step"] if masked else None
    histories = {k: store.get_json(store.get_checkpoint(cid).state.history) for k, cid in points.items()}
    hidden_at = {k: sum(1 for m in h["messages"] if MASK_KEY in (m.get("extra") or {})) for k, h in histories.items()}
    base = common_checks(store, rid, evs)
    reqs = request_checks(store, evs, full)
    mask = masking_checks(store, evs)
    checks = dict(base["checks"])
    checks.update({k: v for k, v in reqs.items() if isinstance(v, bool)})
    checks.update({k: v for k, v in mask.items() if isinstance(v, bool)})
    checks.update({
        "outputs_hidden_more_than_once": mask["events"] >= 2,
        "checkpoints_carry_the_masked_view": first_step is not None and all(
            (hidden_at[k] > 0) == (k >= first_step) for k in hidden_at),
        "masked_checkpoints_still_branchable": first_step is not None and all(
            k in points for k in range(first_step, base["steps"])),
        "accounting_untouched": all(h["counters"]["n_calls"] == k for k, h in histories.items()),
        "passing_submission_ends_run": status["termination_reason"] == "sequence_complete:native_success",
        "status_reports_hiding": (status.get("context_limit") or {}).get("outputs_hidden") == mask["outputs_hidden"],
        "binding_logged": any((e.payload.get("context_limit") or {}).get("window_tokens") == WINDOW
                              for e in lifecycle(evs, "worker_policy_bound")),
    })
    report["N_natural"] = {"checks": checks, "termination": status["termination_reason"], "requests": reqs,
                           "masking": mask, "first_masked_step": first_step, "hidden_outputs_per_point": hidden_at,
                           "request_sizes": [(e.payload.get("context_limit") or {}).get("size")
                                             for e, _, _ in sent_requests(store, evs)],
                           "provider_prompt_tokens": [p for _, _, p in sent_requests(store, evs)]}
    return {"store": store, "root": OUT / "N" / "run" / "store", "run_id": rid, "points": points,
            "first_step": first_step, "steps": base["steps"]}


def check_X(report: dict) -> None:
    store, rid, status = run_natural("X", [
        step("Numbers.", [seq(1, 400)]),
        step(long_text(9000), ["echo ok"]),
        step("Never reached.", ["echo never"])])
    evs = events(store, rid)
    ex = lifecycle(evs, "context_window_exhausted")
    terminal = next(e.payload for e in reversed(evs) if e.kind == "terminal")
    report["X_too_big"] = {"checks": {
        "ends_with_context_window_exhausted": status["termination_reason"] == "context_window_exhausted"
                                              and status["status"] == "completed",
        "exhaustion_logged_with_tokens": len(ex) == 1 and ex[0].payload["tokens_after"]["size"] > WINDOW,
        "nothing_sent_after": sum(e.kind == "model_request" for e in evs) == 2,
        "end_state_graded": terminal["termination"]["terminal_grade_valid"] is True,
    }, "termination": status["termination_reason"], "exhausted": ex[0].payload if ex else None}


# ------------------------------------------------------------------------------------------------ branches
def branch(name: str, spec: dict, script: list[dict]) -> dict:
    out = OUT / "branches" / name / "a0"
    model = ProviderCountingModel(script, model_name="scripted/fixture", model_settings={"max_tokens": RESPONSE_CAP})
    res = run_branch(spec, out_dir=out, model_factory=lambda config: model)
    store = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(spec["source"]["store"]))
    return {"result": res, "store": store, "events": events(store, res["run_id"]), "script_used": model.index}


def stall_of(n: dict, k: int, tag: str, *, ledger: bool) -> dict:
    cid = n["points"][k]
    d = OUT / "stall" / tag
    if ledger:
        build_stall(ReadOnlyStore(n["root"]), n["run_id"], cid, d, ScriptedLabeller(ArtifactStore(d / "labeller")),
                    recipe_cap=120, max_chars=300_000)
    return {"stall_id": tag, "store": str(n["root"]), "run_id": n["run_id"], "checkpoint_id": cid, "dir": str(d),
            "has_boundary": ledger, "step_index": k, "has_recipe": ledger}


def review_call(args: dict) -> dict:
    return {"content": None, "reasoning": "Keep going as it is.",
            "raw_tool_calls": [{"id": "rv-1", "type": "function",
                                "function": {"name": T.REVIEW_TOOL_NAME, "arguments": json.dumps(args)}}],
            "usage": {"completion_tokens": 300, "cost": 0.0}}


def check_branches(report: dict, n: dict) -> None:
    store, rid, points, k0 = n["store"], n["run_id"], n["points"], n["first_step"]
    after = k0 + 1  # a point whose saved context already holds hidden outputs
    c, out = {}, {}

    # B1: arm 1 from a point after hiding
    s1 = stall_of(n, after, "B1", ledger=False)
    # (the solution is already in W at that point; its long command stays in the context, as the agent's own text)
    b1 = branch("B1", arms_v2.arm_spec(s1, "1_continue", 0, purpose="tuning", prefix="m5", caps=None), [
        step("More numbers.", [seq(5001, 5400)]), step("More numbers.", [seq(5401, 5800)]),
        step("Submit.", [SUBMIT]), step("Never reached.", ["echo never"])])
    src_hist = store.get_json(store.get_checkpoint(points[after]).state.history)
    first = sent_requests(b1["store"], b1["events"])[0][1]
    validated = lifecycle(b1["events"], "next_request_validated")[0].payload
    restored = lifecycle(b1["events"], "history_restored")[0].payload
    delivered = b1["store"].get_json(from_jsonable(ArtifactRef, restored["delivered_history_ref"]))
    natural_reqs = [e for e in events(store, rid) if e.kind == "model_request"]
    src_hidden = {i for i, m in enumerate(src_hist["messages"]) if MASK_KEY in (m.get("extra") or {})}
    first_hidden = {i for i, m in enumerate(first["messages"])
                    if m.get("role") == "tool" and m["content"] == T.OUTPUT_PLACEHOLDER}
    b1_full, _ = full_messages(b1["store"], b1["store"].get_json(
        max(b1["store"].list_checkpoints(b1["result"]["run_id"]), key=lambda x: x.ordinal).state.history))
    b1_reqs = request_checks(b1["store"], b1["events"], b1_full)
    b1_mask = masking_checks(b1["store"], b1["events"])
    # the restored context is the saved masked view (hidden outputs and their marks included); its first request
    # keeps them hidden and hides exactly what the natural run hid at the same step
    c["B1_restore_reproduces_masked_view"] = (delivered == src_hist["messages"] and bool(src_hidden)
                                              and src_hidden <= first_hidden)
    c["B1_first_request_equals_natural_request_at_that_step"] = (
        first == json.loads(store.get_bytes(natural_reqs[after].input_ref)))
    c["B1_restore_round_trip_with_hidden_outputs"] = all(v for k_, v in restored["validation"].items()
                                                         if k_ != "revision")
    c["B1_request_validated"] = validated["valid"] and validated["checks"]["context_limit_hid_only_tool_outputs"]
    c["B1_hiding_continues_in_branch"] = b1_mask["events"] >= 1 and b1_mask["logged_with_step_outputs_and_tokens"] \
        and b1_mask["one_fewer_would_not_fit"]
    c["B1_every_request_fits"] = (b1_reqs["every_request_fits_local_size"]
                                  and b1_reqs["every_request_fits_provider_count"]
                                  and b1_reqs["system_and_task_messages_intact"]
                                  and b1_reqs["only_tool_outputs_changed_to_placeholder"]
                                  and b1_reqs["hidden_outputs_are_the_oldest"])
    c["B1_anchor_from_source_record"] = (b1_mask["events"] >= 1 and (lifecycle(b1["events"], "context_limit_masked")[0]
                                         .payload.get("anchor") or {}).get("source") in ("recorded", "this_run"))
    c["B1_passes"] = b1["result"]["end_reason"] == "passing_submission"
    c["B1_result_reports_hiding"] = (b1["result"]["details"]["context_limit"] or {}).get("outputs_hidden") \
        == b1_mask["outputs_hidden"]
    out["B1"] = {"result": b1["result"], "requests": b1_reqs, "masking": b1_mask}

    # B2: arm 1 from the point just before the first hiding: the same request as the natural run's at that step
    s2 = stall_of(n, k0 - 1, "B2", ledger=False)
    b2 = branch("B2", arms_v2.arm_spec(s2, "1_continue", 0, purpose="tuning", prefix="m5",
                                       caps={"new_token_cap": None, "submission_cap": None}) | {
        "budget": {"new_token_cap": None, "submission_cap": None, "cost_limit_usd": 3.0, "step_limit": 1}},
        [step("More numbers.", [seq(1, 3)]), step("Never reached.", ["echo never"])])
    natural_req = natural_reqs[k0 - 1]  # the request of step k0
    branch_req = sent_requests(b2["store"], b2["events"])[0][0]
    c["B2_same_hiding_as_natural_run"] = (json.loads(b2["store"].get_bytes(branch_req.input_ref))
                                          == json.loads(store.get_bytes(natural_req.input_ref))
                                          and branch_req.payload["context_limit"]["local_tokens"]
                                          == natural_req.payload["context_limit"]["local_tokens"]
                                          and [o["message_index"] for o in lifecycle(b2["events"],
                                               "context_limit_masked")[0].payload["outputs"]]
                                          == [o["message_index"] for o in lifecycle(events(store, rid),
                                              "context_limit_masked")[0].payload["outputs"]])
    out["B2"] = {"result": b2["result"]}

    # B3: arm 2 at an early point: the review request fits only once outputs are hidden; the hiding stays in the turn
    early = EARLY
    c["B3_point_is_before_any_hiding"] = early < k0 - 1
    s3 = stall_of(n, early, "B3", ledger=True)
    ledger3 = json.loads((Path(s3["dir"]) / "ledger.json").read_text())
    b3 = branch("B3", arms_v2.arm_spec(s3, "2_ledger_review", 0, purpose="tuning", prefix="m5", caps=None) | {
        "budget": {"new_token_cap": None, "submission_cap": None, "cost_limit_usd": 3.0, "step_limit": 2}},
        [review_call({"rewind": None, "drop": [], "keep": [], "change_nothing": True}),
         step("Continue.", [seq(1, 3)]), step("Never reached.", ["echo never"])])
    turn_mask = masking_checks(b3["store"], b3["events"], mode="review")
    turn_reqs = request_checks(b3["store"], b3["events"], [], mode="review")
    src3 = store.get_json(store.get_checkpoint(points[early]).state.history)
    cont3 = sent_requests(b3["store"], b3["events"])
    view3 = json.loads((Path(s3["dir"]) / "review_view_ledger.json").read_text())["text"]
    review_body = sent_requests(b3["store"], b3["events"], mode="review")[0][1]
    c["B3_review_request_hid_outputs"] = (turn_mask["events"] == 1 and turn_mask["decision_matches_window"]
                                          and turn_mask["logged_with_step_outputs_and_tokens"]
                                          and turn_reqs["every_request_fits_local_size"]
                                          and turn_reqs["system_and_task_messages_intact"]
                                          and turn_reqs["hidden_outputs_are_the_oldest"])
    c["B3_review_view_is_full_record"] = (view3 in review_body["messages"][-1]["content"]
                                          and seq_text(1, 400) in view3)
    c["B3_turn_hiding_not_in_continuation"] = (bool(cont3) and cont3[0][1]["messages"] == strip(src3["messages"])
                                               and not any(MASK_KEY in (m.get("extra") or {})
                                                           for m in src3["messages"]))
    c["B3_valid_change_nothing"] = b3["result"]["choice"]["valid"] and b3["result"]["flags"]["no_change"]
    out["B3"] = {"result": b3["result"], "turn_masking": turn_mask, "turn_requests": turn_reqs,
                 "ledger_items": len(ledger3["items"])}

    # Ledger at a point after hiding: built from the full record
    s4 = stall_of(n, after, "B4", ledger=True)
    ledger4 = json.loads((Path(s4["dir"]) / "ledger.json").read_text())
    ctx4 = load_context(ReadOnlyStore(n["root"]), rid, points[after])
    hidden_idx = [i for i, m in enumerate(src_hist["messages"]) if MASK_KEY in (m.get("extra") or {})]
    delivered = {store.get_bytes(e.delivered_ref).decode("utf-8") for e in events(store, rid)
                 if e.kind == "observation_delivery"}
    out_texts = {f.raw for s in ctx4.steps for f in s.outputs()}
    c["ledger_reads_hidden_outputs_from_record"] = (ctx4.checks["outputs_read_back_from_record"] == hidden_idx
                                                    and bool(hidden_idx) and not ctx4.checks["issues"]
                                                    and T.OUTPUT_PLACEHOLDER not in out_texts
                                                    and out_texts <= delivered)
    c["ledger_quotes_and_fingerprint_check"] = check_quotes(ReadOnlyStore(n["root"]), ledger4)["pass"]
    view4 = json.loads((Path(s4["dir"]) / "review_view_context.json").read_text())["text"]
    c["context_view_is_full_record"] = seq_text(1, 400) in view4 and T.OUTPUT_PLACEHOLDER not in view4

    # B4: arm 2 after hiding: the review request cannot fit -> context_window_exhausted before any continuation
    b4 = branch("B4", arms_v2.arm_spec(s4, "2_ledger_review", 0, purpose="tuning", prefix="m5", caps=None),
                [review_call({"rewind": None, "drop": [], "keep": [], "change_nothing": True}),
                 step("Never reached.", ["echo never"])])
    ex4 = lifecycle(b4["events"], "context_window_exhausted")
    c["B4_review_too_big_ends_branch"] = (b4["result"]["end_reason"] == "context_window_exhausted"
                                          and len(ex4) == 1 and ex4[0].payload.get("mode") == "review"
                                          and not sent_requests(b4["store"], b4["events"])
                                          and not sent_requests(b4["store"], b4["events"], mode="review")
                                          and b4["result"]["choice"]["stopped"] == "ContextWindowExhausted")
    out["B4"] = {"result": b4["result"], "exhausted": ex4[0].payload if ex4 else None}

    # B5: arm 1 after hiding, whose own long answer overflows
    s5 = stall_of(n, after, "B5", ledger=False)
    b5 = branch("B5", arms_v2.arm_spec(s5, "1_continue", 0, purpose="tuning", prefix="m5", caps=None), [
        step(long_text(9000), ["echo ok"]), step("Never reached.", ["echo never"])])
    ex5 = lifecycle(b5["events"], "context_window_exhausted")
    c["B5_branch_too_big_ends_with_reason"] = (b5["result"]["end_reason"] == "context_window_exhausted"
                                               and len(ex5) == 1 and len(sent_requests(b5["store"], b5["events"])) == 1
                                               and b5["result"]["final_share_passed"] is not None)
    out["B5"] = {"result": b5["result"], "exhausted": ex5[0].payload if ex5 else None}
    report["branches"] = {"checks": c, **out}


def seq_text(a: int, b: int) -> str:
    return "\n".join(str(i) for i in range(a, b + 1))


def main() -> None:
    t0 = time.monotonic()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report = {"what": __doc__.split("\n\n")[0], "command": "python configs/v2/engine_e2e/m5_context_limit.py",
              "window_tokens": WINDOW, "response_cap": RESPONSE_CAP, "provider_ratio": RATIO, "provider_calls": 0}
    n = check_N(report)
    check_X(report)
    check_branches(report, n)
    report["all_checks"] = {f"{section}.{name}": value for section, body in report.items()
                            if isinstance(body, dict) and isinstance(body.get("checks"), dict)
                            for name, value in body["checks"].items()}
    report["passed"] = all(report["all_checks"].values())
    report["wall_seconds"] = round(time.monotonic() - t0, 1)
    write(ARTIFACTS / "m5-context-limit.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in report["all_checks"].items() if not v],
                      "first_masked_step": n["first_step"], "steps": n["steps"],
                      "request_sizes": report["N_natural"]["request_sizes"],
                      "end_reasons": {k: v["result"]["end_reason"] for k, v in report["branches"].items()
                                      if isinstance(v, dict) and "result" in v},
                      "seconds": report["wall_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
