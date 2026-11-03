"""Engine end-to-end check of arm set v3 (user, 2026-10-16/17): all six arms on a real stored natural-run checkpoint,
in Docker, with scripted models only (no provider calls).

The natural run goes through the driver (configs/v2/natural.py) on terminalbench-llm-inference-batching-scheduler-v1
with the count-feedback fixture (as m1, m2 and m5) and the context-limit rule bound to an artificial WINDOW, with a
scripted "provider" that counts RATIO prompt tokens per local token (as m5). Its steps: look, numbers, the public
partial control (5 of 6 tests pass), markers A in W (/app) and R (/tmp), a failing submission, an idea and numbers,
markers B plus a failing command, a second identical failing submission (the stall point S), then numbers, the
reference solution and a passing submission (all after S: content no branch may see).

The flat Ledger and the context view of point S are built with track D's code and its scripted (fake) labeller. The
specs come from arms_v2.arm_spec_v3 with P (the judge's stall onset point, an input) = point 4:

  1 continue            nothing changes
  2 Ledger review       flat Ledger; first call invalid (keeps a unit the rewind does not undo), retry valid: W, R and
                        the context to P, O2 dropped, two items from undone steps kept (handed back with their labels)
  3 context review      plain context; the context only to point 6, O2 dropped, step 7 kept
  4 forced rewind to P  W, R, the context and the outputs to P
  5 Ledger read-only    flat Ledger, no tool; a plain-text reply
  6 context read-only   plain context, no tool; first reply calls a tool (one retry), then plain text

Every continuation then runs a probe of the markers and two large outputs, so the context-limit rule hides outputs
in every arm, and ends at its step guard.

Usage: python configs/v2/engine_e2e/m6_arms_v3.py
Writes artifacts/v2/engine/m6-arms-v3.json in the main checkout; stores stay under
~/.cache/trajectory-retention/v2-engine/m6. No provider calls.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

from common import ARTIFACTS, PACKAGE, PARTIAL, REGISTRY_V1, ROOT, SCRATCH, SUBMIT, TASK, events, lifecycle, step, write
from m5_context_limit import ProviderCountingModel, natural, seq, sent_requests, strip
from trajectory_retention.contracts import ArtifactRef, from_jsonable
from trajectory_retention.execution.context_limit import MASK_KEY, RULE_VERSION, ContextLimit
from trajectory_retention.experiments import v2_text as T
from trajectory_retention.experiments.branch_v2 import (BranchStore, KEPT_OUTPUT_VERSION, ReadOnlyArtifactStore,
                                                        run_branch)
from trajectory_retention.experiments.context_edit import steps_of, visible_texts
from trajectory_retention.ledger import templates as LT
from trajectory_retention.ledger.__main__ import build_stall
from trajectory_retention.ledger.checks import check_views, leak_sources
from trajectory_retention.ledger.labeller import ScriptedLabeller
from trajectory_retention.ledger.render import label_text
from trajectory_retention.mining.points import checkpoint_step_index, point_checkpoints
from trajectory_retention.observers.t2.store_ro import ReadOnlyStore
from trajectory_retention.store import ArtifactStore

sys.path.insert(0, str(ROOT / "configs" / "v2"))
import arms_v2  # noqa: E402  track S: spec builder
from study_common import ARMS_V3, lint_worker_text  # noqa: E402

OUT = SCRATCH / "m6"
FIXTURE = "count-feedback-fixture/v1"
WINDOW = 12500        # artificial window, tokens: the v3 review requests fit it, context plus view would not
RESPONSE_CAP = 2000   # every request reserves it
RATIO = 1.17          # the scripted "provider" counts this many prompt tokens per local token (as m5)
MEASURE = ContextLimit(window_tokens=WINDOW, placeholder=T.OUTPUT_PLACEHOLDER)
LIMIT = {"rule": RULE_VERSION, "window_tokens": WINDOW, "source": "m6 e2e (artificial window)"}
S = 8                 # the stall point (after the second failing submission)
P = 4                 # the judge's stall onset point, as supplied to arm 4
PROBE = "cat /tmp/v6/marker; cat /app/v6_note.txt"
BIG_A, BIG_B = seq(5001, 6400), seq(6401, 7800)
IDEA = "Maybe the packer could use smaller batches for the long requests."


def make_plan(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    write(d / "script.json", [])
    plan = {"version": "natural-plan.v2", "plan_id": f"m6-{name}", "registry": str(REGISTRY_V1), "tasks": [TASK],
            "repeats": 1, "model": "scripted/fixture", "settings": {"max_tokens": RESPONSE_CAP}, "store": str(d / "run"),
            "concurrency": 1, "detector": {"rule": "stall-clock/v8.1"},
            "budget": {"cost_limit_usd": 3.0, "step_limit": 500}, "purpose": "stall_supply",
            "tool_timeout_seconds": 60, "context_limit": LIMIT,
            "fixture": {"submission_protocol": FIXTURE, "scripts": {"*": str(d / "script.json")}}}
    write(d / "plan.json", plan)
    return d / "plan.json"


def run_natural() -> tuple[ArtifactStore, Path, str, dict]:
    solve = (PACKAGE / "private/solution/solve.sh").read_text()
    script = [
        step("Look at the task.", ["ls /app/task_file"]),
        step("Numbers.", [seq(1, 400)]),
        step("Run the baseline packer.", [PARTIAL]),
        step("Leave markers.", ["mkdir -p /tmp/v6 && echo A > /tmp/v6/marker && echo A > /app/v6_note.txt"]),
        step("Submit.", [SUBMIT]),
        step(IDEA + " First more numbers.", [seq(401, 800)], reasoning="Thinking about batch sizes."),
        step("I'll update the markers.", ["echo B > /tmp/v6/marker && echo B > /app/v6_note.txt && ls /app/v6_missing"]),
        step("Submit.", [SUBMIT]),
        step("More numbers.", [seq(1201, 1600)]),
        step("Apply a solution.", [solve]),
        step("Submit.", [SUBMIT]),
        step("Never reached.", ["echo never"])]
    plan = natural.load_plan(make_plan("N"))
    job = natural.jobs(plan)[0]
    model = ProviderCountingModel(script, model_name="scripted/fixture", model_settings={"max_tokens": RESPONSE_CAP})
    status = natural.run_job(plan, job, model_factory_override=lambda config: model)
    root = OUT / "N" / "run" / "store"
    return ArtifactStore(root), root, job["run_id"], status


def review_call(args: dict, *, reasoning: str, tokens: int = 600, content: str | None = None) -> dict:
    return {"content": content, "reasoning": reasoning,
            "raw_tool_calls": [{"id": f"rv-{tokens}", "type": "function",
                                "function": {"name": T.REVIEW_TOOL_NAME, "arguments": json.dumps(args)}}],
            "usage": {"completion_tokens": tokens, "cost": 0.0}}


def reply(content: str, *, reasoning: str, tokens: int = 400) -> dict:
    return {"content": content, "commands": [], "reasoning": reasoning, "finish_reason": "stop",
            "usage": {"completion_tokens": tokens, "cost": 0.0}}


CONTINUATION_STEPS = 4  # probe, two large outputs, one more step: the last request holds both large outputs


def continuation() -> list[dict]:
    return [step("Check the state.", [PROBE], tokens=300), step("Numbers.", [BIG_A], tokens=300),
            step("Numbers.", [BIG_B], tokens=300), step("Done.", ["echo done"], tokens=300),
            step("Never reached.", ["echo never"])]


def branch(name: str, spec: dict, script: list[dict]) -> dict:
    out = OUT / "branches" / name / "a0"
    model = ProviderCountingModel(script, model_name="scripted/fixture", model_settings={"max_tokens": RESPONSE_CAP})
    res = run_branch(spec, out_dir=out, model_factory=lambda config: model)
    store = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(spec["source"]["store"]))
    evs = events(store, res["run_id"])
    return {"spec": spec, "result": res, "store": store, "events": evs, "script_used": model.index,
            "script": script}


def first_outputs(b: dict) -> list[str]:
    evs = b["events"]
    first = next((e for e in evs if e.kind == "model_response" and not e.payload.get("mode")), None)
    if first is None:
        return []
    calls = {e.event_id for e in evs if e.kind == "tool_call" and e.payload.get("model_call_id") == first.call_id}
    return [b["store"].get_bytes(e.output_ref).decode("utf-8", "replace") for e in evs
            if e.kind == "tool_result" and e.parent_event_id in calls]


def payload(evs, transition) -> dict:
    found = lifecycle(evs, transition)
    return found[0].payload if found else {}


def private_strings(store, root: Path, run_id: str, ledger: dict) -> dict:
    """What no worker request may contain: private evaluator data of the source run (track D's leak sources), the
    run's worker-visible content after the stall (its future, the reference solution included), and the lane's
    private references: lines of the solution and of the hidden tests, and the hidden test ids."""
    sources = leak_sources(ReadOnlyStore(root), ledger)
    # the harness records the worker-visible strings it binds (v2_text) in its own lifecycle events: they are
    # worker-visible by design, not evaluator data
    bound = {str(v) for v in T.binding().values() if isinstance(v, str)}
    sources["private"] = {s for s in sources["private"] if s not in bound}
    public = set()
    for f in (PACKAGE / "private/environment").rglob("*"):
        if f.is_file():
            try:
                public |= {line.strip() for line in f.read_text().splitlines()}
            except UnicodeDecodeError:
                pass
    refs = set()
    for f in [PACKAGE / "private/solution/solve.sh", *sorted((PACKAGE / "private/tests").glob("*"))]:
        if f.is_file():
            refs |= {line.strip() for line in f.read_text().splitlines()
                     if len(line.strip()) >= 30 and line.strip() not in public}
    checks = json.loads((PACKAGE / "private/checks.json").read_text())["checks"]
    refs |= {c.split("::")[-1] for c in checks}
    return {"private": sources["private"], "future": sources["future"], "references": refs}


def worker_text(body: dict) -> str:
    """Everything the worker reads in one request, decoded: every text of every message (tool outputs and commands
    are JSON in the context) and the tool list. The model name and settings are the harness's own request fields."""
    return "\n".join([t for m in body["messages"] for t in visible_texts(m)]
                     + [json.dumps(body.get("tools"), ensure_ascii=False)])


def leaks_in(text: str, pools: dict) -> dict:
    return {k: sorted(s for s in pool if s in text)[:3] for k, pool in pools.items() if any(s in text for s in pool)}


def main() -> None:
    t0 = time.monotonic()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report: dict = {"what": __doc__.split("\n\n")[0], "command": "python configs/v2/engine_e2e/m6_arms_v3.py",
                    "window_tokens": WINDOW, "response_cap": RESPONSE_CAP, "provider_ratio": RATIO,
                    "stall_point": S, "onset_point_P": P, "provider_calls": 0, "checks": {}}
    c = report["checks"]

    # ---- the natural run and the stall's Ledger files
    store, root, rid, status = run_natural()
    nevs = events(store, rid)
    points = point_checkpoints(store, rid)
    cid = points[S]
    src_hist = store.get_json(store.get_checkpoint(cid).state.history)
    src = src_hist["messages"]
    verdicts = [e.payload for e in lifecycle(nevs, "native_submission_verdict")]
    c["natural_run_passes_after_two_failing_submissions"] = (
        status["termination_reason"] == "sequence_complete:native_success"
        and [v.get("failing_count") for v in verdicts] == [1, 1, 0])
    c["stall_point_branchable_and_unmasked"] = (store.get_checkpoint(cid).branchable
                                                and checkpoint_step_index(store, store.get_checkpoint(cid)) == S
                                                and not any(MASK_KEY in (m.get("extra") or {}) for m in src))
    stall_dir = OUT / "stall" / "m6-N-p8"
    build_stall(ReadOnlyStore(root), rid, cid, stall_dir, ScriptedLabeller(ArtifactStore(stall_dir / "labeller")),
                max_chars=300_000)
    ledger = json.loads((stall_dir / "ledger.json").read_text())
    flat = json.loads((stall_dir / "review_view_ledger_flat.json").read_text())
    ctxv = json.loads((stall_dir / "review_view_context.json").read_text())
    old = json.loads((stall_dir / "review_view_ledger.json").read_text())
    stall = {"stall_id": "m6-N-p8", "store": str(root), "run_id": rid, "checkpoint_id": cid, "dir": str(stall_dir),
             "step_index": S, "has_boundary": True, "has_recipe": False}

    # ---- the flat Ledger: no approach layer
    legend_and_headers = "\n".join(l for l in flat["text"].splitlines() if l.startswith(("Steps are", "⟦S")))
    c["flat_view_has_no_approach_layer"] = (
        flat["kind"] == "ledger_flat" and LT.APPROACHES_HEADER not in flat["text"]
        and not re.search(r"⟦A\d+", flat["text"]) and "approach" not in legend_and_headers.lower()
        and not any(u["kind"] == "approach" for u in flat["units"].values())
        and LT.STATUS_TEXT["survives"] not in flat["text"] and LT.STATUS_TEXT["lost"] not in flat["text"])
    c["old_view_did_have_it"] = LT.APPROACHES_HEADER in old["text"] and bool(re.search(r"⟦A\d+", old["text"]))
    views = check_views(ledger, flat, ctxv)
    c["flat_and_context_views_show_the_same_steps"] = views["pass"]
    items_after_p = sorted((u for u, v in flat["units"].items() if v["kind"] == "item" and P < v["steps"][0] <= S
                            and not v["quote"].endswith("(sic)")), key=lambda u: int(u[1:]))
    keep_items = items_after_p[:2]
    c["items_to_keep_exist"] = len(keep_items) == 2 and len({flat["units"][u]["label"].split(" · ")[0]
                                                             for u in keep_items}) >= 1
    report["ledger"] = {"dir": str(stall_dir), "units": len(flat["units"]), "kept_items": {
        u: flat["units"][u] for u in keep_items}, "view_check": views}

    # ---- the six arms
    # the step guard counts the turn's calls too: turn calls + the continuation steps
    spec = lambda arm, turn_calls, **kw: arms_v2.arm_spec_v3(stall, arm, 0, purpose="tuning", prefix="m6", caps=None,
                                                              onset_point=P, **kw) | {
        "budget": {"new_token_cap": None, "submission_cap": None, "cost_limit_usd": 3.0,
                   "step_limit": turn_calls + CONTINUATION_STEPS}}
    a2 = {"rewind": {"workspace": P, "context": P, "runtime": P, "outputs": None}, "drop": ["O2"],
          "keep": keep_items, "change_nothing": False}
    a2_bad = dict(a2, keep=["S2"])
    a3 = {"rewind": {"workspace": None, "context": 6, "runtime": None, "outputs": None}, "drop": ["O2"],
          "keep": ["S7"], "change_nothing": False}
    r2_bad, r2 = "Rewind and keep step 2 as well.", "Rewinding to point 4 and keeping the idea and the failure."
    r3 = "Undo the last two steps of context, keep step 7."
    r5 = "Two submissions, one hidden test failing each time; the baseline packer is not enough."
    r6 = "I will note what I see in the context."
    note5 = "Notes for myself: the baseline packer passes 5 of 6; next, write my own packer."
    note6 = "Notes for myself: both submissions left one hidden test failing."
    runs = {
        "1_continue": branch("1_continue", spec("1_continue", 0), continuation()),
        "2_ledger_review": branch("2_ledger_review", spec("2_ledger_review", 2, opus_record=True), [
            review_call(a2_bad, reasoning=r2_bad, tokens=500), review_call(a2, reasoning=r2, tokens=700)]
            + continuation()),
        "3_context_review": branch("3_context_review", spec("3_context_review", 1), [
            review_call(a3, reasoning=r3, content="Dropping the long output.")] + continuation()),
        "4_forced_rewind_to_p": branch("4_forced_rewind_to_p", spec("4_forced_rewind_to_p", 0), continuation()),
        "5_ledger_read_only": branch("5_ledger_read_only", spec("5_ledger_read_only", 1), [
            reply(note5, reasoning=r5)] + continuation()),
        "6_context_read_only": branch("6_context_read_only", spec("6_context_read_only", 2), [
            step("Let me look first.", ["ls /app"], tokens=200), reply(note6, reasoning=r6)] + continuation()),
    }
    c["all_six_arms_ran"] = sorted(runs) == sorted(ARMS_V3)
    R = {n: b["result"] for n, b in runs.items()}
    c["no_errors_and_every_branch_ends_at_its_step_guard"] = all(r["end_reason"] == "step_guard" for r in R.values())
    report["end_reasons"] = {n: r["end_reason"] for n, r in R.items()}
    report["errors"] = {n: r.get("error") for n, r in R.items() if r.get("error")}
    c["specs_are_v3_and_arm4_carries_P"] = (all(b["spec"]["version"] == "branch-spec.v3" for b in runs.values())
                                            and runs["4_forced_rewind_to_p"]["spec"]["stall_onset_point"] == P
                                            and set(runs["4_forced_rewind_to_p"]["spec"]["rewind"].values()) == {P})

    pools = private_strings(store, root, rid, ledger)
    report["leak_pools"] = {k: len(v) for k, v in pools.items()}
    # positive controls: the pools hold what they should (the natural run's own later requests carry the solution;
    # its grades carry the hidden test ids)
    later_requests = [worker_text(b) for e, b, _ in sent_requests(store, nevs)][S:]
    c["leak_pools_positive_controls"] = (any(leaks_in(t, {"references": pools["references"]}) for t in later_requests)
                                         and bool(pools["private"]) and bool(pools["future"]))
    src_sent = strip(src)
    per: dict = {}
    for name, b in runs.items():
        st, evs, res = b["store"], b["events"], b["result"]
        p: dict = {}
        turn_reqs = sent_requests(st, evs, mode="review")
        cont = sent_requests(st, evs)
        validated = payload(evs, "next_request_validated")
        compiled = payload(evs, "branch_v2_compiled")
        started = payload(evs, "branch_started")
        donors = {ch: checkpoint_step_index(st, st.get_checkpoint(started[ch])) for ch in
                  ("workspace_donor", "runtime_donor", "history_donor") if started.get(ch)}
        first = cont[0][1]["messages"] if cont else []
        handback = next((st.get_bytes(e.delivered_ref).decode("utf-8") for e in evs if e.kind == "handback_delivery"),
                        None)
        # what the worker reads: the messages and the tools of every request (the model name and settings are the
        # harness's own request fields)
        all_bodies = [worker_text(bd) for _, bd, _ in turn_reqs + cont]
        p["validated"] = bool(validated.get("valid"))
        p["donor_points"] = donors
        p["leaks"] = {i: hit for i, t in enumerate(all_bodies) if (hit := leaks_in(t, pools))}
        p["no_private_or_future_text_in_any_request"] = not p["leaks"] and len(all_bodies) == len(turn_reqs) + len(cont)
        p["continuation_offers_only_bash"] = all([t["function"]["name"] for t in bd.get("tools") or []] == ["bash"]
                                                 for _, bd, _ in cont)
        # the context-limit rule in the continuation: outputs hidden, every request fits, only tool outputs hidden
        masked = [e.payload for e in lifecycle(evs, "context_limit_masked") if not e.payload.get("mode")]
        p["continuation_hid_outputs"] = len(masked) >= 1 and all(
            m["tokens_before"]["size"] > WINDOW >= m["tokens_after"]["size"] for m in masked)
        p["every_continuation_request_fits"] = bool(cont) and all(
            (e.payload.get("context_limit") or {}).get("size", WINDOW + 1) <= WINDOW
            and bd.get("max_tokens") == RESPONSE_CAP for e, bd, _ in cont)
        p["hidden_outputs_are_tool_outputs"] = all(
            bd["messages"][o["message_index"]].get("role") == "tool"
            and bd["messages"][o["message_index"]]["content"] == T.OUTPUT_PLACEHOLDER
            for m in masked for e, bd, _ in cont[-1:] for o in m["outputs"] if o["message_index"] < len(bd["messages"]))
        p["grades_only_at_the_end"] = sum(e.kind == "grade" for e in evs) == 1 and not res["submissions"]
        texts = [t for m in first for t in visible_texts(m)]
        p["generic_line_present"] = any(T.GENERIC_LINE in t for t in texts)
        p["no_stuck_wording_in_engine_text"] = not lint_worker_text(
            "\n".join([handback or ""] + [bd["messages"][-1]["content"].split("=====")[0] for _, bd, _ in turn_reqs]))
        if name in ("2_ledger_review", "3_context_review", "5_ledger_read_only", "6_context_read_only"):
            view = flat if name in ("2_ledger_review", "5_ledger_read_only") else ctxv
            body = turn_reqs[-1][1] if turn_reqs else {"messages": []}
            msgs = body["messages"]
            p["review_request_is_system_task_view"] = all(
                # the retry request also holds the format-error message (the rejected response never enters)
                len(bd["messages"]) in (3, 4) and bd["messages"][:2] == src_sent[:2]
                and bd["messages"][2]["role"] == "user" and view["text"] in bd["messages"][2]["content"]
                and T.SNAPSHOT_WORKSPACE.split("(")[0] in bd["messages"][2]["content"]
                and not any(m in bd["messages"] for m in src_sent[2:]) for _, bd, _ in turn_reqs) and bool(turn_reqs)
            lim = turn_reqs[0][0].payload.get("context_limit") or {} if turn_reqs else {}
            old_style = dict(turn_reqs[0][1], messages=src_sent + [turn_reqs[0][1]["messages"][2]]) if turn_reqs else {}
            old_m = MEASURE.measure(old_style) if old_style else {"local_tokens": 0, "response_cap": 0}
            p["review_sizes"] = {"request": lim.get("size"), "old_style_context_plus_view":
                                 MEASURE.size(old_m["local_tokens"], old_m["response_cap"], lim.get("ratio") or 1.0)}
            p["review_request_fits_window_without_hiding"] = (lim.get("size") is not None and lim["size"] <= WINDOW
                                                              and not any(e.payload.get("mode") for e in
                                                                          lifecycle(evs, "context_limit_masked")))
            p["old_style_request_would_not_fit"] = p["review_sizes"]["old_style_context_plus_view"] > WINDOW
            kept = payload(evs, "review_output_kept")
            kept_msg = first[-2] if len(first) >= 2 else {}  # the kept output, then the hand-back, end the request
            want_reasoning = {"2_ledger_review": r2, "3_context_review": r3, "5_ledger_read_only": r5,
                              "6_context_read_only": r6}[name]
            p["review_thinking_kept_before_handback"] = (
                bool(kept) and kept["rule"].startswith(KEPT_OUTPUT_VERSION) and kept["message_index"] == len(first) - 2
                and kept_msg.get("role") == "assistant" and kept_msg.get("reasoning") == want_reasoning
                and not kept_msg.get("tool_calls") and first[-1]["role"] == "user" and first[-1]["content"] == handback
                and validated["checks"].get("review_output_kept_after_context") is True)
            p["review_prompt_and_view_not_in_continuation"] = (
                validated["checks"].get("intervention_turn_absent") is True
                and not any(view["text"] in t or T.REVIEW_PROMPT in t or T.READ_ONLY_PROMPT in t for t in texts))
            if name in ("2_ledger_review", "3_context_review"):
                args = a2 if name == "2_ledger_review" else a3
                p["turn_offers_only_review_decision"] = all(
                    [t["function"]["name"] for t in bd.get("tools") or []] == [T.REVIEW_TOOL_NAME]
                    for _, bd, _ in turn_reqs)
                p["kept_decision_text"] = T.REVIEW_DECISION_KEPT.format(arguments=json.dumps(args)) in \
                    (kept_msg.get("content") or "")
            else:
                p["turn_exposes_no_tool"] = all("tools" not in bd for _, bd, _ in turn_reqs) and bool(turn_reqs)
                p["kept_reply_text"] = kept_msg.get("content") == (note5 if name == "5_ledger_read_only" else note6)
                p["nothing_changed"] = (set(donors.values()) == {S} and compiled.get("context_edits") == []
                                        and not p["generic_line_present"]
                                        and first[:-2] == src_sent and handback == "\n".join(
                    [T.HANDBACK_HEADER] + [T.GRADER_RECORD_LINE.format(i=g["i"], k=g["k"], n=g["n"])
                                           for g in compiled["grader_record"]]))
        per[name] = p
        report.setdefault("branches", {})[name] = {"result": res, "checks": p, "handback": handback,
                                                   "turn_requests": len(turn_reqs), "continuation_requests": len(cont),
                                                   "first_request_messages": len(first)}

    grader = [T.GRADER_RECORD_LINE.format(i=g["i"], k=g["k"], n=g["n"])
              for g in payload(runs["1_continue"]["events"], "branch_v2_compiled")["grader_record"]]
    c["grader_record_is_the_worker_visible_counts"] = grader == [
        T.GRADER_RECORD_LINE.format(i=i + 1, k=k, n=1) for i, k in enumerate((5, 8))]
    # arm 1
    p1 = per["1_continue"]
    first1 = sent_requests(runs["1_continue"]["store"], runs["1_continue"]["events"])[0][1]["messages"]
    c["arm1_context_unchanged_no_handback"] = (first1 == src_sent and report["branches"]["1_continue"]["handback"] is None
                                               and set(p1["donor_points"].values()) == {S})
    # arm 4: W, R and the context to P
    b4 = runs["4_forced_rewind_to_p"]
    first4 = sent_requests(b4["store"], b4["events"])[0][1]["messages"]
    at_p = strip(store.get_json(store.get_checkpoint(points[P]).state.history)["messages"])
    out4 = "\n".join(first_outputs(b4))
    c["arm4_rewinds_w_r_and_context_to_P"] = (set(per["4_forced_rewind_to_p"]["donor_points"].values()) == {P}
                                              and first4[:-1] == at_p and out4.split() == ["A", "A"])
    c["arm4_handback_generic_and_grader_only"] = report["branches"]["4_forced_rewind_to_p"]["handback"] == "\n".join(
        [T.HANDBACK_HEADER, T.GENERIC_LINE] + grader)
    # arm 2: rewind to P, O2 dropped, kept items with their labels
    b2 = runs["2_ledger_review"]
    hb2 = report["branches"]["2_ledger_review"]["handback"] or ""
    labels = [label_text(ledger, u, flat=True) for u in keep_items]
    c["arm2_kept_items_handed_back_with_labels"] = (
        all(l in hb2 for l in labels) and all(re.match(r"(decision|fact|claim|idea) \(step \d+", l) for l in labels)
        and all(LT.OPEN not in l for l in labels) and LT.STATUS_TEXT["survives"] not in hb2
        and hb2.startswith("\n".join([T.HANDBACK_HEADER, T.GENERIC_LINE] + grader)))
    first2 = sent_requests(b2["store"], b2["events"])[0][1]["messages"]
    o2 = [m for m in first2 if m.get("role") == "tool"][1]
    c["arm2_rewind_to_P_and_O2_dropped"] = (per["2_ledger_review"]["donor_points"] == {
        "workspace_donor": P, "runtime_donor": P, "history_donor": P} and o2["content"] == T.OUTPUT_PLACEHOLDER
        and "A\nA" in "\n".join(first_outputs(b2)))
    c["arm2_invalid_attempt_not_kept"] = (b2["result"]["choice"]["calls"] == 2 and b2["result"]["choice"]["valid"]
                                          and not any(r2_bad in t for m in first2 for t in visible_texts(m)))
    c["arm2_opus_input_is_the_head_only_review_request"] = (lambda o: o["request"]["messages"][:2] == src_sent[:2]
        and len(o["request"]["messages"]) == 3)(json.loads((OUT / "branches/2_ledger_review/a0/opus-review-input.json")
                                                          .read_text()))
    # arm 3: the context to 6 only; W and R stay at S
    hb3 = report["branches"]["3_context_review"]["handback"] or ""
    c["arm3_context_only_rewind_with_kept_step"] = (per["3_context_review"]["donor_points"] == {
        "workspace_donor": S, "runtime_donor": S, "history_donor": 6} and "B\nB" in "\n".join(
        first_outputs(runs["3_context_review"])) and label_text(ledger, "S7") in hb3 and T.GENERIC_LINE in hb3)
    # the generic line exactly in the arms that undo or remove something
    c["generic_line_only_where_something_was_undone"] = {n: per[n]["generic_line_present"] for n in per} == {
        "1_continue": False, "2_ledger_review": True, "3_context_review": True, "4_forced_rewind_to_p": True,
        "5_ledger_read_only": False, "6_context_read_only": False}
    c["arm2_and_arm3_probe_outputs"] = {"2": "\n".join(first_outputs(b2)).split(),
                                        "3": "\n".join(first_outputs(runs["3_context_review"])).split()} == {
        "2": ["A", "A"], "3": ["B", "B"]}
    c["arm6_tool_call_got_one_retry"] = (runs["6_context_read_only"]["result"]["choice"]["calls"] == 2
                                         and runs["6_context_read_only"]["result"]["choice"]["valid"])
    c["review_turn_tokens_counted"] = all(R[n]["review_tokens"] > 0 and R[n]["new_tokens"] >= R[n]["review_tokens"]
                                          for n in ("2_ledger_review", "3_context_review", "5_ledger_read_only",
                                                    "6_context_read_only"))
    # per-branch checks, all arms
    for key in sorted({k for p in per.values() for k, v in p.items() if isinstance(v, bool)}):
        c[f"every_arm:{key}"] = all(p[key] for p in per.values() if key in p and key != "generic_line_present")
    c.pop("every_arm:generic_line_present", None)
    report["passed"] = all(v for v in c.values() if isinstance(v, bool))
    report["seconds"] = round(time.monotonic() - t0, 1)
    write(ARTIFACTS / "m6-arms-v3.json", report)
    print(json.dumps({"passed": report["passed"], "failed": [k for k, v in c.items() if v is False],
                      "end_reasons": report["end_reasons"], "errors": report["errors"],
                      "review_sizes": {n: p.get("review_sizes") for n, p in per.items() if p.get("review_sizes")},
                      "seconds": report["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
