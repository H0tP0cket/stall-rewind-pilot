"""A stub for track E's run_branch, for testing the study machinery without Docker or model calls. SYNTHETIC.

Same signature and result.json shape as trajectory_retention.experiments.branch_v2.run_branch (brief-interfaces.md).
Outcomes are drawn from a synthetic world file (env V2_STUB_WORLD) seeded by the spec's seed, so a rerun of a branch
gives the same result. Like the real engine it writes, per branch: a branch store (ArtifactStore) whose events are
the review or note turn (payload mode), the hand-back message and the continuation steps; result.json (plus the
"store" and "run_id" fields this track asks E to add); and, for review arms with opus_record, the pending
opus-review-input.json holding the exact review request. Nothing here is evidence about the study.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from study_common import ROOT, read_json, worker_text, write_json, validate_decision

sys.path.insert(0, str(ROOT / "src"))
from trajectory_retention.contracts import EventRecord, Usage, Visibility  # noqa: E402
from trajectory_retention.store import ArtifactStore  # noqa: E402

STUB_SYSTEM = "You are a helpful assistant that can interact with a computer."


def _fail_hash(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


class _Events:
    def __init__(self, root: Path, run_id: str):
        self.st = ArtifactStore(root)
        self.run_id = run_id

    def add(self, kind: str, *, output=None, delivered: str | None = None, usage: dict | None = None, **payload):
        out = self.st.put_json(output, visibility=Visibility.WORKER_VISIBLE) if output is not None else None
        dl = self.st.put_bytes(delivered.encode(), media_type="text/plain; charset=utf-8",
                               visibility=Visibility.WORKER_VISIBLE) if delivered is not None else None
        self.st.append_event(EventRecord(run_id=self.run_id, kind=kind, actor="worker" if kind == "model_response"
                                         else "harness", visibility=Visibility.WORKER_VISIBLE, output_ref=out,
                                         delivered_ref=dl, usage=Usage(**usage) if usage else None, payload=payload))


def _choice(rng: random.Random, units: dict, mode: str, last_step: int) -> dict:
    if rng.random() < 0.15:
        rw = None
        drop, keep, nothing = [], [], True
    else:
        nothing = False
        point = rng.randint(max(0, last_step - 8), max(0, last_step - 2))
        if mode == "per_channel":
            rw = {"workspace": point, "context": point, "runtime": point if rng.random() < 0.7 else None,
                  "outputs": None}
        else:
            rw = point
        undone = [u for u, v in units.items() if min(v.get("steps") or [0]) > point]
        kept_ok = [u for u in undone if units[u].get("kind") in ("item", "step", "output")]
        early = [u for u, v in units.items() if max(v.get("steps") or [0]) <= point and v.get("kind") in ("output", "item")]
        keep = rng.sample(kept_ok, min(len(kept_ok), rng.randint(0, 2)))
        drop = rng.sample(early, min(len(early), rng.randint(0, 2)))
    args = {"drop": drop, "keep": keep, "change_nothing": nothing}
    args = ({"rewind": rw} if mode == "per_channel" else {"rewind_to": rw}) | args
    return args


def run_branch(spec: dict, *, out_dir: Path, model_factory=None) -> dict:
    world = read_json(os.environ["V2_STUB_WORLD"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attempt = out.name
    rng = random.Random(f"{spec['seed']}|{attempt}")
    t0 = time.time()
    lo, hi = world.get("delay_seconds", [0.0, 0.0])
    time.sleep(rng.uniform(lo, hi))
    text, _ = worker_text()
    arm, bid = spec["arm"], spec["branch_id"]
    prof = world["runs"][spec["source"]["run_id"]]
    n_tests = prof["n_tests"]
    price = world["price"]
    run_id = f"branch:{bid}:{attempt}"
    base = {"branch_id": bid, "arm": arm, "repeat": spec["repeat"], "store": str(out / "store"), "run_id": run_id,
            "stub": "SYNTHETIC stub_branch (no model, no Docker)"}
    if f"{spec['purpose']}|{bid}" in world.get("error_once", []) and attempt == "a0":
        r = base | {"end_reason": "error", "solved": False, "submissions": [], "best_share_passed": None,
                    "final_share_passed": None, "choice": None, "review_tokens": 0, "note_tokens": 0,
                    "handback_tokens": 0, "new_tokens": 0, "steps": 0, "cost_usd": 0.0, "cost_usd_uncached": 0.0,
                    "restore_seconds": round(rng.uniform(5, 15), 2), "wall_seconds": round(time.time() - t0, 3),
                    "flags": {"mismatch": False, "failed_command": False, "invalid_call": False, "no_change": False},
                    "error": "stub: simulated infrastructure error (restore failed)"}
        write_json(out / "result.json", r)
        return r
    ev = _Events(out / "store", run_id)
    flags = {"mismatch": False, "failed_command": False, "invalid_call": False, "no_change": False}
    budget = spec["budget"]
    used_new = used_cost = used_uncached = 0.0
    review_tokens = note_tokens = handback_tokens = 0
    choice = None
    stall_step = prof["stall_step"]
    undone_or_removed = spec.get("rewind") is not None

    def charge(gen: int, prompt: int):
        nonlocal used_new, used_cost, used_uncached
        cached = int(prompt * 0.9)
        used_new += gen
        used_cost += gen * price["out"] + cached * price["cached_in"] + (prompt - cached) * price["in"]
        used_uncached += gen * price["out"] + prompt * price["in"]

    if spec.get("review") and spec["review"].get("rewind_mode") is None:
        review_tokens = 1000
        charge(review_tokens, 30000)
        ev.add("model_response", output={"content": "(synthetic read-only review)"},
               usage={"generated_tokens": review_tokens, "input_tokens": 30000}, mode="review")
    elif spec.get("review"):
        view = read_json(spec["review"]["view_ref"])
        mode = spec["review"]["rewind_mode"]
        if spec["review"].get("opus_record"):
            import prompt_texts
            rendered = prompt_texts.render_review(spec["review"]["prompt_version"], mode=mode, view_text=view["text"],
                                                  snapshot="/workspace\n  (stub snapshot: files and processes)")
            write_json(out / "opus-review-input.json", {
                "version": "opus-review-input.v1", "branch_id": bid, "arm": arm, "view_ref": spec["review"]["view_ref"],
                "prompt_version": spec["review"]["prompt_version"], "rewind_mode": mode,
                "request": {"messages": [{"role": "system", "content": STUB_SYSTEM},
                                         {"role": "user", "content": rendered["text"]}],
                            "tools": [prompt_texts.tool_schema(spec["review"]["prompt_version"], mode)]}})
        calls, args, valid = 0, None, False
        for _ in range(2):
            calls += 1
            tok = rng.randint(900, 4200)
            review_tokens += tok
            charge(tok, 30000)
            args = _choice(rng, view["units"], mode, stall_step) if rng.random() > 0.06 else {"bogus": True}
            ev.add("model_response", output={"content": None, "reasoning_content": "(stub review reasoning)",
                                             "tool_calls": [{"id": f"r{calls}", "name": "review_decision",
                                                             "arguments": json.dumps(args)}]},
                   usage={"generated_tokens": tok, "input_tokens": 30000}, mode="review", call=calls)
            if not validate_decision(args, mode, view["units"]):
                valid = True
                break
        if not valid:
            flags["invalid_call"] = True
            args = ({"rewind": None} if mode == "per_channel" else {"rewind_to": None}) | {
                "drop": [], "keep": [], "change_nothing": True}
        flags["no_change"] = bool(args["change_nothing"])
        rw = args.get("rewind") if mode == "per_channel" else args.get("rewind_to")
        rewound = rw is not None if mode == "coupled" else rw is not None and any(v is not None for v in rw.values())
        undone_or_removed = rewound or bool(args["drop"])
        choice = {"mode": mode, "calls": calls, "valid": valid, "fallback_change_nothing": not valid, "decision": args,
                  "kept": [{"unit": u, "label": view["units"][u].get("label")} for u in args["keep"]],
                  "dropped": list(args["drop"])}
        flags["mismatch"] = undone_or_removed and rng.random() < 0.05
    if spec.get("note"):
        note_tokens = rng.randint(250, 1300)
        charge(note_tokens, 30000)
        ev.add("model_response", output={"content": "(stub note) " + "x" * note_tokens, "reasoning_content": None},
               usage={"generated_tokens": note_tokens, "input_tokens": 30000}, mode="note")
    if undone_or_removed or spec.get("handback", {}).get("recipe"):
        parts = [text["HANDBACK_HEADER"], text["GENERIC_LINE"], text["GRADER_RECORD_LINE"].format(i=1, k=4, n=3)]
        if spec.get("note"):
            parts.append("(stub note)")
        if spec.get("handback", {}).get("recipe"):
            parts.append(spec["handback"]["recipe"])
        if choice:
            parts += [f"{k['unit']}: {k['label']}" for k in choice["kept"]]
        hb = "\n".join(parts)
        handback_tokens = (len(hb) + 3) // 4
        ev.add("handback_delivery", delivered=hb)
    p_solve = prof["p_solve"].get(arm, prof["p_solve"].get("default", 0.2))
    p_re = prof["p_reentry"].get(arm, prof["p_reentry"].get("default", 0.3))
    will_solve = rng.random() < p_solve
    reenter = rng.random() < p_re
    tests = [f"tests/test_hidden.py::test_{i}" for i in range(n_tests)]
    failing = rng.randint(1, max(1, prof["base_failing"]))
    subs, steps, end = [], 0, None
    same = 0
    while end is None:
        steps += 1
        tok = rng.randint(700, 3200)
        if budget.get("new_token_cap") is not None and used_new + tok > budget["new_token_cap"]:
            end = "token_cap"
            break
        if steps > budget["step_limit"]:
            end = "step_guard"
            break
        charge(tok, 20000 + 900 * steps)
        if used_cost > budget["cost_limit_usd"]:
            end = "cost_guard"
            break
        pool = prof["approach_commands"] if reenter and steps % 2 == 0 else prof["fresh_commands"]
        cmd = pool[(steps + spec["repeat"]) % len(pool)]
        submit = steps % rng.randint(3, 6) == 0
        if submit:
            cmd = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        if rng.random() < 0.03:
            flags["failed_command"] = True
        ev.add("model_response", output={"content": None, "reasoning_content": f"(stub reasoning, step {steps})",
                                         "tool_calls": [{"id": f"c{steps}", "name": "bash",
                                                         "arguments": json.dumps({"command": cmd})}]},
               usage={"generated_tokens": tok, "input_tokens": 20000 + 900 * steps})
        ev.add("observation_delivery", delivered=json.dumps({"returncode": 0, "output": f"(stub output of: {cmd})"}))
        if not submit:
            continue
        if will_solve and (len(subs) >= 2 or rng.random() < 0.4):
            failing = 0
        elif rng.random() < 0.5:
            failing = max(1, failing + rng.choice([-1, 0, 0, 1]))
        ids = tests[:failing]
        h = _fail_hash(ids)
        same = same + 1 if subs and subs[-1]["failing_hash"] == h else 1
        subs.append({"step": steps, "failing_count": failing, "failing_hash": h,
                     "share_passed": round(1 - failing / n_tests, 4), "grade_id": f"grade:{bid}:{attempt}:{len(subs)}"})
        ev.add("native_submission_feedback", delivered=text["SUBMISSION_FEEDBACK"].format(n=failing))
        if failing == 0:
            end = "passing_submission"
        elif same >= 3:
            end = "identical_failures"
        elif budget.get("submission_cap") is not None and len(subs) >= budget["submission_cap"]:
            end = "submission_cap"
    shares = [s["share_passed"] for s in subs]
    if not shares:
        final = round(1 - prof["base_failing"] / n_tests, 4)
    elif end == "passing_submission":
        final = shares[-1]
    else:
        final = round(max(0.0, min(1.0, shares[-1] + rng.choice([-1, 0, 0, 1]) / n_tests)), 4)
    r = base | {"end_reason": end, "solved": end == "passing_submission", "submissions": subs,
                "best_share_passed": max(shares) if shares else None, "final_share_passed": final,
                "choice": choice, "review_tokens": review_tokens, "note_tokens": note_tokens,
                "handback_tokens": handback_tokens, "new_tokens": int(used_new), "steps": steps,
                "cost_usd": round(used_cost, 6), "cost_usd_uncached": round(used_uncached, 6),
                "restore_seconds": round(rng.uniform(5, 15), 2), "wall_seconds": round(time.time() - t0, 3),
                "flags": flags}
    write_json(out / "result.json", r)
    return r
