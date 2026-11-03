"""One-call provider probe for the arm 5 note turn: a request with no tools key after a context full of tool calls.

The engine sends the note turn with the worker's own guarded model class and pinned settings but without any tools
(model.build_request omits the key). Some providers refuse a conversation that contains tool calls when no tools are
declared; this probe checks the pinned one accepts it and answers in plain text. The request is the milestone-1 run A
stall context (scripted, so the content is synthetic) plus the arm 5 note prompt rendered by configs/v2/prompt_texts.py,
exactly as runner._v2_turn builds it; only max_tokens is lowered to bound the cost.

Without --live it only builds and describes the request (no call). With --live it sends one call (two if the first
answer carries a tool call, as in the engine) and writes artifacts/v2/engine/note-turn-probe-<utc>.json (no
credentials). Expected cost: under $0.01 (about 4k input tokens at $0.141/M plus at most 8k output at $0.564/M).

Usage: python configs/v2/engine_e2e/note_turn_probe.py [--live] [--max-tokens 8192]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys

from common import ARTIFACTS, ROOT, SCRATCH
from minisweagent.exceptions import FormatError
from trajectory_retention.experiments.branch_v2 import ReadOnlyArtifactStore, _note_check, _note_parser, \
    live_model_factory
from trajectory_retention.mining.points import checkpoint_step_index

sys.path.insert(0, str(ROOT / "configs" / "v2"))
import prompt_texts  # noqa: E402
from study_common import PROMPT_VERSIONS  # noqa: E402

SETTINGS = ROOT / "configs/v2/settings.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="send the call (spends a few tenths of a cent)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()
    pinned = json.loads(SETTINGS.read_text())
    root = SCRATCH / "m1" / "A" / "run" / "store"
    store = ReadOnlyArtifactStore(root)
    with store._connect() as db:
        run_id = next(r[0] for r in db.execute("SELECT DISTINCT run_id FROM events") if r[0].endswith(":natural"))
    stall = store.get_checkpoint(store.get_record("stall_confirmation", run_id)["primary"]["checkpoint_id"])
    s = checkpoint_step_index(store, stall)
    history = store.get_json(stall.state.history)
    note = prompt_texts.render_note(PROMPT_VERSIONS["5_rewind_own_summary"], n_undone_turns=s, rewind_point=0)
    messages = copy.deepcopy(history["messages"]) + [{"role": "user", "content": note["text"]}]
    settings = dict(pinned["settings"], max_tokens=a.max_tokens)
    from trajectory_retention.contracts import WorkerRequestPolicy
    from trajectory_retention.execution.received_response_guard_v1 import ReceivedResponseGuardV1
    policy = WorkerRequestPolicy()
    live_model_factory()  # the class refuses to exist without the key: load it into the environment (never printed)
    model = ReceivedResponseGuardV1(model_name=pinned["model"], model_settings=settings,
                                    max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
                                    timeout_seconds=policy.timeout_seconds,
                                    usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)
    model.config["tools"], model.action_parser = [], _note_parser
    body = json.loads(model.prepare_request(messages).body)
    described = {"model": body["model"], "has_tools_key": "tools" in body, "messages": len(body["messages"]),
                 "tool_call_messages": sum(1 for m in body["messages"] if m.get("tool_calls")),
                 "chars": len(json.dumps(body["messages"])), "note_prompt": note["version"],
                 "provider": body.get("provider"), "max_tokens": body.get("max_tokens")}
    print(json.dumps(described, indent=1))
    if not a.live:
        print("dry run: no call sent (add --live to send one)")
        return
    calls = []
    for number in (1, 2):
        try:
            message = model.query(copy.deepcopy(messages))
        except FormatError as e:
            calls.append({"call": number, "valid": False, "error": e.messages[0]["content"]})
            messages += e.messages
            continue
        extra, usage = message["extra"], message["extra"]["usage"]
        problem = _note_check(message)
        calls.append({"call": number, "valid": problem is None, "error": problem,
                      "provider": extra["response"].get("provider"),
                      "finish_reason": extra["normalized"]["finish_reason"],
                      "content_chars": len(message.get("content") or ""), "tool_calls": bool(message.get("tool_calls")),
                      "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                      "cost_usd": usage.cost_usd})
        if problem is None:
            break
    out = {"what": "note turn without a tools key, after a context with tool calls", "request": described,
           "calls": calls, "accepted": any(c["valid"] for c in calls),
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    path = ARTIFACTS / f"note-turn-probe-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))
    print(path)


if __name__ == "__main__":
    main()
