"""Zero-cost dry run of pilot spec sets on the real branch engine: Docker restore, review turn, hand-back,
injection splice or workspace debris, one real submission and its delivered feedback. Scripted model only.

Each spec is copied with a short step guard (turn calls + 3) and run with run_branch:
  review arms     one valid review_decision: rewind W, context and R to a few steps before the branch point and keep
                  one unit from the undone steps when there is one (healthy points: same rule)
  read-only arms  one plain-text reply
  then            a look at /app, a submission (graded for real), one more step so the next request shows the
                  feedback line the worker receives
Results are engine evidence only, never worker outcomes.

Usage: python configs/v2/pilot_dryrun.py --out DIR SPEC.json [SPEC.json ...]
Writes DIR/<branch>/check.json per spec and prints one line per branch.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "configs" / "v2"))
from trajectory_retention.execution.model import ScriptedModel  # noqa: E402
from trajectory_retention.experiments import v2_text as T  # noqa: E402
from trajectory_retention.experiments.branch_v2 import BranchStore, ReadOnlyArtifactStore, run_branch  # noqa: E402
from trajectory_retention.experiments.submission_feedback import TEXTS  # noqa: E402

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
OLD_LINE = "Hidden tests still failing:"
RECHECK = False  # --recheck: score stored branches again without running them
USAGE = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200, "cost": 0.0}


def step(content, commands):
    return {"content": content, "commands": commands, "usage": dict(USAGE)}


def decision(spec: dict) -> dict:
    view = json.loads(Path(spec["review"]["view_ref"]).read_text())
    units = view["units"]
    last = max(s for u in units.values() for s in (u.get("steps") or [0]))
    point = max(1, last - 3)
    keep = [u for u, v in sorted(units.items(), key=lambda kv: min(kv[1].get("steps") or [0]))
            if v.get("kind") in ("item", "step", "output") and min(v.get("steps") or [0]) > point][:1]
    return {"rewind": {"workspace": point, "context": point, "runtime": point, "outputs": None},
            "drop": [], "keep": keep, "change_nothing": False}


def script(spec: dict) -> tuple[list, int]:
    arm = spec["arm"]
    turn = []
    if arm in ("2_ledger_review", "3_context_review"):
        args = decision(spec)
        turn = [{"content": None, "reasoning": "Dry-run review.", "usage": dict(USAGE),
                 "raw_tool_calls": [{"id": "rv-dry", "type": "function",
                                     "function": {"name": T.REVIEW_TOOL_NAME, "arguments": json.dumps(args)}}]}]
    elif arm in ("5_ledger_read_only", "6_context_read_only"):
        turn = [{"content": "Dry-run notes.", "commands": [], "reasoning": "Dry-run.", "finish_reason": "stop",
                 "usage": dict(USAGE)}]
    cont = [step("Look.", ["ls /app | head -20"]), step("Submit.", [SUBMIT]), step("Look again.", ["pwd"]),
            step("Never reached.", ["echo never"])]
    return turn + cont, len(turn)


def texts(body: dict) -> list[str]:
    out = []
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [p.get("text", "") for p in c if isinstance(p, dict)]
    return out


def check(spec: dict, out: Path) -> dict:
    s = copy.deepcopy(spec)
    steps, turn_calls = script(s)
    s["budget"] = {**s["budget"], "step_limit": turn_calls + 3, "cost_limit_usd": 3.0}
    holder = {}

    def factory(config):  # one model across the review turn and the continuation, as in engine_e2e/m6
        if "m" not in holder:
            settings = dict(config.model_settings)
            settings.setdefault("max_tokens", 8192)
            holder["m"] = ScriptedModel(steps, model_name=config.model, model_settings=settings)
        return holder["m"]

    done = out / "a0" / "result.json"
    res = (json.loads(done.read_text()) if RECHECK and done.exists() else
           run_branch(s, out_dir=out / "a0", model_factory=factory))
    r = {"branch_id": spec["branch_id"], "arm": spec["arm"], "end_reason": res.get("end_reason"),
         "error": res.get("error"), "error_kind": res.get("error_kind")}
    if res.get("store") and res.get("run_id"):
        store = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(spec["source"]["store"]))
        evs = list(store.iter_events(res["run_id"]))
        lc = lambda t: [e.payload for e in evs if e.kind == "lifecycle" and e.payload.get("transition") == t]
        compiled = (lc("branch_v2_compiled") or [{}])[0]
        reqs = [json.loads(store.get_bytes(e.input_ref)) for e in evs if e.kind == "model_request"]
        verdicts = lc("native_submission_verdict")
        fb = [e for e in evs if e.kind == "native_submission_feedback"]
        last = texts(reqs[-1]) if reqs else []
        bound = (spec.get("submission_feedback") or {}).get("rule")
        r.update(requests=len(reqs), submissions=len(verdicts),
                 failing=[v.get("failing_count") for v in verdicts],
                 feedback_events=len(fb),
                 last_request_has_bound_line=bool(bound) and any(TEXTS[bound] in t for t in last),
                 last_request_old_lines=sum(t.count(OLD_LINE) for t in last),
                 record=compiled.get("injection") if spec.get("injection") else None,
                 effective_points=compiled.get("effective_points"),
                 handback=compiled.get("handback"))
        inj = spec.get("injection")
        if inj and inj.get("channel") == "context":
            injected = [m for b in reqs[:1] for m in b.get("messages", [])
                        if any(isinstance(t, str) and "inj-" in t for t in
                               [c.get("id") for c in m.get("tool_calls") or []] + [m.get("tool_call_id")])]
            r["first_request_injected_messages"] = len(injected)
            first = reqs[0].get("messages", []) if reqs else []
            # the receiving prefixes hold no submissions, so every feedback line in the first request was injected
            r["injected_old_lines"] = sum(str(m.get("content")).count(OLD_LINE) for m in first)
            r["injected_bound_lines"] = sum(m.get("role") == "user" and m.get("content") == TEXTS[bound]
                                            for m in first) if bound else 0
        ok = res.get("error") is None and res.get("error_kind") is None and len(reqs) >= turn_calls + 1
        if verdicts and verdicts[-1].get("failing_count"):
            ok &= r["last_request_has_bound_line"]
        if inj:
            rec = r["record"] or {}
            ok &= rec.get("condition") == inj["condition"]
            if inj.get("channel") == "context":
                # injected tool-call ids carry the namespace; tool results and assistant calls are counted, the
                # rewritten feedback line (a user message) is counted separately
                ok &= (rec.get("spliced") is True and r["first_request_injected_messages"] > 0
                       and r["injected_old_lines"] == 0
                       and r["injected_bound_lines"] == rec.get("feedback_rewritten", 0))
            if inj.get("channel") == "workspace":
                ok &= bool(rec.get("debris_paths"))
        r["ok"] = bool(ok)
    else:
        r["ok"] = False
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recheck", action="store_true")
    ap.add_argument("specs", type=Path, nargs="+")
    a = ap.parse_args()
    global RECHECK
    RECHECK = a.recheck
    for p in a.specs:
        spec = json.loads(p.read_text())
        out = a.out / spec["purpose"] / spec["branch_id"].replace("/", "__")
        try:
            r = check(spec, out)
        except Exception as e:
            r = {"branch_id": spec["branch_id"], "ok": False, "exception": f"{type(e).__name__}: {e}",
                 "trace": traceback.format_exc()[-2000:]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "check.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
        print(json.dumps({k: r.get(k) for k in ("branch_id", "ok", "end_reason", "error", "exception",
                                                  "submissions", "failing", "last_request_has_bound_line",
                                                  "last_request_old_lines")}, default=str), flush=True)


if __name__ == "__main__":
    main()
