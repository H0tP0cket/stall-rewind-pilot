"""End-to-end check of the charged-retry rule for unresolved provider attempts (worst-case-charged-retry.v1).

Natural-v2 runs through the driver (configs/v2/natural.py) on terminalbench-llm-inference-batching-scheduler-v1 with
the count-feedback fixture and a scripted model whose failed attempts carry no usage (the shape of StreamLake's 529,
520, 504 and timeout replies: the generation is unresolved). The plan's settings carry max_tokens 262144 as live
plans do, so each charge is the live bound.

  R recover    rule on: 2 unresolved attempts before step 0 and 1 before step 3, then three identical failing
               submissions end the run normally; every unresolved attempt keeps unknown usage in the ledger and
               carries its worst-case charge; the status reports the charges beside the known spend
  G guard      rule on: 4 + 4 + 2 unresolved attempts on steps 0-2; the charges cross $3 and the cost guard ends the
               run, with the known and charged parts logged separately
  X exhausted  rule on: 5 unresolved attempts on one call (the attempt limit): 4 are charged and retried, the last
               is not, and the run stops on the accounting interruption as before
  O rule off   no unresolved_retry in the plan: 1 unresolved attempt stops the run (Q8), nothing charged
Then a branch (arm 1, continue) from run R's point 2 inherits the rule and retries one unresolved attempt itself.

Usage: python configs/v2/engine_e2e/m7_unresolved_retry.py
Writes artifacts/v2/engine/m7-unresolved-retry.json in the main checkout; stores under
~/.cache/trajectory-retention/v2-engine/m7. No provider calls.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import ARTIFACTS, REGISTRY_V1, ROOT, SCRATCH, SUBMIT, TASK, PARTIAL, events, lifecycle, step, write
from trajectory_retention.accounting import CostLedger
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.mining.points import point_checkpoints
from trajectory_retention.provider_policy import UNRESOLVED_RETRY_RULE
from trajectory_retention.store import ArtifactStore

OUT = SCRATCH / "m7"
DRIVER = ROOT / "configs/v2/natural.py"
SETTINGS = ROOT / "configs/v2/settings.json"
_spec = importlib.util.spec_from_file_location("natural", DRIVER)
natural = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(natural)
FIXTURE = "count-feedback-fixture/v1"
RULE = json.loads(SETTINGS.read_text())["unresolved_retry"]
CAP = 262144


def unresolved(entry: dict, n: int) -> dict:
    return dict(entry, fail_before=n)  # failure_usage absent: the failed attempts carry no usage


def make_plan(name: str, script: list[dict], *, rule: bool = True) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    write(d / "script.json", script)
    plan = {"version": "natural-plan.v2", "plan_id": f"m7-{name}", "registry": str(REGISTRY_V1), "tasks": [TASK],
            "repeats": 1, "model": "scripted/fixture", "settings": {"max_tokens": CAP}, "store": str(d / "run"),
            "concurrency": 1, "detector": {"rule": "stall-clock/v8.1"},
            "budget": {"cost_limit_usd": 3.0, "step_limit": 500}, "purpose": "stall_supply",
            "tool_timeout_seconds": 15, "fixture": {"submission_protocol": FIXTURE, "scripts": {"*": str(d / "script.json")}},
            **({"unresolved_retry": RULE} if rule else {})}
    write(d / "plan.json", plan)
    return d / "plan.json"



def drive(plan) -> dict:
    proc = subprocess.run([sys.executable, str(DRIVER), "--plan", str(plan)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"driver failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def run_of(name: str) -> tuple[ArtifactStore, str, dict]:
    plan = natural.load_plan(OUT / name / "plan.json")
    job = natural.jobs(plan)[0]
    status = json.loads((OUT / name / "run" / "status" / f"{job['key']}.json").read_text())
    return ArtifactStore(OUT / name / "run" / "store"), job["run_id"], status


def attempts(evs) -> list[dict]:
    return [e.payload | {"_call": e.call_id, "_attempt": e.attempt} for e in evs
            if e.kind == "model_attempt" and e.payload.get("phase") == "finished"]


def expected_charge(store, evs, a: dict) -> float:
    started = next(e for e in evs if e.kind == "model_attempt" and e.call_id == a["_call"]
                   and e.attempt == a["_attempt"] and e.payload.get("phase") == "started")
    body = store.get_bytes(started.input_ref)
    cap = json.loads(body)["max_tokens"]
    prices = RULE["worst_case_usd_per_million_tokens"]
    return len(body) * prices["prompt"] / 1e6 + cap * prices["completion"] / 1e6


def summarize(name: str) -> dict:
    store, rid, status = run_of(name)
    evs = events(store, rid)
    att = attempts(evs)
    failed = [a for a in att if a["status"] != "ok"]
    charged = [a for a in failed if a.get("worst_case_charge")]
    ledger = [r for r in CostLedger(store).records(rid) if r.kind == "model"]
    terminal = next(e.payload for e in reversed(evs) if e.kind == "terminal")
    bound = (lifecycle(evs, "worker_policy_bound") or [None])[0]
    guard = lifecycle(evs, "run_guard_boundary")
    charges_exact = all(abs(a["worst_case_charge"]["usd"] - expected_charge(store, evs, a)) < 1e-12 for a in charged)
    return {"store": store, "rid": rid, "status": status, "evs": evs, "failed": failed, "charged": charged,
            "ledger": ledger, "terminal": terminal, "bound": bound.payload if bound else {}, "guard": guard,
            "charges_exact": charges_exact,
            "row": {"termination": terminal["reason"], "status": status["status"], "steps": status["steps"],
                    "failed_attempts": len(failed), "charged_attempts": len(charged),
                    "charges_usd": [round(a["worst_case_charge"]["usd"], 6) for a in charged],
                    "spend_usd_known": status["spend_usd_known"],
                    "spend_usd_worst_case_charged": status.get("spend_usd_worst_case_charged"),
                    "model_attempts_cost_unknown": status["model_attempts_cost_unknown"],
                    "retry_reasons": [(a.get("retry_decision") or {}).get("reason") for a in failed]}}


def main() -> None:
    t0 = time.monotonic()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    look = step("Look around.", ["ls /app/task_file"], cost=0.01)
    partial = step("Run the baseline packer.", [PARTIAL], cost=0.01)
    submit = step("Submit.", [SUBMIT], cost=0.01)
    never = step("Never reached.", ["echo never"], cost=0.01)
    plans = {
        "R": make_plan("R", [unresolved(look, 2), partial, submit, unresolved(submit, 1), submit, never]),
        "G": make_plan("G", [unresolved(look, 4), unresolved(partial, 4), unresolved(look, 2), never, never]),
        "X": make_plan("X", [unresolved(look, 5), never]),
        "O": make_plan("O", [unresolved(look, 1), never], rule=False),
    }
    for p in plans.values():
        drive(p)
    R, G, X, O = (summarize(k) for k in "RGXO")
    checks = {}
    # R: the run recovers and ends normally; usage stays unknown, charges are exact and reported
    checks["R_run_ends_normally"] = R["terminal"]["reason"] == "sequence_complete:native_repeated_failure"
    checks["R_three_charged_attempts"] = len(R["charged"]) == 3 == len(R["failed"])
    checks["R_charged_usage_unknown_in_ledger"] = sum(r.usage.cost_usd is None for r in R["ledger"]) == 3 \
        and R["status"]["model_attempts_cost_unknown"] == 3
    checks["R_charges_exact_bound"] = R["charges_exact"]
    checks["R_status_reports_charges"] = abs(R["status"]["spend_usd_worst_case_charged"]
                                             - sum(a["worst_case_charge"]["usd"] for a in R["charged"])) < 1e-9 \
        and R["status"]["model_attempts_charged_worst_case"] == 3
    checks["R_known_spend_unchanged"] = abs(R["status"]["spend_usd_known"] - 0.01 * R["status"]["steps"]) < 1e-9
    checks["R_rule_bound_and_logged"] = R["bound"].get("unresolved_retry", {}).get("rule") == UNRESOLVED_RETRY_RULE
    checks["R_retry_reason"] = all((a.get("retry_decision") or {}).get("reason") == "unresolved_generation_charged_worst_case"
                                   and a["retry_decision"]["retry"] for a in R["charged"])
    checks["R_no_accounting_incident"] = not any(e.kind == "incident" and e.payload.get("what")
                                                 == "worker_accounting_interruption" for e in R["evs"])
    # G: charges cross the $3 guard; known and charged parts logged separately
    g = G["guard"][0].payload if G["guard"] else {}
    checks["G_cost_guard_ends_run"] = G["status"]["termination_reason"] in ("WorkerCostLimit", "cost_limit") \
        or "cost" in str(G["terminal"]["reason"]).lower()
    checks["G_guard_counts_charges"] = bool(g) and g["guard"] == "cost_limit_usd" and g["observed"] >= 3.0 \
        and abs(g["observed"] - g["observed_known"] - g["observed_worst_case_charged"]) < 1e-9 \
        and g["observed_known"] < 3.0 and g["charged_unknown_attempts"] == 10
    checks["G_stops_after_third_step"] = G["status"]["steps"] == 3
    # X: the attempt limit still stops the run on the accounting interruption
    checks["X_accounting_interruption_at_attempt_limit"] = X["status"]["status"] == "accounting_interrupted" \
        and len(X["failed"]) == 5 and len(X["charged"]) == 4 \
        and (X["failed"][-1].get("retry_decision") or {}).get("reason") == "attempt_limit_unresolved"
    # O: rule off keeps the Q8 stop exactly
    checks["O_rule_off_q8_stop"] = O["status"]["status"] == "accounting_interrupted" and not O["charged"] \
        and len(O["failed"]) == 1 and (O["failed"][0].get("retry_decision") or {}).get("reason") in (None, "unresolved_generation_Q8_stop") \
        and "unresolved_retry" not in O["bound"]

    # a branch (arm 1) from run R's point 2 inherits the rule and retries an unresolved attempt itself
    branch_row = branch_check(R, checks)
    report = {"version": "m7-unresolved-retry.v1", "rule": RULE, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "runs": {k: v["row"] for k, v in zip("RGXO", (R, G, X, O))}, "branch": branch_row, "checks": checks,
              "all_pass": all(checks.values()), "wall_seconds": round(time.monotonic() - t0, 1)}
    write(ARTIFACTS / "m7-unresolved-retry.json", report)
    print(json.dumps({"all_pass": report["all_pass"], "failed": [k for k, v in checks.items() if not v],
                      "runs": report["runs"], "branch": branch_row}, indent=1, default=str))


def branch_check(R: dict, checks: dict) -> dict:
    sys.path.insert(0, str(ROOT / "configs" / "v2"))
    import arms_v2  # noqa: E402
    from trajectory_retention.experiments.branch_v2 import BranchStore, ReadOnlyArtifactStore, run_branch
    store, rid = R["store"], R["rid"]
    points = point_checkpoints(store, rid)
    root = OUT / "R" / "run" / "store"
    stall = {"stall_id": "m7-R-p2", "store": str(root), "run_id": rid, "checkpoint_id": points[2],
             "dir": str(OUT / "stall"), "has_boundary": False, "step_index": 2, "has_recipe": False}
    spec = arms_v2.arm_spec(stall, "1_continue", 0, purpose="tuning", prefix="m7-engine", caps=None)
    model = ScriptedModel([unresolved(step("Submit.", [SUBMIT], cost=0.01), 1),
                           step("Submit.", [SUBMIT], cost=0.01), step("Submit.", [SUBMIT], cost=0.01),
                           step("Never reached.", ["echo never"])],
                          model_name="scripted/fixture", model_settings={"max_tokens": CAP})
    res = run_branch(spec, out_dir=OUT / "branch" / "a0", model_factory=lambda config: model)
    bstore = BranchStore(Path(res["store"]), ReadOnlyArtifactStore(root))
    evs = events(bstore, res["run_id"])
    started = (lifecycle(evs, "run_started") or [None])[0]
    meta = ((started.payload.get("config") or {}).get("metadata") or {}) if started else {}
    failed = [a for a in attempts(evs) if a["status"] != "ok"]
    checks["branch_inherits_rule"] = (meta.get("unresolved_retry") or {}).get("rule") == UNRESOLVED_RETRY_RULE
    checks["branch_retries_charged_attempt"] = len(failed) == 1 and bool(failed[0].get("worst_case_charge")) \
        and res.get("end_reason") not in (None, "technical_failure")
    return {"end_reason": res.get("end_reason"), "steps": res.get("steps"), "cost_usd": res.get("cost_usd"),
            "failed_attempts": len(failed), "charged": [round(a["worst_case_charge"]["usd"], 6) for a in failed
                                                        if a.get("worst_case_charge")], "flags": res.get("flags")}


if __name__ == "__main__":
    main()
