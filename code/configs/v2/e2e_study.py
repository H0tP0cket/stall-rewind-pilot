"""End-to-end check of the track S study machinery on synthetic stalls (3 tasks, 5 stalls + 1 tuning stall), with the
stub runner. SYNTHETIC: no model calls, no Docker, nothing here is evidence about the study.

Builds real run stores (ArtifactStore: events, checkpoints, stall_confirmation records, cost receipts) and Ledger
files in track D's formats (marked fake), then runs every track S command through its CLI, as an operator would:
collect -> screen specs -> driver -> screen/strata (+ extra resamples) -> caps -> tuning -> freeze -> main specs
(refused without --allow-draft) -> driver with a STOP midway -> resume to equal repeats -> full resume with an error
retry -> STOP files (driver then launches nothing) -> healthy checkpoints, harm and
injection specs and runs -> analysis -> Opus record-only dispatch -> re-entry judge inputs, human-check sample.
Writes artifacts/v2/study/{e2e-summary.json, README.md} and the run tree under artifacts/v2/study/run/.

  .venv/bin/python configs/v2/e2e_study.py [--out artifacts/v2/study] [--real-store <natural-run store>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from study_common import (ARMS, HERE, ROOT, MAIN_REPEATS, read_json, stall_id_for, worker_text, write_json)

sys.path.insert(0, str(ROOT / "src"))
from trajectory_retention.accounting import CostLedger  # noqa: E402
from trajectory_retention.contracts import (Checkpoint, EventRecord, NativeState, RuntimeClass, Scope, Usage,  # noqa: E402
                                            Visibility)
from trajectory_retention.store import ArtifactStore  # noqa: E402

PY = sys.executable
TASKS = {"tb2-alpha-v2": 9, "tb2-beta-v2": 6, "tb2-gamma-v2": 4}
# key, task, stall step, approaches, p_solve per arm ("default" for the rest), later_solved
STALLS = [
    ("A1", "tb2-alpha-v2", 18, [(1, 6), (7, 12), (13, 18)],
     {"1_continue": 0.2, "4_forced_rewind": 0.6, "2_ledger_review": 0.6, "3_context_review": 0.4,
      "5_rewind_own_summary": 0.5, "6_rewind_ledger_recipe": 0.65}, False),
    ("A2", "tb2-alpha-v2", 14, [(1, 5), (6, 14)], {"1_continue": 1.0, "default": 0.9}, True),
    ("B1", "tb2-beta-v2", 16, [(1, 8), (9, 16)], {"default": 0.4}, False),
    ("B2", "tb2-beta-v2", 15, [(1, 4), (5, 9), (10, 15)],
     {"1_continue": 0.6, "4_forced_rewind": 0.1, "2_ledger_review": 0.5, "3_context_review": 0.45,
      "5_rewind_own_summary": 0.2, "6_rewind_ledger_recipe": 0.25}, False),
    ("C1", "tb2-gamma-v2", 12, [(1, 12)], {"1_continue": 0.15, "4_forced_rewind": 0.35, "default": 0.3}, False),
    ("T1", "tb2-alpha-v2", 13, [(1, 7), (8, 13)], {"1_continue": 0.3, "default": 0.5}, False),
]
TUNING = ("T1",)
HEALTHY = [("Ha", "tb2-alpha-v2", 14), ("Hb", "tb2-beta-v2", 12), ("Hg", "tb2-gamma-v2", 10)]
P_REENTRY = {"1_continue": 0.7, "2_ledger_review": 0.35, "3_context_review": 0.45, "default": 0.4}
CHECKS: dict[str, dict] = {}
LOG: list[dict] = []


def check(name: str, ok: bool, detail=None) -> bool:
    CHECKS[name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + (f"  {json.dumps(detail, default=str)[:300]}" if detail is not None else ""))
    return ok


def sh(*args: str, env: dict | None = None, expect: int | None = 0) -> tuple[int, str]:
    """Run one track S command the way an operator would; keep its output in the log."""
    p = subprocess.run([PY, "-B", *args], capture_output=True, text=True, env=env or os.environ.copy())
    LOG.append({"cmd": " ".join(Path(a).name if a.endswith(".py") else a for a in args), "rc": p.returncode,
                "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]})
    if expect is not None and p.returncode != expect:
        print(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"command failed (rc {p.returncode}): {' '.join(args)}")
    return p.returncode, p.stdout


def js(out: str):
    return json.loads(out[min(i for i in (out.find("{"), out.find("[")) if i != -1):])


# ---------- synthetic natural runs (real stores) ----------

def commands(task: str, approach: int) -> list[str]:
    t = task.split("-")[1]
    return [f"sed -i 's/limit = {approach}/limit = {approach + 1}/' src/{t}/core_{approach}.py",
            f"python -m pytest tests/test_{t}.py -k case_{approach} -q",
            f"cat src/{t}/core_{approach}.py | head -40",
            f"python scripts/patch_{t}_{approach}.py --apply"]


FRESH = ["grep -rn 'def parse' src/", "python -c 'import sys; print(sys.version)'", "git diff --stat",
         "ls -la src/ tests/", "python -m pytest -q -x", "cat README.md | head -20"]


class Run:
    def __init__(self, st: ArtifactStore, run_id: str, task: str):
        self.st, self.run_id, self.task = st, run_id, task
        self.scope = Scope(task_id=task, stage_id="task", target_id=f"{task}:task", session_id=f"{run_id}/session1",
                           detector_scope_id=f"{run_id}/scope1", requirement_version=f"req:{task}:v1",
                           evaluator_version=f"eval:{task}:v1")
        self.cps = 0

    def ev(self, kind: str, *, response=None, delivered: str | None = None, usage: Usage | None = None, actor="harness",
           visibility=Visibility.WORKER_VISIBLE, **payload) -> EventRecord:
        out = self.st.put_json(response, visibility=Visibility.WORKER_VISIBLE) if response is not None else None
        dl = (self.st.put_bytes(delivered.encode(), media_type="text/plain; charset=utf-8",
                                visibility=Visibility.WORKER_VISIBLE) if delivered is not None else None)
        return self.st.append_event(EventRecord(run_id=self.run_id, kind=kind, actor=actor, scope=self.scope,
                                                visibility=visibility, output_ref=out, delivered_ref=dl, usage=usage,
                                                payload=payload))

    def checkpoint(self, k: int, seq: int) -> str:
        ref = lambda what: self.st.put_json({"synthetic": what, "run": self.run_id, "point": k})
        cid = f"{self.run_id}/C{k}"
        self.st.save_checkpoint(Checkpoint(run_id=self.run_id, scope=self.scope, ordinal=k, event_sequence=seq,
                                           state=NativeState(workspace=ref("W"), history=ref("H"), harness=ref("harness"),
                                                             runtime_class=RuntimeClass.IRRELEVANT),
                                           reasons=("every_point",), checkpoint_id=cid, branchable=True,
                                           validation={"step_index": k, "synthetic": True}))
        return cid


def natural_run(st: ArtifactStore, run_id: str, task: str, n: int, approaches: list[tuple[int, int]], *,
                solved: bool, progress_at: list[int], stall_at: int | None, rng: random.Random) -> dict:
    """n steps; steps past the last approach's end belong to it (a run that solved after its stall)."""
    approaches = approaches[:-1] + [(approaches[-1][0], max(n, approaches[-1][1]))]
    r = Run(st, run_id, task)
    costs = CostLedger(st)
    r.ev("task_disclosure", delivered="You are a helpful assistant that can interact with a computer.", role="system")
    last = r.ev("task_disclosure", delivered=f"Fix the {task} package so its hidden tests pass. "
                + worker_text()[0]["SUBMIT_RULE"], role="user")
    cps = {0: r.checkpoint(0, last.sequence)}
    n_tests = TASKS[task]
    failing = n_tests - 1
    cmds = {}
    for k in range(1, n + 1):
        j = next(i for i, (a, b) in enumerate(approaches, 1) if a <= k <= b)
        a, b = approaches[j - 1]
        submit = k == b or (solved and k == n)
        cmd = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if submit else commands(task, j)[(k - a) % 4]
        cmds[k] = cmd
        gen = rng.randint(600, 2400)
        r.ev("model_response", actor="worker",
             response={"content": None, "reasoning_content": f"Step {k}: approach {j}. I think core_{j} handles the limit.",
                     "tool_calls": [{"id": f"call_{k}", "name": "bash", "arguments": json.dumps({"command": cmd})}]},
             usage=Usage(generated_tokens=gen, input_tokens=4000 + 1500 * k, output_tokens=gen), accepted=True)
        costs.record(run_id=run_id, operation_id=f"call_{k}:attempt1", role="worker",
                     usage=Usage(generated_tokens=gen, cost_usd=round(gen * 4e-7 + (4000 + 1500 * k) * 2e-8, 8)))
        r.ev("tool_call", actor="worker", command=cmd, arguments={"command": cmd})
        out = ("(submitted)" if submit else f"{'total 12' if k % 5 == 1 else 'ok'}: output of step {k} for {cmd[:40]}")
        last = r.ev("observation_delivery", delivered=json.dumps({"returncode": 0, "output": out}), role="tool")
        if submit:
            failing = 0 if (solved and k == n) else max(1, failing - rng.randint(0, 1))
            last = r.ev("native_submission_feedback",
                        delivered=worker_text()[0]["SUBMISSION_FEEDBACK"].format(n=failing))
        cps[k] = r.checkpoint(k, last.sequence)
        anns = []
        if k in progress_at:
            anns.append({"record": "annotation", "label": "P", "label_index": len(anns) + 1, "target_checkpoint_id": cps[k],
                         "score": round(min(0.95, 0.2 + 0.1 * k / n * 5), 3), "annotation_id": f"{run_id}/P{k}"})
        if stall_at == k:
            anns.append({"record": "annotation", "label": "S", "label_index": 1, "target_checkpoint_id": cps[k],
                         "rule_version": "stall-clock/v8.1", "annotation_id": f"{run_id}/S1"})
        if anns:
            r.ev("detector", visibility=Visibility.RESEARCH_PRIVATE, role="output", output=anns)
    r.ev("terminal", visibility=Visibility.RESEARCH_PRIVATE,
         reason="sequence_complete:native_success" if solved else "sequence_complete:native_repeated_failure")
    return {"cps": cps, "cmds": cmds}


# ---------- synthetic Ledger outputs in track D's formats (FAKE labels) ----------

def fake_ledger(n: int, approaches: list[tuple[int, int]], cmds: dict, *, cut: int | None = None) -> dict:
    n = cut or n
    steps, items, aps = [], [], []
    current = max(i for i, (a, _) in enumerate(approaches, 1) if a <= n)
    for k in range(1, n + 1):
        j = next(i for i, (a, b) in enumerate(approaches, 1) if a <= k <= b)
        steps.append({"k": k, "point_before": k - 1, "point_after": k,
                      "flags": {"approach": f"A{j}", "workspace_change": "sed" in cmds[k] or "patch" in cmds[k],
                                "output_current": not (k % 3 == 0 and k < n - 2), "outcome": "fail" if "SUBMIT" in cmds[k] else None,
                                "noise": k % 5 == 1}})
        items.append({"id": f"I{len(items) + 1}", "step": k, "field": "reasoning", "quote": f"core_{j} handles the limit",
                      "type": "fact" if k % 2 else "decision", "survives_rewind": k % 4 != 0})
        if k % 4 == 0:
            items.append({"id": f"I{len(items) + 1}", "step": k, "field": "reasoning", "quote": "the tests pass now",
                          "type": "claim", "claim_status": "contradicted later" if k < n else "unchecked"})
        if k % 6 == 3:
            items.append({"id": f"I{len(items) + 1}", "step": k, "field": "reasoning", "quote": "try a streaming parser",
                          "type": "idea", "idea_status": "never tried"})
    for j, (a, b) in enumerate(approaches, 1):
        if a > n:
            break
        aps.append({"id": f"A{j}", "first_step": a, "last_step": min(b, n),
                    "status": "current" if j == current else "failed",
                    "failure_quote": None if j == current else f"{j} hidden tests still fail"})
    return {"steps": steps, "approaches": aps, "items": items, "synthetic": "FAKE labels (track S e2e)",
            "labeller": {"model": "scripted-fake", "provider": None, "prompt_sha256": "fake", "passes": ["boundary", "full"]}}


def fake_views(led: dict, cmds: dict) -> tuple[dict, dict]:
    ctx_text, led_text, cu, lu = [], ["APPROACHES", "id | steps | status | evidence"], {}, {}
    for ap in led["approaches"]:
        led_text.append(f"{ap['id']} | {ap['first_step']}-{ap['last_step']} | {ap['status']} | {ap['failure_quote'] or ''}")
        lu[ap["id"]] = {"kind": "approach", "steps": [ap["first_step"], ap["last_step"]], "label": ap["status"]}
    for s in led["steps"]:
        k = s["k"]
        block = f"S{k} [step {k}]\n[command] {cmds[k]}\n[output O{k}] output of step {k}"
        ctx_text.append(block)
        tags = [f"[{it['id']} {it['type']}] \"{it['quote']}\"" for it in led["items"] if it["step"] == k]
        led_text.append(block + ("\n" + " ".join(tags) if tags else ""))
        for u in (cu, lu):
            u[f"S{k}"] = {"kind": "step", "steps": [k, k]}
            u[f"O{k}"] = {"kind": "output", "steps": [k, k]}
    for it in led["items"]:
        lu[it["id"]] = {"kind": "item", "steps": [it["step"], it["step"]], "quote": it["quote"],
                        "label": it["type"] + (f" ({it.get('claim_status') or it.get('idea_status')})"
                                               if it.get("claim_status") or it.get("idea_status") else "")}
    return {"text": "\n\n".join(led_text), "units": lu}, {"text": "\n\n".join(ctx_text), "units": cu}


def fake_recipe(led: dict, cap: int | None) -> dict:
    lines = ["Failure evidence: " + next((a["failure_quote"] for a in reversed(led["approaches"]) if a["failure_quote"]),
                                         "(none)")]
    lines += [f"Fact: {it['quote']} (step {it['step']})" for it in led["items"] if it["type"] == "fact" and it.get("survives_rewind")]
    lines += [f"Idea never tried: {it['quote']}" for it in led["items"] if it.get("idea_status") == "never tried"]
    kept, cut_log, used = [], [], 0
    for ln in lines:
        t = (len(ln) + 3) // 4
        if cap is not None and used + t > cap:
            cut_log.append({"cut": ln, "tokens": t})
            continue
        kept.append(ln)
        used += t
    return {"text": "\n".join(kept), "cap_tokens": cap, "cut_log": cut_log, "synthetic": "FAKE (track S e2e)"}


def write_ledger_files(d: Path, led: dict, cmds: dict, boundary: dict | None, cap: int | None) -> None:
    lv, cv = fake_views(led, cmds)
    write_json(d / "ledger.json", led)
    write_json(d / "review_view_ledger.json", lv)
    write_json(d / "review_view_context.json", cv)
    if boundary is not None:
        write_json(d / "boundary.json", boundary)
        write_json(d / "recipe.json", fake_recipe(led, cap))


# ---------- the run ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "v2" / "study"))
    ap.add_argument("--real-store", default=str(Path.home() / "projects/llm-trajectories/artifacts/calibration/"
                                                 "stall-supply-20260927-v1-mbh10-mettlebench/store"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    run = out / "run"
    if run.exists():
        shutil.rmtree(run)
    study, stalls_root, healthy_root = run / "study", run / "stalls", run / "healthy"
    for d in (study, stalls_root, healthy_root):
        d.mkdir(parents=True)
    (study / "SYNTHETIC").write_text("synthetic world from configs/v2/e2e_study.py; not evidence\n")
    t_start = time.time()
    rng = random.Random(20261007)
    S = lambda name: str(HERE / name)

    # 1. synthetic natural runs, stall confirmations and cost receipts in real stores
    stores, world, stall_info = {}, {"version": "stub-world.v1", "note": "SYNTHETIC", "runs": {}, "error_once": [],
                                     "delay_seconds": [0.01, 0.05],
                                     "price": {"in": 1e-7, "cached_in": 2e-8, "out": 4e-7}}, {}
    for task in TASKS:
        stores[task] = ArtifactStore(run / "stores" / task)
    for key, task, n, aps, p, later in STALLS:
        rid = f"v2-e2e:{key.lower()}{rng.randrange(16 ** 8):08x}:natural"
        nr = natural_run(stores[task], rid, task, n + (3 if later else 0), aps, solved=later,
                         progress_at=[2, max(3, aps[0][1])], stall_at=n, rng=rng)
        rec = {"run_id": rid, "task_id": task, "later_solved": later, "reserved": [],
               "primary": {"warning_id": f"{rid}/S1", "checkpoint_id": nr["cps"][n], "step_index": n,
                           "judge_votes": ["stall", "stall", "not_stall"], "judge_receipts": [f"judge/calls/{key}/stall-0.json"]},
               "detector_rule": "stall-clock/v8.1",
               "judge": {"model": "anthropic/claude-haiku-5.5", "provider": "anthropic", "prompt_sha256": "synthetic"}}
        stores[task].put_record("stall_confirmation", rid, rec, run_id=rid)
        cur = aps[-1]
        world["runs"][rid] = {"task_id": task, "n_tests": TASKS[task], "p_solve": p, "p_reentry": P_REENTRY,
                              "base_failing": max(1, TASKS[task] // 2), "stall_step": n,
                              "approach_commands": [nr["cmds"][k] for k in range(cur[0], cur[1] + 1)
                                                    if "SUBMIT" not in nr["cmds"][k]] or FRESH[:1],
                              "fresh_commands": FRESH}
        stall_info[key] = {"run_id": rid, "task": task, "n": n, "aps": aps, "cmds": nr["cmds"],
                           "boundary": {"current_approach_id": f"A{len(aps)}", "current_approach_first_step": cur[0],
                                        "rewind_point": cur[0] - 1, "status": "ok", "synthetic": "FAKE (track S e2e)"}}
    for key, task, n in HEALTHY:
        rid = f"v2-e2e:{key.lower()}{rng.randrange(16 ** 8):08x}:natural"
        nr = natural_run(stores[task], rid, task, n, [(1, n // 2), (n // 2 + 1, n)], solved=True,
                         progress_at=[3, n // 2, n - 2], stall_at=None, rng=rng)
        world["runs"][rid] = {"task_id": task, "n_tests": TASKS[task], "p_solve": {"default": 0.75},
                              "p_reentry": {"default": 0.2}, "base_failing": 1, "stall_step": n,
                              "approach_commands": [c for c in nr["cmds"].values() if "SUBMIT" not in c][:3],
                              "fresh_commands": FRESH}
        stall_info[key] = {"run_id": rid, "task": task, "n": n, "aps": [(1, n // 2), (n // 2 + 1, n)], "cmds": nr["cmds"]}
    world_path = run / "stub-world.json"
    env = os.environ.copy() | {"V2_STUB_WORLD": str(world_path), "PYTHONDONTWRITEBYTECODE": "1"}
    write_json(world_path, world)

    # 2. collect stall dirs from the stores' stall_confirmation records; write track D's files (fake)
    for task in TASKS:
        sh(S("arms_v2.py"), "collect", "--store", str(stores[task].root), "--stalls-root", str(stalls_root))
    healthy_keys = {h[0] for h in HEALTHY}
    sid = {k: stall_id_for(v["task"], v["run_id"], v["n"]) for k, v in stall_info.items() if k not in healthy_keys}
    check("collect: one stall dir per confirmed primary", sorted(p.name for p in stalls_root.iterdir()) == sorted(sid.values()),
          sorted(sid.values()))
    ctx = read_json(stalls_root / sid["A1"] / "stall.json")["context_tokens_at_stall"]
    check("collect: context tokens at the stall read from the store", ctx == 4000 + 1500 * 18, ctx)
    for key, v in stall_info.items():
        if key in sid:
            write_ledger_files(stalls_root / sid[key], fake_ledger(v["n"], v["aps"], v["cmds"]), v["cmds"],
                               v["boundary"], None)
    world["error_once"] = [f"main|{sid['B1']}/4_forced_rewind/9"]
    write_json(world_path, world)

    # 3. screen: arms 1 and 4 x 5 on fresh seeds, guard-only budget
    _, o = sh(S("arms_v2.py"), "screen", "--study", str(study), "--stalls-root", str(stalls_root))
    scr_set = js(o)
    check("screen specs: 6 stalls x 2 arms x 5", scr_set["branches"] == 60, scr_set)
    sp = read_json(study / "specsets" / "screen" / sid["C1"] / "4_forced_rewind__r0.json")
    check("screen arm 4 uses the frozen boundary rewind_point (point 0 for a first-approach stall)",
          sp["rewind"] == {"workspace": 0, "context": 0, "runtime": 0, "outputs": 0} and sp["budget"]["new_token_cap"] is None,
          sp["rewind"])
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "screen.json"), "--out", str(study / "runs" / "screen"),
       "--runner", "stub", "--parallel", "6", "--tag", "screen", env=env)
    tun = sum((["--tuning-stall", sid[k]] for k in TUNING), [])
    _, o = sh(S("screen_v2.py"), "--study", str(study), "--stalls-root", str(stalls_root), *tun)
    scr = read_json(study / "screen" / "screen.json")
    a2 = next(s for s in scr["stalls"] if s["stall_id"] == sid["A2"])
    check("screen: continue 5/5 stall excluded and counted", a2.get("excluded_continue_5of5") and
          scr["counts"]["excluded_continue_5of5"] >= 1, {"A2": a2["primary"]["continue"], "counts": scr["counts"]})
    check("screen: strata rule marked DRAFT and recorded as used", "strata_rule" in scr["draft_rules_used"],
          scr["draft_rules_used"])
    h = scr["undoing_the_current_approach_helps"]["all_complete_screened"]
    check("screen: undoing-helps estimate with intervals (beta and hierarchical)",
          h["solve_rate_difference_beta"]["ci"] is not None and h["solve_rate_difference_hierarchical"]["ci"] is not None, h)
    near = [s["stall_id"] for s in scr["stalls"] if s.get("extra_resamples")]
    if near:
        rc, _ = sh(S("arms_v2.py"), "screen-extra", "--study", str(study), "--stalls-root", str(stalls_root),
                   "--screen-json", str(study / "screen" / "screen.json"), "--round", "1")
        rc, o = sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "screen-extra-1.json"), "--out",
                   str(study / "runs" / "screen-extra-1"), "--runner", "stub", "--tag", "x", env=env, expect=None)
        check("driver refuses a spec set built on DRAFT rules without --allow-draft", rc != 0)
        sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "screen-extra-1.json"), "--out",
           str(study / "runs" / "screen-extra-1"), "--runner", "stub", "--tag", "x", "--allow-draft", env=env)
        sh(S("screen_v2.py"), "--study", str(study), "--stalls-root", str(stalls_root), *tun)
        scr = read_json(study / "screen" / "screen.json")
        n_all = {s["stall_id"]: s["all"]["continue"]["n"] for s in scr["stalls"] if s["stall_id"] in near}
        check("screen: near-cutoff stalls got extra resamples (strata use them, admission does not)",
              all(v == 10 for v in n_all.values()), {"near": near, "continue_n": n_all})
    else:
        check("screen: no stall near a cutoff in this draw (extra-resample path not exercised)", True, scr["counts"])
    seeds = read_json(study / "seed-registry.json")
    check("seeds: unique per branch across spec sets", len(seeds) == len(set(seeds)), len(seeds))

    # 4. caps from the screen (DRAFT rule), tuning on the tuning stall, freeze
    _, o = sh(S("caps.py"), "screen-caps", "--study", str(study))
    caps = js(o)["caps"]
    check("caps: new-token and submission caps calibrated from screen continue branches",
          caps["new_token_cap"] > 0 and caps["submission_cap"] >= 1, caps)
    sh(S("arms_v2.py"), "tuning", "--study", str(study), "--stalls-root", str(stalls_root), "--stall", sid["T1"],
       "--round", "1", "--repeats", "3", "--calibration", str(study / "calibration.json"))
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "tuning-r1.json"), "--out",
       str(study / "runs" / "tuning-r1"), "--runner", "stub", "--tag", "t", "--allow-draft", env=env)
    _, o = sh(S("caps.py"), "tuning", "--study", str(study))
    tun_cal = js(o)
    check("tuning: recipe cap = median arm 5 note length; rewind mode from mismatch rate (never solve rates)",
          tun_cal["recipe_cap_tokens"] > 0 and tun_cal["rewind_mode_main"]["mode"] in ("per_channel", "coupled"), tun_cal)
    for key in sid:
        v = stall_info[key]
        write_json(stalls_root / sid[key] / "recipe.json", fake_recipe(fake_ledger(v["n"], v["aps"], v["cmds"]),
                                                                      tun_cal["recipe_cap_tokens"]))
    _, o = sh(S("caps.py"), "freeze", "--study", str(study))
    frozen = js(o)
    check("freeze: calibration lists the DRAFT rules it rests on", {"cap_rule", "prompt_drafts"} <= set(frozen["draft_rules_used"]),
          frozen)

    # 5. main specs: refused on DRAFT rules, then built for a mechanics test
    rc, _ = sh(S("arms_v2.py"), "main", "--study", str(study), "--stalls-root", str(stalls_root), "--screen-json",
               str(study / "screen" / "screen.json"), "--calibration", str(study / "calibration.json"), expect=None)
    check("main specs refused while strata/cap rules are DRAFT", rc != 0)
    _, o = sh(S("arms_v2.py"), "main", "--study", str(study), "--stalls-root", str(stalls_root), "--screen-json",
              str(study / "screen" / "screen.json"), "--calibration", str(study / "calibration.json"), "--allow-draft")
    main_set = js(o)
    admitted = [s["stall_id"] for s in scr["stalls"] if s.get("admitted") and not s["tuning"]]
    check("main specs: admitted non-tuning stalls x 6 arms x 10 repeats",
          main_set["branches"] == len(admitted) * 6 * MAIN_REPEATS and sid["T1"] not in admitted, main_set)
    m = read_json(study / "specsets" / "main.json")
    blk = m["blocks"][0]
    order = [(j["repeat"], j["arm"]) for j in blk["jobs"][:12]]
    check("main schedule: stall-major, repeat-major inside a stall, arms interleaved",
          order == [(0, arm) for arm in ARMS] + [(1, arm) for arm in ARMS], order[:7])
    s6 = read_json(study / "specsets" / blk["jobs"][5]["spec"])
    s2 = read_json(study / "specsets" / blk["jobs"][1]["spec"])
    check("main specs: arm 2 review (Ledger view, frozen mode, Opus record on repeat 0); arm 6 recipe hand-back; caps set",
          s2["review"]["view_ref"].endswith("review_view_ledger.json") and s2["review"]["opus_record"] and
          s6["handback"]["recipe"] and s6["budget"]["new_token_cap"] == caps["new_token_cap"],
          {"review": s2["review"], "budget": s6["budget"]})

    # 6. driver with a STOP midway, equalize-only resume, full resume, explicit error retry
    out_main = study / "runs" / "main"
    # 2 workers and a STOP when the done count is one past a repeat boundary: at most 2 branches finish after the
    # STOP, so the stall in progress is left with unequal repeats per arm
    cmd = [PY, "-B", S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main),
           "--runner", "stub", "--parallel", "2", "--tag", "main-1", "--allow-draft"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    target = max(8, main_set["branches"] // 3)
    while proc.poll() is None:
        done = len(list(out_main.glob("branches/*/a*/result.json")))
        if done >= target and done % len(ARMS) == 1:
            (out_main / "STOP").write_text("e2e: STOP midway\n")
            break
        time.sleep(0.05)
    proc.wait()
    st1 = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main))[1])
    check("STOP midway: driver stopped launching; the partial block is kept as it is (unequal repeats)",
          (out_main / "STOP").exists() and st1["states"].get("new", 0) > 0 and bool(st1["partial_blocks"])
          and bool(st1["unequal_blocks"]),
          {"states": st1["states"], "partial": st1["partial_blocks"], "unequal": st1["unequal_blocks"],
           "done_per_arm": {u: st1["blocks"][u]["done_per_arm"] for u in st1["partial_blocks"]}})
    (out_main / "STOP").unlink()
    started_blocks = {u for u, b in st1["blocks"].items() if any(v for v in b["done_per_arm"].values())}
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main), "--runner",
       "stub", "--parallel", "4", "--tag", "main-2", "--equalize-only", "--allow-draft", env=env)
    st2 = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main))[1])
    eq = {u: b["done_per_arm"] for u, b in st2["blocks"].items() if u in started_blocks}
    untouched = {u: b["done_per_arm"] for u, b in st2["blocks"].items() if u not in started_blocks}
    check("resume --equalize-only: every started block has equal repeats per arm; nothing new started",
          all(len(set(v.values())) == 1 for v in eq.values()) and all(not any(v.values()) for v in untouched.values()),
          {"started": eq, "untouched": list(untouched)})
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main), "--runner",
       "stub", "--parallel", "4", "--tag", "main-3", "--allow-draft", env=env)
    st3 = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main))[1])
    errored = world["error_once"][0].split("|", 1)[1]
    check("full resume: an infrastructure error stays an error (not retried without --retry-errors)",
          st3["states"].get("error", 0) == (1 if sid["B1"] in admitted else 0), st3["states"])
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main), "--runner",
       "stub", "--parallel", "4", "--tag", "main-4", "--retry-errors", "--allow-draft", env=env)
    st4 = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "main.json"), "--out", str(out_main))[1])
    atts = sorted(p.name for p in (out_main / "branches" / errored.replace("/", "__")).glob("a*")) if sid["B1"] in admitted else []
    check("after --retry-errors: every block has 10 repeats per arm; the failed attempt is kept (a0 error, a1 done)",
          not st4["unequal_blocks"] and st4["states"] == {"done": main_set["branches"]} and
          all(set(b["done_per_arm"].values()) == {MAIN_REPEATS} for b in st4["blocks"].values())
          and (atts == ["a0", "a1"] or sid["B1"] not in admitted), {"states": st4["states"], "attempts": atts})

    # 7. a STOP file in every run dir (the user's stop; there is no spend watchdog, user 2026-10-07)
    stop_dirs = [str(p) for p in sorted((study / "runs").iterdir())] + [str(study / "runs" / "harm"),
                                                                        str(study / "runs" / "injection")]
    for d in stop_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "STOP").write_text("e2e: the user's stop\n")

    # 8. healthy checkpoints, harm and injection specs; the driver launches nothing while STOP stands
    _, o = sh(S("healthy_v2.py"), "select", *sum((["--store", str(stores[t].root)] for t in TASKS), []),
              "--healthy-root", str(healthy_root))
    sel = js(o)
    check("healthy: picked from solved runs, before any stall warning, at most 2 per task",
          sel["picked"] >= 3 and max(sel["per_task"].values()) <= 2, sel)
    for d in sorted(p.parent for p in healthy_root.glob("*/healthy.json")):
        h = read_json(d / "healthy.json")
        info = next(v for v in stall_info.values() if v["run_id"] == h["run_id"])
        led = fake_ledger(info["n"], info["aps"], info["cmds"], cut=h["step_index"])
        lv, _ = fake_views(led, info["cmds"])
        write_json(d / "ledger.json", led)
        write_json(d / "review_view_ledger.json", lv)
    _, o = sh(S("healthy_v2.py"), "donors", "--healthy-root", str(healthy_root), "--stalls-root", str(stalls_root))
    don = js(o)
    ok_conds = {c for v in don.values() for c, s in v.items() if s == "ok"}
    check("injection donors: payloads for all 8 conditions found somewhere; missing ones carry a reason",
          len(ok_conds) == 8, don)
    for name in ("harm", "injection"):
        _, o = sh(S("arms_v2.py"), name, "--study", str(study), "--healthy-root", str(healthy_root), "--calibration",
                  str(study / "calibration.json"), "--allow-draft")
        r = js(o)
        check(f"{name} specs built", r["branches"] > 0, r)
    inj = read_json(study / "specsets" / "injection.json")
    one = read_json(study / "specsets" / inj["blocks"][0]["jobs"][1]["spec"])
    check("injection payload: separate composition fields (receiver, donor cut, splice with renamed tool-call ids)",
          one["injection"]["version"] == "injection-payload.v1" and one["injection"]["receiver"]["checkpoint_id"] ==
          one["source"]["checkpoint_id"], one["injection"])
    _, o = sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "harm.json"), "--out",
              str(study / "runs" / "harm"), "--runner", "stub", "--tag", "harm", "--allow-draft", env=env)
    r = js(o)
    check("while a STOP file stands the driver launches nothing", r["counts"].get("skipped_stop", 0) == r["planned"]
          and not list((study / "runs" / "harm").glob("branches/*/a*/result.json")), r)
    for d in stop_dirs:
        (Path(d) / "STOP").unlink(missing_ok=True)
    harm = read_json(study / "specsets" / "harm.json")
    orphan = harm["blocks"][0]["jobs"][0]["branch_id"]
    odir = study / "runs" / "harm" / "branches" / orphan.replace("/", "__") / "a0"
    odir.mkdir(parents=True)
    (odir / "spec.json").write_text("{}")
    (study / "runs" / "harm" / "claims").mkdir(parents=True, exist_ok=True)
    (study / "runs" / "harm" / "claims" / f"{orphan.replace('/', '__')}__a0").write_text("99999 killed-driver\n")
    for name in ("harm", "injection"):
        sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / f"{name}.json"), "--out",
           str(study / "runs" / name), "--runner", "stub", "--parallel", "6", "--tag", name, "--allow-draft", env=env)
    sth = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "harm.json"), "--out",
                str(study / "runs" / "harm"))[1])
    sh(S("driver_v2.py"), "release", "--out", str(study / "runs" / "harm"), "--branch-id", orphan)
    sh(S("driver_v2.py"), "run", "--specset", str(study / "specsets" / "harm.json"), "--out", str(study / "runs" / "harm"),
       "--runner", "stub", "--tag", "harm-2", "--allow-draft", env=env)
    sth2 = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "harm.json"), "--out",
                 str(study / "runs" / "harm"))[1])
    check("orphan claim from a killed driver: never re-run silently; explicit release, then a fresh attempt",
          sth["states"].get("in_flight") == 1 and sth2["states"] == {"done": harm["branches"]} and
          sorted(p.name for p in odir.parent.iterdir()) == ["a0.released", "a1"],
          {"before": sth["states"], "after": sth2["states"], "attempts": sorted(p.name for p in odir.parent.iterdir())})
    st = js(sh(S("driver_v2.py"), "status", "--specset", str(study / "specsets" / "injection.json"), "--out",
               str(study / "runs" / "injection"))[1])
    check("injection branches ran through the driver (stub)", st["states"] == {"done": inj["branches"]}, st["states"])

    # 9. Opus record-only dispatch and one simulated answer; re-entry inputs, sample and agreement
    _, o = sh(S("opus_review.py"), "pending", "--study", str(study))
    tasks = json.loads(o[o.find("["):])
    check("Opus record-only: one pending input per stall and view (identical repeats deduplicated)",
          len(tasks) == 2 * len(admitted) and {t["arm"] for t in tasks} == {"2_ledger_review", "3_context_review"},
          [t["id"] for t in tasks])
    pend = read_json(study / "opus-review" / "pending.json")
    t0 = pend[0]
    view = read_json(t0["view_ref"])
    undone = [u for u, v in view["units"].items() if v["steps"][0] > 10 and v["kind"] in ("item", "step")][:1]
    args = ({"rewind": {"workspace": 10, "context": 10, "runtime": 10, "outputs": None}} if t0["rewind_mode"] == "per_channel"
            else {"rewind_to": 10}) | {"drop": [], "keep": undone, "change_nothing": False}
    Path(t0["answer"]).write_text(json.dumps({"name": "review_decision", "arguments": args,
                                              "_note": "SIMULATED answer written by e2e_study.py, not an Opus run"}))
    sh(S("opus_review.py"), "record", "--study", str(study))
    rec = read_json(study / "opus-review" / "records" / f"{t0['id']}.json")
    inp_text = Path(t0["input"]).read_text()
    check("Opus record: input holds the exact review request (REVIEW_PROMPT, view, tool); record validated",
          rec["valid"] and worker_text()[0]["REVIEW_PROMPT"] in inp_text and "review_decision" in inp_text
          and inp_text.rstrip().endswith("===== END OF INPUT FILE ====="), {"record": rec["id"], "errors": rec["errors"]})
    _, o = sh(S("reentry_judge.py"), "build", "--study", str(study))
    rb = js(o)
    key = read_json(study / "reentry" / "key.json")
    first_item = sorted(key["items"])[0]
    txt = (study / "reentry" / "inputs" / f"{first_item}.md").read_text()
    consts = worker_text()[0]
    leaks = [s for s in (consts["HANDBACK_HEADER"], consts["GENERIC_LINE"], consts["REVIEW_PROMPT"], *ARMS,
                         "stub note", "stub review", "review_decision") if s in txt]
    blind = not leaks
    check("re-entry inputs: one per finished main branch, blind (no arm, hand-back, review or note turn)",
          rb["items"] == main_set["branches"] and blind and "EARLIER APPROACH" in txt and "CONTINUATION" in txt,
          {"build": rb, "example_item": first_item, "leaks": leaks})
    sim = random.Random(7)
    for item in sorted(key["items"]):
        (study / "reentry" / "inputs" / f"{item}.answer.json").write_text(json.dumps(
            {"reentry": sim.choice(["yes", "no", "no", "unclear"]), "first_reentry_step": None,
             "evidence": "SIMULATED answer written by e2e_study.py, not an Opus run"}))
    sh(S("reentry_judge.py"), "record", "--study", str(study))
    _, o = sh(S("reentry_judge.py"), "sample", "--study", str(study))
    smp = js(o)
    packet_text = "".join(p.read_text() for p in Path(smp["packet"]).iterdir())
    check("re-entry human check: 10% blind sample (no answers or arms in the packet)",
          smp["sampled"] == -(-rb["items"] // 10) and not any(arm in packet_text for arm in ARMS)
          and "SIMULATED" not in packet_text, smp)
    labels = Path(smp["packet"]) / "labels.csv"
    rows = labels.read_text().splitlines()
    labels.write_text("\n".join([rows[0]] + [r.replace(",,,", ",no,,SIMULATED", 1) for r in rows[1:]]) + "\n")
    _, o = sh(S("reentry_judge.py"), "agree", "--study", str(study))
    check("re-entry agreement scored (simulated human labels)", js(o)["labelled"] == smp["sampled"], js(o)["agreement"])

    # 10. analysis, blinded re-estimate, prompt wording lint
    _, o = sh(S("analyze_v2.py"), "--study", str(study))
    an = read_json(study / "analysis" / "analysis.json")
    hd = an["headline"]
    check("analysis: headline arm 2 vs arm 3 on solve and best share with task-clustered intervals and Holm",
          all(hd[o_]["ci"] is not None and "holm" in hd[o_] for o_ in ("solve", "best_share")) and
          hd["solve"]["stalls"] == len(admitted), {o_: {k: hd[o_].get(k) for k in ("estimate", "ci", "p_two_sided")}
                                                   for o_ in ("solve", "best_share")})
    check("analysis: secondaries, splits (2+ approaches, long context, strata, first approach), within-strata vs continue",
          set(an["key_secondaries"]) == {"2_ledger_review vs 1_continue", "6_rewind_ledger_recipe vs 4_forced_rewind",
                                         "6_rewind_ledger_recipe vs 5_rewind_own_summary"}
          and {"approaches", "long_context", "stratum", "first_approach"} <= set(an["splits"]), list(an["splits"]))
    check("analysis: choices, costs with and without caching, re-stalls, Opus records, re-entry, missingness",
          an["choices"] and an["costs"]["main"]["arms"] and an["restalls"] and an["opus_record_only"]
          and an["reentry"], {"choices": list(an["choices"]), "costs": list(an["costs"])})
    _, o = sh(S("analyze_v2.py"), "--study", str(study), "--blinded-reestimate-only")
    br = js(o)
    check("blinded re-estimate (pooled variance only) on the first stalls", "no_interaction" in br["solve"], br["solve"])
    _, o = sh(S("prompt_texts.py"), "lint")
    check("prompt drafts: no worker-visible text says or implies 'stuck'", js(o)["ok"], js(o)["hits"])

    # 11. optional real-store check of the healthy selector (read-only)
    if Path(a.real_store, "journal.sqlite3").exists():
        _, o = sh(S("healthy_v2.py"), "select", "--store", a.real_store, "--healthy-root", str(run / "healthy-real-store"))
        check("healthy selector on a real natural-run store (read-only)", js(o)["picked"] >= 1, js(o))

    shutil.copy(study / "screen" / "screen-report.md", out / "screen-report.md")
    shutil.copy(study / "analysis" / "report.md", out / "analysis-report.md")
    ok = all(c["ok"] for c in CHECKS.values())
    summary = {"version": "track-s-e2e.v1", "synthetic": True, "ok": ok, "at": dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"), "seconds": round(time.time() - t_start, 1), "worker_text_source": worker_text()[1],
        "command": "/Users/sidsri/projects/llm-trajectories/.venv/bin/python configs/v2/e2e_study.py",
        "checks": CHECKS, "counts": {"screen_branches": scr_set["branches"], "main_branches": main_set["branches"],
                                     "admitted_main_stalls": len(admitted), "screen_counts": scr["counts"]},
        "commands": LOG}
    write_json(out / "e2e-summary.json", summary)
    (out / "README.md").write_text(readme(summary))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ({sum(c['ok'] for c in CHECKS.values())}/{len(CHECKS)}) "
          f"in {summary['seconds']} s -> {out}")
    raise SystemExit(0 if ok else 1)


def readme(s: dict) -> str:
    L = ["# Track S end-to-end check (SYNTHETIC)", "",
         "Synthetic world: 3 tasks, 5 stalls plus 1 tuning stall, in real run stores; Ledger files in track D's formats with "
         "fake labels; branches run by the stub runner (no model calls, no Docker). Nothing here is evidence about the study.",
         "", f"Re-create: `{s['command']}` (about {s['seconds']:.0f} s; rebuilds `run/` from scratch).", "",
         f"Result: {'all' if s['ok'] else 'NOT all'} {len(s['checks'])} checks passed. Worker-visible text source: "
         f"{s['worker_text_source']}.", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'pass' if v['ok'] else 'FAIL'} |" for k, v in s["checks"].items()]
    L += ["", "Files: `e2e-summary.json` (every check with its detail and every command run), `screen-report.md`, "
          "`analysis-report.md`, and the run tree under `run/` (stores, stall and healthy dirs, "
          "spec sets, branch results, Opus and re-entry inputs)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
