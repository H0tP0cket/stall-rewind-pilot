"""Staged band screen of admitted terminalbench-v2 tasks: writes natural-run plans and applies the contract's rule.

Never launches anything. Plans use the natural-run plan format of the v2 cross-track interface (one store per plan, so
repeat numbering never collides); ``configs/v2/natural.py --plan <plan>`` runs them after the user's approval.

Rule (docs/research/experiment.md, Tasks and admission; task-admission-criteria.md, Band and staged screen):
  stage 1  4 runs. Drop if solved 4/4, or solved 0/4 with failed runs under 40% of hidden tests. Otherwise stage 2.
  stage 2  6 runs in all. Keep if solved 1-3 of 6. Drop if solved 4 or more. If solved 0/6 with high completion on
           failed runs, go to stage 3; otherwise drop.
  stage 3  12 runs in all. Keep if any run solved.
  band     10-50% solve rate and a median of at least 60% completion on failed runs, over all runs (reported for every
           task, recomputed whenever results arrive; kept tasks continue as stall supply).
Declared readings (flagged for the user): "under 40%" and "high completion" use the median completion of the failed
runs, with high = at least 60% (the band's own threshold); a run's completion is its best submission's share of hidden
tests passed, or its final grade's share when it never submitted; a run solves when a submission passed. Runs that end
in a technical failure (or are interrupted) are recorded, never counted as failures, and are replaced by new runs.

Usage:
  python configs/v2/screen_tasks.py init --registry benchmark_tasks/admission/terminalbench-v2/registry.json \\
      --screen-dir artifacts/v2/screen/<name> --store-root ~/.cache/trajectory-retention/v2-screen/<name> \\
      [--settings configs/v2/settings.json] [--concurrency 4]
  python configs/v2/screen_tasks.py advance --screen-dir artifacts/v2/screen/<name> \\
      [--results outcomes.jsonl ...] [--stores <store> ...] [--supply-repeats N]
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

VERSION = "v2-screen.v1"
PLAN_VERSION = "natural-plan.v2"  # configs/v2/natural.py
MODEL = "deepseek/deepseek-v4.1-flash"
BASE_SETTINGS = {"temperature": 1, "reasoning": {"effort": "max"}, "max_tokens": 262144}
DETECTOR = {"rule": "stall-clock/v8.1"}
BUDGET = {"cost_limit_usd": 3.0, "step_limit": 1_000_000}  # user 2026-10-12: the guard only stops runaway spending, never active work; no step limit (unreachable)
TARGET = {1: 4, 2: 6, 3: 12}
STAGE1_LOW = 0.40
HIGH = 0.60
BAND = (0.10, 0.50)
TECHNICAL = {"technical_failure", "interrupted", "accounting_interrupted", "error"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_settings(path: Path | None, allow_unpinned: bool) -> dict:
    if path is not None and path.is_file():
        data = json.loads(path.read_text())
        settings = data.get("settings", data)
        if not settings.get("provider"):
            raise SystemExit(f"{path} has no provider pin")
        return {**BASE_SETTINGS, **settings}
    if not allow_unpinned:
        raise SystemExit("configs/v2/settings.json (the provider pin) is missing; pass --allow-unpinned only for dry runs")
    return {**BASE_SETTINGS, "provider": None}


def load_context_limit(path: Path | None) -> dict | None:
    """The confirmed context window from the settings file ({rule, window_tokens}), carried into every plan."""
    if path is not None and path.is_file():
        limit = json.loads(path.read_text()).get("context_limit") or {}
        if limit.get("window_tokens") is not None:
            return {"rule": limit["rule"], "window_tokens": limit["window_tokens"], "source": str(path)}
    return None


def load_unresolved_retry(path: Path | None) -> dict | None:
    """The charged-retry rule for unresolved provider attempts from the settings file, carried into every plan."""
    if path is not None and path.is_file():
        return json.loads(path.read_text()).get("unresolved_retry")
    return None


def load_model(path: Path | None) -> str:
    """The pinned worker model from the settings file (its "model" field), else the default."""
    if path is not None and path.is_file():
        return json.loads(path.read_text()).get("model", MODEL)
    return MODEL


def completion(run: dict) -> float | None:
    shares = [s.get("share_passed") for s in run.get("submissions") or [] if s.get("share_passed") is not None]
    return max(shares) if shares else run.get("final_share_passed")


def stats(runs: list[dict]) -> dict:
    valid = [r for r in runs if r.get("status") not in TECHNICAL]
    solved = sum(bool(r.get("solved")) for r in valid)
    failed = [completion(r) for r in valid if not r.get("solved")]
    known = [c for c in failed if c is not None]
    return {"valid_runs": len(valid), "solved": solved, "technical_runs": len(runs) - len(valid),
            "solve_rate": solved / len(valid) if valid else None,
            "failed_completion_median": statistics.median(known) if known else None,
            "failed_completion_missing": len(failed) - len(known)}


def decide(stage: int, s: dict) -> tuple[str, str]:
    """(action, reason) for a task whose stage target of valid runs is met: keep / drop / stage N."""
    n, k, med = s["valid_runs"], s["solved"], s["failed_completion_median"]
    shown = "unknown" if med is None else f"{med:.2f}"
    if stage == 1:
        if k == n:
            return "drop", f"solved {k}/{n}"
        if k == 0 and (med is None or med < STAGE1_LOW):
            return "drop", f"solved 0/{n}; median failed-run completion {shown} < {STAGE1_LOW}"
        return "stage 2", f"solved {k}/{n}"
    if stage == 2:
        if 1 <= k <= 3:
            return "keep", f"solved {k}/{n}"
        if k >= 4:
            return "drop", f"solved {k}/{n}"
        if med is not None and med >= HIGH:
            return "stage 3", f"solved 0/{n}; median failed-run completion {shown} >= {HIGH}"
        return "drop", f"solved 0/{n}; median failed-run completion {shown} < {HIGH}"
    return ("keep", f"solved {k}/{n}") if k >= 1 else ("drop", f"solved 0/{n}")


def in_band(s: dict) -> bool:
    return (s["solve_rate"] is not None and BAND[0] <= s["solve_rate"] <= BAND[1]
            and s["failed_completion_median"] is not None and s["failed_completion_median"] >= HIGH)


def outcomes_from_store(store_path: Path) -> list[dict]:
    """One outcome per natural run in a run store: solved, submission shares and final share, from recorded grades."""
    from trajectory_retention.store import ArtifactStore
    store = ArtifactStore(store_path)
    out = []
    for row in store.list_records("run_result"):
        result = store.get_record("run_result", row["record_id"])
        run_id = result["run_id"]
        events = list(store.iter_events(run_id))
        start = next((e.payload for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "run_started"), {})
        if start.get("treatment") or start.get("resume_from"):
            continue  # branches and resumes are not natural screening runs
        checkpoints = {c.checkpoint_id: c for c in store.list_checkpoints(run_id)}

        def share(checkpoint_id):
            c = checkpoints.get(checkpoint_id)
            return store.get_json(c.evaluation_ref).get("score") if c is not None and c.evaluation_ref else None
        verdicts = [e.payload for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "native_submission_verdict"]
        final = None
        if result.get("final_evaluation_ref"):
            from trajectory_retention.contracts import ArtifactRef, from_jsonable
            final = store.get_json(from_jsonable(ArtifactRef, result["final_evaluation_ref"])).get("score")
        out.append({"task_id": (start.get("task") or {}).get("task_id"), "run_id": run_id, "store": str(store_path),
                    "status": result["status"], "end_reason": result.get("termination_reason"),
                    "solved": any(v.get("status") == "pass" for v in verdicts),
                    "submissions": [{"ordinal": v.get("ordinal"), "status": v.get("status"),
                                     "share_passed": share(v.get("checkpoint_id"))} for v in verdicts],
                    "final_share_passed": final})
    return out


def read_results(results: list[Path], stores: list[Path]) -> list[dict]:
    rows = []
    for path in results:
        text = path.read_text()
        rows += json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    for store in stores:
        rows += outcomes_from_store(store)
    seen, unique = set(), []
    for row in rows:
        key = (row.get("store"), row["run_id"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def write_plan(state: dict, screen_dir: Path, stage: int, tasks: list[str], repeats: int, purpose: str) -> str:
    n = sum(1 for p in state["plans"] if p["stage"] == stage and p["purpose"] == purpose) + 1
    name = f"{purpose}-stage{stage}-{n}-r{repeats}"
    plan = {"version": PLAN_VERSION, "registry": state["registry"], "tasks": sorted(tasks), "repeats": repeats,
            "model": state.get("model", MODEL), "settings": state["settings"],
            **({"context_limit": state["context_limit"]} if state.get("context_limit") else {}),
            **({"submission_stop": state["submission_stop"]} if state.get("submission_stop") else {}),
            **({"submission_feedback": state["submission_feedback"]} if state.get("submission_feedback") else {}),
            **({"unresolved_retry": state["unresolved_retry"]} if state.get("unresolved_retry") else {}),
            "store": str(Path(state["store_root"]) / name),
            "concurrency": state["concurrency"], "detector": DETECTOR, "budget": BUDGET, "purpose": purpose,
            "screen": {"version": VERSION, "stage": stage, "screen_dir": str(screen_dir), "written": now()}}
    path = screen_dir / "plans" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n")
    state["plans"].append({"name": name, "path": str(path.relative_to(screen_dir)), "stage": stage, "purpose": purpose,
                           "tasks": {t: repeats for t in sorted(tasks)}, "store": plan["store"]})
    return str(path)


def init(args) -> dict:
    registry = json.loads(Path(args.registry).read_text())
    tasks = sorted(e["task_id"] for e in registry["tasks"] if e.get("admission") == "ADMITTED")
    if not tasks:
        raise SystemExit("registry has no admitted tasks")
    screen_dir = Path(args.screen_dir)
    if (screen_dir / "screen.json").exists():
        raise SystemExit(f"{screen_dir} already holds a screen; use advance")
    state = {"version": VERSION, "rule": __doc__, "created": now(), "registry": str(Path(args.registry).resolve()),
             "settings": load_settings(Path(args.settings) if args.settings else None, args.allow_unpinned),
             "model": load_model(Path(args.settings) if args.settings else None),
             "context_limit": load_context_limit(Path(args.settings) if args.settings else None),
             "submission_stop": (json.loads(Path(args.settings).read_text()).get("submission_stop")
                                 if args.settings and Path(args.settings).is_file() else None),
             "submission_feedback": (json.loads(Path(args.settings).read_text()).get("submission_feedback")
                                     if args.settings and Path(args.settings).is_file() else None),
             "unresolved_retry": load_unresolved_retry(Path(args.settings) if args.settings else None),
             "store_root": str(Path(args.store_root).expanduser()), "concurrency": args.concurrency,
             "tasks": {t: {"stage": 1, "status": "screening", "history": []} for t in tasks}, "plans": []}
    write_plan(state, screen_dir, 1, tasks, TARGET[1], "screen")
    (screen_dir / "screen.json").write_text(json.dumps(state, indent=2) + "\n")
    return {"tasks": len(tasks), "plans": [p["path"] for p in state["plans"]]}


def advance(args) -> dict:
    screen_dir = Path(args.screen_dir)
    state = json.loads((screen_dir / "screen.json").read_text())
    rows = read_results([Path(p) for p in args.results or []], [Path(p).expanduser() for p in args.stores or []])
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    unknown = sorted(set(by_task) - set(state["tasks"]))
    new_runs: dict[tuple[int, int], list[str]] = {}
    report = {}
    for task, info in sorted(state["tasks"].items()):
        runs = by_task.get(task, [])
        s = stats(runs)
        planned = sum(p["tasks"].get(task, 0) for p in state["plans"] if p["purpose"] == "screen")
        report[task] = {**s, "stage": info["stage"], "status": info["status"], "planned_screen_runs": planned,
                        "in_band": in_band(s)}
        if info["status"] != "screening":
            continue
        if len(runs) < planned:
            report[task]["waiting_for"] = planned - len(runs)
            continue
        target = TARGET[info["stage"]]
        if s["valid_runs"] < target:
            new_runs.setdefault((info["stage"], target - s["valid_runs"]), []).append(task)
            report[task]["replacing"] = target - s["valid_runs"]
            continue
        action, reason = decide(info["stage"], s)
        info["history"].append({"stage": info["stage"], "action": action, "reason": reason, "stats": s, "at": now()})
        if action.startswith("stage"):
            nxt = int(action.split()[1])
            info["stage"] = nxt
            new_runs.setdefault((nxt, TARGET[nxt] - s["valid_runs"]), []).append(task)
        else:
            info["status"] = "kept" if action == "keep" else "dropped"
        report[task].update(action=action, reason=reason, stage=info["stage"], status=info["status"])
    written = [write_plan(state, screen_dir, stage, tasks, repeats, "screen")
               for (stage, repeats), tasks in sorted(new_runs.items())]
    kept = sorted(t for t, i in state["tasks"].items() if i["status"] == "kept")
    if args.supply_repeats and kept:
        written.append(write_plan(state, screen_dir, 0, kept, args.supply_repeats, "stall_supply"))
    band = {"version": VERSION, "written": now(), "rule": __doc__, "results_read": len(rows),
            "unknown_task_results": unknown, "kept_stall_supply": kept,
            "dropped": {t: i["history"][-1]["reason"] for t, i in sorted(state["tasks"].items()) if i["status"] == "dropped"},
            "screening": sorted(t for t, i in state["tasks"].items() if i["status"] == "screening"),
            "final_band": sorted(t for t, r in report.items() if r["in_band"]),
            "tasks": report, "new_plans": written}
    (screen_dir / "band.json").write_text(json.dumps(band, indent=2) + "\n")
    (screen_dir / "screen.json").write_text(json.dumps(state, indent=2) + "\n")
    return {k: band[k] for k in ("results_read", "kept_stall_supply", "dropped", "screening", "final_band", "new_plans")}


# Synthetic screen: one scenario per path through the rule. Each stage lists the runs that stage's plan(s) return, in
# order: (status, solved, completion). The expected outcome is the task's final status and its last decision's action.
SYNTHETIC = {
    "syn-drop-all-solved": ({1: [("completed", True, 1.0)] * 4}, ("dropped", "drop")),
    "syn-drop-low-zero": ({1: [("completed", False, c) for c in (0.1, 0.2, 0.3, 0.5)]}, ("dropped", "drop")),
    "syn-keep-stage2": ({1: [("completed", True, 1.0)] + [("completed", False, 0.7)] * 3,
                         2: [("completed", True, 1.0), ("completed", False, 0.65)]}, ("kept", "keep")),
    "syn-drop-stage2-easy": ({1: [("completed", True, 1.0)] * 2 + [("completed", False, 0.8)] * 2,
                              2: [("completed", True, 1.0)] * 2}, ("dropped", "drop")),
    "syn-drop-stage2-low": ({1: [("completed", False, c) for c in (0.4, 0.45, 0.5, 0.55)],
                             2: [("completed", False, 0.3)] * 2}, ("dropped", "drop")),
    "syn-keep-stage3": ({1: [("completed", False, 0.7)] * 4, 2: [("completed", False, 0.75)] * 2,
                         3: [("completed", True, 1.0)] + [("completed", False, 0.7)] * 5}, ("kept", "keep")),
    "syn-drop-stage3": ({1: [("completed", False, 0.7)] * 4, 2: [("completed", False, 0.75)] * 2,
                         3: [("completed", False, 0.7)] * 6}, ("dropped", "drop")),
    "syn-replace-technical": ({1: [("technical_failure", False, None), ("completed", True, 1.0), ("completed", False, 0.8),
                                   ("completed", False, 0.7), ("completed", False, 0.9)],
                               2: [("completed", False, 0.8), ("interrupted", False, None), ("completed", False, 0.6)]},
                              ("kept", "keep")),
}


def selftest(args) -> dict:
    """Run the whole staged screen on synthetic result files, plan by plan, and check every task's outcome."""
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "registry.json"
    registry.write_text(json.dumps({"tasks": [{"task_id": t, "admission": "ADMITTED"} for t in SYNTHETIC]}, indent=2) + "\n")
    screen_dir, results_dir = out / "screen", out / "results"
    results_dir.mkdir()
    log = [{"step": "init", **init(argparse.Namespace(registry=str(registry), screen_dir=str(screen_dir),
                                                      store_root=str(out / "stores"), settings=args.settings,
                                                      allow_unpinned=args.allow_unpinned, concurrency=4))}]
    served, used = set(), {t: {} for t in SYNTHETIC}
    for _ in range(10):
        state = json.loads((screen_dir / "screen.json").read_text())
        pending = [p for p in state["plans"] if p["name"] not in served and p["purpose"] == "screen"]
        if not pending:
            break
        for plan in pending:
            served.add(plan["name"])
            rows = []
            for task, repeats in plan["tasks"].items():
                runs = SYNTHETIC[task][0][plan["stage"]]
                for i in range(repeats):
                    k = used[task].get(plan["stage"], 0)
                    used[task][plan["stage"]] = k + 1
                    status, solved, share = runs[k]
                    rows.append({"task_id": task, "run_id": f"{plan['name']}-{task}-{i}", "store": plan["store"],
                                 "status": status, "solved": solved, "end_reason": "synthetic",
                                 "submissions": [] if share is None else [{"ordinal": 1, "share_passed": share,
                                                                           "status": "pass" if solved else "fail"}],
                                 "final_share_passed": share})
            (results_dir / f"{plan['name']}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        files = sorted(str(p) for p in results_dir.glob("*.jsonl"))
        log.append({"step": "advance", "results": files,
                    **advance(argparse.Namespace(screen_dir=str(screen_dir), results=files, stores=[], supply_repeats=0))})
    state = json.loads((screen_dir / "screen.json").read_text())
    band = json.loads((screen_dir / "band.json").read_text())
    checks = {}
    for task, (scenario, (status, action)) in SYNTHETIC.items():
        info = state["tasks"][task]
        unused = {stage: len(runs) - used[task].get(stage, 0) for stage, runs in scenario.items()}
        checks[task] = {"expected": [status, action], "got": [info["status"], info["history"][-1]["action"]],
                        "path": [h["action"] for h in info["history"]], "synthetic_runs_unused": unused,
                        "in_band": band["tasks"][task]["in_band"]}
        checks[task]["ok"] = checks[task]["expected"] == checks[task]["got"] and not any(unused.values())
    final = advance(argparse.Namespace(screen_dir=str(screen_dir), results=sorted(str(p) for p in results_dir.glob("*.jsonl")),
                                       stores=[], supply_repeats=6))
    log.append({"step": "advance --supply-repeats 6", **final})
    summary = {"version": VERSION, "written": now(), "ok": all(c["ok"] for c in checks.values()), "checks": checks,
               "plans": [p["path"] for p in json.loads((screen_dir / "screen.json").read_text())["plans"]], "log": log}
    (out / "selftest.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["ok"]:
        raise SystemExit(json.dumps(checks, indent=1))
    return {"ok": True, "tasks": {t: c["path"] for t, c in checks.items()}, "summary": str(out / "selftest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("selftest")
    t.add_argument("--out", required=True)
    t.add_argument("--settings", default="configs/v2/settings.json")
    t.add_argument("--allow-unpinned", action="store_true")
    i = sub.add_parser("init")
    i.add_argument("--registry", default="benchmark_tasks/admission/terminalbench-v2/registry.json")
    i.add_argument("--screen-dir", required=True)
    i.add_argument("--store-root", required=True)
    i.add_argument("--settings", default="configs/v2/settings.json")
    i.add_argument("--allow-unpinned", action="store_true")
    i.add_argument("--concurrency", type=int, default=4)
    a = sub.add_parser("advance")
    a.add_argument("--screen-dir", required=True)
    a.add_argument("--results", nargs="*")
    a.add_argument("--stores", nargs="*")
    a.add_argument("--supply-repeats", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps({"init": init, "advance": advance, "selftest": selftest}[args.command](args), indent=1))


if __name__ == "__main__":
    main()
