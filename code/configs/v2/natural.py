"""Thin natural-run driver for the v2 study: every task x repeat of one plan through MiningRunner.run directly.

  python configs/v2/natural.py --plan plan.json                 # run (resumable; a STOP file halts new runs)
  python configs/v2/natural.py --plan plan.json --dry-run       # validate, load tasks, print the jobs; runs nothing
  python configs/v2/natural.py --plan plan.json --status        # merged per-run status and spend

Plan (brief-interfaces.md): version "natural-plan.v2", registry, tasks, repeats, model, settings (a dict, or a path to
configs/v2/settings.json), store (a directory), concurrency, detector {"rule": "stall-clock/v8.1"}, budget
{"cost_limit_usd": 3.0, "step_limit": 1000000}, purpose screen|stall_supply. Optional: plan_id (default: the plan file
stem), execution_purpose (registry purpose, default "development" as in the earlier Terminal-Bench supply runs),
tool_timeout_seconds (the per-command timeout, default 300 as in those runs), env_file (credentials for live runs),
context_limit {rule: "context-limit/v1", window_tokens} (default: context_limit in the settings file; a live plan
without a confirmed window is refused), unresolved_retry {rule: "worst-case-charged-retry.v1",
worst_case_usd_per_million_tokens {prompt, completion}} (default: unresolved_retry in the settings file; absent, an
attempt whose generation is unresolved stops the run as before, Q8). submission_stop
{rule: "submission-no-improvement/v1", limit: 3, baseline: "first_submission"} defaults to the settings-file value;
when absent, the old submission rule remains. First failure/new best counts as 1; stop at 3. Branches inherit it
and start a fresh count and best-count baseline.
A development check may add "fixture": {"submission_protocol": "count-feedback-fixture/v1", "scripts": {key|"*":
path}}: scripted model responses and the lane-shaped submission protocol around a v1 adapter. Fixture runs are never
study data.

Each run: run_id "<plan_id>:<task_id>.r<NN>:natural" (postrun reads natural runs by this suffix), the natural-v2/v1 run
policy (a checkpoint at every point, submission loop, $3 runaway-spend guard with no real step limit, no wall-clock cap, no token cap), the v2
instance template, detector v8.1 live, no network. Layout under <store>: store/ (one ArtifactStore), claims/ (one
O_EXCL file per run, so no run is ever started twice), status/<key>.json (per-run status, updated at every
checkpoint), logs/, spend.json (spend totals), STOP (create it to start no further runs; runs in flight
finish). A run that was claimed but never finished is reported as orphaned and is never re-run under the same identity.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PLAN_VERSION = "natural-plan.v2"
DRIVER_VERSION = "natural-driver.v2/1"
PURPOSES = ("screen", "stall_supply")
DETECTOR_RULES = {"stall-clock/v8.1": {"stagnation_token_threshold": 10000}}  # 10k generated tokens (contract)
# The natural run has no token cap (contract); the engine's required allowance is set beyond reach.
NO_TOKEN_CAP = 10**9
DEFAULT_TOOL_TIMEOUT = 300
REQUIRED = ("version", "registry", "tasks", "repeats", "model", "settings", "store", "concurrency", "detector", "budget",
            "purpose")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    missing = [k for k in REQUIRED if k not in plan]
    if missing:
        raise ValueError(f"plan lacks {missing}")
    if plan["version"] != PLAN_VERSION:
        raise ValueError(f"plan version must be {PLAN_VERSION}")
    if plan["purpose"] not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if plan["detector"] != {"rule": "stall-clock/v8.1"}:
        raise ValueError("natural v2 runs use detector stall-clock/v8.1 exactly")
    budget = plan["budget"]
    if set(budget) != {"cost_limit_usd", "step_limit"} or not budget["cost_limit_usd"] > 0 \
            or type(budget["step_limit"]) is not int or budget["step_limit"] <= 0:
        raise ValueError("budget is {cost_limit_usd, step_limit}")
    if not isinstance(plan["tasks"], list) or not plan["tasks"] or len(set(plan["tasks"])) != len(plan["tasks"]):
        raise ValueError("tasks must be a non-empty list of distinct task ids")
    if type(plan["repeats"]) is not int or plan["repeats"] <= 0 or type(plan["concurrency"]) is not int \
            or plan["concurrency"] <= 0:
        raise ValueError("repeats and concurrency must be positive integers")
    if any(":" in t for t in plan["tasks"]):
        raise ValueError("task ids may not contain ':'")
    settings = plan["settings"]
    limit = plan.get("context_limit")
    retry = plan.get("unresolved_retry")
    submission_stop = plan.get("submission_stop")
    submission_feedback = plan.get("submission_feedback")
    if isinstance(settings, str):
        bound = json.loads(resolve(path, settings).read_text())
        if bound.get("model") != plan["model"]:
            raise ValueError("settings file is for a different model")
        settings = bound["settings"]
        if limit is None and (bound.get("context_limit") or {}).get("window_tokens") is not None:
            limit = {"rule": bound["context_limit"]["rule"], "window_tokens": bound["context_limit"]["window_tokens"],
                     "source": str(resolve(path, plan["settings"]))}
        if retry is None and bound.get("unresolved_retry") is not None:
            retry = bound["unresolved_retry"]
        if submission_stop is None:
            submission_stop = bound.get("submission_stop")
        if submission_feedback is None:
            submission_feedback = bound.get("submission_feedback")
    if not isinstance(settings, dict):
        raise ValueError("settings must be a dict or a settings.json path")
    if limit is not None:
        from trajectory_retention.execution.context_limit import RULE_VERSION
        if not isinstance(limit, dict) or limit.get("rule") != RULE_VERSION or type(limit.get("window_tokens")) is not int \
                or limit["window_tokens"] <= 0:
            raise ValueError(f"context_limit must be {{rule: {RULE_VERSION}, window_tokens: positive int}}")
    elif plan.get("fixture") is None:
        raise ValueError("the worker's context window is not confirmed: set context_limit.window_tokens in "
                         "configs/v2/settings.json (or the plan's context_limit) before a live run")
    fixture = plan.get("fixture")
    if fixture is not None:
        from trajectory_retention.experiments.v2_fixture import FIXTURES
        if fixture.get("submission_protocol") not in FIXTURES or not isinstance(fixture.get("scripts"), dict):
            raise ValueError("fixture needs a known submission_protocol and scripts")
    elif plan["model"].startswith("scripted/"):
        raise ValueError("scripted models are for fixture plans only")
    if retry is not None:
        from trajectory_retention.provider_policy import unresolved_retry_rule
        retry = unresolved_retry_rule(retry)
    from trajectory_retention.experiments.submission_stop import validate as validate_submission_stop
    submission_stop = validate_submission_stop(submission_stop)
    from trajectory_retention.experiments.submission_feedback import validate as validate_submission_feedback
    submission_feedback = validate_submission_feedback(submission_feedback)
    plan = dict(plan, settings=settings,
                **({"submission_feedback": submission_feedback} if submission_feedback else {}),
                **({"submission_stop": submission_stop} if submission_stop else {}),
                **({"context_limit": limit} if limit else {}),
                **({"unresolved_retry": retry} if retry else {}),
                plan_id=plan.get("plan_id") or path.stem,
                execution_purpose=plan.get("execution_purpose", "development"),
                tool_timeout_seconds=plan.get("tool_timeout_seconds", DEFAULT_TOOL_TIMEOUT))
    if ":" in plan["plan_id"]:
        raise ValueError("plan_id may not contain ':'")
    plan["_path"] = str(path.resolve())
    return plan


def resolve(plan_path: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (plan_path.parent / p).resolve()


def binding_sha(plan: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "_path"}, sort_keys=True).encode()).hexdigest()


def jobs(plan: dict) -> list[dict]:
    """Repeat-major order: every task gets its r00 before any task gets r01 (staged screening adds repeats).

    The key starts with a short hash so that postrun's per-run receipt folders (the key's first 12 characters) and
    labels stay distinct across runs of one task."""
    out = []
    for r in range(plan["repeats"]):
        for task_id in plan["tasks"]:
            h = hashlib.sha256(f"{plan['plan_id']}:{task_id}:{r}".encode()).hexdigest()[:8]
            key = f"{h}-{task_id}.r{r:02d}"
            out.append({"key": key, "task_id": task_id, "repeat": r, "run_id": f"{plan['plan_id']}:{key}:natural"})
    return out


def layout(plan: dict) -> dict[str, Path]:
    root = resolve(Path(plan["_path"]), plan["store"])
    paths = {"root": root, "store": root / "store", "claims": root / "claims", "status": root / "status",
             "logs": root / "logs", "scratch": root / "scratch", "task_scratch": root / "task-scratch"}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths["spend"], paths["stop"], paths["binding"] = root / "spend.json", root / "STOP", root / "plan-binding.json"
    return paths


def bind(plan: dict, paths: dict) -> None:
    """One plan per store directory: a changed plan is a new plan, never a silent edit of a running one."""
    sha = binding_sha(plan)
    prior = read_json(paths["binding"])
    if prior is None:
        write_json(paths["binding"], {"plan_sha256": sha, "plan": {k: v for k, v in plan.items() if k != "_path"},
                                      "driver": DRIVER_VERSION, "bound_at": now()})
    elif prior["plan_sha256"] != sha:
        raise SystemExit("this store is bound to a different plan; use a new store directory")


def run_config(plan: dict, job: dict):
    from trajectory_retention.contracts import RunConfig
    from trajectory_retention.experiments.v2_text import INSTANCE_TEMPLATE_V2_VERSION, binding as text_binding
    from trajectory_retention.mining.runner import V2_NATURAL
    detector = {"rule_version": plan["detector"]["rule"], **DETECTOR_RULES[plan["detector"]["rule"]]}
    return RunConfig(run_id=job["run_id"], campaign_id=plan["plan_id"], purpose=plan["execution_purpose"],
        model=plan["model"], model_settings=plan["settings"], generated_token_limit=NO_TOKEN_CAP,
        max_model_calls=plan["budget"]["step_limit"], cost_limit_usd=plan["budget"]["cost_limit_usd"],
        tool_timeout_seconds=plan["tool_timeout_seconds"], detector_settings=detector, repeat_id=f"r{job['repeat']:02d}",
        metadata={"v2_run_policy": V2_NATURAL, "instance_template": INSTANCE_TEMPLATE_V2_VERSION,
                  "v2_text": text_binding(), "plan_id": plan["plan_id"], "plan_purpose": plan["purpose"],
                  "plan_sha256": binding_sha(plan), "registry": str(resolve(Path(plan["_path"]), plan["registry"])),
                  "task_id": job["task_id"], "repeat": job["repeat"], "driver": DRIVER_VERSION,
                  "token_cap": "none (engine allowance set beyond reach)",
                  **({"context_limit": plan["context_limit"]} if plan.get("context_limit") else {}),
                  **({"unresolved_retry": plan["unresolved_retry"]} if plan.get("unresolved_retry") else {}),
                  **({"submission_stop": plan["submission_stop"]} if plan.get("submission_stop") else {}),
                  **({"submission_feedback": plan["submission_feedback"]} if plan.get("submission_feedback") else {}),
                  "fixture": plan.get("fixture") is not None,
                  **({"submission_fixture": plan["fixture"]["submission_protocol"]} if plan.get("fixture") else {})})


def load_task(plan: dict, task_id: str, scratch: Path):
    from trajectory_retention.registry import read_registry
    registry = read_registry(resolve(Path(plan["_path"]), plan["registry"]))
    if task_id not in registry:
        raise ValueError(f"{task_id} is not in the registry")
    task = registry[task_id].load(scratch_root=scratch, purpose=plan["execution_purpose"])
    if plan.get("fixture") is not None:
        from trajectory_retention.experiments.v2_fixture import FIXTURES
        task = FIXTURES[plan["fixture"]["submission_protocol"]](task)
    if getattr(task, "native_submission_protocol", None) is None:
        raise ValueError(f"{task_id} declares no submission protocol; v2 natural runs need the lane's loop")
    return task


def model_factory(plan: dict, job: dict):
    fixture = plan.get("fixture")
    if fixture is not None:
        from trajectory_retention.execution.model import ScriptedModel
        scripts = fixture["scripts"]
        source = scripts.get(job["key"]) or scripts.get("*")
        script = json.loads(resolve(Path(plan["_path"]), source).read_text())
        # the plan's settings reach the fixture model too (an ordinary response cap sizes worst-case charges)
        return lambda config: ScriptedModel(script, model_name=plan["model"], model_settings=dict(plan["settings"]))
    env_file = Path(plan.get("env_file", "~/.config/trajectory-retention/openrouter.env")).expanduser()
    if not os.environ.get("OPENROUTER_API_KEY") and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.removeprefix("export ").split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    from trajectory_retention.execution.received_response_guard_v1 import ReceivedResponseGuardV1

    def factory(config):
        policy = config.worker_request_policy
        return ReceivedResponseGuardV1(model_name=config.model, model_settings=dict(config.model_settings),
            max_attempts=policy.max_attempts, backoff_seconds=policy.backoff_seconds,
            timeout_seconds=policy.timeout_seconds, usage_recovery_timeout_seconds=policy.usage_recovery_timeout_seconds)
    return factory


def spend_of(store, accounting_ref) -> dict:
    snap = store.get_json(accounting_ref)
    worker = snap["roles"].get("worker", {}).get("cost_usd", {})
    return {"spend_usd_known": worker.get("known_sum", 0.0), "model_calls": snap["totals"]["model_calls"]}


def run_job(plan: dict, job: dict, *, model_factory_override=None) -> dict:
    """One natural run in this process; writes status/<key>.json throughout and returns the final status."""
    from trajectory_retention.detector.core import DetectorConfig
    from trajectory_retention.mining.points import checkpoint_step_index
    from trajectory_retention.mining.runner import MiningRunner
    from trajectory_retention.store import ArtifactStore
    paths = layout(plan)
    status_path = paths["status"] / f"{job['key']}.json"
    store = ArtifactStore(paths["store"])
    status = {"key": job["key"], "run_id": job["run_id"], "task_id": job["task_id"], "repeat": job["repeat"],
              "plan_id": plan["plan_id"], "state": "starting", "pid": os.getpid(), "host": socket.gethostname(),
              "started_at": now(), "fixture": plan.get("fixture") is not None}
    write_json(status_path, status)
    clock = time.monotonic()
    try:
        config = run_config(plan, job)
        task = load_task(plan, job["task_id"], paths["task_scratch"] / job["key"])
        record = task.describe()
        runner = MiningRunner(model_factory=model_factory_override or model_factory(plan, job),
                              worker_image=record.environment_version, network="none",
                              detector_config=DetectorConfig(**config.detector_settings),
                              scratch_root=paths["scratch"])

        def published(checkpoint):
            live = dict(status, state="running", updated_at=now(), checkpoints=checkpoint.ordinal + 1,
                        step_index=checkpoint_step_index(store, checkpoint), **spend_of(store, checkpoint.accounting_ref))
            write_json(status_path, live)

        result = runner.run(task, config, store=store, checkpoint_published=published)
        status.update(final_status(store, job["run_id"], result))
    except BaseException as exc:
        status.update(state="error", error=f"{type(exc).__name__}: {exc}")
        write_json(status_path, dict(status, wall_seconds=time.monotonic() - clock, finished_at=now()))
        raise
    status.update(state="finished", wall_seconds=time.monotonic() - clock, finished_at=now())
    write_json(status_path, status)
    return status


def final_status(store, run_id: str, result) -> dict:
    from trajectory_retention.accounting import CostLedger
    from trajectory_retention.analysis.submissions_v2 import submissions as submission_behaviour
    from trajectory_retention.experiments.branch_v2 import context_limit_summary
    records = CostLedger(store).records(run_id)
    models = [r for r in records if r.kind == "model"]
    events = list(store.iter_events(run_id))
    verdicts = [e.payload for e in events if e.kind == "lifecycle"
                and e.payload.get("transition") == "native_submission_verdict"]
    private = [e.payload for e in events if e.kind == "lifecycle"
               and e.payload.get("transition") == "private_full_solve_logged"]
    terminal = next((e.payload for e in reversed(events) if e.kind == "terminal"), {})
    charged = [e.payload["worst_case_charge"] for e in events if e.kind == "model_attempt"
               and e.payload.get("phase") == "finished" and e.payload.get("worst_case_charge")]
    return {"status": result.status, "termination_reason": result.termination_reason,
            "steps": (terminal.get("counters") or {}).get("calls_this_run"),
            "spend_usd_known": sum(r.usage.cost_usd for r in models if r.usage.cost_usd is not None),
            "model_attempts": len(models), "model_attempts_cost_unknown": sum(r.usage.cost_usd is None for r in models),
            "model_attempts_charged_worst_case": len(charged),
            "spend_usd_worst_case_charged": sum(c["usd"] for c in charged),
            "submissions": [{"step": v["step_index"], "status": v["status"], "failing_count": v.get("failing_count"),
                             "share_passed": v.get("share_passed"), "consecutive_same": v["consecutive_same"],
                             "checkpoint_id": v["checkpoint_id"], "evaluation_id": v["evaluation_id"]} for v in verdicts],
            "solved_by_submission": any(v["status"] == "pass" for v in verdicts),
            "private_full_solves_logged": len(private), "checkpoints": len(result.checkpoint_ids),
            "context_limit": context_limit_summary(events),
            "first_stall_id": result.first_stall_id, "submission_behaviour": submission_behaviour(store, run_id)}


def claim(paths: dict, job: dict) -> bool:
    try:
        fd = os.open(paths["claims"] / job["key"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"run_id": job["run_id"], "pid": os.getpid(), "host": socket.gethostname(), "claimed_at": now()}, f)
    return True


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def merged(plan: dict, paths: dict) -> dict:
    rows, total, unknown, charged, by_state = {}, 0.0, 0, 0.0, {}
    for job in jobs(plan):
        s = read_json(paths["status"] / f"{job['key']}.json")
        claimed = (paths["claims"] / job["key"]).exists()
        state = (s or {}).get("state") or ("claimed" if claimed else "pending")
        if state in ("starting", "running", "claimed") and s is not None and not alive(s.get("pid")):
            state = "orphaned"
        by_state[state] = by_state.get(state, 0) + 1
        if s:
            total += s.get("spend_usd_known") or 0.0
            unknown += s.get("model_attempts_cost_unknown") or 0
            charged += s.get("spend_usd_worst_case_charged") or 0.0
            rows[job["key"]] = {"state": state, "spend_usd_known": s.get("spend_usd_known"),
                                "termination_reason": s.get("termination_reason"), "step_index": s.get("step_index")}
    return {"plan_id": plan["plan_id"], "updated_at": now(), "runs": len(jobs(plan)), "by_state": by_state,
            "spend_usd_known": round(total, 6), "model_attempts_cost_unknown": unknown,
            "spend_usd_worst_case_charged": round(charged, 6),
            "stopped": paths["stop"].exists(), "by_run": rows}


def drive(plan: dict, plan_path: Path) -> dict:
    paths = layout(plan)
    bind(plan, paths)
    # Initialize WAL and schema before parallel children open a brand-new store.
    # Concurrent first opens can otherwise fail before a run or cost is recorded.
    from trajectory_retention.store import ArtifactStore
    ArtifactStore(paths["store"])
    stop_refresh = threading.Event()

    def refresher():
        while not stop_refresh.wait(30):
            write_json(paths["spend"], merged(plan, paths))

    threading.Thread(target=refresher, daemon=True).start()
    queue, lock = list(jobs(plan)), threading.Lock()

    def worker():
        while True:
            if paths["stop"].exists():
                return
            with lock:
                job = next((j for j in queue if claim(paths, j)), None)
                if job is not None:
                    queue.remove(job)
            if job is None:
                return
            with open(paths["logs"] / f"{job['key']}.log", "ab") as log:
                rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path),
                                     "--job", job["key"]], stdout=log, stderr=subprocess.STDOUT).returncode
            s = read_json(paths["status"] / f"{job['key']}.json") or {}
            if s.get("state") != "finished":
                write_json(paths["status"] / f"{job['key']}.json",
                           dict(s, key=job["key"], run_id=job["run_id"], state="error", returncode=rc, finished_at=now()))
            write_json(paths["spend"], merged(plan, paths))

    with cf.ThreadPoolExecutor(plan["concurrency"]) as pool:
        for f in [pool.submit(worker) for _ in range(plan["concurrency"])]:
            f.result()
    stop_refresh.set()
    summary = merged(plan, paths)
    write_json(paths["spend"], summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--job", help=argparse.SUPPRESS)  # internal: one run in a child process
    a = ap.parse_args()
    plan_path = Path(a.plan).resolve()
    plan = load_plan(plan_path)
    if a.job:
        job = next(j for j in jobs(plan) if j["key"] == a.job)
        run_job(plan, job)
        return
    paths = layout(plan)
    if a.status:
        print(json.dumps(merged(plan, paths), indent=1))
        return
    if a.dry_run:
        out = []
        for job in jobs(plan):
            config = run_config(plan, job)
            out.append({"key": job["key"], "run_id": job["run_id"], "max_model_calls": config.max_model_calls,
                        "cost_limit_usd": config.cost_limit_usd, "tool_timeout_seconds": config.tool_timeout_seconds,
                        "claimed": (paths["claims"] / job["key"]).exists()})
        for task_id in plan["tasks"]:
            load_task(plan, task_id, paths["task_scratch"] / f"dry-run-{task_id}")
        print(json.dumps({"plan_id": plan["plan_id"], "plan_sha256": binding_sha(plan), "jobs": out}, indent=1))
        return
    print(json.dumps(drive(plan, plan_path), indent=1))


if __name__ == "__main__":
    main()
