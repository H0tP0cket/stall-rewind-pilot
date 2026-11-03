"""Branch driver for v2 spec sets (track S): stall-major interleaved scheduling, O_EXCL claims, STOP files checked
before every job, resumable partial blocks, status merge, concurrency, and a pluggable runner.

Order: the spec set's blocks (one per stall or healthy checkpoint) in order; inside a block, repeat-major with arms
interleaved (the manifest's job order), so a stall's repeats run close together for the provider cache. A STOP stops
new launches only; running branches finish. On resume, partial blocks (some jobs started, some not) are scheduled
first, then untouched blocks; --equalize-only instead brings each partial block to equal repeats (every arm up to
the highest repeat any arm started) and starts nothing else. Disk is the source of truth: a job is done when one of
its attempts has a result.json whose end_reason is not "error"; claims mark started attempts. Errors are retried
only with --retry-errors, into a new attempt directory (a0, a1, ...); earlier attempts are kept.

  run    --specset M --out O --runner stub|e|module:function [--parallel N] [--tag T] [--stop-file F ...]
         [--spend-stop-usd X] [--equalize-only] [--retry-errors] [--isolation process|thread] [--allow-draft]
  status --specset M --out O         merged status from disk and every driver-status-*.json
  release --out O --branch-id B ...  explicitly release claims left without an outcome by a killed driver (the
                                     attempt dir, if any, is kept and renamed *.released); never automatic
  one    --runner R --spec S --out D  run one branch (used by process isolation)
Runners: "e" = trajectory_retention.experiments.branch_v2:run_branch (track E); "stub" = configs/v2/stub_branch.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

from study_common import (HERE, attempt_dirs, branch_dir, load_specset, read_json, safe, spec_of, write_json)

RUNNERS = {"e": "trajectory_retention.experiments.branch_v2:run_branch", "stub": "stub_branch:run_branch"}
NOW = lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def resolve_runner(name: str):
    mod, fn = RUNNERS.get(name, name).split(":")
    return getattr(importlib.import_module(mod), fn)


def run_one(runner: str, spec: dict, out: Path) -> int:
    """Call the runner once; result.json is the runner's; a raised exception becomes runner-error.json."""
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = resolve_runner(runner)(spec, out_dir=out)
        if not (out / "result.json").exists() and isinstance(result, dict):
            write_json(out / "result.json", result | {"_written_by": "driver (runner returned without writing)"})
        return 0
    except BaseException as exc:  # recorded, never swallowed silently
        write_json(out / "runner-error.json", {"error": f"{type(exc).__name__}: {exc}"[:2000],
                                               "traceback": traceback.format_exc()[-8000:], "at": NOW()})
        return 1


def claims_index(claims: Path) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = defaultdict(list)
    for p in claims.glob("*__a*"):
        if not p.name.endswith(".released"):
            stem, att = p.name.rsplit("__", 1)
            idx[stem].append(att)
    return idx


def job_state(out: Path, bid: str, claims: Path, index: dict | None = None) -> dict:
    """done | error | runner_failed | in_flight (claimed, no outcome) | new, from disk."""
    bdir = branch_dir(out, bid)
    atts = attempt_dirs(bdir)
    states = []
    for a in atts:
        r = read_json(a / "result.json", None)
        if r is not None:
            states.append(("error" if r.get("end_reason") == "error" else "done", a.name, r))
        elif (a / "runner-error.json").exists():
            states.append(("runner_failed", a.name, None))
        else:
            states.append(("in_flight", a.name, None))
    claimed = sorted(index.get(safe(bid), []) if index is not None else
                     (p.name.rsplit("__", 1)[1] for p in claims.glob(f"{safe(bid)}__a*")
                      if not p.name.endswith(".released")))
    released = len(list(bdir.glob("a*.released"))) if bdir.exists() else 0
    for c in claimed:
        if c not in {a.name for a in atts}:
            states.append(("in_flight", c, None))
    if any(s[0] == "done" for s in states):
        kind = "done"
    elif any(s[0] == "in_flight" for s in states):
        kind = "in_flight"
    elif states:
        kind = states[-1][0]
    else:
        kind = "new"
    cost = sum(float((s[2] or {}).get("cost_usd") or 0.0) for s in states)
    return {"state": kind, "attempts": [s[1] for s in states], "next_attempt": len(states) + released, "cost_usd": cost,
            "result": next((s[2] for s in reversed(states) if s[0] == "done"), None)}


def plan(manifest: dict, out: Path, claims: Path, *, equalize_only: bool, retry_errors: bool) -> tuple[list, dict]:
    """Jobs to launch in order, and the block summary they came from."""
    partial, fresh, summary = [], [], {}
    idx = claims_index(claims)
    for block in manifest["blocks"]:
        rows = [(job, job_state(out, job["branch_id"], claims, idx)) for job in block["jobs"]]
        started = [j for j, s in rows if s["state"] != "new"]
        todo = [(j, s) for j, s in rows if s["state"] == "new" or (retry_errors and s["state"] in ("error", "runner_failed"))]
        is_partial = bool(started) and any(s["state"] == "new" for _, s in rows)
        summary[block["unit_id"]] = {"jobs": len(rows), "started": len(started), "partial": is_partial,
                                     "done": sum(s["state"] == "done" for _, s in rows)}
        if equalize_only:
            if not started:
                continue
            level = max(j["repeat"] for j in started) + 1
            todo = [(j, s) for j, s in todo if j["repeat"] < level]
        (partial if started else fresh).extend((block, j, s) for j, s in todo)
    return partial + fresh, summary


def cmd_run(a) -> dict:
    manifest = load_specset(a.specset)
    if manifest["draft_rules_used"] and not a.allow_draft:
        raise SystemExit(f"spec set {manifest['prefix']} was built on unratified rules {manifest['draft_rules_used']}; "
                         "pass --allow-draft only for a mechanics test")
    out = Path(a.out)
    claims = out / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{socket.gethostname().split('.')[0]}-{os.getpid()}"
    status_path = out / f"driver-status-{safe(tag)}.json"
    statuses: dict = read_json(status_path, {})
    jobs, summary = plan(manifest, out, claims, equalize_only=a.equalize_only, retry_errors=a.retry_errors)
    stops = [out / "STOP", *map(Path, a.stop_file)]
    lock = threading.Lock()
    idx = claims_index(claims)
    spent = {"usd": sum(job_state(out, j["branch_id"], claims, idx)["cost_usd"] for b in manifest["blocks"]
                        for j in b["jobs"])}
    counts = defaultdict(int)

    def save():
        write_json(status_path, statuses)

    def stopped() -> str | None:
        hit = next((str(p) for p in stops if p.exists()), None)
        if hit:
            return f"STOP file {hit}"
        if a.spend_stop_usd is not None and spent["usd"] >= a.spend_stop_usd:
            return f"spend {spent['usd']:.2f} >= {a.spend_stop_usd}"
        return None

    def one(item):
        block, job, st = item
        bid = job["branch_id"]
        key = f"{manifest['prefix']}|{bid}"
        with lock:
            why = stopped()
            if why:
                statuses[key] = {"status": "skipped_stop", "reason": why, "at": NOW()}
                counts["skipped_stop"] += 1
                return
            fresh = job_state(out, bid, claims)
            if fresh["state"] in ("done", "in_flight") or (fresh["state"] != "new" and not a.retry_errors):
                counts[f"skipped_{fresh['state']}"] += 1
                return
            att = f"a{fresh['next_attempt']}"
            try:
                fd = os.open(claims / f"{safe(bid)}__{att}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                counts["skipped_claimed"] += 1
                return
            os.write(fd, f"{os.getpid()} {tag} {NOW()}\n".encode())
            os.close(fd)
            statuses[key] = {"status": "running", "attempt": att, "at": NOW()}
            save()
        adir = branch_dir(out, bid) / att
        adir.mkdir(parents=True, exist_ok=True)
        spec = spec_of(manifest, job)
        write_json(adir / "spec.json", spec)
        t0 = time.time()
        if a.isolation == "thread":
            rc = run_one(a.runner, spec, adir)
        else:
            with open(adir / "runner.log", "ab") as log:
                rc = subprocess.run([sys.executable, "-B", str(HERE / "driver_v2.py"), "one", "--runner", a.runner,
                                     "--spec", str(adir / "spec.json"), "--out", str(adir)], stdout=log,
                                    stderr=subprocess.STDOUT, env=os.environ.copy()).returncode
        res = read_json(adir / "result.json", None)
        write_json(adir / "driver.json", {"runner": a.runner, "isolation": a.isolation, "rc": rc, "tag": tag,
                                          "started": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(),
                                          "seconds": round(time.time() - t0, 3)})
        with lock:
            cost = float((res or {}).get("cost_usd") or 0.0)
            spent["usd"] += cost
            state = ("runner_failed" if res is None else "error" if res.get("end_reason") == "error" else "done")
            counts[state] += 1
            statuses[key] = {"status": state, "attempt": att, "end_reason": (res or {}).get("end_reason"),
                             "solved": (res or {}).get("solved"), "cost_usd": cost, "dir": str(adir), "at": NOW()}
            save()

    with cf.ThreadPoolExecutor(max_workers=a.parallel) as pool:
        list(pool.map(one, jobs))
    save()
    r = {"prefix": manifest["prefix"], "planned": len(jobs), "partial_blocks_first": sum(v["partial"] for v in summary.values()),
         "counts": dict(counts), "spent_usd_in_set": round(spent["usd"], 6), "stopped": stopped(), "status_file": str(status_path)}
    print(json.dumps(r, indent=1))
    return r


def status(manifest: dict, out: Path) -> dict:
    """Merged status from disk (authoritative) plus every driver's status file (for skip reasons)."""
    claims = out / "claims"
    idx = claims_index(claims)
    per_block, totals, cost = {}, defaultdict(int), 0.0
    for block in manifest["blocks"]:
        arms = defaultdict(lambda: defaultdict(int))
        started_levels = defaultdict(set)
        for job in block["jobs"]:
            s = job_state(out, job["branch_id"], claims, idx)
            arms[job["arm"]][s["state"]] += 1
            totals[s["state"]] += 1
            cost += s["cost_usd"]
            if s["state"] == "done":
                started_levels[job["arm"]].add(job["repeat"])
        done = {arm: len(v) for arm, v in started_levels.items()}
        all_arms = sorted({j["arm"] for j in block["jobs"]})
        counts = [done.get(arm, 0) for arm in all_arms]
        n_new = sum(v.get("new", 0) for v in arms.values())
        per_block[block["unit_id"]] = {"done_per_arm": {arm: done.get(arm, 0) for arm in all_arms},
                                       "equal_repeats": len(set(counts)) == 1,
                                       "partial": 0 < n_new < len(block["jobs"]),
                                       "states": {k: dict(v) for k, v in arms.items()}}
    merged = {}
    for f in sorted(out.glob("driver-status-*.json")):
        for k, v in read_json(f).items():
            rank = {"done": 4, "error": 3, "runner_failed": 3, "running": 2, "skipped_stop": 1}.get(v.get("status"), 0)
            if k not in merged or rank >= merged[k][0]:
                merged[k] = (rank, v)
    skipped = sum(1 for _, v in merged.values() if v.get("status") == "skipped_stop")
    return {"prefix": manifest["prefix"], "branches": manifest["branches"], "states": dict(totals),
            "cost_usd": round(cost, 6), "stop_file_present": (out / "STOP").exists(),
            "last_skipped_stop_entries": skipped, "blocks": per_block,
            "partial_blocks": [u for u, b in per_block.items() if b["partial"]],
            "unequal_blocks": [u for u, b in per_block.items() if not b["equal_repeats"]]}


def release(out: Path, bids: list[str]) -> dict:
    claims, done = out / "claims", {}
    for bid in bids:
        for c in sorted(claims.glob(f"{safe(bid)}__a*")):
            att = c.name.rsplit("__", 1)[1]
            adir = branch_dir(out, bid) / att
            if (adir / "result.json").exists() or (adir / "runner-error.json").exists():
                continue
            if adir.exists():
                adir.rename(adir.with_name(f"{att}.released"))
            c.rename(c.with_name(c.name + ".released"))
            done.setdefault(bid, []).append(att)
    return {"released": done, "at": NOW()}


def main(argv: list[str] | None = None) -> dict | int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--specset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--tag")
    p.add_argument("--stop-file", action="append", default=[])
    p.add_argument("--spend-stop-usd", type=float)
    p.add_argument("--equalize-only", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--isolation", choices=("process", "thread"), default="process")
    p.add_argument("--allow-draft", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--specset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("release")
    p.add_argument("--out", required=True)
    p.add_argument("--branch-id", action="append", required=True)
    p = sub.add_parser("one")
    p.add_argument("--runner", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "one":
        raise SystemExit(run_one(a.runner, read_json(a.spec), Path(a.out)))
    if a.cmd == "release":
        r = release(Path(a.out), a.branch_id)
        print(json.dumps(r, indent=1))
        return r
    if a.cmd == "status":
        r = status(load_specset(a.specset), Path(a.out))
        print(json.dumps(r, indent=1))
        return r
    return cmd_run(a)


if __name__ == "__main__":
    main()
