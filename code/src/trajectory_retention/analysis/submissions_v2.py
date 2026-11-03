"""Submission behaviour of one v2 natural run or branch, read from what its store already records.

For every graded submission: its step, steps, seconds and worker commands since the previous submission, whether W
(file contents) or R (state outside /workspace) differ from the previous submission's capture, whether the worker ran
a check of its own since then (CHECK_RULE), the failing count and its change, and whether the failing set repeats the
previous one. The run summary shows grader probing (resubmitting with nothing changed) and how often the worker checks
its own work before submitting. Analysis only; the worker and detector never see it.

Steps count from the run's start for natural runs and from the branch start for branches (step_index minus the
branch's call offset, as in the branch result). A branch's first submission has no earlier capture in its own store,
so its change fields are None; commands and time in a branch count from the branch start, after any review or note
turn.

  python -m trajectory_retention.analysis.submissions_v2 --store <store dir> [--run <run_id> ...] [--out file]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from ..mining.points import checkpoint_step_index, point_checkpoints
from ..store import ArtifactStore

VERSION = "submission-behaviour/v1"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
CHECK_RULE = ("own check = a worker bash command since the previous submission that runs tests or a checker: pytest, "
              "unittest, nose, tox, make/cargo/go/npm/yarn/pnpm test, or running (with python, bash, sh, node or ruby, "
              "or directly as ./x or /path/x) a script whose file name has a word starting with test, check, eval, "
              "verify or validate; reading such a file does not count")
CHECK_PATTERN = re.compile(r"\b(pytest|unittest|nose2?|tox)\b|\b(make|cargo|go|npm|yarn|pnpm)\s+(run\s+)?test\b|"
                           r"(\b(python[\d.]*|bash|sh|node|ruby)\s+(-[-\w]*\s+)*|(^|[;&|(]\s*)\.?/)[\w./-]*?"
                           r"(?<![A-Za-z])(test|tests|check|eval|verify|validate)[\w.-]*\.(py|sh|js|rb)\b")


def run_ids(store_dir: Path) -> list[str]:
    db = sqlite3.connect(Path(store_dir) / "journal.sqlite3")
    try:
        return [r[0] for r in db.execute("select distinct run_id from events order by run_id")]
    finally:
        db.close()


def _state(store: Any, checkpoint_id: str | None) -> dict | None:
    if checkpoint_id is None:
        return None
    c = store.get_checkpoint(checkpoint_id)
    v = c.validation or {}
    return {"step": checkpoint_step_index(store, c), "w": v.get("workspace_manifest_digest"), "r": v.get("runtime_digest")}


def _changed(a: dict | None, b: dict | None, key: str) -> bool | None:
    if a is None or b is None or a[key] is None or b[key] is None:
        return None
    return a[key] != b[key]


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def submissions(store: Any, run_id: str) -> dict:
    events = list(store.iter_events(run_id))
    ready = next((e for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "branch_v2_ready"), None)
    branch, offset = ready is not None, int(ready.payload.get("call_offset") or 0) if ready else 0
    prev_state = None if branch else _state(store, point_checkpoints(store, run_id).get(0))
    prev_time = _ts((ready or events[0]).timestamp) if events else None
    rows, cmds, checks, prev = [], [], [], None
    for e in events:
        if e is ready:
            cmds, checks = [], []
        elif e.kind == "tool_call" and e.payload.get("tool_name") == "bash":
            cmd = str(e.payload.get("command") or "")
            if SUBMIT_MARKER not in cmd:
                cmds.append(cmd)
                if CHECK_PATTERN.search(cmd):
                    checks.append(cmd[:200])
        elif e.kind == "lifecycle" and e.payload.get("transition") == "native_submission_verdict":
            p = e.payload
            state = _state(store, p["checkpoint_id"])
            n = p.get("failing_count")
            step = state["step"] - offset
            rows.append({"ordinal": p["ordinal"], "step": step, "steps_since_previous": step - (prev["step"] if prev else 0),
                         "seconds_since_previous": round((_ts(e.timestamp) - prev_time).total_seconds(), 1),
                         "commands_since_previous": len(cmds),
                         "workspace_changed": _changed(prev_state, state, "w"),
                         "runtime_changed": _changed(prev_state, state, "r"),
                         "ran_own_check": bool(checks), "own_checks": checks,
                         "best_failing_count": p.get("best_failing_count"),
                         "no_improvement_count": p.get("no_improvement_count"),
                         "submission_stop": p.get("submission_stop"),
                         "failing_count": n, "failing_change": None if prev is None or n is None
                         or prev["failing_count"] is None else n - prev["failing_count"],
                         "same_failing_set_as_previous": None if prev is None else p["criterion_id"] == prev["criterion"],
                         "outcome": p.get("outcome"), "checkpoint_id": p["checkpoint_id"]})
            prev = rows[-1] | {"criterion": p["criterion_id"]}
            prev_state, prev_time, cmds, checks = state, _ts(e.timestamp), [], []
    later = rows[1:]

    def share(xs: list, test) -> float | None:
        xs = [x for x in xs if x is not None]
        return round(sum(1 for x in xs if test(x)) / len(xs), 3) if xs else None

    return {"run_id": run_id, "version": VERSION, "check_rule": CHECK_RULE, "branch": branch,
            "branch_call_offset": offset if branch else None, "submissions": rows,
            "summary": {
                "submissions": len(rows), "first_submission_step": rows[0]["step"] if rows else None,
                "median_steps_between": statistics.median(r["steps_since_previous"] for r in later) if later else None,
                "share_resubmitted_with_no_change": share(
                    [None if r["workspace_changed"] is None or r["runtime_changed"] is None
                     else not (r["workspace_changed"] or r["runtime_changed"]) for r in later], bool),
                "share_resubmitted_with_workspace_unchanged": share([r["workspace_changed"] for r in later], lambda x: not x),
                "share_with_own_check_before": share([r["ran_own_check"] for r in rows], bool),
                "share_same_failing_set_as_previous": share([r["same_failing_set_as_previous"] for r in later], bool)}}


def summary(store_dir: str | Path, run_id: str) -> dict | None:
    """Run summary for one store, or None when the store is missing."""
    if not (Path(store_dir) / "journal.sqlite3").exists():
        return None
    return submissions(ArtifactStore(store_dir), run_id)["summary"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--run", action="append")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    store = ArtifactStore(a.store)
    out = [submissions(store, r) for r in (a.run or run_ids(Path(a.store)))]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps([{"run_id": r["run_id"], **r["summary"]} for r in out if r["submissions"]], indent=1))


if __name__ == "__main__":
    main()
