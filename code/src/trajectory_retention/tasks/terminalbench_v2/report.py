"""Gate report of the terminalbench-v2 lane: every task considered, from the sealed ledger and the local evidence.

Writes artifacts/v2/lane/gate-report.json (full, local) and the README's results table (committed; counts, statuses and
reasons only, never hidden-test names, grader diagnostics or reference content).
"""
from __future__ import annotations

import datetime
import json
import re

from . import environment as E
from . import source as S
from .package import ADMISSION, EVIDENCE, write_json

README = ADMISSION / "README.md"
COMMANDS = """\
```
# all commands from the repository root, with PYTHONPATH=src and the repo .venv
python -m trajectory_retention.tasks.terminalbench_v2 static              # static screen of all 89 tasks (no Docker)
python -m trajectory_retention.tasks.terminalbench_v2 candidates [--limit N]  # build, gate and restore-check candidates
python -m trajectory_retention.tasks.terminalbench_v2 pipeline <task> ...  # one task: package, gate, restore check
python -m trajectory_retention.tasks.terminalbench_v2 rescan-leak <task>   # after adding a leak_review.py entry
python -m trajectory_retention.tasks.terminalbench_v2 seal                 # admission.json, registry.json, ledger.json
python -m trajectory_retention.tasks.terminalbench_v2 verify-loads         # registry loads for every purpose
python -m trajectory_retention.tasks.terminalbench_v2 report               # this table and artifacts/v2/lane/gate-report.json
python -m trajectory_retention.tasks.terminalbench_v2 restore-private      # re-stage private files on a fresh machine
python configs/v2/screen_tasks.py init --screen-dir artifacts/v2/screen/<name> --store-root ~/.cache/trajectory-retention/v2-screen/<name>
python configs/v2/screen_tasks.py advance --screen-dir artifacts/v2/screen/<name> --stores <store> ...
python configs/v2/screen_tasks.py selftest --out artifacts/v2/lane/screen-synthetic   # the rule on synthetic results
```"""

NOTES = """\
Readings and flags for the user (declared, not ratified):
- Network rule (network-rule/v1): a task is excluded when its instruction asks to fetch a remote resource or its
  reference solution passes only with network; the reference is always run first with network none.
- Leak scan (leak-scan/v1, `leak_scan` in `leak.py`): contents, comments and copies findings pass only with a recorded
  review in `leak_review.py`. The reviews there were written by the build agent and need a human check
  ({reviews}).
- path-tracing is excluded by the user (2026-10-07): it ships `/app/orig`, the compiled original renderer, and
  reversing it is a route to the solution. The leak scan compares text only and does not flag binaries.
- tune-mjcf is excluded by the user (2026-10-07): its public `/app/eval.py` runs essentially the hidden tests' checks,
  so the hidden tests are not hidden and the failing count adds nothing.
- regex-chess ships a public checker the instruction names (`/app/check.py`); the hidden tests share its harness code
  but replay other positions, like public tests plus held-out cases. Kept by the user (2026-10-07).
- regex-chess grades are slow on a working re.json: up to 220 s per grade in the gate (start states grade in about 1 s).
- Images: when this lane's recipe equals the committed terminalbench-v1 recipe, the v1 image is re-tagged instead of
  rebuilt (`adopted_from` in the package and environment records).
- Eligibility: no family split (user), so admitted tasks are eligible for development, calibration and main use."""


def reviewed() -> str:
    from .leak_review import REVIEWS
    tasks = sorted({v["task"] for v in REVIEWS.values()})
    return "; ".join(f"{t}: {sum(v['task'] == t for v in REVIEWS.values())}" for t in tasks) or "none"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


HOLDS = """\
Why most candidates are held: every image build that runs apt-get fails on this Mac. The Docker VM disk (98 GB) is full:
0 bytes are available to non-root users, so apt's sandbox user cannot write and apt reports "invalid signature" (seen on
path-tracing-reverse, make-mips-interpreter and sparql-university). Root can still write into the reserved blocks
(about 5 GB), which is how the four admitted tasks and the two python-based builds ran. The other candidates need apt
in the task Dockerfile, in the worker layer (to add python3 to a base that has none) or in the grader layer. Freeing or
enlarging the Docker disk is the user's call (this lane never prunes the shared daemon); then `candidates` resumes the
held tasks in build order. The table shortens this cause to "Docker disk full"."""


def short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def table_reason(text: str) -> str:
    text = re.sub(r"; apt-get cannot run in this Mac's Docker VM.*$", "; Docker disk full", text)
    text = text.replace(" (apt-get update: 'invalid signature', which here means the Docker VM disk is full: apt's "
                        "sandbox user cannot write)", " (apt-get: Docker disk full)")
    return short(text)


def admitted_detail(task: str, task_id: str) -> dict:
    base = ADMISSION / "tasks" / task_id
    gate = json.loads((base / "gate.json").read_text())
    restore = json.loads((base / "restore-check.json").read_text())
    env = json.loads((base / "environment.json").read_text())
    items = gate["items"]
    return {"hidden_tests": items["hidden_tests"]["count"], "start_share_passed": items["starter_fails"]["start_share_passed"],
            "reference_full_solves": f"{items['reference']['full_solves']}/{items['reference']['grades']}",
            "flaky": {k: items["flaky_tests"][k] for k in ("start_identical", "reference_identical")},
            "network": {k: items["network"].get(k) for k in ("rule", "reference_offline_full_solve", "solve_network_commands")},
            "leak_scan": items["leak_scan"]["counts"], "emulation": items["emulation"]["worker_architecture"],
            "grader_isolation": {k: items["grader_isolation"].get(k) for k in ("tamper_equals_start", "grader_network",
                                                                                "private_files_in_w_or_image")},
            "grade_wall_seconds": gate["timing"], "restore_check": {"passed": restore["passed"],
                                                                     "restore_seconds": restore["restore_seconds"]},
            "images": {k: {"id": v["image"]["id"], "adopted_from_v1": bool(v.get("adopted"))} for k, v in env.items()}}


def write() -> dict:
    ledger = json.loads((ADMISSION / "ledger.json").read_text())
    registry = json.loads((ADMISSION / "registry.json").read_text())
    rows = []
    for t in ledger["tasks"]:
        row = dict(t)
        if t["status"] == "ADMITTED":
            row["detail"] = admitted_detail(t["task"], t["task_id"])
        rows.append(row)
    try:
        free = round(E.docker_free_bytes() / 1e9, 1)
    except Exception as exc:  # recorded, never invented
        free = f"unmeasured: {type(exc).__name__}"
    report = {"version": "terminalbench-v2.gate-report.v1", "written": now(), "source_revision": S.SOURCE_REVISION,
              "registry_revision": S.REGISTRY_REVISION, "counts": ledger["counts"],
              "admitted": [e["task_id"] for e in registry["tasks"]], "docker_free_gb_at_report": free,
              "tasks": rows}
    write_json(EVIDENCE / "gate-report.json", report)
    order = {"ADMITTED": 0, "HOLD": 1, "EXCLUDED": 2}
    table = ["| Task | Status | Hidden tests | Stage | Reason |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (order[r["status"]], r["task"])):
        tests = r.get("hidden_tests_collected") or r.get("hidden_tests_static")
        tests = "not counted" if tests is None else tests
        reason = "; ".join(r.get("reasons") or []) if r["status"] != "ADMITTED" else "every gate item and the restore check passed"
        table.append(f"| {r['task']} | {r['status']} | {tests} | {r.get('stage')} | {table_reason(reason)} |")
    counts = ledger["counts"]
    text = "\n".join([
        "# terminalbench-v2 admission",
        "",
        f"Terminal-Bench 2 (source {S.SOURCE_REVISION[:12]}, registry {S.REGISTRY_REVISION[:12]}), all {S.TASK_COUNT} tasks "
        "considered. Lane code: `src/trajectory_retention/tasks/terminalbench_v2/`; packages: "
        "`benchmark_tasks/packages/terminalbench-v2/` (public part only; private tests and references live outside Git "
        "under `~/.cache/trajectory-retention/terminalbench-v2/private/`, re-staged by `restore-private`).",
        "",
        "Gate per task (`gate.py`, no model calls): per-test scoring; at least 3 hidden tests; grader isolation (network "
        "none, tamper control, private files absent from W and the worker image); reference 3/3; starter fails; flaky "
        "check (start and reference graded 3 times each); network rule; leak scan (names, hashes, contents, comments); no "
        "emulation; restore check through the real runner with ScriptedModel (`restore_check.py`); runtime class "
        "CAPTURED. Static filters run first (`static.py`). EXCLUDED is a failed rule; HOLD is a task this Mac could not "
        "carry through, with the reason.",
        "",
        f"Counts: {counts.get('ADMITTED', 0)} admitted, {counts.get('HOLD', 0)} held, {counts.get('EXCLUDED', 0)} "
        f"excluded. Generated {report['written']} by `report`; full evidence is local under `artifacts/v2/lane/`.",
        "",
        NOTES.replace("{reviews}", reviewed()),
        "",
        HOLDS,
        "",
        "## Commands",
        "",
        COMMANDS,
        "",
        "## Results",
        "",
        *table,
        "",
    ])
    README.write_text(text)
    (EVIDENCE / "gate-report.md").write_text(text)
    return {"counts": counts, "admitted": report["admitted"], "readme": str(README),
            "report": str(EVIDENCE / "gate-report.json")}
