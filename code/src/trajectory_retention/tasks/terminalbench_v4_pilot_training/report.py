"""Gate report of the terminalbench-v4-pilot-training lane: every ratified task, from the committed ledgers and the local evidence.

Adapted from the terminalbench-v2 lane (lanes never import each other). Writes <evidence>/gate-report.json (full,
local) and the admission README (committed; statuses, counts, reasons and deviations only, never hidden-test names,
grader diagnostics or reference content).
"""
from __future__ import annotations

import datetime
import json

from . import environment as E
from . import source as S
from .package import ADMISSION, EVIDENCE, write_json

README = ADMISSION / "README.md"
COMMANDS = """\
```
# all commands from the repository root, with PYTHONPATH=src and the repo .venv
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training static              # static screen of the ratified tasks
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training run                 # build, gate, restore-check in ratified order
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training pipeline <task> ...  # one task: package, gate, restore check
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training rescan-leak <task>   # after adding a leak_review.py entry
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training record               # committed evidence, ledgers, registry (no seal)
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training seal                 # the user's step: admission.json + registry
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training verify-loads         # registry loads for every purpose
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training report               # this README and the local gate report
python -m trajectory_retention.tasks.terminalbench_v4_pilot_training restore-private      # re-stage private files on a fresh machine
python configs/v2/engine_e2e/m4_lane_v4_natural.py                         # scripted natural runs on the lane (no provider)
```"""


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def short(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def reviewed(by_agent: bool) -> str:
    from .leak_review import AGENT, REVIEWS
    rows = [v for v in REVIEWS.values() if (v["reviewer"] == AGENT) == by_agent]
    tasks = sorted({v["task"] for v in rows})
    return "; ".join(f"{t}: {sum(v['task'] == t for v in rows)}" for t in tasks) or "none"


def write() -> dict:
    ledger = json.loads((ADMISSION / "ledger.json").read_text())
    registry = json.loads((ADMISSION / "registry.json").read_text())
    rows = []
    for t in ledger["tasks"]:
        row = dict(t)
        path = ADMISSION / "tasks" / t["task_id"] / "ledger.json"
        if path.is_file():
            row["detail"] = json.loads(path.read_text())
        rows.append(row)
    try:
        disk = E.docker_disk()
    except Exception as exc:  # recorded, never invented
        disk = {"error": f"unmeasured: {type(exc).__name__}"}
    report = {"version": "terminalbench-v4-pilot-training.gate-report.v1", "written": now(), "source_revision": S.SOURCE_REVISION,
              "counts": ledger["counts"], "admitted": [e["task_id"] for e in registry["tasks"]],
              "awaiting_seal": registry.get("awaiting_seal", []), "docker_disk_at_report": disk, "tasks": rows}
    write_json(EVIDENCE / "gate-report.json", report)
    order = {"ADMITTED": 0, "VALIDATED": 1, "HOLD": 2, "EXCLUDED": 3}
    table = ["| Task | Status | Hidden tests | Stage | Reason |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (order[r["status"]], S.RATIFIED_TASKS.index(r["task"]))):
        tests = r.get("hidden_tests_collected") or r.get("hidden_tests_static")
        tests = "not counted" if tests is None else tests
        reason = ("every gate item and the restore check passed" + ("; awaiting the seal" if r["status"] == "VALIDATED"
                                                                    else "")
                  if r["status"] in ("ADMITTED", "VALIDATED") else "; ".join(r.get("reasons") or []))
        table.append(f"| {r['task']} | {r['status']} | {tests} | {r.get('stage')} | {short(reason)} |")
    deviations = []
    for r in rows:
        devs = (r.get("detail") or {}).get("deviations")
        if devs:
            deviations += [f"- **{r['task']}**", *[f"  - {d}" for d in devs]]
    counts = ledger["counts"]
    text = "\n".join([
        "# terminalbench-v4-pilot-training admission",
        "",
        f"Terminal-Bench 4 ({S.SOURCE_URL}, tag {S.SOURCE_TAG}, commit {S.SOURCE_REVISION[:8]}), the tasks the user "
        "ratified on 2026-10-12, in the ratified order. Lane code: `src/trajectory_retention/tasks/terminalbench_v4_pilot_training/`; "
        "packages: `benchmark_tasks/packages/terminalbench-v4-pilot-training/` (public part only; private tests and references live "
        "outside Git under `~/.cache/trajectory-retention/terminalbench-v4-pilot-training/private/`, re-staged by `restore-private`).",
        "",
        "Grading reproduces Harbor's separate verifier offline (`grading.py`): the submitted W and captured R are "
        "restored into a fresh worker container, the task.toml artifacts are copied out and into a fresh verifier "
        "container built from the task's tests/Dockerfile, and the task's own tests/test.sh runs there with network "
        "none; a researcher-only recorder plugin gives per-test pytest node ids; Harbor's reward file decides the native "
        "pass. Submission protocol: the v2 protocol (failing-set hash criterion, failing count only, 3 identical failing "
        "sets end the run, grades on the captured W and R), text from `experiments/v2_text.py`.",
        "",
        "Gate per task (`gate.py`, no model calls): per-test scoring; at least 3 hidden tests; grader isolation (separate "
        "verifier, network none, tamper control, private files absent from W and the worker image); reference 3/3; "
        "starter fails; flaky check (start and reference graded 3 times each); network rule; leak scan; no emulation; "
        "start fidelity (native /app owner and mode); restore check through the real runner with ScriptedModel; runtime "
        "class CAPTURED. VALIDATED tasks are admitted only by the user's `seal`.",
        "",
        f"Counts: {counts.get('ADMITTED', 0)} admitted, {counts.get('VALIDATED', 0)} validated (awaiting the seal), "
        f"{counts.get('HOLD', 0)} held, {counts.get('EXCLUDED', 0)} excluded. Generated {report['written']} by "
        "`report`; full evidence is local under `artifacts/v2/lane-v4/`.",
        "",
        f"Leak reviews recorded by the build agent (need a human check): {reviewed(True)}. Leak reviews by the user: "
        f"{reviewed(False)}.",
        "",
        "## Commands",
        "",
        COMMANDS,
        "",
        "## Results",
        "",
        *table,
        "",
        "## Deviations from the official environment (per package)",
        "",
        *(deviations or ["(no package built)"]),
        "",
    ])
    README.write_text(text)
    (EVIDENCE / "gate-report.md").write_text(text)
    return {"counts": counts, "admitted": report["admitted"], "awaiting_seal": report["awaiting_seal"],
            "readme": str(README), "report": str(EVIDENCE / "gate-report.json")}
