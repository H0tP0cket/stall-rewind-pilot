"""Raw pytest recorder output -> frozen per-test statuses (terminalbench-v4-pilot-photonic).

Copied from the terminalbench-v2 lane (``observe``, ``_status``, ``translate``; lanes never import each other) with one
change for TB4's native ``tests/test.sh``: it can end before pytest starts (production-planning runs a replay script
under ``set -e`` first). When the grading transport and test.sh ran but the recorder never loaded, every check is
BLOCKED (``pytest_not_started``), which counts against the submitted state like any failing test; Harbor itself finds no
reward file there. A recorder that never loads on a correct state is caught by the gate (reference 3 of 3).
"""
from __future__ import annotations

import json
from pathlib import Path

from trajectory_retention.contracts import CheckResult, CheckStatus


def observe(output: Path) -> dict:
    path = Path(output) / "reports.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                rows.append({"event": "unparseable"})
    reports: dict[str, dict] = {}
    duplicates = []
    for row in rows:
        if row.get("event") == "report":
            phases = reports.setdefault(row["nodeid"], {})
            if row["when"] in phases:
                duplicates.append(row["nodeid"])
            phases[row["when"]] = row
    return {"plugin_loaded": any(r.get("event") == "plugin_loaded" for r in rows),
            "plugin_loads": sum(r.get("event") == "plugin_loaded" for r in rows),
            "session_started": any(r.get("event") == "session_start" for r in rows),
            "session_finished": [r["exitstatus"] for r in rows if r.get("event") == "session_finish"],
            "collected": next((r["nodeids"] for r in rows if r.get("event") == "collected"), None),
            "collect_errors": [{"nodeid": r["nodeid"], "text": r.get("text", "")} for r in rows if r.get("event") == "collect_error"],
            "reports": reports, "duplicates": sorted(set(duplicates)),
            "unparseable": sum(r.get("event") == "unparseable" for r in rows)}


def _status(phases: dict) -> tuple[CheckStatus, str, bool | None]:
    setup, call, teardown = phases.get("setup"), phases.get("call"), phases.get("teardown")
    if call is None:
        if setup is not None and setup["outcome"] == "failed":
            return CheckStatus.ERROR_TASK, "setup_error", None
        if setup is not None and setup["outcome"] == "skipped":
            return (CheckStatus.PASS, "xfailed", True) if setup["wasxfail"] else (CheckStatus.FAIL, "skipped", None)
        return CheckStatus.NOT_RUN, "no_call_phase", None
    if call["outcome"] == "passed":
        if teardown is not None and teardown["outcome"] == "failed":
            return CheckStatus.ERROR_TASK, "teardown_error", True
        return CheckStatus.PASS, "xpassed" if call["wasxfail"] else "passed", True
    if call["outcome"] == "skipped":
        return (CheckStatus.PASS, "xfailed", True) if call["wasxfail"] else (CheckStatus.FAIL, "skipped", True)
    return CheckStatus.FAIL, "failed", True


def translate(expected: list[str], observation: dict, receipt: dict) -> tuple[CheckResult, ...]:
    """Raw pytest observations -> frozen check statuses.

    INFRA_ERROR (invalid grade) is reserved for harness faults: a container or transport failed, or the recorder output
    is inconsistent (duplicate reports, unparseable lines, more than one pytest session). Everything that happens once
    the verifier's own test.sh runs is attributed to the submitted state (the same frozen verifier image passes the
    reference): test.sh ending before pytest, startup or collection errors, crashes and hangs become BLOCKED /
    ERROR_TASK / FAIL checks.
    """
    driver = receipt.get("driver") or {}
    timed_out = bool(driver.get("timed_out"))
    invalid = (receipt.get("infrastructure_error") or observation["duplicates"] or observation["unparseable"]
               or observation.get("plugin_loads", 0) > 1)
    if invalid:
        reason = receipt.get("infrastructure_error") or "invalid native grading observation"
        return tuple(CheckResult(check_id=cid, status=CheckStatus.INFRA_ERROR,
                                 details={"reachable": None, "reason": reason}) for cid in expected)
    if not observation["plugin_loaded"]:
        raw = "suite_timeout" if timed_out else "pytest_not_started"
        return tuple(CheckResult(check_id=cid, status=CheckStatus.BLOCKED, raw_status=raw,
                                 details={"reachable": False, "blocked_by": ["verifier_test_sh"],
                                          "blocking_reason": ("the verifier timed out before pytest started" if timed_out
                                                              else "the verifier's test.sh ended before pytest started"),
                                          "test_sh_returncode": driver.get("returncode")}) for cid in expected)
    startup_failed = not observation["session_started"]
    checks = []
    for cid in expected:
        phases = observation["reports"].get(cid)
        if phases is None:
            filename = cid.split("::", 1)[0]
            blockers = [e["nodeid"] for e in observation["collect_errors"]
                        if not e["nodeid"] or e["nodeid"] == filename or filename.startswith(e["nodeid"].rstrip("/") + "/")
                        or e["nodeid"].endswith("conftest.py")]
            if startup_failed and not timed_out:
                checks.append(CheckResult(check_id=cid, status=CheckStatus.BLOCKED, raw_status="pytest_startup_error",
                    details={"reachable": False, "blocked_by": ["pytest_startup"],
                             "blocking_reason": "pytest failed to start a session",
                             "test_sh_returncode": driver.get("returncode")}))
            elif blockers:
                checks.append(CheckResult(check_id=cid, status=CheckStatus.BLOCKED, raw_status="collection_error",
                    details={"reachable": False, "blocked_by": blockers, "blocking_reason": "collection_error"}))
            elif timed_out:
                checks.append(CheckResult(check_id=cid, status=CheckStatus.BLOCKED, raw_status="suite_timeout",
                    details={"reachable": None, "blocked_by": ["grade_timeout"], "blocking_reason": "suite_timeout"}))
            elif not observation["session_finished"]:
                checks.append(CheckResult(check_id=cid, status=CheckStatus.BLOCKED, raw_status="pytest_aborted",
                    details={"reachable": None, "blocked_by": ["pytest_aborted"],
                             "blocking_reason": "pytest ended before reporting this test",
                             "test_sh_returncode": driver.get("returncode")}))
            else:
                checks.append(CheckResult(check_id=cid, status=CheckStatus.NOT_RUN, raw_status="not_collected",
                    details={"reachable": None, "blocking_reason": "missing_native_check"}))
            continue
        status, raw, reachable = _status(phases)
        text = next((p.get("text") for p in phases.values() if p.get("text")), "")
        checks.append(CheckResult(check_id=cid, status=status, raw_status=raw,
            details={"reachable": reachable, "phases": {k: v["outcome"] for k, v in phases.items()},
                     "diagnostic_tail": text[-600:] if text else ""}))
    return tuple(checks)
