"""Raw pytest recorder output -> frozen per-test statuses (terminalbench-v2).

Copied unchanged from ``tasks/swesmith_long/grading.py`` (``observe``, ``_status``, ``translate``): the v1 lane imports
them from there, and its seals hash that file, so this lane keeps its own copy instead of importing across lanes.
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
            "session_started": any(r.get("event") == "session_start" for r in rows),
            "session_finished": [r["exitstatus"] for r in rows if r.get("event") == "session_finish"],
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

    INFRA_ERROR (invalid grade) is reserved for harness faults: the container or
    driver failed, pytest never started with the recorder, or the recorder output
    is inconsistent. Once pytest has started, failures to load the pristine
    conftest/configuration, collection/import errors, crashes and hangs are
    attributed to the submitted W (the same frozen environment and test-support
    set pass at the reference) and become BLOCKED / ERROR_TASK / FAIL checks.
    """
    driver = receipt.get("driver") or {}
    timed_out = bool(driver.get("timed_out"))
    invalid = (receipt.get("infrastructure_error") or not observation["plugin_loaded"]
               or observation["duplicates"] or observation["unparseable"])
    if invalid:
        reason = receipt.get("infrastructure_error") or "invalid native grading observation"
        return tuple(CheckResult(check_id=cid, status=CheckStatus.INFRA_ERROR,
                                 details={"reachable": None, "reason": reason}) for cid in expected)
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
                             "blocking_reason": "pristine conftest/configuration failed to load with the submitted code",
                             "pytest_returncode": driver.get("returncode")}))
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
                             "pytest_returncode": driver.get("returncode")}))
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
