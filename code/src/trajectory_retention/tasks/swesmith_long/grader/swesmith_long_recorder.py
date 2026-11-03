"""Researcher-only pytest plugin: append one JSON line per native report.

Mounted read-only into the disposable grading container; never part of W.
Node ids are pytest's own ids (the ids SWE-smith's verbose log parser reads).
"""
import json
import os
import time

_PATH = os.environ.get("SWESMITH_LONG_REPORT", "/out/reports.jsonl")
_HANDLE = open(_PATH, "a", encoding="utf-8")


def _emit(row):
    _HANDLE.write(json.dumps(row, sort_keys=True) + "\n")
    _HANDLE.flush()


# Early -p plugins import before conftest/configuration loading. This marker
# separates "pytest started with the recorder" from harness/environment faults.
_emit({"event": "plugin_loaded", "time": time.time()})


def _text(report, limit=1500):
    try:
        value = report.longreprtext
    except Exception:  # pragma: no cover - defensive
        value = str(getattr(report, "longrepr", ""))
    return value[-limit:] if value else ""


def pytest_sessionstart(session):
    import pytest
    _emit({"event": "session_start", "pytest": pytest.__version__, "time": time.time()})


def pytest_collectreport(report):
    if report.failed:
        _emit({"event": "collect_error", "nodeid": report.nodeid, "text": _text(report)})


def pytest_runtest_logreport(report):
    _emit({"event": "report", "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
           "wasxfail": hasattr(report, "wasxfail"), "duration": round(report.duration, 6),
           "text": _text(report) if report.outcome != "passed" else ""})


def pytest_sessionfinish(session, exitstatus):
    _emit({"event": "session_finish", "exitstatus": int(exitstatus), "time": time.time()})
