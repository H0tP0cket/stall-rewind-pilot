"""Researcher-only pytest plugin: one JSON line per native pytest report.

Streamed into /tb-grader of a disposable grading container only; never part of
W, the worker image or the worker's view. Node ids are pytest's own ids.
"""
import json
import time

_HANDLE = open("/tb-grader/out/reports.jsonl", "a", encoding="utf-8")


def _emit(row):
    _HANDLE.write(json.dumps(row, sort_keys=True) + "\n")
    _HANDLE.flush()


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


def pytest_collection_finish(session):
    _emit({"event": "collected", "nodeids": [item.nodeid for item in session.items]})


def pytest_runtest_logreport(report):
    _emit({"event": "report", "nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
           "wasxfail": hasattr(report, "wasxfail"), "duration": round(report.duration, 6),
           "text": _text(report) if report.outcome != "passed" else ""})


def pytest_sessionfinish(session, exitstatus):
    _emit({"event": "session_finish", "exitstatus": int(exitstatus), "time": time.time()})
