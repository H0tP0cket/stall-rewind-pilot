"""Researcher-only pytest plugin: one JSON line per native pytest report (terminalbench-v4-pilot-heatpump).

Loaded into the task's own pytest run inside a disposable verifier container through environment variables the grader
adds to the native ``tests/test.sh`` invocation: ``PYTEST_ADDOPTS=-p tb_recorder``, with the module importable through
``PYTHONPATH=/tb-grader`` or, when test.sh sets its own PYTHONPATH, through the user site directory under
``PYTHONUSERBASE=/tb-grader/user``. On import it removes all three again, so every process the tests start sees the
verifier's native environment. Output goes
to /tb-grader/out (root only); it is never part of W, the worker image or the worker's view. Node ids are pytest's own.
"""
import json
import os
import time

_MARK_PATH = "/tb-grader"
_MARK_OPTS = "-p tb_recorder"
_paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != _MARK_PATH]
if _paths:
    os.environ["PYTHONPATH"] = os.pathsep.join(_paths)
else:
    os.environ.pop("PYTHONPATH", None)
if os.environ.get("PYTEST_ADDOPTS") == _MARK_OPTS:
    os.environ.pop("PYTEST_ADDOPTS")
if os.environ.get("PYTHONUSERBASE") == _MARK_PATH + "/user":
    os.environ.pop("PYTHONUSERBASE")

_HANDLE = open("/tb-grader/out/reports.jsonl", "a", encoding="utf-8")


def _emit(row):
    _HANDLE.write(json.dumps(row, sort_keys=True) + "\n")
    _HANDLE.flush()


_emit({"event": "plugin_loaded", "time": time.time(), "pid": os.getpid()})


def _text(report, limit=1500):
    try:
        value = report.longreprtext
    except Exception:  # pragma: no cover - defensive
        value = str(getattr(report, "longrepr", ""))
    return value[-limit:] if value else ""


def pytest_sessionstart(session):
    import pytest
    _emit({"event": "session_start", "pytest": pytest.__version__, "rootdir": str(session.config.rootpath),
           "args": [str(a) for a in session.config.args], "time": time.time()})


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
