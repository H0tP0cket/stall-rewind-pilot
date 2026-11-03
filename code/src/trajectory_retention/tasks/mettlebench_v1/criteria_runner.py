"""Research-private all-criteria observation of an unmodified MettleBench evaluator.

Runs only inside a disposable grading container. It imports the upstream
verifier module and calls its own ``main()`` with the upstream arguments, after
wrapping every ``CHECKS`` entry. Each wrapper records the check's own outcome
and then reports success to ``main`` so the native loop continues through every
criterion in the native order, with the native call arguments (manifests,
ground truth) and shared module state. Native early stopping is therefore the
only behavior changed; the native verdict itself is produced separately by the
unmodified command.

Status translation (frozen, mettlebench.criteria-runner.v1):
  returned normally / CheckResult.ok true      -> PASS
  CheckFailure or a type the native loop maps to a failed point -> FAIL
  any other exception raised by the check      -> ERROR_TASK
  check never invoked by main                   -> NOT_RUN
  main() itself raised before/after the checks  -> recorded as main_error
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import importlib.util
import io
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

VERSION = "mettlebench.criteria-runner.v1"


class CheckTimeout(Exception):
    pass


def _handled_failure_types(source: str, module) -> tuple[type, ...]:
    """Exception types the verifier's own main loop converts into a failed point."""
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    names: set[str] = set()
    for node in ast.walk(main):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            for elt in (node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]):
                names.add(ast.unparse(elt))
    types: list[type] = []
    for name in sorted(names):
        if name in {"Exception", "BaseException"}:
            continue  # generic handlers are recorded as ERROR_TASK with the native mapping noted
        obj = module
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, type) and issubclass(obj, BaseException):
            types.append(obj)
    return tuple(types)


def _generic_handler(source: str) -> bool:
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return any(isinstance(n, ast.ExceptHandler) and n.type is not None and ast.unparse(n.type) == "Exception"
               for n in ast.walk(main))


def run(verifier: Path, workspace: Path, native_output: Path, per_check_timeout: float,
        progress: Path | None = None) -> dict:
    source = verifier.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mettle_private_verifier", verifier)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    checklist = json.loads((verifier.parent / "verifier_checklist.json").read_text(encoding="utf-8"))
    order = [p["id"] for p in checklist["points"]]
    handled = _handled_failure_types(source, module)
    generic = _generic_handler(source)
    outcomes: dict[str, dict] = {}

    def wrap(check_id, fn):
        def wrapper(*args, **kwargs):
            if check_id in outcomes:
                outcomes[check_id]["repeat_invocations"] = outcomes[check_id].get("repeat_invocations", 0) + 1
            started = time.monotonic()
            record = {"id": check_id}

            def on_alarm(signum, frame):
                raise CheckTimeout(f"check exceeded {per_check_timeout}s")
            previous = signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, per_check_timeout)
            try:
                value = fn(*args, **kwargs)
            except CheckTimeout as exc:
                record.update(status="ERROR_TASK", raw="check_timeout", exception_type="CheckTimeout", message=str(exc))
                value = None
            except BaseException as exc:  # noqa: BLE001 - recorded, never propagated to main
                if isinstance(exc, KeyboardInterrupt):
                    raise
                message = getattr(exc, "message", None) or str(exc)
                excerpt = getattr(exc, "excerpt", None)
                if isinstance(exc, getattr(module, "CheckFailure", ())) or (handled and isinstance(exc, handled)):
                    record.update(status="FAIL", raw="native_failed_point")
                else:
                    record.update(status="ERROR_TASK", raw="native_generic_failed_point" if generic else "native_error",
                                  traceback=traceback.format_exc()[-4000:])
                record.update(exception_type=type(exc).__name__, message=str(message)[:4000],
                              excerpt=(str(excerpt)[:4000] if excerpt else None))
                value = None
            else:
                ok = getattr(value, "ok", True) if value is not None else True
                if ok is False:
                    record.update(status="FAIL", raw="check_result_not_ok",
                                  excerpt=str(getattr(value, "excerpt", "") or "")[:4000])
                else:
                    record.update(status="PASS", raw="returned")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            record["seconds"] = round(time.monotonic() - started, 4)
            if check_id not in outcomes:
                outcomes[check_id] = record
            if progress is not None:
                # Incremental receipt: an outer kill still leaves attributable observations.
                progress.write_text(json.dumps({"version": VERSION, "partial": True,
                                                "criteria": list(outcomes.values())}, sort_keys=True) + "\n")
            # Report success to the native loop so it proceeds to the next criterion.
            if value is not None and hasattr(value, "ok"):
                if dataclasses.is_dataclass(value):
                    return dataclasses.replace(value, ok=True)
                value.ok = True
                return value
            return value
        return wrapper

    checks = module.CHECKS
    if isinstance(checks, dict):
        module.CHECKS = {k: wrap(k, v) for k, v in checks.items()}
        declared = list(checks)
    else:
        module.CHECKS = [(k, wrap(k, v)) for k, v in checks]
        declared = [k for k, _ in checks]
    main_error = None
    sys.argv = [str(verifier), "--workspace", str(workspace), "--output", str(native_output)]
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module.main()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        main_error = {"type": type(exc).__name__, "message": str(exc)[:4000], "traceback": traceback.format_exc()[-4000:]}
    rows = []
    for index, cid in enumerate(order, start=1):
        row = outcomes.get(cid) or {"id": cid, "status": "NOT_RUN", "raw": "not_invoked"}
        rows.append({**row, "index": index})
    return {"version": VERSION, "declared_check_order": declared, "checklist_order": order,
            "order_matches": declared == order, "criteria": rows, "main_error": main_error,
            "seconds": round(time.monotonic() - started, 4),
            "stdout_tail": stdout.getvalue()[-4000:], "stderr_tail": stderr.getvalue()[-4000:]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-check-timeout", type=float, default=900.0)
    args = parser.parse_args()
    result = run(args.verifier, args.workspace, args.native_output, args.per_check_timeout,
                 progress=args.output.with_name("criteria-progress.json"))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
