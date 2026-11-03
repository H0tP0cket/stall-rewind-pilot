"""Explicit missing checkpoints for unsupported live background state (Q8).

Opt-in technical continuations only. No archive is certified, no grade is
invented, and submission/initial/terminal capture failures remain fatal.
"""
from __future__ import annotations

POLICY = "background-capture-gap/v1"
ERRORS = {
    "Live Unix socket state is not restorable by this filesystem adapter",
    "Unix socket file capture requires no non-harness processes",
}
CHANGING = (
    "workspace changed during capture or archive incomplete:",
    "runtime archive mismatch:",
    "Runtime changed during archive capture:",
    "Runtime changed during socket-file capture",
)


def diagnose(env, errors):
    # Only this sealed adapter's known limitations. Do not soften storage,
    # transport, permission, unknown capture, or arbitrary adapter failures.
    from ..tasks.terminalbench_v4_pilot_training.worker_runtime import OfflineWorkerEnvironment
    if type(env) is not OfflineWorkerEnvironment or not errors:
        return None
    if not all(e in ERRORS or e.startswith(CHANGING) for e in errors):
        return None
    processes = env.process_table()
    foreign = [p for p in processes if env._process_identity(p) not in env._harness_processes]
    # Content mismatches require current evidence of a background process.
    # The socket-specific errors already establish unsupported state at capture.
    if any(e not in ERRORS for e in errors) and not foreign:
        return None
    return {"processes": processes, "foreign_processes": foreign,
            "diagnostic_only": True, "snapshot_validated": False}
