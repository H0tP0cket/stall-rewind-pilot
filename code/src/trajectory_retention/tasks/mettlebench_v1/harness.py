"""Admission tooling: run a private script in a real worker container, capture W/R, grade.

Used for reference development, controls and qualification. Never used by the
mining runner. Scripts are copied to a private temporary path outside /workspace
and deleted before capture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import time
import uuid
from typing import Any

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from trajectory_retention.execution.archives import extract_directory, pack_directory
from trajectory_retention.execution.environment import DockerWorkerEnvironment, fidelity_manifest_from_tar
from trajectory_retention.mining.runtime_support import bind_runtime_support

MODULE_DIR = Path(__file__).resolve().parent
IMAGE_LOCK = MODULE_DIR / "environment/image-lock.json"
STEP_MARKER = "# @step"


def image_id(upstream_id: str | None = None) -> str:
    lock = json.loads(IMAGE_LOCK.read_text())
    variant = lock.get("task_variants", {}).get(upstream_id or "", "default")
    return lock["images"][variant]["image_id"]


def split_steps(script: str) -> tuple[str, list[str]]:
    """Header (always run) and the ordered ``# @step`` sections of a witness script."""
    header, steps, current = [], [], None
    for line in script.splitlines(keepends=True):
        if line.startswith(STEP_MARKER):
            current = [line]
            steps.append(current)
        elif current is None:
            header.append(line)
        else:
            current.append(line)
    return "".join(header), ["".join(s) for s in steps]


def prefix_script(script: str, steps: int | None) -> str:
    if steps is None:
        return script
    header, parts = split_steps(script)
    if steps > len(parts):
        raise ValueError(f"Script has only {len(parts)} steps")
    return header + "".join(parts[:steps])


def start_worker(adapter, *, network: str = "none") -> tuple[DockerWorkerEnvironment, Path]:
    env = DockerWorkerEnvironment(image=adapter.manifest["environment_image_id"], network=network,
                                  name_prefix="tr-mettle-admission")
    host = Path(tempfile.mkdtemp(prefix="mettle-start-"))
    ws = host / "W"
    try:
        adapter.materialize(ws)
        data = pack_directory(ws, default_owner=(env.worker_uid, env.worker_gid))
        env.restore_workspace(data, fidelity_manifest_from_tar(data))
    except BaseException:
        env.close()
        raise
    finally:
        shutil.rmtree(host, ignore_errors=True)
    return env, ws


def run_private_script(env: DockerWorkerEnvironment, script: str, *, timeout: float = 1800) -> dict[str, Any]:
    token = uuid.uuid4().hex[:12]
    private = f"/tmp/.mettle-witness-{token}"
    env._exec(f"mkdir -p {private} && chmod 700 {private} && cat > {private}/solve.sh", input_bytes=script.encode())
    started = time.monotonic()
    proc = env._exec(f"cd /workspace && timeout -s KILL {int(timeout)} bash {private}/solve.sh /workspace",
                     timeout=timeout + 60, check=False)
    seconds = time.monotonic() - started
    env._exec(f"rm -rf {private}")
    return {"returncode": proc.returncode, "seconds": seconds,
            "stdout_tail": proc.stdout.decode(errors="replace")[-6000:],
            "stderr_tail": proc.stderr.decode(errors="replace")[-6000:]}


def capture(env: DockerWorkerEnvironment) -> tuple[bytes, dict, dict]:
    w_data, w_fidelity = env.capture_workspace()
    runtime = env.capture_runtime()
    return w_data, w_fidelity, runtime


def grade_capture(adapter, env: DockerWorkerEnvironment, w_data: bytes, runtime: dict | None, out: Path):
    """Runner-equivalent isolated grade of a capture (W extracted on the host, R exported read-only)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ws = out / "workspace"
    extract_directory(w_data, ws)
    export = None
    if runtime is not None:
        record = adapter.describe()
        runtime = bind_runtime_support(record, adapter.stage("task"), runtime)
        archive_path = archive_sha = None
        if runtime.get("archive"):
            archive_path = out / "runtime.tar"
            archive_path.write_bytes(runtime["archive"])
            archive_sha = hashlib.sha256(runtime["archive"]).hexdigest()
        export = RuntimeExport(base_image_id=env.image_id, runtime_class=RuntimeClass(runtime["runtime_class"]),
            snapshot={"rule": runtime["rule"], "entries": runtime["entries"], "changed": runtime["changed"],
                      "deleted": runtime["deleted"], "processes": runtime["processes"],
                      "foreign_processes": runtime["foreign_processes"], "coverage": runtime["coverage"],
                      "runtime_class": runtime["runtime_class"], "runtime_support": runtime["runtime_support"],
                      "irrelevance_evidence": runtime.get("irrelevance_evidence"), "mounts": runtime.get("mounts"),
                      "initial_mounts": runtime.get("initial_mounts"),
                      "mount_state_changed": runtime.get("mount_state_changed"),
                      "workspace_fidelity_manifest": fidelity_manifest_from_tar(w_data),
                      "workspace_default_owner": [env.worker_uid, env.worker_gid], "process_restore_supported": False},
            archive=archive_path, archive_sha256=archive_sha,
            workspace_snapshot_sha256=hashlib.sha256(w_data).hexdigest(), environment=dict(env.env))
    started = time.monotonic()
    evaluation = adapter.grade(ws, stage_id="task", private_output_dir=out / "private", runtime=export)
    elapsed = time.monotonic() - started
    shutil.rmtree(ws, ignore_errors=True)
    return evaluation, elapsed


def summarize(evaluation) -> dict[str, Any]:
    return {"valid": evaluation.valid, "score": evaluation.score, "full_solve": evaluation.full_solve,
            "native": evaluation.details.get("native"), "consistency": evaluation.details.get("native_consistency"),
            "statuses": {c.check_id.removeprefix("criterion::"): c.status.value for c in evaluation.checks},
            "infrastructure_errors": evaluation.details.get("infrastructure_errors"),
            "grade_seconds": evaluation.details.get("grade_seconds")}


def statuses(evaluation) -> list[tuple[str, str, str | None]]:
    return [(c.check_id, c.status.value, c.raw_status) for c in evaluation.checks]


def run_and_grade(adapter, script: str | None, out: Path, *, steps: int | None = None, timeout: float = 1800,
                  regrade: bool = False) -> dict[str, Any]:
    env, _ = start_worker(adapter)
    try:
        report: dict[str, Any] = {"steps": steps}
        if script is not None:
            report["script"] = run_private_script(env, prefix_script(script, steps), timeout=timeout)
        w_data, _fid, runtime = capture(env)
        report["runtime_class_measured"] = runtime["runtime_class"]
        report["foreign_processes"] = len(runtime["foreign_processes"])
        evaluation, elapsed = grade_capture(adapter, env, w_data, runtime, out / "grade1")
        report["grade"] = summarize(evaluation)
        report["grade_wall_seconds"] = elapsed
        report["w_sha256"] = hashlib.sha256(w_data).hexdigest()
        if regrade:
            second, elapsed2 = grade_capture(adapter, env, w_data, runtime, out / "grade2")
            report["regrade"] = summarize(second)
            report["regrade_wall_seconds"] = elapsed2
            report["deterministic"] = statuses(evaluation) == statuses(second) and evaluation.score == second.score
        report["_evaluation"] = evaluation
        return report
    finally:
        env.close()
