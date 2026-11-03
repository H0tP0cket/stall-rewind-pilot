"""Private Terminal-Bench grading in a disposable container restored from the submitted W/R.

Native Terminal-Bench grades the live agent container after the agent stops:
Harbor copies ``tests/`` to ``/tests`` and runs ``tests/test.sh`` there as root
from the task's working directory (``/app``). Here (disclosed adaptation) the
submitted state is captured W (``/workspace`` = ``/app``) plus captured
filesystem R (``docker diff`` outside W); both are restored into a fresh
container of the grader image (worker image + pre-built verifier environment),
with network none, and the native pytest command is run there. Live worker
processes are not restored (the runner records them; filesystem only).

R entries under grader-reserved roots (``/tb-verifier``, ``/tb-grader``,
``/tests``, ``/logs``) are dropped before restoration and listed in the
receipt: grading material is installed after restoration and must not be
shadowed by worker-created files there.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from typing import Any

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from trajectory_retention.execution.archives import pack_directory
from trajectory_retention.execution.environment import DockerWorkerEnvironment, RUNTIME_RULE_VERSION, fidelity_manifest_from_tar

from .environment import NATIVE_WORKDIR, RESERVED_ROOTS, VERIFIER_ROOT

GRADER_DIR = Path(__file__).with_name("grader")
GRADER_FILES = ("tb_recorder.py", "run_native.py", "pytest.ini")
TRANSPORT_GRACE = 180

_SCRIPT = r"""
set -u
mkdir -p /logs/verifier
cd / && tar -xf - --no-same-owner >/dev/null || exit 90
chmod 700 /tb-grader
cd {workdir} || {{ echo '{{"returncode": 91, "seconds": 0, "cd_failed": true}}' > /tb-grader/out/meta.json; cd /tb-grader/out && exec tar -cf - .; }}
start=$(date +%s.%N)
timeout -s KILL {timeout} {python} -I /tb-grader/run_native.py {flags} > /tb-grader/out/pytest.log 2>&1
rc=$?
end=$(date +%s.%N)
echo "{{\"returncode\": $rc, \"start\": $start, \"end\": $end}}" > /tb-grader/out/meta.json
cp /logs/verifier/ctrf.json /tb-grader/out/ctrf.json 2>/dev/null || true
cd /tb-grader/out && exec tar -cf - .
"""


def _image_env(image_id: str) -> list[str]:
    info = json.loads(subprocess.run(["docker", "image", "inspect", image_id], capture_output=True, check=True).stdout)[0]
    return info.get("Config", {}).get("Env") or []


def _reserved(path: str) -> bool:
    p = "/" + path.lstrip("/")
    return any(p == root or p.startswith(root + "/") for root in RESERVED_ROOTS)


def filter_runtime(snap: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop captured R entries under grader-reserved roots (listed, never silently)."""
    dropped = sorted({"/" + k.lstrip("/") for k in snap.get("entries", {}) if _reserved(k)}
                     | {p for p in snap.get("deleted", []) if _reserved(p)})
    if not dropped:
        return snap, []
    out = dict(snap)
    out["entries"] = {k: v for k, v in snap["entries"].items() if not _reserved(k)}
    out["changed"] = [p for p in snap.get("changed", []) if not _reserved(p)]
    out["deleted"] = [p for p in snap.get("deleted", []) if not _reserved(p)]
    if snap.get("archive"):
        buf = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(snap["archive"])) as src, \
                tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as dst:
            for member in src.getmembers():
                if _reserved(member.name):
                    continue
                dst.addfile(member, src.extractfile(member) if member.isreg() else None)
        out["archive"] = buf.getvalue() if out["entries"] else None
    return out, dropped


def grader_bundle(tests_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        def add(path: Path, arcname: str, mode: int | None = None):
            info = tf.gettarinfo(str(path), arcname=arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            if mode is not None:
                info.mode = mode
            if path.is_file():
                with open(path, "rb") as handle:
                    tf.addfile(info, handle)
            else:
                tf.addfile(info)
        for directory, arc in ((tests_dir, "tests"),):
            add(directory, arc, 0o755)
            for path in sorted(Path(directory).rglob("*")):
                if path.name == "test.sh" and path.parent == Path(directory):
                    continue  # the native driver script is replaced by run_native.py (recorded)
                add(path, arc + "/" + path.relative_to(directory).as_posix())
        info = tarfile.TarInfo("tb-grader"); info.type = tarfile.DIRTYPE; info.mode = 0o700
        tf.addfile(info)
        for name in GRADER_FILES:
            add(GRADER_DIR / name, "tb-grader/" + name, 0o644)
        info = tarfile.TarInfo("tb-grader/out"); info.type = tarfile.DIRTYPE; info.mode = 0o700
        tf.addfile(info)
    return buf.getvalue()


def validate_runtime(runtime: RuntimeExport, worker_image_id: str, workspace: Path) -> None:
    if runtime.base_image_id != worker_image_id:
        raise ValueError("Submitted R image differs from the frozen worker image")
    if runtime.snapshot.get("mount_state_changed"):
        raise ValueError("Unsupported mounted runtime state")
    if runtime.runtime_class not in (RuntimeClass.CAPTURED, RuntimeClass.UNSUPPORTED):
        raise ValueError("Unexpected runtime class")
    if runtime.runtime_class == RuntimeClass.UNSUPPORTED and not runtime.snapshot.get("foreign_processes"):
        raise ValueError("Unsupported runtime state other than live processes")
    if runtime.snapshot.get("rule") != RUNTIME_RULE_VERSION or "entries" not in runtime.snapshot:
        raise ValueError("Unsupported runtime capture rule/fidelity")
    if not isinstance(runtime.snapshot.get("workspace_fidelity_manifest"), dict):
        raise ValueError("Paired W fidelity absent")
    if runtime.snapshot.get("changed") and runtime.archive is None:
        raise ValueError("Changed R archive absent")
    if runtime.archive is not None and runtime.archive.resolve().is_relative_to(workspace):
        raise ValueError("Private R export inside W")


def run_private_grade(*, grader_image_id: str, worker_image_id: str, tests_dir: Path, workspace: Path, output: Path,
                      runtime: RuntimeExport | None, timeout: int, pytest_flags: list[str], cpus: str,
                      memory: str) -> dict[str, Any]:
    """Grade an isolated host copy of W (+ captured R); write private receipts into ``output``."""
    workspace = Path(workspace).resolve()
    output = Path(output).resolve()
    if output.is_relative_to(workspace):
        raise ValueError("Private output in W")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[str, Any] = {"infrastructure_errors": [], "baseline_runtime": runtime is None, "files": {}}
    started = time.monotonic()
    env = None
    try:
        snap: dict[str, Any] = {}
        if runtime is not None:
            validate_runtime(runtime, worker_image_id, workspace)
            snap = dict(runtime.snapshot)
            snap["archive"] = runtime.archive.read_bytes() if runtime.archive else None
            if snap.get("foreign_processes"):
                result["runtime_restore_scope"] = "filesystem only; live worker processes not restored"
                result["foreign_processes"] = snap["foreign_processes"]
                snap["runtime_class"] = RuntimeClass.CAPTURED.value
            snap, dropped = filter_runtime(snap)
            result["reserved_runtime_paths_dropped"] = dropped
        owner = tuple(snap.get("workspace_default_owner", (0, 0)))
        packed = pack_directory(workspace, ownership=snap.get("workspace_fidelity_manifest"), default_owner=owner)
        fidelity = fidelity_manifest_from_tar(packed)
        if runtime is not None and fidelity != runtime.snapshot["workspace_fidelity_manifest"]:
            raise ValueError("W export differs from paired execution fidelity")
        env = DockerWorkerEnvironment(image=grader_image_id, network="none", name_prefix="tr-tb-grade",
                                      env=dict(runtime.environment) if runtime else {}, memory=memory, cpus=cpus)
        receipt: dict[str, Any] = {}
        if env.image_id != grader_image_id:
            raise ValueError("Grading image differs from the frozen grader image")
        if runtime is not None and (snap.get("changed") or snap.get("deleted") or snap.get("archive")):
            receipt["runtime_restore"] = env.restore_runtime(snap)
        elif runtime is not None:
            receipt["runtime_restore"] = {"validated": None, "empty_capture": True, "processes_restored": False}
        receipt["workspace_restore"] = env.restore_workspace(packed, fidelity)
        receipt["binding"] = env.binding()
        script = _SCRIPT.format(workdir=NATIVE_WORKDIR, timeout=int(timeout), python=VERIFIER_ROOT + "/venv/bin/python",
                                flags=shlex.quote(json.dumps(list(pytest_flags))))
        command = ["docker", "exec", "-i", "--user", "0:0", "-w", "/", "-e", "PYTHONDONTWRITEBYTECODE=1"]
        for key, value in env.env.items():
            command += ["-e", f"{key}={value}"]
        # uvx runs the tool with its environment's bin directory first on PATH (measured with uv 0.12.11;
        # VIRTUAL_ENV is not set), so a test's subprocess "python" is the verifier environment's interpreter.
        image_path = next((e.split("=", 1)[1] for e in _image_env(grader_image_id) if e.startswith("PATH=")),
                          "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        command += ["-e", f"PATH={VERIFIER_ROOT}/venv/bin:{image_path}"]
        command += [env.container_id, "bash", "-c", script]
        receipt["argv"] = command[:-1] + ["<grading script>"]
        exec_started = time.monotonic()
        proc = subprocess.run(command, input=grader_bundle(tests_dir), capture_output=True,
                              timeout=int(timeout) + TRANSPORT_GRACE)
        receipt["exec_seconds"] = time.monotonic() - exec_started
        receipt["exec_returncode"] = proc.returncode
        receipt["exec_stderr"] = proc.stderr.decode(errors="replace")[-4000:]
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f"grading exec transport failed ({proc.returncode})")
        files = {}
        with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
            for member in tf.getmembers():
                if member.isreg():
                    files[Path(member.name).name] = tf.extractfile(member).read()
        for name, data in files.items():
            (output / name).write_bytes(data)
        meta = json.loads(files["meta.json"])
        if "start" in meta:
            meta["seconds"] = float(meta["end"]) - float(meta["start"])
        receipt["meta"] = meta
        result["receipt"] = receipt
        result["pytest_seconds"] = meta.get("seconds")
        result["returncode"] = meta["returncode"]
        result["timed_out"] = meta["returncode"] in (137, -9) and (meta.get("seconds") or 0) >= timeout - 2
        result["exit"] = json.loads(files["exit.json"]) if "exit.json" in files else None
        result["files"] = sorted(files)
    except Exception as exc:  # recorded as INFRA_ERROR by the adapter
        result["infrastructure_errors"].append({"type": type(exc).__name__, "message": str(exc)[:4000]})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # pragma: no cover - cleanup failure is recorded
                result["infrastructure_errors"].append({"type": "CleanupError", "message": str(exc)[:1000]})
        result["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
