"""Private MettleBench grading in disposable containers restored from exported W/R.

Two passes run in separate disposable containers created from the frozen task
image, each restored from the same exported W (and captured R when supplied):

* native: the unmodified upstream command
  ``python tasks/<id>/verifier/verifier.py --workspace /workspace --output ...``
  producing the native verdict (prefix score and first-failure feedback);
* criteria: ``criteria_runner.py`` observes every criterion in native order.

The private verifier bundle is streamed into ``/mettle-grader`` of the
disposable container only; it never touches the live worker, its image or W.
Grading is on an isolated copy; the live worker is never mounted.
"""
from __future__ import annotations

import concurrent.futures
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

GRADER_ROOT = "/mettle-grader"
NATIVE_TIMEOUT = 1500
CRITERIA_TIMEOUT = 2400
PER_CHECK_TIMEOUT = 900
LABEL = "trajectory-retention.lane=mettlebench"

_SCRIPT = r"""
set -u
umask 077
G={root}
rm -rf "$G" && mkdir -p "$G" && cd "$G" && tar -xf - >/dev/null
cd /workspace
start=$(python3 -c 'import time;print(time.monotonic())')
if [ "$MODE" = native ]; then
  timeout -s KILL {native_timeout} python "$G/tasks/$TID/verifier/verifier.py" --workspace /workspace --output "$G/out/native.json" > "$G/out/native.log" 2>&1
else
  timeout -s KILL {criteria_timeout} python "$G/criteria_runner.py" --verifier "$G/tasks/$TID/verifier/verifier.py" \
    --workspace /workspace --native-output "$G/out/continue-native.json" --output "$G/out/criteria.json" \
    --per-check-timeout {per_check} > "$G/out/criteria.log" 2>&1
fi
rc=$?
end=$(python3 -c 'import time;print(time.monotonic())')
python3 -c 'import json,sys;print(json.dumps({{"returncode":int(sys.argv[1]),"seconds":float(sys.argv[3])-float(sys.argv[2])}}))' "$rc" "$start" "$end" > "$G/out/meta.json"
cd "$G/out" && exec tar -cf - .
"""


def grader_bundle(package: Path, task_upstream_id: str, runner_source: Path) -> bytes:
    """Tar of the private verifier tree for one task, plus the criteria runner."""
    buf = io.BytesIO()
    private = package / "private"
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        def add(path: Path, arcname: str):
            info = tf.gettarinfo(str(path), arcname=arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 0
            if path.is_file():
                with open(path, "rb") as handle:
                    tf.addfile(info, handle)
            else:
                tf.addfile(info)
        add(private / "upstream/tasks/_verifier_v2.py", "tasks/_verifier_v2.py")
        task_root = private / "upstream/tasks" / task_upstream_id
        for path in sorted(task_root.rglob("*")):
            add(path, "tasks/" + task_upstream_id + "/" + path.relative_to(task_root).as_posix())
        overlay = private / "adaptation/overlay"
        if overlay.is_dir():
            for path in sorted(overlay.rglob("*")):
                add(path, path.relative_to(overlay).as_posix())
        add(runner_source, "criteria_runner.py")
        info = tarfile.TarInfo("out")
        info.type = tarfile.DIRTYPE
        info.mode = 0o700
        tf.addfile(info)
    return buf.getvalue()


def _validate_runtime(runtime: RuntimeExport, image_id: str, workspace: Path) -> None:
    if runtime.base_image_id != image_id:
        raise ValueError("Submitted R image differs from frozen task image")
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


def _one_pass(mode: str, *, image_id: str, upstream_id: str, bundle: bytes, packed: bytes,
              fidelity: dict, runtime: RuntimeExport | None, snap: dict) -> dict[str, Any]:
    env = DockerWorkerEnvironment(image=image_id, network="none", name_prefix=f"tr-mettle-grade-{mode}",
                                  env=dict(runtime.environment) if runtime else {}, memory="4g", cpus="2")
    receipt: dict[str, Any] = {"mode": mode}
    try:
        if env.image_id != image_id:
            raise ValueError("Grading image differs from frozen task image")
        if runtime is not None and (snap.get("changed") or snap.get("deleted") or snap.get("archive")):
            receipt["runtime_restore"] = env.restore_runtime(snap)
        elif runtime is not None:
            # Captured R had no filesystem change outside W: the fresh container from the same
            # image already equals it; restoring an empty diff is a no-op (not re-measured).
            receipt["runtime_restore"] = {"validated": None, "empty_capture": True, "processes_restored": False}
        receipt["workspace_restore"] = env.restore_workspace(packed, fidelity)
        receipt["binding"] = env.binding()
        script = _SCRIPT.format(root=GRADER_ROOT, native_timeout=NATIVE_TIMEOUT, criteria_timeout=CRITERIA_TIMEOUT,
                                per_check=PER_CHECK_TIMEOUT)
        command = ["docker", "exec", "-i", "--user", "0:0", "-w", "/workspace", "-e", f"TID={upstream_id}",
                   "-e", f"MODE={mode}", "-e", "PYTHONDONTWRITEBYTECODE=1"]
        for key, value in env.env.items():
            command += ["-e", f"{key}={value}"]
        command += [env.container_id, "bash", "-c", script]
        receipt["argv"] = command[:-1] + ["<grading script>"]
        started = time.monotonic()
        proc = subprocess.run(command, input=bundle, capture_output=True,
                              timeout=max(NATIVE_TIMEOUT, CRITERIA_TIMEOUT) + 120)
        receipt["exec_seconds"] = time.monotonic() - started
        receipt["exec_returncode"] = proc.returncode
        receipt["exec_stderr"] = proc.stderr.decode(errors="replace")[-4000:]
        if proc.returncode != 0:
            raise RuntimeError(f"grading exec transport failed ({proc.returncode})")
        files = {}
        with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as tf:
            for member in tf.getmembers():
                if member.isreg():
                    files[Path(member.name).name] = tf.extractfile(member).read()
        receipt["files"] = sorted(files)
        receipt["meta"] = json.loads(files["meta.json"])
        return {"receipt": receipt, "files": files}
    finally:
        env.close()


def run_private_grade(*, package: Path, upstream_id: str, image_id: str, workspace: Path, output: Path,
                      runtime: RuntimeExport | None, runner_source: Path) -> dict[str, Any]:
    """Grade an already isolated host copy of W; write private receipts into ``output``."""
    workspace = Path(workspace).resolve()
    if Path(output).resolve().is_relative_to(workspace):
        raise ValueError("Private output in W")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[str, Any] = {"infrastructure_errors": [], "baseline_runtime": runtime is None}
    started = time.monotonic()
    try:
        snap = {}
        if runtime is not None:
            _validate_runtime(runtime, image_id, workspace)
            snap = dict(runtime.snapshot)
            snap["archive"] = runtime.archive.read_bytes() if runtime.archive else None
            if snap.get("foreign_processes"):
                # Live processes are never restorable; grade the captured filesystem W/R and say so.
                result["runtime_restore_scope"] = "filesystem only; live worker processes not restored"
                result["foreign_processes"] = snap["foreign_processes"]
                snap["runtime_class"] = RuntimeClass.CAPTURED.value
        owner = tuple(snap.get("workspace_default_owner", (0, 0)))
        packed = pack_directory(workspace, ownership=snap.get("workspace_fidelity_manifest"), default_owner=owner)
        fidelity = fidelity_manifest_from_tar(packed)
        if runtime is not None and fidelity != snap["workspace_fidelity_manifest"]:
            raise ValueError("W export differs from paired execution fidelity")
        bundle = grader_bundle(package, upstream_id, runner_source)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = {mode: pool.submit(_one_pass, mode, image_id=image_id, upstream_id=upstream_id, bundle=bundle,
                                         packed=packed, fidelity=fidelity, runtime=runtime, snap=snap)
                       for mode in ("native", "criteria")}
            passes = {mode: f.result() for mode, f in futures.items()}
        for mode, value in passes.items():
            for name, data in value["files"].items():
                (output / f"{mode}-{name}").write_bytes(data)
        result["receipts"] = {m: v["receipt"] for m, v in passes.items()}
        native_meta = passes["native"]["receipt"]["meta"]
        criteria_meta = passes["criteria"]["receipt"]["meta"]
        result["native_seconds"] = native_meta["seconds"]
        result["criteria_seconds"] = criteria_meta["seconds"]
        if "native.json" in passes["native"]["files"]:
            result["native"] = json.loads(passes["native"]["files"]["native.json"])
        elif native_meta["returncode"] in (124, 137):
            result["native"] = {"status": "deadline_killed", "first_failure": None, "metrics": {}}
        else:
            raise RuntimeError(f"native verifier produced no result (rc={native_meta['returncode']})")
        criteria_files = passes["criteria"]["files"]
        if "criteria.json" in criteria_files:
            result["criteria"] = json.loads(criteria_files["criteria.json"])
        elif criteria_meta["returncode"] in (124, 137):
            # Outer deadline killed the observation: attribute to the task, keep observed prefix.
            progress = (json.loads(criteria_files["criteria-progress.json"])
                        if "criteria-progress.json" in criteria_files else {"criteria": []})
            result["criteria"] = {"criteria": progress["criteria"], "killed": True, "main_error": None,
                                  "kill_returncode": criteria_meta["returncode"]}
        else:
            raise RuntimeError(f"criteria runner produced no result (rc={criteria_meta['returncode']})")
    except Exception as exc:  # recorded as INFRA_ERROR by the adapter
        result["infrastructure_errors"].append({"type": type(exc).__name__, "message": str(exc)[:4000]})
    finally:
        result["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps({k: v for k, v in result.items()
                                                         if k not in {"native", "criteria"}}, indent=2, default=str) + "\n")
    return result
