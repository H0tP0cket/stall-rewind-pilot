"""Private TB4 grading: Harbor's separate verifier, reproduced offline on a captured W and R (terminalbench-v4).

Native TB4 (Harbor, separate verifier mode, ``source.HARBOR_FILES`` at ``source.HARBOR_REVISION``): after the agent
stops, Harbor copies each path declared in task.toml ``artifacts`` (plus its conventional publish dir ``/logs/artifacts``,
prepended) out of the agent container with ``docker cp``, starts a fresh container of the verifier image built from
``tests/Dockerfile`` (which bakes the hidden tests at /tests), empties ``/logs/verifier`` (mkdir -p, chmod 777), uploads
each artifact to its original path (a file: ``mkdir -p`` + ``chmod 777`` of its parent, then ``docker cp``; a directory:
emptied, then its contents copied), runs ``chmod +x /tests/test.sh`` as root and then
``bash -c "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"`` as the image's default user from the image's
WORKDIR, and reads the reward from /logs/verifier/reward.json or reward.txt.

Here the agent container is the submitted state: captured W (``/workspace`` = ``/app``) and captured filesystem R are
restored into a fresh worker-image container (network none) and the artifacts are copied out of it exactly as Harbor
copies them; everything after that follows Harbor on a fresh verifier container with network none and the task's cpus
and memory. Additions (disclosed): a researcher-only pytest recorder (``grader/tb_recorder.py``) is loaded into the
task's own pytest run through ``PYTEST_ADDOPTS`` and made importable through ``PYTHONPATH`` and, for a test.sh that sets
its own PYTHONPATH (e.g. session-window-debug's ``PYTHONPATH=/app python3 -m pytest``), Python's user site directory
(``PYTHONUSERBASE``); it removes all three again on import, so the per-test
statuses carry pytest node ids (the native ctrf.json drops parametrize ids); the run is bounded by the task's verifier
timeout. Live worker processes are not restored. Both containers are removed by their own IDs.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import tarfile
import time
import uuid
from typing import Any

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from trajectory_retention.execution.archives import pack_directory
from trajectory_retention.execution.environment import (DockerWorkerEnvironment, RUNTIME_RULE_VERSION,
                                                        fidelity_manifest_from_tar)

from .source import remove_tree
from .environment import LABEL, NATIVE_WORKDIR

GRADER_DIR = Path(__file__).with_name("grader")
GRADER_FILES = ("tb_recorder.py",)
TRANSPORT_GRACE = 180
CONVENTION_DIR = "/logs/artifacts"
VERIFIER_DIR = "/logs/verifier"
TEST_SCRIPT = "/tests/test.sh"
TEST_STDOUT = VERIFIER_DIR + "/test-stdout.txt"
RECORDER_ENV = {"PYTHONPATH": "/tb-grader", "PYTHONUSERBASE": "/tb-grader/user", "PYTEST_ADDOPTS": "-p tb_recorder"}
# The recorder's copies under PYTHONUSERBASE, at the user site directory of each CPython the verifier image may run
# pytest with (lib/pythonX.Y[t]/site-packages; a venv or -s/-I run ignores the user site and relies on PYTHONPATH).
USER_SITE_VERSIONS = tuple(f"3.{minor}" for minor in range(6, 16)) + tuple(f"3.{minor}t" for minor in range(13, 16))
OWNER_LABEL = "trajectory-retention.owner"

_RUN = r"""
start=$(date +%s.%N)
timeout -s KILL {timeout} bash -c {command}
rc=$?
end=$(date +%s.%N)
echo "{{\"returncode\": $rc, \"start\": $start, \"end\": $end}}" > /tb-grader/out/meta.json
"""


def harbor_command() -> str:
    """Harbor's verifier command (build_execution_command with a stdout path)."""
    return f"({TEST_SCRIPT}) > {TEST_STDOUT} 2>&1"


def worker_path(path: str) -> str:
    """Where a declared artifact path lives in the worker image (/app is a symlink to /workspace there)."""
    if path == NATIVE_WORKDIR or path.startswith(NATIVE_WORKDIR + "/"):
        return "/workspace" + path[len(NATIVE_WORKDIR):]
    return path


def _docker(*args: str, input_bytes: bytes | None = None, timeout: float = 600, check: bool = True
            ) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], input=input_bytes, capture_output=True, timeout=timeout)
    if check and proc.returncode:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed ({proc.returncode}): "
                           f"{proc.stderr.decode(errors='replace')[-2000:]}")
    return proc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grader_bundle() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, mode in (("tb-grader", 0o700), ("tb-grader/out", 0o700)):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, mode
            tf.addfile(info)
        sites = [f"tb-grader/user/lib/python{version}/site-packages" for version in USER_SITE_VERSIONS]
        for name in ("tb-grader/user", "tb-grader/user/lib",
                     *(d for site in sites for d in (site.rsplit("/", 1)[0], site))):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o700
            tf.addfile(info)
        for name in GRADER_FILES:
            data = (GRADER_DIR / name).read_bytes()
            for target in ("tb-grader", *sites):
                info = tarfile.TarInfo(target + "/" + name)
                info.size, info.mode = len(data), 0o644
                tf.addfile(info, io.BytesIO(data))
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


def pack_submitted(workspace: Path, ownership: dict, owner: tuple[int, int]) -> tuple[bytes, list[str]]:
    """Pack the host copy of W with its recorded owners and modes.

    ``execution.archives.extract_directory`` gives a container file its owner cannot read (e.g. rs-archive-clone's
    root-owned 0111 ``/app/artifacts/archive-tool``) owner-read on the host copy only, which ``pack_directory`` does not
    undo. This undoes exactly that projection: a regular file whose recorded mode lacks owner-read and whose host mode
    is the recorded mode's transport projection plus owner-read gets the recorded mode back. Any other mode difference
    is kept, so it still fails the fidelity check against the paired capture."""
    packed = pack_directory(workspace, ownership=ownership, default_owner=owner)
    out, restored = io.BytesIO(), []
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:") as src, \
            tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as dst:
        for member in src.getmembers():
            name = member.name[2:] if member.name.startswith("./") else member.name
            previous = (ownership or {}).get(name) or {}
            mode = previous.get("mode")
            if (member.isreg() and mode is not None and not mode & stat.S_IRUSR
                    and (member.mode & 0o7777) == ((mode & 0o755) | stat.S_IRUSR)):
                member.mode = mode
                restored.append(name)
            dst.addfile(member, src.extractfile(member) if member.isreg() else None)
    return (out.getvalue() if restored else packed), restored


class VerifierContainer:
    """A fresh container of the verifier image, kept alive as Harbor keeps it (sh -c 'sleep infinity')."""

    def __init__(self, image_id: str, *, cpus: str, memory: str):
        self.name = f"tr-tbv4-verify-{uuid.uuid4().hex[:10]}"
        self.owner = uuid.uuid4().hex
        self.container_id = None
        inspect = json.loads(_docker("image", "inspect", image_id).stdout)[0]
        self.image_id = inspect["Id"]
        self.image_user = inspect.get("Config", {}).get("User") or ""
        self.image_workdir = inspect.get("Config", {}).get("WorkingDir") or "/"
        self.image_env = inspect.get("Config", {}).get("Env") or []
        self.argv = ["run", "-d", "--name", self.name, "--label", LABEL, "--label", f"{OWNER_LABEL}={self.owner}",
                     "--network", "none", "--cpus", cpus, "--memory", memory, self.image_id, "sh", "-c", "sleep infinity"]
        self.container_id = _docker(*self.argv, timeout=120).stdout.decode().strip()
        if not self.container_id:
            raise RuntimeError("verifier container launch returned no ID")

    def root(self, command: str, *, input_bytes: bytes | None = None, timeout: float = 600, check: bool = True
             ) -> subprocess.CompletedProcess:
        return _docker("exec", "-i", "--user", "0:0", self.container_id, "bash", "-c", command,
                       input_bytes=input_bytes, timeout=timeout, check=check)

    def binding(self) -> dict:
        # The network as the daemon reports it for the live container, not as requested.
        live = json.loads(_docker("inspect", self.container_id).stdout)[0]
        return {"container_id": self.container_id, "name": self.name, "image_id": live["Image"],
                "network": live["HostConfig"]["NetworkMode"],
                "attached_networks": sorted((live.get("NetworkSettings") or {}).get("Networks") or {}),
                "image_user": self.image_user or "root (image default)", "workdir": self.image_workdir,
                "argv": self.argv}

    def close(self) -> None:
        if self.container_id:
            proc = _docker("rm", "-f", self.container_id, timeout=60, check=False)
            if proc.returncode and b"no such container" not in proc.stderr.lower():
                raise RuntimeError(f"verifier container cleanup failed: {proc.stderr.decode(errors='replace')}")
            self.container_id = None


def ensure_dirs_command(path: str) -> str:
    q = shlex.quote(path)
    return f"mkdir -p {q} && chmod 777 {q}"


def empty_dirs_command(path: str) -> str:
    q = shlex.quote(path)
    return (f"if [ -L {q} ] || {{ [ -e {q} ] && [ ! -d {q} ]; }}; then rm -rf {q}; fi && mkdir -p {q} && "
            f"find {q} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + && chmod 777 {q}")


def copy_out(src: DockerWorkerEnvironment, artifact: str, host: Path) -> dict:
    """Harbor's download of one artifact from the (restored) agent container, best effort."""
    path = worker_path(artifact)
    q = shlex.quote(path)
    probe = _docker("exec", "--user", "0:0", src.container_id, "sh", "-c",
                    f"if [ -d {q} ]; then echo dir; elif [ -e {q} ] || [ -L {q} ]; then echo file; else echo missing; fi",
                    timeout=120).stdout.decode().strip()
    row: dict[str, Any] = {"source": artifact, "worker_path": path, "kind": probe}
    if probe == "missing":
        return row
    if probe == "dir":
        host.mkdir(parents=True, exist_ok=True)
        proc = _docker("cp", f"{src.container_id}:{path}/.", str(host), check=False)
    else:
        host.parent.mkdir(parents=True, exist_ok=True)
        proc = _docker("cp", f"{src.container_id}:{path}", str(host), check=False)
    row["download_returncode"] = proc.returncode
    if proc.returncode:  # Harbor records a failed download and does not upload it
        row["download_error"] = proc.stderr.decode(errors="replace")[-500:]
        if host.is_dir() and not host.is_symlink():
            shutil.rmtree(host, ignore_errors=True)
        else:
            host.unlink(missing_ok=True)
        return row
    if host.is_symlink():
        row["symlink_target"] = str(host.readlink())
    elif host.is_file():
        row.update(size=host.stat().st_size, sha256=_sha(host), mode=oct(host.stat().st_mode & 0o7777))
    elif host.is_dir():
        row["files"] = {p.relative_to(host).as_posix(): _sha(p) for p in sorted(host.rglob("*"))
                        if p.is_file() and not p.is_symlink()}
    return row


def upload(verifier: VerifierContainer, row: dict, host: Path) -> None:
    """Harbor's re-materialization of one collected artifact at its original path (skipped when not collected)."""
    target = row["source"]
    if not (host.exists() or host.is_symlink()):
        row["uploaded"] = False
        return
    if host.is_dir() and not host.is_symlink():
        verifier.root(empty_dirs_command(target))
        _docker("cp", f"{host}/.", f"{verifier.container_id}:{target}")
    else:
        parent = PurePosixPath(target).parent.as_posix()
        if parent and parent != target:
            verifier.root(ensure_dirs_command(parent))
        _docker("cp", str(host), f"{verifier.container_id}:{target}")
    row["uploaded"] = True


def run_private_grade(*, verifier_image_id: str, worker_image_id: str, artifacts: list[str], workspace: Path,
                      output: Path, runtime: RuntimeExport | None, timeout: int, cpus: str, memory: str,
                      recorder_args: str | None = None) -> dict[str, Any]:
    """Grade an isolated copy of the submitted W (+ captured R); write private receipts into ``output``."""
    workspace = Path(workspace).resolve()
    output = Path(output).resolve()
    if output == workspace or output.is_relative_to(workspace):
        raise ValueError("Private output in W")
    if runtime is None:
        raise ValueError("terminalbench-v4 grades captured W and R only; baseline R is not the measured state")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[str, Any] = {"infrastructure_errors": [], "files": {}, "artifacts": []}
    started = time.monotonic()
    src = verifier = None
    try:
        validate_runtime(runtime, worker_image_id, workspace)
        snap: dict[str, Any] = dict(runtime.snapshot)
        snap["archive"] = runtime.archive.read_bytes() if runtime.archive else None
        if snap.get("foreign_processes"):
            result["runtime_restore_scope"] = "filesystem only; live worker processes not restored"
            result["foreign_processes"] = snap["foreign_processes"]
            snap["runtime_class"] = RuntimeClass.CAPTURED.value
        owner = tuple(snap.get("workspace_default_owner", (0, 0)))
        packed, unread = pack_submitted(workspace, snap.get("workspace_fidelity_manifest") or {}, owner)
        if unread:
            result["owner_unreadable_modes_restored"] = unread
        fidelity = fidelity_manifest_from_tar(packed)
        if fidelity != runtime.snapshot["workspace_fidelity_manifest"]:
            raise ValueError("W export differs from paired execution fidelity")
        receipt: dict[str, Any] = {"harbor_command": harbor_command(), "recorder_env": RECORDER_ENV}
        # 1. The submitted state in a fresh worker container (the agent container Harbor collects from).
        src = DockerWorkerEnvironment(image=worker_image_id, network="none", name_prefix="tr-tbv4-gsrc",
                                      env=dict(runtime.environment), memory=memory, cpus=cpus)
        if src.image_id != worker_image_id:
            raise ValueError("Source image differs from the frozen worker image")
        if snap.get("changed") or snap.get("deleted") or snap.get("archive"):
            receipt["runtime_restore"] = src.restore_runtime(snap)
        else:
            receipt["runtime_restore"] = {"validated": None, "empty_capture": True, "processes_restored": False}
        receipt["workspace_restore"] = src.restore_workspace(packed, fidelity)
        receipt["source_binding"] = src.binding()
        # 2. Collect: the convention dir first, then every declared artifact (Harbor's order).
        staging = output / "collected"
        if staging.exists():
            remove_tree(staging)
        rows, hosts = [], []
        for i, artifact in enumerate([CONVENTION_DIR, *artifacts]):
            host = staging / f"{i:02d}" / (PurePosixPath(artifact).name or "root")
            rows.append(copy_out(src, artifact, host))
            hosts.append(host)
        src.close()
        src = None
        # 3. A fresh verifier container: empty /logs/verifier, upload, grader bundle, chmod, run test.sh.
        verifier = VerifierContainer(verifier_image_id, cpus=cpus, memory=memory)
        if verifier.image_id != verifier_image_id:
            raise ValueError("Verifier image differs from the frozen verifier image")
        receipt["verifier_binding"] = verifier.binding()
        verifier.root(empty_dirs_command(VERIFIER_DIR))
        rows[0].setdefault("note", "Harbor's agent publish dir; it exists (empty) in every agent container")
        verifier.root(empty_dirs_command(CONVENTION_DIR))
        for row, host in zip(rows, hosts):
            if row["kind"] != "missing" and "download_error" not in row:
                upload(verifier, row, host)
            else:
                row["uploaded"] = False
        result["artifacts"] = rows
        verifier.root("tar -C / -xf - --no-same-owner", input_bytes=grader_bundle())
        verifier.root(f"chmod +x {TEST_SCRIPT}")
        env = dict(RECORDER_ENV)
        if recorder_args:
            env["PYTEST_ADDOPTS"] = env["PYTEST_ADDOPTS"] + " " + recorder_args
        script = _RUN.format(timeout=int(timeout), command=shlex.quote(harbor_command()))
        command = ["exec", "-i"]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        command += [verifier.container_id, "bash", "-c", script]
        receipt["argv"] = ["docker", *command[:-1], "<grading script>"]
        exec_started = time.monotonic()
        proc = _docker(*command, timeout=int(timeout) + TRANSPORT_GRACE, check=False)
        receipt["exec_seconds"] = time.monotonic() - exec_started
        receipt["exec_returncode"] = proc.returncode
        receipt["exec_stderr"] = proc.stderr.decode(errors="replace")[-4000:]
        if proc.returncode != 0:
            raise RuntimeError(f"grading exec transport failed ({proc.returncode})")
        # 4. Read the verifier's outputs and the recorder's.
        collected = verifier.root("tar -C / -cf - logs/verifier tb-grader/out", timeout=300).stdout
        files: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(collected), mode="r:") as tf:
            for member in tf.getmembers():
                if member.isreg():
                    files[member.name] = tf.extractfile(member).read()
        (output / "verifier").mkdir(exist_ok=True)
        for name, data in files.items():
            base = Path(name).name
            target = output / base if name.startswith("tb-grader/out/") else output / "verifier" / base
            target.write_bytes(data)
        meta = json.loads(files["tb-grader/out/meta.json"])
        meta["seconds"] = float(meta["end"]) - float(meta["start"])
        receipt["meta"] = meta
        result["receipt"] = receipt
        result["test_sh_seconds"] = meta["seconds"]
        result["returncode"] = meta["returncode"]
        result["timed_out"] = meta["returncode"] in (137, -9) and meta["seconds"] >= timeout - 2
        result["reward"] = read_reward(files)
        result["files"] = sorted(files)
    except Exception as exc:  # recorded as INFRA_ERROR by the adapter
        result["infrastructure_errors"].append({"type": type(exc).__name__, "message": str(exc)[:4000]})
    finally:
        for box in (src, verifier):
            if box is not None:
                try:
                    box.close()
                except Exception as exc:  # pragma: no cover - cleanup failure is recorded
                    result["infrastructure_errors"].append({"type": "CleanupError", "message": str(exc)[:1000]})
        result["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def read_reward(files: dict[str, bytes]) -> dict:
    """Harbor's reward: reward.json if present, else reward.txt; absent or unparseable is recorded, never invented."""
    if "logs/verifier/reward.json" in files:
        try:
            data = json.loads(files["logs/verifier/reward.json"])
            value = data.get("reward") if isinstance(data, dict) else data
            return {"source": "reward.json", "value": float(value), "raw": data}
        except (ValueError, TypeError, AttributeError) as exc:
            return {"source": "reward.json", "value": None, "error": str(exc)[:200]}
    if "logs/verifier/reward.txt" in files:
        text = files["logs/verifier/reward.txt"].decode(errors="replace").strip()
        try:
            return {"source": "reward.txt", "value": float(text), "raw": text}
        except ValueError:
            return {"source": "reward.txt", "value": None, "raw": text[:100]}
    return {"source": None, "value": None, "error": "no reward file (Harbor: RewardFileNotFoundError)"}
