"""Disposable worker containers that build admission control states (start, reference, tamper) and capture W and R.

Adapted from the terminalbench-v2 lane (lanes never import each other). The natural start is restored from the package's
numeric-owner start archive exactly as the runner restores it (``initial_workspace_archive``), not re-packed with the
worker as owner. Every control state is captured exactly as the runner captures a checkpoint (W tar + fidelity
manifest, filesystem R diff + process table) and graded through the lane's grader with that captured R; nothing is
graded on baseline R.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from trajectory_retention.execution.archives import extract_directory
from trajectory_retention.execution.environment import DockerWorkerEnvironment, fidelity_manifest_from_tar

from . import environment as E


def directory_tar(root: Path, arcname: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in [Path(root), *sorted(Path(root).rglob("*"))]:
            info = tf.gettarinfo(str(path), arcname=arcname + ("/" + path.relative_to(root).as_posix() if path != Path(root) else ""))
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            if path.is_file() and not path.is_symlink():
                with open(path, "rb") as handle:
                    tf.addfile(info, handle)
            else:
                tf.addfile(info)
    return buf.getvalue()


class Disposable:
    """A worker-image container restored to the natural start (the package's numeric-owner /app archive)."""

    def __init__(self, *, worker_image_id: str, workspace_archive: Path, root: Path, memory: str, cpus: str,
                 network: str = "none", timeout: float = 3600):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        data = gzip.decompress(Path(workspace_archive).read_bytes())
        from .services import ServiceWorkerEnvironment
        from .source import EVIDENCE
        plan = json.loads((EVIDENCE / 'service-build.json').read_text())
        self.env = ServiceWorkerEnvironment(service_plan=plan, image=worker_image_id, network=network, name_prefix="tr-tbv4-gate",
                                           memory=memory, cpus=cpus, timeout=timeout)
        self.restore = self.env.restore_workspace(data, fidelity_manifest_from_tar(data))
        self.network = "per-run-internal-isolated/v1"

    def run(self, command: str, *, timeout: float, input_bytes: bytes | None = None, user: str | None = None) -> dict:
        started = time.monotonic()
        argv = ["docker", "exec", "-i", "-w", E.NATIVE_WORKDIR, *(["--user", user] if user else []),
                self.env.container_id, "bash", "-c", command]
        try:
            proc = subprocess.run(argv, input=input_bytes, capture_output=True, timeout=timeout)
            code, out = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            code, out = "timeout", (exc.stdout or b"") + (exc.stderr or b"")
        return {"returncode": code, "seconds": round(time.monotonic() - started, 3), "network": self.network,
                "output_tail": out.decode(errors="replace")[-3000:]}

    def put_directory(self, source: Path, target: str) -> None:
        """Stream a host directory into the container at ``target`` (root-owned), e.g. the reference at /solution."""
        parent, name = target.rstrip("/").rsplit("/", 1)
        subprocess.run(["docker", "exec", "-i", "--user", "0:0", self.env.container_id, "tar", "-C", parent or "/",
                        "-xf", "-"], input=directory_tar(source, name), check=True, capture_output=True, timeout=600)

    def root_listing(self) -> str:
        """Owner and mode of /app itself (the natural start keeps the native values)."""
        proc = subprocess.run(["docker", "exec", "--user", "0:0", self.env.container_id, "stat", "-c", "%a %u %g",
                               "/workspace"], capture_output=True, timeout=60)
        return proc.stdout.decode().strip()

    def capture(self, label: str) -> tuple[Path, RuntimeExport, dict]:
        data, _ = self.env.capture_workspace()
        runtime = self.env.capture_runtime()
        base = self.root / label
        workspace = base / "W"
        extract_directory(data, workspace)
        archive_path = None
        if runtime.get("archive"):
            archive_path = base / "runtime.tar"
            archive_path.write_bytes(runtime["archive"])
        export = RuntimeExport(
            base_image_id=self.env.image_id, runtime_class=RuntimeClass(runtime["runtime_class"]),
            snapshot={"rule": runtime["rule"], "entries": runtime["entries"], "changed": runtime["changed"],
                      "deleted": runtime["deleted"], "processes": runtime["processes"],
                      "foreign_processes": runtime["foreign_processes"], "coverage": runtime["coverage"],
                      "runtime_class": runtime["runtime_class"], "mounts": runtime.get("mounts"),
                      "initial_mounts": runtime.get("initial_mounts"),
                      "mount_state_changed": runtime.get("mount_state_changed"),
                      "workspace_fidelity_manifest": fidelity_manifest_from_tar(data),
                      "workspace_default_owner": [self.env.worker_uid, self.env.worker_gid],
                      "process_restore_supported": False},
            archive=archive_path, archive_sha256=hashlib.sha256(runtime["archive"]).hexdigest() if archive_path else None,
            workspace_snapshot_sha256=hashlib.sha256(data).hexdigest(), environment=dict(self.env.env))
        summary = {"runtime_changed": runtime["changed"], "runtime_deleted": runtime["deleted"],
                   "foreign_processes": [{k: p.get(k) for k in ("pid", "cmdline")} for p in runtime["foreign_processes"]],
                   "runtime_class": runtime["runtime_class"], "workspace_bytes": len(data),
                   "workspace_sha256": hashlib.sha256(data).hexdigest()}
        (base / "capture-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return workspace, export, summary

    def close(self) -> None:
        self.env.close()
