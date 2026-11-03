"""Restore submitted R away from the immutable evaluator and bridge CLI calls.

Restoration is delegated to the execution environment. The evaluator exchanges
only public test inputs and submitted output through a private host mailbox;
the restored submission container has no evaluator, reference, or mailbox mount.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
from typing import Any

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from trajectory_retention.execution.archives import pack_directory


def tree_tar(root: Path, *, ownership: dict | None = None,
             default_owner: tuple[int, int] | None = None) -> bytes:
    """Repack exported W without substituting host owners or rounding mtimes.

    The host export is a transport projection. Its paired runtime export holds
    container ownership and modes, including permission bits sanitized on the
    host. File bytes and exact modification times still come from the export.
    """
    return pack_directory(root, ownership=ownership, default_owner=default_owner)


class RuntimeGradeSession:
    def __init__(self, runtime: RuntimeExport, workspace: Path, public_assets: Path,
                 mailbox: Path, evaluator_container: str, fixture_container_path: str,
                 *, baseline_only: bool = False):
        self.runtime, self.mailbox = runtime, mailbox
        self.evaluator_container = evaluator_container
        self.fixture_path = fixture_container_path
        self.env = None
        self.errors: list[dict] = []
        self.calls: list[dict] = []
        self.validation: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread = None
        self._validate_export(workspace)
        if not baseline_only and (
                not isinstance(runtime.snapshot.get("workspace_fidelity_manifest"), dict)
                or "workspace_default_owner" not in runtime.snapshot):
            raise ValueError("Runtime grading requires paired workspace fidelity and default owner")
        try:
            # Import at use time so baseline-only packaging never requires a
            # worker SDK import. No second snapshot/restoration implementation.
            from trajectory_retention.execution.environment import DockerWorkerEnvironment, fidelity_manifest_from_tar
            self.env = DockerWorkerEnvironment(image=runtime.base_image_id, network="none",
                                               env=dict(runtime.environment), name_prefix="tr-private-submission")
            if self.env.image_id != runtime.base_image_id:
                raise ValueError("Runtime base image must be its exact immutable image ID")
            snapshot = dict(runtime.snapshot)
            snapshot["archive"] = runtime.archive.read_bytes() if runtime.archive else None
            if baseline_only:
                self.validation["runtime"] = {"baseline_only": True,
                    "basis": "fresh container at declared image; no worker R restoration claimed",
                    "initial_runtime_manifest": self.env.runtime_manifest()}
            else:
                self.validation["runtime"] = self.env.restore_runtime(snapshot)
            owner = snapshot.get("workspace_default_owner", [self.env.worker_uid, self.env.worker_gid])
            if (not isinstance(owner, (list, tuple)) or len(owner) != 2
                    or any(type(value) is not int or value < 0 for value in owner)):
                raise ValueError("Invalid paired workspace default owner")
            data = tree_tar(workspace, ownership=snapshot.get("workspace_fidelity_manifest"),
                            default_owner=tuple(owner))
            self.validation["workspace"] = self.env.restore_workspace(data, fidelity_manifest_from_tar(data))
            self.validation["workspace_transport"] = {
                "rule": "shared pack_directory; paired container ownership/modes and exact host-export mtime_ns",
                "paired_fidelity_manifest_present": "workspace_fidelity_manifest" in snapshot,
                "default_owner": list(owner),
            }
            self.env._exec("mkdir -p /public-assets && tar -C /public-assets -xpf -",
                           input_bytes=tree_tar(public_assets), timeout=60)
            self.validation["binding"] = self.env.binding()
            self.validation["runtime_archive_sha256"] = runtime.archive_sha256
            self.validation["workspace_snapshot_sha256"] = runtime.workspace_snapshot_sha256
            self.validation["private_evaluator_mounts"] = []
            self.validation["runtime_mode"] = "execution_restored_separate_submission_container"
            self._thread = threading.Thread(target=self._serve, name="file-backup-grade-bridge", daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def _validate_export(self, workspace: Path) -> None:
        r = self.runtime
        if r.runtime_class == RuntimeClass.UNSUPPORTED or r.snapshot.get("foreign_processes"):
            raise ValueError("Runtime has unsupported live process state")
        if not r.base_image_id.startswith("sha256:") or len(r.base_image_id) != 71:
            raise ValueError("Unpinned runtime base image")
        if r.archive:
            archive = r.archive.resolve(strict=True)
            if archive.is_relative_to(workspace.resolve()):
                raise ValueError("Runtime archive must be an execution-owned export outside W")
            if hashlib.sha256(archive.read_bytes()).hexdigest() != r.archive_sha256:
                raise ValueError("Runtime archive digest mismatch")
        if r.snapshot.get("changed") and r.archive is None:
            raise ValueError("Runtime changed paths have no archive")
        if r.runtime_class == RuntimeClass.CAPTURED and "entries" not in r.snapshot:
            raise ValueError("Captured runtime lacks a content/fidelity verification manifest")
        if r.runtime_class == RuntimeClass.IRRELEVANT:
            if not r.snapshot.get("irrelevance_evidence"):
                raise ValueError("R_IRRELEVANT requires task-specific evidence")
        if r.snapshot.get("runtime_class") not in (None, r.runtime_class.value):
            raise ValueError("Runtime snapshot class disagrees with export")

    def _fixture_root(self, argv: list[str]) -> str:
        if "--schedule" not in argv:
            raise ValueError("Evaluator invocation lacks fixture schedule")
        schedule = Path(argv[argv.index("--schedule") + 1])
        container_root = Path(self.fixture_path)
        relative = schedule.parent.relative_to(container_root)
        if ".." in relative.parts or not relative.parts:
            raise ValueError("Fixture path escapes isolated grading fixture tree")
        return str(schedule.parent)

    def _evaluator_exec(self, command: str, *, data: bytes | None = None) -> bytes:
        proc = subprocess.run(["docker", "exec", "-i", self.evaluator_container, "sh", "-c", command],
                              input=data, capture_output=True, timeout=60)
        if proc.returncode:
            raise RuntimeError("Trusted evaluator fixture transport failed: " + proc.stderr.decode(errors="replace"))
        return proc.stdout

    def _invoke(self, request: dict) -> dict:
        argv = request["argv"]
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            raise ValueError("Invalid evaluator invocation")
        remote = self._fixture_root(argv)
        cwd = request["cwd"]
        if cwd != "/workspace" and not Path(cwd).is_relative_to(self.fixture_path):
            raise ValueError("Unknown submitted working directory")
        fixture_bytes = self._evaluator_exec(f"tar --format=pax --numeric-owner -C {shlex.quote(remote)} -cf - .")
        self.env._exec(f"rm -rf -- {shlex.quote(remote)} && mkdir -p {shlex.quote(remote)} && tar -C {shlex.quote(remote)} -xpf -",
                       input_bytes=fixture_bytes, timeout=60)
        command = shlex.join(["python", "/workspace/backup_scheduler.py", *argv])
        started = time.monotonic()
        proc = self.env._exec(command, cwd=cwd, check=False, timeout=32)
        # Return only this test's public fixture effects. Safe extraction rejects
        # special files, traversal and symlinks escaping the fixture root.
        exported = self.env._exec(f"tar --format=pax --numeric-owner -C {shlex.quote(remote)} -cf - .", timeout=60).stdout
        with tarfile.open(fileobj=io.BytesIO(exported), mode="r:*") as archive:
            for member in archive.getmembers():
                # Validate without materializing on a potentially case-insensitive
                # host filesystem. Both test worlds remain on Linux filesystems.
                tarfile.data_filter(member, "/file-backup-validated-fixture")
        self._evaluator_exec(
            f"find {shlex.quote(remote)} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + && "
            f"tar -C {shlex.quote(remote)} -xpf -", data=exported)
        result = {"returncode": proc.returncode, "stdout_base64": base64.b64encode(proc.stdout).decode(),
                  "stderr_base64": base64.b64encode(proc.stderr).decode(),
                  "elapsed_seconds": time.monotonic() - started}
        self.calls.append({"argv": argv, "cwd": cwd, **result})
        return result

    def _serve(self) -> None:
        while not self._stop.is_set():
            for path in sorted(self.mailbox.glob("*.request.json")):
                response = path.with_name(path.name.replace(".request.json", ".response.json"))
                if response.exists():
                    continue
                try:
                    result = self._invoke(json.loads(path.read_text()))
                except BaseException as exc:
                    result = {"infrastructure_error": f"{type(exc).__name__}: {exc}"}
                    self.errors.append({"request": path.name, **result})
                temp = response.with_suffix(".tmp")
                temp.write_text(json.dumps(result))
                temp.replace(response)
            self._stop.wait(0.01)

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=60)
        if self.env:
            try:
                self.validation["graded_workspace_after"] = self.env.workspace_fidelity_manifest()
            except Exception as exc:
                self.validation["graded_workspace_after_error"] = repr(exc)
            self.env.close()
            self.env = None
