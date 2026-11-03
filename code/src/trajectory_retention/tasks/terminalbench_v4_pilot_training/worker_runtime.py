"""Offline hostname resolution and faithful capture of unbound Unix socket files.

PyTorch leaves socket inodes after its workers exit. GNU tar skips these inodes.
Only sockets absent from /proc/net/unix are reconstructed; live processes and
mounted-state changes retain the shared runner's unsupported classification.
No task files, training commands, tests or network interfaces are changed.
"""
from __future__ import annotations

import io
import json
import shlex
import tarfile

from trajectory_retention.contracts import RuntimeClass
from trajectory_retention.execution.environment import (
    DockerWorkerEnvironment, CaptureError, RestoreError, comparable,
    fidelity_manifest_from_tar, manifest_diff, RUNTIME_IGNORED_PREFIXES,
)

RULE = "offline-hostname-stale-unix-files/v1"
SOCKET_HEADER = "trajectory_retention.stale_unix_socket_files"


class OfflineWorkerEnvironment(DockerWorkerEnvironment):
    def _initialize_state(self):
        # network=none omits Docker's usual hostname entry. Keep all sockets on
        # loopback while allowing native torchrun/Spark to resolve their own name.
        self._exec("printf '127.0.0.1 %s\\n' \"$(hostname)\" >> /etc/hosts")
        super()._initialize_state()

    def binding(self):
        return {**super().binding(), "runtime_rule": RULE,
                "local_hostname_resolution": "own hostname maps to 127.0.0.1; network mode unchanged"}

    def capture_runtime(self):
        manifest = self.runtime_manifest()
        sockets = {k: v for k, v in manifest["entries"].items() if v["type"] == "s"}
        if not sockets:
            result = super().capture_runtime()
            result["rule"] = RULE
            return result
        active = self._exec("cat /proc/net/unix").stdout.decode().splitlines()[1:]
        bound = {parts[7] for line in active if len(parts := line.split(maxsplit=7)) == 8}
        if any("/" + path in bound for path in sockets):
            raise CaptureError("Live Unix socket state is not restorable by this filesystem adapter")
        entries = {k: v for k, v in manifest["entries"].items() if k not in sockets}
        listing = "".join("/" + path + "\n" for path in sorted(entries)).encode()
        raw = self._exec("tar --format=pax --numeric-owner --no-recursion -cf - -T /dev/stdin",
                         input_bytes=listing, timeout=900).stdout
        archived = {k.lstrip("/"): v for k, v in fidelity_manifest_from_tar(raw).items()}
        if comparable(archived) != comparable(entries):
            raise CaptureError("Runtime changed during archive capture: " + "; ".join(manifest_diff(entries, archived)))
        processes = self.process_table()
        foreign = [p for p in processes if self._process_identity(p) not in self._harness_processes]
        # Do not certify even stale inodes if a background process could rebind
        # them between the liveness check and the archive's final validation.
        if foreign:
            raise CaptureError("Unix socket file capture requires no non-harness processes")
        if self.runtime_manifest() != manifest:
            raise CaptureError("Runtime changed during socket-file capture")
        out = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source, tarfile.open(
                fileobj=out, mode="w", format=tarfile.PAX_FORMAT,
                pax_headers={SOCKET_HEADER: json.dumps(sockets, sort_keys=True)}) as target:
            for member in source:
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
        mount_changed = manifest["mounts"] != self._initial_mount_state
        return {"rule": RULE, "entries": manifest["entries"], "changed": sorted(manifest["entries"]),
                "deleted": manifest["deleted"], "archive": out.getvalue(), "processes": processes,
                "foreign_processes": foreign,
                "runtime_class": (RuntimeClass.UNSUPPORTED if mount_changed else RuntimeClass.CAPTURED).value,
                "mounts": manifest["mounts"], "initial_mounts": self._initial_mount_state,
                "mount_state_changed": mount_changed,
                "coverage": {"filesystem_outside_workspace": "captured_with_fidelity_manifest",
                             "stale_unix_socket_files": "PAX metadata; require no bound sockets or foreign processes",
                             "processes": "recorded_not_restorable", "network_state": "not_captured",
                             "shell_state": "none_by_construction", "excluded_path_roots": list(RUNTIME_IGNORED_PREFIXES),
                             "mounted_storage": "baseline divergence is unsupported"}}

    def restore_runtime(self, runtime):
        sockets = {}
        if runtime.get("archive"):
            with tarfile.open(fileobj=io.BytesIO(runtime["archive"]), mode="r:*") as tf:
                sockets = json.loads(tf.pax_headers.get(SOCKET_HEADER, "{}"))
        expected = {k: v for k, v in runtime.get("entries", {}).items() if v["type"] == "s"}
        if sockets != expected:
            raise RestoreError("Socket metadata differs from recorded fidelity manifest")
        if not sockets:
            return super().restore_runtime(runtime)
        if runtime.get("runtime_class") == RuntimeClass.UNSUPPORTED.value or runtime.get("mount_state_changed"):
            raise RestoreError("Runtime contains unsupported state")
        self._exec("cd / && tar --numeric-owner -xpf -", input_bytes=runtime["archive"], timeout=900)
        script = '''import json,os,socket,stat,sys
for rel,meta in json.load(sys.stdin).items():
    if rel.startswith('/') or '..' in rel.split('/'):
        raise ValueError('Invalid socket path')
    path='/'+rel
    if os.path.lexists(path):
        raise ValueError('Socket path already exists: '+path)
    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:s.bind(path)
    finally:s.close()
    os.chown(path,meta['uid'],meta['gid']);os.chmod(path,meta['mode'])
    os.utime(path,ns=(meta['mtime_ns'],meta['mtime_ns']))
'''
        self._exec("python3 -c " + shlex.quote(script), input_bytes=json.dumps(sockets).encode())
        # Reuse all shared deletion, mount and fidelity validation; tar excludes
        # the recreated inodes and therefore does not replace them on this pass.
        result = super().restore_runtime(runtime)
        return {**result, "rule": RULE, "stale_unix_socket_files_restored": len(sockets)}
