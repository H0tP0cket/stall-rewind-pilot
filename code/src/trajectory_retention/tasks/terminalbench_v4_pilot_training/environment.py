"""Native linux/arm64 rebuild of a Terminal-Bench 4 task: agent environment, worker layer and separate verifier image.

Adapted from the terminalbench-v2 lane (lanes never import each other). The official TB4 images
(``harborframework/terminal-bench:<task>-{environment,verifier}-<hash>``) are linux/amd64 only and are never pulled;
each task's ``environment/Dockerfile`` and ``tests/Dockerfile`` are rebuilt natively for linux/arm64 from the same build
contexts, with every ``FROM`` pinned by digest to the arm64 variant of the base release the official image was built from
(``official.py``). Build steps that download packages run at build time with network, as upstream.

Three images per task, each tagged ``tr-tbv4-pilot-training/<task>:<kind>``:

1. native   - environment/Dockerfile, unchanged except the digest-pinned FROM (the agent environment).
2. worker   - native + the lane adaptation layer: ``python3`` only if the native image lacks it, ``/workspace`` created,
              native ``/app`` replaced by a symlink to ``/workspace``; WORKDIR /app and the native USER kept. The native
              ``/app`` (numeric owners and modes) becomes the package's natural-start W.
3. verifier - tests/Dockerfile, unchanged except the digest-pinned FROM, built from the task's tests/ directory as
              upstream (so it carries the hidden tests at /tests, as the official verifier image does). Private: never
              given to the worker.

The pinned Dockerfile is written outside the build context, so ``COPY . /tests/`` copies exactly the upstream tests/.
No solution, test or reference file is copied into the native or worker image.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time

from .source import remove_tree
from . import source as S

LANE = S.LANE
LABEL = "trajectory-retention.lane=" + LANE
TAG = "tr-tbv4-pilot-training/{task}:{kind}"
BUILD_ROOT = Path.home() / ".cache/trajectory-retention/v4-pilot-training/build"
# Docker Desktop's VM disk is shared with other sessions. A build must leave at least this many free bytes (root-reserved
# blocks included, measured inside a container) after its estimated size.
MIN_FREE_BYTES = int(float(os.environ.get("TBV4_TRAINING_MIN_FREE_GB", "3")) * 1e9)
# Non-root users (apt's _apt sandbox user during builds, a task's own USER at run time) cannot use the root-reserved
# blocks: with nothing left for them apt update fails with "invalid signature" and worker writes fail with ENOSPC, however
# much the root-inclusive figure shows (seen 2026-10-12: 5.2 GB free, 0 available to non-root). A build also needs this.
MIN_NON_ROOT_BYTES = int(float(os.environ.get("TBV4_TRAINING_MIN_NON_ROOT_GB", "0.5")) * 1e9)
# Estimated on-disk size of a rebuilt image = this factor x the official amd64 image's compressed layer bytes.
SIZE_FACTOR = 2.5
PROBE_IMAGE = "alpine:3"
PLATFORM = "linux/arm64"
NATIVE_WORKDIR = "/app"
KINDS = ("native", "worker", "verifier")


def run(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float | None = None, check: bool = True,
        cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout, cwd=cwd)
    if check and proc.returncode:
        raise RuntimeError(f"{shlex.join(cmd[:6])} ... failed ({proc.returncode}): "
                           f"{proc.stderr.decode(errors='replace')[-3000:]}")
    return proc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_info(ref: str) -> dict:
    info = json.loads(run(["docker", "image", "inspect", ref]).stdout)[0]
    return {"id": info["Id"], "architecture": info.get("Architecture"), "os": info.get("Os"),
            "size": info.get("Size"), "layers": info.get("RootFS", {}).get("Layers", []),
            "workdir": info.get("Config", {}).get("WorkingDir"), "user": info.get("Config", {}).get("User") or "",
            "env": info.get("Config", {}).get("Env") or [], "entrypoint": info.get("Config", {}).get("Entrypoint"),
            "cmd": info.get("Config", {}).get("Cmd")}


def resolve_base(ref: str) -> dict:
    """Fallback when the official base is not identified: pull the tag for linux/arm64 and pin its repo digest."""
    run(["docker", "pull", "--platform", PLATFORM, ref], timeout=1800)
    info = json.loads(run(["docker", "image", "inspect", ref]).stdout)[0]
    repo = ref.split(":")[0]
    digests = [d for d in info.get("RepoDigests", []) if d.split("@")[0] in (repo, "docker.io/library/" + repo,
                                                                            "library/" + repo)]
    if not digests:
        raise RuntimeError("No repo digest for base " + ref + ": " + str(info.get("RepoDigests")))
    return {"ref": ref, "pinned": ref + "@" + digests[0].split("@", 1)[1]}


def pinned_dockerfile(task: str, role: str) -> tuple[str, list[dict], dict]:
    """The upstream Dockerfile of one role with every FROM pinned to the arm64 variant of the official base release."""
    from . import official as O
    path = S.task_dir(task) / ("environment/Dockerfile" if role == "environment" else "tests/Dockerfile")
    text = path.read_text()
    official, row = O.official_config(task, role)
    bases = []

    def pin(match: re.Match) -> str:
        ref = match.group(2)
        found = O.match_base(ref, official)
        if found.get("pinned"):
            O.pull_arm64(found["pinned"])
            found["fallback"] = None
        else:
            found.update(resolve_base(O.split_ref(ref)[0] + ":" + O.split_ref(ref)[1]))
            found["fallback"] = "official base not identified; current tag digest used"
        bases.append(found)
        return match.group(1) + found["pinned"] + match.group(3)

    pinned = re.sub(r"(?im)^(FROM\s+)(\S+)(.*)$", pin, text)
    summary = {"image": row["pinned_ref"], "release_ref": row.get("release_ref"), "role": role,
               "architecture": official.get("architecture"), "created": official.get("created"),
               "env": [e for e in official.get("config", {}).get("Env", []) if "VERSION" in e],
               "user": official.get("config", {}).get("User") or "", "workdir": official.get("config", {}).get("WorkingDir"),
               "layers": len(official.get("rootfs", {}).get("diff_ids", []))}
    return pinned, bases, summary


def official_compressed_bytes(task: str, role: str) -> int | None:
    """Sum of the official amd64 image's compressed layer sizes (registry manifest; nothing pulled)."""
    from . import official as O
    try:
        row = O.official_images(task)[role]
        repo, digest = row["pinned_ref"].split(":", 1)[0], row["digest"]
        m = O.manifest(repo, digest, O.token(repo))
        if "manifests" in m:
            pick = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"]
            m = O.manifest(repo, pick[0]["digest"], O.token(repo))
        return sum(layer.get("size", 0) for layer in m.get("layers", []))
    except Exception:  # noqa: BLE001 - an unknown size is reported as unknown, never invented
        return None


def estimate_bytes(task: str, kind: str) -> int | None:
    if kind == "worker":
        return 50_000_000
    size = official_compressed_bytes(task, "environment" if kind == "native" else "verifier")
    return int(size * SIZE_FACTOR) if size is not None else None


def _build(context: Path, tag: str, *, dockerfile_text: str, workdir: Path, timeout: float = 5400,
           labels: dict | None = None) -> dict:
    """docker build with the Dockerfile and log kept outside the context (COPY . copies only upstream bytes)."""
    workdir.mkdir(parents=True, exist_ok=True)
    dockerfile = workdir / "Dockerfile.tb-lane"
    dockerfile.write_text(dockerfile_text)
    cmd = ["docker", "build", "--platform", PLATFORM, "-f", str(dockerfile), "-t", tag, "--label", LABEL,
           "--progress", "plain"]
    for k, v in (labels or {}).items():
        cmd += ["--label", f"{k}={v}"]
    started = time.monotonic()
    proc = run(cmd + [str(context)], timeout=timeout, check=False)
    seconds = time.monotonic() - started
    log = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    (workdir / "build.log").write_text(log)
    if proc.returncode:
        return {"result": "failed", "seconds": seconds, "returncode": proc.returncode, "log_tail": log[-4000:],
                "dockerfile_sha256": sha256_bytes(dockerfile_text.encode())}
    return {"result": "built", "seconds": seconds, "tag": tag, "image": image_info(tag),
            "dockerfile_sha256": sha256_bytes(dockerfile_text.encode())}


def context_manifest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): S.sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def build_role(task: str, kind: str) -> dict:
    """Build the native (environment/) or verifier (tests/) image from a fresh copy of the upstream context."""
    role, sub = ("environment", "environment") if kind == "native" else ("verifier", "tests")
    upstream = S.task_dir(task) / sub
    base = BUILD_ROOT / task / kind
    if base.exists():
        remove_tree(base)
    context = base / "context"
    shutil.copytree(upstream, context, symlinks=True)
    dockerfile, bases, official = pinned_dockerfile(task, role)
    result = _build(context, TAG.format(task=task, kind=kind), dockerfile_text=dockerfile, workdir=base,
                    labels={"terminalbench.task": task, "terminalbench.source": S.SOURCE_REVISION,
                            "terminalbench.role": role})
    result.update({"bases": bases, "official_image": official,
                   "upstream_dockerfile_sha256": S.sha256_file(upstream / "Dockerfile"),
                   "context_manifest": context_manifest(upstream),
                   "context_identical_to_upstream": context_manifest(context) == context_manifest(upstream)})
    if kind == "verifier":
        result["private"] = "carries the hidden tests at /tests (as the official verifier image); never given to the worker"
    return result


def id_ref(task: str, kind: str, image_id: str) -> str:
    """BuildKit cannot use a bare image ID in FROM: tag the exact ID immutably and return that tag."""
    ref = f"tr-tbv4-pilot-training/{task}:{kind}-id-{image_id.split(':')[-1][:16]}"
    run(["docker", "tag", image_id, ref])
    if image_info(ref)["id"] != image_id:
        raise RuntimeError("ID tag does not resolve to the recorded image")
    return ref


WORKER_DOCKERFILE = """# terminalbench-v4-pilot-training worker adaptation layer (disclosed in every package)
FROM {native}
USER root
RUN if command -v python3 >/dev/null 2>&1; then echo "python3 present: $(python3 -V)"; else \\
      export DEBIAN_FRONTEND=noninteractive && apt-get update && \\
      apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*; fi
RUN rm -rf {app} && mkdir -p /workspace && ln -s /workspace {app}
WORKDIR {app}
{user}
"""


def build_worker(task: str, native: dict) -> dict:
    base = BUILD_ROOT / task / "worker"
    if base.exists():
        remove_tree(base)
    context = base / "context"
    context.mkdir(parents=True)
    info = native["image"]
    if info.get("workdir") != NATIVE_WORKDIR:
        raise ValueError(f"Native WORKDIR is {info.get('workdir')!r}, lane requires {NATIVE_WORKDIR}")
    user = f"USER {info['user']}" if info.get("user") not in ("", "root", "0", "0:0") else ""
    text = WORKER_DOCKERFILE.format(native=id_ref(task, "native", info["id"]), app=NATIVE_WORKDIR, user=user)
    result = _build(context, TAG.format(task=task, kind="worker"), dockerfile_text=text, workdir=base,
                    labels={"terminalbench.task": task, "terminalbench.native_image": info["id"]})
    if result["result"] == "built":
        result["extends_native_layers"] = result["image"]["layers"][:len(info["layers"])] == info["layers"]
        result["python3"] = probe(result["image"]["id"], "command -v python3 && python3 -V").stdout.decode().strip()
    return result


def export_app(image_id: str) -> bytes:
    """The native image's /app as a tar rooted at '.', numeric owners and modes kept (the './' entry included)."""
    cmd = ["docker", "run", "--rm", "--network", "none", "--label", LABEL, "--user", "0:0", "--entrypoint", "tar",
           image_id, "-C", NATIVE_WORKDIR, "--numeric-owner", "--format=pax", "-cf", "-", "."]
    return run(cmd, timeout=900).stdout


def app_tar_manifest(data: bytes) -> dict:
    """Entries of the exported /app tar (relative to /app; '.' is /app itself): type, mode, owner, size, sha256, link."""
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            rel = m.name[2:] if m.name.startswith("./") else m.name
            rel = rel or "."
            kind = "f" if m.isreg() else "d" if m.isdir() else "l" if m.issym() else "h" if m.islnk() else "o"
            digest = sha256_bytes(tf.extractfile(m).read()) if m.isreg() else None
            entries[rel] = {"type": kind, "mode": m.mode & 0o7777, "size": m.size if m.isreg() else None,
                            "sha256": digest, "link": m.linkname if (m.issym() or m.islnk()) else None,
                            "uid": m.uid, "gid": m.gid}
    return dict(sorted(entries.items()))


def gzip_bytes(data: bytes) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return out.getvalue()


def probe(image_id: str, script: str, *, timeout: float = 300, network: str = "none", user: str | None = None
          ) -> subprocess.CompletedProcess:
    return run(["docker", "run", "--rm", "--label", LABEL, "--network", network, *(["--user", user] if user else []),
                "--entrypoint", "sh", image_id, "-c", script], timeout=timeout, check=False)


def docker_disk() -> dict:
    """Free bytes on the Docker VM's root filesystem: root-reserved blocks included, and what non-root users can use."""
    out = run(["docker", "run", "--rm", "--network", "none", "--label", LABEL, PROBE_IMAGE, "stat", "-f", "-c",
               "%f %a %S", "/"], timeout=120).stdout.decode().split()
    return {"free_bytes": int(out[0]) * int(out[2]), "non_root_available_bytes": int(out[1]) * int(out[2])}


def docker_free_bytes() -> int:
    return docker_disk()["free_bytes"]


class DiskHeadroom(RuntimeError):
    """The Docker VM disk would fall below the floor; the build was not started (reason "docker disk")."""


def require_headroom(what: str, estimate: int | None = None) -> dict:
    disk = docker_disk()
    need = MIN_FREE_BYTES + (estimate or 0)
    if disk["free_bytes"] < need:
        raise DiskHeadroom(f"docker disk: {disk['free_bytes'] / 1e9:.1f} GB free, {what} is estimated at "
                           f"{(estimate or 0) / 1e9:.1f} GB and must leave {MIN_FREE_BYTES / 1e9:.1f} GB; not started "
                           f"(the disk is shared with other sessions)")
    if disk["non_root_available_bytes"] < MIN_NON_ROOT_BYTES + (estimate or 0):
        raise DiskHeadroom(f"docker disk: {disk['non_root_available_bytes'] / 1e9:.1f} GB available to non-root users "
                           f"({disk['free_bytes'] / 1e9:.1f} GB free counting root-reserved blocks); {what} needs "
                           f"{(estimate or 0) / 1e9:.1f} GB plus {MIN_NON_ROOT_BYTES / 1e9:.1f} GB that apt's non-root "
                           f"user and the task's user can write; not started")
    return {**disk, "estimate_bytes": estimate, "floor_bytes": MIN_FREE_BYTES, "non_root_floor_bytes": MIN_NON_ROOT_BYTES}
