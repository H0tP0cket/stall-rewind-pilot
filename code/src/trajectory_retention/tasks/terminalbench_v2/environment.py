"""Native linux/arm64 rebuild of a Terminal-Bench 2 task environment, plus worker and grader images.

Copied from the terminalbench-v1 lane (lanes never import each other); the image recipes are unchanged, so tasks
the v1 lane already built reuse Docker's layer cache. Images are tagged and labelled for terminalbench-v2.

The official task images (``alexgshaw/<task>:20251031``) are single-platform
x86_64; this host emulates x86_64 about 27x slower, so each task's pinned
``environment/Dockerfile`` is rebuilt natively for linux/arm64 from the same
build context. Every ``FROM`` is pinned to the arm64 variant of the base
release the official image was built from (``official.py``: layer-prefix match,
or the CPython release for python bases), by registry digest.
Build steps that download packages run at build time with network (as upstream).

Three images per task, each built FROM the previous image's exact ID:

1. native  - the upstream Dockerfile, unchanged except the digest-pinned FROM.
2. worker  - native + the lane adaptation layer: ``python3`` (the shared
   harness supervisor and capture probes run ``python3``; installed only when the
   native image lacks it), ``/workspace`` created, and native ``/app`` replaced
   by a symlink to ``/workspace``. The native ``/app`` bytes become the
   package's natural-start W (``public/workspace.tar.gz``).
3. grader  - worker + ``/tb-verifier``: the Python 3.13 environment that the
   native ``tests/test.sh`` would create at grade time with
   ``uvx -p 3.13 -w <pins>`` (uv 0.9.5), built ahead of time because grading
   runs with network none. uv's default interpreter choice is kept, as at a
   native grade: the image's own CPython 3.13 when it has one (python:3.13
   bases), otherwise uv's managed CPython 3.13 under ``/tb-verifier``. Never
   used for the worker.

No solution, test or reference file is copied into any image.
"""
from __future__ import annotations

import gzip
import os
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import time

from . import source as S

LANE = "terminalbench-v2"
LABEL = "trajectory-retention.lane=" + LANE
TAG = "tr-tbv2/{task}:{kind}"
BUILD_ROOT = Path.home() / ".cache/trajectory-retention/v2-L/build"
# Docker Desktop's VM disk is shared with other sessions; builds refuse to start below this many free bytes
# (root-reserved blocks included, measured inside a container).
MIN_FREE_BYTES = int(float(os.environ.get("TBV2_MIN_FREE_GB", "3")) * 1e9)
PROBE_IMAGE = "alpine:3"
PLATFORM = "linux/arm64"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.9.5@sha256:7bf0a77e82321b0f2019874f7ff40f2a58fa2ca851a816179931098d77e36d5e"
UV_IMAGE_INDEX = "ghcr.io/astral-sh/uv:0.9.5"
VERIFIER_ROOT = "/tb-verifier"
NATIVE_WORKDIR = "/app"
RESERVED_ROOTS = ("/tb-verifier", "/tb-grader", "/tests", "/logs")

# Native tests/test.sh template (most TB2 tasks at the pin share it apart from
# the canary comment, the ``-w`` pins, extra apt packages on the curl line and
# extra pytest flags). Parsed strictly; any other shape holds the task.
TEST_SH_BODY = [
    "#!/bin/bash",
    "apt-get update",
    "__APT__",
    "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh",
    "source $HOME/.local/bin/env",
    'if [ "$PWD" = "/" ]; then',
    'echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."',
    "exit 1",
    "fi",
    "__UVX__",
    "if [ $? -eq 0 ]; then",
    "echo 1 > /logs/verifier/reward.txt",
    "else",
    "echo 0 > /logs/verifier/reward.txt",
    "fi",
]
ALLOWED_PYTEST_FLAGS = ("-v",)


def run(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float | None = None, check: bool = True,
        cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout, cwd=cwd)
    if check and proc.returncode:
        raise RuntimeError(f"{shlex.join(cmd[:6])} ... failed ({proc.returncode}): "
                           f"{proc.stderr.decode(errors='replace')[-3000:]}")
    return proc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_test_sh(text: str) -> dict:
    """Parse the native test.sh into the verifier recipe, or raise if it is not the shared template.

    Returns python version, ``uvx -w`` pins, grade-time apt packages (besides curl)
    and the pytest arguments.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not (ln.startswith("#") and not ln.startswith("#!"))]
    joined, buf = [], ""
    for ln in lines:
        if ln.endswith("\\"):
            buf += ln[:-1].strip() + " "
            continue
        joined.append((buf + ln).strip())
        buf = ""
    uvx = [ln for ln in joined if ln.startswith("uvx ")]
    if len(uvx) != 1:
        raise ValueError("test.sh: expected exactly one uvx line")
    apt = [ln for ln in joined if ln.startswith("apt-get install -y curl")]
    if len(apt) != 1:
        raise ValueError("test.sh: expected one 'apt-get install -y curl ...' line")
    body = [("__UVX__" if ln == uvx[0] else "__APT__" if ln == apt[0] else ln) for ln in joined]
    if body != TEST_SH_BODY:
        raise ValueError("test.sh differs from the native template: " + json.dumps(body)[:800])
    apt_packages = shlex.split(apt[0])[4:]
    if not all(re.fullmatch(r"[a-z0-9][a-z0-9.+\-]*(=[A-Za-z0-9.+:~\-]+)?", p) for p in apt_packages):
        raise ValueError("Unexpected apt package token in test.sh")
    tokens = shlex.split(uvx[0])
    python, pins, rest, i = None, [], [], 1
    while i < len(tokens):
        if tokens[i] == "-p":
            python = tokens[i + 1]; i += 2
        elif tokens[i] == "-w":
            pins.append(tokens[i + 1]); i += 2
        else:
            rest = tokens[i:]; break
    head = ["pytest", "--ctrf", "/logs/verifier/ctrf.json", "/tests/test_outputs.py"]
    flags = rest[len(head):]
    if (rest[:len(head)] != head or "-rA" not in flags or any(f not in ("-rA",) + ALLOWED_PYTEST_FLAGS for f in flags)
            or python != "3.13" or not all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+", p) for p in pins)):
        raise ValueError(f"Unexpected uvx invocation: {uvx[0]}")
    if not any(p.startswith("pytest==") for p in pins) or not any(p.startswith("pytest-json-ctrf==") for p in pins):
        raise ValueError("uvx pins lack pytest/pytest-json-ctrf")
    return {"python": python, "with": pins, "apt_packages": apt_packages, "pytest_args": rest[1:],
            "uvx_line": uvx[0], "apt_line": apt[0]}


def resolve_base(ref: str) -> dict:
    """Pull an upstream base for linux/arm64 and return its repo digest pin."""
    if "@sha256:" in ref:
        name = ref
    else:
        run(["docker", "pull", "--platform", PLATFORM, ref], timeout=1800)
        info = json.loads(run(["docker", "image", "inspect", ref]).stdout)[0]
        repo = ref.split(":")[0] if ":" in ref.split("/")[-1] else ref
        digests = [d for d in info.get("RepoDigests", []) if d.split("@")[0] in (repo, "docker.io/library/" + repo,
                                                                                    "library/" + repo)]
        if not digests:
            raise RuntimeError("No repo digest for base " + ref + ": " + str(info.get("RepoDigests")))
        name = ref + "@" + digests[0].split("@", 1)[1]
    return {"ref": ref, "pinned": name}


def image_info(ref: str) -> dict:
    info = json.loads(run(["docker", "image", "inspect", ref]).stdout)[0]
    return {"id": info["Id"], "architecture": info.get("Architecture"), "os": info.get("Os"),
            "size": info.get("Size"), "layers": info.get("RootFS", {}).get("Layers", []),
            "workdir": info.get("Config", {}).get("WorkingDir"), "user": info.get("Config", {}).get("User") or "",
            "env": info.get("Config", {}).get("Env") or [], "entrypoint": info.get("Config", {}).get("Entrypoint"),
            "cmd": info.get("Config", {}).get("Cmd")}


def _build(context: Path, tag: str, *, dockerfile_text: str, timeout: float = 5400, labels: dict | None = None) -> dict:
    (context / "Dockerfile.tb-lane").write_text(dockerfile_text)
    cmd = ["docker", "build", "--platform", PLATFORM, "-f", str(context / "Dockerfile.tb-lane"), "-t", tag,
           "--label", LABEL, "--progress", "plain"]
    for k, v in (labels or {}).items():
        cmd += ["--label", f"{k}={v}"]
    started = time.monotonic()
    proc = run(cmd + [str(context)], timeout=timeout, check=False)
    seconds = time.monotonic() - started
    log = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
    (context / (tag.split(":")[-1] + ".build.log")).write_text(log)
    if proc.returncode:
        return {"result": "failed", "seconds": seconds, "returncode": proc.returncode, "log_tail": log[-4000:],
                "dockerfile_sha256": sha256_bytes(dockerfile_text.encode())}
    info = image_info(tag)
    return {"result": "built", "seconds": seconds, "tag": tag, "image": info,
            "dockerfile_sha256": sha256_bytes(dockerfile_text.encode())}


def pinned_native_dockerfile(task: str, checkout: Path = S.CHECKOUT) -> tuple[str, list[dict], dict]:
    """Pin every FROM to the arm64 variant of the base release the official image was built from."""
    from . import official as O
    text = (checkout / task / "environment/Dockerfile").read_text()
    official = O.official_config(task)
    bases = []

    def pin(match: re.Match) -> str:
        ref = match.group(2)
        found = O.match_base(ref, official)
        if found.get("pinned"):
            O.pull_arm64(found["pinned"])
            found["fallback"] = None
        else:
            found.update(resolve_base(ref))
            found["fallback"] = "official base not identified; current tag digest used"
        bases.append(found)
        return match.group(1) + found["pinned"] + match.group(3)

    text = re.sub(r"(?im)^(FROM\s+)(\S+)(.*)$", pin, text)
    summary = {"image": f"alexgshaw/{task}:20251031", "architecture": official.get("architecture"),
               "created": official.get("created"),
               "env": [e for e in official.get("config", {}).get("Env", []) if "VERSION" in e],
               "layers": len(official.get("rootfs", {}).get("diff_ids", []))}
    return text, bases, summary


def build_native(task: str, checkout: Path = S.CHECKOUT) -> dict:
    context = BUILD_ROOT / task / "native"
    if context.exists():
        shutil.rmtree(context)
    shutil.copytree(checkout / task / "environment", context, symlinks=True)
    dockerfile, bases, official = pinned_native_dockerfile(task, checkout)
    result = _build(context, TAG.format(task=task, kind="native"), dockerfile_text=dockerfile,
                    labels={"terminalbench.task": task, "terminalbench.source": S.SOURCE_REVISION})
    result.update({"bases": bases, "official_image": official, "upstream_dockerfile_sha256": S.sha256_file(checkout / task / "environment/Dockerfile"),
                   "context_manifest": {p.relative_to(checkout / task / "environment").as_posix(): S.sha256_file(p)
                                        for p in sorted((checkout / task / "environment").rglob("*")) if p.is_file()}})
    return result


def export_path(image_id: str, path: str) -> bytes:
    """tar stream of one path from a stopped container of the image (the container is removed by ID)."""
    cid = run(["docker", "create", "--label", LABEL, image_id, "true"]).stdout.decode().strip()
    try:
        return run(["docker", "cp", f"{cid}:{path}", "-"], timeout=1800).stdout
    finally:
        run(["docker", "rm", "-f", cid], check=False)


def probe(image_id: str, script: str, *, timeout: float = 300, network: str = "none") -> subprocess.CompletedProcess:
    return run(["docker", "run", "--rm", "--label", LABEL, "--network", network, "--entrypoint", "sh", image_id,
                "-c", script], timeout=timeout, check=False)


WORKER_DOCKERFILE = """# terminalbench-v2 worker adaptation layer (disclosed in every package)
FROM {native}
USER root
RUN if command -v python3 >/dev/null 2>&1; then echo "python3 present: $(python3 -V)"; else \\
      export DEBIAN_FRONTEND=noninteractive && apt-get update && \\
      apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*; fi
RUN rm -rf {app} && mkdir -p /workspace && ln -s /workspace {app}
WORKDIR {app}
{user}
"""

GRADER_DOCKERFILE = """# terminalbench-v2 private grader layer: native test.sh verifier environment, built ahead (network none at grade)
FROM {uv} AS uv
FROM {worker} AS build
USER root
COPY --from=uv /uv /uvx /tb-uv/
ENV UV_PYTHON_INSTALL_DIR={root}/python UV_CACHE_DIR=/tmp/tb-uv-cache UV_NO_CONFIG=1
RUN /tb-uv/uv venv --python {python} {root}/venv \\
 && /tb-uv/uv pip install --python {root}/venv/bin/python {pins} \\
 && {root}/venv/bin/python -c "import sys,json;print(json.dumps({{'executable':sys.executable,'version':sys.version}}))" \\
      > {root}/interpreter.json \\
 && /tb-uv/uv pip freeze --python {root}/venv/bin/python > {root}/freeze.txt \\
 && rm -rf /tmp/tb-uv-cache
FROM {worker}
USER root
{apt}COPY --from=build {root} {root}
{user}
"""


def id_ref(task: str, kind: str, image_id: str) -> str:
    """BuildKit cannot use a bare image ID in FROM: tag the exact ID immutably and return that tag."""
    ref = f"tr-tbv2/{task}:{kind}-id-{image_id.split(':')[-1][:16]}"
    run(["docker", "tag", image_id, ref])
    if image_info(ref)["id"] != image_id:
        raise RuntimeError("ID tag does not resolve to the recorded image")
    return ref


def build_worker(task: str, native: dict) -> dict:
    context = BUILD_ROOT / task / "worker"
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    info = native["image"]
    if info.get("workdir") != NATIVE_WORKDIR:
        raise ValueError(f"Native WORKDIR is {info.get('workdir')!r}, lane requires {NATIVE_WORKDIR}")
    user = f"USER {info['user']}" if info.get("user") not in ("", "root", "0", "0:0") else ""
    text = WORKER_DOCKERFILE.format(native=id_ref(task, "native", info["id"]), app=NATIVE_WORKDIR, user=user)
    result = _build(context, TAG.format(task=task, kind="worker"), dockerfile_text=text,
                    labels={"terminalbench.task": task, "terminalbench.native_image": info["id"]})
    if result["result"] == "built":
        result["extends_native_layers"] = result["image"]["layers"][:len(info["layers"])] == info["layers"]
    return result


def build_grader(task: str, worker: dict, test_spec: dict) -> dict:
    context = BUILD_ROOT / task / "grader"
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    apt = ""
    if test_spec["apt_packages"]:
        apt = ("RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y "
               + " ".join(shlex.quote(p) for p in test_spec["apt_packages"]) + " && rm -rf /var/lib/apt/lists/*\n")
    user = worker["image"].get("user") or ""
    text = GRADER_DOCKERFILE.format(uv=UV_IMAGE, worker=id_ref(task, "worker", worker["image"]["id"]), root=VERIFIER_ROOT, apt=apt,
                                    user=f"USER {user}" if user not in ("", "root", "0", "0:0") else "",
                                    python=test_spec["python"], pins=" ".join(shlex.quote(p) for p in test_spec["with"]))
    result = _build(context, TAG.format(task=task, kind="grader"), dockerfile_text=text,
                    labels={"terminalbench.task": task, "terminalbench.worker_image": worker["image"]["id"]})
    if result["result"] == "built":
        worker_layers = worker["image"]["layers"]
        result["extends_worker_layers"] = result["image"]["layers"][:len(worker_layers)] == worker_layers
        meta = probe(result["image"]["id"], f"cat {VERIFIER_ROOT}/interpreter.json; echo; cat {VERIFIER_ROOT}/freeze.txt")
        out = meta.stdout.decode(errors="replace").splitlines()
        result["interpreter"] = json.loads(out[0]) if out and out[0].startswith("{") else None
        result["freeze"] = [ln for ln in out[1:] if ln.strip()]
    return result


def app_tar_manifest(data: bytes) -> dict:
    """Entries of the exported /app tar (relative to /app): type, mode, size, sha256, link."""
    entries = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            name = m.name
            if name == "app":
                continue
            if not name.startswith("app/"):
                raise ValueError("Unexpected member outside app/: " + name)
            rel = name[4:]
            kind = "f" if m.isreg() else "d" if m.isdir() else "l" if m.issym() else "h" if m.islnk() else "o"
            digest = sha256_bytes(tf.extractfile(m).read()) if m.isreg() else None
            entries[rel] = {"type": kind, "mode": m.mode & 0o7777, "size": m.size if m.isreg() else None,
                            "sha256": digest, "link": m.linkname if (m.issym() or m.islnk()) else None,
                            "uid": m.uid, "gid": m.gid}
    return dict(sorted(entries.items()))


def workspace_tar_gz(app_tar: bytes) -> bytes:
    """Re-root the exported /app tar at '.', deterministic order/mtimes kept from the image."""
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(app_tar)) as src, \
            gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as dst:
        members = sorted(src.getmembers(), key=lambda m: m.name)
        for m in members:
            if m.name == "app":
                continue
            data = src.extractfile(m).read() if m.isreg() else None
            m.name = "./" + m.name[4:]
            if m.islnk():
                m.linkname = "./" + m.linkname[4:]
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
    return out.getvalue()


def docker_free_bytes() -> int:
    """Free bytes on the Docker VM's root filesystem, root-reserved blocks included."""
    out = run(["docker", "run", "--rm", "--network", "none", "--label", LABEL, PROBE_IMAGE, "stat", "-f", "-c", "%f %S", "/"],
              timeout=120).stdout.decode().split()
    return int(out[0]) * int(out[1])


def require_headroom(what: str) -> int:
    free = docker_free_bytes()
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Docker VM disk headroom {free / 1e9:.1f} GB is below {MIN_FREE_BYTES / 1e9:.1f} GB; "
                           f"not starting {what} (the disk is shared with other sessions)")
    return free
