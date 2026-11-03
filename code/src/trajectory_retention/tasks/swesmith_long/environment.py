"""Native arm64 rebuild of an official SWE-smith Python environment.

The official images are x86_64; under this host's Docker Desktop emulation a
Python loop measured ~27x slower than native, so worker/grader images are
rebuilt natively from the official environment's exact package pins. The
project itself is installed editable against /workspace from the clean mirror
tree through a build-time bind mount, then /workspace is emptied in the same
layer: no project source (clean or buggy) is stored in any image layer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

from .official import installed_packages, normalize

LANE_LABEL = "trajectory-retention.lane=swesmith-long"
BASES = {
    "3.9.21": "python:3.9.21-slim-bookworm@sha256:40007fe18a72a2e7166be350d52dab86b9fe18f2de08e6a38e26422fb247e81e",
    "3.10.16": "python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512",
    "3.11.11": "python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5",
    "3.12.8": "python:3.12.8-slim-bookworm@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0",
}
MINOR_DEFAULT = {"3.9": "3.9.21", "3.10": "3.10.16", "3.11": "3.11.11", "3.12": "3.12.8"}


def base_for(official_python: str) -> tuple[str, str]:
    if official_python in BASES:
        return official_python, BASES[official_python]
    minor = ".".join(official_python.split(".")[:2])
    if minor not in MINOR_DEFAULT:
        raise ValueError("Unsupported official Python version: " + official_python)
    chosen = MINOR_DEFAULT[minor]
    return chosen, BASES[chosen]


def build_backend(clean: Path) -> str:
    pyproject = clean / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text())
        backend = data.get("build-system", {}).get("build-backend")
        if backend:
            return backend
    if (clean / "setup.py").is_file() or (clean / "setup.cfg").is_file():
        return "setuptools.build_meta:__legacy__"
    raise ValueError("No recognizable Python build configuration")


# Declared, family-specific pin exclusions (disclosed in family/package records).
# Only packages that the repository's test suite never imports are eligible.
EXCLUDED_PINS = {
    "pdfminer__pdfminer.six.1a8bd2f7": {
        "atheris": "fuzzing-only dependency (imported only under fuzzing/, never by tests/); no linux/arm64 "
                   "wheel and its sdist requires a libFuzzer-enabled Clang",
    },
}


def requirements(official: dict, project: str, exclude: dict | None = None) -> tuple[list[str], dict]:
    exclude = {normalize(k): v for k, v in (exclude or {}).items()}
    pins, dropped = [], []
    projects = [p for p in official["packages"] if p["editable"]]
    if [normalize(p["name"]) for p in projects] != [normalize(project)]:
        raise ValueError(f"Expected exactly the project editable, found {[p['name'] for p in projects]}")
    for package in official["packages"]:
        name = normalize(package["name"])
        if package["editable"] or name == "pip" or name in exclude:
            dropped.append({"name": package["name"], "version": package["version"],
                            "reason": "project editable install" if package["editable"] else
                            "base image installer" if name == "pip" else "declared exclusion: " + exclude[name]})
            continue
        if package["direct_url"] and not package["direct_url"].get("url", "").startswith("file:///croot"):
            raise ValueError("Non-index direct URL dependency: " + package["name"])
        pins.append(f"{package['name']}=={package['version']}")
    return sorted(set(pins), key=str.lower), {"dropped": dropped, "project": projects[0]}


DOCKERFILE = """# syntax=docker/dockerfile:1
FROM {base}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC LANG=C.UTF-8
{apt}COPY requirements.txt collect_generated.py /opt/swesmith-long/
RUN python -m pip install --no-deps {binary}-r /opt/swesmith-long/requirements.txt
RUN python -m pip check > /opt/swesmith-long/pip-check.txt 2>&1 || true
RUN --mount=type=bind,source=clean,target=/opt/clean-source \\
    mkdir -p /workspace && cp -a /opt/clean-source/. /workspace/ \\
 && SETUPTOOLS_SCM_PRETEND_VERSION={version} SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{scm_name}={version} \\
    python -m pip install --no-deps {editable_settings}-e /workspace \\
 && python /opt/swesmith-long/collect_generated.py /opt/clean-source /workspace /opt/swesmith-long/generated \\
 && rm -rf /workspace && mkdir /workspace && rm -rf /root/.cache /tmp/* \\
 && python -m pip freeze --all > /opt/swesmith-long/environment.lock
WORKDIR /workspace
LABEL trajectory-retention.lane="swesmith-long" trajectory-retention.swesmith-repo="{repo_key}"
"""


def write_context(repo_key: str, official: dict, clean: Path, context: Path, *, project: str,
                  compile_tools: bool) -> dict:
    version, base = base_for(official["python_version"])
    pins, notes = requirements(official, project, EXCLUDED_PINS.get(repo_key))
    backend = build_backend(clean)
    editable_settings = "--config-settings editable_mode=compat " if backend.startswith("setuptools") else ""
    context.mkdir(parents=True, exist_ok=True)
    (context / "requirements.txt").write_text("\n".join(pins) + "\n")
    (context / "collect_generated.py").write_bytes(Path(__file__).with_name("build_support").joinpath("collect_generated.py").read_bytes())
    apt = ("RUN apt-get update && apt-get install -y --no-install-recommends build-essential "
           "&& rm -rf /var/lib/apt/lists/*\n") if compile_tools else ""
    text = DOCKERFILE.format(base=base, apt=apt, binary="" if compile_tools else "--only-binary=:all: ",
                             version=notes["project"]["version"] or "0.0.0",
                             scm_name=re.sub(r"[-.]", "_", normalize(project)).upper(),
                             editable_settings=editable_settings, repo_key=repo_key)
    (context / "Dockerfile").write_text(text)
    if not (context / "clean").exists():
        raise ValueError("Clean source must be exported into the build context")
    return {"python_official": official["python_version"], "python_base": version, "base_image": base,
            "requirements_sha256": hashlib.sha256((context / "requirements.txt").read_bytes()).hexdigest(),
            "dockerfile_sha256": hashlib.sha256(text.encode()).hexdigest(), "build_backend": backend,
            "editable_mode": "compat" if editable_settings else "backend-default", "compile_tools": compile_tools,
            "pins": len(pins), "dropped": notes["dropped"], "project_version": notes["project"]["version"]}


def build(context: Path, tag: str, log: Path) -> str:
    with log.open("wb") as handle:
        result = subprocess.run(["docker", "build", "--platform", "linux/arm64", "--label", LANE_LABEL,
                                 "-t", tag, str(context)], stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"docker build failed ({result.returncode}); see {log}")
    return image_id(tag)


def image_id(tag: str) -> str:
    return json.loads(subprocess.run(["docker", "image", "inspect", tag], check=True, capture_output=True).stdout)[0]["Id"]


def extract_path(image: str, path: str, destination: Path) -> None:
    """Copy one image path out without starting the image (docker create/cp)."""
    import io
    import tarfile
    created = subprocess.run(["docker", "create", "--label", LANE_LABEL, image, "true"],
                             check=True, capture_output=True).stdout.decode().strip()
    try:
        data = subprocess.run(["docker", "cp", f"{created}:{path}", "-"], check=True, capture_output=True).stdout
    finally:
        subprocess.run(["docker", "rm", "-f", created], capture_output=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        prefix = Path(path).name + "/"
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("Unexpected link/device in generated files: " + member.name)
            if not member.name.startswith(prefix):
                continue
            member.name = member.name[len(prefix):]
            archive.extract(member, destination, filter="data")
