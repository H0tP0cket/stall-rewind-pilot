"""Build and verify a frozen NL2Repo task package and its offline worker image.

Layout (hashed in package.json):
  public/instruction.md          native prompt
  public/workspace/start.md      the only native input file (W0)
  private/test_commands.json     native grading commands
  private/test_files.json        native test paths removed from the agent's workspace before grading
  private/test_case_count.txt    native fixed denominator
  private/worker-requirements.txt  the grading image's third-party packages minus the target (worker image input)
  private/Dockerfile.worker      worker image recipe (official python at the grading version, pinned by digest + those packages)

The worker image never contains the target package, the hidden tests or the reference packaging; it is built from a
clean base, not from the grading image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import source as S

PACKAGE_VERSION = "nl2repo-package/v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect(ref: str) -> dict:
    return json.loads(subprocess.run(["docker", "image", "inspect", ref], check=True, capture_output=True).stdout)[0]


def grading_packages(image_id: str) -> list[dict]:
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", image_id, "-m", "pip",
                          "list", "--format=json", "-v", "--disable-pip-version-check"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def split_target(packages: list[dict], task: str) -> tuple[list[dict], list[dict]]:
    """Target = the task's own distribution plus anything not installed into site-packages (editable or local-path
    installs of the reference). Some grading images (deepdiff) install the reference as a normal site-package."""
    target = [p for p in packages if p.get("editable_project_location") or _norm(p["name"]) == _norm(task)
              or "site-packages" not in (p.get("location") or "site-packages")]
    names = {p["name"] for p in target}
    return target, [p for p in packages if p["name"] not in names and p["name"] not in ("pip", "setuptools", "wheel")]


def build(*, task: str, bench_dir: Path, out: Path) -> dict:
    bench, out = Path(bench_dir), Path(out)
    if out.exists():
        raise ValueError("Package directory already exists; packages are immutable")
    src = bench / "test_files" / task
    grading_ref = S.GRADING_IMAGE.format(task=task)
    grading = _inspect(grading_ref)
    version = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", grading["Id"], "-c",
                              "import platform;print(platform.python_version())"], check=True, capture_output=True,
                             text=True).stdout.strip()
    base_digest = S.WORKER_BASES[version]
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", "-q", base_digest], check=True, capture_output=True)
    base = _inspect(base_digest)
    if grading.get("Architecture") != "amd64" or base.get("Architecture") != "amd64":
        raise ValueError("NL2Repo images must be linux/amd64")
    target, deps = split_target(grading_packages(grading["Id"]), task)
    if not target:
        # Some grading images (asteval) keep the reference only as a source tree in /project, not as a distribution.
        probe = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "test", grading["Id"],
                                "-d", f"/project/{task}"], capture_output=True)
        if probe.returncode != 0:
            raise ValueError("Could not identify the target package in the grading image")
        target = [{"name": task, "version": None, "reference_tree": "/project"}]
    (out / "public/workspace").mkdir(parents=True)
    (out / "private").mkdir()
    shutil.copyfile(src / "start.md", out / "public/workspace/start.md")
    (out / "public/instruction.md").write_text(S.NATIVE_PROMPT + "\n", encoding="utf-8")
    for name in ("test_commands.json", "test_files.json", "test_case_count.txt"):
        shutil.copyfile(src / name, out / "private" / name)
    reqs = "".join(f"{p['name']}=={p['version']}\n" for p in sorted(deps, key=lambda p: p["name"].lower()))
    (out / "private/worker-requirements.txt").write_text(reqs)
    (out / "private/Dockerfile.worker").write_text(
        f"FROM --platform=linux/amd64 {base_digest}\n"
        "COPY worker-requirements.txt /tmp/worker-requirements.txt\n"
        "RUN pip install --no-cache-dir --no-deps -r /tmp/worker-requirements.txt && rm /tmp/worker-requirements.txt "
        "&& rm -rf /root/.cache\n"
        "WORKDIR /workspace\n")
    tag = f"tr-nl2repo-worker:{task}-v1"
    subprocess.run(["docker", "build", "--platform", "linux/amd64", "-q", "-t", tag, "-f",
                    str(out / "private/Dockerfile.worker"), str(out / "private")], check=True, capture_output=True)
    worker = _inspect(tag)
    count = int((out / "private/test_case_count.txt").read_text().strip())
    manifest = {
        "package_version": PACKAGE_VERSION, "task_id": f"nl2repo-{task}-v1", "upstream_task_id": task,
        "family_id": f"nl2repo:{task}", "family_split": "unassigned", "source_benchmark": "NL2RepoBench",
        "source_url": S.SOURCE_URL, "source_revision": S.SOURCE_REVISION,
        "grading_image": grading_ref, "grading_image_id": grading["Id"], "grading_image_digests": grading.get("RepoDigests") or [],
        "worker_base": base_digest, "python_version": version, "environment_image": tag, "environment_image_id": worker["Id"],
        "environment_architecture": worker.get("Architecture"),
        "target_packages": [{k: p.get(k) for k in ("name", "version", "reference_tree") if k in p} for p in target],
        "test_case_count": count, "lifecycle": S.LIFECYCLE, "native_budget": S.NATIVE_BUDGET,
        "deviations": list(S.DEVIATIONS),
        "files": {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (out / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_package(package: Path) -> dict:
    package = Path(package)
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("package_version") != PACKAGE_VERSION:
        raise ValueError("Unsupported NL2Repo package version")
    present = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()} - {"package.json"}
    if present != set(manifest["files"]):
        raise ValueError("Frozen package changed: file set")
    for rel, digest in manifest["files"].items():
        if _sha(package / rel) != digest:
            raise ValueError(f"Frozen package changed: {rel}")
    return manifest
