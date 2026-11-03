"""Build and verify a frozen ProgramBench task package.

Layout (every file hashed in package.json):
  public/instruction.md    native task text the worker receives
  public/workspace.tar     the cleanroom image's /workspace, byte-for-byte (numeric owners, modes; the reference
                           ./executable stays root-owned and execute-only)
  private/task.yaml        upstream task metadata (repo, commit, eval_clean_hashes)
  private/tests.json       upstream per-branch test lists and ignored tests (frozen denominator)
  private/blobs/tests/<branch>.tar.gz   hidden tests + upstream source + reference executable (grader-only)

Only public/ ever reaches a worker. private/ is read by the host-side grader driver alone.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

import yaml

from . import source as S

PACKAGE_VERSION = "programbench-package/v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_of(test_name: str) -> str:
    """Feedback group: the test module stem (``eval.tests.test_basic.TestX.test_y`` -> ``test_basic``)."""
    parts = test_name.split(".")
    for part in parts[:-1]:
        if part.startswith("test_"):
            return part
    return parts[0] if len(parts) > 1 else "tests"


def frozen_tests(tests: dict) -> dict:
    """Scored test identities (``branch/name``) from active branches minus ignored tests."""
    scored, ignored, groups = [], [], {}
    for branch, info in sorted((tests.get("branches") or {}).items()):
        if info.get("ignored"):
            continue
        skip = {t["name"] for t in info.get("ignored_tests") or []}
        for name in info.get("tests") or []:
            key = f"{branch}/{name}"
            (ignored if name in skip else scored).append(key)
            if name not in skip:
                groups[group_of(name)] = groups.get(group_of(name), 0) + 1
    return {"scored": scored, "ignored": ignored, "groups": dict(sorted(groups.items()))}


def native_task_text(agent_config: dict) -> str:
    system = agent_config["agent"]["system_template"]
    first = "You are a helpful assistant that can interact with a computer.\n"
    if not system.startswith(first):
        raise ValueError("Native system template no longer starts with the shared first line")
    instance = agent_config["agent"]["instance_template"]
    start, end = instance.find("<system_information>"), instance.find("</system_information>")
    if start < 0 or end < 0:
        raise ValueError("Native instance template lost its system_information block")
    instance = instance[:start].rstrip("\n") + "\n" + instance[end + len("</system_information>"):].lstrip("\n")
    if "{{" in instance or "{%" in instance:
        raise ValueError("Native instance template has unrendered variables")
    return system[len(first):].lstrip("\n") + "\n" + instance


def export_native_workspace(image_id: str) -> bytes:
    cmd = ["docker", "run", "--rm", "--network", "none", "--user", "0:0", "--entrypoint", "tar", image_id,
           "-C", "/workspace", "--numeric-owner", "--format=pax", "-cf", "-", "."]
    return subprocess.run(cmd, check=True, capture_output=True, timeout=600).stdout


def build(*, instance_id: str, wheel: Path, blob_dir: Path, agent_config: Path, out: Path,
          family_split: str = "unassigned") -> dict:
    wheel, blob_dir, out = Path(wheel), Path(blob_dir), Path(out)
    if _sha(wheel) != S.WHEEL_SHA256:
        raise ValueError("ProgramBench wheel differs from the pinned release")
    if out.exists():
        raise ValueError("Package directory already exists; packages are immutable")
    image_ref = f"programbench/{instance_id.replace('__', '_1776_')}:{S.IMAGE_TAG}"
    inspect = json.loads(subprocess.run(["docker", "image", "inspect", image_ref], check=True, capture_output=True).stdout)[0]
    (out / "public").mkdir(parents=True)
    (out / "private/blobs/tests").mkdir(parents=True)
    with zipfile.ZipFile(wheel) as zf:
        base = f"programbench/data/tasks/{instance_id}/"
        (out / "private/task.yaml").write_bytes(zf.read(base + "task.yaml"))
        (out / "private/tests.json").write_bytes(zf.read(base + "tests.json"))
    tests = json.loads((out / "private/tests.json").read_text())
    task_yaml = yaml.safe_load((out / "private/task.yaml").read_text())
    active = [b for b, info in sorted(tests["branches"].items()) if not info.get("ignored")]
    for branch in active:
        shutil.copyfile(blob_dir / f"{branch}.tar.gz", out / f"private/blobs/tests/{branch}.tar.gz")
    config = yaml.safe_load(Path(agent_config).read_text())
    (out / "public/instruction.md").write_text(native_task_text(config), encoding="utf-8")
    workspace = export_native_workspace(inspect["Id"])
    (out / "public/workspace.tar").write_bytes(workspace)
    with tarfile.open(fileobj=io.BytesIO(workspace)) as tf:
        members = {m.name.removeprefix("./"): m for m in tf.getmembers()}
    exe = members.get("executable")
    if exe is None or exe.uid != 0 or (exe.mode & 0o777) != 0o111:
        raise ValueError("Native workspace lost the root-owned execute-only reference executable")
    frozen = frozen_tests(tests)
    repo = instance_id.split(".")[0]
    manifest = {
        "package_version": PACKAGE_VERSION, "task_id": f"programbench-{instance_id}-v1", "upstream_task_id": instance_id,
        "family_id": f"programbench:{repo}", "family_split": family_split, "source_benchmark": "ProgramBench",
        "source_url": S.HARNESS_URL, "source_revision": S.HARNESS_REVISION, "wheel": S.WHEEL_NAME,
        "wheel_sha256": S.WHEEL_SHA256, "tests_dataset": S.TESTS_DATASET, "tests_dataset_revision": S.TESTS_DATASET_REVISION,
        "agent_config": S.AGENT_CONFIG_URL, "agent_config_sha256": _sha(Path(agent_config)),
        "upstream": {"repository": task_yaml.get("repository"), "commit": task_yaml.get("commit"),
                     "language": task_yaml.get("language"), "eval_clean_hashes": len(task_yaml.get("eval_clean_hashes") or [])},
        "environment_image": image_ref, "environment_image_id": inspect["Id"],
        "environment_image_digests": inspect.get("RepoDigests") or [], "environment_architecture": inspect.get("Architecture"),
        "grader_image": image_ref, "active_branches": active,
        "frozen_tests": {"scored": len(frozen["scored"]), "ignored": len(frozen["ignored"]), "groups": frozen["groups"],
                         "rule": "active branches' tests minus ignored_tests; group = test module stem"},
        "lifecycle": S.LIFECYCLE, "native_environment": S.NATIVE_ENVIRONMENT, "native_budget": S.NATIVE_BUDGET,
        "worker_environment": {k: list(v) if isinstance(v, tuple) else v for k, v in S.WORKER_ENVIRONMENT.items()},
        "deviations": list(S.DEVIATIONS),
        "files": {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    (out / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_package(package: Path) -> dict:
    package = Path(package)
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("package_version") != PACKAGE_VERSION:
        raise ValueError("Unsupported ProgramBench package version")
    present = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()} - {"package.json"}
    if present != set(manifest["files"]):
        raise ValueError("Frozen package changed: file set")
    for rel, digest in manifest["files"].items():
        if _sha(package / rel) != digest:
            raise ValueError(f"Frozen package changed: {rel}")
    return manifest
