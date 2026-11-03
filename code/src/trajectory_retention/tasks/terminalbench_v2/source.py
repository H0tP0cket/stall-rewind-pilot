"""Pinned Terminal-Bench 2 source identity and task grouping for the terminalbench-v2 lane.

Copied from the terminalbench-v1 lane (lanes never import each other). Upstream:
https://github.com/laude-institute/terminal-bench-2 (Harbor task format). Task bytes are read from the local checkout at
``SOURCE_REVISION`` and each task's instruction, environment, solution and tests are checked byte-identical to the
Harbor registry revision ``REGISTRY_REVISION`` (69671fba); only task.toml/README.md/.gitignore differ (metadata).

The v2 study has no family split (user decision). Tasks that share reference material are still grouped into one
family id so that analysis can cluster them; the grouping is copied from the v1 lane's recorded scan.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

LANE = "terminalbench-v2"
SOURCE_URL = "https://github.com/laude-institute/terminal-bench-2"
SOURCE_REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"  # 2026-04-29 "Add task metadata to task manifests"
REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
REGISTRY_DATASET = "terminal-bench@2.0"
REGISTRY_REVISION = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"  # git_commit_id of every terminal-bench@2.0 task
LICENSE = "Apache-2.0"
CHECKOUT = Path.home() / ".cache/terminalbench/terminal-bench-2"
TASK_COUNT = 89
CONTENT_EXCLUDES = ("README.md", ".gitignore", "task.toml")

# Cross-task shared reference material (recorded by the v1 lane's scan of the pinned Dockerfiles, solutions and tests).
SHARED_REFERENCE_GROUPS = {
    "build-pmars": ("build-pmars", "winning-avg-corewars"),
    "make-doom-for-mips": ("make-doom-for-mips", "make-mips-interpreter"),
    "path-tracing": ("path-tracing", "path-tracing-reverse"),
    "qemu-alpine-ssh": ("qemu-alpine-ssh", "qemu-startup"),
}
SPLIT_POLICY = "none: the v2 study has no family split (user decision 2026-10-06); every admitted task may supply stalls"


def package_task_id(task: str) -> str:
    return "terminalbench-v2-" + task


def family_id(task: str) -> str:
    key = next((k for k, members in SHARED_REFERENCE_GROUPS.items() if task in members), task)
    return "terminal-bench:" + key


def git(*args: str, checkout: Path = CHECKOUT) -> bytes:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True).stdout


def task_ids(checkout: Path = CHECKOUT) -> list[str]:
    return sorted(p.name for p in Path(checkout).iterdir() if (p / "task.toml").is_file())


def verify_checkout(checkout: Path = CHECKOUT) -> dict:
    head = git("rev-parse", "HEAD", checkout=checkout).decode().strip()
    if head != SOURCE_REVISION:
        raise ValueError(f"Terminal-Bench checkout at {head}, expected {SOURCE_REVISION}")
    if git("status", "--porcelain", "--untracked-files=no", checkout=checkout).strip():
        raise ValueError("Terminal-Bench checkout has local changes")
    tasks = task_ids(checkout)
    if len(tasks) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} tasks, found {len(tasks)}")
    return {"source_url": SOURCE_URL, "source_revision": head, "task_count": len(tasks)}


def content_tree(task: str, revision: str, checkout: Path = CHECKOUT) -> dict[str, str]:
    """{path: blob sha} of a task at a revision, excluding metadata-only files."""
    out = {}
    for line in git("ls-tree", "-r", revision, "--", task + "/", checkout=checkout).decode().splitlines():
        meta, path = line.split("\t", 1)
        rel = path[len(task) + 1:]
        if rel not in CONTENT_EXCLUDES:
            out[rel] = meta.split()[2]
    return out


def registry_identity(task: str, checkout: Path = CHECKOUT) -> dict:
    here, registry = content_tree(task, SOURCE_REVISION, checkout), content_tree(task, REGISTRY_REVISION, checkout)
    return {"registry_dataset": REGISTRY_DATASET, "registry_revision": REGISTRY_REVISION,
            "content_identical_to_registry": here == registry,
            "content_differences": sorted(set(here.items()) ^ set(registry.items()))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
