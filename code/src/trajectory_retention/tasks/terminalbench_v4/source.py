"""Pinned Terminal-Bench 4 source identity, the ratified task list and lane paths for the terminalbench-v4 lane.

Upstream: https://github.com/harbor-framework/terminal-bench at tag v4.0.0 (commit 452bf305, 2026-08-25), the release every
TB4 leaderboard trial ran. Task bytes are read from the local checkout ``CHECKOUT`` at that commit; the checkout must be
clean. Lanes never import each other: this module is new for v4 and shares nothing with ``terminalbench_v2``.

Only the tasks the user ratified (2026-10-12) are considered, in the ratified order: tier 1, then tier 2, then tier 3. No family split (the v2 study
has none); each task is its own family.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

LANE = "terminalbench-v4"
SOURCE_URL = "https://github.com/harbor-framework/terminal-bench"
SOURCE_TAG = "v4.0.0"
SOURCE_REVISION = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
LICENSE = "Apache-2.0"
CHECKOUT = Path(os.environ.get("TBV4_CHECKOUT", Path.home() / ".cache/terminalbench/terminal-bench-4"))
TASKS_DIR = CHECKOUT / "tasks"
RATIFIED_TASKS = ("rs-archive-clone", "gsea-proteomics", "mvcc-lsm-compaction", "production-planning",
                  "hof-topology-interpenetration",
                  "atrx-vep-crispr", "satb-audio-transcription", "telecom-entity-resolution",
                  "retro-console-soc", "batched-eval-parity", "session-window-debug")
RATIFICATION = ("user, 2026-10-12: the TB4 tasks of the ranked tiers (\"yeah these are good for now\"), in this order: "
                "tier 1 (first five), tier 2 (next three), tier 3 (last three); no others")
SPLIT_POLICY = "none: the v2 study has no family split (user decision 2026-10-06); every admitted task may supply stalls"
# Harbor's separate-verifier semantics are reproduced from this source revision (read, never imported or run).
HARBOR_URL = "https://github.com/harbor-framework/harbor"
HARBOR_REVISION = "4778ceba98314a3b0fe119249f7ac84fc0af08b7"
HARBOR_FILES = ("src/harbor/trial/trial.py (_run_separate_verifier)", "src/harbor/trial/artifact_handler.py",
                "src/harbor/verifier/verifier.py", "src/harbor/environments/base.py (empty_dirs, ensure_dirs)",
                "src/harbor/environments/docker/docker.py (exec)", "src/harbor/environments/docker/docker_unix.py",
                "src/harbor/utils/scripts.py (build_execution_command)",
                "src/harbor/environments/docker/docker-compose-build.yaml (keep-alive command)")

WORKTREE = Path(__file__).resolve().parents[4]


def repository_root(path: Path = WORKTREE) -> Path:
    """The main checkout that owns this tree (a git worktree's .git file points into the main .git directory).

    Local, gitignored evidence goes under the main checkout's artifacts/ so it outlives a worktree. Plain file reads;
    no git command runs here."""
    marker = path / ".git"
    if marker.is_file():
        text = marker.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = (path / text.split(":", 1)[1].strip()).resolve()
            common = gitdir / "commondir"
            if common.is_file():
                return (gitdir / common.read_text().strip()).resolve().parent
    return path


EVIDENCE = Path(os.environ.get("TBV4_EVIDENCE", repository_root() / "artifacts" / "v2" / "lane-v4"))


def package_task_id(task: str) -> str:
    return "terminalbench-v4-" + task


def family_id(task: str) -> str:
    return "terminal-bench-4:" + task


def git(*args: str, checkout: Path = CHECKOUT) -> bytes:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True).stdout


def verify_checkout(checkout: Path = CHECKOUT) -> dict:
    head = git("rev-parse", "HEAD", checkout=checkout).decode().strip()
    if head != SOURCE_REVISION:
        raise ValueError(f"Terminal-Bench 4 checkout at {head}, expected {SOURCE_REVISION}")
    if git("status", "--porcelain", "--untracked-files=no", checkout=checkout).strip():
        raise ValueError("Terminal-Bench 4 checkout has local changes")
    missing = [t for t in RATIFIED_TASKS if not (checkout / "tasks" / t / "task.toml").is_file()]
    if missing:
        raise ValueError("Ratified tasks missing from the checkout: " + ", ".join(missing))
    return {"source_url": SOURCE_URL, "source_tag": SOURCE_TAG, "source_revision": head}


def task_tree(task: str, checkout: Path = CHECKOUT) -> dict[str, str]:
    """{path: git blob sha} of one task at the pinned revision (every file, metadata included)."""
    out = {}
    for line in git("ls-tree", "-r", SOURCE_REVISION, "--", f"tasks/{task}/", checkout=checkout).decode().splitlines():
        meta, path = line.split("\t", 1)
        out[path[len(f"tasks/{task}/"):]] = meta.split()[2]
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_dir(task: str) -> Path:
    if task not in RATIFIED_TASKS:
        raise ValueError(f"{task} is not one of the ratified terminalbench-v4 tasks")
    return TASKS_DIR / task


def remove_tree(path) -> None:
    """shutil.rmtree for host copies of W: a captured W can hold read-only directories (rs-archive-clone's
    /app/artifacts is 0555), which plain rmtree cannot empty. Only our own scratch and evidence copies are removed."""
    import os
    import shutil
    import stat as st

    def retry(func, target, _exc):
        parent = os.path.dirname(target)
        os.chmod(parent, os.stat(parent).st_mode | st.S_IWUSR | st.S_IXUSR | st.S_IRUSR)
        if os.path.isdir(target) and not os.path.islink(target):
            os.chmod(target, os.stat(target).st_mode | st.S_IWUSR | st.S_IXUSR | st.S_IRUSR)
            shutil.rmtree(target, onexc=retry)
        else:
            func(target)

    shutil.rmtree(path, onexc=retry)
