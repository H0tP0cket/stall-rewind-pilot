"""Official SWE-smith mirror trees: clean start commit and per-instance bug branches.

The clean tree is the mirror's `main` ("Initial commit" at the profile's pinned
upstream commit). The natural start is clean + the dataset bug patch, verified
against the mirror's instance branch (bug commit = branch~1; the branch head only
removes the FAIL_TO_PASS test files, which this lane keeps as a disclosed
adaptation).
"""
from __future__ import annotations

import hashlib
import io
import os
import posixpath
from pathlib import Path
import shutil
import subprocess
import tarfile

MIRROR_ORG = "swesmith"


def _env(cwd: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")}
    env["GIT_CEILING_DIRECTORIES"] = str(Path(cwd).resolve().parent)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=cwd, env=_env(cwd), input=input_bytes, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def mirror(repo_key: str, cache: Path) -> Path:
    path = cache / "mirrors" / (repo_key + ".git")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--bare", "--single-branch", "--branch", "main",
                        f"https://github.com/{MIRROR_ORG}/{repo_key}.git", str(path)],
                       check=True, capture_output=True, env=_env(path.parent))
    return path


def fetch_branch(repository: Path, branch: str) -> str:
    git(repository, "fetch", "--quiet", "origin", f"+refs/heads/{branch}:refs/heads/{branch}")
    return git(repository, "rev-parse", branch).decode().strip()


def export(repository: Path, rev: str, destination: Path) -> str:
    """Export a commit tree (no .git) and return the commit id."""
    if destination.exists():
        raise FileExistsError(destination)
    commit = git(repository, "rev-parse", rev + "^{commit}").decode().strip()
    data = git(repository, "archive", "--format=tar", commit)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                base = posixpath.dirname(member.name) if member.issym() else ""
                target = posixpath.normpath(posixpath.join(base, member.linkname))
                if member.linkname.startswith("/") or target == ".." or target.startswith("../"):
                    raise ValueError("Link escapes source tree: " + member.name)
        archive.extractall(destination, filter="tar")
    return commit


def manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[rel] = "symlink:" + os.readlink(path)
        elif path.is_file():
            result[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def apply_patch(tree: Path, patch: str) -> None:
    env = _env(tree)
    discovery = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=tree, env=env, capture_output=True)
    if discovery.returncode == 0:
        raise ValueError("Patch target is inside an unrelated Git worktree")
    for args in (["--check"], []):
        result = subprocess.run(["git", "apply", *args, "-"], cwd=tree, env=env, input=patch.encode(),
                                capture_output=True)
        if result.returncode:
            raise ValueError("Bug patch failed: " + result.stderr.decode(errors="replace"))


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, symlinks=True)
