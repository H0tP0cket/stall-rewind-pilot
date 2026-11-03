"""Private NL2Repo grading: the native post-processor on an exported W, in a disposable grading container.

Steps (native order): copy W, delete the agent's packaging files anywhere and the task's test paths, copy the result
over the grading image's /workspace (its hidden tests and reference packaging stay), set PYTHONPATH=/workspace, run
each native command split with shlex and no shell, score sum("(\\d+) passed") / test_case_count capped at 1. Added:
a per-command timeout and --junitxml on pytest commands (exact passed/total and failing test files).
"""
from __future__ import annotations

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
import uuid
import xml.etree.ElementTree as ET

from . import source as S

LABEL = "trajectory-retention.lane=nl2repo"


def prepare_submission(workspace: Path, dest: Path, test_files: list[str]) -> dict:
    shutil.copytree(workspace, dest, symlinks=True)
    removed = []
    for root, _dirs, files in os.walk(dest):
        for name in files:
            if name in S.PACKAGE_FILES or name in S.TAMPER_FILES or name.endswith(".pth"):
                os.remove(os.path.join(root, name))
                removed.append(os.path.relpath(os.path.join(root, name), dest))
    for rel in test_files:
        target = dest / rel
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
            removed.append(rel + "/")
        elif target.exists() or target.is_symlink():
            target.unlink()
            removed.append(rel)
    return {"removed": removed}


def _tar(directory: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(directory.rglob("*")):
            info = tf.gettarinfo(str(path), arcname="./" + path.relative_to(directory).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            if path.is_file() and not path.is_symlink():
                with open(path, "rb") as handle:
                    tf.addfile(info, handle)
            else:
                tf.addfile(info)
    return buf.getvalue()


def with_junit(command: str) -> str:
    return command + f" --junitxml={S.JUNIT_PATH}" if "pytest" in command.lower() else command


def junit_counts(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    cases = []
    for case in root.iter("testcase"):
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = child.tag
            elif child.tag == "skipped":
                outcome = "skipped"
        classname = case.get("classname") or ""
        cases.append({"file": (classname.split(".")[:-1] or [classname]), "outcome": outcome, "classname": classname,
                      "name": case.get("name")})
    return {"cases": cases}


def test_file_of(classname: str, name: str) -> str:
    """JUnit classname `tests.test_jwt.TestJWT` -> `tests/test_jwt.py` (the module path, no function or class)."""
    parts = classname.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test"):
            return "/".join(parts[:i]) + ".py"
    return classname.replace(".", "/") + ".py" if classname else "unknown"


def run_private_grade(*, package: Path, manifest: dict, workspace: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    test_files = json.loads((package / "private/test_files.json").read_text())
    commands = json.loads((package / "private/test_commands.json").read_text())
    count = manifest["test_case_count"]
    sub = output / "submission"
    prep = prepare_submission(Path(workspace), sub, test_files)
    name = f"tr-nl2repo-grade-{uuid.uuid4().hex[:10]}"
    run = ["docker", "run", "-d", "--name", name, "--label", LABEL, "-w", "/workspace", "-e",
           "PYTHONPATH=/workspace:", "--entrypoint", "tail", manifest["grading_image_id"], "-f", "/dev/null"]
    infra, results = [], []
    try:
        proc = subprocess.run(run, capture_output=True, text=True)
        if proc.returncode != 0:
            return {"infrastructure_errors": [{"message": "grading container: " + proc.stderr.strip()[-500:]}],
                    "seconds": time.monotonic() - started, "submission": prep}
        cp = subprocess.run(["docker", "exec", "-i", name, "tar", "-xf", "-", "-C", "/workspace"], input=_tar(sub),
                            capture_output=True)
        if cp.returncode != 0:
            return {"infrastructure_errors": [{"message": "copy-in failed: " + cp.stderr.decode()[-500:]}],
                    "seconds": time.monotonic() - started, "submission": prep}
        for command in commands:
            argv = shlex.split(with_junit(command))
            t0 = time.monotonic()
            try:
                r = subprocess.run(["docker", "exec", "-w", "/workspace", name, "timeout", "-s", "KILL",
                                    str(S.GRADE_COMMAND_TIMEOUT), *argv], capture_output=True, text=True,
                                   timeout=S.GRADE_COMMAND_TIMEOUT + 120)
                results.append({"command": command, "exit_code": r.returncode, "seconds": time.monotonic() - t0,
                                "output": r.stdout + r.stderr})
            except subprocess.TimeoutExpired:
                results.append({"command": command, "exit_code": -9, "seconds": time.monotonic() - t0, "output": ""})
        junit = subprocess.run(["docker", "exec", name, "cat", S.JUNIT_PATH], capture_output=True, text=True)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    native_passed = sum(int(m.group(1)) for r in results if "pytest" in r["command"].lower()
                        for line in r["output"].split("\n") for m in [re.search(r"(\d+) passed", line)] if m)
    (output / "commands.json").write_text(json.dumps(results, indent=1))
    cases = junit_counts(junit.stdout)["cases"] if junit.returncode == 0 and junit.stdout.strip() else []
    return {"infrastructure_errors": infra, "seconds": time.monotonic() - started, "submission": prep,
            "native": {"passed_regex": native_passed, "test_case_count": count,
                       "native_score": min(native_passed / count, 1) if count else 0.0,
                       "exit_codes": [r["exit_code"] for r in results], "junit_found": bool(cases)},
            "cases": [{"file": test_file_of(c["classname"], c["name"]), "outcome": c["outcome"]} for c in cases]}
