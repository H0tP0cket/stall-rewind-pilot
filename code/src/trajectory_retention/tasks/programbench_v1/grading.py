"""Private ProgramBench grading: exported W -> submission.tar.gz -> upstream Evaluator in the grader venv.

The grader venv holds only programbench==1.2.4 (its wheel lists every hidden test name, so it never enters a
worker image). Hidden-test blobs are linked into a per-grade private directory outside W. The grade uses the
cleanroom image tag only after checking that the tag resolves to the package's pinned image id.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time


GRADER_ENV = "TR_PROGRAMBENCH_GRADER_PYTHON"
DEFAULT_GRADER = Path.home() / ".cache/trajectory-retention/programbench-grader-1.2.4/bin/python"
DRIVER = Path(__file__).with_name("grade_driver.py")
GRADE_TIMEOUT = 4 * 3600
# Upstream run-level error codes caused by the submission itself (no buildable ./executable). Anything else that
# stops the whole run is a grader/infrastructure fault and invalidates the grade.
WORKER_ERROR_CODES = {"compile_failed", "copy_executable_failed", "hash_executable_failed"}


def grader_python() -> Path:
    path = Path(os.environ.get(GRADER_ENV) or DEFAULT_GRADER)
    if not path.exists():
        raise RuntimeError(f"ProgramBench grader venv missing: {path} (see programbench_v1/README in the package notes)")
    return path


def grader_cpus() -> int:
    """Upstream default is 10 CPUs per grading container; cap it at what the Docker host has."""
    ncpu = subprocess.run(["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True).stdout.strip()
    return max(1, min(10, int(ncpu) if ncpu.isdigit() else 10))


def ensure_grader(wheel: Path, venv: Path | None = None) -> Path:
    """Create the isolated grader venv (uv) from the pinned wheel. Idempotent."""
    venv = Path(venv or DEFAULT_GRADER.parent.parent)
    py = venv / "bin/python"
    if not py.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), str(wheel)], check=True)
    version = subprocess.run([str(py), "-c", "import programbench;print(programbench.__version__)"],
                             check=True, capture_output=True, text=True).stdout.strip()
    if version != "1.2.4":
        raise RuntimeError(f"Grader venv has programbench {version}, expected 1.2.4")
    return py


def reference_hash(package: Path) -> str:
    import hashlib
    with tarfile.open(package / "public/workspace.tar") as tf:
        return hashlib.sha256(tf.extractfile("./executable").read()).hexdigest()


def submission_archive(workspace: Path, dest: Path) -> dict:
    """tar.gz of W like the native `tar -czf ... -C /workspace .`; the reference ./executable is left out (the
    upstream grader deletes any submitted ./executable before compiling)."""
    workspace = Path(workspace)
    import hashlib
    skipped, hashes = [], {}
    with tarfile.open(dest, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(workspace.rglob("*")):
            rel = path.relative_to(workspace).as_posix()
            if rel == "executable":
                skipped.append(rel)
                continue
            if path.is_file() and not path.is_symlink() and not os.access(path, os.R_OK):
                path.chmod(path.stat().st_mode | stat.S_IRUSR)  # private scratch copy of W only
            tf.add(path, arcname="./" + rel, recursive=False)
            if path.is_file() and not path.is_symlink():
                hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"skipped": skipped, "bytes": dest.stat().st_size, "hashes": hashes}


def run_private_grade(*, package: Path, manifest: dict, workspace: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    infra: list[dict] = []
    tag_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", manifest["environment_image"]],
                            capture_output=True, text=True).stdout.strip()
    if tag_id != manifest["environment_image_id"]:
        return {"infrastructure_errors": [{"message": f"image tag resolves to {tag_id or 'nothing'}, not the pinned id"}],
                "seconds": time.monotonic() - started}
    blob_root = output / "blobs"
    (blob_root / manifest["upstream_task_id"]).mkdir(parents=True)
    os.symlink((package / "private/blobs/tests").resolve(), blob_root / manifest["upstream_task_id"] / "tests")
    sub = output / "submission.tar.gz"
    packed = submission_archive(workspace, sub)
    cpus = grader_cpus()
    proc = subprocess.run([str(grader_python()), "-B", str(DRIVER), "--package", str(package), "--submission", str(sub),
                           "--blob-root", str(blob_root), "--output", str(output / "upstream"), "--docker-cpus", str(cpus)],
                          capture_output=True, text=True, timeout=GRADE_TIMEOUT)
    (output / "driver.log").write_text(proc.stdout + proc.stderr)
    summary_path = output / "upstream/summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        infra.append({"message": f"grade driver exit {proc.returncode}", "log": str(output / 'driver.log')})
        packed.pop("hashes", None)
        return {"infrastructure_errors": infra, "seconds": time.monotonic() - started, "submission": packed}
    summary = json.loads(summary_path.read_text())
    summary["grader_docker_cpus"] = cpus
    # Tamper guard (declared deviation): upstream eval_clean_hashes can be empty (e.g. seqtk), so a byte copy of the
    # reference executable shipped under another name is flagged here. The grader deletes ./executable before
    # compiling, so a copy must arrive as a submitted file. A built binary equal to the reference is NOT flagged:
    # compiling the upstream source reproduces it byte-for-byte (seen in the reference control); its hash is recorded.
    gold = reference_hash(package)
    tamper = sorted(rel for rel, h in packed.pop("hashes").items() if h == gold)
    summary["built_equals_reference"] = summary.get("executable_hash") == gold
    if summary["error_code"] and summary["error_code"] not in WORKER_ERROR_CODES:
        infra.append({"message": f"upstream evaluator stopped: {summary['error_code']}"})
    return {"infrastructure_errors": infra, "summary": summary, "submission": packed, "tamper": tamper,
            "seconds": time.monotonic() - started}
