"""v3 private grading: grading.py's container run with the v3 driver.

The only difference from grading.run_suite is the in-container driver
(grader/grade_driver_v3.py), which skips Git metadata (every ``.git`` path
component) when it copies the submitted W. Observation parsing and check
translation are grading.observe / grading.translate, unchanged. grading.py is
not edited because v1/v2 packages bind its exact bytes.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import uuid

from .grading import GRADER_DIR, HOST_GRACE_SECONDS, LANE_LABEL

DRIVER = "grade_driver_v3.py"


def run_suite(image_id: str, workspace: Path, output: Path, *, pristine: Path | None, spec: dict) -> dict:
    """Run the grading container; returns a receipt. Never raises for task failures."""
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output == workspace or output.is_relative_to(workspace):
        raise ValueError("Private grade output must be outside submitted W")
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("Submitted W must be a real directory")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(output.iterdir()):
        raise ValueError("Preserve earlier private grade output; choose a fresh directory")
    (output / "spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    empty = output / ".empty-pristine"
    if pristine is None:
        empty.mkdir()
    name = "tr-swesmith-long-grade-" + uuid.uuid4().hex[:12]
    command = ["docker", "run", "--rm", "--name", name, "--label", LANE_LABEL, "--network", "none",
               "--memory", "4g", "--cpus", "2", "--pids-limit", "1024", "--entrypoint", "python3",
               "-e", "PYTHONHASHSEED=0", "-e", "TZ=UTC",
               "--mount", f"type=bind,src={workspace},dst=/submission,readonly",
               "--mount", f"type=bind,src={(pristine or empty).resolve()},dst=/pristine,readonly",
               "--mount", f"type=bind,src={GRADER_DIR.resolve()},dst=/grader,readonly",
               "--mount", f"type=bind,src={output},dst=/out",
               image_id, "-I", "-B", "/grader/" + DRIVER, "/out/spec.json"]
    receipt = {"container_name": name, "image_id": image_id, "argv": command, "infrastructure_error": None}
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, timeout=spec["timeout_seconds"] + HOST_GRACE_SECONDS)
        receipt["docker_returncode"] = result.returncode
        (output / "docker.stdout").write_bytes(result.stdout)
        (output / "docker.stderr").write_bytes(result.stderr)
        if result.returncode != 0:
            receipt["infrastructure_error"] = f"grading container exited {result.returncode}"
    except subprocess.TimeoutExpired:
        receipt["infrastructure_error"] = "host transport deadline expired"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    receipt["wall_seconds"] = round(time.monotonic() - started, 3)
    driver = output / "driver.json"
    receipt["driver"] = json.loads(driver.read_text()) if driver.is_file() else None
    if receipt["driver"] is None and receipt["infrastructure_error"] is None:
        receipt["infrastructure_error"] = "grading driver produced no receipt"
    if pristine is None:
        empty.rmdir()
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
