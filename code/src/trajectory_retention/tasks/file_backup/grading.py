"""Task-private Docker grading, operating only on an execution-exported W."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from trajectory_retention.contracts import RuntimeClass, RuntimeExport
from .packaging import file_manifest, verify_package


def run_grade(package: Path, workspace_export: Path, *, stage: int,
              output: Path, scratch_root: Path, adapted: bool = False,
              runtime: RuntimeExport | None = None) -> dict:
    operation_started = time.monotonic()
    manifest = verify_package(package)
    workspace_export = workspace_export.resolve()
    if not workspace_export.is_dir() or stage not in range(1, 5):
        raise ValueError("Expected existing isolated workspace and stage 1..4")
    output = output.resolve()
    if output == workspace_export or workspace_export in output.parents:
        raise ValueError("Private grading output cannot be inside submitted W")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    scratch_root.mkdir(parents=True, exist_ok=True)
    before = file_manifest(workspace_export)
    name = "retention-file-backup-grade-" + uuid.uuid4().hex[:12]
    module_dir = Path(__file__).parent
    scoring_dir = module_dir.parents[1] / "scoring" / "file_backup"
    with tempfile.TemporaryDirectory(prefix="file-backup-", dir=scratch_root) as tmp:
        root = Path(tmp).resolve()
        work = root / "workspace"
        shutil.copytree(workspace_export, work, symlinks=True)
        private = root / "private"
        private.mkdir(mode=0o700)
        shutil.copytree(package / "private/tests", private / "tests")
        instrumentation = private / "instrumentation"
        instrumentation.mkdir(mode=0o700)
        (instrumentation / "pytest.ini").write_text("[pytest]\n")
        shutil.copy2(module_dir / "native_capture.py", instrumentation / "scb_probe_plugin.py")
        shutil.copy2(module_dir / "submission.py", instrumentation / "submission.py")
        # Root-only parent prevents submitted code from reading test sources,
        # notwithstanding world-readable files inside their immutable bind.
        (private / "tests").chmod(0o700)
        test_paths = [f"/tests/test_checkpoint_{i}.py" for i in range(1, stage + 1)]
        if adapted:
            shutil.copy2(scoring_dir / "supplemental_checks.py", private / "tests/test_supplemental.py")
            rubric = json.loads((package / "private/rubric.json").read_text())
            test_paths += ["/tests/test_supplemental.py::test_" + c["check_id"].split("::", 1)[1]
                           for c in rubric["proposed_supplemental_checks"] if stage in c["applicable_stages"]]
        args = ["-c", "/instrumentation/pytest.ini", "--rootdir=/tests", "--confcutdir=/tests",
                "-p", "scb_probe_plugin", "-p", "no:cacheprovider", "-q", *test_paths,
                "--checkpoint", f"checkpoint_{stage}", "--entrypoint",
                "python -I /instrumentation/submission.py", "--timeout=30", "--timeout-method=thread"]
        runtime_session = None
        runtime_errors = []
        runtime_validation = {"runtime_mode": "baseline_only_preflight", "worker_R_verified": False}
        restore_elapsed_seconds = 0.0
        bridge_mounts = []
        # Even baseline-only probes use a separate submission container. Host
        # bind-mount permission semantics are not an isolation boundary.
        if runtime is None:
            runtime = RuntimeExport(base_image_id=manifest["environment_image_id"],
                runtime_class=RuntimeClass.CAPTURED,
                snapshot={"entries": {}, "changed": [], "deleted": [],
                          "scope": "declared frozen baseline; not a worker runtime capture"})
            baseline_only = True
        else:
            baseline_only = False
        if runtime is not None:
            from .runtime_grading import RuntimeGradeSession
            mailbox = private / "mailbox"
            mailbox.mkdir(mode=0o700)
            fixture_path = "/tmp/file-backup-eval-" + uuid.uuid4().hex[:12]
            # This is a fresh Linux directory, never a host bind mount.
            args += ["--basetemp", fixture_path]
            bridge_mounts = ["--env", "FILE_BACKUP_RUNTIME_BRIDGE=1",
                             "--mount", f"type=bind,src={mailbox},dst=/bridge"]
            try:
                restore_started = time.monotonic()
                runtime_session = RuntimeGradeSession(runtime, work, package / "public/workspace",
                                                       mailbox, name, fixture_path, baseline_only=baseline_only)
                runtime_validation = runtime_session.validation
                runtime_validation["baseline_only_preflight"] = baseline_only
                runtime_validation["worker_R_verified"] = not baseline_only
            except Exception as exc:
                runtime_errors.append({"phase": "runtime_restore", "error": f"{type(exc).__name__}: {exc}"})
            finally:
                restore_elapsed_seconds = time.monotonic() - restore_started
        (instrumentation / "run_grader.py").write_text(
            "import sys\nsys.path[:0] = ['/instrumentation', '/tests']\nimport pytest\n"
            f"raise SystemExit(pytest.main({args!r}))\n")
        command = ["docker", "run", "--rm", "--init", "--name", name, "--network", "none",
                   "--memory", "2g", "--cpus", "1", "--pids-limit", "256",
                   "--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "FOWNER",
                   "--cap-add", "DAC_OVERRIDE", "--cap-add", "SETUID", "--cap-add", "SETGID",
                   "--security-opt", "no-new-privileges", "--workdir", "/workspace",
                   "--entrypoint", "python", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                   "--env", f"SCB_STAGE={stage}",
                   "--env", "FILE_BACKUP_CAPTURE_MODE=" + ("spec_compatible_v1" if adapted else "native"),
                   "--mount", f"type=bind,src={private / 'tests'},dst=/tests,readonly",
                   "--mount", f"type=bind,src={instrumentation},dst=/instrumentation,readonly",
                   "--mount", f"type=bind,src={(package / 'public/workspace').resolve()},dst=/public-assets,readonly",
                   "--mount", f"type=bind,src={output},dst=/probe-output",
                   *bridge_mounts, manifest["environment_image_id"], "-I", "/instrumentation/run_grader.py"]
        # pytest-timeout is explicitly loaded; autoload never imports W plugins.
        args[0:0] = ["-p", "pytest_timeout"]
        (instrumentation / "run_grader.py").write_text(
            "import sys\nsys.path[:0] = ['/instrumentation', '/tests']\nimport pytest\n"
            f"raise SystemExit(pytest.main({args!r}))\n")
        start = time.monotonic()
        timeout = False
        error = None
        try:
            if runtime_errors:
                returncode = None
                (output / "stderr.txt").write_text(json.dumps(runtime_errors, indent=2))
                (output / "stdout.txt").write_text("")
            else:
                with (output / "stdout.txt").open("w") as stdout, (output / "stderr.txt").open("w") as stderr:
                    result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600)
                returncode = result.returncode
        except subprocess.TimeoutExpired:
            timeout, returncode = True, None
            subprocess.run(["docker", "stop", "--time", "1", name], capture_output=True, timeout=30)
        except OSError as exc:
            returncode, error = None, repr(exc)
        finally:
            if runtime_session is not None:
                runtime_session.close()
                runtime_errors += runtime_session.errors
                (output / "runtime_calls.json").write_text(json.dumps(runtime_session.calls, indent=2) + "\n")
        after = file_manifest(workspace_export)
        raw_path = output / "results.json"
        receipt = {
            "task_id": manifest["task_id"], "stage": stage,
            "source_revision": manifest["source_revision"], "image_id": manifest["environment_image_id"],
            "command": command, "returncode": returncode, "timeout": timeout, "launch_error": error,
            "elapsed_seconds": time.monotonic() - start,
            "total_operation_seconds": time.monotonic() - operation_started,
            "restore_elapsed_seconds": restore_elapsed_seconds,
            "source_workspace_before": before, "source_workspace_after": after,
            "source_unchanged": before == after, "host_export_copy_after": file_manifest(work),
            "output_files": file_manifest(output), "results_present": raw_path.exists(),
            "adaptation": manifest.get("adaptation_version") if adapted else None,
            "runtime_validation": runtime_validation, "runtime_errors": runtime_errors,
            "isolation": {"boundary": "separate evaluator and submission containers",
                          "private_directories_mode": "0700 (artifact hygiene only)",
                          "network": "none", "copied_workspace_only": True,
                          "reference_mount": False,
                          "runtime_coverage": "grader only; execution validates worker R separately"},
        }
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
