"""In-container grading driver (disposable container, network none).

1. Copy the submitted W (read-only mount) to /workspace.
2. Restore the frozen test-support set: protected test directories are
   replaced wholesale, protected files are overwritten from the pristine
   copy, and conftest/pytest-config files absent at the start are removed.
3. Run the repository's pytest on the selected test files with the private
   recorder plugin. The driver never reads or writes the host workspace.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time


def unique(values):
    seen, result = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main(spec_path):
    spec = json.load(open(spec_path))
    started = time.time()
    receipt = {"driver": "swesmith-long.grade-driver.v1", "phases": {}}
    work = "/workspace"
    if os.path.lexists(work):
        shutil.rmtree(work)
    shutil.copytree("/submission", work, symlinks=True)
    receipt["phases"]["copy_seconds"] = round(time.time() - started, 3)
    removed, restored = [], []
    if spec["restore_test_support"]:
        overlay(work, spec, removed, restored)
    receipt["overlay"] = {"applied": bool(spec["restore_test_support"]), "removed": sorted(set(removed)),
                          "restored_count": len(restored)}
    receipt["phases"]["overlay_seconds"] = round(time.time() - started, 3)
    run_pytest(work, spec, receipt, started)
    return 0


def overlay(work, spec, removed, restored):
    for rel in spec["protected_dirs"]:
        path = os.path.join(work, rel)
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
            removed.append(rel)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            removed.append(rel)
    config_names = set(spec["config_names"])
    protected_files = set(spec["protected_files"])
    for root, dirs, files in os.walk(work):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), work)
            if (name == "conftest.py" or (name in config_names and os.path.dirname(rel) == "")) and rel not in protected_files:
                os.unlink(os.path.join(root, name))
                removed.append(rel)
    for rel in spec["protected_files"]:
        target = os.path.join(work, rel)
        parent = os.path.dirname(target)
        if os.path.islink(parent) or (os.path.exists(parent) and not os.path.isdir(parent)):
            os.unlink(parent)
        os.makedirs(parent, exist_ok=True)
        if os.path.lexists(target):
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        shutil.copy2(os.path.join("/pristine", rel), target, follow_symlinks=False)
        restored.append(rel)


def run_pytest(work, spec, receipt, started):
    env = dict(os.environ)
    env["PYTHONPATH"] = "/grader" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SWESMITH_LONG_REPORT"] = "/out/reports.jsonl"
    # The repository's native command is the `pytest` console script (cwd is
    # not prepended to sys.path, unlike `python -m pytest`).
    launcher = shutil.which("pytest", path=env.get("PATH"))
    command = ([launcher] if launcher else [sys.executable, "-m", "pytest"]) + ["-p", "no:cacheprovider", "-p", "no:randomly",
               "-p", "swesmith_long_recorder", "--disable-warnings", "--color=no", "--tb=short", "-q",
               "--continue-on-collection-errors"] + unique(list(spec["pytest_args"]) + list(spec["test_files"]))
    receipt["command"] = command
    t0 = time.time()
    with open("/out/pytest.log", "wb") as log:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            receipt["returncode"] = process.wait(timeout=spec["timeout_seconds"])
            receipt["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            receipt["returncode"] = None
            receipt["timed_out"] = True
    receipt["phases"]["pytest_seconds"] = round(time.time() - t0, 3)
    receipt["phases"]["total_seconds"] = round(time.time() - started, 3)
    with open("/out/driver.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
