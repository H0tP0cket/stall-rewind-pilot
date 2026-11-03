"""In-container grading driver, v3 (disposable container, network none).

Identical to grade_driver.py except for step 1: Git metadata is not part of
the submission. Every path component named ``.git`` (the start repository and
any repository the worker created) is skipped when the read-only /submission
is copied to /workspace; Git itself never tracks such a path. The skipped
paths are listed in the receipt. Steps 2 and 3 reuse grade_driver.py
unchanged (test-support overlay, pytest with the private recorder).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script directory
import grade_driver as base  # noqa: E402

VCS_NAMES = (".git",)


def main(spec_path):
    spec = json.load(open(spec_path))
    started = time.time()
    receipt = {"driver": "swesmith-long.grade-driver.v3", "phases": {}}
    work = "/workspace"
    if os.path.lexists(work):
        shutil.rmtree(work)
    skipped = []

    def ignore(directory, names):
        chosen = [name for name in names if name in VCS_NAMES]
        skipped.extend(os.path.relpath(os.path.join(directory, name), "/submission") for name in chosen)
        return chosen

    shutil.copytree("/submission", work, symlinks=True, ignore=ignore)
    receipt["vcs_metadata_skipped"] = sorted(skipped)
    receipt["phases"]["copy_seconds"] = round(time.time() - started, 3)
    removed, restored = [], []
    if spec["restore_test_support"]:
        base.overlay(work, spec, removed, restored)
    receipt["overlay"] = {"applied": bool(spec["restore_test_support"]), "removed": sorted(set(removed)),
                          "restored_count": len(restored)}
    receipt["phases"]["overlay_seconds"] = round(time.time() - started, 3)
    base.run_pytest(work, spec, receipt, started)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
