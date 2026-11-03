"""Host-side ProgramBench grade: the unmodified upstream Evaluator on one submission.

Runs in the isolated grader venv (programbench==1.2.4 only); it never imports trajectory_retention and never
touches a worker container. Usage:
  python grade_driver.py --package P --submission S.tar.gz --blob-root B --output O
where B/<instance_id>/tests/<branch>.tar.gz are the package's private blobs. Writes O/eval.json (upstream
EvaluationResult, all tests incl. ignored) and O/summary.json (frozen-denominator tallies, no test contents).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--package", "--submission", "--blob-root", "--output"):
        ap.add_argument(name, required=True)
    ap.add_argument("--docker-cpus", type=int, default=10)
    a = ap.parse_args()
    import os
    os.environ["PROGRAMBENCH_BLOB_DIR"] = a.blob_root
    from programbench import __version__
    from programbench.eval.eval import Evaluator
    from programbench.utils.load_data import get_active_branches, get_ignored_branches, get_ignored_tests

    package, out = Path(a.package), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "package.json").read_text())
    instance = {**yaml.safe_load((package / "private/task.yaml").read_text()),
                **json.loads((package / "private/tests.json").read_text())}
    instance_id = manifest["upstream_task_id"]
    branches = get_active_branches(instance)
    image = manifest["environment_image"].rsplit(":", 1)[0]
    started = time.monotonic()
    evaluator = Evaluator(image_name=image, solution_branch="submission", submission_archive=Path(a.submission),
                          blob_dir=Path(a.blob_root) / instance_id, tests_branches=branches,
                          remove_hashes=instance.get("eval_clean_hashes", []), image_tag=manifest["environment_image"].rsplit(":", 1)[1],
                          tests_by_branch={b: instance["branches"][b]["tests"] for b in branches},
                          ignored_tests=get_ignored_tests(instance), ignored_branches=get_ignored_branches(instance),
                          instance_id=instance_id, docker_cpus=a.docker_cpus, branch_workers=1, branch_retries=1)
    result = evaluator.run()
    seconds = time.monotonic() - started
    (out / "eval.json").write_text(result.model_dump_json(indent=1))
    ignored = get_ignored_tests(instance)
    native = result.without_ignored(ignored)
    statuses = {t.full_name: t.status for t in result.test_results}
    summary = {"programbench_version": __version__, "seconds": seconds, "error_code": result.error_code,
               "executable_hash": result.executable_hash, "native_score": native.score,
               "native_n_resolved": native.n_resolved, "native_n_tests": len(native),
               "branch_errors": {b: [e.error_code for e in errs] for b, errs in result.test_branch_errors.items()},
               "log_steps": [{"step": e.get("step"), "returncode": e.get("returncode"), "wall_time": e.get("wall_time")}
                             for e in result.log],
               "statuses": statuses}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
