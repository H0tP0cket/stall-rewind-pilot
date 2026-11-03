"""Pinned FeatureBench sources and the native level-1 lifecycle this lane reproduces.

Native (FeatureBench @ SOURCE_REVISION): infer/runtime.py::_initialize_level1 builds the agent's /testbed from the
unmasked /root/my_repo (copy, apply the masking patch, delete the F2P test file, clear bytecode, re-initialise git,
delete /root/my_repo, clear package caches); the agent works in /testbed as the unprivileged `fbagent` with the model
API as its only network route; the submission is `git diff --cached <root commit>` without binary files.
harness/runtime.py::run_instance_level1 grades in a fresh container of the same image: rebuild the masked baseline,
apply the agent patch, delete and restore the F2P file via reverse test_patch, run pytest once on the F2P file and once
per P2P file. resolved = F2P exit 0 and every P2P exit 0; pass_rate = passed / parsed F2P ids.
"""
from __future__ import annotations

SOURCE_URL = "https://github.com/LiberCoders/FeatureBench"
SOURCE_REVISION = "8d4e347"  # v0.2.3
DATASET = "LiberCoders/FeatureBench"
DATASET_REVISION = "v1.1"
DATASET_SHA = "76b4a4566e04f4bcc13c35125d4f301791efa736"
PARQUET_SHA256 = {"lite": "8740f33eb89e443e8e32b46c98508be05a2db2cbf65f856ff38e4e7730b1cc99",
                  "full": "1989012f1d7003bde5e79cb5d6f7b5580c1597ae36d533019a53a1da44b2c160"}
# Images carry only a moving `latest` tag; pinned by manifest digest on 2026-09-28 (all linux/amd64).
IMAGE_DIGESTS = {
    "libercoders/featurebench-specs_seaborn-instance_52738fbb":
        "sha256:fe27a1a5205fa0939453e79d0f40c74cb402bad8ea1d42fff7de5e2106e77ce1",
    "libercoders/featurebench-specs_pandas-instance_15f634a6":
        "sha256:e484ae407255b4c2c481a842e476c3c3285d2032738adf0abd410fb5f776dab7",
    "libercoders/featurebench-specs_astropy-instance_493bb78b":
        "sha256:4b5e7489d8bee8e20b53f94c74833e6d9ee0163673ac4a36eecb0372ef01fda2",
    "libercoders/featurebench-specs_sphinx-instance_6191afb1":
        "sha256:15b434473e16386e33c864d8b7f4a3bddb9c5585961758c0ed0f06b51b3f0268",
    "libercoders/featurebench-specs_sympy-instance_3d889fef":
        "sha256:207ff124011e74d2b6abbdce2f8d3244546949cd0abfb7949d7dd414368f8935",
}
CONDA = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
TASK_USER = "fbagent"
NATIVE_BUDGET = {"openhands_max_iterations": 500, "timeout_seconds": 3600, "token_budget": None, "cost_cap": None}
WORKER_ENVIRONMENT = {"user": TASK_USER, "cpus": "4", "memory": "16g"}
# Files an agent could use to rewrite pytest outcomes; changes to them are reverted before the tests run.
GUARDED_NAMES = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "sitecustomize.py",
                 "usercustomize.py")

# Native package-cache purge (infer/runtime.py::clear_package_caches, mounts check dropped: the build has none).
CLEAR_CACHES = r"""
shopt -s nullglob
for p in /download /opt/miniconda3/pkgs /opt/conda/pkgs /root/.cache/pip /root/.cache/uv /root/.conda/pkgs \
         /home/*/.cache/pip /home/*/.cache/uv /home/*/.conda/pkgs; do
  [ -e "$p" ] && find "$(readlink -f -- "$p")" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
done
rm -rf /tmp/pip-* /tmp/uv-*
"""

LIFECYCLE = {
    "policy_id": "featurebench.native-level1-single-submission.v1",
    "stage_release_rule": ("single native assignment (problem statement with interface descriptions); the run ends on "
                           "the first voluntary submission or a guard; the final W is graded once; no feedback"),
    "context_overflow": "native: full history, no condenser, no truncation; overflow ends the run and the partial W is graded",
    "source_evidence": (SOURCE_URL + "/blob/" + SOURCE_REVISION + "/featurebench/infer/runtime.py",
                        SOURCE_URL + "/blob/" + SOURCE_REVISION + "/featurebench/harness/runtime.py"),
}

DEVIATIONS = (
    "Scaffold is the shared mini-SWE bash worker (FeatureBench supports mini-swe-agent natively; the leaderboard mostly "
    "uses OpenHands CodeAct and its stuck detector, which is not reproduced).",
    "Worker has no network at all (native allows only the model API through a proxy) and no package index.",
    "The worker's /testbed is a symlink to the runner's /workspace (W is captured at /workspace); the editable install "
    "still resolves through /testbed. The testbed conda env is first on PATH instead of activated in .bashrc.",
    "Observation cap is the shared 10,000-character head/tail.",
    "Binding budgets are the study's x86 guards (1,000 model calls / 12 h / $10); native 500 iterations / 3,600 s recorded.",
    "The score's denominator is the F2P node ids the reference passes in a reference run of this package (native uses "
    "the ids parsed from each run, so a collection error scores 0/0, and counts module-level skips); passed follows "
    "native PASSED/XFAIL semantics. full_solve = every frozen id passes and no P2P test the reference passes fails "
    "(native `resolved` is recorded alongside).",
    "Grading-tamper guard: agent changes to conftest.py, pytest.ini, tox.ini, setup.cfg, pyproject.toml, "
    "sitecustomize.py, usercustomize.py and *.pth files are reverted before the tests run (a conftest hook can rewrite "
    "outcomes).",
    "Grading containers run without network (native eval uses the default bridge).",
)
