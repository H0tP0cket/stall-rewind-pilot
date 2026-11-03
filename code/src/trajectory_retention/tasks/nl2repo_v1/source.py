"""Pinned NL2RepoBench sources and the native lifecycle this lane reproduces.

Native (NL2RepoBench @ SOURCE_REVISION, openhands/openhands_app.py and openhands/post_processor.py): the agent
starts from a workspace holding only start.md in a generic runtime; after `finish`, the post-processor zips the
workspace, deletes the agent's packaging files and the task's test paths, builds FROM the task's grading image with
the workspace copied over /workspace (keeping the image's hidden tests and reference packaging), runs the task's test
commands (split with shlex, no shell) and scores sum(regex "(\\d+) passed") / test_case_count, capped at 1.
"""
from __future__ import annotations

SOURCE_URL = "https://github.com/multimodal-art-projection/NL2RepoBench"
SOURCE_REVISION = "781a1da1ee41fb8edb0bed22f586d69111610edf"
GRADING_IMAGE = "ghcr.io/multimodal-art-projection/nl2repobench/{task}:1.0"
# Worker base = official python image at the grading image's interpreter version, linux/amd64 manifests (resolved
# from the multi-platform indexes on 2026-09-28).
WORKER_BASES = {"3.10.11": "python@sha256:fc1d7d6e1df26c603b5729e92fff684a56e9d1b8c6db263c6169a1f250e80787",
                "3.11.7": "python@sha256:a526139339e938726256809dad42ba8dd4edf3133194dbe6f62d400aff6bb886"}
NATIVE_PROMPT = ("According to the start.md in the workspace, implement the entire project as per the requirements "
                 "specified in the document, ensuring that the final product can be directly run in the current "
                 "directory. The running requirements should comply with the <API Usage Guide> section of the "
                 "document. Please complete this task step by step.")
PACKAGE_FILES = ("setup.py", "pyproject.toml", "setup.cfg", "requirements.txt", "requirements-dev.txt",
                 "requirements-test.txt", "tox.ini", "pytest.ini", "poetry.lock", "Pipfile", "Pipfile.lock",
                 "environment.yml", "conda-env.yaml", "manifest.in", "MANIFEST.in")
NATIVE_BUDGET = {"max_iterations": 500, "wall_clock": None, "cost_cap": None, "sandbox_command_timeout_seconds": 120}
GRADE_COMMAND_TIMEOUT = 1800
JUNIT_PATH = "/tmp/nl2repo-junit.xml"
TAMPER_FILES = ("conftest.py", "sitecustomize.py", "usercustomize.py")

LIFECYCLE = {
    "policy_id": "nl2repo.native-single-submission.v1",
    "stage_release_rule": ("single native assignment from start.md; the run ends on the first voluntary submission or "
                           "a guard; the final W is graded once; no feedback reaches the worker"),
    "context_overflow": "native: full history; a context-window overflow ends the run and the partial W is graded",
    "source_evidence": (SOURCE_URL + "/blob/" + SOURCE_REVISION + "/openhands/openhands_app.py",
                        SOURCE_URL + "/blob/" + SOURCE_REVISION + "/openhands/post_processor.py"),
}

DEVIATIONS = (
    "Scaffold is the shared mini-SWE bash worker, not OpenHands CodeAct; no browsing, no auto-continue message.",
    "Worker has no network and no package index (user decision 2026-09-28). The worker image is the official python image at "
    "the grading image's version with "
    "the grading image's third-party packages pre-installed at the same versions, excluding the target package.",
    "Observation cap is the shared 10,000-character head/tail, not OpenHands' 30,000-character max_message_chars.",
    "Binding budgets are the study's x86 guards (1,000 model calls / 12 h / $10); native 500 iterations recorded.",
    "Each grading command gets a 1,800 s timeout (native has none).",
    "Grading-tamper guard: agent-written conftest.py, sitecustomize.py, usercustomize.py and *.pth files are deleted "
    "before grading (a conftest hook can rewrite test outcomes; PYTHONPATH=/workspace imports sitecustomize).",
    "pytest commands additionally write JUnit XML (--junitxml) so passed/total and failing test files are exact; "
    "the native regex score is kept alongside.",
)
