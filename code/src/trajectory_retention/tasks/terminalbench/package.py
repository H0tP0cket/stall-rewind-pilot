"""Build one terminalbench-v1 package: images, natural start, frozen checks, manifest.

Raw build evidence goes to artifacts/task-admission/terminalbench/tasks/<task>/
(gitignored); the package binds image IDs, upstream hashes and every file.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import tomllib

from . import environment as E
from . import source as S
from .adapter import ADAPTER_ENTRY, SCHEMA, evaluator_sources, extract_workspace, file_manifest, sha256
from .grading import run_private_grade

WORKTREE = Path(__file__).resolve().parents[4]
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / E.LANE
EVIDENCE = WORKTREE / "artifacts" / "task-admission" / "terminalbench"
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / E.LANE
PACKAGE_VERSION = "1"
HOLD_GRADE_SECONDS = 600
FLAG_GRADE_SECONDS = 120

ADAPTATION = {
    "version": "terminalbench.adaptation.v1",
    "items": [
        "Harness: the shared mini-SWE bash harness (one bash command per tool call through docker exec, the shared "
        "submission marker) replaces Terminus 2's tmux keystroke interface; instruction.md is delivered byte-for-byte.",
        "Environment: the upstream environment/Dockerfile is rebuilt natively for linux/arm64 from the same build "
        "context (the official alexgshaw/<task>:20251031 images are x86_64-only). Every FROM is pinned by digest to "
        "the arm64 variant of the base release the official image was built from, identified from the official "
        "image's registry config (layer-prefix match, e.g. ubuntu:noble-20251001; for python bases the CPython "
        "release, e.g. python:3.13.7-slim-bookworm, because re-published python tags do not reproduce layer digests). "
        "apt/pip/curl steps re-resolve at build time as upstream (image IDs recorded).",
        "Workspace: the native /app directory becomes W (/workspace); /app is a symlink to /workspace in the worker "
        "image so every absolute /app path in the instruction and tests is unchanged.",
        "Harness interpreter: python3 is added to the worker image only when the native image lacks it (the harness "
        "supervisor and capture probes run python3); recorded per package.",
        "Network: worker and grader containers run with network none; natively allow_internet is true, so the worker "
        "cannot install packages or download at run time. Tasks whose reference solution needs network are held.",
        "Grading: native TB runs tests/test.sh inside the live agent container after the agent stops. Here the "
        "submitted W and captured filesystem R are restored into a fresh container of the grader image (worker image "
        "+ verifier layer) and the native pytest command runs there; live processes are not restored; R entries under "
        "/tb-verifier, /tb-grader, /tests and /logs are dropped (listed per grade).",
        "Verifier environment: the Python environment that test.sh creates at grade time (uv 0.9.5, uvx -p 3.13 with "
        "the same -w pins, uv's default interpreter choice: the image's own CPython 3.13 where it has one, otherwise "
        "uv's managed CPython 3.13) and its grade-time apt packages are installed ahead of time in the grader image "
        "only (transitive versions resolved at build time and recorded).",
        "Grader driver: test.sh is replaced by an equivalent pytest.main call with the same arguments and uvx's PATH "
        "(verifier environment bin first), plus a researcher-only recorder plugin, a private empty ini file, and "
        "rootdir/confcutdir fixed to /tests.",
        "Resources: the shared engine runs the worker container with 1 CPU and 2 GB unless a launch overrides it; the "
        "grader uses the task.toml cpus/memory. The native agent timeout is not enforced by the task; launch limits "
        "(calls, tokens, wall clock) apply.",
    ],
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n")


def package_task_id(task: str) -> str:
    return f"terminalbench-{task}-v{PACKAGE_VERSION}"


def task_toml(task: str) -> dict:
    return tomllib.loads((S.CHECKOUT / task / "task.toml").read_text())


def native_limits(task: str) -> dict:
    t = task_toml(task)
    env = t.get("environment", {})
    return {"cpus": env.get("cpus"), "memory_mb": env.get("memory_mb"), "storage_mb": env.get("storage_mb"),
            "gpus": env.get("gpus"), "allow_internet": env.get("allow_internet"),
            "agent_timeout_sec": t.get("agent", {}).get("timeout_sec"),
            "verifier_timeout_sec": t.get("verifier", {}).get("timeout_sec"),
            "build_timeout_sec": env.get("build_timeout_sec"), "official_image": env.get("docker_image"),
            "difficulty": t.get("metadata", {}).get("difficulty"), "category": t.get("metadata", {}).get("category")}


def build_images(task: str, *, reuse: bool = True) -> dict:
    out = EVIDENCE / "tasks" / task
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    spec = E.parse_test_sh((S.CHECKOUT / task / "tests/test.sh").read_text())
    chain_reused = True
    for kind in ("native", "worker", "grader"):
        path = out / f"{kind}-build.json"
        if reuse and chain_reused and path.is_file():
            rec = json.loads(path.read_text())
            current_policy = kind != "native" or "official_image" in rec
            if rec.get("result") == "built" and current_policy:
                try:
                    if E.image_info(rec["tag"])["id"] == rec["image"]["id"]:
                        records[kind] = rec
                        continue
                except Exception:
                    pass
        chain_reused = False
        if kind == "native":
            rec = E.build_native(task)
        elif kind == "worker":
            rec = E.build_worker(task, records["native"])
        else:
            rec = E.build_grader(task, records["worker"], spec)
        write_json(path, rec)
        if rec.get("result") != "built":
            return {"status": "HOLD", "reason": f"{kind} image build failed", "build": rec, "records": records}
        records[kind] = rec
    return {"status": "BUILT", "records": records, "test_spec": spec}


def collect_checks(task: str, grader_id: str, worker_id: str, workspace: Path, flags: list[str], limits: dict) -> dict:
    out = Path(tempfile.mkdtemp(prefix="collect-", dir=EVIDENCE / "tasks" / task))
    result = run_private_grade(grader_image_id=grader_id, worker_image_id=worker_id,
                               tests_dir=S.CHECKOUT / task / "tests", workspace=workspace, output=out, runtime=None,
                               timeout=300, pytest_flags=["--collect-only", "-q"], cpus=str(limits["cpus"] or 1),
                               memory=f"{limits['memory_mb'] or 2048}m")
    collected = []
    for line in (out / "reports.jsonl").read_text().splitlines() if (out / "reports.jsonl").is_file() else []:
        row = json.loads(line)
        if row.get("event") == "collected":
            collected = row["nodeids"]
    return {"checks": collected, "receipt": str(out), "infrastructure_errors": result["infrastructure_errors"]}


def build_package(task: str, *, split: str, reuse: bool = True) -> dict:
    from .catalog import METRICS
    metrics = METRICS.get(task)
    images = build_images(task, reuse=reuse)
    if images["status"] != "BUILT":
        return images
    rec = images["records"]
    limits = native_limits(task)
    task_id = package_task_id(task)
    staging = EVIDENCE / "tasks" / task / "package-staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "public").mkdir(parents=True)
    (staging / "private").mkdir()
    app_tar = E.export_path(rec["native"]["image"]["id"], E.NATIVE_WORKDIR)
    app_manifest = E.app_tar_manifest(app_tar)
    symlinks = sorted(k for k, v in app_manifest.items() if v["type"] in ("l", "h"))
    (staging / "public/workspace.tar.gz").write_bytes(E.workspace_tar_gz(app_tar))
    shutil.copy2(S.CHECKOUT / task / "instruction.md", staging / "public/instruction.md")
    for name in ("tests", "solution", "environment"):
        shutil.copytree(S.CHECKOUT / task / name, staging / "private" / name, symlinks=True)
    shutil.copy2(S.CHECKOUT / task / "task.toml", staging / "private/task.toml")
    probe_ws = Path(tempfile.mkdtemp(prefix="start-", dir=EVIDENCE / "tasks" / task)) / "W"
    probe_ws.mkdir()
    extract_workspace(staging / "public/workspace.tar.gz", probe_ws)
    spec = images["test_spec"]
    flags = [a for a in spec["pytest_args"] if a not in ("--ctrf", "/logs/verifier/ctrf.json", "/tests/test_outputs.py")]
    collected = collect_checks(task, rec["grader"]["image"]["id"], rec["worker"]["image"]["id"], probe_ws, flags, limits)
    write_json(EVIDENCE / "tasks" / task / "collect.json", collected)
    if collected["infrastructure_errors"] or not collected["checks"]:
        return {"status": "HOLD", "reason": "native test collection failed", "collect": collected}
    write_json(staging / "private/checks.json", {"checks": collected["checks"],
               "rule": "pytest node ids collected from the native tests/test_outputs.py (rootdir /tests) in the grader "
                       "image; one check per collected test"})
    if metrics:
        write_json(staging / "private/metrics.json", {"metrics": metrics})
    native_python = E.probe(rec["native"]["image"]["id"], "command -v python3 && python3 -V")
    python_added = native_python.returncode != 0
    manifest = {
        "schema_version": SCHEMA, "task_id": task_id, "package_version": PACKAGE_VERSION, "lane": E.LANE,
        "upstream_task": task, "family_id": S.family_id(task), "planned_split": split,
        "adapter_entry": ADAPTER_ENTRY, "source_url": S.SOURCE_URL, "source_revision": S.SOURCE_REVISION,
        "license": S.LICENSE, "registry_identity": S.registry_identity(task),
        "upstream_hashes": {"instruction.md": S.sha256_file(S.CHECKOUT / task / "instruction.md"),
                            "task.toml": S.sha256_file(S.CHECKOUT / task / "task.toml"),
                            "tests/test.sh": S.sha256_file(S.CHECKOUT / task / "tests/test.sh"),
                            "solution/solve.sh": S.sha256_file(S.CHECKOUT / task / "solution/solve.sh")},
        "native_limits": limits,
        "environment": {
            "platform": E.PLATFORM,
            "native": {"image_id": rec["native"]["image"]["id"], "tag": rec["native"]["tag"],
                       "bases": rec["native"]["bases"], "upstream_dockerfile_sha256": rec["native"]["upstream_dockerfile_sha256"],
                       "dockerfile_sha256": rec["native"]["dockerfile_sha256"]},
            "worker": {"image_id": rec["worker"]["image"]["id"], "tag": rec["worker"]["tag"],
                       "dockerfile_sha256": rec["worker"]["dockerfile_sha256"],
                       "extends_native_layers": rec["worker"].get("extends_native_layers"),
                       "python3_added": python_added},
            "grader": {"image_id": rec["grader"]["image"]["id"], "tag": rec["grader"]["tag"],
                       "dockerfile_sha256": rec["grader"]["dockerfile_sha256"], "uv_image": E.UV_IMAGE,
                       "extends_worker_layers": rec["grader"].get("extends_worker_layers"),
                       "interpreter": rec["grader"].get("interpreter"), "freeze": rec["grader"].get("freeze"),
                       "grade_time_apt_packages": spec["apt_packages"]},
        },
        "grading": {"timeout_seconds": int(limits["verifier_timeout_sec"] or 900), "pytest_flags": flags,
                    "cpus": str(limits["cpus"] or 1), "memory": f"{limits['memory_mb'] or 2048}m",
                    "native_uvx_line": spec["uvx_line"], "native_apt_line": spec["apt_line"]},
        "native_start": {"source": "native image " + E.NATIVE_WORKDIR, "entries": len(app_manifest),
                         "regular_bytes": sum(v["size"] or 0 for v in app_manifest.values()),
                         "symlinks_or_hardlinks": symlinks, "manifest_sha256": E.sha256_bytes(
                             json.dumps(app_manifest, sort_keys=True).encode())},
        "adaptation": ADAPTATION,
        "worker_prompt": None,
        "evaluator_sources": evaluator_sources(),
    }
    package = PACKAGES / task_id
    if package.exists():
        shutil.rmtree(package)
    package.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, package, symlinks=True)
    manifest["public_files"] = file_manifest(package / "public")
    manifest["private_files"] = file_manifest(package / "private")
    write_json(package / "package.json", manifest)
    # Bind the rendered worker prompt (needs the loaded stage).
    from .adapter import TerminalBenchTaskAdapter
    adapter = TerminalBenchTaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    prompt = adapter.worker_prompt("task")
    manifest["worker_prompt"] = prompt.binding()
    write_json(package / "package.json", manifest)
    TerminalBenchTaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    write_json(EVIDENCE / "tasks" / task / "app-manifest.json", app_manifest)
    return {"status": "PACKAGED", "task_id": task_id, "package": str(package), "checks": collected["checks"],
            "symlinks": symlinks, "start_bytes": manifest["native_start"]["regular_bytes"],
            "package_sha256": sha256(package / "package.json")}
