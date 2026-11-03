"""Build and verify one MettleBench task package (mettlebench.package.v1).

Layout (benchmark_tasks/packages/<task_id>/):
  package.json                      frozen manifest (hashes of every file)
  LICENSE, NOTICE.md, licenses/      upstream license/attribution material
  public/instruction.md             upstream prompt.md, byte-identical
  public/workspace.tar.gz           upstream dataset archive, byte-identical
  private/upstream/tasks/...        _verifier_v2.py + tasks/<id>/verifier/** (+ prompt.md)
  private/adaptation/...            explicit versioned adaptations (overlay files, record)
  private/rubric.json               frozen criterion scoring map
  private/reference/solve.sh        independently authored forward-only witness

Only ``public/`` reaches the worker (materialize extracts the archive into W).
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from . import source as S
from .scoring import build_rubric

SCHEMA = "mettlebench.package.v1"
MODULE_DIR = Path(__file__).resolve().parent
ADAPTER_ENTRY = "trajectory_retention.tasks.mettlebench_v1.adapter:MettleBenchTaskAdapter"
BOUND_FILES = ("__init__.py", "adapter.py", "criteria_runner.py", "grading.py", "package.py", "scoring.py",
               "source.py", "environment/Dockerfile", "environment/requirements.txt", "environment/python-lock-base.txt",
               "environment/python-lock-torch.txt")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(path: Path) -> dict[str, str]:
    out = {}
    for p in sorted(Path(path).rglob("*")):
        if p.is_symlink():
            raise ValueError("Package contains a symlink: " + str(p))
        if p.is_file():
            out[p.relative_to(path).as_posix()] = sha(p)
    return out


def implementation_binding() -> dict[str, str]:
    return {"tasks/mettlebench_v1/" + name: sha(MODULE_DIR / name) for name in BOUND_FILES}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def archive_manifest(archive: Path) -> dict[str, dict]:
    """size/sha256 of every regular file in the shipped archive, keyed relative to W."""
    files = {}
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if m.isreg():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                data = tf.extractfile(m).read()
                files[rel] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


# Explicit, reviewed adaptations. Anything not listed here is upstream-exact.
def task_adaptations(upstream_id: str, archive: Path) -> tuple[list[dict], dict[str, bytes]]:
    adaptations: list[dict] = []
    overlay: dict[str, bytes] = {}
    if upstream_id == "programbench_bat_cache_handoff":
        manifest = {"files": archive_manifest(archive)}
        overlay[f"tasks/{upstream_id}/internal/initial_state_manifest.yaml"] = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        adaptations.append({
            "id": "mettlebench.bat.initial-state-manifest.v1",
            "kind": "evaluator_data_reconstruction",
            "reason": ("Upstream verifier.load_manifest() reads tasks/<id>/internal/initial_state_manifest.yaml, "
                       "which is not in the pinned repository; the evaluator cannot start without it."),
            "change": ("Reconstructed as JSON {files: {relpath: {size, sha256}}} for every regular file of the shipped "
                       "dataset archive. Only P06_no_forbidden_edits reads it (size+sha256 of PROTECTED_PATHS)."),
            "public_change": False, "prompt_change": False, "checklist_change": False})
    return adaptations, overlay


COMMON_ADAPTATIONS = [
    {"id": "mettlebench.environment.shared-image.v1", "kind": "environment",
     "change": ("One pinned Docker image (python:3.12-slim-trixie, arm64; root user; default git identity and "
                "safe.directory='*') with git, git-filter-repo, openssl, gcc/make, sqlite3, rsync, bc, jq, zstd, lz4, xxd, "
                "hexdump, x86_64 glibc for the bundled prebuilt executables (run through Docker Desktop emulation), and "
                "the Python packages the workspaces and evaluators import (exact lock; the torch stack only in the variant "
                "image for the task that imports it). Upstream declares only Python "
                "3.12+ and git (+django/pypdf/cryptography for some evaluators); the authors' image is not published."),
     "public_change": False, "prompt_change": False, "checklist_change": False},
    {"id": "mettlebench.feedback.bare-string.v1", "kind": "harness",
     "change": ("Native Continue feedback is the evaluator's first_failure.feedback delivered verbatim as a user message; "
                "the authors' wrapper text is unpublished. Shared mini-SWE tool interface and submission command."),
     "public_change": False, "prompt_change": False, "checklist_change": False},
    {"id": "mettlebench.grading.isolated-copy.v1", "kind": "evaluation_location",
     "change": ("The unmodified evaluator runs as root in a disposable container restored from the captured W/R, "
                "never in the live worker container."),
     "public_change": False, "prompt_change": False, "checklist_change": False},
]


def build_package(upstream_id: str, destination: Path, *, image_id: str, reference: Path, split: str,
                  cache: Path = S.DEFAULT_CACHE, reference_meta: dict | None = None) -> dict:
    verified = S.verify_source(cache)
    row = next(r for r in verified["manifest"] if r["task_id"] == upstream_id)
    src = cache / "src"
    task_dir = src / "tasks" / upstream_id
    archive = cache / "hf/workspaces" / f"{upstream_id}.tar.gz"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    public, private = destination / "public", destination / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    shutil.copy2(task_dir / "prompt.md", public / "instruction.md")
    shutil.copy2(archive, public / "workspace.tar.gz")
    up = private / "upstream/tasks"
    up.mkdir(parents=True)
    shutil.copy2(src / "tasks/_verifier_v2.py", up / "_verifier_v2.py")
    shutil.copytree(task_dir, up / upstream_id)
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(src / name, destination / name)
    shutil.copytree(src / "licenses", destination / "licenses")
    adaptations, overlay = task_adaptations(upstream_id, archive)
    for rel, data in overlay.items():
        target = private / "adaptation/overlay" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_json(private / "adaptation/record.json", {"task_specific": adaptations, "common": COMMON_ADAPTATIONS})
    checklist = json.loads((task_dir / "verifier/verifier_checklist.json").read_text())
    write_json(private / "rubric.json", build_rubric(checklist, upstream_id=upstream_id))
    (private / "reference").mkdir()
    shutil.copy2(reference, private / "reference/solve.sh")
    if reference_meta is not None:
        write_json(private / "reference/meta.json", reference_meta)
    task_id = S.package_task_id(upstream_id)
    binding = implementation_binding()
    manifest = {
        "schema_version": SCHEMA, "task_id": task_id, "package_version": "1", "upstream_task_id": upstream_id,
        "source_benchmark": row["source_benchmark"], "source_url": S.SOURCE_URL, "source_revision": S.SOURCE_REVISION,
        "dataset_url": S.DATASET_URL, "dataset_revision": S.DATASET_REVISION,
        "family_id": S.family_for(upstream_id), "family_split": split, "split_rule": S.SPLIT_RULE,
        "environment_image_id": image_id, "grader_image_id": image_id, "adapter_entry": ADAPTER_ENTRY,
        "archive": {"path": "public/workspace.tar.gz", **verified["archives"][upstream_id]},
        "native_protocol": S.NATIVE_PROTOCOL, "criteria": len(checklist["points"]),
        "upstream_files": {"prompt.md": sha(task_dir / "prompt.md"), "_verifier_v2.py": sha(src / "tasks/_verifier_v2.py"),
                           **{("verifier/" + k): v for k, v in inventory(task_dir / "verifier").items()}},
        "adaptations": [a["id"] for a in adaptations + COMMON_ADAPTATIONS],
        "reference": {"path": "private/reference/solve.sh", "sha256": sha(private / "reference/solve.sh"),
                      "origin": ("Independent researcher witness written from the public prompt/workspace and checked "
                                 "against the private evaluator. The gated upstream reference runs were not used.")},
        "public_files": inventory(public), "private_files": inventory(private),
        "license_files": {p: sha(destination / p) for p in ["LICENSE", "NOTICE.md", *sorted(
            "licenses/" + q for q in inventory(destination / "licenses"))]},
        "implementation_binding": binding,
    }
    write_json(destination / "package.json", manifest)
    return manifest


def verify_package(package: Path) -> dict:
    package = Path(package)
    m = json.loads((package / "package.json").read_text())
    if m.get("schema_version") != SCHEMA:
        raise ValueError("Wrong task package schema")
    if inventory(package / "public") != m["public_files"] or inventory(package / "private") != m["private_files"]:
        raise ValueError("Frozen task package differs")
    if {p: sha(package / p) for p in m["license_files"]} != m["license_files"]:
        raise ValueError("License binding differs")
    if implementation_binding() != m["implementation_binding"]:
        raise ValueError("Source binding differs")
    if sha(package / m["archive"]["path"]) != m["archive"]["sha256"]:
        raise ValueError("Workspace archive differs")
    return m


def extract_workspace(archive: Path, workspace: Path) -> None:
    """Upstream download_data.py semantics (tarfile filter='tar'), rooted at W."""
    workspace = Path(workspace)
    with tarfile.open(archive) as tf:
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if parts[0] != "workspace":
                raise ValueError("Unexpected archive member outside workspace/: " + m.name)
            if len(parts) == 1:
                continue
            m.name = parts[1]
            members.append(m)
        tf.extractall(workspace, members=members, filter="tar")
