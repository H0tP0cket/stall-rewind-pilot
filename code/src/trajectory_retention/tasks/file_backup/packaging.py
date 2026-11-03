"""Build a pinned file_backup package with separate public and grading material.

No shared record/storage types are defined here. The integration contracts own
those; this module owns only the task's source files and package manifest.
"""
from __future__ import annotations

import hashlib
import ast
import json
import shutil
import subprocess
from pathlib import Path

SOURCE_REVISION = "ef6a9dd13911566b6b01075ca121758c9f7b5c5f"
RUNNER_REVISION = "06b5c0687d4c05ee502e9696a4d0c22fc1eec5e0"
IMAGE_ID = "sha256:b2260f806c09e154f35b6e04e6c8adfce3b5a414bff5a8453b22d1f45d92c37e"
TASK_ID = "scb-file-backup-v1"


def grading_contract_projection() -> str:
    """Bind the shared records used by grading, excluding unrelated store APIs."""
    contracts = Path(__file__).parents[2] / "contracts.py"
    names = {"RuntimeExport", "RuntimeClass", "Evaluation", "CheckResult", "CheckStatus",
             "CapabilityResult", "StageSpec", "TaskRecord", "LifecyclePolicy", "AdmissionStatus",
             "ArtifactRef", "Visibility", "to_jsonable", "from_jsonable", "new_id", "utc_now",
             "TaskAdapter", "ScoringAdapter"}
    parsed = ast.parse(contracts.read_text())
    selected = [node for node in parsed.body if getattr(node, "name", None) in names or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCHEMA_VERSION" for t in node.targets))]
    if {node.name for node in selected if hasattr(node, "name")} != names:
        raise ValueError("Shared grading contract definitions changed")
    return hashlib.sha256(ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def file_manifest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}


def build_package(source_repo: Path, destination: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Unexpected source revision: {revision}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing package: {destination}")
    source = source_repo / "file_backup"
    destination.mkdir(parents=True)
    public = destination / "public"
    private = destination / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    shutil.copytree(source / "files", public / "workspace" / "files")
    (public / "requirements").mkdir()
    (private / "original_specs").mkdir()
    for stage in range(1, 5):
        spec = source / f"checkpoint_{stage}.md"
        shutil.copy2(spec, private / "original_specs" / spec.name)
        rendered = spec.read_text().replace("%%%ENTRYPOINT:entry_file%%%", "backup_scheduler.py").replace(
            "%%%ENTRYPOINT:entry_command%%%", "python /workspace/backup_scheduler.py")
        (public / "requirements" / spec.name).write_text(rendered)
        shutil.copytree(source / "solutions" / f"checkpoint_{stage}", private / "references" / f"checkpoint_{stage}")
    shutil.copytree(source / "tests", private / "tests")
    # The official runner injects declared static assets at tests/assets.
    shutil.copytree(source / "files", private / "tests" / "assets" / "files")
    shutil.copy2(source / "config.yaml", private / "source_config.yaml")
    shutil.copy2(source_repo / "LICENSE", destination / "LICENSE")
    manifest = {
        "package_version": "scb.file_backup.package.v1",
        "task_id": TASK_ID, "family_id": "slopcode:file_backup",
        "source_url": "https://github.com/gabeorlanski/scb-problems",
        "source_revision": revision, "runner_revision": RUNNER_REVISION,
        "environment_image_id": IMAGE_ID, "platform": "linux/arm64",
        "entrypoint": ["python", "/workspace/backup_scheduler.py"],
        "stages": [f"checkpoint_{i}" for i in range(1, 5)],
        "initial_workspace": "public/workspace",
        "requirements": "public/requirements",
        "private_tests": "private/tests", "private_references": "private/references",
        "changes": ["Python entrypoint placeholders bound to backup_scheduler.py.",
                    "Declared files assets copied into evaluator tests/assets as in source runner."],
        "semantic_corrections": [],
        "public_files": file_manifest(public), "private_files": file_manifest(private),
        "license_sha256": hashlib.sha256((destination / "LICENSE").read_bytes()).hexdigest(),
        "admission_status": "hold",
    }
    (destination / "package.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_package(package: Path) -> dict:
    manifest = json.loads((package / "package.json").read_text())
    for zone in ("public", "private"):
        if any(p.is_symlink() for p in (package / zone).rglob("*")):
            raise ValueError("Pinned package zones cannot contain unbound symlinks")
        if file_manifest(package / zone) != manifest[f"{zone}_files"]:
            raise ValueError(f"Package {zone} bytes no longer match frozen manifest")
    if hashlib.sha256((package / "LICENSE").read_bytes()).hexdigest() != manifest["license_sha256"]:
        raise ValueError("Package license hash changed")
    return manifest


def build_adapted_package(original: Path, destination: Path, *, addenda: Path,
                          rubric: Path) -> dict:
    """Create a separate disclosed version, including original reference bytes."""
    from .reference_adaptation import adapt_reference
    original_manifest = verify_package(original)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(original, destination)
    private = destination / "private"
    (private / "references").rename(private / "upstream_references")
    (private / "references").mkdir(mode=0o700)
    witness_changes = []
    for stage in range(1, 5):
        source = private / "upstream_references" / f"checkpoint_{stage}"
        target = private / "references" / f"checkpoint_{stage}"
        if stage < 3:
            shutil.copytree(source, target)
        else:
            witness_changes.append(adapt_reference(source, target, stage))
    text = addenda.read_text()
    for section in text.split("## Checkpoint ")[1:]:
        stage = int(section[0])
        title, body = section[2:].split("\n", 1)
        path = destination / "public/requirements" / f"checkpoint_{stage}.md"
        path.write_text(path.read_text().rstrip() + "\n\n## Versioned clarification: " + title + "\n" + body.rstrip() + "\n")
    shutil.copy2(addenda, private / "adaptation_addenda.md")
    shutil.copy2(rubric, private / "rubric.json")
    (private / "reference_adaptation.json").write_text(json.dumps(witness_changes, indent=2) + "\n")
    (private / "upstream_package.json").write_text(json.dumps(original_manifest, indent=2) + "\n")
    manifest = {**original_manifest,
        "package_version": "scb.file_backup.persistence-adapted.package.v1-candidate",
        "task_id": "scb-file-backup-persistence-v1-candidate",
        "adaptation_version": "scb.file_backup.persistence-and-observability.v1-candidate",
        "original_package_sha256": hashlib.sha256((original / "package.json").read_bytes()).hexdigest(),
        "semantic_corrections": [
            "CP3+ supplied --backup persists full files and generated pack archives; omitted backup remains event-only.",
            "CP4 comparison and skip accounting are whole-pack; no new stale archive deletion requirement.",
            "Accept permitted ISO timestamp spelling and omitted optional completion counters; present values remain checked.",
            "Thirteen specification-derived supplemental probes fill evaluator coverage and generic-crash gaps.",
            "Binary capability and equal spec-group scoring is a disclosed research score, separate from raw native pass counts.",
        ],
        "witness_kind": "unchanged upstream CP1/2; disclosed persistence-repaired upstream CP3/4",
        "public_files": file_manifest(destination / "public"), "private_files": file_manifest(private),
        "admission_status": "hold",
    }
    (destination / "package.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_package(args.source, args.destination), indent=2))
