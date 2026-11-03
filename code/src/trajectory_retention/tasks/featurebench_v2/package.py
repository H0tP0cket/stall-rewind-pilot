"""Build and verify a frozen FeatureBench level-1 task package, its derived worker image and its frozen F2P ids.

Layout (hashed in package.json):
  public/instruction.md        native problem statement (the native instruction, verbatim)
  public/workspace.tar         native W0: /testbed after the native level-1 initialisation (gitignored, hash-bound)
  private/row.json             the HF row at the pinned dataset revision
  private/build/*              worker-image recipe (native init script, mask patch, Dockerfiles)
  private/f2p_ids.json         F2P node ids frozen from the reference run (the score denominator)
  private/gold.json            reference-run summary (F2P/P2P exits, grade seconds)
  private/leak_scan.json       worker-image scan: no /root/my_repo, no F2P file, no importable copy of a masked file,
                               empty package caches, single-commit git history

The worker image is the pinned original image after the native initialisation (the masked repo, the F2P file deleted,
bytecode and package caches purged, /root/my_repo deleted), with /testbed replaced by a symlink to the runner's
/workspace. Grading always starts a fresh container of the original image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd

from trajectory_retention.execution.archives import extract_directory
from . import source as S
from .grading import CLEAR_BYTECODE, GIT_INIT, _baseline, run_private_grade

PACKAGE_VERSION = "featurebench-package/v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect(ref: str) -> dict:
    return json.loads(subprocess.run(["docker", "image", "inspect", ref], check=True, capture_output=True).stdout)[0]


def load_row(instance_id: str, parquet_dir: Path) -> dict:
    for split, digest in S.PARQUET_SHA256.items():
        path = Path(parquet_dir) / f"data/{split}-00000-of-00001.parquet"
        if _sha(path) != digest:
            raise ValueError(f"FeatureBench {split} parquet differs from the pinned revision")
        df = pd.read_parquet(path)
        hit = df[df.instance_id == instance_id]
        if len(hit):
            row = hit.iloc[0].to_dict()
            row["FAIL_TO_PASS"] = [str(x) for x in row["FAIL_TO_PASS"]]
            row["PASS_TO_PASS"] = [str(x) for x in row["PASS_TO_PASS"]]
            if not isinstance(row["repo_settings"], str):
                row["repo_settings"] = json.dumps(row["repo_settings"])
            row["split_found_in"] = split
            return row
    raise ValueError("Instance not in the pinned FeatureBench splits")


def test_command(row: dict) -> tuple[str, int]:
    settings = json.loads(row["repo_settings"])
    cmd = {"pydantic/pydantic": "pytest -rA -v --color=no"}.get(row["repo"]) or \
        settings.get("test_cmd", "pytest -rA -p no:cacheprovider --color=no")
    one = settings.get("timeout_one", 10)
    if one:
        cmd = f"{cmd} --timeout={one}"
    if settings.get("use_uv"):
        raise ValueError("uv-run tasks are not supported by this adapter")
    return cmd, int(settings.get("timeout_run", 600))


def masked_paths(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)))


def removed_markers(patch: str, n: int = 3) -> list[str]:
    lines = {l[1:].strip() for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")}
    lines = [l for l in lines if len(l) > 30 and not l.startswith(("#", '"', "'")) and "'" not in l]
    return sorted(lines, key=lambda l: (-len(l), l))[:n]


def build_script(f2p: str) -> str:
    return (_baseline(f2p, strict=True) + CLEAR_BYTECODE + GIT_INIT + S.CLEAR_CACHES +
            "rm -rf /root/my_repo /tmp/grade\n"
            f"getent group {S.TASK_USER} >/dev/null || groupadd --system {S.TASK_USER}\n"
            f"id -u {S.TASK_USER} >/dev/null 2>&1 || useradd --system --gid {S.TASK_USER} --home-dir /home/{S.TASK_USER} "
            f"--create-home --shell /bin/bash {S.TASK_USER}\n"
            f"chown -R {S.TASK_USER}:{S.TASK_USER} /testbed\n")


def leak_scan(image: str, row: dict) -> dict:
    f2p = row["FAIL_TO_PASS"][0]
    finds = " ".join(f"-path '*/{p}' -o -path '*/{str(Path(p).parent)}/__pycache__/{Path(p).stem}.*.pyc' -o"
                     for p in masked_paths(row["patch"]) if p.endswith(".py")).removesuffix(" -o")
    greps = " ".join(f"-e {json.dumps(m)}" for m in removed_markers(row["patch"]))
    script = (
        "echo MY_REPO=$(test -e /root/my_repo && echo present || echo absent)\n"
        f"echo F2P=$(test -e /testbed/{f2p} && echo present || echo absent)\n"
        f"echo COPIES_BEGIN; find / -xdev \\( -path /proc -o -path /sys -o -path /workspace -o -path /testbed \\) -prune "
        f"-o \\( {finds} \\) -print 2>/dev/null; echo COPIES_END\n"
        f"echo MARKERS_BEGIN; grep -rlF --include='*.py' {greps} / --exclude-dir=proc --exclude-dir=sys "
        "--exclude-dir=workspace --exclude-dir=testbed 2>/dev/null | head -20; echo MARKERS_END\n"
        "echo CACHES_BEGIN; for p in /opt/miniconda3/pkgs /root/.cache/pip /root/.cache/uv /root/.conda/pkgs "
        "/home/*/.cache/pip; do [ -d \"$p\" ] && [ -n \"$(ls -A \"$p\" 2>/dev/null)\" ] && echo \"$p\"; done; echo CACHES_END\n")
    out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "0:0", "--entrypoint", "bash", image,
                          "-c", script], capture_output=True, text=True, timeout=3600).stdout

    def block(name):
        return [l for l in out.split(name + "_BEGIN", 1)[1].split(name + "_END", 1)[0].splitlines() if l.strip()]
    result = {"my_repo_absent": "MY_REPO=absent" in out, "f2p_absent": "F2P=absent" in out,
              "masked_file_copies": block("COPIES"), "removed_code_markers_found_in": block("MARKERS"),
              "nonempty_caches": block("CACHES"), "markers": removed_markers(row["patch"])}
    result["clean"] = (result["my_repo_absent"] and result["f2p_absent"] and not result["masked_file_copies"]
                       and not result["removed_code_markers_found_in"] and not result["nonempty_caches"])
    return result


def reference_workspace(package: Path, dest: Path) -> Path:
    """Native W0 with the masking patch reversed = the unmasked upstream implementation (grader-side only)."""
    extract_directory((Path(package) / "public/workspace.tar").read_bytes(), dest)
    row = json.loads((Path(package) / "private/row.json").read_text())
    patch = row["patch"] if row["patch"].endswith("\n") else row["patch"] + "\n"
    subprocess.run(["git", "apply", "-R", "--whitespace=fix", "-"], cwd=dest, input=patch.encode(), check=True,
                   capture_output=True)
    return dest


def build(*, instance_id: str, parquet_dir: Path, out: Path, family_split: str = "unassigned") -> dict:
    out = Path(out)
    if out.exists():
        raise ValueError("Package directory already exists; packages are immutable")
    row = load_row(instance_id, parquet_dir)
    digest = S.IMAGE_DIGESTS[row["image_name"]]
    original = _inspect(f"{row['image_name']}@{digest}")
    if original.get("Architecture") != "amd64":
        raise ValueError("FeatureBench images must be linux/amd64")
    f2p = row["FAIL_TO_PASS"][0]
    if len(row["FAIL_TO_PASS"]) != 1:
        raise ValueError("Level-1 tasks carry exactly one F2P file")
    cmd, timeout_run = test_command(row)
    short = instance_id.split(".")[0].split("__")[1] + "-" + instance_id.split(".")[2]
    (out / "public").mkdir(parents=True)
    (out / "private/build").mkdir(parents=True)
    (out / "public/instruction.md").write_text(row["problem_statement"], encoding="utf-8")
    (out / "private/row.json").write_text(json.dumps(row, indent=1, sort_keys=True, default=str) + "\n")
    bdir = out / "private/build"
    (bdir / "mask.patch").write_text(row["patch"] if row["patch"].endswith("\n") else row["patch"] + "\n")
    (bdir / "init.sh").write_text(build_script(f2p))
    (bdir / "Dockerfile.init").write_text(
        f"FROM {row['image_name']}@{digest}\n"
        "COPY mask.patch /tmp/grade/mask.patch\nCOPY init.sh /tmp/fb-init.sh\n"
        "RUN bash /tmp/fb-init.sh && rm -f /tmp/fb-init.sh\n")
    init_tag = f"tr-featurebench-init:{short}-v2"
    subprocess.run(["docker", "build", "--network", "none", "--platform", "linux/amd64", "-q", "-t", init_tag, "-f",
                    str(bdir / "Dockerfile.init"), str(bdir)], check=True, capture_output=True)
    w0 = subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "0:0", "--entrypoint", "tar", init_tag,
                         "-C", "/testbed", "--numeric-owner", "--format=pax", "-cf", "-", "."],
                        check=True, capture_output=True, timeout=1800).stdout
    (out / "public/workspace.tar").write_bytes(w0)
    (bdir / "Dockerfile.worker").write_text(
        f"FROM {init_tag}\n"
        f"RUN rm -rf /testbed && mkdir -p /workspace && chown {S.TASK_USER}:{S.TASK_USER} /workspace && "
        "ln -s /workspace /testbed\n"
        "ENV PATH=/opt/miniconda3/envs/testbed/bin:$PATH CONDA_DEFAULT_ENV=testbed\n"
        "WORKDIR /workspace\n")
    tag = f"tr-featurebench-worker:{short}-v2"
    subprocess.run(["docker", "build", "--network", "none", "--platform", "linux/amd64", "-q", "-t", tag, "-f",
                    str(bdir / "Dockerfile.worker"), str(bdir)], check=True, capture_output=True)
    worker = _inspect(tag)
    scan = leak_scan(worker["Id"], row)
    (out / "private/leak_scan.json").write_text(json.dumps(scan, indent=1) + "\n")
    manifest = {
        "package_version": PACKAGE_VERSION, "task_id": f"featurebench-{short}-v2", "upstream_task_id": instance_id,
        "family_id": f"featurebench:{row['repo']}", "family_split": family_split, "source_benchmark": "FeatureBench",
        "source_url": S.SOURCE_URL, "source_revision": S.SOURCE_REVISION, "dataset": S.DATASET,
        "dataset_revision": S.DATASET_REVISION, "dataset_sha": S.DATASET_SHA, "split_found_in": row["split_found_in"],
        "grading_image": f"{row['image_name']}@{digest}", "grading_image_id": original["Id"],
        "environment_image": tag, "environment_image_id": worker["Id"], "environment_architecture": worker.get("Architecture"),
        "f2p_file": f2p, "p2p_files": row["PASS_TO_PASS"], "test_command": cmd, "timeout_run": timeout_run,
        "masked_files": masked_paths(row["patch"]), "lifecycle": S.LIFECYCLE, "native_budget": S.NATIVE_BUDGET,
        "deviations": list(S.DEVIATIONS), "leak_scan_clean": scan["clean"],
    }
    # Freeze the F2P denominator from a reference run in the original image.
    logs = Path.home() / ".cache/trajectory-retention/featurebench-gold" / manifest["task_id"]
    shutil.rmtree(logs, ignore_errors=True)
    ref = reference_workspace(out, logs / "ref")
    gold = run_private_grade(package=out, manifest=manifest, workspace=ref, output=logs / "grade")
    if gold["infrastructure_errors"]:
        raise RuntimeError(f"reference grade failed: {gold['infrastructure_errors']}")
    # Frozen denominator = the F2P ids the reference passes; ids it cannot pass (module-level skips, environment
    # failures) are recorded, not scored. P2P ids the reference fails are not counted as agent regressions.
    ids = sorted(t for t, v in gold["f2p_status"].items() if v in ("PASSED", "XFAIL"))
    nonpass = sorted(t for t, v in gold["f2p_status"].items() if v not in ("PASSED", "XFAIL"))
    (out / "private/f2p_ids.json").write_text(json.dumps(ids, indent=1) + "\n")
    (out / "private/gold.json").write_text(json.dumps(
        {"native": gold["native"], "nonpass_f2p": nonpass, "p2p_failed_ids": gold["p2p_failed_ids"],
         "seconds": gold["seconds"], "submission": gold["submission"], "guarded": gold["guarded"]}, indent=1) + "\n")
    manifest.update({"f2p_count": len(ids), "reference_nonpass_f2p": nonpass,
                     "reference_p2p_failed_ids": gold["p2p_failed_ids"], "reference_resolved": gold["native"]["resolved"]})
    manifest["files"] = {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_package(package: Path) -> dict:
    package = Path(package)
    manifest = json.loads((package / "package.json").read_text())
    if manifest.get("package_version") != PACKAGE_VERSION:
        raise ValueError("Unsupported FeatureBench package version")
    present = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()} - {"package.json"}
    if present != set(manifest["files"]):
        raise ValueError("Frozen package changed: file set")
    for rel, digest in manifest["files"].items():
        if _sha(package / rel) != digest:
            raise ValueError(f"Frozen package changed: {rel}")
    return manifest
