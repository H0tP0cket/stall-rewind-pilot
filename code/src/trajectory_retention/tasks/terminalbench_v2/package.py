"""Build one terminalbench-v2 package: images, natural start, frozen hidden-test ids, manifest.

Raw build evidence goes to artifacts/v2/lane/tasks/<task>/ (gitignored). The package directory in Git holds only
package.json and public/; the private files (hidden tests, reference, build context, task.toml, checks.json) go to
``adapter.PRIVATE_ROOT/<task_id>`` outside Git and are rebuilt byte-for-byte from the pinned checkout by
``restore_private``. package.json binds every file by hash, the image IDs and the evaluator sources.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import tempfile
import tomllib

from . import environment as E
from . import source as S
from .adapter import (ADAPTER_ENTRY, LANE, PRIVATE_ROOT, SCHEMA, TerminalBenchV2TaskAdapter, evaluator_sources,
                      file_manifest, sha256)
from .grading import run_private_grade
from .states import Disposable

WORKTREE = Path(__file__).resolve().parents[4]
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / LANE
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / LANE
EVIDENCE = WORKTREE / "artifacts" / "v2" / "lane"
PACKAGE_VERSION = "1"
PRIVATE_PARTS = ("tests", "solution", "environment")

ADAPTATION = {
    "version": "terminalbench-v2.adaptation.v1",
    "items": [
        "Harness: the shared mini-SWE bash harness (one bash command per tool call through docker exec, the shared "
        "submission marker) replaces Terminus 2's tmux keystroke interface; instruction.md is delivered byte-for-byte.",
        "Submission loop (study-declared): each submission grades the captured W and R and the worker hears only the "
        "count of hidden tests still failing; the run ends at a passing submission or 3 identical failing sets.",
        "Environment: the upstream environment/Dockerfile is rebuilt natively for linux/arm64 from the same build "
        "context (the official alexgshaw/<task>:20251031 images are x86_64-only); every FROM is pinned by digest to "
        "the arm64 variant of the base release the official image was built from. apt/pip/curl steps re-resolve at "
        "build time as upstream (image IDs recorded).",
        "Workspace: the native /app directory becomes W (/workspace); /app is a symlink to /workspace in the worker "
        "image so every absolute /app path in the instruction and tests is unchanged.",
        "Harness interpreter: python3 is added to the worker image only when the native image lacks it.",
        "Network: worker and grader containers run with network none (natively allow_internet is true); tasks whose "
        "reference needs the network are excluded.",
        "Grading: the submitted W and captured filesystem R are restored into a fresh grader container (worker image + "
        "verifier layer) and the native pytest command runs there; live processes are not restored; R entries under "
        "/tb-verifier, /tb-grader, /tests and /logs are dropped (listed per grade).",
        "Verifier environment: the Python environment test.sh would create at grade time (uv 0.9.5, uvx -p 3.13 with the "
        "same -w pins) and its grade-time apt packages are installed ahead of time in the grader image only.",
        "Grader driver: test.sh is replaced by an equivalent pytest.main call with the same arguments, a "
        "researcher-only recorder plugin, a private empty ini file, and rootdir/confcutdir fixed to /tests.",
        "Resources: the worker and grader containers use the task.toml cpus and memory.",
    ],
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n")


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


def v1_dockerfile(task: str, kind: str, records: dict, spec: dict) -> str:
    """The Dockerfile text the terminalbench-v1 lane used for this image (its recipe with v1's tag and lane names)."""
    def ref(k: str) -> str:
        return f"tr-terminalbench/{task}:{k}-id-{records[k]['image']['id'].split(':')[-1][:16]}"
    if kind == "native":
        return E.pinned_native_dockerfile(task)[0]
    if kind == "worker":
        info = records["native"]["image"]
        user = f"USER {info['user']}" if info.get("user") not in ("", "root", "0", "0:0") else ""
        return E.WORKER_DOCKERFILE.replace("terminalbench-v2", "terminalbench-v1").format(
            native=ref("native"), app=E.NATIVE_WORKDIR, user=user)
    apt = ""
    if spec["apt_packages"]:
        apt = ("RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y "
               + " ".join(shlex.quote(p) for p in spec["apt_packages"]) + " && rm -rf /var/lib/apt/lists/*\n")
    user = records["worker"]["image"].get("user") or ""
    return E.GRADER_DOCKERFILE.replace("terminalbench-v2", "terminalbench-v1").format(
        uv=E.UV_IMAGE, worker=ref("worker"), root=E.VERIFIER_ROOT, apt=apt,
        user=f"USER {user}" if user not in ("", "root", "0", "0:0") else "", python=spec["python"],
        pins=" ".join(shlex.quote(p) for p in spec["with"]))


def adopt_v1(task: str, kind: str, records: dict, spec: dict) -> tuple[dict | None, str]:
    """Reuse the image the terminalbench-v1 lane built from the identical recipe (its committed environment record binds
    the Dockerfile hash and image ID) instead of rebuilding: a rebuild would re-resolve apt/pip packages. The image is
    re-tagged for this lane; nothing of the v1 lane is changed. (record, note); the record is None, with the reason in
    the note, when no identical recorded image exists locally."""
    record = WORKTREE / "benchmark_tasks/admission/terminalbench-v1/tasks" / f"terminalbench-{task}-v1" / "environment.json"
    if not record.is_file():
        return None, "no committed terminalbench-v1 environment record"
    v1 = json.loads(record.read_text()).get(kind) or {}
    image_id = (v1.get("image") or {}).get("id")
    text = v1_dockerfile(task, kind, records, spec)
    if not image_id or E.sha256_bytes(text.encode()) != v1.get("dockerfile_sha256"):
        return None, f"the v1 {kind} recipe differs from this lane's (for example a different base-image pin)"
    try:
        info = E.image_info(image_id)
    except Exception:
        return None, f"the v1 {kind} image is not present locally"
    if info["id"] != image_id or info["architecture"] != "arm64":
        return None, f"the v1 {kind} image is not the recorded arm64 image"
    if kind != "native":
        parent = records["native" if kind == "worker" else "worker"]["image"]["layers"]
        if info["layers"][:len(parent)] != parent:
            return None, f"the v1 {kind} image does not extend the adopted parent"
    tag = E.TAG.format(task=task, kind=kind)
    E.run(["docker", "tag", image_id, tag])
    rec = {"result": "built", "adopted": True, "seconds": 0.0, "tag": tag, "image": E.image_info(tag),
           "dockerfile_sha256": v1["dockerfile_sha256"], "bases": v1.get("bases"),
           "upstream_dockerfile_sha256": v1.get("upstream_dockerfile_sha256"),
           "interpreter": v1.get("interpreter"), "freeze": v1.get("freeze"),
           "adopted_from": {"lane": "terminalbench-v1", "record": str(record.relative_to(WORKTREE)),
                            "record_sha256": E.sha256_bytes(record.read_bytes()), "image_id": image_id,
                            "rule": "identical recipe: the v1 Dockerfile hash equals this lane's recipe rendered with "
                                    "v1's tag names; image present locally, arm64, extending the adopted parent"}}
    if kind == "native":
        rec["official_image"] = None
    elif kind == "worker":
        rec["extends_native_layers"] = True
    else:
        rec["extends_worker_layers"] = True
    return rec, "adopted"


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
            if rec.get("result") == "built":
                try:
                    if E.image_info(rec["tag"])["id"] == rec["image"]["id"]:
                        records[kind] = rec
                        continue
                except Exception:
                    pass
        chain_reused = False
        rec, note = adopt_v1(task, kind, records, spec)
        if rec is None:
            E.require_headroom(f"the {kind} image build of {task}")
            rec = (E.build_native(task) if kind == "native" else E.build_worker(task, records["native"])
                   if kind == "worker" else E.build_grader(task, records["worker"], spec))
            rec["v1_adoption"] = note
        write_json(path, rec)
        if rec.get("result") != "built":
            reason = f"{kind} image build failed"
            if "invalid signature" in (rec.get("log_tail") or ""):
                reason += (" (apt-get update: 'invalid signature', which here means the Docker VM disk is full: apt's "
                           "sandbox user cannot write)")
            return {"status": "HOLD", "reason": f"{reason}; v1 image not adopted: {note}", "build": rec,
                    "records": records}
        records[kind] = rec
    return {"status": "BUILT", "records": records, "test_spec": spec}


def stage_private(task: str, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in PRIVATE_PARTS:
        shutil.copytree(S.CHECKOUT / task / name, target / name, symlinks=True)
    shutil.copy2(S.CHECKOUT / task / "task.toml", target / "task.toml")


def collect_checks(task: str, rec: dict, workspace_archive: Path, private: Path, flags: list[str], limits: dict) -> dict:
    """Collect the hidden-test ids in the grader image from the captured natural start (W and R)."""
    root = EVIDENCE / "tasks" / task / "collect"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    box = Disposable(worker_image_id=rec["worker"]["image"]["id"], workspace_archive=workspace_archive, root=root / "box",
                     memory=f"{limits['memory_mb'] or 2048}m", cpus=str(limits["cpus"] or 1))
    try:
        start_w, start_r, _ = box.capture("start")
    finally:
        box.close()
    out = root / "grade"
    result = run_private_grade(grader_image_id=rec["grader"]["image"]["id"], worker_image_id=rec["worker"]["image"]["id"],
                               tests_dir=private / "tests", workspace=start_w, output=out, runtime=start_r,
                               timeout=300, pytest_flags=["--collect-only", "-q"], cpus=str(limits["cpus"] or 1),
                               memory=f"{limits['memory_mb'] or 2048}m")
    collected = []
    for line in (out / "reports.jsonl").read_text().splitlines() if (out / "reports.jsonl").is_file() else []:
        row = json.loads(line)
        if row.get("event") == "collected":
            collected = row["nodeids"]
    return {"checks": collected, "receipt": str(out), "infrastructure_errors": result["infrastructure_errors"]}


def build_package(task: str, *, reuse: bool = True) -> dict:
    images = build_images(task, reuse=reuse)
    if images["status"] != "BUILT":
        return images
    rec = images["records"]
    limits = native_limits(task)
    task_id = S.package_task_id(task)
    staging = EVIDENCE / "tasks" / task / "package-staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "public").mkdir(parents=True)
    app_tar = E.export_path(rec["native"]["image"]["id"], E.NATIVE_WORKDIR)
    app_manifest = E.app_tar_manifest(app_tar)
    symlinks = sorted(k for k, v in app_manifest.items() if v["type"] in ("l", "h"))
    (staging / "public/workspace.tar.gz").write_bytes(E.workspace_tar_gz(app_tar))
    shutil.copy2(S.CHECKOUT / task / "instruction.md", staging / "public/instruction.md")
    private_staging = staging / "private"
    stage_private(task, private_staging)
    spec = images["test_spec"]
    flags = [a for a in spec["pytest_args"] if a not in ("--ctrf", "/logs/verifier/ctrf.json", "/tests/test_outputs.py")]
    collected = collect_checks(task, rec, staging / "public/workspace.tar.gz", private_staging, flags, limits)
    write_json(EVIDENCE / "tasks" / task / "collect.json", collected)
    if collected["infrastructure_errors"] or not collected["checks"]:
        return {"status": "HOLD", "reason": "grader: native test collection failed in the grader image", "collect": collected}
    write_json(private_staging / "checks.json", {"checks": collected["checks"],
               "rule": "pytest node ids collected from the native tests/test_outputs.py (rootdir /tests) in the grader "
                       "image on the captured natural start; one hidden test per collected node id"})
    native_python = E.probe(rec["native"]["image"]["id"], "command -v python3 && python3 -V")
    manifest = {
        "schema_version": SCHEMA, "task_id": task_id, "package_version": PACKAGE_VERSION, "lane": LANE,
        "upstream_task": task, "family_id": S.family_id(task), "split_policy": S.SPLIT_POLICY,
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
                       "architecture": rec["native"]["image"].get("architecture"), "bases": rec["native"]["bases"],
                       "upstream_dockerfile_sha256": rec["native"]["upstream_dockerfile_sha256"],
                       "dockerfile_sha256": rec["native"]["dockerfile_sha256"]},
            "worker": {"image_id": rec["worker"]["image"]["id"], "tag": rec["worker"]["tag"],
                       "architecture": rec["worker"]["image"].get("architecture"),
                       "dockerfile_sha256": rec["worker"]["dockerfile_sha256"],
                       "extends_native_layers": rec["worker"].get("extends_native_layers"),
                       "python3_added": native_python.returncode != 0},
            "grader": {"image_id": rec["grader"]["image"]["id"], "tag": rec["grader"]["tag"],
                       "dockerfile_sha256": rec["grader"]["dockerfile_sha256"], "uv_image": E.UV_IMAGE,
                       "extends_worker_layers": rec["grader"].get("extends_worker_layers"),
                       "interpreter": rec["grader"].get("interpreter"), "freeze": rec["grader"].get("freeze"),
                       "grade_time_apt_packages": spec["apt_packages"]},
            "adopted_from_v1": {k: rec[k]["adopted_from"] for k in ("native", "worker", "grader") if rec[k].get("adopted")},
        },
        "grading": {"timeout_seconds": int(limits["verifier_timeout_sec"] or 900), "pytest_flags": flags,
                    "cpus": str(limits["cpus"] or 1), "memory": f"{limits['memory_mb'] or 2048}m",
                    "native_uvx_line": spec["uvx_line"], "native_apt_line": spec["apt_line"]},
        "native_start": {"source": "native image " + E.NATIVE_WORKDIR, "entries": len(app_manifest),
                         "regular_bytes": sum(v["size"] or 0 for v in app_manifest.values()),
                         "symlinks_or_hardlinks": symlinks, "manifest_sha256": E.sha256_bytes(
                             json.dumps(app_manifest, sort_keys=True).encode())},
        "private_location": "adapter.PRIVATE_ROOT/<task_id> (outside Git; restore_private rebuilds it from the pinned checkout)",
        "adaptation": ADAPTATION,
        "worker_prompt": None,
        "evaluator_sources": evaluator_sources(),
    }
    package = PACKAGES / task_id
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copytree(staging / "public", package / "public")
    private = PRIVATE_ROOT / task_id
    if private.exists():
        shutil.rmtree(private)
    private.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(private_staging, private, symlinks=True)
    manifest["public_files"] = file_manifest(package / "public")
    manifest["private_files"] = file_manifest(private)
    write_json(package / "package.json", manifest)
    adapter = TerminalBenchV2TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    manifest["worker_prompt"] = adapter.worker_prompt("task").binding()
    write_json(package / "package.json", manifest)
    TerminalBenchV2TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    write_json(EVIDENCE / "tasks" / task / "app-manifest.json", app_manifest)
    shutil.rmtree(private_staging)
    return {"status": "PACKAGED", "task_id": task_id, "package": str(package), "hidden_tests": len(collected["checks"]),
            "symlinks": symlinks, "start_bytes": manifest["native_start"]["regular_bytes"],
            "package_sha256": sha256(package / "package.json")}


def restore_private(task_id: str) -> dict:
    """Rebuild a package's private files outside Git from the pinned checkout and check them against package.json."""
    package = PACKAGES / task_id
    manifest = json.loads((package / "package.json").read_text())
    S.verify_checkout()
    target = PRIVATE_ROOT / task_id
    staging = Path(tempfile.mkdtemp(prefix="private-", dir=PRIVATE_ROOT.parent if PRIVATE_ROOT.parent.exists() else None))
    stage_private(manifest["upstream_task"], staging / "p")
    receipt = EVIDENCE / "tasks" / manifest["upstream_task"] / "collect.json"
    if receipt.is_file():
        collected = json.loads(receipt.read_text())["checks"]
    else:  # re-collect in the package's own grader image
        env = manifest["environment"]
        rec = {"worker": {"image": {"id": env["worker"]["image_id"]}}, "grader": {"image": {"id": env["grader"]["image_id"]}}}
        (EVIDENCE / "tasks" / manifest["upstream_task"]).mkdir(parents=True, exist_ok=True)
        collected = collect_checks(manifest["upstream_task"], rec, package / "public/workspace.tar.gz", staging / "p",
                                   manifest["grading"]["pytest_flags"], manifest["native_limits"])["checks"]
    write_json(staging / "p" / "checks.json", {"checks": collected,
               "rule": "pytest node ids collected from the native tests/test_outputs.py (rootdir /tests) in the grader "
                       "image on the captured natural start; one hidden test per collected node id"})
    if file_manifest(staging / "p") != manifest["private_files"]:
        shutil.rmtree(staging)
        raise ValueError("Rebuilt private files differ from the package manifest")
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(staging / "p"), target)
    shutil.rmtree(staging)
    return {"task_id": task_id, "private": str(target), "files": len(manifest["private_files"])}
