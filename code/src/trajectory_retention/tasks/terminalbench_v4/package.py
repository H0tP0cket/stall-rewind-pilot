"""Build one terminalbench-v4 package: images, natural start, frozen hidden-test ids, deviations, manifest.

Raw build evidence goes to ``source.EVIDENCE/tasks/<task>/`` (local, gitignored). The package directory in Git holds only
package.json and public/; the private files (hidden tests, reference, build context, task.toml, README.md, cheat/,
checks.json) go to ``adapter.PRIVATE_ROOT/<task_id>`` outside Git and are rebuilt byte-for-byte from the pinned checkout
by ``restore_private``. package.json binds every file by hash, the image IDs, the evaluator sources and the recorded
deviations from the official environment.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

from .source import remove_tree
from . import environment as E
from . import source as S
from .adapter import (ADAPTER_ENTRY, LANE, PRIVATE_ROOT, SCHEMA, TerminalBenchV4TaskAdapter, evaluator_sources,
                      file_manifest, sha256)
from .grading import CONVENTION_DIR, harbor_command, run_private_grade
from .states import Disposable

WORKTREE = S.WORKTREE
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / LANE
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / LANE
EVIDENCE = S.EVIDENCE
PACKAGE_VERSION = "1"
PRIVATE_PARTS = ("tests", "solution", "environment", "cheat")
PRIVATE_FILES = ("task.toml", "README.md")
CHECKS_RULE = ("pytest node ids collected by the task's own tests/test.sh in the verifier image (recorder plugin, "
               "--collect-only) on the captured natural start; one hidden test per collected node id")

ADAPTATION = {
    "version": "terminalbench-v4.adaptation.v1",
    "items": [
        "Harness: the shared mini-SWE bash harness (one bash command per tool call through docker exec, the shared "
        "submission marker) replaces Harbor's agent; instruction.md is delivered as Harbor delivers it (leading canary "
        "comment lines stripped by Harbor's strip_canary), otherwise byte-for-byte.",
        "Submission loop (study-declared): each submission grades the captured W and R and the worker hears only the "
        "count of hidden tests still failing; the run ends at a passing submission or 3 identical failing sets.",
        "Environment: environment/Dockerfile and tests/Dockerfile are rebuilt natively for linux/arm64 from the same "
        "build contexts (the official harborframework/terminal-bench images are linux/amd64 only); every FROM is pinned "
        "by digest to the arm64 variant of the base release the official image was built from. apt/pip/curl steps "
        "re-resolve at build time as upstream (image IDs recorded).",
        "Workspace: the native /app directory becomes W (/workspace); /app is a symlink to /workspace in the worker "
        "image, so every absolute /app path in the instruction and tests is unchanged. The natural start is the native "
        "/app with its numeric owners and modes (and /app's own owner and mode).",
        "Harness interpreter: python3 is added to the worker image only when the native image lacks it.",
        "Network: worker, source and verifier containers run with network none (TB4 runs with open internet); tasks "
        "whose reference needs the network are excluded.",
        "Grading: Harbor's separate verifier reproduced offline (grading.py): the submitted W and captured filesystem R "
        "are restored into a fresh worker container, the declared artifacts (and /logs/artifacts) are copied out with "
        "docker cp and into a fresh verifier container as Harbor does, and the native tests/test.sh runs there with "
        "Harbor's command; live processes are not restored (Harbor's verifier receives files only).",
        "Grader recorder: a researcher-only pytest plugin is loaded into the task's own pytest run via PYTEST_ADDOPTS, "
        "importable through PYTHONPATH or (when test.sh sets its own PYTHONPATH) the user site under PYTHONUSERBASE "
        "(all removed again on import), so per-test statuses carry pytest node ids; Harbor's reward file decides the "
        "native pass.",
        "Resources: the worker and verifier containers use the task.toml cpus and memory; storage_mb is not enforced.",
    ],
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=str) + "\n")


def task_toml(task: str) -> dict:
    return tomllib.loads((S.task_dir(task) / "task.toml").read_text())


def native_limits(task: str) -> dict:
    t = task_toml(task)
    env = t.get("environment", {})
    return {"cpus": env.get("cpus"), "memory_mb": env.get("memory_mb"), "storage_mb": env.get("storage_mb"),
            "gpus": env.get("gpus"), "allow_internet": env.get("allow_internet"),
            "agent_timeout_sec": t.get("agent", {}).get("timeout_sec"),
            "verifier_timeout_sec": t.get("verifier", {}).get("timeout_sec"),
            "verifier_environment_mode": t.get("verifier", {}).get("environment_mode"),
            "verifier_user": t.get("verifier", {}).get("user"), "verifier_env": t.get("verifier", {}).get("env") or {},
            "build_timeout_sec": env.get("build_timeout_sec")}


def artifacts(task: str) -> list[str]:
    values = task_toml(task).get("artifacts") or []
    if not values or not all(isinstance(v, str) and v.startswith("/") for v in values):
        raise ValueError(f"{task}: task.toml artifacts must be a non-empty list of absolute paths")
    return list(values)


def build_images(task: str, *, reuse: bool = True) -> dict:
    out = EVIDENCE / "tasks" / task
    out.mkdir(parents=True, exist_ok=True)
    records, disk = {}, {}
    chain_reused = True
    for kind in E.KINDS:
        path = out / f"{kind}-build.json"
        if reuse and (chain_reused or kind == "verifier") and path.is_file():
            rec = json.loads(path.read_text())
            if rec.get("result") == "built":
                try:
                    if E.image_info(rec["tag"])["id"] == rec["image"]["id"]:
                        records[kind] = rec
                        continue
                except Exception:
                    pass
        if kind != "verifier":
            chain_reused = False
        estimate = E.estimate_bytes(task, kind)
        try:
            disk[kind] = E.require_headroom(f"the {kind} image build of {task}", estimate)
        except E.DiskHeadroom as exc:
            return {"status": "HOLD", "reason": str(exc), "docker_disk": True, "records": records}
        rec = E.build_role(task, "native") if kind == "native" else (
            E.build_worker(task, records["native"]) if kind == "worker" else E.build_role(task, "verifier"))
        rec["disk_before"] = disk[kind]
        rec["disk_after"] = E.docker_disk()
        write_json(path, rec)
        if rec.get("result") != "built":
            tail = rec.get("log_tail") or ""
            disk_full = any(s in tail for s in ("No space left", "invalid signature", "no space left"))
            before = rec["disk_before"]
            reason = (f"docker disk: the {kind} image build failed with "
                      + ("'No space left on device'" if "o space left" in tail else
                         "apt 'invalid signature' (apt's non-root user cannot write)")
                      + f"; Docker VM disk before the build: {before['free_bytes'] / 1e9:.1f} GB free counting "
                        f"root-reserved blocks, {before['non_root_available_bytes'] / 1e9:.1f} GB available to non-root"
                      if disk_full else f"{kind} image build failed")
            return {"status": "HOLD", "reason": reason, "docker_disk": disk_full, "build": rec, "records": records}
        records[kind] = rec
    return {"status": "BUILT", "records": records}


def stage_private(task: str, target: Path) -> None:
    if target.exists():
        remove_tree(target)
    target.mkdir(parents=True)
    root = S.task_dir(task)
    for name in PRIVATE_PARTS:
        if (root / name).is_dir():
            shutil.copytree(root / name, target / name, symlinks=True)
    for name in PRIVATE_FILES:
        if (root / name).is_file():
            shutil.copy2(root / name, target / name)


def verifier_checks(task: str, verifier: dict) -> dict:
    """Facts the grader relies on, read from the built verifier image (never assumed)."""
    info = verifier["image"]
    env = dict(e.split("=", 1) for e in info["env"] if "=" in e)
    tests_hash = E.probe(info["id"], "cd /tests && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum",
                         user="0:0")
    image_tests = {}
    for line in tests_hash.stdout.decode().splitlines():
        digest, _, rel = line.partition("  ")
        image_tests[rel[2:]] = digest
    upstream = {k: v for k, v in E.context_manifest(S.task_dir(task) / "tests").items()}
    tools = E.probe(info["id"], "for t in bash timeout tar date find chmod mkdir; do command -v $t >/dev/null || echo "
                                "missing:$t; done; echo ok", user="0:0").stdout.decode()
    out = {"user": info.get("user") or "", "workdir": info.get("workdir") or "/",
           "user_is_root": info.get("user") in ("", "root", "0", "0:0"),
           "recorder_env_free": not {"PYTHONPATH", "PYTHONUSERBASE", "PYTEST_ADDOPTS"} & set(env),
           "tests_in_image_equal_upstream": image_tests == upstream, "tests_in_image": len(image_tests),
           "tools": "ok" if tools.strip().endswith("ok") and "missing:" not in tools else tools.strip()}
    out["passed"] = (out["user_is_root"] and out["recorder_env_free"] and out["tests_in_image_equal_upstream"]
                     and out["tools"] == "ok")
    return out


def collect_checks(task: str, rec: dict, workspace_archive: Path, limits: dict, grading: dict) -> dict:
    """Collect the hidden-test ids: the task's own test.sh, recorder with --collect-only, on the captured natural start."""
    root = EVIDENCE / "tasks" / task / "collect"
    if root.exists():
        remove_tree(root)
    root.mkdir(parents=True)
    box = Disposable(worker_image_id=rec["worker"]["image"]["id"], workspace_archive=workspace_archive, root=root / "box",
                     memory=grading["memory"], cpus=grading["cpus"])
    try:
        start_w, start_r, _ = box.capture("start")
    finally:
        box.close()
    out = root / "grade"
    result = run_private_grade(verifier_image_id=rec["verifier"]["image"]["id"], worker_image_id=rec["worker"]["image"]["id"],
                               artifacts=grading["artifacts"], workspace=start_w, output=out, runtime=start_r,
                               timeout=grading["timeout_seconds"], cpus=grading["cpus"], memory=grading["memory"],
                               recorder_args="--collect-only")
    collected = []
    for line in (out / "reports.jsonl").read_text().splitlines() if (out / "reports.jsonl").is_file() else []:
        row = json.loads(line)
        if row.get("event") == "collected":
            collected = row["nodeids"]
    return {"checks": collected, "receipt": str(out), "infrastructure_errors": result["infrastructure_errors"]}


def deviations(task: str, rec: dict, limits: dict, app_manifest: dict) -> list[str]:
    """Every difference from the official TB4 environment for this task (recorded per package)."""
    out = []
    for kind, role in (("native", "environment"), ("verifier", "verifier")):
        for base in rec[kind]["bases"]:
            line = (f"{role} image: rebuilt for linux/arm64 (official {rec[kind]['official_image']['image']} is "
                    f"linux/amd64); FROM {base['ref']} pinned to {base['pinned']}")
            if base.get("match") == "layer_prefix":
                line += f" (the arm64 variant of {base['matched_tag']}, whose amd64 layers prefix the official image)"
            elif base.get("match") == "python_version_only":
                line += (f" (same CPython {base.get('official_python_version')} release as the official image; the "
                         f"re-published tag's layers differ)")
            else:
                line += " (official base release not identified; the tag's current arm64 digest)"
            if "@sha256:" in base["ref"]:
                line += "; the Dockerfile's own digest pin is an amd64 manifest and cannot run natively here"
            out.append(line)
        out.append(f"{role} image: apt/pip/curl steps re-resolved at this build (official images were built "
                   f"{str(rec[kind]['official_image'].get('created'))[:10]}); package versions may differ where the "
                   f"Dockerfile does not pin them")
    if task == "rs-archive-clone":
        out.append("environment: the Dockerfile's own `uname -m` branch installs the shipped aarch64 build of "
                   "/app/artifacts/archive-tool (archive-tool-linux-aarch64); the official image carries the x86_64 "
                   "build. The hidden fixtures were generated by the reference tool; the gate's reference 3/3 checks the "
                   "aarch64 build against them")
    worker = rec["worker"]
    if not str(worker.get("python3", "")).startswith("/"):
        out.append("worker image: python3 added for the harness")
    out.append("worker image: /app is a symlink to /workspace (W); the natural start restores the native /app with its "
               f"numeric owners and modes (/app itself {oct(app_manifest['.']['mode'])} uid {app_manifest['.']['uid']})")
    out.append("network: none for worker and verifier (TB4 default: open internet)")
    out.append(f"resources: cpus {limits['cpus']}, memory {limits['memory_mb']} MB as task.toml; storage_mb "
               f"{limits['storage_mb']} not enforced (Docker Desktop)")
    out.append("agent: the study's mini-SWE bash harness and submission loop instead of Harbor's agent; no wall-clock "
               f"cap (the instruction's {int(limits['agent_timeout_sec'] or 0)}-second line is shown unchanged)")
    out.append("grading: Harbor's separate verifier reproduced offline (grading.py) on the captured W and R, network "
               "none; a researcher-only pytest recorder plugin is added through PYTEST_ADDOPTS/PYTHONPATH/PYTHONUSERBASE")
    return out


def build_package(task: str, static_row: dict, *, reuse: bool = True) -> dict:
    images = build_images(task, reuse=reuse)
    if images["status"] != "BUILT":
        return images
    rec = images["records"]
    limits = native_limits(task)
    if limits["verifier_environment_mode"] != "separate" or limits["verifier_user"] is not None or limits["verifier_env"]:
        return {"status": "HOLD", "reason": "grader: task.toml verifier is not plain separate mode (user/env set)"}
    task_id = S.package_task_id(task)
    grading = {"mode": "harbor-separate-verifier", "artifacts": artifacts(task), "convention_dir": CONVENTION_DIR,
               "timeout_seconds": int(limits["verifier_timeout_sec"] or 900), "cpus": str(limits["cpus"] or 1),
               "memory": f"{limits['memory_mb'] or 2048}m", "harbor_command": harbor_command(),
               "harbor_revision": S.HARBOR_REVISION}
    vcheck = verifier_checks(task, rec["verifier"])
    write_json(EVIDENCE / "tasks" / task / "verifier-checks.json", vcheck)
    if not vcheck["passed"]:
        return {"status": "HOLD", "reason": "grader: verifier image outside what the lane's grader reproduces: "
                                            + json.dumps({k: v for k, v in vcheck.items() if k != "passed"})[:400]}
    staging = EVIDENCE / "tasks" / task / "package-staging"
    if staging.exists():
        remove_tree(staging)
    (staging / "public").mkdir(parents=True)
    app_tar = E.export_app(rec["native"]["image"]["id"])
    app_manifest = E.app_tar_manifest(app_tar)
    (staging / "public/workspace.tar.gz").write_bytes(E.gzip_bytes(app_tar))
    shutil.copy2(S.task_dir(task) / "instruction.md", staging / "public/instruction.md")
    private_staging = staging / "private"
    stage_private(task, private_staging)
    collected = collect_checks(task, rec, staging / "public/workspace.tar.gz", limits, grading)
    write_json(EVIDENCE / "tasks" / task / "collect.json", collected)
    if collected["infrastructure_errors"] or not collected["checks"]:
        return {"status": "HOLD", "reason": "grader: test collection through the task's test.sh failed in the verifier "
                                            "image", "collect": collected}
    write_json(private_staging / "checks.json", {"checks": collected["checks"], "rule": CHECKS_RULE})
    links = sorted(k for k, v in app_manifest.items() if v["type"] in ("l", "h"))
    manifest = {
        "schema_version": SCHEMA, "task_id": task_id, "package_version": PACKAGE_VERSION, "lane": LANE,
        "upstream_task": task, "family_id": S.family_id(task), "split_policy": S.SPLIT_POLICY,
        "adapter_entry": ADAPTER_ENTRY, "source_url": S.SOURCE_URL, "source_tag": S.SOURCE_TAG,
        "source_revision": S.SOURCE_REVISION, "license": S.LICENSE, "ratification": S.RATIFICATION,
        "upstream_tree": S.task_tree(task),
        "upstream_hashes": {name: S.sha256_file(S.task_dir(task) / name) for name in
                            ("instruction.md", "task.toml", "environment/Dockerfile", "tests/Dockerfile", "tests/test.sh",
                             "solution/solve.sh")},
        "native_limits": limits,
        "environment": {
            "platform": E.PLATFORM,
            "native": {"image_id": rec["native"]["image"]["id"], "tag": rec["native"]["tag"],
                       "architecture": rec["native"]["image"].get("architecture"), "bases": rec["native"]["bases"],
                       "official_image": rec["native"]["official_image"],
                       "upstream_dockerfile_sha256": rec["native"]["upstream_dockerfile_sha256"],
                       "dockerfile_sha256": rec["native"]["dockerfile_sha256"],
                       "user": rec["native"]["image"].get("user"), "workdir": rec["native"]["image"].get("workdir")},
            "worker": {"image_id": rec["worker"]["image"]["id"], "tag": rec["worker"]["tag"],
                       "architecture": rec["worker"]["image"].get("architecture"),
                       "dockerfile_sha256": rec["worker"]["dockerfile_sha256"],
                       "extends_native_layers": rec["worker"].get("extends_native_layers"),
                       "python3": rec["worker"].get("python3")},
            "verifier": {"image_id": rec["verifier"]["image"]["id"], "tag": rec["verifier"]["tag"],
                         "architecture": rec["verifier"]["image"].get("architecture"), "bases": rec["verifier"]["bases"],
                         "official_image": rec["verifier"]["official_image"],
                         "upstream_dockerfile_sha256": rec["verifier"]["upstream_dockerfile_sha256"],
                         "dockerfile_sha256": rec["verifier"]["dockerfile_sha256"], "checks": vcheck,
                         "private": "carries the hidden tests at /tests; never given to the worker"},
        },
        "grading": grading,
        "native_start": {"source": "native image " + E.NATIVE_WORKDIR + " (tar --numeric-owner, './' included)",
                         "entries": len(app_manifest), "app_root": app_manifest.get("."),
                         "regular_bytes": sum(v["size"] or 0 for v in app_manifest.values()),
                         "symlinks_or_hardlinks": links,
                         "manifest_sha256": E.sha256_bytes(json.dumps(app_manifest, sort_keys=True).encode())},
        "private_location": "adapter.PRIVATE_ROOT/<task_id> (outside Git; restore_private rebuilds it from the pinned checkout)",
        "adaptation": ADAPTATION,
        "deviations": deviations(task, rec, limits, app_manifest),
        "static": {k: static_row.get(k) for k in ("hidden_tests_static", "static_verdict", "static_reasons",
                                                  "solve_time_installs", "x86_only_binaries")},
        "worker_prompt": None,
        "evaluator_sources": evaluator_sources(),
    }
    package = PACKAGES / task_id
    if package.exists():
        remove_tree(package)
    package.mkdir(parents=True)
    shutil.copytree(staging / "public", package / "public")
    private = PRIVATE_ROOT / task_id
    if private.exists():
        remove_tree(private)
    private.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(private_staging, private, symlinks=True)
    manifest["public_files"] = file_manifest(package / "public")
    manifest["private_files"] = file_manifest(private)
    write_json(package / "package.json", manifest)
    adapter = TerminalBenchV4TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    manifest["worker_prompt"] = adapter.worker_prompt("task").binding()
    write_json(package / "package.json", manifest)
    TerminalBenchV4TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    write_json(EVIDENCE / "tasks" / task / "app-manifest.json", app_manifest)
    remove_tree(private_staging)
    return {"status": "PACKAGED", "task_id": task_id, "package": str(package), "hidden_tests": len(collected["checks"]),
            "start_bytes": manifest["native_start"]["regular_bytes"], "package_sha256": sha256(package / "package.json")}


def restore_private(task_id: str) -> dict:
    """Rebuild a package's private files outside Git from the pinned checkout and check them against package.json."""
    package = PACKAGES / task_id
    manifest = json.loads((package / "package.json").read_text())
    S.verify_checkout()
    target = PRIVATE_ROOT / task_id
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="private-", dir=PRIVATE_ROOT))
    stage_private(manifest["upstream_task"], staging / "p")
    receipt = EVIDENCE / "tasks" / manifest["upstream_task"] / "collect.json"
    if receipt.is_file():
        collected = json.loads(receipt.read_text())["checks"]
    else:  # re-collect in the package's own verifier image on its natural start
        env = manifest["environment"]
        rec = {"worker": {"image": {"id": env["worker"]["image_id"]}}, "verifier": {"image": {"id": env["verifier"]["image_id"]}}}
        collected = collect_checks(manifest["upstream_task"], rec, package / "public/workspace.tar.gz",
                                   manifest["native_limits"], manifest["grading"])["checks"]
    write_json(staging / "p" / "checks.json", {"checks": collected, "rule": CHECKS_RULE})
    if file_manifest(staging / "p") != manifest["private_files"]:
        remove_tree(staging)
        raise ValueError("Rebuilt private files differ from the package manifest")
    if target.exists():
        remove_tree(target)
    shutil.move(str(staging / "p"), target)
    remove_tree(staging)
    return {"task_id": task_id, "private": str(target), "files": len(manifest["private_files"])}


def unpack_start(package: Path) -> bytes:
    return gzip.decompress((Path(package) / "public/workspace.tar.gz").read_bytes())
