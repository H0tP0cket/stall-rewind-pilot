"""Family environment preparation, instance packaging and mechanical validation.

No model/provider calls. Every step writes raw evidence under
artifacts/task-admission/swesmith-long/ and returns a compact record.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from . import environment, source
from .adapter import SWESmithLongTaskAdapter, evaluator_sources, file_manifest, sha256
from .grading import CONFIG_NAMES, observe, run_suite
from .official import extract_metadata, installed_packages, normalize
from .selection import DATASET, DATASET_REVISION, SWESMITH_SOURCE_REVISION, is_test_path, patch_files

WORKTREE = Path(__file__).resolve().parents[4]
CACHE = Path.home() / ".cache" / "swesmith-long"
EVIDENCE = WORKTREE / "artifacts" / "task-admission" / "swesmith-long"
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / "swesmith-long-v1"
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / "swesmith-long-v1"
TEST_DIRS = ("tests", "test", "specs")
REFERENCE_RUNS = 3
GRADE_TIMEOUT = 900
ADAPTATION = {
    "version": "swesmith-long.adaptation.v1",
    "changes": [
        "FAIL_TO_PASS test files are kept in the worker's start tree (SWE-smith's branch head removes them); "
        "the worker can run the repository's own tests and see failures. Tests are upstream public files.",
        "No Git history or mirror branches in W (they contain the clean reference); the start tree is the "
        "verified bug commit's files plus files the editable install generates natively (e.g. version modules).",
        "Environment rebuilt natively for linux/arm64 from the official x86_64 image's exact package pins "
        "(official image emulation measured ~27x slower on this host). Python patch version per base image.",
        "Network disabled for worker and grader (dependencies are preinstalled).",
        "Grading restores the frozen test-support set (test directories, test files, conftest.py, root pytest "
        "configuration) before running pytest with --continue-on-collection-errors; only W is graded.",
        "PASS_TO_PASS tests that do not pass in every reference run in the rebuilt environment are excluded and listed.",
    ],
    "unchanged": ["problem_statement text (worker instruction) verbatim", "bug patch and FAIL_TO_PASS identities",
                  "pytest node identities and repository test command"],
}


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, **kwargs)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def official_image(repo_key: str) -> str:
    return "jyangballin/swesmith.x86_64." + repo_key.lower().replace("__", "_1776_")


def slug(repo_key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", repo_key.split("__", 1)[1].rsplit(".", 1)[0].lower()).strip("-")


def task_id_for(instance_id: str) -> str:
    repo_key = instance_id.rsplit(".", 1)[0]
    return f"swesmith-{slug(repo_key)}-{instance_id.rsplit('__', 1)[1]}-v1"


def native_pytest_args(profile: dict) -> tuple[list[str], str]:
    """Profile test command minus conda activation and log-format flags."""
    raw = profile["fields"].get("test_cmd")
    if isinstance(raw, dict):
        raw = raw["unparsed"]
    command = raw or "pytest --disable-warnings --color=no --tb=no --verbose"
    command = command.split(";")[-1].strip().strip("'\"f")
    visible = command
    args = command.split()
    if not args or args[0] != "pytest":
        raise ValueError("Unsupported native test command: " + str(raw))
    kept = [a for a in args[1:] if a not in ("--disable-warnings", "--color=no", "--tb=no", "--verbose", "-v", "-rA")]
    return kept, visible


def inventory(image_id: str, output: Path) -> list:
    here = Path(__file__).with_name("build_support")
    name = "tr-swesmith-long-scan-" + hashlib.sha256(output.as_posix().encode()).hexdigest()[:10]
    result = run(["docker", "run", "--rm", "--name", name, "--label", environment.LANE_LABEL, "--network", "none",
                  "--entrypoint", "python3", "--mount", f"type=bind,src={here},dst=/scan,readonly",
                  image_id, "-I", "/scan/scan_image.py"], timeout=900)
    rows = json.loads(result.stdout)
    output.write_text(json.dumps(rows) + "\n")
    return rows


def image_leak_scan(image_id: str, clean: Path, project_top_levels: list[str], output: Path) -> dict:
    rows = inventory(image_id, output / "image-inventory.json")
    clean_py = {sha256(p): p.relative_to(clean).as_posix() for p in clean.rglob("*.py")
                if p.is_file() and not p.is_symlink() and not is_test_path(p.relative_to(clean).as_posix())
                and p.stat().st_size > 200}
    identical = [{"image_path": path, "clean_file": clean_py[digest]} for path, size, digest in rows if digest in clean_py]
    named = [path for path, _, _ in rows
             if any(f"/site-packages/{top}/" in path or path.endswith(f"/site-packages/{top}.py") for top in project_top_levels)]
    archives = [path for path, _, _ in rows if re.search(r"\.(whl|tar\.gz|zip|egg)$", path)
                and any(normalize(Path(path).name).startswith(normalize(t) + "-") for t in project_top_levels)]
    caches = [path for path, _, _ in rows if "/.cache/pip/" in path or "/root/.cache/" in path]
    return {"files_scanned": len(rows), "identical_to_clean_source": identical, "project_package_copies": named,
            "project_archives": archives, "pip_cache_files": caches,
            "leak": bool(identical or named or archives or caches),
            "rule": "no image file (outside /workspace) identical to any clean non-test source .py >200 bytes; "
                    "no project top-level package or archive in the image; no pip cache"}


def prepare_family(family: dict, *, force: bool = False) -> dict:
    key = family["repo_key"]
    out = EVIDENCE / "families" / key
    record_path = out / "family.json"
    if record_path.is_file() and not force:
        return json.loads(record_path.read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    record = {"repo_key": key, "family_id": family["family_id"], "split": family["split"], "status": "running",
              "official_image": official_image(key)}
    try:
        official_dir = CACHE / "official" / key
        if not (official_dir / "official.json").is_file():
            if official_dir.exists():
                shutil.rmtree(official_dir)
            meta = extract_metadata(official_image(key), official_dir)
            write_json(official_dir / "official.json", meta)
        meta = json.loads((official_dir / "official.json").read_text())
        packages = installed_packages(official_dir)
        write_json(out / "official-packages.json", {"image": meta, **packages})
        record["official"] = {"manifest_digest": meta["manifest_digest"], "python": packages["python_version"],
                              "packages": len(packages["packages"])}
        project = [p for p in packages["packages"] if p["editable"]]
        if len(project) != 1:
            raise RuntimeError(f"expected one editable project, found {[p['name'] for p in project]}")
        project_name = project[0]["name"]
        repository = source.mirror(key, CACHE)
        context = CACHE / "build" / key
        if context.exists():
            shutil.rmtree(context)
        context.mkdir(parents=True)
        commit = source.export(repository, "main", context / "clean")
        record["mirror"] = {"url": f"https://github.com/{source.MIRROR_ORG}/{key}", "main_commit": commit}
        tag = f"tr-swesmith-long/{key.lower()}:v1"
        attempts = []
        image_id = None
        for compile_tools in (False, True):
            info = environment.write_context(key, packages, context / "clean", context, project=project_name,
                                             compile_tools=compile_tools)
            log = out / f"build-{'compile' if compile_tools else 'binary'}.log"
            started = time.monotonic()
            try:
                image_id = environment.build(context, tag, log)
                attempts.append({**info, "result": "built", "seconds": round(time.monotonic() - started, 1)})
                break
            except RuntimeError as exc:
                attempts.append({**info, "result": str(exc), "seconds": round(time.monotonic() - started, 1)})
        record["environment_attempts"] = attempts
        shutil.copy2(context / "Dockerfile", out / "Dockerfile")
        shutil.copy2(context / "requirements.txt", out / "requirements.txt")
        if image_id is None:
            record.update(status="HOLD", reason="arm64 environment rebuild failed from official pins")
            return record
        record["environment"] = {"tag": tag, "image_id": image_id, **attempts[-1]}
        generated = context / "generated"
        environment.extract_path(image_id, "/opt/swesmith-long/generated", generated)
        environment.extract_path(image_id, "/opt/swesmith-long", out / "image-opt")
        record["generated_files"] = sorted(file_manifest(generated))
        reference = context / "reference"
        shutil.copytree(context / "clean", reference, symlinks=True)
        shutil.copytree(generated, reference, symlinks=True, dirs_exist_ok=True)
        args, visible = native_pytest_args(family["profile"])
        record["native_test_command"] = visible
        record["pytest_args"] = args
        runs = []
        for index in range(REFERENCE_RUNS):
            grade_dir = out / f"reference-run-{index + 1}"
            spec = {"restore_test_support": False, "protected_dirs": [], "protected_files": [], "config_names": [],
                    "pytest_args": args, "test_files": [], "timeout_seconds": GRADE_TIMEOUT}
            receipt = run_suite(image_id, reference, grade_dir, pristine=None, spec=spec)
            observation = observe(grade_dir)
            statuses = {nid: _simple_status(phases) for nid, phases in observation["reports"].items()}
            write_json(grade_dir / "statuses.json", statuses)
            runs.append({"wall_seconds": receipt["wall_seconds"], "infrastructure_error": receipt["infrastructure_error"],
                         "collect_errors": [e["nodeid"] for e in observation["collect_errors"]],
                         "counts": dict(Counter(statuses.values())), "statuses": statuses})
        record["reference_runs"] = [{k: v for k, v in r.items() if k != "statuses"} for r in runs]
        ids = set().union(*(r["statuses"] for r in runs))
        stable_pass = sorted(i for i in ids if all(r["statuses"].get(i) == "passed" for r in runs))
        flaky = sorted(i for i in ids if len({r["statuses"].get(i) for r in runs}) > 1)
        write_json(out / "reference-stable-pass.json", stable_pass)
        record["reference_stable_pass"] = len(stable_pass)
        record["reference_flaky"] = flaky
        top_levels = sorted({p.name.removesuffix(".py") for p in (context / "clean").glob("*")
                             if (p.is_dir() and (p / "__init__.py").exists()) or (p.suffix == ".py" and p.name != "setup.py")} |
                            {p.name.removesuffix(".py") for p in (context / "clean" / "src").glob("*")
                             if (context / "clean" / "src").is_dir() and ((p.is_dir() and (p / "__init__.py").exists()) or p.suffix == ".py")})
        top_levels = [t for t in top_levels if t not in TEST_DIRS and not t.startswith(("test", "conftest", "noxfile", "docs"))]
        record["project_top_levels"] = top_levels
        record["image_leak_scan"] = image_leak_scan(image_id, context / "clean", top_levels, out)
        record["status"] = "READY" if not record["image_leak_scan"]["leak"] and not any(
            r["infrastructure_error"] for r in runs) else "HOLD"
        if record["status"] == "HOLD":
            record["reason"] = "image leak or reference infrastructure error"
        return record
    except Exception as exc:  # environment/source failures hold the whole family
        record.update(status="HOLD", reason=f"{type(exc).__name__}: {exc}"[:2000])
        return record
    finally:
        write_json(record_path, record)


def _simple_status(phases: dict) -> str:
    setup, call, teardown = phases.get("setup"), phases.get("call"), phases.get("teardown")
    if call is None:
        if setup and setup["outcome"] == "skipped":
            return "xfailed" if setup["wasxfail"] else "skipped"
        return "error"
    if call["outcome"] == "passed":
        return "error" if teardown and teardown["outcome"] == "failed" else "passed"
    if call["outcome"] == "skipped":
        return "xfailed" if call["wasxfail"] else "skipped"
    return "failed"


def protected_set(tree: Path) -> tuple[list[str], list[str]]:
    dirs, files = [], []
    for path in sorted(tree.rglob("*")):
        rel = path.relative_to(tree).as_posix()
        parts = rel.split("/")
        if path.is_dir() and not path.is_symlink() and parts[-1] in TEST_DIRS and not any(p in TEST_DIRS for p in parts[:-1]):
            dirs.append(rel)
    for path in sorted(tree.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        rel = path.relative_to(tree).as_posix()
        inside = any(rel == d or rel.startswith(d + "/") for d in dirs)
        if (inside or path.name == "conftest.py" or (rel.endswith(".py") and is_test_path(rel))
                or ("/" not in rel and path.name in CONFIG_NAMES)):
            files.append(rel)
    return dirs, files


def build_start(row: dict, family: dict, work: Path) -> dict:
    """Clean + generated + bug patch, verified against the mirror bug commit."""
    key = family["repo_key"]
    reference = CACHE / "build" / key / "reference"
    start = work / "start"
    source.copy_tree(reference, start)
    source.apply_patch(start, row["patch"])
    repository = source.mirror(key, CACHE)
    head = source.fetch_branch(repository, row["instance_id"])
    bug_commit = source.git(repository, "rev-parse", head + "~1").decode().strip()
    branch_tree = work / "mirror-bug-commit"
    source.export(repository, bug_commit, branch_tree)
    generated = set(family["generated_files"])
    ours = {k: v for k, v in source.manifest(start).items() if k not in generated}
    theirs = source.manifest(branch_tree)
    removed = source.git(repository, "diff", "--name-status", bug_commit, head).decode().split("\n")
    f2p_files = sorted({x.split("::", 1)[0] for x in row["FAIL_TO_PASS"]})
    head_changes = sorted(line for line in removed if line.strip())
    materialized = materialize_doc_symlinks(start) if ours == theirs else []
    return {"start": start, "mirror_branch_head": head, "mirror_bug_commit": bug_commit,
            "symlinks_materialized": materialized,
            "start_equals_mirror_bug_commit": ours == theirs,
            "differences": sorted(set(ours.items()) ^ set(theirs.items()))[:20],
            "branch_head_changes": head_changes,
            "branch_head_only_removes_f2p_files": head_changes == [f"D\t{f}" for f in f2p_files]}


def materialize_doc_symlinks(tree: Path) -> list[dict]:
    """The shared runner cannot restore W symlinks from this macOS host (host lstat
    mode 0755 vs container 0777 fails its fidelity check). Documentation-only links
    to in-tree regular files are replaced by copies (disclosed adaptation); any
    other symlink holds the instance."""
    links = sorted(p for p in tree.rglob("*") if p.is_symlink())
    result = []
    for link in links:
        rel = link.relative_to(tree).as_posix()
        target = link.resolve()
        if not rel.startswith("docs/") or not target.is_file() or not target.is_relative_to(tree.resolve()):
            raise RuntimeError(f"start tree has {len(links)} symlinks (e.g. {rel}); the shared runner's W restore "
                               "rejects symlinks on this host (mode 0755 vs 0777); non-documentation links are not rewritten")
        result.append({"path": rel, "target": target.relative_to(tree.resolve()).as_posix()})
    for item in result:
        path = tree / item["path"]
        path.unlink()
        shutil.copy2(tree / item["target"], path)
    return result


def package_instance(row: dict, family: dict) -> dict:
    instance_id = row["instance_id"]
    task_id = task_id_for(instance_id)
    out = EVIDENCE / "tasks" / task_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = CACHE / "work" / task_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    record = {"task_id": task_id, "instance_id": instance_id, "repo_key": family["repo_key"],
              "family_id": family["family_id"], "split": family["split"], "status": "running"}
    try:
        built = build_start(row, family, work)
        record["provenance"] = {k: v for k, v in built.items() if k != "start"}
        if not built["start_equals_mirror_bug_commit"]:
            raise RuntimeError("start tree differs from the mirror bug commit")
        start = built["start"]
        family_dir = EVIDENCE / "families" / family["repo_key"]
        stable = set(json.loads((family_dir / "reference-stable-pass.json").read_text()))
        f2p = list(dict.fromkeys(row["FAIL_TO_PASS"]))
        p2p_all = [x for x in dict.fromkeys(row["PASS_TO_PASS"]) if x not in set(f2p)]
        missing_f2p = [x for x in f2p if x not in stable]
        record["f2p_not_stably_passing_at_reference"] = missing_f2p
        if missing_f2p:
            record.update(status="HOLD", reason=f"{len(missing_f2p)} FAIL_TO_PASS tests do not pass stably at the reference in the rebuilt environment")
            return record
        p2p = [x for x in p2p_all if x in stable]
        p2p_excluded = [{"check_id": x, "reason": "not stably passing at reference in rebuilt environment"}
                        for x in p2p_all if x not in stable]
        dirs, files = protected_set(start)
        test_files = sorted({x.split("::", 1)[0] for x in f2p + p2p})
        package = PACKAGES / task_id
        if package.exists():
            shutil.rmtree(package)
        (package / "public").mkdir(parents=True)
        (package / "private").mkdir()
        source.copy_tree(start, package / "public/workspace")
        (package / "public/instructions.md").write_text(row["problem_statement"])
        (package / "private/bug.patch").write_text(row["patch"])
        write_json(package / "private/source-row.json", {k: row[k] for k in (
            "instance_id", "repo", "image_name", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS", "patch")})
        pristine = package / "private/pristine"
        for rel in files:
            target = pristine / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(start / rel, target, follow_symlinks=False)
        write_json(package / "private/checks.json", {"fail_to_pass": f2p, "pass_to_pass": p2p,
                                                     "pass_to_pass_excluded": p2p_excluded})
        write_json(package / "private/grading.json", {"protected_dirs": dirs, "protected_files": files,
            "pytest_args": family["pytest_args"], "test_files": test_files, "timeout_seconds": GRADE_TIMEOUT})
        write_json(package / "private/adaptation.json", ADAPTATION)
        files_changed = patch_files(row["patch"])
        manifest = {"schema_version": "swesmith-long.package.v1", "task_id": task_id, "package_version": task_id,
            "adapter_entry": "trajectory_retention.tasks.swesmith_long.adapter:SWESmithLongTaskAdapter",
            "source_url": f"https://huggingface.co/datasets/{DATASET}", "dataset_revision": DATASET_REVISION,
            "swesmith_source": f"https://github.com/SWE-bench/SWE-smith/tree/{SWESMITH_SOURCE_REVISION}",
            "upstream_instance_id": instance_id, "repo_key": family["repo_key"], "family_id": family["family_id"],
            "planned_split": family["split"], "mirror": {**family["mirror"], **record["provenance"]},
            "official_environment": {"image": family["official_image"], **family["official"]},
            "environment_image_id": family["environment"]["image_id"], "environment_tag": family["environment"]["tag"],
            "environment_build": {k: family["environment"][k] for k in (
                "base_image", "python_official", "python_base", "requirements_sha256", "dockerfile_sha256",
                "build_backend", "editable_mode", "compile_tools", "project_version", "dropped")},
            "generated_files": family["generated_files"],
            "visible_feedback_command": family["native_test_command"] + " (run in /workspace)",
            "bug_shape": {"files": [f["a"] for f in files_changed], "hunks": sum(f["hunks"] for f in files_changed),
                          "strategy": instance_id.rsplit(".", 1)[-1].split("__")[0]},
            "counts": {"fail_to_pass": len(f2p), "pass_to_pass": len(p2p), "pass_to_pass_excluded": len(p2p_excluded),
                       "fail_to_pass_functions": len({x.split('[', 1)[0] for x in f2p})},
            "adaptation": {**ADAPTATION, "symlinks_materialized": built["symlinks_materialized"]},
            "evaluator_sources": evaluator_sources(),
            "public_files": file_manifest(package / "public"), "private_files": file_manifest(package / "private")}
        write_json(package / "package.json", manifest)
        record["package"] = str(package.relative_to(WORKTREE))
        record["package_sha256"] = sha256(package / "package.json")
        record["counts"] = manifest["counts"]
        record["bug_shape"] = manifest["bug_shape"]
        record["validation"] = validate(package, work, row, out)
        record["status"] = "VALIDATED" if record["validation"]["passed"] else "HOLD"
        if record["status"] == "HOLD":
            record["reason"] = "; ".join(record["validation"]["failures"])
        return record
    except Exception as exc:
        record.update(status="HOLD", reason=f"{type(exc).__name__}: {exc}"[:2000])
        return record
    finally:
        write_json(out / "package-record.json", record)


def _grade(adapter: SWESmithLongTaskAdapter, tree: Path, out: Path, label: str) -> dict:
    evaluation = adapter.grade(tree, stage_id="repair", private_output_dir=out / label)
    statuses = {c.check_id: c.status.value for c in evaluation.checks}
    return {"label": label, "valid": evaluation.valid, "score": evaluation.score, "full_solve": evaluation.full_solve,
            "f2p_passed": evaluation.details["fail_to_pass_passed"], "p2p_passed": evaluation.details["pass_to_pass_passed"],
            "p2p_regressions": len(evaluation.details["pass_to_pass_regressions"] or []),
            "wall_seconds": evaluation.details["grade_wall_seconds"],
            "status_counts": {k: v for k, v in evaluation.details["status_counts"].items() if v},
            "statuses_sha256": hashlib.sha256(json.dumps(statuses, sort_keys=True).encode()).hexdigest(),
            "_statuses": statuses}


def _hunk_patches(patch: str) -> list[tuple[str, str]]:
    """Split a unified diff into one self-contained patch per hunk."""
    result, header, hunk, name = [], [], [], None
    def flush():
        if hunk:
            result.append((f"{name}#{len([r for r in result if r[0].startswith(name + '#')]) + 1}",
                           "\n".join(header + hunk) + "\n"))
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            flush(); hunk = []
            header, name = [line], line.split(" b/", 1)[1]
        elif line.startswith("@@"):
            flush(); hunk = [line]
        elif hunk:
            hunk.append(line)
        else:
            header.append(line)
    flush()
    return result


def validate(package: Path, work: Path, row: dict, out: Path, adapter_cls=SWESmithLongTaskAdapter) -> dict:
    adapter = adapter_cls(package, scratch_root=CACHE / "grade-scratch")
    failures = []
    natural = work / "natural"
    adapter.materialize(natural)
    natural_equal = file_manifest(natural) == file_manifest(package / "public/workspace")
    grades = {}
    for label in ("start-1", "start-2"):
        grades[label] = _grade(adapter, natural, out, label)
    reference = work / "reference"
    source.copy_tree(natural, reference)
    subprocess.run(["git", "apply", "-R", "-"], cwd=reference, input=row["patch"].encode(), check=True,
                   capture_output=True, env=source._env(reference))
    for label in ("reference-1", "reference-2"):
        grades[label] = _grade(adapter, reference, out, label)
    partial = []
    chosen = None
    hunks = _hunk_patches(row["patch"])
    # Scripted partial controls: reverse one bug hunk at a time, then (if none
    # earns strictly partial credit) all hunks except one. Grader controls only.
    attempts = [(name, [hunk]) for name, hunk in hunks]
    if len(hunks) > 2:
        attempts += [("all-but-" + name, [h for n, h in hunks if n != name]) for name, _ in hunks]
    for name, selected in attempts:
        label = "partial-" + re.sub(r"[^A-Za-z0-9]+", "-", name)
        tree = work / label
        source.copy_tree(natural, tree)
        applied = all(subprocess.run(["git", "apply", "-R", "-"], cwd=tree, input=h.encode(), capture_output=True,
                                     env=source._env(tree)).returncode == 0 for h in selected)
        if not applied:
            partial.append({"hunk": name, "applied": False})
            continue
        result = _grade(adapter, tree, out, label)
        partial.append({"hunk": name, "applied": True, **{k: v for k, v in result.items() if k != "_statuses"}})
        if result["valid"] and 0 < result["score"] < 1:
            chosen = name
            break
    tamper = work / "tamper"
    source.copy_tree(natural, tamper)
    for rel in adapter.grading["protected_dirs"]:
        shutil.rmtree(tamper / rel, ignore_errors=True)
    protected = set(adapter.grading["protected_files"])
    # Only genuine test-support files are tampered (doctest F2P ids live in source modules).
    for rel in {x.split("::", 1)[0] for x in adapter.scorer.f2p} & protected:
        path = tamper / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_tampered():\n    assert True\n")
    (tamper / "conftest.py").write_text("import pytest\n\n@pytest.hookimpl(tryfirst=True)\ndef pytest_runtest_call(item):\n    pass\n")
    grades["tamper-tests"] = _grade(adapter, tamper, out, "tamper-tests")
    starts = [grades["start-1"], grades["start-2"]]
    refs = [grades["reference-1"], grades["reference-2"]]
    checks = {
        "natural_start_equals_package": natural_equal,
        "no_git_directory_in_start": not (natural / ".git").exists(),
        "start_valid": all(g["valid"] for g in starts),
        "start_no_f2p_passing": all(g["f2p_passed"] == 0 for g in starts),
        "start_deterministic": starts[0]["statuses_sha256"] == starts[1]["statuses_sha256"],
        "reference_valid": all(g["valid"] for g in refs),
        "reference_full_solve": all(g["full_solve"] is True and g["score"] == 1.0 for g in refs),
        "reference_deterministic": refs[0]["statuses_sha256"] == refs[1]["statuses_sha256"],
        "partial_control_strictly_between": chosen is not None,
        "tamper_control_equals_start": grades["tamper-tests"]["statuses_sha256"] == starts[0]["statuses_sha256"],
    }
    leak = start_leak_scan(natural, row["patch"], work)
    checks["start_has_no_reference_copy"] = not leak["leak"]
    failures = [k for k, v in checks.items() if not v]
    times = [g["wall_seconds"] for g in grades.values() if g.get("wall_seconds") is not None] + [
        p["wall_seconds"] for p in partial if p.get("wall_seconds") is not None]
    start_statuses = starts[0]["_statuses"]
    summary = {"passed": not failures, "failures": failures, "checks": checks,
               "grades": {k: {x: y for x, y in v.items() if x != "_statuses"} for k, v in grades.items()},
               "partial_controls": partial, "partial_control_hunk": chosen, "start_leak_scan": leak,
               "start_f2p_status_counts": dict(Counter(start_statuses[c] for c in adapter.scorer.f2p)),
               "start_p2p_status_counts": dict(Counter(start_statuses[c] for c in adapter.scorer.p2p)),
               "grade_seconds": {"max": max(times), "median": sorted(times)[len(times) // 2]},
               "grade_time_flag_over_120s": max(times) > 120,
               "evaluator_version": adapter.scorer.version}
    write_json(out / "validation.json", summary)
    return {k: v for k, v in summary.items() if k not in ("grades", "partial_controls")} | {
        "grades": {k: {x: y for x, y in v.items() if x in ("valid", "score", "full_solve", "wall_seconds", "status_counts")}
                   for k, v in summary["grades"].items()},
        "partial_controls": [{k: p.get(k) for k in ("hunk", "applied", "score", "valid")} for p in partial]}


def start_leak_scan(start: Path, patch: str, work: Path) -> dict:
    clean = work / "clean-mutated"
    clean.mkdir(exist_ok=True)
    mutated = [f["a"] for f in patch_files(patch)]
    reference_hashes = {}
    for rel in mutated:
        copy = clean / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(start / rel, copy)
    subprocess.run(["git", "apply", "-R", "-"], cwd=clean, input=patch.encode(), check=True, capture_output=True,
                   env=source._env(clean))
    for rel in mutated:
        reference_hashes[sha256(clean / rel)] = rel
    copies = [{"path": p.relative_to(start).as_posix(), "reference_of": reference_hashes[sha256(p)]}
              for p in start.rglob("*") if p.is_file() and not p.is_symlink() and sha256(p) in reference_hashes]
    git_metadata = [p.relative_to(start).as_posix() for p in start.rglob(".git")]
    return {"reference_file_copies": copies, "git_metadata": git_metadata,
            "leak": bool(copies or git_metadata),
            "rule": "no file in W identical to the reference version of a mutated file; no .git "
                    "(the rest of W equals the verified mirror bug commit plus install-generated files)"}
