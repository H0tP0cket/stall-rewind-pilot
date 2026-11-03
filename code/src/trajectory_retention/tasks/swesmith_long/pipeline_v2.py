"""Build and validate v2 (native SWE-smith start) successors of the sealed v1 tasks.

SWE-smith (pinned source 9b74ac08, swesmith/harness/gather.py) commits each
instance as two commits on its mirror branch: "Bug Patch", then "Remove F2P
Tests", which deletes every file holding a FAIL_TO_PASS test
(profiles/python.py:get_test_files = unique file parts of the FAIL_TO_PASS ids).
The agent works on the branch head; evaluation checks out HEAD~1 (tests back),
applies the agent patch and reverts changes to FAIL_TO_PASS/PASS_TO_PASS files
(harness/utils.py). v2 reproduces the worker-visible part exactly: W = files of
the mirror branch head (+ the install-generated files v1 already disclosed).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import pipeline, source
from .adapter import file_manifest, sha256
from .adapter_v2 import ADAPTER_ENTRY, SCHEMA, SWESmithLongV2TaskAdapter, evaluator_sources
from .pipeline import CACHE, EVIDENCE, WORKTREE, write_json

PACKAGES_V1 = WORKTREE / "benchmark_tasks" / "packages" / "swesmith-long-v1"
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / "swesmith-long-v2"
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / "swesmith-long-v2"
EVIDENCE_V2 = EVIDENCE / "v2"
ADAPTATION = {
    "version": "swesmith-long.adaptation.v2",
    "predecessor": "swesmith-long.adaptation.v1",
    "changes": [
        "Worker start is the SWE-smith mirror branch head: every FAIL_TO_PASS test file deleted by the native "
        "'Remove F2P Tests' commit is withheld from W (v1 adaptation 1 removed). PASS_TO_PASS tests in other "
        "files stay visible, as natively; PASS_TO_PASS tests inside withheld files are withheld with them.",
        "No Git history in W (stricter than native: the native branch history contains the clean reference and "
        "the withheld tests).",
        "Environment rebuilt natively for linux/arm64 from the official x86_64 image's exact package pins "
        "(unchanged from v1; same images).",
        "Network disabled for worker and grader.",
        "Grading restores the frozen test-support set (which includes the withheld files) before running pytest "
        "with --continue-on-collection-errors on every tracked FAIL_TO_PASS and PASS_TO_PASS test; only W is graded.",
        "PASS_TO_PASS tests that did not pass in every reference run in the rebuilt environment stay excluded (as v1).",
        "Documentation-only symlinks replaced by copies (as v1; python-pinyin only).",
    ],
    "unchanged": ["problem_statement text (worker instruction) byte-identical to v1 and the dataset",
                  "bug patch, FAIL_TO_PASS/PASS_TO_PASS identities, score rule, grader mechanism, images"],
}


def v2_id(v1_id: str) -> str:
    if not v1_id.endswith("-v1"):
        raise ValueError(v1_id)
    return v1_id[:-3] + "-v2"


def distinctive_lines(withheld: dict[str, bytes], others: list[Path], limit: int = 400) -> list[str]:
    """Stripped lines (>= 25 chars) of withheld files that occur in no other file of the full tree."""
    candidates = set()
    for data in withheld.values():
        candidates |= {line.strip() for line in data.decode("utf-8", "ignore").splitlines() if len(line.strip()) >= 25}
    for path in others:
        if not candidates:
            break
        try:
            text = path.read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        candidates -= {line.strip() for line in text.splitlines()}
    return sorted(candidates)[:limit]


def build(v1_id: str) -> dict:
    task_id = v2_id(v1_id)
    out = EVIDENCE_V2 / "tasks" / task_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = CACHE / "work-v2" / task_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    old = PACKAGES_V1 / v1_id
    old_manifest = json.loads((old / "package.json").read_text())
    row = json.loads((old / "private/source-row.json").read_text())
    grading = json.loads((old / "private/grading.json").read_text())
    key = old_manifest["repo_key"]
    record = {"task_id": task_id, "predecessor": v1_id, "instance_id": row["instance_id"], "repo_key": key,
              "family_id": old_manifest["family_id"], "split": old_manifest["planned_split"], "status": "running"}
    try:
        repository = source.mirror(key, CACHE)
        head = source.fetch_branch(repository, row["instance_id"])
        bug_commit = source.git(repository, "rev-parse", head + "~1").decode().strip()
        changes = [line.split("\t") for line in source.git(repository, "diff", "--name-status", bug_commit, head)
                   .decode().splitlines() if line.strip()]
        f2p_files = sorted({x.split("::", 1)[0] for x in row["FAIL_TO_PASS"]})
        withheld = sorted(path for status, path in changes)
        record["native_removal"] = {"branch_head": head, "bug_commit": bug_commit, "changes": changes,
                                    "f2p_files": f2p_files}
        if any(status != "D" for status, _ in changes) or withheld != f2p_files:
            raise RuntimeError("mirror branch head is not exactly the removal of the FAIL_TO_PASS files")
        not_test_support = [p for p in withheld if p not in set(grading["protected_files"])]
        if not_test_support:
            record.update(status="HOLD", reason=(
                "native 'Remove F2P Tests' deletes non-test source module(s) " + ", ".join(not_test_support) +
                " (FAIL_TO_PASS doctest ids); the native start is not importable and native evaluation reverts "
                "that module to its bug-commit version, so a faithful native grade is not defined here"))
            return record
        # Worker start: mirror branch head + v1's install-generated files (+ doc-symlink copies).
        head_tree = work / "mirror-branch-head"
        source.export(repository, head, head_tree)
        pipeline.materialize_doc_symlinks(head_tree)
        start = work / "start"
        source.copy_tree(old / "public/workspace", start)
        for path in withheld:
            (start / path).unlink()
        generated = set(old_manifest["generated_files"])
        ours = {k: v for k, v in source.manifest(start).items() if k not in generated}
        theirs = source.manifest(head_tree)
        record["start_equals_mirror_branch_head"] = ours == theirs
        if ours != theirs:
            raise RuntimeError("v2 start differs from the mirror branch head: "
                               + str(sorted(set(ours.items()) ^ set(theirs.items()))[:5]))
        withheld_bytes = {p: (old / "public/workspace" / p).read_bytes() for p in withheld}
        for p, data in withheld_bytes.items():
            if (old / "private/pristine" / p).read_bytes() != data:
                raise RuntimeError("private test-support copy differs from the withheld file: " + p)
        package = PACKAGES / task_id
        if package.exists():
            shutil.rmtree(package)
        package.mkdir(parents=True)
        shutil.copytree(old / "private", package / "private", symlinks=True)
        (package / "public").mkdir()
        source.copy_tree(start, package / "public/workspace")
        shutil.copy2(old / "public/instructions.md", package / "public/instructions.md")
        if (package / "public/instructions.md").read_bytes() != row["problem_statement"].encode():
            raise RuntimeError("problem statement is not byte-identical")
        write_json(package / "private/withheld.json", {
            "rule": "files deleted by the mirror branch head commit ('Remove F2P Tests'); = unique file parts of FAIL_TO_PASS ids",
            "files": [{"path": p, "sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)} for p, d in withheld_bytes.items()]})
        write_json(package / "private/adaptation.json", ADAPTATION)
        p2p_hidden = sorted(c for c in json.loads((old / "private/checks.json").read_text())["pass_to_pass"]
                            if c.split("::", 1)[0] in set(withheld))
        manifest = {**{k: v for k, v in old_manifest.items() if k not in ("public_files", "private_files")},
                    "schema_version": SCHEMA, "task_id": task_id, "package_version": task_id,
                    "adapter_entry": ADAPTER_ENTRY, "adaptation": ADAPTATION, "evaluator_sources": evaluator_sources(),
                    "predecessor": {"task_id": v1_id, "package_sha256": sha256(old / "package.json")},
                    "native_start": {"mirror_branch_head": head, "withheld_files": withheld,
                                     "pass_to_pass_in_withheld_files": len(p2p_hidden)},
                    "public_files": file_manifest(package / "public"), "private_files": file_manifest(package / "private")}
        write_json(package / "package.json", manifest)
        record["package"] = str(package.relative_to(WORKTREE))
        record["package_sha256"] = sha256(package / "package.json")
        record["counts"] = {**old_manifest["counts"], "withheld_files": len(withheld),
                            "pass_to_pass_withheld_with_files": len(p2p_hidden)}
        record["bug_shape"] = old_manifest["bug_shape"]
        others = [p for p in (old / "public/workspace").rglob("*") if p.is_file() and not p.is_symlink()
                  and p.relative_to(old / "public/workspace").as_posix() not in set(withheld)]
        needles = distinctive_lines(withheld_bytes, others)
        write_json(out / "needles.json", needles)
        record["validation"] = pipeline.validate(package, work, row, out, adapter_cls=SWESmithLongV2TaskAdapter)
        leak = withheld_leak_scan(package, work / "natural", withheld_bytes, needles, key)
        record["withheld_leak_scan"] = leak
        record["validation"]["checks"]["withheld_tests_absent_from_workspace_and_image"] = not leak["leak"]
        record["validation"]["failures"] = [k for k, v in record["validation"]["checks"].items() if not v]
        record["validation"]["passed"] = not record["validation"]["failures"]
        record["status"] = "VALIDATED" if record["validation"]["passed"] else "HOLD"
        if record["status"] == "HOLD":
            record["reason"] = "; ".join(record["validation"]["failures"])
        return record
    except Exception as exc:
        record.update(status="HOLD", reason=f"{type(exc).__name__}: {exc}"[:2000])
        return record
    finally:
        write_json(out / "package-record.json", record)


def withheld_leak_scan(package: Path, natural: Path, withheld: dict[str, bytes], needles: list[str], key: str) -> dict:
    hashes = {hashlib.sha256(d).hexdigest(): p for p, d in withheld.items()}
    present = [p for p in withheld if (natural / p).exists() or (natural / p).is_symlink()]
    copies = [{"path": f.relative_to(natural).as_posix(), "withheld": hashes[sha256(f)]}
              for f in natural.rglob("*") if f.is_file() and not f.is_symlink() and sha256(f) in hashes]
    wanted = set(needles)
    line_hits = []
    for f in natural.rglob("*"):
        if f.is_file() and not f.is_symlink():
            found = {line.strip() for line in f.read_bytes().decode("utf-8", "ignore").splitlines()} & wanted
            if found:
                line_hits.append({"path": f.relative_to(natural).as_posix(), "lines": sorted(found)[:3]})
    inventory = json.loads((EVIDENCE / "families" / key / "image-inventory.json").read_text())
    image_copies = [{"image_path": path, "withheld": hashes[d]} for path, _, d in inventory if d in hashes]
    image_needles = json.loads((EVIDENCE_V2 / "families" / key / "image-needles.json").read_text())
    per_file = json.loads((EVIDENCE_V2 / "families" / key / "needles.json").read_text())
    # Generic lines of a withheld test (imports, dunder signatures) also occur in third-party
    # packages (e.g. jinja2 imports markupsafe). An image file counts as a copy of a withheld
    # file when it contains >= max(3, 20%) of that file's distinctive lines (all, if fewer than 3).
    task_hits, image_line_hits = [], []
    for hit in image_needles["hits"]:
        task, _, path = hit["owner"].partition("::")
        if task != package.name:
            continue
        total = len(per_file[hit["owner"]])
        threshold = total if total < 3 else max(3, -(-total // 5))
        hit = {**hit, "withheld_file": path, "file_needles": total, "threshold": threshold}
        task_hits.append(hit)
        if hit["count"] >= threshold:
            image_line_hits.append(hit)
    return {"withheld_paths_present_in_W": present, "identical_copies_in_W": copies,
            "distinctive_lines_checked": len(needles), "distinctive_line_hits_in_W": line_hits,
            "identical_copies_in_image": image_copies, "distinctive_line_hits_in_image": image_line_hits,
            "image_matches_below_threshold": sorted((h for h in task_hits if h["count"] < h["threshold"]),
                                                    key=lambda h: -h["count"])[:10],
            "image_max_fraction_of_a_withheld_file_matched": max((h["count"] / h["file_needles"] for h in task_hits), default=0.0),
            "image_files_scanned": image_needles["files_scanned"],
            "leak": bool(present or copies or line_hits or image_copies or image_line_hits),
            "rule": "withheld paths absent from W; no W or image file byte-identical to a withheld file; no W file "
                    "containing any distinctive (>=25 char, not elsewhere in the full tree) line of a withheld file; "
                    "no image file containing >= max(3, 20%) of one withheld file's distinctive lines (all, if fewer than 3)"}


def family_image_needles(key: str, v1_ids: list[str]) -> dict:
    """One container per family: search the worker image for withheld-test lines of all its tasks."""
    needles = {}
    for v1_id in v1_ids:
        old = PACKAGES_V1 / v1_id
        row = json.loads((old / "private/source-row.json").read_text())
        withheld = sorted({x.split("::", 1)[0] for x in row["FAIL_TO_PASS"]})
        data = {p: (old / "public/workspace" / p).read_bytes() for p in withheld if (old / "public/workspace" / p).is_file()}
        others = [p for p in (old / "public/workspace").rglob("*") if p.is_file() and not p.is_symlink()
                  and p.relative_to(old / "public/workspace").as_posix() not in set(withheld)]
        for path, content in data.items():
            needles[v2_id(v1_id) + "::" + path] = distinctive_lines({path: content}, others)
    out = EVIDENCE_V2 / "families" / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "needles.json").write_text(json.dumps(needles, indent=1) + "\n")
    image = json.loads((PACKAGES_V1 / v1_ids[0] / "package.json").read_text())["environment_image_id"]
    here = Path(__file__).with_name("build_support")
    name = "tr-swesmith-long-needles-" + hashlib.sha256(key.encode()).hexdigest()[:10]
    result = subprocess.run(["docker", "run", "--rm", "--name", name, "--label", "trajectory-retention.lane=swesmith-long",
                             "--network", "none", "--entrypoint", "python3",
                             "--mount", f"type=bind,src={here},dst=/scan,readonly",
                             "--mount", f"type=bind,src={out},dst=/needles,readonly",
                             image, "-I", "/scan/scan_needles.py", "/needles/needles.json"],
                            check=True, capture_output=True, timeout=900)
    report = json.loads(result.stdout)
    report["image_id"] = image
    write_json(out / "image-needles.json", report)
    return report


def rescan(task_id: str) -> dict:
    """Recompute the withheld-test leak scan for an already validated v2 task (rule revision only)."""
    out = EVIDENCE_V2 / "tasks" / task_id
    record = json.loads((out / "package-record.json").read_text())
    if "validation" not in record:
        return record
    package = PACKAGES / task_id
    old = PACKAGES_V1 / record["predecessor"]
    withheld = json.loads((package / "private/withheld.json").read_text())["files"]
    withheld_bytes = {r["path"]: (package / "private/pristine" / r["path"]).read_bytes() for r in withheld}
    per_file = json.loads((EVIDENCE_V2 / "families" / record["repo_key"] / "needles.json").read_text())
    needles = sorted({line for r in withheld for line in per_file[task_id + "::" + r["path"]]})
    natural = CACHE / "work-v2" / task_id / "natural"
    leak = withheld_leak_scan(package, natural, withheld_bytes, needles, record["repo_key"])
    record["withheld_leak_scan"] = leak
    checks = record["validation"]["checks"]
    checks["withheld_tests_absent_from_workspace_and_image"] = not leak["leak"]
    record["validation"]["failures"] = [k for k, v in checks.items() if not v]
    record["validation"]["passed"] = not record["validation"]["failures"]
    record["leak_rule_revision"] = "per-withheld-file threshold (v2 rule 2); grades unchanged"
    if record["status"] in ("VALIDATED", "HOLD") and record.get("reason", "") in (None, "", "withheld_tests_absent_from_workspace_and_image"):
        record["status"] = "VALIDATED" if record["validation"]["passed"] else "HOLD"
        record["reason"] = None if record["validation"]["passed"] else "; ".join(record["validation"]["failures"])
    write_json(out / "package-record.json", record)
    return record
