"""v4: new calibration families built by the unchanged v1 -> v2 -> v3 pipeline (no model/provider calls).

Per family (selection_v4 order), exactly the steps that produced the v3 tasks:

1. v1 family environment (pipeline.prepare_family): official pins, native arm64
   image, three reference runs of the whole suite, clean-source image scan.
2. Per instance, v1 package + validation (pipeline.package_instance), v2 native
   start with the FAIL_TO_PASS files withheld (pipeline_v2.family_image_needles,
   pipeline_v2.build), then the v3 format: git-enabled family image
   (pipeline_v3.build_family_image, pipeline_v3.family_scans), one-commit
   ``workspace-git.tar`` (pipeline_v3.build_git_start), the v3 gates
   (pipeline_v3.validate, pipeline_v3.withheld_leak_scan, pipeline_v3.compare_with_v2),
   the G1 grade-time gate, and the MiningRunner fixture (runner_check).

The v1 and v2 packages are intermediate build stages: they are written to a
gitignored staging directory (``artifacts/task-admission/swesmith-long/v4/stage``),
never to the sealed v1/v2 package trees. Final packages are v3-format packages in
``benchmark_tasks/packages/swesmith-long-v4/<task>-v4`` loaded by adapter_v4.
Raw evidence (gitignored) goes where the v1/v2/v3 steps always write it, keyed by
the new task ids, plus ``artifacts/task-admission/swesmith-long/v4/``.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

from . import pipeline, pipeline_v2, pipeline_v3, source
from .adapter import file_manifest, sha256
from .adapter_v3 import GIT_TAR, SCHEMA
from .adapter_v4 import ADAPTER_ENTRY, LANE, SWESmithLongV4TaskAdapter
from .pipeline import CACHE, EVIDENCE, WORKTREE, write_json
from .selection_v4 import CRITERIA_V4, GRADE_SECONDS_MAX

STAGE = EVIDENCE / "v4" / "stage"  # gitignored, inside the worktree (records use worktree-relative paths)
PACKAGES_V1_STAGE = STAGE / "packages" / "swesmith-long-v1"
PACKAGES_V2_STAGE = STAGE / "packages" / "swesmith-long-v2"
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / LANE
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / LANE
EVIDENCE_V4 = EVIDENCE / "v4"
WORK = CACHE / "work-v4"
SELECTION = EVIDENCE_V4 / "selection.json"
ADAPTATION_NOTE = ("v3 package format and adaptation record (swesmith-long.adaptation.v3) unchanged; first issued in the "
                   "v4 lane release; intermediate v1/v2 build stages are not published")


def v4_id(v2_id: str) -> str:
    if not v2_id.endswith("-v2"):
        raise ValueError(v2_id)
    return v2_id[:-3] + "-v4"


@contextmanager
def staged(key: str, v2_ids: list[str]):
    """Point the unchanged v1/v2/v3 steps at the staging package trees for this family."""
    saved = (pipeline.PACKAGES, pipeline_v2.PACKAGES_V1, pipeline_v2.PACKAGES, pipeline_v3.PACKAGES_V2,
             pipeline_v3.family_packages)
    pipeline.PACKAGES = PACKAGES_V1_STAGE
    pipeline_v2.PACKAGES_V1 = PACKAGES_V1_STAGE
    pipeline_v2.PACKAGES = PACKAGES_V2_STAGE
    pipeline_v3.PACKAGES_V2 = PACKAGES_V2_STAGE
    pipeline_v3.family_packages = lambda k: list(v2_ids) if k == key else saved[4](k)
    try:
        yield
    finally:
        (pipeline.PACKAGES, pipeline_v2.PACKAGES_V1, pipeline_v2.PACKAGES, pipeline_v3.PACKAGES_V2,
         pipeline_v3.family_packages) = saved


def rows_for(ids: list[str]) -> dict[str, dict]:
    import pandas as pd
    frame = pd.read_pickle(CACHE / "combine.pkl")
    frame = frame[frame.instance_id.isin(ids)]
    return {r["instance_id"]: {k: (list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v) for k, v in r.items()}
            for r in frame.to_dict("records")}


# --------------------------------------------------------------------------- final (v3-format) build
def build_v4(v2_id: str, family: dict, selection_sha256: str) -> dict:
    """pipeline_v3.build with the v4 identity, staging predecessor and the G1 gate."""
    task_id = v4_id(v2_id)
    out = EVIDENCE_V4 / "tasks" / task_id
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    work = WORK / task_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    record = {"task_id": task_id, "predecessor_stage": v2_id, "status": "running"}
    try:
        old = PACKAGES_V2_STAGE / v2_id
        old_manifest = json.loads((old / "package.json").read_text())
        row = json.loads((old / "private/source-row.json").read_text())
        record.update(instance_id=row["instance_id"], repo_key=old_manifest["repo_key"],
                      family_id=old_manifest["family_id"], split=old_manifest["planned_split"])
        from .adapter_v2 import SWESmithLongV2TaskAdapter
        SWESmithLongV2TaskAdapter(old, scratch_root=CACHE / "grade-scratch")  # staged v2 package verifies
        package = PACKAGES / task_id
        shutil.rmtree(package, ignore_errors=True)
        package.mkdir(parents=True)
        shutil.copytree(old / "private", package / "private", symlinks=True)
        (package / "public").mkdir()
        source.copy_tree(old / "public/workspace", package / "public/workspace")
        shutil.copy2(old / "public/instructions.md", package / "public/instructions.md")
        if (package / "public/instructions.md").read_bytes() != row["problem_statement"].encode():
            raise RuntimeError("problem statement is not byte-identical to the dataset")
        write_json(package / "private/adaptation.json", pipeline_v3.ADAPTATION)
        materialized = (old_manifest["adaptation"].get("symlinks_materialized")
                        or old_manifest["mirror"].get("symlinks_materialized") or [])
        git_start = pipeline_v3.build_git_start(package, family["image_id"], work,
                                                old_manifest["native_start"]["mirror_branch_head"],
                                                old_manifest["repo_key"], materialized)
        shutil.copy2(work / "workspace-git.tar", package / GIT_TAR)
        record["git_start"] = {k: v for k, v in git_start.items() if k != "blobs"}
        if not git_start["native_tree_differences_explained"]:
            raise RuntimeError("start commit tree differs from the native branch head beyond disclosed symlink copies: "
                               + str(git_start["native_tree_differences"][:5]))
        from ..prompts import SWESMITH_INSTANCE_TEMPLATE_SHA256, swesmith_prompt_v3
        from trajectory_retention.contracts import StageSpec
        text = (package / "public/instructions.md").read_text()
        prompt = swesmith_prompt_v3(StageSpec(stage_id="repair", target_id=task_id + ":repair", ordinal=1,
                                              instructions=text, requirement_version="", evaluator_version=""))
        manifest = {**{k: v for k, v in old_manifest.items() if k not in ("public_files", "private_files")},
                    "schema_version": SCHEMA, "task_id": task_id, "package_version": task_id,
                    "adapter_entry": ADAPTER_ENTRY,
                    "adaptation": {**pipeline_v3.ADAPTATION, "symlinks_materialized": materialized},
                    "evaluator_sources": pipeline_v3.evaluator_sources(),
                    "lane_release": {"lane": LANE, "note": ADAPTATION_NOTE,
                                     "selection": {"criteria": CRITERIA_V4["version"], "selection_sha256": selection_sha256}},
                    "predecessor": {"task_id": v2_id, "package_sha256": sha256(old / "package.json"),
                                    "published": False, "stage": "intermediate v2 build stage (staging, not published)"},
                    "environment_image_id": family["image_id"], "environment_tag": family["tag"],
                    "environment_git": {k: family[k] for k in ("base_image_id", "base_image_tag", "dockerfile_sha256",
                                                               "git_version", "system_git_config", "dpkg_added")},
                    "git_start": git_start,
                    "worker_prompt": {"version": prompt.version, "rendered_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                                      "template_sha256": SWESMITH_INSTANCE_TEMPLATE_SHA256, "source": prompt.source},
                    "public_files": file_manifest(package / "public"), "private_files": file_manifest(package / "private")}
        write_json(package / "package.json", manifest)
        record["package"] = str(package.relative_to(WORKTREE))
        record["package_sha256"] = sha256(package / "package.json")
        record["counts"] = old_manifest["counts"]
        record["bug_shape"] = old_manifest["bug_shape"]
        record["native_start"] = old_manifest["native_start"]
        SWESmithLongV4TaskAdapter(package, scratch_root=CACHE / "grade-scratch")  # the v4 loader accepts it
        record["validation"] = pipeline_v3.validate(package, work, row, out)
        withheld = json.loads((package / "private/withheld.json").read_text())["files"]
        withheld_bytes = {r["path"]: (package / "private/pristine" / r["path"]).read_bytes() for r in withheld}
        per_file = json.loads((pipeline_v3.EVIDENCE_V3 / "families" / record["repo_key"] / "needles.json").read_text())
        needles = sorted({line for r in withheld for line in per_file[v2_id + "::" + r["path"]]})
        leak = pipeline_v3.withheld_leak_scan(package, work / "natural", withheld_bytes, needles, record["repo_key"])
        record["withheld_leak_scan"] = leak
        record["v2_comparison"] = pipeline_v3.compare_with_v2(record["validation"], v2_id)
        checks = record["validation"]["checks"]
        checks["withheld_tests_absent_from_workspace_git_and_image"] = not leak["leak"]
        checks["grades_equal_v2"] = record["v2_comparison"]["all_equal"]
        checks["G1_every_grade_within_60s"] = record["validation"]["grade_seconds"]["max"] <= GRADE_SECONDS_MAX
        record["validation"]["failures"] = [k for k, v in checks.items() if not v]
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


# --------------------------------------------------------------------------- family orchestration
def process_family(repo_key: str) -> list[dict]:
    from .runner_check import run_check
    selection = json.loads(SELECTION.read_text())
    selection_sha = sha256(SELECTION)
    family = next(f for f in selection["families"] if f["repo_key"] == repo_key)
    if family["split"] != "calibration" or family["v4_status"] != "DRAWN":
        raise ValueError(f"{repo_key} is not a drawn calibration family ({family['split']}, {family['v4_status']})")
    ledger_path = EVIDENCE_V4 / "candidates" / f"{repo_key}.json"
    ledger: list[dict] = []

    def save():
        write_json(ledger_path, ledger)

    env = pipeline.prepare_family({k: family[k] for k in ("repo_key", "family_id", "split", "profile")})
    fam_summary = {k: env.get(k) for k in ("status", "reason", "reference_stable_pass", "reference_flaky")}
    fam_summary["reference_wall_seconds"] = [r.get("wall_seconds") for r in env.get("reference_runs") or []]
    write_json(EVIDENCE_V4 / "families" / repo_key / "environment.json", fam_summary)
    if env["status"] != "READY":
        for iid in family["candidates"]:
            ledger.append({"instance_id": iid, "role": "candidate", "status": "HOLD", "stage": "family-environment",
                           "reason": "family environment: " + str(env.get("reason"))})
        for iid in family["reserves"]:
            ledger.append({"instance_id": iid, "role": "reserve", "status": "NOT_USED",
                           "reason": "family environment held"})
        save()
        return ledger
    env["profile"] = family["profile"]
    rows = rows_for(family["candidates"] + family["reserves"])
    reserves = list(family["reserves"])
    queue = [(iid, "candidate") for iid in family["candidates"]]
    v1_ok: list[str] = []
    v2_ok: list[str] = []
    image = None
    while queue:
        iid, role = queue.pop(0)
        row = rows[iid]
        entry = {"instance_id": iid, "role": role}
        ledger.append(entry)
        failed = None
        with staged(repo_key, v2_ok):
            v1 = pipeline.package_instance(row, env)
            entry.update(v1_task=v1["task_id"], v1_status=v1["status"], counts=v1.get("counts"),
                         bug_shape=v1.get("bug_shape"),
                         v1_grade_seconds=(v1.get("validation") or {}).get("grade_seconds"))
            if v1["status"] != "VALIDATED":
                failed = ("v1-package", v1.get("reason"))
            if failed is None:
                v1_ok.append(v1["task_id"])
                pipeline_v2.family_image_needles(repo_key, v1_ok)
                v2 = pipeline_v2.build(v1["task_id"])
                entry.update(v2_task=v2["task_id"], v2_status=v2["status"])
                if v2["status"] != "VALIDATED":
                    failed = ("v2-native-start", v2.get("reason"))
            if failed is None:
                v2_ok.append(v2["task_id"])
        if failed is None:
            with staged(repo_key, v2_ok):
                image_path = pipeline_v3.EVIDENCE_V3 / "families" / repo_key / "image.json"
                if image is None:
                    image = pipeline_v3.build_family_image(repo_key)
                    if not image["image_ok"]:
                        write_json(image_path, image)
                        entry.update(status="HOLD", stage="family-git-image", reason="v3 image check failed")
                        for later, later_role in queue:
                            ledger.append({"instance_id": later, "role": later_role, "status": "HOLD",
                                           "stage": "family-git-image", "reason": "family git image check failed"})
                        for later in reserves:
                            ledger.append({"instance_id": later, "role": "reserve", "status": "NOT_USED",
                                           "reason": "family git image held"})
                        save()
                        return ledger
                # Needles grow with every staged v2 task: rescan the git image each time.
                image["scans"] = pipeline_v3.family_scans(repo_key, image["image_id"])
                write_json(image_path, image)
            final = build_v4(v2["task_id"], image, selection_sha)
            entry.update(task_id=final["task_id"], v4_status=final["status"],
                         grade_seconds=(final.get("validation") or {}).get("grade_seconds"),
                         grades_equal_v2=(final.get("v2_comparison") or {}).get("all_equal"),
                         partial_control_hunk=(final.get("validation") or {}).get("partial_control_hunk"))
            if final["status"] != "VALIDATED":
                failed = ("v4-format-gates", final.get("reason"))
            else:
                summary = run_check(PACKAGES / final["task_id"], EVIDENCE_V4 / "tasks" / final["task_id"] / "runner-check",
                                    adapter_cls=SWESmithLongV4TaskAdapter)
                entry["runner_check"] = "passed" if summary.get("passed") else "failed"
                if not summary.get("passed"):
                    failed = ("runner-check", summary.get("error") or ", ".join(
                        k for k, v in summary.get("assertions", {}).items() if not v))
        if failed is None:
            entry.update(status="VALIDATED")
        else:
            entry.update(status="HOLD", stage=failed[0], reason=str(failed[1])[:1500])
            if reserves:
                queue.append((reserves.pop(0), "reserve"))
        save()
        print(json.dumps({k: entry.get(k) for k in ("instance_id", "role", "status", "stage", "reason", "task_id")}),
              flush=True)
    for iid in reserves:
        ledger.append({"instance_id": iid, "role": "reserve", "status": "NOT_USED"})
    save()
    return ledger


# --------------------------------------------------------------------------- selection record
def write_selection(last_position: int) -> dict:
    from .selection_v4 import FIRST_POSITION, OFFICIAL_IMAGE_MAX_BYTES, extension, image_size
    import pandas as pd
    frozen = json.loads((EVIDENCE / "selection.json").read_text())
    frame = pd.read_pickle(CACHE / "combine.pkl")
    rows = [{k: (list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v) for k, v in r.items()}
            for r in frame.to_dict("records")]
    shards = sorted((CACHE / "hf").glob("train-*.parquet"))
    from .selection_v4 import DATASET_SHARDS_SHA256
    shard_hashes = [sha256(p) for p in shards]
    reviews = json.loads((ADMISSION / "issue-review.json").read_text())["reviews"]
    order = frozen["eligible_repository_order"]
    sizes_path = EVIDENCE_V4 / "official-image-sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.is_file() else {}
    for position in range(FIRST_POSITION, last_position + 1):
        if position % 2 == 0 and order[position] not in sizes:
            sizes[order[position]] = image_size(order[position])
    write_json(sizes_path, sizes)
    record = extension(rows, CACHE / "SWE-smith" / "swesmith" / "profiles" / "python.py", frozen, reviews, sizes,
                       range(FIRST_POSITION, last_position + 1))
    record["dataset_shards_sha256_match_pin"] = shard_hashes == DATASET_SHARDS_SHA256
    record["frozen_v1_selection_sha256"] = sha256(EVIDENCE / "selection.json")
    record["issue_review_sha256"] = sha256(ADMISSION / "issue-review.json")
    record["official_image_max_bytes"] = OFFICIAL_IMAGE_MAX_BYTES
    write_json(SELECTION, record)
    committed = {k: v for k, v in record.items()}
    committed["families"] = [{k: v for k, v in f.items() if k != "profile"} | (
        {"profile_class": f["profile"]["class"]} if f.get("profile") else {}) for f in record["families"]]
    write_json(ADMISSION / "selection.json", committed)
    return record
