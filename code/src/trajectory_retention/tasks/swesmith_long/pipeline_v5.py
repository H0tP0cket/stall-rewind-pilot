"""v5: harder calibration instances (published-trajectory hard band), built by the unchanged v4 pipeline steps.

Selection (selection_v5, declared before packaging) is recorded by
``write_selection``: the external-evidence snapshot, the merged I2 verdicts, the
comment-leak verdicts and the ranked walk. Building reuses the v1 -> v2 -> v3
steps exactly as pipeline_v4 does, with v5 identities and staging directories.
No model or provider calls.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .adapter import sha256
from .pipeline import CACHE, EVIDENCE, WORKTREE, write_json
from .selection_v5 import TRAJECTORIES

LANE = "swesmith-long-v5"
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / LANE
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / LANE
EVIDENCE_V5 = EVIDENCE / "v5"
SELECTION = EVIDENCE_V5 / "selection.json"

SCOUT = Path("/private/tmp/claude-501/-Users-sidsri-projects-llm-trajectories/aa27fe0b-bf94-4143-b375-a8ab4c0d80a8/"
             "scratchpad/task-swarm/swesmith")
STALL_SUPPLY = Path(__file__).resolve().parents[4]  # the stall-supply branch was merged into main
LEAKS = STALL_SUPPLY / "benchmark_tasks/admission/stall-supply-v1/swesmith-leaks"
LEAK_LIST = LEAKS / "comment-leak-external-candidates.json"
COMMENT_RULE = LEAKS / "scripts/comment_rule.py"

EVIDENCE_COLUMNS = ["instance_id", "repo_key", "strategy", "family_position", "family_split", "family_used",
                    "seeded_rank_in_family", "official_image_bytes", "f2p", "f2p_functions", "f2p_files", "p2p",
                    "bug_files", "bug_hunks", "I1_passes", "I1_disclosed_hunks", "attempts", "resolved_attempts",
                    "resolve_rate", "attempts_c37", "resolved_c37", "attempts_c35", "resolved_c35", "attempts_g4o",
                    "resolved_g4o", "runs", "succ_steps_min", "succ_steps_median", "succ_steps_max",
                    "succ_tokens_final_context_median_est", "fail_steps_median", "fail_tokens_final_context_median_est",
                    "fail_hit_step_cap", "fail_ended_submit", "hunks_measurable", "fail_max_hunks_written_frac",
                    "fail_min_hunks_written_frac", "issue_text_same_as_pinned_frac", "network_risk", "timing_risk",
                    "grade_time_risk"]


def _value(text: str):
    if text in ("", "nan", "NaN"):
        return None
    if text in ("True", "False"):
        return text == "True"
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and "." not in text else number


def evidence_snapshot() -> Path:
    """Committed copy of the scout's published-trajectory evidence for every calibration-position row."""
    source = SCOUT / "all_eligible_with_evidence.csv"
    with source.open() as handle:
        rows = [{k: _value(r[k]) for k in EVIDENCE_COLUMNS} for r in csv.DictReader(handle)
                if r["family_split"] == "calibration"]
    verdicts = {}
    for path in sorted((SCOUT / "work/i2").glob("verdicts*.json")):
        verdicts.update(json.loads(path.read_text()))
    ids = {r["instance_id"] for r in rows}
    snapshot = {
        "schema_version": "swesmith-long.external-evidence.v5",
        "note": ("Published SWE-agent outcomes (other people's runs) for every lane-eligible (E1-E7) instance with at "
                 "least one published trajectory in a calibration-position family. Copied from the external-evidence "
                 "scout (research output, no model calls). Columns as in the scout's all_eligible_with_evidence.csv."),
        "trajectories": TRAJECTORIES,
        "sources_sha256": {name: sha256(SCOUT / name) for name in
                           ("all_eligible_with_evidence.csv", "ranked_all_tiers.csv", "candidates_top60.csv",
                            "README.md", "work/rank.py", "work/traj_features.py", "work/hunk_coverage.py")},
        "scout_i2_sources_sha256": {p.name: sha256(p) for p in sorted((SCOUT / "work/i2").glob("verdicts*.json"))},
        "rows": sorted(rows, key=lambda r: r["instance_id"]),
        "scout_i2_verdicts": {k: v for k, v in sorted(verdicts.items()) if k in ids},
    }
    path = ADMISSION / "external-evidence.json"
    write_json(path, snapshot)
    return path


def drawn_before() -> set[str]:
    """Every instance drawn (candidate/reserve) in v1 or drawn/walked in v4."""
    admission = WORKTREE / "benchmark_tasks" / "admission"
    ids = set()
    for family in json.loads((admission / "swesmith-long-v1/selection.json").read_text())["families"]:
        ids.update(family["candidates"] + family["reserves"])
    for family in json.loads((admission / "swesmith-long-v4/selection.json").read_text())["families"]:
        ids.update(family.get("candidates") or [])
        ids.update(family.get("reserves") or [])
        ids.update(step["instance_id"] for step in family.get("walk") or [])
    return ids


def merged_reviews(snapshot: dict) -> dict[str, dict]:
    lane = json.loads((WORKTREE / "benchmark_tasks/admission/swesmith-long-v4/issue-review.json").read_text())["reviews"]
    ours = json.loads((ADMISSION / "issue-review.json").read_text())["reviews"]
    merged = {}
    for source, reviews in (("v5 review", ours), ("external-evidence scout review", snapshot["scout_i2_verdicts"]),
                            ("lane v4 review", lane)):
        for iid, review in reviews.items():
            merged[iid] = {"verdict": review["verdict"], "note": review.get("note"), "source": source}
    return merged  # later sources overwrite earlier ones: lane v4 > scout > v5


def comment_leak_verdicts() -> tuple[dict[str, dict], dict]:
    verdicts, meta = {}, {"list": str(LEAK_LIST), "available": LEAK_LIST.is_file()}
    if LEAK_LIST.is_file():
        meta["list_sha256"] = sha256(LEAK_LIST)
        data = json.loads(LEAK_LIST.read_text())
        meta["list_rule"] = data.get("status")
        for row in data["candidates"]:
            verdicts[row["instance_id"]] = {"leak": bool(row["exclude"]), "verdict": row.get("verdict"),
                                            "note": row.get("note"), "source": "stall-supply comment-leak list"}
    review = ADMISSION / "comment-leak-review.json"
    if review.is_file():
        meta["v5_review_sha256"] = sha256(review)
        for iid, row in json.loads(review.read_text())["reviews"].items():
            verdicts.setdefault(iid, {"leak": row["leak"], "note": row["note"],
                                      "source": "v5 review with comment_rule.py"})
    if COMMENT_RULE.is_file():
        meta["comment_rule_sha256"] = sha256(COMMENT_RULE)
    return verdicts, meta


def family_positions() -> dict[str, int]:
    frozen = json.loads((EVIDENCE / "selection.json").read_text())
    return {key: i for i, key in enumerate(frozen["eligible_repository_order"])}


def write_selection() -> dict:
    from .selection_v5 import CRITERIA_V5, walk
    from .selection_v4 import DATASET_SHARDS_SHA256
    snapshot_path = evidence_snapshot()
    snapshot = json.loads(snapshot_path.read_text())
    reviews = merged_reviews(snapshot)
    leaks, leak_meta = comment_leak_verdicts()
    positions = family_positions()
    ranked = walk(snapshot["rows"], family_positions=positions, drawn_before=drawn_before(), reviews=reviews,
                  comment_leak=leaks)
    shards = sorted((CACHE / "hf").glob("train-*.parquet"))
    record = {"criteria": CRITERIA_V5,
              "criteria_sha256": hashlib.sha256(json.dumps(CRITERIA_V5, sort_keys=True).encode()).hexdigest(),
              "dataset_shards_sha256_match_pin": [sha256(p) for p in shards] == DATASET_SHARDS_SHA256,
              "frozen_v1_selection_sha256": sha256(EVIDENCE / "selection.json"),
              "external_evidence_sha256": sha256(snapshot_path),
              "issue_review_v5_sha256": sha256(ADMISSION / "issue-review.json"),
              "comment_leak": leak_meta,
              "ranked": ranked,
              "counts": {status: sum(r["status"] == status for r in ranked)
                         for status in sorted({r["status"] for r in ranked})}}
    committed = ADMISSION / "selection.json"
    if committed.is_file():
        previous = json.loads(committed.read_text())
        history = previous.get("previous_selection_sha256", [])
        if previous.get("ranked") != ranked or previous.get("criteria_sha256") != record["criteria_sha256"]:
            history = history + [sha256(committed)]
        record["previous_selection_sha256"] = history
        record["revision_note"] = ("recomputed with more verdicts (comment-leak list; I2 reviews made when a family "
                                   "still had free slots, per the I2 rule's 'reviewed only until needed'); the rule "
                                   "itself (criteria_sha256) is unchanged")
    write_json(SELECTION, record)
    write_json(committed, record)
    return record


# --------------------------------------------------------------------------- families
def family_record(repo_key: str) -> dict:
    """Family metadata as v4 records it (position, split, SWE-smith profile, official image size)."""
    from .selection import parse_profiles
    from .selection_v4 import image_size
    position = family_positions()[repo_key]
    if position % 2:
        raise ValueError(f"{repo_key} is a held-out-position family (position {position}); never used")
    upstream = repo_key.rsplit(".", 1)[0].replace("__", "/", 1)
    profiles = parse_profiles(CACHE / "SWE-smith" / "swesmith" / "profiles" / "python.py")
    sizes_path = EVIDENCE_V5 / "official-image-sizes.json"
    sizes = json.loads(sizes_path.read_text()) if sizes_path.is_file() else {}
    if repo_key not in sizes:
        sizes[repo_key] = image_size(repo_key)
        write_json(sizes_path, sizes)
    return {"order": position, "repo_key": repo_key, "upstream": upstream, "family_id": "project:" + upstream,
            "split": "calibration", "profile": profiles[repo_key.lower()],
            "official_image_compressed_bytes": sizes[repo_key]}


TRANSIENT = ("TimeoutError", "URLError", "ConnectionResetError", "RemoteDisconnected", "IncompleteRead")


def prepare_environment(repo_key: str) -> dict:
    """v1 family environment step (pipeline.prepare_family, unchanged; reused when already recorded)."""
    from . import pipeline
    from .selection_v4 import OFFICIAL_IMAGE_MAX_BYTES
    family = family_record(repo_key)
    if family["official_image_compressed_bytes"] > OFFICIAL_IMAGE_MAX_BYTES:
        raise ValueError("R4: official image exceeds the resource bound")
    spec = {k: family[k] for k in ("repo_key", "family_id", "split", "profile")}
    env = pipeline.prepare_family(spec)
    retries_path = EVIDENCE_V5 / "families" / repo_key / "transient-retries.json"
    retries = json.loads(retries_path.read_text()) if retries_path.is_file() else []
    # A download timeout while streaming the official image metadata is an infrastructure failure of this host's
    # network, not a measurement of the family: record it and retry (at most twice), clearing the partial cache.
    while (env["status"] != "READY" and str(env.get("reason", "")).startswith(TRANSIENT) and len(retries) < 2):
        import shutil
        retries.append({"reason": env.get("reason")})
        write_json(retries_path, retries)
        shutil.rmtree(CACHE / "official" / repo_key, ignore_errors=True)
        env = pipeline.prepare_family(spec, force=True)
    summary = {k: env.get(k) for k in ("status", "reason", "reference_stable_pass", "reference_flaky")}
    summary["reference_wall_seconds"] = [r.get("wall_seconds") for r in env.get("reference_runs") or []]
    summary["official_image_compressed_bytes"] = family["official_image_compressed_bytes"]
    summary["order"] = family["order"]
    write_json(EVIDENCE_V5 / "families" / repo_key / "environment.json", summary)
    return env


# --------------------------------------------------------------------------- build (unchanged v4 steps)
STAGE = EVIDENCE_V5 / "stage"                       # gitignored; intermediate v1/v2 packages and their evidence
PACKAGES_V1_STAGE = STAGE / "packages" / "swesmith-long-v1"
PACKAGES_V2_STAGE = STAGE / "packages" / "swesmith-long-v2"
STAGE_EVIDENCE_V2 = STAGE / "evidence" / "v2"       # v2 family needles / task records for the v5 builds
STAGE_EVIDENCE_V3 = STAGE / "evidence" / "v3"       # v3 family image scans for the v5 builds
WORK = CACHE / "work-v5"
ADAPTATION_NOTE = ("v3 package format and adaptation record (swesmith-long.adaptation.v3) unchanged; first issued in the "
                   "v5 lane release (published-hard-band selection); intermediate v1/v2 build stages are not published")


def v5_id(v2_id: str) -> str:
    if not v2_id.endswith("-v2"):
        raise ValueError(v2_id)
    return v2_id[:-3] + "-v5"


def _staged(key: str, v2_ids: list[str]):
    """Point the unchanged v1/v2/v3 steps at v5 staging trees (packages and family/task evidence).

    Redirecting the v2/v3 evidence roots keeps the v2/v3 family records of families that already
    have v2/v3/v4 tasks unchanged (the needle scan of a family is recomputed for the new tasks only).
    """
    from contextlib import contextmanager
    from . import pipeline, pipeline_v2, pipeline_v3

    @contextmanager
    def manager():
        saved = (pipeline.PACKAGES, pipeline_v2.PACKAGES_V1, pipeline_v2.PACKAGES, pipeline_v2.EVIDENCE_V2,
                 pipeline_v3.PACKAGES_V2, pipeline_v3.family_packages, pipeline_v3.EVIDENCE_V2, pipeline_v3.EVIDENCE_V3)
        pipeline.PACKAGES = PACKAGES_V1_STAGE
        pipeline_v2.PACKAGES_V1 = PACKAGES_V1_STAGE
        pipeline_v2.PACKAGES = PACKAGES_V2_STAGE
        pipeline_v2.EVIDENCE_V2 = STAGE_EVIDENCE_V2
        pipeline_v3.PACKAGES_V2 = PACKAGES_V2_STAGE
        pipeline_v3.family_packages = lambda k: list(v2_ids) if k == key else saved[5](k)
        pipeline_v3.EVIDENCE_V2 = STAGE_EVIDENCE_V2
        pipeline_v3.EVIDENCE_V3 = STAGE_EVIDENCE_V3
        try:
            yield
        finally:
            (pipeline.PACKAGES, pipeline_v2.PACKAGES_V1, pipeline_v2.PACKAGES, pipeline_v2.EVIDENCE_V2,
             pipeline_v3.PACKAGES_V2, pipeline_v3.family_packages, pipeline_v3.EVIDENCE_V2,
             pipeline_v3.EVIDENCE_V3) = saved
    return manager()


def rows_for(ids: list[str]) -> dict[str, dict]:
    import pandas as pd
    frame = pd.read_pickle(CACHE / "combine.pkl")
    frame = frame[frame.instance_id.isin(ids)]
    return {r["instance_id"]: {k: (list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v) for k, v in r.items()}
            for r in frame.to_dict("records")}


def build_v5(v2_id: str, family: dict, selection_sha256: str) -> dict:
    """pipeline_v4.build_v4 with the v5 identity, v5 staging and the G1 gate (steps unchanged)."""
    import shutil
    from . import pipeline_v3, source
    from .adapter import file_manifest
    from .adapter_v3 import GIT_TAR, SCHEMA
    from .adapter_v5 import ADAPTER_ENTRY, SWESmithLongV5TaskAdapter
    from .selection_v4 import GRADE_SECONDS_MAX
    from .selection_v5 import CRITERIA_V5
    task_id = v5_id(v2_id)
    out = EVIDENCE_V5 / "tasks" / task_id
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
        SWESmithLongV2TaskAdapter(old, scratch_root=CACHE / "grade-scratch")
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
                                     "selection": {"criteria": CRITERIA_V5["version"], "selection_sha256": selection_sha256}},
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
        for key in ("counts", "bug_shape", "native_start"):
            record[key] = old_manifest[key]
        SWESmithLongV5TaskAdapter(package, scratch_root=CACHE / "grade-scratch")
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


def family_git_image(repo_key: str) -> dict:
    """The v3 git image of the family: reuse the recorded v3/v4 image when its tag still resolves to it, else build."""
    from . import pipeline_v3
    from .environment import image_id
    for recorded in (STAGE_EVIDENCE_V3 / "families" / repo_key / "image.json",
                     EVIDENCE / "v3" / "families" / repo_key / "image.json"):
        if not recorded.is_file():
            continue
        image = json.loads(recorded.read_text())
        if image.get("image_ok") and image_id(image["tag"]) == image["image_id"]:
            image = {k: v for k, v in image.items() if k != "scans"}
            image.setdefault("reused_from", str(recorded.relative_to(WORKTREE)))
            (pipeline_v3.EVIDENCE_V3 / "families" / repo_key).mkdir(parents=True, exist_ok=True)
            return image
    return pipeline_v3.build_family_image(repo_key)


def process_instances(repo_key: str, instance_ids: list[str]) -> list[dict]:
    """Build, gate and runner-check the given instances of one family (rank order); append to its ledger."""
    from . import pipeline, pipeline_v2, pipeline_v3
    from .adapter_v5 import SWESmithLongV5TaskAdapter
    from .runner_check import run_check
    selection_sha = sha256(SELECTION)
    ledger_path = EVIDENCE_V5 / "candidates" / f"{repo_key}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
    env = prepare_environment(repo_key)
    if env["status"] != "READY":
        for iid in instance_ids:
            ledger.append({"instance_id": iid, "status": "HOLD", "stage": "family-environment",
                           "reason": "family environment: " + str(env.get("reason"))})
        write_json(ledger_path, ledger)
        return ledger
    env["profile"] = family_record(repo_key)["profile"]
    rows = rows_for(instance_ids)
    v1_ok = [r["v1_task"] for r in ledger if r.get("v1_status") == "VALIDATED"]
    v2_ok = [r["v2_task"] for r in ledger if r.get("v2_status") == "VALIDATED"]
    image = None
    for iid in instance_ids:
        entry = {"instance_id": iid}
        ledger.append(entry)
        failed = None
        with _staged(repo_key, v2_ok):
            v1 = pipeline.package_instance(rows[iid], env)
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
            with _staged(repo_key, v2_ok):
                image_path = pipeline_v3.EVIDENCE_V3 / "families" / repo_key / "image.json"
                if image is None:
                    image = family_git_image(repo_key)
                    if not image["image_ok"]:
                        write_json(image_path, image)
                        entry.update(status="HOLD", stage="family-git-image", reason="v3 image check failed")
                        write_json(ledger_path, ledger)
                        return ledger
                image["scans"] = pipeline_v3.family_scans(repo_key, image["image_id"])
                write_json(image_path, image)
                final = build_v5(v2["task_id"], image, selection_sha)
            entry.update(task_id=final["task_id"], v5_status=final["status"],
                         grade_seconds=(final.get("validation") or {}).get("grade_seconds"),
                         grades_equal_v2=(final.get("v2_comparison") or {}).get("all_equal"),
                         partial_control_hunk=(final.get("validation") or {}).get("partial_control_hunk"))
            if final["status"] != "VALIDATED":
                failed = ("v5-format-gates", final.get("reason"))
            else:
                summary = run_check(PACKAGES / final["task_id"], EVIDENCE_V5 / "tasks" / final["task_id"] / "runner-check",
                                    adapter_cls=SWESmithLongV5TaskAdapter)
                entry["runner_check"] = "passed" if summary.get("passed") else "failed"
                if not summary.get("passed"):
                    failed = ("runner-check", summary.get("error") or ", ".join(
                        k for k, v in summary.get("assertions", {}).items() if not v))
        if failed is None:
            entry["status"] = "VALIDATED"
        else:
            entry.update(status="HOLD", stage=failed[0], reason=str(failed[1])[:1500])
        write_json(ledger_path, ledger)
        print(json.dumps({k: entry.get(k) for k in ("instance_id", "status", "stage", "reason", "task_id")}), flush=True)
    return ledger


# --------------------------------------------------------------------------- stall rationale (per new task)
LIFECYCLE_NOTE = ("SWE-smith lifecycle is one submission per run (native): a worker that submits early ends the run, so "
                  "GPT-6 Luna over-commit on these tasks shows up only as a wrong final submission (partial score), not "
                  "as a stall; stuck stalls are expected from a worker that keeps working without passing the hidden "
                  "FAIL_TO_PASS tests (withheld natively).")


def stall_rationale(task_id: str) -> dict:
    """Published evidence and the reason this task is expected to produce stuck stalls (no model calls)."""
    record = json.loads((EVIDENCE_V5 / "tasks" / task_id / "package-record.json").read_text())
    selection = json.loads(SELECTION.read_text())
    step = next(s for s in selection["ranked"] if s["instance_id"] == record["instance_id"])
    snapshot = json.loads((ADMISSION / "external-evidence.json").read_text())
    row = next(r for r in snapshot["rows"] if r["instance_id"] == record["instance_id"])
    hunks, measurable = row["bug_hunks"], row["hunks_measurable"]
    best, worst = row["fail_max_hunks_written_frac"], row["fail_min_hunks_written_frac"]
    published = {k: row[k] for k in ("attempts", "resolved_attempts", "attempts_c37", "resolved_c37", "attempts_c35",
                                     "resolved_c35", "attempts_g4o", "resolved_g4o", "succ_steps_median",
                                     "fail_steps_median", "fail_tokens_final_context_median_est", "fail_hit_step_cap",
                                     "fail_ended_submit", "hunks_measurable", "fail_max_hunks_written_frac",
                                     "fail_min_hunks_written_frac", "issue_text_same_as_pinned_frac", "runs")}
    reasons = []
    if step["tier"] == "C":
        reasons.append(f"No published attempt resolved it ({row['resolved_attempts']}/{row['attempts']}, "
                       f"{row['attempts_c37']} by Claude 3.7 Sonnet), yet a failed attempt wrote the fix for "
                       f"{best:.0%} of the {measurable} measurable bug hunks: agents found part of the fix and stopped "
                       "short, the profile of our observed genuine stalls (isodate-1gq8eopf, python-pinyin-injlhl59).")
        if best < 1.0:
            reasons.append("At least one hunk was never written by any failed attempt: the missing bug is the likely "
                           "place for a stuck loop (tests keep failing after the easy bugs are fixed).")
        else:
            reasons.append("A failed attempt wrote every measurable hunk and still did not resolve: an incomplete or "
                           "wrong variant of a fix, or an unmeasurable hunk, stays wrong; the hidden FAIL_TO_PASS tests "
                           "give no direct signal about which.")
    else:
        reasons.append(f"Published agents resolved it in {row['resolved_attempts']} of {row['attempts']} attempts "
                       f"(median {row['succ_steps_median']} steps when solved, {row['fail_steps_median']} when not): "
                       "solvable but unreliable, with failed attempts that ran as long as the solved ones.")
    reasons.append(f"Multi-bug instance: {hunks} bug hunks in {len(record['bug_shape']['files'])} "
                   f"file(s); {record['counts']['fail_to_pass']} FAIL_TO_PASS tests, withheld natively, so the worker "
                   "sees only the issue text and the remaining visible tests.")
    if step.get("I2") and step["I2"].get("note"):
        reasons.append("Issue text vs bug patch (I2 review): " + step["I2"]["note"])
    reasons.append(LIFECYCLE_NOTE)
    out = {"task_id": task_id, "instance_id": record["instance_id"], "tier": step["tier"], "rank": step["rank"],
           "published_evidence": published, "evidence_source": snapshot["trajectories"],
           "expected_stall_type": "stuck (primary); over-commit visible only as a wrong final submission",
           "why": reasons, "status": "expectation from published outcomes; not measured on our workers"}
    write_json(ADMISSION / "tasks" / task_id / "stall-rationale.json", out)
    return out
