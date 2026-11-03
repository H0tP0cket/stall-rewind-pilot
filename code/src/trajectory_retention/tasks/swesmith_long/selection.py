"""Frozen, seeded candidate selection over the pinned SWE-smith dataset.

Criteria are declared here before sampling and are purely mechanical
(dataset metadata, bug patch shape, repository profile shape). Environment
feasibility is measured later and recorded per candidate; it never changes
which candidates were drawn.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import random
import re

DATASET = "SWE-bench/SWE-smith"
DATASET_REVISION = "ea6d7173829c7ec8fa16c22055699ff2e9188091"
SWESMITH_SOURCE_REVISION = "9b74ac08118a85c39c356802f7961893af73e07f"  # same pin as tasks/prompts.py
SEED = 20260923
PER_FAMILY = 3          # candidates drawn per repository family
RESERVES_PER_FAMILY = 2  # consumed only after an instance-specific (not environment) failure
FAMILY_TARGET = 14      # repositories processed in seeded order (stop once reached)

# Families already present in this repository's admission registry / other
# benchmark intakes (same upstream project => same family). They are excluded
# so that the new families can be split cleanly.
EXISTING_FAMILIES = frozenset({
    "sqlfluff/sqlfluff", "andialbrecht/sqlparse", "marshmallow-code/marshmallow",
    "more-itertools/more-itertools", "msiemens/tinydb", "pydata/xarray", "scott-griffiths/bitstring",
    "sybrenstuvel/python-rsa", "team-simpy/simpy", "tkem/cachetools", "pallets/flask",
})

CRITERIA = {
    "version": "swesmith-long.selection.v1",
    "dataset": DATASET, "dataset_revision": DATASET_REVISION, "seed": SEED,
    "instance": {
        "E1_python_pytest": "every FAIL_TO_PASS and PASS_TO_PASS id is a pytest node id '<path>.py::<name>'",
        "E2_multi_bug": "SWE-smith strategy combine_file or combine_module (several combined bugs)",
        "E3_issue_text": "non-empty problem_statement (the native worker instruction)",
        "E4_f2p_count": "8 <= |FAIL_TO_PASS| <= 60 (partial credit that can move; bounded grading)",
        "E5_f2p_functions": ">= 4 distinct FAIL_TO_PASS test functions after removing parametrization",
        "E6_patch_shape": "bug patch modifies only existing non-test .py files and has >= 2 hunks",
        "E7_p2p_count": "|PASS_TO_PASS| <= 1500 (grade-time proxy)",
    },
    "repository": {
        "R1_instances": ">= 3 eligible instances in the repository",
        "R2_new_family": "upstream project not already a family in this repository's intake: " + ", ".join(sorted(EXISTING_FAMILIES)),
        "R3_standard_profile": "SWE-smith PythonProfile without custom log_parser/get_test_cmd/get_test_files overrides",
    },
    "sampling": {
        "repository_order": "sorted eligible repository keys shuffled with random.Random(seed)",
        "instances": f"per repository: sorted eligible instance ids shuffled with random.Random(seed:repo_key); first {PER_FAMILY} are candidates, next {RESERVES_PER_FAMILY} are reserves",
        "reserve_rule": "a reserve is packaged only when a candidate fails for an instance-specific reason; environment failures hold the whole family",
        "stop_rule": f"process repositories in order until {FAMILY_TARGET} repositories have been attempted",
    },
    "measured_later": [
        "arm64 environment rebuild from the official image's pinned packages (HOLD family if it fails)",
        "reference passes all FAIL_TO_PASS; start fails them (HOLD instance otherwise)",
        "grade wall time (flag > 120 s)", "deterministic repeat grades", "leak scan",
    ],
    "split": {
        "rule": "families alternate by seeded repository order: 1st calibration, 2nd held_out, 3rd calibration, ...",
        "decided_before": "any model outcome (no model calls are made by this lane)",
    },
}

NODE = re.compile(r"^[^\s:]+\.py::\S")


def is_test_path(path: str) -> bool:
    """SWE-smith RepoProfile._is_test_path for Python files (profiles/base.py)."""
    root, _, file = path.rpartition("/")
    if file.lower().startswith("test") or file.rsplit(".", 1)[0].endswith("test"):
        return True
    return any(x in root.split("/") for x in ("tests", "test", "specs"))


def patch_files(patch: str) -> list[dict]:
    files, current = [], None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/(\S+) b/(\S+)$", line)
            current = {"a": m.group(1) if m else None, "b": m.group(2) if m else None, "hunks": 0,
                       "new": False, "deleted": False, "rename": False}
            files.append(current)
        elif current is not None:
            if line.startswith("new file mode"):
                current["new"] = True
            elif line.startswith("deleted file mode"):
                current["deleted"] = True
            elif line.startswith("rename from") or line.startswith("similarity index"):
                current["rename"] = True
            elif line.startswith("@@"):
                current["hunks"] += 1
    return files


def test_function(node_id: str) -> str:
    return node_id.split("[", 1)[0]


def instance_checks(row: dict) -> dict[str, bool]:
    f2p, p2p = list(row["FAIL_TO_PASS"]), list(row["PASS_TO_PASS"])
    strategy = row["instance_id"].rsplit(".", 1)[-1].split("__", 1)[0]
    files = patch_files(row["patch"])
    return {
        "E1_python_pytest": bool(f2p) and all(NODE.match(x) for x in f2p + p2p),
        "E2_multi_bug": strategy in ("combine_file", "combine_module"),
        "E3_issue_text": bool((row.get("problem_statement") or "").strip()),
        "E4_f2p_count": 8 <= len(f2p) <= 60,
        "E5_f2p_functions": len({test_function(x) for x in f2p}) >= 4,
        "E6_patch_shape": bool(files) and sum(f["hunks"] for f in files) >= 2 and all(
            f["a"] == f["b"] and f["a"] and f["a"].endswith(".py") and not is_test_path(f["a"])
            and not (f["new"] or f["deleted"] or f["rename"]) for f in files),
        "E7_p2p_count": len(p2p) <= 1500,
    }


def parse_profiles(source: Path) -> dict[str, dict]:
    """Static read of swesmith/profiles/python.py; keyed by owner/repo@commit8."""
    tree = ast.parse(source.read_text())
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not any(
                isinstance(b, ast.Name) and b.id == "PythonProfile" for b in node.bases):
            continue
        fields, methods = {}, []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                name = item.target.id
                try:
                    fields[name] = ast.literal_eval(item.value)
                except ValueError:
                    value = item.value
                    if (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "field"
                            and value.keywords and isinstance(value.keywords[0].value, ast.Lambda)):
                        try:
                            fields[name] = ast.literal_eval(value.keywords[0].value.body)
                        except ValueError:
                            fields[name] = {"unparsed": ast.unparse(value)}
                    else:
                        fields[name] = {"unparsed": ast.unparse(value)}
            elif isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        if {"owner", "repo", "commit"} <= set(fields):
            key = f"{fields['owner']}__{fields['repo']}.{fields['commit'][:8]}"
            result[key.lower()] = {"class": node.name, "fields": fields, "methods": methods}
    return result


def select(rows: list[dict], profiles: dict[str, dict]) -> dict:
    by_repo: dict[str, list[dict]] = {}
    excluded_rows = {}
    for row in rows:
        checks = instance_checks(row)
        repo_key = row["instance_id"].rsplit(".", 1)[0]
        if all(checks.values()):
            by_repo.setdefault(repo_key, []).append(row)
        else:
            excluded_rows[row["instance_id"]] = [k for k, v in checks.items() if not v]
    repos = []
    for repo_key in sorted(by_repo):
        owner_repo = by_repo[repo_key][0]["repo"]
        upstream = repo_key.rsplit(".", 1)[0].replace("__", "/", 1)
        profile = profiles.get(repo_key.lower())
        reasons = []
        if len(by_repo[repo_key]) < 3:
            reasons.append("R1_instances")
        if upstream.lower() in {f.lower() for f in EXISTING_FAMILIES}:
            reasons.append("R2_new_family")
        if profile is None or set(profile["methods"]) & {"log_parser", "get_test_cmd", "get_test_files"}:
            reasons.append("R3_standard_profile")
        repos.append({"repo_key": repo_key, "upstream": upstream, "mirror": owner_repo,
                      "eligible_instances": len(by_repo[repo_key]), "excluded_by": reasons,
                      "profile": profile})
    eligible = [r for r in repos if not r["excluded_by"]]
    order = sorted(r["repo_key"] for r in eligible)
    random.Random(SEED).shuffle(order)
    families = []
    for position, repo_key in enumerate(order[:FAMILY_TARGET]):
        ids = sorted(r["instance_id"] for r in by_repo[repo_key])
        random.Random(f"{SEED}:{repo_key}").shuffle(ids)
        info = next(r for r in eligible if r["repo_key"] == repo_key)
        families.append({"order": position, "repo_key": repo_key, "upstream": info["upstream"],
                         "family_id": "project:" + info["upstream"],
                         "split": "calibration" if position % 2 == 0 else "held_out",
                         "candidates": ids[:PER_FAMILY], "reserves": ids[PER_FAMILY:PER_FAMILY + RESERVES_PER_FAMILY],
                         "profile": info["profile"]})
    return {"criteria": CRITERIA, "criteria_sha256": hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest(),
            "rows_considered": len(rows), "rows_eligible": sum(len(v) for v in by_repo.values()),
            "repositories": [{k: v for k, v in r.items() if k != "profile"} for r in repos],
            "eligible_repository_order": order, "families": families}


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True, help="pickle of combine_* rows from the pinned parquet shards")
    parser.add_argument("--profiles", type=Path, required=True, help="swesmith/profiles/python.py at SWESMITH_SOURCE_REVISION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_pickle(args.rows)
    rows = [{k: (list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v) for k, v in r.items()}
            for r in frame.to_dict("records")]
    result = select(rows, parse_profiles(args.profiles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"eligible_rows": result["rows_eligible"], "eligible_repositories": len(result["eligible_repository_order"]),
                      "families": [(f["repo_key"], f["split"]) for f in result["families"]]}, indent=1))
