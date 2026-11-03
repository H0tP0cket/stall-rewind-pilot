"""v4 extension of the frozen v1 selection: more calibration families in the same seeded order.

Declared before any v4 instance was packaged. Nothing frozen by v1 changes:
dataset pin, eligibility criteria E1-E7 and R1-R3, seed, the eligible
repository order (``selection.json``), each repository's seeded instance order
and the split rule (position 0, 2, 4, ... calibration; 1, 3, 5, ... held_out).

v4 continues that order after the 14 v1 families (positions 14, 15, ...):

* A family keeps the split its position gives it. Held-out-position families are
  assigned ``held_out`` and recorded; v4 does not package them (deferred), and
  they can never enter calibration.
* Calibration-position families are processed in order until the admitted new
  calibration tasks reach the target (stop at a family boundary).
* Per family, the seeded instance order is walked and two new instance filters
  are applied (I1 mechanical, I2 reviewed, both on dataset text only); the first
  PER_FAMILY passing instances are candidates, the next RESERVES_PER_FAMILY are
  reserves (v1 reserve rule).
* R4: a family whose official environment image exceeds the resource bound is
  held without an attempted rebuild (recorded; its split does not change).
* Added measured gate: every grade of the final build takes <= GRADE_SECONDS_MAX.

No model or provider calls; no outcome of any model enters selection.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import random
import re

from .selection import (CRITERIA, EXISTING_FAMILIES, PER_FAMILY, RESERVES_PER_FAMILY, SEED, FAMILY_TARGET,
                        instance_checks, parse_profiles)

FIRST_POSITION = FAMILY_TARGET          # 14: first family after the v1 families
CALIBRATION_TARGET = 20                 # "about 20 more calibration tasks"
OFFICIAL_IMAGE_MAX_BYTES = 4_000_000_000
GRADE_SECONDS_MAX = 60.0
MIN_FRAGMENT_CHARS = 6
MIN_LINE_CHARS = 10

DATASET_SHARDS_SHA256 = [
    "c9078e71de617fcb1c6e2719b79f09890e93c3e88a61c5ee9cd7bb79850d714d",
    "3e8af6d9125bfbc5835386e8f992f054de6cda6f57bf0cb7ac3a038f0e11502e",
    "e17367bdd0fc87fe3bede13f64f0f74c79d71a61a90ff4b444b201ab21d50f5a",
    "dc8d53b2452eb970fde590546a11f830afee293a7cb8a908ca2c6b93d9a445b4",
    "b6e9c84aaf149e11a417bb25e66ffccf6f793596bbd6f706fb2ae5a88fd7b360",
    "e9e3025562ae112f6fb80d53c83971e8d418b271641481ca8715881c9cb274f1",
    "78dba1970cbd3c466466043e83ed1d0505b45e0e3f1cad0234d40cbbce6fedbd",
    "27aa219993d814884296b4798a19bbf8ddbf636e3c0b47578cbaa1fb780a74c2",
    "e3cd42eae5ba5e0543341cf1a4a3f5ea1b0647408883709c6ee5725d20b3ae73",
    "74dcce80661ef5bbb700709f05e1b23f5934fbd9fece160201c6aa0a86aa9d61",
    "360c6b5b058dd7404c874d5691482a9be86a77e889008a71a51f55eaee6d3086",
]

CRITERIA_V4 = {
    "version": "swesmith-long.selection.v4-extension",
    "extends": CRITERIA["version"],
    "extends_criteria_sha256": hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest(),
    "unchanged": ["dataset pin", "E1-E7 instance eligibility", "R1-R3 repository eligibility", f"seed {SEED}",
                  "eligible repository order (v1 selection.json)", "per-repository seeded instance order",
                  "split rule: even position calibration, odd position held_out",
                  f"{PER_FAMILY} candidates + {RESERVES_PER_FAMILY} reserves per family; reserves only after an "
                  "instance-specific failure; an environment failure holds the whole family"],
    "family_extension": {
        "order": f"continue the frozen eligible repository order at position {FIRST_POSITION}",
        "held_out_positions": "assigned held_out and recorded; not packaged in v4 (deferred); never calibration",
        "calibration_positions": "processed in order",
        "stop_rule": f"stop after the first calibration-position family (in order) at which the admitted new "
                     f"calibration tasks reach >= {CALIBRATION_TARGET}; a family processed after that point is "
                     "recorded but not admitted",
        "R4_resource_bound": f"official environment image (compressed layers) <= {OFFICIAL_IMAGE_MAX_BYTES} bytes; a "
                             "larger family is held (environment: resource bound) without an attempted rebuild",
    },
    "instance_filters": {
        "I1_no_fix_code_in_issue": (
            "for no bug hunk does the problem statement contain the fix code: (a) a clean-side changed line of the "
            f"hunk (whitespace-normalized, >= {MIN_LINE_CHARS} non-space characters, not a comment, not also a "
            "buggy-side line), or (b) every clean-side changed fragment of the hunk (token diff between the "
            f"buggy and clean changed lines, one token of context, >= {MIN_FRAGMENT_CHARS} characters; "
            "trailing '# ...' comments ignored). Stricter than 'not every bug': one disclosed hunk rejects."),
        "I2_issue_reviewed": (
            "the full problem statement is read before packaging and compared with the bug patch. A hunk's "
            "correction counts as stated when the text names the code element responsible (function, method, "
            "class, argument, parameter, condition, property) and the wrong-versus-right change for it (e.g. "
            "'X instead of Y', 'inverted', 'arguments swapped', 'missing __init__', 'returns None instead of the "
            "loader', 'splits only once'); symptoms, expected outputs and lists of affected APIs do not count. "
            "Hunks changing only comments, docstrings or blank lines are not counted. Reject when corrections "
            "are stated for more than half of the counted hunks; borderline cases are rejected. Verdicts and "
            "notes: issue-review.json. Instances are reviewed in seeded order only until the family has its "
            "candidates and reserves."),
        "I2_amendment": (
            "Wording revised after reading the first texts and before any v4 packaging: the committed draft "
            "(e8816941) rejected only when every hunk's correction was stated; django__daphne combine_file__ngqv80py, "
            "an 8-item enumerated fix list stating 6 of 9 hunks, passed that literal rule, so the threshold "
            "became 'more than half of the counted hunks'."),
        "drawing": f"walk the seeded instance order; the first {PER_FAMILY} instances passing I1 and I2 are "
                   f"candidates, the next {RESERVES_PER_FAMILY} are reserves",
    },
    "measured_gates_added": {
        "G1_grade_time": f"every grade of the final (v3-format) validation (start x2, reference x2, partial controls, "
                         f"tamper) takes <= {GRADE_SECONDS_MAX} s inside the grading container; otherwise HOLD",
    },
    "decided_before": "any v4 packaging outcome and any model outcome (no model calls are made by this lane)",
}


# --------------------------------------------------------------------------- I1: fix code in the issue text
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\"[^\"\n]*\"|'[^'\n]*'|==|!=|<=|>=|//|\*\*|->|"
                   r"[-+*/%<>=!&|^~:\[\](){},.@]")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _code(line: str) -> str:
    """Drop a trailing '# ...' comment (SWE-smith bug comments) outside simple string literals."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def patch_hunks(patch: str) -> list[dict]:
    """Per hunk: file, buggy-side (+) and clean-side (-) changed lines of the SWE-smith bug patch."""
    hunks, current, name = [], None, None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current, name = None, line.split(" b/", 1)[-1]
        elif line.startswith("@@"):
            current = {"file": name, "header": line, "clean": [], "buggy": []}
            hunks.append(current)
        elif current is not None and line.startswith("-") and not line.startswith("---"):
            current["clean"].append(line[1:])
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            current["buggy"].append(line[1:])
    return hunks


def _fragments(clean: list[str], buggy: list[str]) -> list[list[str]]:
    a = TOKEN.findall(" ".join(_code(x) for x in buggy))
    b = TOKEN.findall(" ".join(_code(x) for x in clean))
    result = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            fragment = b[max(0, j1 - 1):min(len(b), j2 + 1)]
            if len("".join(fragment)) >= MIN_FRAGMENT_CHARS:
                result.append(fragment)
    return result


def _fragment_in(text: str, fragment: list[str]) -> bool:
    return re.search(r"\s*".join(re.escape(t) for t in fragment), text) is not None


def issue_disclosure(problem_statement: str, patch: str) -> dict:
    """I1: which bug hunks have their fix code in the issue text."""
    text, flat = problem_statement, _norm(problem_statement)
    rows = []
    for hunk in patch_hunks(patch):
        buggy = {_norm(_code(x)) for x in hunk["buggy"]}
        lines = []
        for raw in hunk["clean"]:
            line = _norm(_code(raw))
            if len(line.replace(" ", "")) >= MIN_LINE_CHARS and not raw.strip().startswith("#") and line not in buggy:
                lines.append(line)
        line_hits = [line for line in lines if line in flat]
        fragments = _fragments(hunk["clean"], hunk["buggy"])
        fragment_hits = [" ".join(f) for f in fragments if _fragment_in(text, f)]
        all_fragments = bool(fragments) and len(fragment_hits) == len(fragments)
        rows.append({"file": hunk["file"], "header": hunk["header"][:120], "clean_lines_checked": len(lines),
                     "clean_line_hits": line_hits, "fragments_checked": len(fragments),
                     "fragment_hits": fragment_hits, "disclosed": bool(line_hits) or all_fragments})
    return {"hunks": len(rows), "disclosed_hunks": sum(r["disclosed"] for r in rows),
            "passes": not any(r["disclosed"] for r in rows), "per_hunk": rows}


# --------------------------------------------------------------------------- family extension
def eligible(rows: list[dict]) -> dict[str, list[dict]]:
    by_repo: dict[str, list[dict]] = {}
    for row in rows:
        if all(instance_checks(row).values()):
            by_repo.setdefault(row["instance_id"].rsplit(".", 1)[0], []).append(row)
    return by_repo


def seeded_instances(repo_key: str, repo_rows: list[dict]) -> list[str]:
    ids = sorted(r["instance_id"] for r in repo_rows)
    random.Random(f"{SEED}:{repo_key}").shuffle(ids)
    return ids


def extension(rows: list[dict], profiles_source: Path, frozen_selection: dict, reviews: dict[str, dict],
              image_sizes: dict[str, int | None], positions: range) -> dict:
    """Recompute the frozen order, then draw v4 candidates for the given positions (no packaging)."""
    by_repo = eligible(rows)
    profiles = parse_profiles(profiles_source)
    # Recompute exactly as selection.select() did; the order must equal the frozen record.
    keys = []
    for repo_key in sorted(by_repo):
        upstream = repo_key.rsplit(".", 1)[0].replace("__", "/", 1)
        profile = profiles.get(repo_key.lower())
        if (len(by_repo[repo_key]) >= 3 and upstream.lower() not in {f.lower() for f in EXISTING_FAMILIES}
                and profile is not None and not set(profile["methods"]) & {"log_parser", "get_test_cmd", "get_test_files"}):
            keys.append(repo_key)
    order = sorted(keys)
    random.Random(SEED).shuffle(order)
    if order != frozen_selection["eligible_repository_order"]:
        raise ValueError("Recomputed eligible repository order differs from the frozen v1 selection")
    for family in frozen_selection["families"]:
        ids = seeded_instances(family["repo_key"], by_repo[family["repo_key"]])
        if ids[:PER_FAMILY] != family["candidates"] or ids[PER_FAMILY:PER_FAMILY + RESERVES_PER_FAMILY] != family["reserves"]:
            raise ValueError("Recomputed seeded instance order differs from the frozen v1 draw: " + family["repo_key"])
    rows_by_id = {r["instance_id"]: r for rows_ in by_repo.values() for r in rows_}
    families, pending_review = [], []
    for position in positions:
        repo_key = order[position]
        upstream = repo_key.rsplit(".", 1)[0].replace("__", "/", 1)
        split = "calibration" if position % 2 == 0 else "held_out"
        family = {"order": position, "repo_key": repo_key, "upstream": upstream, "family_id": "project:" + upstream,
                  "split": split, "eligible_instances": len(by_repo[repo_key]),
                  "profile": profiles[repo_key.lower()]}
        if split == "held_out":
            family["v4_status"] = "ASSIGNED_HELD_OUT_NOT_PROCESSED"
            families.append(family)
            continue
        size = image_sizes.get(repo_key)
        family["official_image_compressed_bytes"] = size
        if size is None:
            family["v4_status"] = "IMAGE_SIZE_UNKNOWN"
        elif size > OFFICIAL_IMAGE_MAX_BYTES:
            family["v4_status"] = "HOLD_R4_RESOURCE_BOUND"
        else:
            family["v4_status"] = "DRAWN"
        walk, drawn = [], []
        for iid in (seeded_instances(repo_key, by_repo[repo_key]) if family["v4_status"] == "DRAWN" else []):
            if len(drawn) == PER_FAMILY + RESERVES_PER_FAMILY:
                break
            row = rows_by_id[iid]
            i1 = issue_disclosure(row["problem_statement"], row["patch"])
            step = {"instance_id": iid, "I1_passes": i1["passes"], "I1_disclosed_hunks": i1["disclosed_hunks"],
                    "hunks": i1["hunks"]}
            if i1["passes"]:
                review = reviews.get(iid)
                if review is None:
                    step["I2"] = "PENDING"
                    pending_review.append(iid)
                    walk.append(step)
                    break
                step["I2"] = review["verdict"]
                if review["verdict"] == "pass":
                    drawn.append(iid)
            walk.append(step)
        family["walk"] = walk
        family["candidates"] = drawn[:PER_FAMILY]
        family["reserves"] = drawn[PER_FAMILY:PER_FAMILY + RESERVES_PER_FAMILY]
        if family["v4_status"] == "DRAWN" and len(drawn) < PER_FAMILY + RESERVES_PER_FAMILY and not any(
                s.get("I2") == "PENDING" for s in walk):
            family["draw_note"] = "seeded order exhausted before 3 candidates + 2 reserves passed I1/I2"
        families.append(family)
    return {"criteria": CRITERIA_V4,
            "criteria_sha256": hashlib.sha256(json.dumps(CRITERIA_V4, sort_keys=True).encode()).hexdigest(),
            "frozen_order_verified": True, "frozen_v1_draws_verified": True,
            "families": families, "pending_review": pending_review}


def image_size(repo_key: str) -> int:
    from .official import manifest
    from .pipeline import official_image
    info = manifest(official_image(repo_key))
    layers = info["manifest"].get("layers")
    if not layers:
        raise ValueError("Expected a single-platform image manifest")
    return sum(layer["size"] for layer in layers)
