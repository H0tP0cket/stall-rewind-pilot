"""Pinned Terminal-Bench 2 source identity, task families and the frozen family split.

Upstream: https://github.com/laude-institute/terminal-bench-2 (Harbor task
format: ``instruction.md``, ``task.toml``, ``environment/Dockerfile`` plus build
context, ``solution/solve.sh``, ``tests/test.sh`` + ``tests/test_outputs.py``).
The Harbor dataset registry entry ``terminal-bench@2.0`` pins every task to
``REGISTRY_REVISION``; this lane reads the task bytes at ``SOURCE_REVISION``
(TB2.1 metadata commit) and records, per task, that its instruction,
environment, solution and tests are byte-identical to the registry revision.

Family split rule (``SPLIT_RULE``) is fixed here before any image build,
reference grade or model run for this lane.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess

SOURCE_URL = "https://github.com/laude-institute/terminal-bench-2"
SOURCE_REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"  # 2026-04-29 "Add task metadata to task manifests"
REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
REGISTRY_DATASET = "terminal-bench@2.0"
REGISTRY_REVISION = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"  # git_commit_id of every terminal-bench@2.0 task
LICENSE = "Apache-2.0"
DEFAULT_CACHE = Path.home() / ".cache/terminalbench"
CHECKOUT = DEFAULT_CACHE / "terminal-bench-2"
TASK_COUNT = 89

# Files a task's canonical content is compared on between SOURCE_REVISION and
# REGISTRY_REVISION (task.toml/README.md/.gitignore changed only in metadata).
CONTENT_EXCLUDES = ("README.md", ".gitignore", "task.toml")

# --------------------------------------------------------------------------- families
# 1. A Terminal-Bench task's family is the task itself: "terminal-bench:<task>".
# 2. MettleBench terminal_* tasks derived from a TB task are the SAME family and
#    keep MettleBench's recorded split (mettlebench.family-split.v1, commit record
#    docs/research/mettlebench-admission/README.md). Recorded values, copied here:
METTLEBENCH_INHERITED = {
    "caffe-cifar-10": "calibration",
    "cobol-modernization": "calibration",
    "code-from-image": "calibration",
    "build-cython-ext": "calibration",
    "train-fasttext": "calibration",
    "financial-document-processor": "calibration",
    "fix-git": "calibration",
    "kv-store-grpc": "calibration",
    "hf-model-inference": "calibration",
    "mcmc-sampling-stan": "calibration",
    "openssl-selfsigned-cert": "calibration",
    "overfull-hbox": "calibration",
    "protein-assembly": "calibration",
    "sam-cell-seg": "calibration",
    "db-wal-recovery": "held_out",
    "dna-assembly": "held_out",
    "git-leak-recovery": "held_out",
    "git-multibranch": "held_out",
    "sanitize-git-repo": "held_out",
    "multi-source-data-merger": "held_out",
    "build-pmars": "held_out",
    "build-pov-ray": "held_out",
    "pypi-server": "held_out",
    "pytorch-model-recovery": "held_out",
    "rstan-to-pystan": "held_out",
    "write-compressor": "held_out",
}
# 3. Families already used for development/calibration in the current registry
#    can never be held out (task-admission-criteria.md, "Family split").
FORCED_CALIBRATION = {
    "llm-inference-batching-scheduler": "tb2-batching-scheduler-artifact-v1..v9 (family "
                                        "terminal-bench:llm-inference-batching-scheduler) were development-family "
                                        "tasks in benchmark_tasks/admission/registry.json and ran in the 2026-09 "
                                        "old-era calibration",
}
# 4. Shared-reference grouping (cross-task copies): TB tasks are one family when
#    one task's environment build or shipped files contain another task's
#    reference solution or deliverable, when they share hidden test/reference
#    files, when their hidden reference sources are near-identical, or when they
#    are built on the same specific third-party input artefact. Determined from
#    the pinned Dockerfiles/solutions/tests only (scan recorded in family-split.json).
SHARED_REFERENCE_GROUPS = {
    "build-pmars": {
        "members": ("build-pmars", "winning-avg-corewars"),
        "reason": "winning-avg-corewars/environment/Dockerfile runs the build-pmars reference solution "
                  "(apt-get source pmars; sed -i 's/-DXWINGRAPHX//g' Makefile; LIB = -lncurses; make; install) "
                  "and leaves the built /app/pmars-*/ source tree in the task's start state",
    },
    "make-doom-for-mips": {
        "members": ("make-doom-for-mips", "make-mips-interpreter"),
        "reason": "make-mips-interpreter ships the MIPS doomgeneric ELF (make-doom-for-mips's deliverable, built "
                  "in its Dockerfile) and make-doom-for-mips ships vm.js (a MIPS interpreter, "
                  "make-mips-interpreter's deliverable); both share tests/test_outputs.py and tests/reference.jpg "
                  "byte-for-byte",
    },
    "path-tracing": {
        "members": ("path-tracing", "path-tracing-reverse"),
        "reason": "hidden reference sources environment/orig.c differ only in '#define SS_FACTOR' (3 vs 1); "
                  "path-tracing ships the compiled /app/orig of that program",
    },
    "qemu-alpine-ssh": {
        "members": ("qemu-alpine-ssh", "qemu-startup"),
        "reason": "both boot the same Alpine 3.19.0 x86_64 ISO under QEMU (same downloaded input artefact)",
    },
}
SPLIT_SEED = 20260923  # the repository's family-split seed (swesmith-long, mettlebench)
SPLIT_RULE = (
    "terminalbench.family-split.v1: families = TB2 task ids at SOURCE_REVISION, merged by SHARED_REFERENCE_GROUPS "
    "(family id = 'terminal-bench:<group key>'); a family containing a MettleBench-derived task inherits that "
    "recorded split (held_out wins if members disagree); FORCED_CALIBRATION families are calibration; every other "
    "family id is sorted, shuffled with random.Random(20260923), and positions 0,2,4,... (1st, 3rd, 5th, ...) are "
    "calibration, positions 1,3,5,... held_out. Fixed before any image build, reference grade or model run in this "
    "lane. Only calibration families are packaged for use; held-out families are never loaded for "
    "development/calibration and main loading stays closed.")

# Pre-pivot (archived, 2026-08-31/09-01, GLM-5.3 / Luna reasoning-off) Terminal-Bench conversions that were
# mined. MettleBench's split did not treat archived pre-pivot exposure as development use; this lane follows
# that precedent and records the exposure only.
PRE_PIVOT_MINED = ("custom-memory-heap-crash", "password-recovery", "prove-plus-comm", "vulnerable-secret",
                   "regex-log", "constraints-scheduling")


def family_key(task: str) -> str:
    for key, group in SHARED_REFERENCE_GROUPS.items():
        if task in group["members"]:
            return key
    return task


def family_id(task: str) -> str:
    return "terminal-bench:" + family_key(task)


def task_ids(checkout: Path = CHECKOUT) -> list[str]:
    return sorted(p.name for p in Path(checkout).iterdir() if (p / "task.toml").is_file())


def family_split(tasks: list[str]) -> dict:
    """Frozen split over the pinned task list. Returns {family_id: record}."""
    members: dict[str, list[str]] = {}
    for task in sorted(set(tasks)):
        members.setdefault(family_id(task), []).append(task)
    records: dict[str, dict] = {}
    free = []
    for fam, ts in sorted(members.items()):
        inherited = {t: METTLEBENCH_INHERITED[t] for t in ts if t in METTLEBENCH_INHERITED}
        forced = [t for t in ts if t in FORCED_CALIBRATION]
        if inherited:
            split = "held_out" if "held_out" in inherited.values() else "calibration"
            if forced and split == "held_out":
                raise ValueError("Forced calibration family inherits held_out: " + fam)
            records[fam] = {"members": ts, "split": split, "basis": "mettlebench_inherited",
                            "inherited_from": inherited}
        elif forced:
            records[fam] = {"members": ts, "split": "calibration", "basis": "forced_calibration",
                            "reason": FORCED_CALIBRATION[forced[0]]}
        else:
            free.append(fam)
    order = sorted(free)
    random.Random(SPLIT_SEED).shuffle(order)
    for position, fam in enumerate(order):
        records[fam] = {"members": members[fam], "split": "calibration" if position % 2 == 0 else "held_out",
                        "basis": "seeded_parity", "seeded_position": position}
    for fam, rec in records.items():
        key = fam.split(":", 1)[1]
        if key in SHARED_REFERENCE_GROUPS:
            rec["grouping_reason"] = SHARED_REFERENCE_GROUPS[key]["reason"]
        rec["pre_pivot_mined_members"] = [t for t in rec["members"] if t in PRE_PIVOT_MINED]
    return dict(sorted(records.items()))


def task_split(task: str, split: dict) -> str:
    return split[family_id(task)]["split"]


def git(*args: str, checkout: Path = CHECKOUT) -> bytes:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True).stdout


def verify_checkout(checkout: Path = CHECKOUT) -> dict:
    head = git("rev-parse", "HEAD", checkout=checkout).decode().strip()
    if head != SOURCE_REVISION:
        raise ValueError(f"Terminal-Bench checkout at {head}, expected {SOURCE_REVISION}")
    if git("status", "--porcelain", "--untracked-files=no", checkout=checkout).strip():
        raise ValueError("Terminal-Bench checkout has local changes")
    tasks = task_ids(checkout)
    if len(tasks) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} tasks, found {len(tasks)}")
    return {"source_url": SOURCE_URL, "source_revision": head, "task_count": len(tasks)}


def content_tree(task: str, revision: str, checkout: Path = CHECKOUT) -> dict[str, str]:
    """{path: blob sha} of a task at a revision, excluding metadata-only files."""
    out = {}
    for line in git("ls-tree", "-r", revision, "--", task + "/", checkout=checkout).decode().splitlines():
        meta, path = line.split("\t", 1)
        rel = path[len(task) + 1:]
        if rel in CONTENT_EXCLUDES:
            continue
        out[rel] = meta.split()[2]
    return out


def registry_identity(task: str, checkout: Path = CHECKOUT) -> dict:
    here, registry = content_tree(task, SOURCE_REVISION, checkout), content_tree(task, REGISTRY_REVISION, checkout)
    return {"registry_dataset": REGISTRY_DATASET, "registry_revision": REGISTRY_REVISION,
            "content_identical_to_registry": here == registry,
            "content_differences": sorted(set(here.items()) ^ set(registry.items()))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_reference_scan(checkout: Path = CHECKOUT) -> dict:
    """Mechanical scan behind SHARED_REFERENCE_GROUPS (recorded, not used to assign families)."""
    import collections
    import re
    root = Path(checkout)
    tasks = task_ids(checkout)
    by_hash = collections.defaultdict(set)
    for t in tasks:
        for sub in ("environment", "solution", "tests"):
            for p in sorted((root / t / sub).rglob("*")):
                if p.is_file() and p.stat().st_size > 200 and p.name not in ("Dockerfile", "test.sh"):
                    by_hash[sha256_file(p)].add(t + "/" + p.relative_to(root / t).as_posix())
    shared_files = sorted(sorted(v) for v in by_hash.values() if len({x.split("/")[0] for x in v}) > 1)

    def lines(p: Path) -> set[str]:
        keep = set()
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if len(line) < 25 or line.startswith("#") or "apt-get" in line or ("pip" in line and "install" in line):
                continue
            keep.add(line)
        return keep

    solution_in_env = []
    for a in tasks:
        solve = root / a / "solution/solve.sh"
        if not solve.is_file():
            continue
        wanted = lines(solve)
        for b in tasks:
            dockerfile = root / b / "environment/Dockerfile"
            if a == b or not dockerfile.is_file():
                continue
            text = dockerfile.read_text(errors="replace")
            common = sorted(line for line in wanted if line in text)
            if len(common) >= 2:
                solution_in_env.append({"solution_of": a, "found_in_environment_of": b, "lines": common})
    urls = collections.defaultdict(set)
    for t in tasks:
        for p in (root / t).rglob("*"):
            if p.is_file() and (p.name == "Dockerfile" or p.suffix in (".sh", ".md", ".toml")):
                for m in re.findall(r"https?://[^\s'\"<>)]+", p.read_text(errors="replace")):
                    urls[m].add(t)
    shared_urls = {u: sorted(ts) for u, ts in sorted(urls.items()) if len(ts) > 1}
    return {"shared_files": shared_files, "solution_lines_in_other_environment": solution_in_env,
            "shared_urls": shared_urls,
            "notes": ("git-leak-recovery's generic 'git reflog expire'/'git gc' lines match other Dockerfiles' history "
                      "scrubbing and are not a shared reference; shared package-index/tooling URLs (uv installer, "
                      "download.pytorch.org/whl/cpu, localhost) are not input artefacts.")}


def write_split(out: Path, checkout: Path = CHECKOUT) -> dict:
    ident = verify_checkout(checkout)
    tasks = task_ids(checkout)
    split = family_split(tasks)
    record = {"schema_version": "terminalbench.family-split.v1", **ident,
              "registry": {"dataset": REGISTRY_DATASET, "revision": REGISTRY_REVISION, "url": REGISTRY_URL},
              "rule": SPLIT_RULE, "seed": SPLIT_SEED, "forced_calibration": FORCED_CALIBRATION,
              "mettlebench_inherited": METTLEBENCH_INHERITED,
              "shared_reference_groups": {k: {"members": list(v["members"]), "reason": v["reason"]}
                                          for k, v in SHARED_REFERENCE_GROUPS.items()},
              "shared_reference_scan": shared_reference_scan(checkout),
              "pre_pivot_mined": list(PRE_PIVOT_MINED),
              "families": split,
              "tasks": {t: {"family_id": family_id(t), "split": split[family_id(t)]["split"]} for t in tasks},
              "counts": {s: {"families": sum(r["split"] == s for r in split.values()),
                             "tasks": sum(len(r["members"]) for r in split.values() if r["split"] == s)}
                         for s in ("calibration", "held_out")},
              "outcomes_consulted": "none (no image build, grade or model run in this lane before this record)"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return record
