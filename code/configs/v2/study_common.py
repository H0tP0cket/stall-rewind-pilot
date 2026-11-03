"""Shared pieces of the v2 study operations (track S): arm names, sizes, paths, seeds, stall and branch records.

Every other configs/v2 study script imports this module. It never calls a model and never touches Docker.
Rules that the user has not ratified are named constants with status "DRAFT"; anything built on them records that.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

SPEC_VERSION = "branch-spec.v2"
SPECSET_VERSION = "specset.v2"
ARMS = ("1_continue", "2_ledger_review", "3_context_review", "4_forced_rewind", "5_rewind_own_summary",
        "6_rewind_ledger_recipe")
REVIEW_VIEWS = {"2_ledger_review": "review_view_ledger.json", "3_context_review": "review_view_context.json"}
REWIND_ARMS = ("4_forced_rewind", "5_rewind_own_summary", "6_rewind_ledger_recipe")
SCREEN_ARMS = ("1_continue", "4_forced_rewind")
TUNING_ARMS = ("2_ledger_review", "3_context_review", "5_rewind_own_summary", "6_rewind_ledger_recipe")
HARM_ARM = "2_ledger_review"
INJECTION_CONDITIONS = ("continue", "false_success_claim", "failed_approach_promising",
                        "failed_attempt_failure_shown", "stale_output", "workspace_debris", "small_neutral",
                        "large_neutral")
PURPOSES = ("screen", "tuning", "main", "injection", "harm")
MAIN_REPEATS, SCREEN_REPEATS, HARM_REPEATS, INJECTION_REPEATS = 10, 5, 4, 4
GUARD = {"cost_limit_usd": 3.0, "step_limit": 1_000_000}  # user 2026-10-12: the guard only stops runaway spending, never active work; no step limit (unreachable)
MAX_STALLS_PER_TASK = 3
MAX_HEALTHY_PER_TASK = 2
END_REASONS = ("passing_submission", "identical_failures", "no_improvement", "submission_cap", "token_cap", "cost_guard", "step_guard",
               "worker_ended", "error")  # worker_ended: the worker ended the branch itself (an unsolved outcome)
PROMPT_VERSIONS = {"2_ledger_review": "review-ledger.v0-draft", "3_context_review": "review-context.v0-draft",
                   "5_rewind_own_summary": "note.v0-draft"}

# Arm set v3 (user, 2026-10-16/17), built beside the set above (branch-spec.v3 in experiments/branch_v2.py):
# 1 continue; 2 Ledger review and 3 context review (the review turn as before, but its request is the system and task
# messages plus the view, and its own output stays in the context); 4 forced rewind of W, R and the context to P, the
# judge's stall onset point (a per-stall input); 5 Ledger read-only and 6 context read-only (one turn showing the view
# with no tool, its output kept, then continue). The Ledger view of arms 2 and 5 is the flat Ledger (no approach layer).
SPEC_VERSION_V3 = "branch-spec.v3"
ARMS_V3 = ("1_continue", "2_ledger_review", "3_context_review", "4_forced_rewind_to_p", "5_ledger_read_only",
           "6_context_read_only")
REVIEW_VIEWS_V3 = {"2_ledger_review": "review_view_ledger_flat.json", "3_context_review": "review_view_context.json",
                   "5_ledger_read_only": "review_view_ledger_flat.json",
                   "6_context_read_only": "review_view_context.json"}
READ_ONLY_ARMS_V3 = ("5_ledger_read_only", "6_context_read_only")
PROMPT_VERSIONS_V3 = {"2_ledger_review": "review-ledger.v1-draft", "3_context_review": "review-context.v1-draft",
                      "5_ledger_read_only": "readonly-ledger.v1-draft", "6_context_read_only": "readonly-context.v1-draft"}

# Worker-visible text. Track E owns trajectory_retention.experiments.v2_text; until it is merged, the provisional
# wording from brief-interfaces.md is the single fallback copy (it is never used when v2_text exists).
_PROVISIONAL_TEXT = {
    "SUBMIT_RULE": "To submit, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone.",
    "SUBMISSION_FEEDBACK": "Hidden tests still failing: {n}.",
    "REVIEW_PROMPT": "You can review your work and rewind or drop any part of it, or change nothing.",
    "GENERIC_LINE": "Part of your earlier work was undone or removed from view.",
    "HANDBACK_HEADER": "Notes carried over:",
    "GRADER_RECORD_LINE": "Submission {i} (step {k}): {n} hidden tests failing.",
    "OUTPUT_PLACEHOLDER": "[output removed]",
    "READ_ONLY_PROMPT": "You can review your work. Nothing can be changed in this turn.",
}

# No worker-visible text may say or imply "you seem stuck" (contract, shared rules). Checked on every prompt file.
FORBIDDEN_WORKER_PHRASES = (r"\bstuck\b", r"\bstall", r"\bstruggl", r"not making progress", r"no progress",
                            r"going (?:round|around) in circles", r"\bin circles\b", r"\bloop(?:ing)?\b",
                            r"\bspinning\b", r"\bunproductive\b", r"\bwasted\b", r"\bgive up\b")


def worker_text() -> tuple[dict[str, str], str]:
    """The worker-visible constants and where they came from ("v2_text" or "provisional-brief")."""
    try:
        from trajectory_retention.experiments import v2_text  # type: ignore
    except ImportError:
        return dict(_PROVISIONAL_TEXT), "provisional-brief"
    return {k: getattr(v2_text, k, v) for k, v in _PROVISIONAL_TEXT.items()}, "v2_text"


def lint_worker_text(text: str) -> list[str]:
    return [m.group(0) for rx in FORBIDDEN_WORKER_PHRASES for m in re.finditer(rx, text, re.I)]


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_json(path: str | Path, default: Any = ...) -> Any:
    p = Path(path)
    if default is not ... and not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path: str | Path, obj: Any, *, indent: int | None = 1) -> Path:
    """Atomic write (temp file in the same directory, then rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=p.parent)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(obj, indent=indent, default=str, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
    return p


@contextmanager
def file_lock(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def safe(name: str) -> str:
    """A path-safe form of an id ("/" -> "__", ":" -> "~", anything else odd -> "_")."""
    return re.sub(r"[^A-Za-z0-9._~-]", "_", name.replace("/", "__").replace(":", "~"))


def short_hash(text: str, n: int = 10) -> str:
    return sha256_bytes(text.encode())[:n]


def stall_id_for(task_id: str, run_id: str, step_index: int) -> str:
    return f"{safe(task_id)}--{short_hash(run_id)}--p{step_index}"


def branch_id(stall_id: str, arm: str, repeat: int) -> str:
    return f"{stall_id}/{arm}/{repeat}"


def seed_for(purpose: str, prefix: str, bid: str) -> int:
    """Deterministic, per branch; the purpose and set prefix are in the hash, so screen, tuning and arm seeds differ."""
    return int(sha256_bytes(f"v2-seed|{purpose}|{prefix}|{bid}".encode())[:8], 16) & 0x7FFFFFFF


def register_seeds(study_root: Path, entries: Iterable[tuple[int, str]]) -> None:
    """Refuse a seed already registered to a different branch (fresh seeds are never reused across sets)."""
    reg_path = study_root / "seed-registry.json"
    with file_lock(study_root / ".seed-registry.lock"):
        reg = read_json(reg_path, {})
        clashes = []
        for seed, owner in entries:
            prior = reg.get(str(seed))
            if prior is not None and prior != owner:
                clashes.append((seed, prior, owner))
            reg[str(seed)] = owner
        if clashes:
            raise SystemExit(f"seed reuse refused: {clashes[:3]}")
        write_json(reg_path, reg, indent=None)


# ---------- stalls and healthy checkpoints ----------

def load_stall(stall_dir: str | Path) -> dict:
    """stall.json plus the paths of the Ledger outputs that exist (missing ones stay missing)."""
    d = Path(stall_dir)
    s = read_json(d / "stall.json")
    s["dir"] = str(d.resolve())
    for name in ("boundary.json", "ledger.json", "recipe.json", *REVIEW_VIEWS.values(), *REVIEW_VIEWS_V3.values(),
                 "injection_donors.json"):
        s[f"has_{name.removesuffix('.json')}"] = (d / name).exists()
    return s


def stall_dirs(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/stall.json"))


def healthy_dirs(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/healthy.json"))


def view_units(view_path: str | Path) -> dict:
    v = read_json(view_path)
    if not isinstance(v.get("text"), str) or not isinstance(v.get("units"), dict) or not v["units"]:
        raise ValueError(f"review view without text or units: {view_path}")
    return v["units"]


# ---------- spec sets and branch results ----------

def specset_path(study_root: Path, prefix: str) -> Path:
    return study_root / "specsets" / f"{prefix}.json"


def runs_dir(study_root: Path, prefix: str) -> Path:
    return study_root / "runs" / prefix


def load_specset(path: str | Path) -> dict:
    m = read_json(path)
    if m.get("version") != SPECSET_VERSION:
        raise SystemExit(f"not a {SPECSET_VERSION} manifest: {path}")
    m["_path"] = str(Path(path).resolve())
    return m


def spec_of(manifest: dict, job: dict) -> dict:
    return read_json(Path(manifest["_path"]).parent / job["spec"])


def attempt_dirs(branch_dir: Path) -> list[Path]:
    return sorted((p for p in branch_dir.glob("a*") if p.is_dir() and p.name[1:].isdigit()),
                  key=lambda p: int(p.name[1:]))


def latest_result(branch_dir: Path) -> tuple[dict | None, list[dict]]:
    """(latest attempt result that is not an error, every attempt's result or failure). Errors are kept, not used."""
    attempts = []
    for a in attempt_dirs(branch_dir):
        r = read_json(a / "result.json", None)
        if r is None:
            f = read_json(a / "runner-error.json", None)
            attempts.append({"attempt": a.name, "status": "runner_failed" if f else "no_result", "error": f})
        else:
            attempts.append({"attempt": a.name, "status": "error" if r.get("end_reason") == "error" else "done",
                             "result": r, "dir": str(a)})
    good = [x for x in attempts if x["status"] == "done"]
    return (good[-1]["result"] | {"_dir": good[-1]["dir"]} if good else None), attempts


def branch_dir(out: Path, bid: str) -> Path:
    return out / "branches" / safe(bid)


def iter_results(manifest: dict, out: Path) -> Iterable[tuple[dict, dict, dict | None, list[dict]]]:
    """(block, job, latest good result or None, attempts) for every job of a spec set, in schedule order."""
    for block in manifest["blocks"]:
        for job in block["jobs"]:
            res, attempts = latest_result(branch_dir(out, job["branch_id"]))
            yield block, job, res, attempts


def draft_rules(*rules: dict) -> list[str]:
    return [r["name"] for r in rules if str(r.get("status", "")).upper().startswith("DRAFT")]


# ---------- review decisions (record-only validation; track E's validator is authoritative for branches) ----------

def validate_decision(args: Any, mode: str, units: dict) -> list[str]:
    """Problems with one review_decision call under the interface rules ([] = valid)."""
    errs: list[str] = []
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    want = {"drop", "keep", "change_nothing", "rewind" if mode == "per_channel" else "rewind_to"}
    if set(args) != want:
        errs.append(f"keys {sorted(args)} != {sorted(want)}")
    steps = [s for u in units.values() for s in (u.get("steps") or [])]
    last = max(steps) if steps else 0
    point = lambda v: v is None or (isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= last)
    if mode == "per_channel":
        rw = args.get("rewind")
        if rw is not None:
            if not isinstance(rw, dict) or set(rw) != {"workspace", "context", "runtime", "outputs"}:
                errs.append("rewind must be null or have exactly workspace, context, runtime, outputs")
                rw = None
            elif not all(point(v) for v in rw.values()):
                errs.append(f"rewind points must be null or integers 0..{last}")
        context_point = (rw or {}).get("context")
        rewinds = bool(rw) and any(v is not None for v in rw.values())
    else:
        context_point = args.get("rewind_to")
        if not point(context_point):
            errs.append(f"rewind_to must be null or an integer 0..{last}")
            context_point = None
        rewinds = context_point is not None
    drop, keep = args.get("drop"), args.get("keep")
    for name, ids in (("drop", drop), ("keep", keep)):
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            errs.append(f"{name} must be a list of unit IDs")
        else:
            unknown = [i for i in ids if i not in units]
            if unknown:
                errs.append(f"{name} names unknown units {unknown[:5]}")
    if not isinstance(args.get("change_nothing"), bool):
        errs.append("change_nothing must be true or false")
    elif args["change_nothing"] and (rewinds or drop or keep):
        errs.append("change_nothing is true but the call also rewinds, drops or keeps")
    if isinstance(keep, list) and keep:
        if context_point is None:
            errs.append("keep names units but the context is not rewound")
        else:
            early = [i for i in keep if i in units and min(units[i].get("steps") or [0]) <= context_point]
            if early:
                errs.append(f"keep may name only units from undone steps; not undone: {early[:5]}")
    if isinstance(drop, list) and isinstance(keep, list) and set(drop) & set(keep):
        errs.append("a unit is both dropped and kept")
    return errs
