"""Cheap static screen of all 89 Terminal-Bench 2 tasks, before any image build (no Docker, no model calls).

Exclusion rules applied here (every other gate item needs a build):
  hidden_tests  fewer than 3 test functions in tests/test_outputs.py (literal ``parametrize`` lists expand; a
                non-literal parametrize leaves the count unknown and the task goes on to collection in the grader image)
  gpu           task.toml asks for a GPU
  emulation     the environment or instruction runs a QEMU virtual machine (an x86 guest under emulation)
  network       the instruction tells the worker to fetch a remote resource (a non-local URL on a line that also says
                clone/download/fetch/install/wget/curl/pull)
  grader        tests/test.sh is outside the native template the prebuilt offline verifier reproduces (held, not excluded)
Recorded but not deciding: network commands in solution/solve.sh (the dynamic rule decides: the reference is run with
the network off), extra test or solution files, and published per-task results used only to order the builds.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import re
import tomllib

from . import environment as E
from . import source as S

MIN_HIDDEN_TESTS = 3
EXTERNAL_RESULTS = Path(__file__).resolve().parents[4] / (
    "benchmark_tasks/admission/stall-supply-v1/scouting/stall-evidence-sources/tb2_harbor_2026_per_task.csv")
SOLVE_NETWORK = re.compile(
    r"\b(apt-get (?:update|install|source)|apt (?:update|install)|pip3? install|uv (?:pip install|add|sync|venv)|uvx |"
    r"curl |wget |git clone|npm (?:i|install|ci)|cargo (?:install|build|fetch)|go (?:get|install|mod download)|"
    r"huggingface|hf_hub|from_pretrained|urlretrieve|requests\.get|conda install|gem install|opam (?:install|init)|"
    r"install\.packages)")
URL = re.compile(r"https?://([^/\s:)\"'>`]+)[^\s)\"'>`]*")
FETCH_VERB = re.compile(r"\b(clone|download|fetch|install|wget|curl|pull)\b", re.I)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "server")
EMULATION = re.compile(r"\bqemu-system|\bqemu\b|\bQEMU\b")


def count_tests(path: Path) -> tuple[int | None, bool]:
    """(collected test count, parametrize seen); None when a parametrize argument list is not a literal."""
    if not path.is_file():
        return 0, False
    tree = ast.parse(path.read_text())
    seen, unknown = False, False

    def walk(body) -> int:
        nonlocal seen, unknown
        total = 0
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                n = 1
                for d in node.decorator_list:
                    if "parametrize" in ast.unparse(d):
                        seen = True
                        if isinstance(d, ast.Call) and len(d.args) >= 2 and isinstance(d.args[1], (ast.List, ast.Tuple)):
                            n *= len(d.args[1].elts)
                        else:
                            unknown = True
                total += n
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                total += walk(node.body)
        return total

    total = walk(tree.body)
    return (None if unknown else total), seen


def remote_fetch_lines(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        hosts = [h for h in URL.findall(line) if not h.startswith(LOCAL_HOSTS)]
        if hosts and FETCH_VERB.search(line):
            hits.append(line.strip()[:300])
    return hits


def external_results() -> dict[str, dict]:
    if not EXTERNAL_RESULTS.is_file():
        return {}
    with EXTERNAL_RESULTS.open() as handle:
        return {row["task"]: row for row in csv.DictReader(handle)}


def screen_task(task: str, checkout: Path = S.CHECKOUT, external: dict | None = None) -> dict:
    root = checkout / task
    toml = tomllib.loads((root / "task.toml").read_text())
    env = toml.get("environment", {})
    tests, parametrized = count_tests(root / "tests/test_outputs.py")
    instruction = (root / "instruction.md").read_text(errors="replace")
    dockerfile = (root / "environment/Dockerfile").read_text(errors="replace") if (root / "environment/Dockerfile").is_file() else ""
    solve = (root / "solution/solve.sh").read_text(errors="replace") if (root / "solution/solve.sh").is_file() else ""
    try:
        test_spec, test_sh_error = E.parse_test_sh((root / "tests/test.sh").read_text()), None
    except (ValueError, OSError) as exc:
        test_spec, test_sh_error = None, str(exc)[:300]
    fetch = remote_fetch_lines(instruction)
    emulation = sorted({m for m in EMULATION.findall(dockerfile + "\n" + instruction)})
    record = {
        "task": task, "task_id": S.package_task_id(task), "family_id": S.family_id(task),
        "hidden_tests_static": tests, "parametrize": parametrized,
        "test_files": sorted(p.name for p in (root / "tests").iterdir()),
        "solution_files": sorted(p.relative_to(root / "solution").as_posix() for p in (root / "solution").rglob("*") if p.is_file()),
        "gpus": env.get("gpus"), "cpus": env.get("cpus"), "memory_mb": env.get("memory_mb"),
        "allow_internet": env.get("allow_internet"),
        "difficulty": toml.get("metadata", {}).get("difficulty"), "category": toml.get("metadata", {}).get("category"),
        "instruction_remote_fetch": fetch, "emulation_mentions": emulation,
        "solve_network_commands": sorted(set(SOLVE_NETWORK.findall(solve))),
        "test_sh_template": test_sh_error is None, "test_sh_error": test_sh_error,
        "grade_time_apt_packages": test_spec["apt_packages"] if test_spec else None,
        "external": {k: v for k, v in (external or {}).get(task, {}).items() if k != "task"} or None,
        "registry_identity": S.registry_identity(task, checkout),
    }
    reasons = []
    if tests is not None and tests < MIN_HIDDEN_TESTS:
        reasons.append(("EXCLUDED", f"hidden_tests: {tests} test function(s) in tests/test_outputs.py; at least "
                                    f"{MIN_HIDDEN_TESTS} required"))
    if (env.get("gpus") or 0) > 0:
        reasons.append(("EXCLUDED", f"gpu: task.toml asks for {env.get('gpus')} GPU(s)"))
    if emulation:
        reasons.append(("EXCLUDED", "emulation: the environment or instruction runs a QEMU virtual machine"))
    if fetch:
        reasons.append(("EXCLUDED", "network: the instruction asks the worker to fetch a remote resource: " + fetch[0][:160]))
    if test_sh_error:
        reasons.append(("HOLD", "grader: tests/test.sh is outside the native template the offline verifier reproduces"))
    if not record["registry_identity"]["content_identical_to_registry"]:
        reasons.append(("HOLD", "source: task content differs from the registry revision 69671fba"))
    record["static_verdict"] = ("EXCLUDED" if any(k == "EXCLUDED" for k, _ in reasons)
                                else "HOLD" if reasons else "CANDIDATE")
    record["static_reasons"] = [r for _, r in reasons]
    return record


def build_priority(row: dict, cached: set[str]) -> tuple:
    """Order candidates for building (never decides admission): v1-cached images first (no new disk), then tasks
    whose published DeepSeek pass rate is nearest the 10-50% band, then more hidden tests, then name."""
    ext = row.get("external") or {}
    try:
        deepseek = float(ext.get("Termi/DeepSeek-"))
    except (TypeError, ValueError):
        deepseek = None
    distance = abs(deepseek - 0.3) if deepseek is not None else 1.0
    return (row["task"] not in cached, round(distance, 3), -(row["hidden_tests_static"] or 0), row["task"])


def screen_all(checkout: Path = S.CHECKOUT) -> dict:
    ident = S.verify_checkout(checkout)
    external = external_results()
    rows = [screen_task(t, checkout, external) for t in S.task_ids(checkout)]
    counts = {v: sum(r["static_verdict"] == v for r in rows) for v in ("CANDIDATE", "HOLD", "EXCLUDED")}
    return {"schema_version": "terminalbench-v2.static-screen.v1", **ident,
            "registry": {"dataset": S.REGISTRY_DATASET, "revision": S.REGISTRY_REVISION},
            "rules": __doc__, "min_hidden_tests": MIN_HIDDEN_TESTS,
            "external_results": {"path": str(EXTERNAL_RESULTS.relative_to(EXTERNAL_RESULTS.parents[5])),
                                 "use": "build order only; never an admission input"},
            "counts": counts, "tasks": rows}


def write(out: Path, checkout: Path = S.CHECKOUT) -> dict:
    record = screen_all(checkout)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
