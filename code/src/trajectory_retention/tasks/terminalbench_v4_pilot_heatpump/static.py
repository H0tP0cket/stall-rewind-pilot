"""Static screen of the ratified TB4 tasks, before any image build (no Docker, no model calls).

Adapted from the terminalbench-v2 lane's static screen (lanes never import each other). Rules applied here (every other
gate item needs a build):
  hidden_tests  fewer than 3 pytest test functions in the files tests/test.sh runs (literal ``parametrize`` lists
                expand; a non-literal parametrize leaves the count to collection in the verifier image)      EXCLUDED
  gpu           task.toml asks for a GPU                                                                         EXCLUDED
  emulation     the environment Dockerfile or the instruction runs QEMU or another CPU emulator                  EXCLUDED
  network       the instruction tells the worker to fetch a remote resource (a non-local URL on a line that also says
                clone/download/fetch/install/wget/curl/pull)                                                     EXCLUDED
  x86           an x86-64 executable or shared library in the agent build context (environment/), including inside a
                zip or tar archive there, with no aarch64 counterpart: on this arm64 Mac it runs only under emulation,
                so the task needs the x86 box                                                                      HOLD
  grader        not Harbor's plain separate verifier with a pytest test.sh (compose sidecars, MCP servers, collect
                hooks, no tests/Dockerfile, or a test.sh that does not run pytest with --ctrf)                      HOLD
Recorded but not deciding: package installs and downloads in solution/solve.sh (the dynamic network rule decides: the
reference runs with the network off first).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

from . import source as S

MIN_HIDDEN_TESTS = 3
SOLVE_INSTALLS = re.compile(
    r"(apt-get (?:update|install)|apt (?:update|install)|pip3? install|\"\$PIP\" install|\$PIP install|uv pip install|"
    r"uv (?:add|sync)|uvx |curl |wget |git clone|npm (?:i|install|ci)\b|cargo (?:install|fetch)|conda install|"
    r"requests\.get|urlretrieve|from_pretrained|snapshot_download)")
URL = re.compile(r"https?://([^/\s:)\"'>`]+)[^\s)\"'>`]*")
FETCH_VERB = re.compile(r"\b(clone|download|fetch|install|wget|curl|pull)\b", re.I)
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "warranty-portal", "asset-ledger", "document-vault", "returns-ledger", "compliance-ledger", "warranty-inbox")
EMULATION = re.compile(r"\bqemu(?:-system|-user|-x86_64|-aarch64)?\b|\bbox64\b|\bFEX-Emu\b|\bbinfmt_misc\b", re.I)
ELF_MACHINE = {0x3E: "x86-64", 0x03: "i386", 0xB7: "aarch64"}
PYTEST_TARGET = re.compile(r"(/tests/[\w./-]+\.py|test_[\w-]+\.py)")


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


def pytest_targets(test_sh: str, tests: Path) -> list[str]:
    """The test files test.sh hands to pytest (paths under /tests, or bare test_*.py names found in tests/)."""
    found = []
    for line in test_sh.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for match in PYTEST_TARGET.findall(line):
            name = Path(match).name
            if (tests / name).is_file() and name.startswith("test") and name not in found:
                found.append(name)
    return found


def remote_fetch_lines(text: str) -> list[str]:
    hits = []
    for line in text.splitlines():
        hosts = [h for h in URL.findall(line) if not h.startswith(LOCAL_HOSTS)]
        if hosts and FETCH_VERB.search(line):
            hits.append(line.strip()[:300])
    return hits


def _elf_machine(head: bytes) -> str | None:
    if head[:4] != b"\x7fELF" or len(head) < 20:
        return None
    machine = int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")
    return ELF_MACHINE.get(machine, f"e_machine {machine}")


def elf_members(root: Path) -> list[dict]:
    """Every ELF file in a build context, including members of zip and tar archives (path, machine)."""
    out = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(root).as_posix()
        with open(path, "rb") as handle:
            head = handle.read(64)
        machine = _elf_machine(head)
        if machine:
            out.append({"path": rel, "machine": machine})
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.file_size >= 20 and not info.is_dir():
                        with zf.open(info) as handle:
                            m = _elf_machine(handle.read(64))
                        if m:
                            out.append({"path": f"{rel}!{info.filename}", "machine": m})
        elif path.suffix in (".tar", ".tgz", ".gz", ".xz", ".bz2") and tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                for member in tf:
                    if member.isreg() and member.size >= 20:
                        m = _elf_machine(tf.extractfile(member).read(64))
                        if m:
                            out.append({"path": f"{rel}!{member.name}", "machine": m})
    return out


def x86_only(elves: list[dict]) -> list[dict]:
    """x86 ELF files with no aarch64 counterpart (same path with x86_64/amd64 replaced by aarch64/arm64)."""
    arm = {e["path"] for e in elves if e["machine"] == "aarch64"}
    out = []
    for e in elves:
        if e["machine"] not in ("x86-64", "i386"):
            continue
        twins = {re.sub(r"x86_64|x86-64|amd64", alt, e["path"]) for alt in ("aarch64", "arm64")}
        if not twins & arm:
            out.append(e)
    return out


def screen_task(task: str) -> dict:
    root = S.task_dir(task)
    toml = tomllib.loads((root / "task.toml").read_text())
    env = toml.get("environment", {})
    verifier = toml.get("verifier", {})
    test_sh = (root / "tests/test.sh").read_text(errors="replace") if (root / "tests/test.sh").is_file() else ""
    targets = pytest_targets(test_sh, root / "tests")
    counts = {t: count_tests(root / "tests" / t) for t in targets}
    tests = None if any(c is None for c, _ in counts.values()) or not counts else sum(c for c, _ in counts.values())
    if task != "heat-pump-warranty":
        raise ValueError("This lane implements only heat-pump-warranty native case grading")
    tests = len(json.loads((root / 'tests/expected.json').read_text())['cases'])
    targets = ['test_scoring.py', 'expected.json']
    instruction = (root / "instruction.md").read_text(errors="replace")
    dockerfile = (root / "environment/Dockerfile").read_text(errors="replace") if (root / "environment/Dockerfile").is_file() else ""
    solve = (root / "solution/solve.sh").read_text(errors="replace") if (root / "solution/solve.sh").is_file() else ""
    fetch = remote_fetch_lines(instruction)
    emulation = sorted(set(EMULATION.findall(dockerfile + "\n" + instruction)))
    elves = elf_members(root / "environment")
    x86 = x86_only(elves)
    grader = []
    if verifier.get("environment_mode") != "separate":
        grader.append("verifier environment_mode is not separate")
    if not (root / "tests/Dockerfile").is_file():
        grader.append("no tests/Dockerfile")
    # The ratified local services have a lane-specific capture/restore implementation.
    if env.get("mcp_servers"):
        grader.append("MCP servers")
    if verifier.get("collect"):
        grader.append("verifier collect hooks")
    if 'python3 "$TEST_DIR/test_scoring.py"' not in test_sh:
        grader.append("unrecognized native case-grader command")
    installs = sorted({m.strip() for m in SOLVE_INSTALLS.findall(solve)})
    record = {
        "task": task, "task_id": S.package_task_id(task), "family_id": S.family_id(task),
        "hidden_tests_static": tests, "pytest_targets": targets,
        "parametrize": any(p for _, p in counts.values()),
        "test_files": sorted(p.relative_to(root / "tests").as_posix() for p in (root / "tests").rglob("*") if p.is_file())[:50],
        "solution_files": sorted(p.relative_to(root / "solution").as_posix() for p in (root / "solution").rglob("*") if p.is_file()),
        "artifacts": toml.get("artifacts"), "gpus": env.get("gpus"), "cpus": env.get("cpus"),
        "memory_mb": env.get("memory_mb"), "allow_internet": env.get("allow_internet", "absent (TB4 default: open internet)"),
        "verifier_mode": verifier.get("environment_mode"),
        "instruction_remote_fetch": fetch, "emulation_mentions": emulation,
        "elf_binaries": elves[:20], "elf_binary_count": len(elves), "x86_only_binaries": x86[:20],
        "x86_only_binary_count": len(x86), "solve_time_installs": installs, "grader_shape": grader,
        "upstream_tree_files": len(S.task_tree(task)),
    }
    reasons = []
    if tests is not None and tests < MIN_HIDDEN_TESTS:
        reasons.append(("EXCLUDED", f"hidden_tests: {tests} pytest test(s) in {', '.join(targets)}; at least "
                                    f"{MIN_HIDDEN_TESTS} required"))
    if (env.get("gpus") or 0) > 0:
        reasons.append(("EXCLUDED", f"gpu: task.toml asks for {env.get('gpus')} GPU(s)"))
    if emulation:
        reasons.append(("EXCLUDED", "emulation: the environment or instruction runs a CPU emulator"))
    if fetch:
        reasons.append(("EXCLUDED", "network: the instruction asks the worker to fetch a remote resource: " + fetch[0][:160]))
    if x86:
        reasons.append(("HOLD", f"x86: {len(x86)} x86-64 executable(s) or librar(ies) in the agent build context with no "
                                f"aarch64 counterpart (for example {x86[0]['path']}); on this arm64 Mac they run only "
                                "under emulation, so the task needs the x86 box"))
    if grader:
        reasons.append(("HOLD", "grader: " + "; ".join(grader)))
    record["static_verdict"] = ("EXCLUDED" if any(k == "EXCLUDED" for k, _ in reasons)
                                else "HOLD" if reasons else "CANDIDATE")
    record["static_reasons"] = [r for _, r in reasons]
    return record


def screen_all() -> dict:
    ident = S.verify_checkout()
    rows = [screen_task(t) for t in S.RATIFIED_TASKS]
    counts = {v: sum(r["static_verdict"] == v for r in rows) for v in ("CANDIDATE", "HOLD", "EXCLUDED")}
    return {"schema_version": "terminalbench-v4-pilot-heatpump.static-screen.v1", **ident, "ratification": S.RATIFICATION,
            "rules": __doc__, "min_hidden_tests": MIN_HIDDEN_TESTS, "counts": counts, "tasks": rows}


def write(out: Path) -> dict:
    record = screen_all()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return record
