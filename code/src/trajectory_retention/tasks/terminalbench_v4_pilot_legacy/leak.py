"""Leak scan (leak-scan/v1, terminalbench-v4-pilot-legacy copy): does the starter reveal the reference solution or the hidden tests?

Copied from the terminalbench-v2 lane (lanes never import each other) with TB4's file layout: the hidden test files are
the pytest targets tests/test.sh runs (not always tests/test_outputs.py), and the task README.md (it carries the
author's "Solution explanation") and cheat/ (an adversarial oracle some tasks ship) count as solution material.

Scanned: the natural start W (every text file up to 4 MB, plus every blob of any git repository in it) and the worker
image's whole filesystem (names and hashes). Compared against the private material:
  solution   what the reference run contributes (per file it writes or changes: the lines not in its base, the base
             being the file's start version, or for a new file the starter file it shares most lines with) and every
             solution/ file except solve.sh (its effect is the reference W)
  tests      tests/test_outputs.py, and each test-support file minus the lines the task's own build context
             (environment/) also ships to the worker
Lines and comments carrying the Terminal-Bench canary string are ignored everywhere (every upstream file has them).

Findings:
  locations a path under /tests, /solution or /oracle in the worker image
  hashes    a solution/ file, tests/test_outputs.py, tests/test.sh, or a test-support file the build context does not
            ship, of at least 64 bytes, found byte-for-byte in W or the worker image
  contents  a block copy: at least 3 distinct distinctive lines of one private source found in one starter file (or git
            blob), or one distinctive line of at least 80 characters
  comments  a code comment in W sharing at least 2 distinct 5-word sequences with one private source (sequences that
            also occur in instruction.md do not count)
  copies    a file the reference writes or changes that is byte-identical to a starter file at another path
A distinctive line is at least 30 characters after collapsing whitespace, is not an import line, and does not occur in
instruction.md. Locations and hashes always fail the scan. A contents, comments or copies finding fails the scan unless
it is covered by a recorded review (``leak_review.py``) that judged it not to reveal the solution or the hidden tests, with the
reason (for example, the reference reuses starter code). Starter files sharing a basename with a private file and
comments carrying hint words (bug, fix, hint, answer, solution, should be, ...) are listed but do not decide the scan.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tarfile
from pathlib import Path

from . import environment as E

VERSION = "leak-scan/v1"
SUSPICIOUS_NAMES = ("solve.sh", "test_outputs.py", "test.sh")
IMAGE_PREFIXES = ("/tests/", "/solution/", "/oracle/")
HIDDEN_DEFAULT = ("tests/test_outputs.py", "tests/test.sh")
SOLUTION_LIKE = ("solution", "cheat")
MIN_LINE = 30
LONG_LINE = 80
BLOCK = 3
SHINGLE = 5
MIN_SHARED_SHINGLES = 2
TEXT_LIMIT = 4_000_000
CANARY = re.compile(r"canary", re.I)
HINT = re.compile(r"\b(bug|bugs|fix|fixme|hint|answer|solution|cheat|should be|wrong|expected|hidden test|todo)\b", re.I)
COMMENT_STYLES = {
    "hash": ({".py", ".sh", ".bash", ".rb", ".pl", ".r", ".R", ".yaml", ".yml", ".toml", ".cfg", ".conf", ".ini",
              ".mk", ".cmake", ".jl", ".nim", ".tcl", ".ps1", ".dockerfile"}, ("Makefile", "Dockerfile")),
    "slash": ({".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".js", ".mjs", ".ts", ".tsx", ".java", ".go", ".rs", ".cs",
               ".swift", ".kt", ".scala", ".php", ".css", ".proto", ".sol", ".zig", ".v", ".sv"}, ()),
    "dash": ({".sql", ".lua", ".hs", ".elm", ".ada", ".adb"}, ()),
    "semicolon": ({".lisp", ".scm", ".ss", ".el", ".clj", ".asm", ".s", ".red"}, ()),
    "percent": ({".tex", ".m", ".erl", ".pro"}, ()),
}


def norm(line: str) -> str:
    return " ".join(line.split())


def distinctive(line: str, instruction: set[str]) -> bool:
    text = norm(line)
    return (len(text) >= MIN_LINE and not text.startswith(("import ", "from ", "#include", "using ", "require("))
            and text not in instruction and sum(ch.isalnum() for ch in text) >= 12 and not CANARY.search(text))


def read_text(path: Path) -> str | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > TEXT_LIMIT:
            return None
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            return None
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def git_objects(workspace: Path) -> dict[str, str]:
    """Text of every blob in every git repository inside W (history can hold what the work tree does not)."""
    found = {}
    for git_dir in sorted(p for p in workspace.rglob(".git") if p.is_dir()):
        repo = git_dir.parent
        label = repo.relative_to(workspace).as_posix() or "."
        proc = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch"],
                              capture_output=True, timeout=600)
        if proc.returncode:
            found[f"{label}:git:<error>"] = ""
            continue
        data, pos = proc.stdout, 0
        while pos < len(data):
            end = data.index(b"\n", pos)
            sha, kind, size = data[pos:end].decode().split()
            body = data[end + 1:end + 1 + int(size)]
            pos = end + 2 + int(size)
            if kind == "blob" and b"\0" not in body[:8192]:
                try:
                    found[f"{label}:git:{sha}"] = body.decode("utf-8")
                except UnicodeDecodeError:
                    pass
    return found


def starter_texts(workspace: Path) -> dict[str, str]:
    texts = {}
    for path in sorted(workspace.rglob("*")):
        if ".git" in path.relative_to(workspace).parts:
            continue
        text = read_text(path)
        if text is not None:
            texts[path.relative_to(workspace).as_posix()] = text
    texts.update(git_objects(workspace))
    return texts


def reference_new_lines(start: Path, reference: Path) -> dict[str, list[str]]:
    """Per reference-W file the reference wrote or changed: its contribution, i.e. the lines not in its base. The base of
    a changed file is its start version; the base of a new file is the starter file sharing the most lines with it (a
    file derived from a starter file, such as a tuned copy, contributes only what it changes)."""
    starter = {}
    for path in sorted(start.rglob("*")):
        if ".git" not in path.relative_to(start).parts and (text := read_text(path)) is not None:
            starter[path.relative_to(start).as_posix()] = {norm(x) for x in text.splitlines()}
    out = {}
    for path in sorted(reference.rglob("*")):
        rel = path.relative_to(reference)
        if ".git" in rel.parts or (text := read_text(path)) is None:
            continue
        lines = {norm(x) for x in text.splitlines() if norm(x)}
        if rel.as_posix() in starter:
            if starter[rel.as_posix()] == lines:
                continue
            base = starter[rel.as_posix()]
        else:
            base = max(starter.values(), key=lambda s: len(s & lines), default=set())
        new = [x for x in text.splitlines() if norm(x) and norm(x) not in base]
        if new:
            out["reference-W/" + rel.as_posix()] = new
    return out


def reference_copies(start: Path, reference: Path) -> list[dict]:
    """Files the reference wrote or changed that are byte-identical to a starter file at another path."""
    starter = {}
    for path in sorted(start.rglob("*")):
        if path.is_file() and not path.is_symlink() and ".git" not in path.relative_to(start).parts:
            starter.setdefault(sha256_path(path), []).append(path.relative_to(start).as_posix())
    out = []
    for path in sorted(reference.rglob("*")):
        rel = path.relative_to(reference).as_posix()
        if not path.is_file() or path.is_symlink() or ".git" in Path(rel).parts or path.stat().st_size < 64:
            continue
        before = start / rel
        if before.is_file() and not before.is_symlink() and sha256_path(before) == sha256_path(path):
            continue
        others = [p for p in starter.get(sha256_path(path), []) if p != rel]
        if others:
            out.append({"reference_file": rel, "identical_starter_files": others})
    return out


def comments(name: str, text: str) -> list[str]:
    suffix, base = Path(name).suffix, Path(name).name
    style = next((k for k, (exts, names) in COMMENT_STYLES.items() if suffix in exts or base in names), None)
    if style is None:
        return []
    if style == "slash":
        found = re.findall(r"//([^\n]*)", text) + [c for block in re.findall(r"/\*(.*?)\*/", text, re.S)
                                                   for c in block.splitlines()]
    else:
        marker = {"hash": "#", "dash": "--", "semicolon": ";", "percent": "%"}[style]
        found = [line.split(marker, 1)[1] for line in text.splitlines()
                 if marker in line and not line.lstrip().startswith("#!")]
    out = [norm(c.strip(" */#;%-")) for c in found]
    return [c for c in out if len(c) >= 12 and not CANARY.search(c)]


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return {tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def scan_image(image_id: str, hashes: dict[str, str]) -> dict:
    """Hash every regular file of the image's flattened filesystem (what a container can read)."""
    cid = E.run(["docker", "create", "--label", E.LABEL, image_id, "true"]).stdout.decode().strip()
    hits, names, locations, files = [], [], [], 0
    try:
        proc = subprocess.Popen(["docker", "export", cid], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                name = "/" + member.name.lstrip("./")
                if member.isreg():
                    files += 1
                    if member.size >= 64:
                        digest = hashlib.sha256()
                        handle = tf.extractfile(member)
                        for block in iter(lambda: handle.read(1 << 20), b""):
                            digest.update(block)
                        if digest.hexdigest() in hashes:
                            hits.append({"path": name, "matches": hashes[digest.hexdigest()]})
                if name.rsplit("/", 1)[-1] in SUSPICIOUS_NAMES:
                    names.append(name)
                if name.startswith(IMAGE_PREFIXES):
                    locations.append(name)
        proc.wait(timeout=900)
    finally:
        E.run(["docker", "rm", "-f", cid], check=False)
    return {"image_id": image_id, "files_scanned": files, "hash_hits": hits, "names": sorted(set(names)),
            "locations": sorted(set(locations))}


def sources(private: Path, start: Path, reference: Path, instruction: str, hidden: tuple[str, ...] = HIDDEN_DEFAULT) -> dict:
    """Private hashes and distinctive-line pools (see the module rule)."""
    instruction_lines = {norm(x) for x in instruction.splitlines()}
    shipped_hashes, shipped_lines = {}, set()
    for path in sorted((private / "environment").rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "Dockerfile":
            if path.stat().st_size >= 64:
                shipped_hashes.setdefault(sha256_path(path), "environment/" + path.relative_to(private / "environment").as_posix())
            shipped_lines |= {norm(x) for x in (read_text(path) or "").splitlines()}
    hashes, pools, support_shipped = {}, {}, []
    for sub in (*SOLUTION_LIKE, "tests"):
        for path in sorted((private / sub).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            label = f"{sub}/{path.relative_to(private / sub).as_posix()}"
            digest, text = sha256_path(path), read_text(path)
            support = sub == "tests" and label not in hidden
            if support and digest in shipped_hashes:
                support_shipped.append(label)
            elif path.stat().st_size >= 64:
                hashes[digest] = label
            if text is None or label in ("solution/solve.sh", "tests/test.sh"):
                continue
            lines = {norm(x) for x in text.splitlines() if distinctive(x, instruction_lines)}
            pools[label] = lines - shipped_lines if support else lines
    readme = private / "README.md"
    if readme.is_file() and (text := read_text(readme)) is not None:
        pools["README.md"] = {norm(x) for x in text.splitlines() if distinctive(x, instruction_lines)}
    for label, lines in reference_new_lines(start, reference).items():
        pools[label] = {norm(x) for x in lines if distinctive(x, instruction_lines)}
    return {"hashes": hashes, "pools": pools, "shipped_hashes": shipped_hashes, "support_shipped": support_shipped}


def scan(*, task: str, private: Path, instruction: str, start: Path, reference: Path, worker_image_id: str,
         reviews: dict[str, dict] | None = None, hidden: tuple[str, ...] = HIDDEN_DEFAULT) -> dict:
    reviews = reviews or {}
    src = sources(private, start, reference, instruction, hidden)
    hashes, pools, shipped_hashes = src["hashes"], src["pools"], src["shipped_hashes"]
    instruction_grams = shingles(instruction)
    texts = starter_texts(start)
    private_names = {Path(p).name for p in hashes.values()} | set(SUSPICIOUS_NAMES)
    names = sorted("/app/" + p.relative_to(start).as_posix() for p in start.rglob("*") if p.name in private_names)
    hash_hits, env_exposures = [], []
    for path in sorted(start.rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest = sha256_path(path)
            if digest in hashes:
                hash_hits.append({"path": "/app/" + path.relative_to(start).as_posix(), "matches": hashes[digest]})
            elif digest in shipped_hashes:
                env_exposures.append({"path": "/app/" + path.relative_to(start).as_posix(), "matches": shipped_hashes[digest]})
    image = scan_image(worker_image_id, hashes)
    findings = []
    starter_lines = {name: {norm(x) for x in text.splitlines()} for name, text in texts.items()}
    for label, pool in pools.items():
        for name, lines in starter_lines.items():
            shared = sorted(pool & lines)
            if len(shared) >= BLOCK or any(len(x) >= LONG_LINE for x in shared):
                findings.append({"kind": "contents", "source": label, "starter_file": name, "lines": len(shared),
                                 "longest": max(len(x) for x in shared),
                                 "id": fingerprint(task, "contents", label, name, *shared)})
    for copy in reference_copies(start, reference):
        findings.append({"kind": "copies", "source": "reference-W/" + copy["reference_file"],
                         "starter_file": copy["identical_starter_files"][0],
                         "id": fingerprint(task, "copies", copy["reference_file"], *copy["identical_starter_files"])})
    grams = {label: set().union(*(shingles(x) for x in pool)) - instruction_grams if pool else set()
             for label, pool in pools.items()}
    hint_flags = []
    for name, text in texts.items():
        for comment in comments(name.split(":git:")[0], text):
            mine = shingles(comment)
            for label, pool in grams.items():
                shared = mine & pool
                if len(shared) >= MIN_SHARED_SHINGLES:
                    findings.append({"kind": "comments", "source": label, "starter_file": name,
                                     "shared_5grams": len(shared), "id": fingerprint(task, "comments", label, name, comment)})
            if HINT.search(comment):
                hint_flags.append({"starter_file": name, "comment": comment[:200]})
    for f in findings:
        f["review"] = reviews.get(f["id"])
        f["unresolved"] = f["review"] is None or f["review"].get("verdict") != "not_a_leak"
    hard = {"locations": image["locations"], "hashes": hash_hits + image["hash_hits"]}
    unresolved = [f for f in findings if f["unresolved"]]
    failed_on = sorted([k for k, v in hard.items() if v] + sorted({f["kind"] for f in unresolved}))
    return {"version": VERSION, "rule": __doc__, "passed": not failed_on, "failed_on": failed_on, **hard,
            "findings": findings, "unresolved_findings": len(unresolved),
            "reviewed_findings": sum(1 for f in findings if f["review"] is not None),
            "names": names, "worker_image_names": image["names"], "test_support_shipped_by_task": src["support_shipped"],
            "private_sources_compared": sorted(pools), "distinctive_lines_compared": sum(len(p) for p in pools.values()),
            "starter_files_scanned": len(texts), "git_objects_scanned": sum(":git:" in k for k in texts),
            "worker_image_files_scanned": image["files_scanned"], "environment_exposures": env_exposures,
            "comment_hint_flags": hint_flags[:200], "comment_hint_flag_count": len(hint_flags)}


def explain(result: dict, *, private: Path, start: Path, reference: Path, instruction: str,
            hidden: tuple[str, ...] = HIDDEN_DEFAULT) -> list[dict]:
    """Reviewer view of every finding: the shared lines or the matching comments (private; never committed)."""
    src = sources(private, start, reference, instruction, hidden)
    texts = starter_texts(start)
    instruction_grams = shingles(instruction)
    out = []
    for f in result["findings"]:
        pool = src["pools"].get(f["source"], set())
        starter = texts.get(f["starter_file"], "")
        if f["kind"] == "contents":
            out.append({**f, "shared_lines": sorted(pool & {norm(x) for x in starter.splitlines()})[:20]})
        elif f["kind"] == "copies":
            out.append(f)
        else:
            grams = set().union(*(shingles(x) for x in pool)) - instruction_grams if pool else set()
            hit = [c for c in comments(f["starter_file"].split(":git:")[0], starter)
                   if len(shingles(c) & grams) >= MIN_SHARED_SHINGLES]
            out.append({**f, "comments": hit[:10]})
    return out
