"""Private FeatureBench grading: native patch extraction and native level-1 evaluation in disposable containers.

Container A (extract) rebuilds the native masked baseline as a git commit, lays the submitted W over it and takes the
native submission: `git add -A`, binary files unstaged, `git diff --cached <root>`, binary hunks dropped. The baseline
comes from the image, not from W's own .git, so a rewritten history cannot change the patch.
Container B (evaluate) is harness/runtime.py::run_instance_level1: fresh masked baseline, apply the agent patch,
(added) revert guarded pytest-hook files, delete and restore the F2P file via reverse test_patch, run pytest on the F2P
file and on each P2P file. Both containers use the pinned original image, run as root without network, and are never
the worker's container.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
import uuid

from . import source as S

LABEL = "trajectory-retention.lane=featurebench"


def _baseline(f2p: str, *, strict: bool) -> str:
    mask = ("git apply --whitespace=fix /tmp/grade/mask.patch" if strict else
            "git apply --whitespace=fix /tmp/grade/mask.patch; echo MASK_EXIT=$? > /tmp/grade/out/mask.txt")
    return (f"{S.CONDA}\n" + ("set -e\n" if strict else "") +
            "rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/\n"
            f"cd /testbed && {mask}\n"
            f"rm -f /testbed/{shlex.quote(f2p)}\n")


GIT_INIT = ("cd /testbed && rm -rf .git && git init -q && git config user.email fb@bench.com && "
            "git config user.name FeatureBench && git add -A && "
            "git commit -q -m 'Initial commit for FeatureBench evaluation' --allow-empty\n")
CLEAR_BYTECODE = ("cd /testbed && find . -type d -name __pycache__ -prune -exec rm -rf -- {} + && "
                  "find . -type f \\( -name '*.pyc' -o -name '*.pyo' \\) -delete\n")
UNSTAGE_BINARY = ("cd /testbed && git diff --cached --numstat --no-renames --diff-filter=ACMRTD | "
                  "awk -F '\\t' '$1==\"-\" || $2==\"-\" {print $3}' | "
                  "while IFS= read -r f; do git reset HEAD -- \"$f\" >/dev/null 2>&1 || true; done\n")


def extract_script(f2p: str) -> str:
    return (_baseline(f2p, strict=True) + CLEAR_BYTECODE + GIT_INIT +
            "find /testbed -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf -- {} +\n"
            "tar -xf /tmp/grade/w.tar -C /testbed\n"
            "cd /testbed && find . -mindepth 2 -type d -name .git -prune -exec rm -rf -- {} +\n"
            "git add -A\n" + UNSTAGE_BINARY +
            "git diff --no-color --cached \"$(git rev-list --max-parents=0 HEAD | head -1)\" > /tmp/grade/agent.patch\n")


def evaluate_script(f2p: str, p2p: list[str], test_cmd: str, timeout_run: int) -> str:
    guard = "|".join(S.GUARDED_NAMES) + "|*.pth"
    lines = [_baseline(f2p, strict=False), GIT_INIT,
             "if [ ! -s /tmp/grade/agent.patch ] || [ -z \"$(tr -d '[:space:]' < /tmp/grade/agent.patch)\" ]; then "
             "echo empty_patch > /tmp/grade/out/error.txt; exit 0; fi\n",
             "git apply --whitespace=fix --verbose /tmp/grade/agent.patch > /tmp/grade/out/apply.txt 2>&1 || "
             "{ echo patch_apply_failed > /tmp/grade/out/error.txt; exit 0; }\n",
             "git add -A\n",
             "git diff --cached --name-only --no-renames HEAD | while IFS= read -r f; do\n"
             f"  case \"$(basename -- \"$f\")\" in {guard})\n"
             "    if git cat-file -e \"HEAD:$f\" 2>/dev/null; then git checkout -q HEAD -- \"$f\"; "
             "else git rm -q --cached -f -- \"$f\" >/dev/null; rm -f -- \"$f\"; fi\n"
             "    echo \"$f\" >> /tmp/grade/out/guarded.txt;;\n  esac\ndone\n",
             f"rm -f /testbed/{shlex.quote(f2p)}\n",
             "git apply --reverse --whitespace=fix /tmp/grade/test_patch.diff > /tmp/grade/out/restore.txt 2>&1 || "
             "echo restore_failed >> /tmp/grade/out/restore.txt\n"]
    for i, path in enumerate([f2p, *p2p]):
        name = "f2p" if i == 0 else f"p2p{i}"
        lines.append(f"cd /testbed && timeout -s KILL {timeout_run} {test_cmd} /testbed/{shlex.quote(path)} "
                     f"> /tmp/grade/out/{name}.txt 2>&1; echo $? > /tmp/grade/out/{name}.exit\n")
    return "".join(lines)


def remove_binary_diffs(patch: str) -> str:
    """infer/runtime.py::_remove_binary_diffs."""
    if not patch:
        return ""
    out, skip = [], False
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            skip = False
        if "Binary files" in line or "GIT binary patch" in line:
            while out and not out[-1].startswith("diff --git"):
                out.pop()
            if out and out[-1].startswith("diff --git"):
                out.pop()
            skip = True
            continue
        if not skip:
            out.append(line)
    cleaned = "\n".join(out)
    return cleaned + "\n" if patch.endswith("\n") and cleaned and not cleaned.endswith("\n") else cleaned


def parse_log_pytest(log: str) -> dict[str, str]:
    """harness/test_parsers.py::parse_log_pytest (the parser used for every lane task)."""
    status = {}
    lines = (log or "").split("\n")
    start = next((i + 1 for i, line in enumerate(lines) if "short test summary info" in line.lower()), None)
    if start is None:
        return status
    for raw in lines[start:]:
        line = raw.strip()
        if line.startswith("==="):
            break
        for word in ("PASSED", "FAILED", "ERROR", "XFAIL"):
            if line.startswith(word + " "):
                m = re.match(word + r"\s+(.+?)(?:\s+-\s+.*)?$", line)
                if m:
                    status[m.group(1).strip()] = word
        if line.startswith("SKIPPED "):
            m = re.match(r"SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$", line)
            if m:
                status[m.group(1).strip()] = "SKIPPED"
    return status


def group_of(node_id: str) -> str | None:
    parts = node_id.split("::")
    return parts[1] if len(parts) >= 3 else None


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def workspace_tar(workspace: Path) -> bytes:
    """Submitted W without its own top-level .git (the baseline is rebuilt from the image)."""
    buf = io.BytesIO()
    root = Path(workspace)
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            info = tf.gettarinfo(str(path), arcname="./" + rel)
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            if info.isreg():
                with open(path, "rb") as handle:
                    tf.addfile(info, handle)
            else:
                tf.addfile(info)
    return buf.getvalue()


class _Container:
    def __init__(self, image_id: str):
        self.name = f"tr-featurebench-grade-{uuid.uuid4().hex[:10]}"
        proc = subprocess.run(["docker", "run", "-d", "--name", self.name, "--label", LABEL, "--network", "none",
                               "--user", "0:0", "--entrypoint", "tail", image_id, "-f", "/dev/null"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("grading container: " + proc.stderr.strip()[-500:])

    def put(self, files: dict[str, bytes]):
        subprocess.run(["docker", "exec", self.name, "mkdir", "-p", "/tmp/grade/out"], check=True, capture_output=True)
        cp = subprocess.run(["docker", "exec", "-i", self.name, "tar", "-xf", "-", "-C", "/tmp/grade"],
                            input=_tar_bytes(files), capture_output=True)
        if cp.returncode != 0:
            raise RuntimeError("copy-in failed: " + cp.stderr.decode()[-500:])

    def put_tar(self, data: bytes, dest: str):
        cp = subprocess.run(["docker", "exec", "-i", self.name, "sh", "-c", f"cat > {dest}"], input=data, capture_output=True)
        if cp.returncode != 0:
            raise RuntimeError("copy-in failed: " + cp.stderr.decode()[-500:])

    def bash(self, script: str, timeout: int):
        return subprocess.run(["docker", "exec", "-i", self.name, "bash", "-s"], input=script.encode(),
                              capture_output=True, timeout=timeout)

    def get(self, path: str) -> bytes | None:
        proc = subprocess.run(["docker", "exec", self.name, "cat", path], capture_output=True)
        return proc.stdout if proc.returncode == 0 else None

    def get_dir(self, path: str) -> dict[str, bytes]:
        proc = subprocess.run(["docker", "exec", self.name, "tar", "-C", path, "-cf", "-", "."], capture_output=True)
        if proc.returncode != 0:
            return {}
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
            return {m.name.removeprefix("./"): tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def run_private_grade(*, package: Path, manifest: dict, workspace: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    row = json.loads((package / "private/row.json").read_text())
    f2p, p2p = row["FAIL_TO_PASS"][0], list(row["PASS_TO_PASS"])
    test_cmd, timeout_run = manifest["test_command"], manifest["timeout_run"]
    patches = {"mask.patch": row["patch"].encode() if row["patch"].endswith("\n") else (row["patch"] + "\n").encode(),
               "test_patch.diff": row["test_patch"].encode() if row["test_patch"].endswith("\n")
               else (row["test_patch"] + "\n").encode()}
    image = manifest["grading_image_id"]
    result: dict = {"infrastructure_errors": [], "submission": {}}
    try:
        a = _Container(image)
        try:
            a.put(patches)
            a.put_tar(workspace_tar(Path(workspace)), "/tmp/grade/w.tar")
            proc = a.bash(extract_script(f2p), timeout=1800)
            raw = a.get("/tmp/grade/agent.patch")
            if proc.returncode != 0 or raw is None:
                result["infrastructure_errors"].append({"message": "patch extraction failed: "
                                                        + proc.stderr.decode(errors="replace")[-800:]})
                return result | {"seconds": time.monotonic() - started}
        finally:
            a.remove()
        patch = remove_binary_diffs(raw.decode("utf-8", errors="replace"))
        (output / "agent.patch").write_text(patch)
        changed = re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)
        result["submission"] = {"patch_bytes": len(patch.encode()), "files_changed": len(changed),
                                "f2p_written_by_agent": f2p in changed}
        b = _Container(image)
        try:
            b.put({**patches, "agent.patch": patch.encode()})
            proc = b.bash(evaluate_script(f2p, p2p, test_cmd, timeout_run), timeout=(1 + len(p2p)) * (timeout_run + 60) + 1200)
            files = b.get_dir("/tmp/grade/out")
        finally:
            b.remove()
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        result["infrastructure_errors"].append({"message": str(exc)[-800:]})
        return result | {"seconds": time.monotonic() - started}
    for name, data in files.items():
        (output / name).write_bytes(data)
    text = {k: v.decode("utf-8", errors="replace") for k, v in files.items()}
    if "MASK_EXIT=0" not in text.get("mask.txt", ""):
        result["infrastructure_errors"].append({"message": "mask patch did not apply in the grading image"})
    if "restore_failed" in text.get("restore.txt", ""):
        result["infrastructure_errors"].append({"message": "F2P restore (reverse test_patch) failed"})
    error = text.get("error.txt", "").strip() or None
    runs = {}
    for i, path in enumerate([f2p, *p2p]):
        name = "f2p" if i == 0 else f"p2p{i}"
        if f"{name}.exit" in text:
            runs[path] = {"kind": "f2p" if i == 0 else "p2p", "exit_code": int(text[f"{name}.exit"].strip()),
                          "status": parse_log_pytest(text.get(f"{name}.txt", ""))}
    f2p_run = runs.get(f2p)
    p2p_ok = [runs[p]["exit_code"] == 0 for p in p2p if p in runs]
    result.update({
        "seconds": time.monotonic() - started, "native_error": error,
        "guarded": [g for g in text.get("guarded.txt", "").splitlines() if g],
        "f2p_status": f2p_run["status"] if f2p_run else {},
        "native": {"f2p_success": bool(f2p_run and f2p_run["exit_code"] == 0),
                   "p2p_success": all(p2p_ok) if len(p2p_ok) == len(p2p) else False,
                   "f2p_exit": f2p_run and f2p_run["exit_code"],
                   "p2p_exits": {p: runs[p]["exit_code"] for p in p2p if p in runs},
                   "parsed_f2p": len(f2p_run["status"]) if f2p_run else 0,
                   "native_pass_rate": (round(sum(v in ("PASSED", "XFAIL") for v in f2p_run["status"].values())
                                              / len(f2p_run["status"]), 4) if f2p_run and f2p_run["status"] else 0.0)},
        "p2p_failed_ids": sorted(t for p in p2p if p in runs for t, v in runs[p]["status"].items()
                                 if v in ("FAILED", "ERROR")),
    })
    result["native"]["resolved"] = error is None and result["native"]["f2p_success"] and result["native"]["p2p_success"]
    return result
