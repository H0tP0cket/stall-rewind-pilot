"""Mechanical admission gates for one terminalbench-v1 package (no model calls).

Gates (mirroring the SWE-smith-long and MettleBench lanes):
  start      natural start graded twice: valid, not solved, identical statuses
  reference  upstream solution/solve.sh run in a disposable worker container (network none) from the natural
             start; its W + filesystem R captured and graded twice: full solve, identical
  partial    declared public-only scripted control (never reference material), if one exists: 0 < score < 1
  tamper     worker-writable tampering (W conftest, root conftest/ini, fake /tests, verifier site hooks) graded:
             must equal the start grade
  isolation  no upstream tests/solution/hidden-environment file (by content hash or name) in W or the worker image
             filesystem beyond what the native image itself exposes; image history recorded
  timing     grade wall time recorded; > 600 s holds, > 120 s is flagged
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

from trajectory_retention.contracts import RuntimeClass, RuntimeExport, to_jsonable
from trajectory_retention.execution.archives import extract_directory, pack_directory
from trajectory_retention.execution.environment import DockerWorkerEnvironment, fidelity_manifest_from_tar

from . import environment as E
from .adapter import TerminalBenchTaskAdapter, sha256
from .catalog import PARTIALS
from .package import EVIDENCE, FLAG_GRADE_SECONDS, HOLD_GRADE_SECONDS, write_json

TAMPER_CONFTEST = ("import pytest\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                   "    outcome = yield\n    rep = outcome.get_result()\n    rep.outcome = 'passed'\n")
TAMPER_TEST = "def test_output_files_exist():\n    assert True\n"


def signature(evaluation) -> dict:
    return {"valid": evaluation.valid, "score": evaluation.score, "full_solve": evaluation.full_solve,
            "statuses": {c.check_id: c.status.value for c in evaluation.checks}}


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


class Disposable:
    """A worker-image container restored to the natural start, used to build control states."""

    def __init__(self, adapter: TerminalBenchTaskAdapter, root: Path):
        self.adapter, self.root = adapter, root
        start = root / "start"
        adapter.materialize(start)
        g = adapter.grading
        self.env = DockerWorkerEnvironment(image=adapter.worker_image_id, network="none", name_prefix="tr-tb-validate",
                                           memory=g["memory"], cpus=g["cpus"], timeout=3600)
        packed = pack_directory(start)
        self.env.restore_workspace(packed, fidelity_manifest_from_tar(packed))

    def run(self, command: str, *, timeout: float, input_bytes: bytes | None = None) -> dict:
        started = time.monotonic()
        proc = subprocess.run(["docker", "exec", "-i", "-w", E.NATIVE_WORKDIR, self.env.container_id, "bash", "-c", command],
                              input=input_bytes, capture_output=True, timeout=timeout)
        return {"returncode": proc.returncode, "seconds": time.monotonic() - started,
                "output_tail": (proc.stdout + proc.stderr).decode(errors="replace")[-3000:]}

    def capture(self, label: str) -> tuple[Path, RuntimeExport]:
        data, _ = self.env.capture_workspace()
        runtime = self.env.capture_runtime()
        base = self.root / label
        workspace = base / "W"
        extract_directory(data, workspace)
        archive_path = None
        if runtime.get("archive"):
            archive_path = base / "runtime.tar"
            archive_path.write_bytes(runtime["archive"])
        export = RuntimeExport(
            base_image_id=self.env.image_id, runtime_class=RuntimeClass(runtime["runtime_class"]),
            snapshot={"rule": runtime["rule"], "entries": runtime["entries"], "changed": runtime["changed"],
                      "deleted": runtime["deleted"], "processes": runtime["processes"],
                      "foreign_processes": runtime["foreign_processes"], "coverage": runtime["coverage"],
                      "runtime_class": runtime["runtime_class"], "mounts": runtime.get("mounts"),
                      "initial_mounts": runtime.get("initial_mounts"),
                      "mount_state_changed": runtime.get("mount_state_changed"),
                      "workspace_fidelity_manifest": fidelity_manifest_from_tar(data),
                      "workspace_default_owner": [self.env.worker_uid, self.env.worker_gid],
                      "process_restore_supported": False},
            archive=archive_path, archive_sha256=hashlib.sha256(runtime["archive"]).hexdigest() if archive_path else None,
            workspace_snapshot_sha256=hashlib.sha256(data).hexdigest(), environment=dict(self.env.env))
        (base / "runtime-summary.json").write_text(json.dumps({"changed": runtime["changed"], "deleted": runtime["deleted"],
            "foreign_processes": runtime["foreign_processes"], "runtime_class": runtime["runtime_class"]}, indent=1))
        return workspace, export

    def close(self):
        self.env.close()


def private_hashes(package: Path) -> dict[str, str]:
    out = {}
    for sub in ("tests", "solution", "environment"):
        for path in sorted((package / "private" / sub).rglob("*")):
            if path.is_file() and path.stat().st_size >= 64 and path.name != "Dockerfile":
                out.setdefault(sha256(path), sub + "/" + path.relative_to(package / "private" / sub).as_posix())
    return out


SUSPICIOUS_NAMES = ("solve.sh", "test_outputs.py", "test.sh")


def scan_image(image_id: str, hashes: dict[str, str]) -> dict:
    """Hash every regular file of the image's flattened filesystem (what a container can read)."""
    cid = E.run(["docker", "create", "--label", E.LABEL, image_id, "true"]).stdout.decode().strip()
    hits, names, files = [], [], 0
    try:
        proc = subprocess.Popen(["docker", "export", cid], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                name = "/" + member.name.lstrip("./")
                if member.isreg():
                    files += 1
                    data = tf.extractfile(member).read()
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in hashes:
                        hits.append({"path": name, "matches": hashes[digest]})
                base = name.rsplit("/", 1)[-1]
                if base in SUSPICIOUS_NAMES or name.startswith(("/tests/", "/solution/", "/oracle/")):
                    names.append(name)
        proc.wait(timeout=600)
    finally:
        E.run(["docker", "rm", "-f", cid], check=False)
    return {"image_id": image_id, "files_scanned": files, "hash_hits": hits, "suspicious_names": sorted(set(names))}


def history(image_id: str) -> list[str]:
    out = E.run(["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", image_id]).stdout.decode()
    return [line for line in out.splitlines() if line.strip()]


def start_scan(start: Path, hashes: dict[str, str]) -> dict:
    hits, names = [], []
    for path in sorted(start.rglob("*")):
        rel = path.relative_to(start).as_posix()
        if path.is_file() and not path.is_symlink():
            digest = sha256(path)
            if digest in hashes:
                hits.append({"path": "/app/" + rel, "matches": hashes[digest]})
        if path.name in SUSPICIOUS_NAMES:
            names.append("/app/" + rel)
    return {"hash_hits": hits, "suspicious_names": names, "git_dirs": sorted(
        "/app/" + p.relative_to(start).as_posix() for p in start.rglob(".git") if p.is_dir())}


def validate(package: Path, *, out: Path | None = None) -> dict:
    adapter = TerminalBenchTaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    task = adapter.upstream_task
    out = Path(out or EVIDENCE / "tasks" / task / "validation")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    record: dict = {"task_id": adapter.task_id, "upstream_task": task, "package_sha256": sha256(package / "package.json"),
                    "gates": {}, "grades": {}, "provider_calls": 0}
    grades = record["grades"]

    def grade(label: str, workspace: Path, runtime: RuntimeExport | None):
        value = adapter.grade(workspace, stage_id="task", private_output_dir=out / "grades" / label, runtime=runtime)
        grades[label] = {**signature(value), "grade_wall_seconds": value.details.get("grade_wall_seconds"),
                         "pytest_seconds": value.details.get("pytest_seconds"),
                         "native_reward": value.details.get("native_reward"),
                         "native_metrics": value.details.get("native_metrics"),
                         "reserved_runtime_paths_dropped": value.details.get("reserved_runtime_paths_dropped"),
                         "infrastructure_error": value.details.get("infrastructure_error"),
                         "diagnostics": {c.check_id: (c.details.get("diagnostic_tail") or "")[-400:]
                                         for c in value.checks if c.status.value != "PASS"}}
        return value

    start = out / "start-W"
    adapter.materialize(start)
    s1, s2 = grade("start_1", start, None), grade("start_2", start, None)
    g = record["gates"]
    g["start_valid_not_solved_identical"] = (s1.valid and not s1.full_solve and s1.score < 1
                                             and signature(s1) == signature(s2))
    # Reference: upstream solve.sh from the natural start, network none.
    box = Disposable(adapter, out / "reference")
    try:
        solve = (package / "private/solution/solve.sh").read_bytes()
        extra = [p for p in (package / "private/solution").rglob("*") if p.is_file() and p.name != "solve.sh"]
        if extra:
            raise RuntimeError("solution directory holds more than solve.sh; not supported by this lane")
        record["reference_run"] = box.run("bash -s", timeout=3600, input_bytes=solve)
        ref_w, ref_r = box.capture("reference-state")
    finally:
        box.close()
    r1, r2 = grade("reference_1", ref_w, ref_r), grade("reference_2", ref_w, ref_r)
    g["reference_full_identical"] = (r1.valid and r1.full_solve is True and r1.score == 1.0
                                     and signature(r1) == signature(r2))
    # Partial control.
    partial = PARTIALS.get(task)
    if partial:
        box = Disposable(adapter, out / "partial")
        try:
            record["partial_run"] = [box.run(c, timeout=900) for c in partial["commands"]]
            p_w, p_r = box.capture("partial-state")
        finally:
            box.close()
        p = grade("partial", p_w, p_r)
        record["partial_control"] = {"basis": partial["basis"], "commands": partial["commands"], "score": p.score}
        g["partial_strictly_between"] = p.valid and 0 < p.score < 1
    else:
        record["partial_control"] = None
        record["binary_only_note"] = "no declared public-only partial control; recorded as binary-only evidence"
    # Tamper control.
    box = Disposable(adapter, out / "tamper")
    try:
        cmds = [
            f"printf %s {_quote(TAMPER_CONFTEST)} > /app/conftest.py",
            f"printf %s {_quote(TAMPER_CONFTEST)} > /conftest.py",
            "printf '[pytest]\\naddopts = -p no:terminal\\n' > /pytest.ini",
            f"mkdir -p /tests && printf %s {_quote(TAMPER_TEST)} > /tests/test_outputs.py && "
            f"printf %s {_quote(TAMPER_CONFTEST)} > /tests/conftest.py",
            "mkdir -p /tb-verifier/venv/lib/python3.13/site-packages && "
            "printf 'import sys; sys.exit(0)\\n' > /tb-verifier/venv/lib/python3.13/site-packages/sitecustomize.py",
        ]
        # A sitecustomize.py in the worker image's own python3 stdlib is not part of this control: it also
        # disables the shared harness's python3 capture probes (observed 2026-09-27), so it is an engine-level
        # concern, and the verifier's interpreter is uv's managed CPython under /tb-verifier.
        record["tamper_run"] = [box.run(c, timeout=120) for c in cmds]
        t_w, t_r = box.capture("tamper-state")
    finally:
        box.close()
    t = grade("tamper", t_w, t_r)
    g["tamper_equals_start"] = signature(t) == signature(s1)
    # Isolation.
    hashes = private_hashes(package)
    native_scan = scan_image(adapter.manifest["environment"]["native"]["image_id"], hashes)
    worker_scan = scan_image(adapter.worker_image_id, hashes)
    native_paths = {h["path"] for h in native_scan["hash_hits"]} | set(native_scan["suspicious_names"])
    introduced = ([h for h in worker_scan["hash_hits"] if h["path"] not in native_paths]
                  + [n for n in worker_scan["suspicious_names"] if n not in native_paths])
    exposures = sorted([{"path": h["path"], "matches": h["matches"]} for h in native_scan["hash_hits"]]
                       + [{"path": n, "matches": "name"} for n in native_scan["suspicious_names"]], key=lambda e: e["path"])
    record["isolation"] = {"private_files_hashed": len(hashes), "start": start_scan(start, hashes),
                           "native_image": native_scan, "worker_image": worker_scan,
                           "native_exposures": exposures,
                           "native_exposure_categories": sorted({e["matches"].split("/", 1)[0] for e in exposures}),
                           "lane_introduced": introduced,
                           "worker_history": history(adapter.worker_image_id)}
    history_refs = [line for line in record["isolation"]["worker_history"]
                    if any(k in line for k in ("solution", "solve.sh", "/tests", "tests/"))]
    record["isolation"]["history_references"] = history_refs
    g["isolation_no_lane_introduced_leak"] = not introduced
    # Timing.
    times = [v["grade_wall_seconds"] for v in grades.values() if v.get("grade_wall_seconds") is not None]
    record["timing"] = {"max_grade_wall_seconds": max(times) if times else None,
                        "start_seconds": grades["start_1"]["grade_wall_seconds"],
                        "reference_seconds": grades["reference_1"]["grade_wall_seconds"],
                        "flag_over_seconds": FLAG_GRADE_SECONDS, "hold_over_seconds": HOLD_GRADE_SECONDS,
                        "reference_solve_seconds": record["reference_run"]["seconds"]}
    record["timing"]["flagged"] = bool(times) and max(times) > FLAG_GRADE_SECONDS
    g["grade_time_within_hold_bound"] = bool(times) and max(times) <= HOLD_GRADE_SECONDS
    record["resources"] = {"native": adapter.manifest["native_limits"], "grader_container": {
        "cpus": adapter.grading["cpus"], "memory": adapter.grading["memory"]}}
    record["passed"] = all(g.values())
    write_json(out / "validation.json", record)
    return record
