"""Admission gate for one packaged terminalbench-v2 task (Docker; no model calls).

Gate items (docs/research/task-admission-criteria.md), each recorded with its evidence and a passed flag:
  per_test_scoring  every grade gives a status for every frozen hidden test and a share passed
  hidden_tests      at least 3 hidden tests collected in the grader image (count recorded)
  grader_isolation  grading on an isolated copy of captured W and R, grader network none, verifier prebuilt;
                    worker-writable tampering (W conftest, root conftest/ini, fake /tests, verifier site hooks) leaves
                    the grade equal to the start grade; no private file reaches W or the worker image (leak scan)
  reference         the upstream reference (solution/ at /solution, solve.sh run from /app, network none) graded 3 times
                    on its captured W and R: full solve 3 of 3
  starter_fails     the natural start, captured and graded 3 times, is not solved
  flaky_tests       the 3 start grades have identical per-test statuses, and so do the 3 reference grades
  network           network rule (network-rule/v1): the task needs network if the instruction tells the worker to fetch
                    a remote resource (static screen), or the reference fails with the network off; when it fails
                    offline it is re-run once with the network on to tell "needs network" from "reference broken"
  leak_scan         leak-scan/v1 (leak.py)
  emulation         worker image is linux/arm64, the static screen found no QEMU guest, and no x86 executable in W
  restore_check     the scripted runner session (restore_check.py) restores a checkpoint and re-runs its next command
                    with identical output
  runtime_class     declared CAPTURED: the reference's R changes and processes are recorded; filesystem R restores in
                    the restore check; live processes are not restored
Raw receipts stay under artifacts/v2/lane/tasks/<task>/gate (gitignored); the committed admission evidence is compact
and holds no test names, diagnostics or reference content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from . import environment as E
from . import leak as L
from . import static as ST
from .adapter import TerminalBenchV2TaskAdapter, sha256
from .leak_review import reviews_for
from .package import EVIDENCE, write_json
from .states import Disposable

GATE_VERSION = "terminalbench-v2.gate.v1"
GRADES_PER_STATE = 3
TAMPER_CONFTEST = ("import pytest\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                   "    outcome = yield\n    rep = outcome.get_result()\n    rep.outcome = 'passed'\n")
TAMPER_TEST = "def test_output_files_exist():\n    assert True\n"
X86_MACHINES = {62: "x86-64", 3: "i386"}


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def signature(evaluation) -> dict:
    return {"valid": evaluation.valid, "score": evaluation.score, "full_solve": evaluation.full_solve,
            "statuses": {c.check_id: c.status.value for c in evaluation.checks}}


def compact(sig: dict) -> dict:
    """Commit-safe view of a grade: counts and a hash of the per-test statuses, no test names."""
    statuses = sig["statuses"]
    return {"valid": sig["valid"], "score": sig["score"], "full_solve": sig["full_solve"],
            "passed": sum(v == "PASS" for v in statuses.values()), "total": len(statuses),
            "statuses_sha256": hashlib.sha256(json.dumps(statuses, sort_keys=True).encode()).hexdigest()}


def elf_machines(workspace: Path) -> list[dict]:
    found = []
    for path in sorted(workspace.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.stat().st_size >= 20:
            with open(path, "rb") as handle:
                head = handle.read(20)
            if head[:4] == b"\x7fELF":
                little = head[5] == 1
                machine = int.from_bytes(head[18:20], "little" if little else "big")
                kind = int.from_bytes(head[16:18], "little" if little else "big")
                found.append({"path": "/app/" + path.relative_to(workspace).as_posix(), "e_machine": machine,
                              "e_type": kind, "x86": machine in X86_MACHINES})
    return found


def leak_item(leak: dict) -> dict:
    return {"passed": leak["passed"], "version": leak["version"], "failed_on": leak["failed_on"],
            "counts": {"locations": len(leak["locations"]), "hashes": len(leak["hashes"]),
                       "findings": len(leak["findings"]), "reviewed": leak["reviewed_findings"],
                       "unresolved": leak["unresolved_findings"], "names": len(leak["names"])},
            "reviews": [{"id": f["id"], "kind": f["kind"], "verdict": (f["review"] or {}).get("verdict"),
                         "reason": (f["review"] or {}).get("reason")} for f in leak["findings"]],
            "comment_hint_flags": leak["comment_hint_flag_count"], "starter_files_scanned": leak["starter_files_scanned"],
            "git_objects_scanned": leak["git_objects_scanned"],
            "worker_image_files_scanned": leak["worker_image_files_scanned"]}


def rescan_leak(package: Path) -> dict:
    """Re-run only the leak scan on the gate's captured start and reference W (after recording reviews)."""
    adapter = TerminalBenchV2TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    task = adapter.upstream_task
    out = EVIDENCE / "tasks" / task / "gate"
    record = json.loads((out / "gate.json").read_text())
    if record["package_sha256"] != sha256(package / "package.json"):
        raise ValueError("Gate record binds a different package; re-run the gate")
    start_w, ref_w = out / "start" / "start-state" / "W", out / "reference" / "reference-state" / "W"
    instruction = (package / "public/instruction.md").read_text()
    leak = L.scan(task=task, private=adapter.private, instruction=instruction, start=start_w, reference=ref_w,
                  worker_image_id=adapter.worker_image_id, reviews=reviews_for(task))
    write_json(out / "leak.json", leak)
    write_json(out / "leak-explain.json", L.explain(leak, private=adapter.private, start=start_w, reference=ref_w,
                                                   instruction=instruction))
    record["items"]["leak_scan"] = leak_item(leak)
    iso = record["items"]["grader_isolation"]
    iso["private_files_in_w_or_image"] = len(leak["hashes"]) + len(leak["locations"])
    iso["passed"] = (iso["tamper_equals_start"] and iso["grader_network"] == "none"
                     and iso["private_files_in_w_or_image"] == 0)
    record["passed"] = all(i["passed"] for i in record["items"].values())
    record["leak_rescanned"] = True
    write_json(out / "gate.json", record)
    return record


def run_reference(adapter, root: Path, network: str) -> tuple[dict, Path, object, dict]:
    g = adapter.grading
    timeout = max(3600, int(adapter.manifest["native_limits"].get("agent_timeout_sec") or 0) * 2)
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=adapter.package / "public/workspace.tar.gz",
                     root=root, memory=g["memory"], cpus=g["cpus"], network=network, timeout=timeout)
    try:
        box.put_directory(adapter.private / "solution", "/solution")
        run = box.run("bash /solution/solve.sh", timeout=timeout)
        run["cleanup"] = box.run("rm -rf /solution", timeout=120, user="0:0")["returncode"]
        w, r, summary = box.capture("reference-state")
    finally:
        box.close()
    return run, w, r, summary


def validate(package: Path, *, static_row: dict | None = None, out: Path | None = None) -> dict:
    adapter = TerminalBenchV2TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    task = adapter.upstream_task
    static_row = static_row or ST.screen_task(task)
    out = Path(out or EVIDENCE / "tasks" / task / "gate")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    record: dict = {"version": GATE_VERSION, "task_id": adapter.task_id, "upstream_task": task,
                    "package_sha256": sha256(package / "package.json"), "items": {}, "grades": {}, "provider_calls": 0}
    grades, items = record["grades"], record["items"]
    sigs: dict[str, dict] = {}

    def grade(label: str, workspace: Path, runtime) -> dict:
        value = adapter.grade(workspace, stage_id="task", private_output_dir=out / "grades" / label, runtime=runtime)
        sigs[label] = signature(value)
        grades[label] = {**compact(sigs[label]), "grade_wall_seconds": value.details.get("grade_wall_seconds"),
                         "infrastructure_error": value.details.get("infrastructure_error"),
                         "reserved_runtime_paths_dropped": value.details.get("reserved_runtime_paths_dropped")}
        (out / "grades" / label / "signature.json").write_text(json.dumps(sigs[label], indent=1) + "\n")
        return sigs[label]

    g = adapter.grading
    E.require_headroom(f"the admission gate of {task}")
    # Natural start, captured from a fresh worker container and graded 3 times.
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=package / "public/workspace.tar.gz",
                     root=out / "start", memory=g["memory"], cpus=g["cpus"])
    try:
        start_w, start_r, start_summary = box.capture("start-state")
    finally:
        box.close()
    start = [grade(f"start_{i}", start_w, start_r) for i in range(1, GRADES_PER_STATE + 1)]
    # Reference with the network off.
    ref_run, ref_w, ref_r, ref_summary = run_reference(adapter, out / "reference", "none")
    record["reference_run"] = {k: ref_run[k] for k in ("returncode", "seconds", "network", "cleanup")}
    record["reference_runtime"] = {"changed_paths": len(ref_summary["runtime_changed"]),
                                   "deleted_paths": len(ref_summary["runtime_deleted"]),
                                   "foreign_processes": ref_summary["foreign_processes"],
                                   "runtime_class": ref_summary["runtime_class"]}
    (out / "reference" / "run.json").write_text(json.dumps(ref_run, indent=1) + "\n")
    reference = [grade(f"reference_{i}", ref_w, ref_r) for i in range(1, GRADES_PER_STATE + 1)]
    ref_ok = all(s["valid"] and s["full_solve"] is True and s["score"] == 1.0 for s in reference)
    hidden = len(adapter.scorer.expected)
    every = list(sigs.values())
    items["per_test_scoring"] = {"passed": all(s["valid"] and len(s["statuses"]) == hidden and s["score"] is not None
                                               for s in every),
                                 "evidence": "every grade: one status per frozen hidden test, share passed = passing/all"}
    items["hidden_tests"] = {"passed": hidden >= ST.MIN_HIDDEN_TESTS, "count": hidden, "minimum": ST.MIN_HIDDEN_TESTS}
    items["starter_fails"] = {"passed": all(s["valid"] and s["full_solve"] is not True and s["score"] < 1 for s in start),
                              "start_share_passed": start[0]["score"]}
    items["reference"] = {"passed": ref_ok, "full_solves": sum(s["full_solve"] is True for s in reference),
                          "grades": GRADES_PER_STATE, "solve_seconds": ref_run["seconds"]}
    items["flaky_tests"] = {"passed": all(s == start[0] for s in start) and all(s == reference[0] for s in reference),
                            "start_identical": all(s == start[0] for s in start),
                            "reference_identical": all(s == reference[0] for s in reference)}
    # Network rule.
    network = {"instruction_remote_fetch": static_row["instruction_remote_fetch"],
               "solve_network_commands": static_row["solve_network_commands"],
               "reference_offline_full_solve": ref_ok}
    if not ref_ok:
        on_run, on_w, on_r, _ = run_reference(adapter, out / "reference-network", "bridge")
        on = grade("reference_network_on", on_w, on_r)
        network["reference_network_on"] = {"returncode": on_run["returncode"], "seconds": on_run["seconds"],
                                           "full_solve": on["full_solve"] is True, "score": on["score"]}
        network["classification"] = ("needs network: the reference fails offline and passes with the network on"
                                     if on["full_solve"] is True else
                                     "reference fails with and without the network (not a network finding)")
    items["network"] = {"passed": ref_ok and not static_row["instruction_remote_fetch"], "rule": "network-rule/v1",
                        **network}
    # Grader isolation: tamper control.
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=package / "public/workspace.tar.gz",
                     root=out / "tamper", memory=g["memory"], cpus=g["cpus"])
    try:
        cmds = [f"printf %s {_quote(TAMPER_CONFTEST)} > /app/conftest.py",
                f"printf %s {_quote(TAMPER_CONFTEST)} > /conftest.py",
                "printf '[pytest]\\naddopts = -p no:terminal\\n' > /pytest.ini",
                f"mkdir -p /tests && printf %s {_quote(TAMPER_TEST)} > /tests/test_outputs.py && "
                f"printf %s {_quote(TAMPER_CONFTEST)} > /tests/conftest.py",
                "mkdir -p /tb-verifier/venv/lib/python3.13/site-packages && "
                "printf 'import sys; sys.exit(0)\\n' > /tb-verifier/venv/lib/python3.13/site-packages/sitecustomize.py"]
        tamper_runs = [box.run(c, timeout=120) for c in cmds]
        t_w, t_r, _ = box.capture("tamper-state")
    finally:
        box.close()
    tamper = grade("tamper", t_w, t_r)
    receipt = json.loads((out / "grades" / "start_1" / "receipt.json").read_text())
    binding = receipt.get("receipt", {}).get("binding", {})
    # Leak scan (also the isolation scan of private files in W and the worker image).
    instruction = (package / "public/instruction.md").read_text()
    leak = L.scan(task=task, private=adapter.private, instruction=instruction, start=start_w, reference=ref_w,
                  worker_image_id=adapter.worker_image_id, reviews=reviews_for(task))
    write_json(out / "leak.json", leak)
    write_json(out / "leak-explain.json", L.explain(leak, private=adapter.private, start=start_w, reference=ref_w,
                                                   instruction=instruction))
    items["grader_isolation"] = {
        "passed": (tamper == start[0] and binding.get("network") == "none" and not leak["hashes"]
                   and not leak["locations"]),
        "tamper_equals_start": tamper == start[0], "tamper_commands_ran": [r["returncode"] for r in tamper_runs],
        "grader_network": binding.get("network"), "grader_image_id": binding.get("image_id"),
        "graded_runtime": "captured W and captured R on every grade (baseline R refused by the grader)",
        "private_files_in_w_or_image": len(leak["hashes"]) + len(leak["locations"])}
    items["leak_scan"] = leak_item(leak)
    # Emulation.
    elves = elf_machines(start_w)
    arch = adapter.manifest["environment"]["worker"].get("architecture") or E.image_info(adapter.worker_image_id)["architecture"]
    items["emulation"] = {"passed": arch == "arm64" and not static_row["emulation_mentions"] and not any(e["x86"] for e in elves),
                          "worker_architecture": arch, "qemu_mentions": static_row["emulation_mentions"],
                          "elf_files_in_w": len(elves), "x86_elf_in_w": [e["path"] for e in elves if e["x86"]][:20]}
    items["runtime_class"] = {"passed": True, "declared": "CAPTURED",
                              "evidence": {"reference_runtime": record["reference_runtime"],
                                           "start_runtime_class": start_summary["runtime_class"],
                                           "restore": "see restore_check"}}
    times = [v["grade_wall_seconds"] for v in grades.values() if v.get("grade_wall_seconds") is not None]
    record["timing"] = {"max_grade_wall_seconds": max(times) if times else None,
                        "median_grade_wall_seconds": sorted(times)[len(times) // 2] if times else None,
                        "reference_solve_seconds": ref_run["seconds"]}
    record["passed"] = all(i["passed"] for i in items.values())
    write_json(out / "gate.json", record)
    return record
