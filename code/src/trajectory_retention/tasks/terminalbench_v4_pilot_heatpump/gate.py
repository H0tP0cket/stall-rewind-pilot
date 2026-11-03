"""Admission gate for one packaged terminalbench-v4-pilot-heatpump task (Docker; no model calls).

Adapted from the terminalbench-v2 gate (lanes never import each other). Gate items (docs/research/
task-admission-criteria.md), each recorded with its evidence and a passed flag:
  per_test_scoring  every grade gives a status for every frozen hidden test and a share passed
  hidden_tests      at least 3 hidden tests collected by the task's own test.sh in the verifier image (count recorded)
  grader_isolation  grading in a separate verifier container (Harbor separate mode) on copies of the declared artifacts
                    of captured W and R, network none, tests baked into the private verifier image; worker-writable
                    tampering (conftest/ini files in W and R, a fake /tests, a planted reward file, files in Harbor's
                    publish dir) leaves the grade equal to the start grade; no private file reaches W or the worker image
  reference         the upstream reference (solution/ at /solution; Harbor's oracle: chmod +x as root, then
                    ``(/solution/solve.sh) > log`` as the worker from /app, network none) graded 3 times on its captured
                    W and R: full solve 3 of 3
  starter_fails     the natural start, captured and graded 3 times, is not solved
  flaky_tests       the 3 start grades have identical per-test statuses, and so do the 3 reference grades
  network           network-rule/v1: the task needs network if the instruction tells the worker to fetch a remote
                    resource (static screen), or the reference fails with the network off; when it fails offline it is
                    re-run once with the network on to tell "needs network" from "reference broken"
  leak_scan         leak-scan/v1 (leak.py, TB4 layout)
  emulation         worker image linux/arm64, no emulator in the task, no x86 executable in W or the build context
  start_fidelity    the natural start's /app (owner, mode) equals the native image's /app
  restore_check     the scripted runner session (restore_check.py) restores a checkpoint and re-runs its next command
                    with identical output
  runtime_class     declared CAPTURED: the reference's R changes and processes are recorded; filesystem R restores in
                    the restore check; live processes are not restored (Harbor's verifier receives files only)
Raw receipts stay under source.EVIDENCE/tasks/<task>/gate (local, gitignored); the committed admission evidence is
compact and holds no test names, diagnostics or reference content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .source import remove_tree
from . import environment as E
from . import leak as L
from . import static as ST
from .adapter import TerminalBenchV4TaskAdapter, sha256
from .leak_review import reviews_for
from .package import EVIDENCE, write_json
from .states import Disposable
from .services import docker
from trajectory_retention.execution.archives import extract_directory

GATE_VERSION = "terminalbench-v4-pilot-heatpump.gate.v1"
GRADES_PER_STATE = 3
TAMPER_CONFTEST = ("import pytest\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                   "    outcome = yield\n    rep = outcome.get_result()\n    rep.outcome = 'passed'\n")
ENOSPC = "No space left on device"
TAMPER_TEST ="def test_planted():\n    assert True\n"
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


def hidden_files(static_row: dict) -> tuple[str, ...]:
    return tuple("tests/" + t for t in static_row["pytest_targets"]) + ("tests/test.sh",)


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


def leak_view(adapter, workspace: Path, target: Path) -> Path:
    """Scan task-visible /app/packet in R as well as the (initially empty) W."""
    if target.exists(): remove_tree(target)
    shutil.copytree(workspace, target / 'workspace')
    data = docker('run', '--rm', '--network', 'none', '--entrypoint', 'tar',
                  adapter.worker_image_id, '-C', '/app', '-cf', '-', '.').stdout
    extract_directory(data, target / 'app')
    return target


def rescan_leak(package: Path, static_row: dict) -> dict:
    """Re-run only the leak scan on the gate's captured start and reference W (after recording reviews)."""
    adapter = TerminalBenchV4TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    task = adapter.upstream_task
    out = EVIDENCE / "tasks" / task / "gate"
    record = json.loads((out / "gate.json").read_text())
    if record["package_sha256"] != sha256(package / "package.json"):
        raise ValueError("Gate record binds a different package; re-run the gate")
    start_w, ref_w = out / "start" / "start-state" / "W", out / "reference" / "reference-state" / "W"
    start_w = leak_view(adapter, start_w, out / 'leak-start')
    ref_w = leak_view(adapter, ref_w, out / 'leak-reference')
    instruction = (package / "public/instruction.md").read_text()
    hidden = hidden_files(static_row)
    leak = L.scan(task=task, private=adapter.private, instruction=instruction, start=start_w, reference=ref_w,
                  worker_image_id=adapter.worker_image_id, reviews=reviews_for(task), hidden=hidden)
    write_json(out / "leak.json", leak)
    write_json(out / "leak-explain.json", L.explain(leak, private=adapter.private, start=start_w, reference=ref_w,
                                                   instruction=instruction, hidden=hidden))
    record["items"]["leak_scan"] = leak_item(leak)
    iso = record["items"]["grader_isolation"]
    iso["private_files_in_w_or_image"] = len(leak["hashes"]) + len(leak["locations"])
    iso["passed"] = (iso["tamper_equals_start"] and iso["verifier_network"] == "none"
                     and iso["private_files_in_w_or_image"] == 0)
    record["passed"] = all(i["passed"] for i in record["items"].values())
    record["leak_rescanned"] = True
    write_json(out / "gate.json", record)
    return record


def run_reference(adapter, root: Path, network: str) -> tuple[dict, Path, object, dict]:
    g = adapter.grading
    timeout = max(3600, int(adapter.manifest["native_limits"].get("agent_timeout_sec") or 0))
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=adapter.package / "public/workspace.tar.gz",
                     root=root, memory=g["memory"], cpus=g["cpus"], network=network, timeout=timeout)
    try:
        box.put_directory(adapter.private / "solution", "/solution")
        box.run("chmod +x /solution/solve.sh", timeout=120, user="0:0")
        run = box.run("export DEBIAN_FRONTEND=noninteractive; (/solution/solve.sh) > /tmp/tbv4-oracle.txt 2>&1; "
                      "rc=$?; tail -c 3000 /tmp/tbv4-oracle.txt; exit $rc", timeout=timeout)
        run["cleanup"] = box.run("rm -rf /solution /tmp/tbv4-oracle.txt", timeout=120, user="0:0")["returncode"]
        w, r, summary = box.capture("reference-state")
    finally:
        box.close()
    return run, w, r, summary


def validate(package: Path, *, static_row: dict, out: Path | None = None) -> dict:
    adapter = TerminalBenchV4TaskAdapter(package, scratch_root=EVIDENCE / "scratch")
    task = adapter.upstream_task
    out = Path(out or EVIDENCE / "tasks" / task / "gate")
    if out.exists():
        remove_tree(out)
    out.mkdir(parents=True)
    record: dict = {"version": GATE_VERSION, "task_id": adapter.task_id, "upstream_task": task,
                    "package_sha256": sha256(package / "package.json"), "items": {}, "grades": {}, "provider_calls": 0}
    grades, items = record["grades"], record["items"]
    sigs: dict[str, dict] = {}

    def grade(label: str, workspace: Path, runtime) -> dict:
        value = adapter.grade(workspace, stage_id="task", private_output_dir=out / "grades" / label, runtime=runtime)
        sigs[label] = signature(value)
        grades[label] = {**compact(sigs[label]), "grade_wall_seconds": value.details.get("grade_wall_seconds"),
                         "test_sh_seconds": value.details.get("test_sh_seconds"),
                         "native_reward": value.details.get("native_reward"),
                         "infrastructure_error": value.details.get("infrastructure_error"),
                         "artifacts_uploaded": sum(bool(a.get("uploaded")) for a in value.details.get("artifacts") or [])}
        (out / "grades" / label / "signature.json").write_text(json.dumps(sigs[label], indent=1) + "\n")
        return sigs[label]

    g = adapter.grading
    record["disk_at_start"] = E.docker_disk()
    # Natural start, captured from a fresh worker container and graded 3 times.
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=package / "public/workspace.tar.gz",
                     root=out / "start", memory=g["memory"], cpus=g["cpus"])
    try:
        app_root = box.root_listing()
        # The worker user must be able to write at all: with the Docker VM disk down to its root-reserved blocks a
        # non-root user (rs-archive-clone's `agent`) gets ENOSPC everywhere, which is the disk, not the task.
        write_probe = box.run("p=/tmp/.tbv4-write-probe; head -c 65536 /dev/zero > $p; rc=$?; rm -f $p; exit $rc",
                              timeout=120)
        start_w, start_r, start_summary = box.capture("start-state")
    finally:
        box.close()
    record["worker_write_probe"] = {"returncode": write_probe["returncode"],
                                    "output_tail": write_probe["output_tail"][-300:]}
    if write_probe["returncode"] != 0:
        record["docker_disk"] = {"seen_in": ["worker user write probe: "
                                             + " ".join(write_probe["output_tail"].split())[-200:]],
                                 "disk": record["disk_at_start"]}
        record["passed"] = False
        record["stopped"] = "the worker user cannot write in a fresh worker container; gate not run further"
        write_json(out / "gate.json", record)
        return record
    native_root = adapter.manifest["native_start"]["app_root"]
    items["start_fidelity"] = {"passed": app_root == f"{oct(native_root['mode'])[2:]} {native_root['uid']} {native_root['gid']}",
                               "natural_start_app_root": app_root, "native_app_root": native_root,
                               "evidence": "stat -c '%a %u %g' /workspace after the runner-equivalent restore"}
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
    items["hidden_tests"] = {"passed": hidden >= ST.MIN_HIDDEN_TESTS, "count": hidden, "minimum": ST.MIN_HIDDEN_TESTS,
                             "static_count": static_row.get("hidden_tests_static")}
    items["starter_fails"] = {"passed": all(s["valid"] and s["full_solve"] is not True and s["score"] < 1 for s in start),
                              "start_share_passed": start[0]["score"]}
    items["reference"] = {"passed": ref_ok, "full_solves": sum(s["full_solve"] is True for s in reference),
                          "grades": GRADES_PER_STATE, "solve_seconds": ref_run["seconds"],
                          "share_passed": [s["score"] for s in reference]}
    items["flaky_tests"] = {"passed": all(s == start[0] for s in start) and all(s == reference[0] for s in reference),
                            "start_identical": all(s == start[0] for s in start),
                            "reference_identical": all(s == reference[0] for s in reference)}
    # Network rule.
    network = {"instruction_remote_fetch": static_row["instruction_remote_fetch"],
               "solve_time_installs": static_row["solve_time_installs"],
               "reference_offline_full_solve": ref_ok}
    network['classification'] = 'approved per-run local services; external network prohibited'
    isolation_path = EVIDENCE / 'services-check.json'
    isolation = json.loads(isolation_path.read_text())
    network['isolation_check_sha256'] = sha256(isolation_path)
    network['isolation_assertions'] = isolation['assertions']
    items["network"] = {"passed": ref_ok and isolation.get('passed') is True and not static_row["instruction_remote_fetch"],
                        "rule": "network-rule/local-services-v1", **network}
    nospace = ['reference run'] if ENOSPC in ref_run['output_tail'] else []
    # Grader isolation: tamper control (worker-writable places a grader could read).
    box = Disposable(worker_image_id=adapter.worker_image_id, workspace_archive=package / "public/workspace.tar.gz",
                     root=out / "tamper", memory=g["memory"], cpus=g["cpus"])
    try:
        cmds = [f"printf %s {_quote(TAMPER_CONFTEST)} > /app/conftest.py",
                "printf '[pytest]\\naddopts = -p no:terminal\\n' > /app/pytest.ini",
                f"printf %s {_quote(TAMPER_CONFTEST)} > /tmp/conftest.py",
                f"printf %s {_quote(TAMPER_CONFTEST)} > /conftest.py",
                "printf '[pytest]\\naddopts = -p no:terminal\\n' > /pytest.ini",
                f"mkdir -p /tests && printf %s {_quote(TAMPER_TEST)} > /tests/test_planted.py && "
                f"printf %s {_quote(TAMPER_CONFTEST)} > /tests/conftest.py",
                "mkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt",
                f"mkdir -p /logs/artifacts && printf %s {_quote(TAMPER_CONFTEST)} > /logs/artifacts/conftest.py",
                "mkdir -p /tb-grader/out && echo '{\"event\": \"plugin_loaded\"}' > /tb-grader/out/reports.jsonl"]
        tamper_runs = [box.run(c, timeout=120) for c in cmds]
        nospace += ["tamper commands"] if any(ENOSPC in r["output_tail"] for r in tamper_runs) else []
        t_w, t_r, _ = box.capture("tamper-state")
    finally:
        box.close()
    tamper = grade("tamper", t_w, t_r)
    receipt = json.loads((out / "grades" / "start_1" / "receipt.json").read_text())
    vbinding = receipt.get("receipt", {}).get("verifier_binding", {})
    sbinding = receipt.get("receipt", {}).get("source_binding", {})
    # Leak scan (also the isolation scan of private files in W and the worker image).
    start_w = leak_view(adapter, start_w, out / 'leak-start')
    ref_w = leak_view(adapter, ref_w, out / 'leak-reference')
    instruction = (package / "public/instruction.md").read_text()
    hidden_paths = hidden_files(static_row)
    leak = L.scan(task=task, private=adapter.private, instruction=instruction, start=start_w, reference=ref_w,
                  worker_image_id=adapter.worker_image_id, reviews=reviews_for(task), hidden=hidden_paths)
    write_json(out / "leak.json", leak)
    write_json(out / "leak-explain.json", L.explain(leak, private=adapter.private, start=start_w, reference=ref_w,
                                                   instruction=instruction, hidden=hidden_paths))
    items["grader_isolation"] = {
        "passed": (tamper == start[0] and vbinding.get("network") == "none" and sbinding.get("network") == "none"
                   and not leak["hashes"] and not leak["locations"]),
        "tamper_equals_start": tamper == start[0], "tamper_commands_ran": [r["returncode"] for r in tamper_runs],
        "verifier_network": vbinding.get("network"), "source_network": sbinding.get("network"),
        "verifier_image_id": vbinding.get("image_id"),
        "transfer": "declared artifacts only (Harbor separate mode): " + json.dumps(g["artifacts"]),
        "graded_runtime": "captured W and captured R on every grade (baseline R refused by the grader)",
        "private_files_in_w_or_image": len(leak["hashes"]) + len(leak["locations"])}
    items["leak_scan"] = leak_item(leak)
    # Emulation.
    elves = elf_machines(start_w)
    arch = adapter.manifest["environment"]["worker"].get("architecture") or E.image_info(adapter.worker_image_id)["architecture"]
    items["emulation"] = {"passed": (arch == "arm64" and not static_row["emulation_mentions"]
                                     and not static_row["x86_only_binary_count"] and not any(e["x86"] for e in elves)),
                          "worker_architecture": arch, "emulator_mentions": static_row["emulation_mentions"],
                          "x86_only_in_build_context": static_row["x86_only_binary_count"],
                          "elf_files_in_w": len(elves), "x86_elf_in_w": [e["path"] for e in elves if e["x86"]][:20],
                          "elf_machines_in_w": sorted({e["e_machine"] for e in elves})}
    items["runtime_class"] = {"passed": start_summary["runtime_class"] == "R_CAPTURED" and ref_summary["runtime_class"] == "R_CAPTURED", "declared": "CAPTURED",
                              "evidence": {"reference_runtime": record["reference_runtime"],
                                           "start_runtime_class": start_summary["runtime_class"],
                                           "restore": "see restore_check"}}
    times = [v["grade_wall_seconds"] for v in grades.values() if v.get("grade_wall_seconds") is not None]
    record["timing"] = {"max_grade_wall_seconds": max(times) if times else None,
                        "median_grade_wall_seconds": sorted(times)[len(times) // 2] if times else None,
                        "reference_solve_seconds": ref_run["seconds"]}
    record["disk_at_end"] = E.docker_disk()
    if nospace:
        record["docker_disk"] = {"seen_in": [f"{label}: No space left on device" for label in nospace],
                                 "disk": record["disk_at_end"]}
    record["passed"] = all(i["passed"] for i in items.values()) and not nospace
    write_json(out / "gate.json", record)
    return record
