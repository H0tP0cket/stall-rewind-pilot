"""Per-task technical qualification (no model/provider calls).

Gates (see docs/research/mettlebench-admission/README.md):
  natural_start       materialized W equals the shipped archive; no extra files
  worker_isolation    live scripted worker sees no grader/witness paths or private markers
  reference_full      private witness grades 1.0 and native pass
  start_low           untouched start < 1.0 with native fail
  partial_between     a witness step prefix scores strictly between start and 1.0
  deterministic       start and reference regraded with identical per-check statuses
  visible_feedback    native submission feedback delivered through the actual runner
  runner_round_trip   resume from a mid-run checkpoint regrades identically
  private_absent      worker requests contain no research-private grade material
Grade wall time is recorded; > 120 s is flagged, not failed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

from . import source as S
from .adapter import MettleBenchTaskAdapter
from .harness import image_id, run_and_grade, split_steps
from .package import build_package, sha, verify_package, write_json

REPO = Path(__file__).resolve().parents[4]
REFERENCES = REPO / "benchmark_tasks/mettlebench-references"
PACKAGES = REPO / "benchmark_tasks/packages"
ADMISSION = REPO / "benchmark_tasks/admission"
ARTIFACTS = REPO / "artifacts/task-admission/mettlebench"
QUALIFICATION_VERSION = "mettlebench.qualification.v1"
PRIVATE_MARKERS = ("criterion::", "observed_status", "/mettle-grader", "mettle-witness", "verifier_checklist",
                   "_verifier_v2", "failure_feedback")
ISOLATION_PROBE = ("for p in /mettle-grader /tmp/.mettle-witness*; do [ -e \"$p\" ] && echo MB_PRIVATE_PATH:$p; done; "
                   "grep -rIl --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev -e _verifier_v2 -e verifier_checklist "
                   "-e failure_feedback / 2>/dev/null | head -5 | sed 's/^/MB_PRIVATE_MARKER:/'; "
                   "echo probe > /tmp/mb-isolation-probe.txt; echo MB_ISOLATION_PROBE_DONE")


def _tracked(path: Path) -> bool:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(path / "package.json")],
                         capture_output=True)
    return out.returncode == 0


def ensure_package(upstream_id: str) -> Path:
    task_id = S.package_task_id(upstream_id)
    package = PACKAGES / task_id
    reference = REFERENCES / upstream_id / "solve.sh"
    if not reference.is_file():
        raise ValueError("No private witness for " + upstream_id)
    ids = [r["task_id"] for r in S.verify_source()["manifest"]]
    split = S.family_split(ids)[S.family_for(upstream_id)]
    meta_path = REFERENCES / upstream_id / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None
    if package.exists():
        if _tracked(package):
            manifest = verify_package(package)
            if manifest["reference"]["sha256"] != sha(reference):
                raise ValueError("Committed package differs from the current witness; issue a new package version")
            return package
        shutil.rmtree(package)
    build_package(upstream_id, package, image_id=image_id(upstream_id), reference=reference, split=split, reference_meta=meta)
    return package


def natural_start(adapter: MettleBenchTaskAdapter) -> dict[str, Any]:
    archive = adapter.package / "public/workspace.tar.gz"
    expected: dict[str, Any] = {}
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel:
                continue
            if m.isreg():
                expected[rel] = ("f", hashlib.sha256(tf.extractfile(m).read()).hexdigest())
            elif m.issym():
                expected[rel] = ("l", m.linkname)
            elif m.isdir():
                expected[rel] = ("d", None)
    tmp = Path(tempfile.mkdtemp(prefix="mettle-natural-"))
    try:
        ws = tmp / "W"
        adapter.materialize(ws)
        actual: dict[str, Any] = {}
        for p in sorted(ws.rglob("*")):
            rel = p.relative_to(ws).as_posix()
            if p.is_symlink():
                actual[rel] = ("l", str(Path(p.readlink())))
            elif p.is_dir():
                actual[rel] = ("d", None)
            else:
                actual[rel] = ("f", sha(p))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    private = {sha(p) for p in (adapter.package / "private").rglob("*") if p.is_file()}
    public = {v[1] for v in expected.values() if v[0] == "f"}
    leaked = sorted(k for k, v in actual.items() if v[0] == "f" and v[1] in private and v[1] not in public)
    return {"equal_to_archive": actual == expected, "entries": len(actual),
            "extra": sorted(set(actual) - set(expected))[:20], "missing": sorted(set(expected) - set(actual))[:20],
            "private_file_copies_in_W": leaked,
            "note": "Private files byte-identical to shipped public files (e.g. pristine test copies) are not leaks."}


def _call(command: str) -> dict:
    return {"content": "scripted admission control", "commands": [command],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0}}


def runner_control(adapter: MettleBenchTaskAdapter, script: str, out: Path) -> dict[str, Any]:
    from trajectory_retention.contracts import RunConfig
    from trajectory_retention.detector.core import DetectorConfig
    from trajectory_retention.execution.model import ScriptedModel
    from trajectory_retention.mining.runner import MiningRunner
    from trajectory_retention.store import ArtifactStore

    header, steps = split_steps(script)
    first = header + steps[0]
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    # step 1 (graded), submission (native feedback delivered, session reopened), isolation probe.
    entries = [_call(first), _call(submit), _call(ISOLATION_PROBE)]
    model = ScriptedModel(entries, model_name="scripted/mettlebench-admission")
    store = ArtifactStore(out / "store")
    detector = DetectorConfig(stagnation_token_threshold=100000)
    image = adapter.manifest["environment_image_id"]
    common = dict(campaign_id="mettlebench-admission-controls", purpose="development",
                  model="scripted/mettlebench-admission", model_settings={}, generated_token_limit=100000,
                  tool_timeout_seconds=1800.0)
    runner = MiningRunner(model_factory=lambda config: model, worker_image=image, detector_config=detector,
                          scratch_root=out / "scratch", network="none")
    run_id = "natural-" + adapter.manifest["task_id"]
    result = runner.run(adapter, RunConfig(run_id=run_id, max_model_calls=len(entries), **common), store=store)
    events = list(store.iter_events(run_id))
    verdicts = [e.payload for e in events if e.kind == "lifecycle" and e.payload.get("transition") == "native_submission_verdict"]
    deliveries = [e for e in events if e.kind == "native_submission_feedback"]
    requests = [e for e in events if e.kind == "model_request"]
    bodies = [store.get_bytes(e.input_ref).decode("utf-8", errors="replace") for e in requests]
    delivered_texts = [store.get_bytes(e.delivered_ref).decode() for e in deliveries]
    report: dict[str, Any] = {"status": result.status, "termination_reason": result.termination_reason,
                              "model_calls": model.index, "checkpoints": len(result.checkpoint_ids),
                              "native_verdicts": verdicts, "feedback_deliveries": len(deliveries)}
    # Every delivered feedback string equals the native feedback of the graded submission and reaches the next request.
    checkpoints = {c.checkpoint_id: c for c in store.list_checkpoints(run_id)}
    feedback_ok = bool(verdicts)
    for verdict, text in zip([v for v in verdicts if v["outcome"] == "continue"], delivered_texts):
        ev = store.get_json(checkpoints[verdict["checkpoint_id"]].evaluation_ref)
        feedback_ok &= text == (ev["details"]["native"] or {}).get("feedback")
        feedback_ok &= any(json.dumps(text)[1:-1] in body for body in bodies)
    report["native_feedback_delivered"] = feedback_ok and len(deliveries) == sum(v["outcome"] == "continue" for v in verdicts)
    leaks = sorted({m for m in PRIVATE_MARKERS for body in bodies if m in body})
    report["private_markers_in_requests"] = leaks
    probe = [e for e in events if e.kind == "tool_result" and e.output_ref is not None]
    probe_text = store.get_bytes(probe[-1].output_ref).decode(errors="replace") if probe else ""
    report["isolation_probe"] = probe_text[-2000:]
    report["worker_isolation"] = ("MB_ISOLATION_PROBE_DONE" in probe_text and "MB_PRIVATE_PATH" not in probe_text
                                  and "MB_PRIVATE_MARKER" not in probe_text) if model.index == len(entries) else None
    # Restore round trip from a mid-run graded, branchable, non-initial checkpoint.
    graded = [c for c in store.list_checkpoints(run_id) if c.evaluation_ref and c.branchable and c.ordinal > 0]
    graded = [c for c in graded if store.get_json(c.evaluation_ref)["valid"]]
    report["branchable_graded_checkpoints"] = [c.checkpoint_id for c in graded]
    if graded:
        source = graded[len(graded) // 2]
        source_eval = store.get_json(source.evaluation_ref)
        resume_model = ScriptedModel([_call("true")], model_name="scripted/mettlebench-admission")
        runner2 = MiningRunner(model_factory=lambda config: resume_model, worker_image=image, detector_config=detector,
                               scratch_root=out / "scratch-resume", network="none")
        resume_id = "resume-" + adapter.manifest["task_id"]
        result2 = runner2.run(adapter, RunConfig(run_id=resume_id, max_model_calls=1, **common), store=store,
                              resume=store.get_checkpoint(source.checkpoint_id))
        restored = [c for c in store.list_checkpoints(resume_id) if "branch_start" in c.reasons]
        report["resume"] = {"status": result2.status, "termination_reason": result2.termination_reason,
                            "source_checkpoint": source.checkpoint_id, "source_reasons": list(source.reasons)}
        if restored and restored[0].evaluation_ref:
            restored_eval = store.get_json(restored[0].evaluation_ref)
            key = lambda e: ([(c["check_id"], c["status"], c["raw_status"]) for c in e["checks"]], e["score"],
                             (e["details"].get("native") or {}).get("first_failure_id"))
            report["resume"].update(source_score=source_eval["score"], restored_score=restored_eval["score"],
                                    identical=key(source_eval) == key(restored_eval),
                                    restored_workspace_digest=restored[0].validation.get("workspace_fidelity_digest"),
                                    source_workspace_digest=source.validation.get("workspace_fidelity_digest"))
            restored_events = list(store.iter_events(resume_id))
            report["resume"]["history_restored"] = any(e.kind == "lifecycle" and e.payload.get("transition") == "history_restored"
                                                       and all(e.payload["validation"][k] for k in e.payload["validation"] if k != "revision")
                                                       for e in restored_events)
    report["store"] = str(out / "store")
    return report


def feedback_commands(prompt: str) -> list[str]:
    found = re.findall(r"`([^`]*(?:python|pytest|make|bash |sh |\./)[^`]*)`", prompt)
    return list(dict.fromkeys(c.strip() for c in found))[:8]


def qualify(upstream_id: str) -> dict[str, Any]:
    package = ensure_package(upstream_id)
    adapter = MettleBenchTaskAdapter(package, scratch_root=ARTIFACTS / "scratch")
    task_id = adapter.manifest["task_id"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ARTIFACTS / task_id / stamp
    out.mkdir(parents=True)
    script = (package / "private/reference/solve.sh").read_text()
    n_steps = len(split_steps(script)[1])
    report: dict[str, Any] = {"version": QUALIFICATION_VERSION, "task_id": task_id, "upstream_task_id": upstream_id,
                              "package_sha256": sha(package / "package.json"), "image_id": adapter.manifest["environment_image_id"],
                              "family_id": adapter.manifest["family_id"], "family_split": adapter.manifest["family_split"],
                              "criteria": adapter.manifest["criteria"], "witness_steps": n_steps, "started": stamp,
                              "provider_calls": 0}
    report["natural_start"] = natural_start(adapter)

    def clean(r):
        r = dict(r)
        r.pop("_evaluation", None)
        return r
    start = clean(run_and_grade(adapter, None, out / "start", regrade=True))
    reference = clean(run_and_grade(adapter, script, out / "reference", regrade=True))
    report["start"], report["reference"] = start, reference
    partial = None
    tried = []
    meta_path = package / "private/reference/meta.json"
    hint = json.loads(meta_path.read_text()).get("partial_steps") if meta_path.is_file() else None
    order = ([hint] if isinstance(hint, int) and 0 < hint < n_steps else []) + [k for k in range(1, n_steps) if k != hint]
    for k in order:
        r = clean(run_and_grade(adapter, script, out / f"partial-{k}", steps=k, regrade=True))
        tried.append({"steps": k, "score": r["grade"]["score"], "deterministic": r["deterministic"]})
        if r["grade"]["valid"] and start["grade"]["score"] < r["grade"]["score"] < 1.0:
            partial = r
            break
    report["partial_search"] = tried
    report["partial"] = partial
    report["runner_control"] = runner_control(adapter, script, out / "runner")
    prompt = (package / "public/instruction.md").read_text()
    report["visible_feedback"] = {"native_submission_feedback": "first unsatisfied criterion feedback on each submission",
                                  "prompt_named_commands": feedback_commands(prompt)}
    rc = report["runner_control"]
    gates = {
        "natural_start": report["natural_start"]["equal_to_archive"] and not report["natural_start"]["private_file_copies_in_W"],
        "worker_isolation": rc.get("worker_isolation") is not False and not rc["private_markers_in_requests"],
        "reference_full": bool(reference["grade"]["valid"] and reference["grade"]["score"] == 1.0
                               and (reference["grade"]["native"] or {}).get("status") == "pass"),
        "start_low": bool(start["grade"]["valid"] and start["grade"]["score"] < 1.0
                          and (start["grade"]["native"] or {}).get("status") != "pass"),
        "partial_between": partial is not None,
        "deterministic": bool(start.get("deterministic") and reference.get("deterministic")
                              and (partial is None or partial.get("deterministic"))),
        "native_consistency": all((g["grade"].get("consistency") or {}).get("consistent") for g in [start, reference] + ([partial] if partial else [])),
        "visible_feedback": bool(rc.get("native_feedback_delivered")),
        "runner_round_trip": bool((rc.get("resume") or {}).get("identical") and (rc.get("resume") or {}).get("history_restored")),
        "private_absent": not rc["private_markers_in_requests"],
    }
    report["gates"] = gates
    report["grade_seconds"] = {"start": start["grade"]["grade_seconds"], "reference": reference["grade"]["grade_seconds"],
                               "reference_wall": reference["grade_wall_seconds"]}
    report["slow_grade_flag"] = (reference["grade"]["grade_seconds"] or 0) > 120
    report["passed"] = all(gates.values())
    report["summary"] = {"start": start["grade"]["score"], "partial": partial["grade"]["score"] if partial else None,
                         "partial_steps": partial["steps"] if partial else None, "reference": reference["grade"]["score"],
                         "grade_seconds": reference["grade"]["grade_seconds"], "gates_failed": [k for k, v in gates.items() if not v]}
    write_json(out / "qualification.json", report)
    compact = {k: report[k] for k in ("version", "task_id", "upstream_task_id", "package_sha256", "image_id", "family_id",
                                      "family_split", "criteria", "witness_steps", "started", "provider_calls",
                                      "gates", "passed", "summary", "grade_seconds", "slow_grade_flag", "partial_search",
                                      "visible_feedback")}
    compact["natural_start"] = {k: report["natural_start"][k] for k in ("equal_to_archive", "entries", "private_file_copies_in_W")}
    compact["statuses"] = {"start": start["grade"]["statuses"], "reference": reference["grade"]["statuses"],
                           "partial": partial["grade"]["statuses"] if partial else None}
    compact["native"] = {"start": start["grade"]["native"], "reference": reference["grade"]["native"]}
    compact["runner_control"] = {k: rc.get(k) for k in ("status", "termination_reason", "model_calls", "checkpoints",
                                                         "feedback_deliveries", "native_feedback_delivered",
                                                         "private_markers_in_requests", "worker_isolation", "resume")}
    compact["runner_control"]["native_verdicts"] = [{k: v.get(k) for k in ("ordinal", "status", "criterion_id",
                                                                           "consecutive_same", "outcome")}
                                                    for v in rc["native_verdicts"]]
    compact["local_evidence"] = str(out.relative_to(REPO))
    write_json(ADMISSION / task_id / "qualification.json", compact)
    report["compact_path"] = str(ADMISSION / task_id / "qualification.json")
    return report
