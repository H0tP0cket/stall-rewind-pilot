"""Campaign entry point. Scientific settings are explicit in a frozen input file."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys

from .campaign import Campaign, CampaignConfig, PhaseJournal
from .contracts import from_jsonable, to_jsonable
from .store import ArtifactStore
from .verification import verify_store


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _json_input(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value: " + value)
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


class _ManualPreview:
    """Keep deterministic rebuild artifacts in memory during read-only validation."""
    def __init__(self, store):
        self.store, self.blobs, self.records = store, {}, {}

    def __getattr__(self, key):
        return getattr(self.store, key)

    def put_bytes(self, data, **kwargs):
        import hashlib
        from .contracts import ArtifactRef
        kwargs.setdefault("media_type", "application/octet-stream")
        ref = ArtifactRef(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), **kwargs)
        self.blobs[ref.sha256] = data
        return ref

    def put_json(self, value, **kwargs):
        from .contracts import canonical_json
        return self.put_bytes(canonical_json(value), media_type="application/json", **kwargs)

    def put_record(self, kind, record_id, value, *, run_id=None):
        from .contracts import canonical_json
        try:
            previous = self.get_record(kind, record_id)
        except KeyError:
            pass
        else:
            if canonical_json(previous) != canonical_json(value):
                raise ValueError(f"Immutable manual source record changed: {kind}/{record_id}")
        self.records[kind, record_id] = to_jsonable(value)
        return self.put_json(value)

    def get_bytes(self, ref):
        return self.blobs[ref.sha256] if ref.sha256 in self.blobs else self.store.get_bytes(ref)

    def get_json(self, ref):
        return json.loads(self.get_bytes(ref))

    def get_record(self, kind, record_id):
        return self.records[kind, record_id] if (kind, record_id) in self.records else self.store.get_record(kind, record_id)


def _manual_command(args) -> int:
    from .contracts import TreatmentManifest
    from .experiments.manual import build_manual_manifest, validate_manual_manifest
    from .preflight import _ReadOnlyStore
    retained = _ReadOnlyStore(args.store)
    if getattr(args, "output", None) is not None and args.output.resolve().is_relative_to(retained.root):
        raise ValueError("Manual output must be outside the retained evidence store")
    if args.manual_command == "packet":
        # Registers one immutable research_packet record plus its exact worker-visible bytes; idempotent.
        # Treatment legality (run, checkpoint at/before the decision, fresh history) is checked at build.
        from .experiments.research_packets import register_research_packet
        record = register_research_packet(ArtifactStore(args.store), _json_input(args.record),
                                          args.text.read_bytes().decode("utf-8"))
        print(json.dumps({"packet_id": record["packet_id"], "label": record["label"],
                          "token_count": record["token_count"], "text_sha256": record["text_sha256"],
                          "source_checkpoint_id": record["source_checkpoint_id"]}, indent=2))
        return 0
    if args.manual_command == "build":
        # Building may add immutable derived packet/target records; validation
        # and completed readback never need a writable source store.
        spec = _json_input(args.spec)
        preview = _ManualPreview(retained)
        manifest = build_manual_manifest(preview, spec)
        problems = validate_manual_manifest(manifest, preview)
        if problems:
            raise ValueError("Manual manifest invalid: " + "; ".join(problems))
        manifest = build_manual_manifest(ArtifactStore(args.store), spec)
        _write_json(args.output, manifest)
        print(json.dumps({"valid": True, "treatment_id": manifest.treatment_id,
                          "manifest_path": str(args.output), "executed": False}, indent=2))
        return 0
    manifest = from_jsonable(TreatmentManifest, _json_input(args.manifest))
    if args.manual_command == "validate":
        for ref in manifest.packet_refs:
            retained.get_bytes(ref)
        problems = validate_manual_manifest(manifest, _ManualPreview(retained))
        report = {"valid": not problems, "problems": problems, "treatment_id": manifest.treatment_id, "executed": False}
        if args.output is not None:
            _write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if not problems else 1
    from .manual_execution import manual_run_config, reuse_manual_experiment, run_manual_experiment
    horizons = tuple(args.horizon)
    previous = reuse_manual_experiment(retained, manifest, experiment_id=args.experiment_id,
                                       horizons=horizons, export_dir=args.output)
    if previous is not None:
        print(json.dumps(to_jsonable(previous), indent=2))
        return 0 if previous["pipeline_status"] == "outcomes_recorded" else 1
    problems = validate_manual_manifest(manifest, _ManualPreview(retained))
    if problems:
        raise ValueError("Manual manifest invalid: " + "; ".join(problems))
    run_config = manual_run_config(_ManualPreview(retained), manifest, args.experiment_id)
    config = CampaignConfig(campaign_id=args.experiment_id, mining=run_config, mode="mine")
    from .registry import read_registry
    from .launch_sources import verify_declared_sources, verify_loaded_task
    entries = read_registry(args.registry, root=args.artifact_root)
    if args.task_id not in entries:
        raise ValueError("Task ID not present in registry")
    verify_declared_sources(config, entries[args.task_id])
    task = entries[args.task_id].load(scratch_root=args.scratch / "grader", purpose=run_config.purpose)
    verify_loaded_task(config, task.describe())
    source = retained.get_checkpoint(manifest.source_checkpoint_id)
    start = next(e.payload for e in retained.iter_events(source.run_id)
                 if e.kind == "lifecycle" and e.payload.get("transition") == "run_started")
    if start.get("task") != to_jsonable(task.describe()):
        raise ValueError("Manual recovery task differs from the recorded natural source")
    try:
        source_campaign = retained.get_record("campaign", start["config"]["campaign_id"])
    except KeyError:
        source_campaign = {}
    if source_campaign.get("fixture"):
        raise ValueError("Live manual execution refuses a scripted fixture campaign")
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    from .factory import live_campaign
    store = ArtifactStore(args.store)
    live = live_campaign(store, task, config, scratch=args.scratch)
    result = run_manual_experiment(store, task, live.runner, manifest, experiment_id=args.experiment_id,
                                  horizons=horizons, export_dir=args.output)
    print(json.dumps(to_jsonable(result), indent=2))
    return 0 if result["pipeline_status"] == "outcomes_recorded" else 1


@contextmanager
def _batch_drain_signals(token):
    """Signals request orderly draining; always restore the caller's handlers."""
    installed = {}
    def request(signum, frame):
        token.request(signal.Signals(signum).name)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            signal.signal(signum, request)
            installed[signum] = previous
        yield
    finally:
        for signum, previous in installed.items():
            signal.signal(signum, previous)


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"trajectory-retention: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trajectory-retention")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run/resume one declared admitted-task campaign")
    run.add_argument("--config", type=Path, required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--package", type=Path, help="Legacy explicit file_backup package")
    source.add_argument("--task-id", help="Admitted task identity from the registry")
    run.add_argument("--registry", type=Path, default=Path("benchmark_tasks/admission/registry-mettlebench-v2.json"))
    run.add_argument("--artifact-root", type=Path)
    run.add_argument("--store", type=Path, required=True)
    run.add_argument("--scratch", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--env-file", type=Path, help="Load credentials locally; values never enter config/artifacts")
    batch = sub.add_parser("batch", help="Execute a declared cohort with bounded independent-job concurrency")
    batch.add_argument("--plan", type=Path, required=True)
    batch.add_argument("--registry", type=Path, default=Path("benchmark_tasks/admission/registry-mettlebench-v2.json"))
    batch.add_argument("--artifact-root", type=Path)
    for key in ("store", "scratch", "output"):
        batch.add_argument("--" + key, type=Path, required=True)
    batch.add_argument("--env-file", type=Path)
    mining_plan = sub.add_parser("mining-plan", help="Prepare or validate an offline mining-first plan; never launch work")
    mining_commands = mining_plan.add_subparsers(dest="plan_command", required=True)
    template = mining_commands.add_parser("template", help="Copy declared cohort/model identities with unresolved settings as null")
    template.add_argument("--cohort", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    validate = mining_commands.add_parser("validate", help="Report every unresolved/invalid setting without execution")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    build = mining_commands.add_parser("build", help="Write a BatchPlan only from a fully explicit mining plan")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    recovery_plan = sub.add_parser("recovery-plan", help="Prepare recovery from verified saved mines; never launch work")
    recovery_commands = recovery_plan.add_subparsers(dest="plan_command", required=True)
    recovery_template = recovery_commands.add_parser("template", help="Copy the declared cohort; leave recovery, observer and Q14 choices unresolved")
    recovery_template.add_argument("--cohort", type=Path, required=True)
    recovery_template.add_argument("--output", type=Path, required=True)
    for name in ("validate", "build"):
        command = recovery_commands.add_parser(name, help="Validate saved sources read-only" if name == "validate"
            else "Write a BatchPlan only after explicit settings and saved-source verification")
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--store", type=Path, help="Existing saved-mine evidence; without it the plan remains nonlaunchable")
        command.add_argument("--output", type=Path, required=name == "build")
    manual = sub.add_parser("manual", help="Build, validate or execute one explicit manual experiment")
    manual_commands = manual.add_subparsers(dest="manual_command", required=True)
    manual_build = manual_commands.add_parser("build", help="Resolve retained donors and save deterministic packet views; no model calls")
    manual_build.add_argument("--spec", type=Path, required=True)
    manual_build.add_argument("--store", type=Path, required=True)
    manual_build.add_argument("--output", type=Path, required=True)
    manual_packet = manual_commands.add_parser("packet", help="Register one immutable external research packet "
                                               "(exact worker-visible bytes and provenance); no model calls")
    manual_packet.add_argument("--record", type=Path, required=True, help="Packet body JSON (research-packet.v1 fields)")
    manual_packet.add_argument("--text", type=Path, required=True, help="Exact UTF-8 packet text delivered to the worker")
    manual_packet.add_argument("--store", type=Path, required=True)
    manual_validate = manual_commands.add_parser("validate", help="Check a manifest against its immutable source evidence")
    manual_validate.add_argument("--manifest", type=Path, required=True)
    manual_validate.add_argument("--store", type=Path, required=True)
    manual_validate.add_argument("--output", type=Path)
    manual_run = manual_commands.add_parser("run", help="Execute or read back one bounded manual experiment")
    manual_run.add_argument("--manifest", type=Path, required=True)
    manual_run.add_argument("--experiment-id", required=True)
    manual_run.add_argument("--horizon", type=int, action="append", required=True)
    manual_run.add_argument("--task-id", required=True)
    manual_run.add_argument("--registry", type=Path, default=Path("benchmark_tasks/admission/registry-mettlebench-v2.json"))
    manual_run.add_argument("--artifact-root", type=Path)
    for key in ("store", "scratch", "output"):
        manual_run.add_argument("--" + key, type=Path, required=True)
    manual_run.add_argument("--env-file", type=Path)
    preflight = sub.add_parser("preflight", help="Check offline launch bindings without worker/observer calls")
    preflight_source = preflight.add_mutually_exclusive_group(required=True)
    preflight_source.add_argument("--config", type=Path)
    preflight_source.add_argument("--plan", type=Path)
    preflight.add_argument("--task-id", help="Required with --config; batches declare task IDs in their plan")
    preflight.add_argument("--registry", type=Path, default=Path("benchmark_tasks/admission/registry-mettlebench-v2.json"))
    preflight.add_argument("--artifact-root", type=Path)
    preflight.add_argument("--store", type=Path, help="Existing store containing main-study freeze evidence")
    preflight.add_argument("--output", type=Path, help="Optional JSON report file; stdout always receives the report")
    tasks = sub.add_parser("tasks", help="Inspect registered admission status without model calls")
    tasks.add_argument("--registry", type=Path, default=Path("benchmark_tasks/admission/registry-mettlebench-v2.json"))
    tasks.add_argument("--artifact-root", type=Path)
    for name in ("verify", "inspect", "status"):
        command = sub.add_parser(name)
        command.add_argument("--store", type=Path, required=True)
        if name == "status":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--launch-plan", type=Path, action="append", default=[], help="Optional pending mining plan; repeat for multiple plans")
            command.add_argument("--admissions", type=Path, action="append", default=[], help="Optional JSON array of task_id/admission rows; repeat for multiple files")
    args = parser.parse_args(argv)
    if args.command == "manual":
        return _manual_command(args)
    if args.command == "mining-plan":
        from .launch_plan import (build_mining_batch, load_mining_launch_plan,
                                  mining_launch_template, validate_mining_launch_plan)
        if args.plan_command == "template":
            cohort = load_mining_launch_plan(args.cohort)
            plan = mining_launch_template(cohort)
            report = validate_mining_launch_plan(plan)
            _write_json(args.output, plan)
            report["template_path"] = str(args.output)
        else:
            plan = load_mining_launch_plan(args.plan)
            report = validate_mining_launch_plan(plan)
            if args.plan_command == "build" and report["launchable"]:
                built = build_mining_batch(plan)
                _write_json(args.output, built)
                report.update(batch_plan_path=str(args.output), campaign_count=len(built.jobs))
            elif args.plan_command == "validate" and args.output:
                _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["launchable"] else 1
    if args.command == "recovery-plan":
        from .launch_plan import (build_recovery_batch, load_recovery_launch_plan,
                                  recovery_launch_template, validate_recovery_launch_plan)
        from .preflight import _ReadOnlyStore
        if args.plan_command == "template":
            plan = recovery_launch_template(load_recovery_launch_plan(args.cohort))
            report = validate_recovery_launch_plan(plan)
            _write_json(args.output, plan)
            report["template_path"] = str(args.output)
        else:
            plan = load_recovery_launch_plan(args.plan)
            store = _ReadOnlyStore(args.store) if args.store is not None else None
            if store is not None and args.output is not None and args.output.resolve().is_relative_to(store.root):
                raise ValueError("Recovery plan output must be outside the read-only saved-mine store")
            report = validate_recovery_launch_plan(plan, store=store)
            if args.plan_command == "build" and report["launchable"]:
                built = build_recovery_batch(plan, store=store)
                _write_json(args.output, built)
                report.update(batch_plan_path=str(args.output), campaign_count=len(built.jobs))
            elif args.plan_command == "validate" and args.output:
                _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["launchable"] else 1
    if args.command == "preflight":
        from .preflight import preflight_campaigns
        report = preflight_campaigns(config=args.config, plan=args.plan, task_id=args.task_id,
                                    registry=args.registry, artifact_root=args.artifact_root, store=args.store)
        encoded = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        return 0 if report["passed"] else 1
    if args.command == "tasks":
        from .registry import read_registry
        entries = read_registry(args.registry, root=args.artifact_root)
        print(json.dumps([t.entry for t in entries.values()], indent=2))
        return 0
    if args.command == "status":
        from .launch_plan import load_mining_launch_plan
        from .preflight import _ReadOnlyStore
        from .status import write_status
        admissions = []
        for path in args.admissions:
            rows = _json_input(path)
            if not isinstance(rows, list) or any(not isinstance(row, dict) or
                    not isinstance(row.get("task_id"), str) or not row["task_id"].strip() or
                    not isinstance(row.get("admission"), str) or
                    row.get("admission") not in {"ADMITTED", "HOLD", "PENDING", "REJECTED", "not_recorded"} for row in rows):
                raise ValueError(f"{path}: expected an array of task_id/admission rows")
            admissions.extend(rows)
        store = _ReadOnlyStore(args.store)
        protected = [args.output.resolve()]
        if args.output.suffix != ".json":
            protected.append(args.output.with_suffix(".json").resolve())
        if any(path == store.database or path.is_relative_to(store.blobs) for path in protected):
            raise ValueError("Status output must not overwrite the journal or immutable blobs")
        print(write_status(store, args.output,
            launch_plans=[load_mining_launch_plan(path) for path in args.launch_plan], admissions=admissions))
        return 0
    store = ArtifactStore(args.store)
    if args.command == "verify":
        report = verify_store(store)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    if args.command == "inspect":
        reports = [store.get_record("campaign_result", row["record_id"])
                   for row in store.list_records("campaign_result")]
        phases = {"started": len(store.list_records("phase_started")),
                  "finished": len(store.list_records("phase_finished"))}
        print(json.dumps({"campaigns": reports, "phases": phases}, indent=2))
        return 0
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if args.command == "batch":
        from .factory import BatchCancellation, BatchPlan, run_batch
        plan = from_jsonable(BatchPlan, _json_input(args.plan))
        if any(job.config.fixture for job in plan.jobs):
            parser.error("batch invokes real providers; fixture plans are for explicit host/integration tests")
        token = BatchCancellation()
        with _batch_drain_signals(token):
            result = run_batch(plan, registry=args.registry, artifact_root=args.artifact_root, store=store,
                               scratch=args.scratch, output=args.output, cancellation=token)
        print(json.dumps(to_jsonable(result), indent=2))
        return 0 if result["passed"] else 1
    config = from_jsonable(CampaignConfig, _json_input(args.config))
    if config.fixture:
        parser.error("run invokes a real provider; fixture campaigns belong to the fixture integration tests")
    from .campaign import reuse_completed_campaign
    previous = reuse_completed_campaign(store, config, export_dir=args.output)
    if previous is not None:
        print(json.dumps(to_jsonable(previous), indent=2))
        return 0 if previous["pipeline_status"] == "outcomes_recorded" else 1
    from .launch_sources import verify_declared_sources, verify_loaded_task
    if args.task_id:
        from .registry import read_registry
        entries = read_registry(args.registry, root=args.artifact_root)
        if args.task_id not in entries:
            parser.error("Task ID not present in registry")
        verify_declared_sources(config, entries[args.task_id])
        task = entries[args.task_id].load(scratch_root=args.scratch / "grader", purpose=config.mining.purpose)
    else:
        if config.mining.metadata.get("source_binding") is not None:
            raise ValueError("Pinned launch source binding requires --task-id and its registry")
        from .tasks.file_backup import FileBackupTaskAdapter
        task = FileBackupTaskAdapter(args.package, scratch_root=args.scratch / "grader")
    if config.mining.metadata.get("source_binding") is not None:
        verify_loaded_task(config, task.describe())
    from .factory import live_campaign
    result = live_campaign(store, task, config, scratch=args.scratch).run(config, export_dir=args.output)
    report = verify_store(store)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "integrity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(to_jsonable(result), indent=2))
    return 0 if report["valid"] and result["pipeline_status"] == "outcomes_recorded" else 1


if __name__ == "__main__":
    sys.exit(main())
