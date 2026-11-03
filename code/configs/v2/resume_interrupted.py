"""Recover an interrupted pre-submission natural run, preserving protocol and remaining guard.

--probe uses one harmless scripted command in a separate restored container (no API).
Default is a read-only preflight; --run starts the real worker. Source evidence is never edited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trajectory_retention.contracts import to_jsonable
from trajectory_retention.mining.branch_store import BranchStore, ReadOnlyArtifactStore
from trajectory_retention.mining.technical_resume import prepare


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


class ProbeComplete(BaseException):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--source-run', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prior-attempt', type=Path, action='append', default=[],
                   help='Completed interrupted continuation directory; charge its actual attempts too')
    p.add_argument('--background-capture-gaps', action='store_true',
                   help='Record unsupported background snapshots as non-branchable gaps (Q8)')
    p.add_argument('--current-conditions', action='store_true',
                   help='Bind the current submission feedback, submission stop and provider retry rules from '
                        'configs/v2/settings.json (sources that never submitted only)')
    p.add_argument('--continuation', type=Path,
                   help='Interrupted technical continuation directory (restart.json + store) to resume from its own '
                        'latest point; --store/--source-run name its original source')
    p.add_argument('--script', type=Path, help='Scripted validation only: JSON list of responses, zero provider calls')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--probe', action='store_true')
    a = p.parse_args()
    if a.script and not a.run:
        raise ValueError('--script requires --run and replaces the provider with ScriptedModel')
    source = ReadOnlyArtifactStore(a.store)
    source_run = a.source_run
    if a.continuation:
        from trajectory_retention.mining.branch_store import ReadOnlyBranchStore
        prev = json.loads((a.continuation / 'restart.json').read_text())
        if prev.get('source_run_id') != a.source_run:
            raise ValueError('--continuation does not continue --source-run')
        source = ReadOnlyBranchStore(a.continuation / 'store', source)
        source_run = prev['run_id']
    out = a.out.expanduser().resolve()
    run_id = 'technical-resume-v2:' + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    prior_attempts = []
    for path in a.prior_attempt:
        prior = json.loads((path / 'restart.json').read_text())
        prior_attempts.append({'store': str((path / 'store').resolve()), 'run_id': prior['run_id']})
    from trajectory_retention.mining.capture_gaps import POLICY as GAP_POLICY
    conditions = None
    if a.current_conditions:
        settings = json.loads((ROOT / 'configs/v2/settings.json').read_text())
        from trajectory_retention.experiments.submission_feedback import validate as feedback_rule
        from trajectory_retention.experiments.submission_stop import validate as stop_rule
        from trajectory_retention.provider_policy import unresolved_retry_rule
        conditions = {'submission_feedback': feedback_rule(settings['submission_feedback']),
                      'submission_stop': stop_rule(settings['submission_stop']),
                      'unresolved_retry': unresolved_retry_rule(settings['unresolved_retry'])}
    cp, config, spec, start = prepare(source, source_run, run_id, prior_attempts=prior_attempts,
                                    capture_gap_policy=GAP_POLICY if a.background_capture_gaps else None,
                                    conditions=conditions)
    from trajectory_retention.experiments.branch_v2 import _load_task, live_model_factory
    scratch = Path.home() / '.cache/trajectory-retention/technical-resume' / run_id.split(':')[1]
    task = _load_task(start, scratch / 'task')
    if to_jsonable(task.describe()) != start['task']:
        raise ValueError('task identity changed; refusing resume')
    receipt = {**spec, 'run_id': run_id, 'source_store': str(source.root), 'store': str(out / 'store'),
               'pid': os.getpid(), 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'probe': a.probe, 'paid_execution': a.run and not a.script, 'state': 'preflight_passed',
               **({'scripted_validation': True, 'script_sha256': hashlib.sha256(a.script.read_bytes()).hexdigest()}
                  if a.script else {}),
               'config': to_jsonable(config)}
    if not (a.run or a.probe):
        print(json.dumps(receipt, indent=2)); return
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'restart.json', receipt)
    store = BranchStore(out / 'store', source)
    from trajectory_retention.detector.core import DetectorConfig
    from trajectory_retention.execution.model import ScriptedModel
    from trajectory_retention.mining.runner import MiningRunner
    if a.script:
        script = json.loads(a.script.read_text())
        factory = lambda c: ScriptedModel(script, model_name=c.model, model_settings=dict(c.model_settings))
    elif a.probe:
        factory = lambda c: ScriptedModel([{'content': 'Restore integrity probe.',
            'commands': ["printf 'TECHNICAL_RESTORE_OK\\n'; pwd"],
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 100, 'cost': 0.0}}],
            model_name=c.model, model_settings=dict(c.model_settings))
    else:
        factory = live_model_factory()
    runner = MiningRunner(model_factory=factory, worker_image=task.describe().environment_version,
        detector_config=DetectorConfig(**config.detector_settings), scratch_root=scratch / 'runner', network='none')
    def checkpoint(c):
        write(out / 'status.json', {**receipt, 'state': 'running', 'checkpoint_id': c.checkpoint_id,
            'step': c.validation.get('step_index'), 'branchable': c.branchable,
            'updated_at': dt.datetime.now(dt.timezone.utc).isoformat()})
        if a.probe and c.validation.get('step_index', 0) > spec['restored_step']:
            raise ProbeComplete()
    try:
        result = runner.run(task, config, store=store, resume=cp, checkpoint_published=checkpoint)
    except ProbeComplete:
        evs = list(store.iter_events(run_id))
        restored = next(e.payload for e in evs if e.kind == 'lifecycle' and
                        e.payload.get('transition') == 'history_restored')
        checks = {k: v for k, v in restored['validation'].items() if k != 'revision'}
        checks['exactly_one_scripted_response'] = sum(e.kind == 'model_response' for e in evs) == 1
        checks['restorable_next_point'] = any(c.branchable and c.validation.get('step_index') ==
            spec['restored_step'] + 1 for c in store.list_checkpoints(run_id))
        checks['no_new_user_message'] = not any(e.kind in ('handback', 'treatment_packet', 'native_submission_feedback')
                                               for e in evs)
        report = {**receipt, 'state': 'probe_passed' if all(checks.values()) else 'probe_failed',
                  'provider_calls': 0, 'checks': checks, 'all_pass': all(checks.values())}
        write(out / 'result.json', report); write(out / 'status.json', report)
        print(json.dumps({'state': report['state'], 'checks': checks}, indent=2))
        if not report['all_pass']: raise SystemExit(1)
        return
    report = {**receipt, 'state': 'finished', 'result': to_jsonable(result),
              'finished_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(out / 'result.json', report); write(out / 'status.json', report)
    print(json.dumps({'state': 'finished', 'status': result.status, 'reason': result.termination_reason}))


if __name__ == '__main__':
    main()
