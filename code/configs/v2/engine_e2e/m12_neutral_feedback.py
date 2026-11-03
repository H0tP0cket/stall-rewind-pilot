"""Real runner/restore checks of neutral feedback; scripted models, zero API calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from common import ARTIFACTS, ROOT, SUBMIT, events, lifecycle, requests, step, write
import m8_submission_stop as fixture
from trajectory_retention.experiments.submission_feedback import RULE_V1, SETTING, TEXTS, text, validate

TEXT = text(SETTING)  # current rule's delivered text (v2)
TEXT_V1 = TEXTS[RULE_V1]
from trajectory_retention.experiments.branch_v2 import BranchPlan, BranchStore, ReadOnlyArtifactStore, run_branch
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.mining.points import point_checkpoints


def main():
    checks = {}
    neutral = fixture.run('neutral', [7, 8, 9, 0], feedback=SETTING)
    neutral_v1 = fixture.run('neutral-v1', [7, 8, 9, 0], feedback={'rule': RULE_V1})
    old = fixture.run('old-count', [7, 8, 9, 0])
    checks['current_rule_is_v2'] = SETTING == {'rule': 'neutral-submission-feedback/v2'} and \
        TEXT == 'The task is not complete yet. Continue working.'
    for name, r in [('neutral', neutral), ('neutral_v1', neutral_v1), ('old', old)]:
        checks[name + '_private_counts'] = [v['failing_count'] for v in r['verdicts']] == [7, 8, 9]
        checks[name + '_q56'] = r['terminal'] == 'sequence_complete:native_no_improvement'
        checks[name + '_counter'] = [v['no_improvement_count'] for v in r['verdicts']] == [1, 2, 3]
        deliveries = [r['store'].get_bytes(e.delivered_ref).decode() for e in r['events']
                      if e.kind == 'native_submission_feedback']
        checks[name + '_deliveries'] = deliveries == {'neutral': [TEXT, TEXT], 'neutral_v1': [TEXT_V1, TEXT_V1],
            'old': ['Hidden tests still failing: 7.', 'Hidden tests still failing: 8.']}[name]
    checks['neutral_next_requests'] = all('Hidden tests still failing:' not in q for r in (neutral, neutral_v1)
        for q in requests(r['store'], r['events']))
    sys.path.insert(0, str(ROOT / 'configs/v2'))
    import arms_v2
    arms_v2.configure_arm_set('v3')
    results = []
    for name, source, override in [('inherited', neutral, False), ('historical', old, True)]:
        root = source['store'].root
        points = point_checkpoints(source['store'], source['run_id'])
        stall = {'stall_id': 'neutral-' + name, 'store': str(root), 'run_id': source['run_id'],
                 'checkpoint_id': points[2], 'step_index': 2}
        spec = arms_v2.arm_spec_v3(stall, '4_forced_rewind_to_p', 0, purpose='tuning', prefix='m12',
                                  caps=None, onset_point=0)
        if not override:
            spec.pop('submission_feedback')  # engine must inherit the neutral source even for old spec shape
        fixture.COUNTS, fixture.SAME_SET = [7, 8, 9, 0], False
        model = ScriptedModel([step('Submit.', [SUBMIT]) for _ in fixture.COUNTS], model_name='scripted/fixture')
        b = run_branch(spec, out_dir=fixture.OUT / ('branch-' + name), model_factory=lambda c: model)
        store = BranchStore(Path(b['store']), ReadOnlyArtifactStore(root))
        evs = events(store, b['run_id'])
        vs = lifecycle(evs, 'native_submission_verdict')
        checks[name + '_branch_q56'] = b['end_reason'] == 'no_improvement' and len(vs) == 3
        checks[name + '_branch_private_counts'] = [e.payload['failing_count'] for e in vs] == [7, 8, 9]
        checks[name + '_branch_neutral_deliveries'] = [store.get_bytes(e.delivered_ref).decode()
            for e in evs if e.kind == 'native_submission_feedback'] == [TEXT, TEXT]
        reqs = requests(store, evs)
        checks[name + '_rewind_no_count_leak'] = bool(reqs) and all('hidden tests failing' not in q.lower()
            and 'hidden tests still failing' not in q.lower() and 'failing_count' not in q for q in reqs)
        checks[name + '_submission_record_retained'] = 'Submission 1 (step 1).' in reqs[0]
        # Prefix-only judge record must recognize neutral delivery, without deriving its count from private grades.
        sys.path.insert(0, str(ROOT / 'configs/v2/judge_v2'))
        from record import RunView
        view = RunView(source['store'], source['run_id'])
        if name == 'inherited':
            parsed = view.record(2)
            checks['neutral_judge_record'] = (TEXT in parsed.render() and not parsed.issues and
                all(s['failing'] is None for s in parsed.submissions()) and
                'Hidden tests still failing' not in parsed.render())
        results.append({'name': name, 'run_id': b['run_id'], 'store': b['store'], 'end_reason': b['end_reason']})
    for bad in ({'rule': 'typo'}, {'rule': SETTING['rule'], 'text': '7 failing'}, {'rule': RULE_V1, 'text': TEXT}):
        try:
            validate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed policy accepted')
    report = {'provider_calls': 0, 'fixture': 'scripted counts; real Docker execution, grading and restores',
              'policy': SETTING, 'store_root': str(fixture.OUT), 'branches': results, 'checks': checks,
              'all_pass': all(checks.values())}
    write(ARTIFACTS / 'm12-neutral-feedback.json', report)
    print(json.dumps(report, indent=2))
    if not report['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
