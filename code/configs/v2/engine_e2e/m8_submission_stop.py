"""Q56 E2E: real Docker execution/capture/grading with scripted models and submission-verdict fixtures.

The fixture replaces only the adapter's verdict to exercise exact count sequences (not measured task scores).
Checks driver/settings/metadata, feedback delivery, terminal status, branch inheritance/reset, and old-plan behaviour.
No provider calls; sealed task bytes untouched. Failure modes: off-by-one, reset to zero, regression treated as
progress, changing failure sets bypassing the stop, stopping a passing submission, missing counts treated as zero,
inheriting source counters into a branch, and silently changing old plans.
"""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from pathlib import Path
from common import ARTIFACTS, ROOT, SCRATCH, REGISTRY_V1, TASK, SUBMIT, events, lifecycle, requests, step, write
from trajectory_retention.experiments.v2_fixture import CountFeedbackFixture, FIXTURES
from trajectory_retention.experiments.submission_stop import SETTING
from trajectory_retention.execution.model import ScriptedModel
from trajectory_retention.mining.points import point_checkpoints
from trajectory_retention.store import ArtifactStore

spec = importlib.util.spec_from_file_location('natural', ROOT / 'configs/v2/natural.py')
natural = importlib.util.module_from_spec(spec)
spec.loader.exec_module(natural)
OUT = SCRATCH / 'm8' / time.strftime('%Y%m%dT%H%M%S')
FIXTURE = 'm8-scripted-counts/v1'
COUNTS = []
SAME_SET = False

class VerdictFixture(CountFeedbackFixture):
    def __init__(self, inner):
        super().__init__(inner)
        self.i = 0
        self.verdicts = {}
    def native_submission_verdict(self, evaluation):
        # Real grade must exist and be valid before injecting a test-only verdict.
        if evaluation is None or not evaluation.valid:
            return {'status': 'unavailable'}
        key = evaluation.evaluation_id
        if key in self.verdicts:
            return self.verdicts[key]
        # Result rendering may re-read submissions and grade the final state. A fixture verdict must be idempotent.
        if self.i >= len(COUNTS):
            return super().native_submission_verdict(evaluation)
        n = COUNTS[self.i]
        self.i += 1
        value = {'status': 'pass' if n == 0 else 'fail', 'failing_count': n,
                'criterion_id': None if n == 0 else ('fixed' if SAME_SET else f'fixture-set-{self.i}'),
                'feedback': None if n == 0 else f'Hidden tests still failing: {n}.',
                'share_passed': evaluation.score}  # real grade; count is explicitly a fixture
        self.verdicts[key] = value
        return value

FIXTURES[FIXTURE] = VerdictFixture

def run(name, counts, *, enabled=True, same=False, feedback=None):
    global COUNTS, SAME_SET
    COUNTS, SAME_SET = counts, same
    d = OUT / name
    write(d / 'script.json', [step('Submit.', [SUBMIT]) for _ in counts])
    # Exercise settings-file loading, not just direct RunConfig construction.
    write(d / 'settings.json', {'model':'scripted/fixture', 'settings':{},
                               **({'submission_feedback': feedback} if feedback else {}),
                               **({'submission_stop': SETTING} if enabled else {})})
    p = {'version':'natural-plan.v2', 'plan_id':f'm8-{name}', 'registry':str(REGISTRY_V1), 'tasks':[TASK],
         'repeats':1, 'model':'scripted/fixture', 'settings':str(d / 'settings.json'), 'store':str(d / 'run'),
         'concurrency':1, 'detector':{'rule':'stall-clock/v8.1'}, 'budget':{'cost_limit_usd':3.,'step_limit':len(counts)},
         'purpose':'stall_supply', 'tool_timeout_seconds':15,
         'fixture':{'submission_protocol':FIXTURE, 'scripts':{'*':str(d / 'script.json')}}}
    write(d / 'plan.json',p)
    plan = natural.load_plan(d / 'plan.json'); job = natural.jobs(plan)[0]
    status = natural.run_job(plan,job)
    store = ArtifactStore(d / 'run/store'); evs = events(store,job['run_id'])
    vs = [e.payload for e in lifecycle(evs,'native_submission_verdict')]
    terminal = next(e.payload['reason'] for e in reversed(evs) if e.kind=='terminal')
    return {'status':status,'verdicts':vs,'terminal':terminal,'store':store,'run_id':job['run_id'],'events':evs}

def main():
    cases = {'regression':([7,8,9,0],3,'native_no_improvement',[1,2,3]),
             'improving':([7,6,5,0],4,'native_success',[1,1,1,0]),
             'reset':([7,6,6,6,0],4,'native_no_improvement',[1,1,2,3]),
             'pass_boundary':([7,9,0],3,'native_success',[1,2,0]),
             'old_plan':([7,8,9,0],4,'native_success',None),
             'identical':([7,7,7,0],3,'native_repeated_failure',[1,2,3]),
             'invalid_count':([7,None,0],1,None,None)}
    results, checks = {}, {}
    for name,(counts,n,end,counters) in cases.items():
        r=run(name,counts,enabled=name!='old_plan',same=name=='identical'); results[name]=r
        vs=r['verdicts']
        checks[name+'_submission_count']=len(vs)==n
        if end:
            checks[name+'_terminal']=r['terminal']=='sequence_complete:'+end
        else:
            checks[name+'_technical_failure']='technical' in r['status']['status'] or 'technical' in r['terminal']
        if counters is not None:
            checks[name+'_counter']=[v.get('no_improvement_count') for v in vs]==counters
        if name=='old_plan': checks['old_plan_no_new_rule']=all('submission_stop' not in v for v in vs)
        checks[name+'_grade_binding']=all(any(e.kind=='grade' and e.event_id==v['grade_event_id']
                                                    for e in r['events']) for v in vs)
        if end and name!='old_plan':
            checks[name+'_no_extra_call']=sum(e.kind=='model_response' for e in r['events'])==n
        print(name,r['terminal'],flush=True)
    # Point 2 already has a stretch of two. Branch must reset, not stop on its first submission.
    global COUNTS, SAME_SET
    COUNTS, SAME_SET = [7,8,9,0], False
    r=results['regression']; root=OUT/'regression/run/store'
    sys.path.insert(0,str(ROOT/'configs/v2'))
    import arms_v2
    from trajectory_retention.experiments.branch_v2 import run_branch, BranchStore, ReadOnlyArtifactStore
    points=point_checkpoints(r['store'],r['run_id'])
    stall={'stall_id':'m8-p2','store':str(root),'run_id':r['run_id'],'checkpoint_id':points[2],
           'dir':str(OUT/'stall'),'has_boundary':False,'step_index':2,'has_recipe':False}
    branch_spec=arms_v2.arm_spec(stall,'1_continue',0,purpose='tuning',prefix='m8',caps=None)
    model=ScriptedModel([step('Submit.',[SUBMIT]) for _ in COUNTS],model_name='scripted/fixture')
    b=run_branch(branch_spec,out_dir=OUT/'branch',model_factory=lambda c:model)
    bs=BranchStore(Path(b['store']),ReadOnlyArtifactStore(root)); be=events(bs,b['run_id'])
    bv=[e.payload for e in lifecycle(be,'native_submission_verdict')]
    checks['branch_end_reason']=b['end_reason']=='no_improvement'
    checks['branch_fresh_counter']=[v.get('no_improvement_count') for v in bv]==[1,2,3]
    checks['branch_inherits_rule']=all(v.get('submission_stop')==SETTING for v in bv) and len(bv)==3
    # Every next request sees the count-only feedback; internal rule fields never reach the worker.
    for name,r in results.items():
        reqs=requests(r['store'],r['events'])
        checks[name+'_no_rule_leak']=all('submission-no-improvement' not in x and 'no_improvement_count' not in x for x in reqs)
        for fb in [e for e in r['events'] if e.kind=='native_submission_feedback']:
            text=r['store'].get_bytes(fb.delivered_ref).decode()
            checks[name+'_feedback_'+str(fb.sequence)]=text.startswith('Hidden tests still failing: ')
    rows={name:{'terminal':r['terminal'],'counts':[v['failing_count'] for v in r['verdicts']],
                'counters':[v.get('no_improvement_count') for v in r['verdicts']], 'status':r['status']['status']}
          for name,r in results.items()}
    report={'rule':SETTING,'fixture':'scripted verdicts on real Docker run/grade path; not task performance',
            'provider_calls':0,'store_root':str(OUT),'runs':rows,'branch_end_reason':b['end_reason'],
            'checks':checks,'all_pass':all(checks.values())}
    write(ARTIFACTS/'m8-submission-stop.json',report)
    print(json.dumps({'all_pass':report['all_pass'],'failed':[k for k,v in checks.items() if not v],
                      'checks':len(checks)},indent=2))
    if not report['all_pass']: raise SystemExit(1)

if __name__=='__main__': main()
