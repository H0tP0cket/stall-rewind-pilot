"""Build a human-only fixed-unit labeling deck from completed pilot source prefixes. No model calls."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'configs/v2/judge_v2')]
import record as R
from trajectory_retention.ledger import units as U, prompts as P
from trajectory_retention.ledger.steps import load_context

def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def packed(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=ROOT/'artifacts/v2/pilot-20261027-terra')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/v2/unit-review')
    
    p.add_argument('--max-steps',type=int,default=None,help='Optional starter cap; default full selected prefixes')
    a=p.parse_args()
    if a.max_steps is not None and a.max_steps<1: p.error('max-steps must be positive')
    progress=json.loads((a.pilot/'progress.json').read_text())
    candidates=[]
    for folder,field,singular in [('healthy','healthy_selected','healthy'),('stalls','stalls_selected','stall')]:
        for ident in progress[field]:
            path=a.pilot/folder/ident/(singular+'.json')
            src=json.loads(path.read_text())
            candidates.append({**src,'source_file':str(path),'source_file_sha256':sha(path.read_text())})
    candidates.sort(key=lambda x:sha(x['run_id']))
    roots={x['store'] for x in candidates}
    if len(roots)!=1: raise ValueError('Selected sources must currently share a store')
    store=R.open_readonly(Path(next(iter(roots))))
    public=[]; private=[]; excluded=[]; payloads={}
    try:
        for source in candidates:
            rid=source['run_id']; run=R.RunView(store,rid)
            cap=min(a.max_steps or source['step_index'],source['step_index'],run.n_steps)
            if cap<1:
                excluded.append({'run_id':rid,'reason':'No eligible prefix'}); continue
            cp=run.point_checkpoint(cap)
            if cp is None: excluded.append({'run_id':rid,'reason':f'No saved checkpoint at {cap}'}); continue
            ctx=load_context(store,rid,cp.checkpoint_id)
            if ctx.stall_point!=cap or ctx.checks.get('issues'):
                excluded.append({'run_id':rid,'reason':'Context has issues','checks':ctx.checks}); continue
            seg=U.segment_context(ctx)
            coverage=U.segmentation_coverage(ctx,seg)
            if not coverage['pass']: raise ValueError(coverage)
            runkey=sha(source['store']+rid)[:16]
            def visible(u): return {'id':u['id'],'kind':u['kind'],'step':u['step'],'text':R.mask(u['text'])}
            task=[visible(u) for u in seg['task']]
            steps=[]
            source_steps=[]
            for k in range(1,cap+1):
                prefix={'task':task,'steps':[{ 'k':j,'units':[visible(u) for u in seg['steps'][j]]} for j in range(1,k+1)]}
                body=packed(prefix); fingerprint=sha(body)
                filename=f'data/{runkey}-{k}-{fingerprint[:12]}.json'
                payloads[filename]=body
                current=seg['steps'][k]
                steps.append({'k':k,'file':filename,'sha256':fingerprint,
                              'tag_units':sum(u['kind'] in U.TAGGABLE for u in current),
                              'commands':sum(u['kind']=='command' for u in current),
                              'outputs':sum(u['kind'] in U.EVIDENCE for u in current)})
                source_steps.append({'k':k,'display_prefix_sha256':fingerprint,'source_step_sha256':sha(U.step_text(seg,k))})
            public.append({'id':runkey,'name':f'Trace {len(public)+1:02d}','steps':steps,
                           'work':{'steps':cap,'tag_units':sum(x['tag_units'] for x in steps),'commands':sum(x['commands'] for x in steps)}})
            private.append({'id':runkey,'run_id':rid,'store':source['store'],'selection_file':source['source_file'],'selection_file_sha256':source['source_file_sha256'],'worker_model':run.model,
                            'run_policy':run.policy,'checkpoint_id':cp.checkpoint_id,'cut_event_sequence':ctx.cut,
                            'source_task_sha256':sha(U.task_text(seg)),'steps':source_steps,'coverage':coverage})
    finally: shutil.rmtree(Path(store.database).parent)
    protocol=U.units_protocol_record()
    data={'schema':'human-unit-deck/v1','runs':public,'protocol':protocol,
          'definitions':{'tags':P.UNITS_TAGS,'rests_on':P.UNITS_RESTS_ON,'claim_status':P.UNITS_STATUS},
          'selection':f'Current selected pilot checkpoints in deterministic run-hash order; sequential human labels from step 1. '+(f'Starter capped at {a.max_steps} steps.' if a.max_steps else 'Full selected prefixes.'),
          'coverage':{'source_runs':len(candidates),'selected_runs':len(candidates),'loaded_runs':len(public),
                      'steps':sum(r['work']['steps'] for r in public),'tag_units':sum(r['work']['tag_units'] for r in public),
                      'commands':sum(r['work']['commands'] for r in public),'excluded_runs':len(excluded)},
          'limitations':['Exploratory starter coverage, not a representative validation sample or a completed human gate.',
                        'Completed source runs were selected by the pilot. Outcomes, helper labels and hidden evidence are withheld.',
                        'Only this human’s frozen labels define earlier claims/observations; unclear labels make that eligibility incomplete.',
                        'No computed final claim status is shown. Each evidence judgement is only about the current step’s outputs/feedback.']}
    data['deck_id']=sha(packed(data))[:20]
    a.out.mkdir(parents=True,exist_ok=True)
    # Never overwrite different generated evidence or saved user labels.
    def write(path,text):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_text()!=text: raise ValueError(f'Existing different artifact: {path}; choose --out NEW_DIR')
        path.write_text(text)
    for name,text in payloads.items(): write(a.out/name,text)
    template=(Path(__file__).parent/'units.html').read_text()
    write(a.out/'index.html',template.replace('/*__UNIT_DECK__*/',packed(data).replace('<','\\u003c')))
    write(a.out/'manifest.json',json.dumps(data,indent=2))
    write(a.out/'provenance.json',json.dumps({'pilot':str(a.pilot),'selection_ids':{'healthy':progress['healthy_selected'],'stalls':progress['stalls_selected']},
                                           'runs':private,'excluded':excluded},indent=2))
    print(json.dumps({'url':str(a.out/'index.html'),'deck_id':data['deck_id'],**data['coverage']},indent=2))
if __name__=='__main__':main()
