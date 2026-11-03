(async()=>{
const checks=[];const check=(name,ok)=>{checks.push({name,passed:!!ok});if(!ok)throw Error(name)};
const until=async(fn)=>{for(let i=0;i<300;i++){if(fn())return;await new Promise(r=>setTimeout(r,20))}throw Error('Timed out waiting for UI')};
try{
startSession('browser-e2e');await until(()=>prefix&&!loading);
check('real prefix loaded and hash verified',prefix.steps.length===1&&k===1);
check('no future steps loaded or rendered',prefix.steps.every(s=>s.k<=k)&&!document.getElementById('step-2'));
check('no preanswer labels',Object.keys(currentDraft.tags).length===0&&Object.keys(currentDraft.claim_status).length===0&&Object.keys(state.labels).length===0);
check('only visible unit fields',allUnits().every(u=>Object.keys(u).every(x=>['id','kind','step','text'].includes(x))));
const typed=currentUnits().filter(u=>['reasoning','message'].includes(u.kind)),command=currentUnits().find(u=>u.kind==='command'),ev=evidenceUnits();check('real fixtures have prose command and output',typed.length>1&&command&&ev.length);
const id=typed[0].id;tagChange(id,'claim');tagChange(id,'observation');check('multiple types supported',currentDraft.tags[id].tags.join(',')==='claim,observation');tagChange(id,'none');check('explicit none distinct from blank',currentDraft.tags[id].state==='answered'&&currentDraft.tags[id].tags.length===0&&!currentDraft.tags[typed[1].id]);tagChange(id,'unclear');check('unclear distinct from none',currentDraft.tags[id].state==='unclear');tagChange(id,'claim');
check('current claim is eligible dependency',targets().some(u=>u.id===id));check('current outputs excluded from dependencies',!targets().some(u=>u.id===ev[0].id));
for(const u of typed)if(!currentDraft.tags[u.id])currentDraft.tags[u.id]={state:'answered',tags:[]};for(const c of currentUnits().filter(u=>u.kind==='command'))currentDraft.rests_on[c.id]={state:'answered',units:[prefix.task[0].id]};
currentDraft.claim_status[id]={state:'answered',verdict:'confirms',evidence:ev[0].id};renderEditor();check('current output confirms claim with evidence',validateAnnotations(currentDraft,prefix,[]).length===0);
currentDraft.claim_status[id].evidence='S999.o1';check('future evidence rejected',validateAnnotations(currentDraft,prefix,[]).length>0);currentDraft.claim_status[id]={state:'answered',verdict:'contradicts',evidence:ev[0].id};check('contradiction with current evidence accepted',validateAnnotations(currentDraft,prefix,[]).length===0);
writeDraft();check('draft persists',readJSON(storageBase+'/session/browser-e2e',{}).drafts[labelKey()].claim_status[id].verdict==='contradicts');
const r=run.id;await finish();check('frozen then next prefix revealed',saved(r,1)&&k===2&&prefix.steps.length===2);check('own earlier claims drive eligibility',claims().some(u=>u.id===id)&&targets().some(u=>u.id===id));check('frozen history hashes recorded',saved(r,1).prior_answer_sha256.length===0&&saved(r,1).answer_sha256.length===64);
await load(1);check('earlier answer read only',readonly&&$('finish').disabled&&[...$('editor').querySelectorAll('.chips button')].every(b=>b.disabled));check('future exposure retained',seen[r]===2);
let output;download=(name,text)=>{output=JSON.parse(text)};exportData();check('export preserves annotations provenance hashes',output.labels[r+':1'].prefix_sha256===run.steps[0].sha256&&output.annotator==='browser-e2e'&&output.exposure[r]===2);
state.labels={};state.drafts={};await importData(output);check('import roundtrip reconstructs sequential history',saved(r,1).annotations.tags[id].tags.includes('claim'));
const bad=structuredClone(output);bad.labels[r+':1'].annotations.tags[id].tags=[];let rejected=false;try{await importData(bad)}catch{rejected=true}check('tampered answer hash rejected',rejected);
const kept=state.labels;state.labels={};const missing=structuredClone(output);delete missing.labels[r+':1'].annotations.tags[id];let changed=missing.labels[r+':1'];delete changed.answer_sha256;changed.answer_sha256=await hash(JSON.stringify(changed));rejected=false;try{await importData(missing)}catch{rejected=true}check('missing type cannot masquerade as explicit none',rejected&&Object.keys(state.labels).length===0);
const future=structuredClone(output);changed=future.labels[r+':1'];delete future.labels[r+':1'];changed.step=2;changed.prefix_sha256=run.steps[1].sha256;delete changed.answer_sha256;changed.answer_sha256=await hash(JSON.stringify(changed));future.labels[r+':2']=changed;rejected=false;try{await importData(future)}catch{rejected=true}check('later step without earlier human history rejected',rejected&&Object.keys(state.labels).length===0);state.labels=kept;
const wrong={...output,deck_id:'other'};rejected=false;try{await importData(wrong)}catch{rejected=true}check('wrong deck rejected',rejected);
startSession('browser-e2e-second');await until(()=>prefix&&!loading);check('new session blank but earlier exposure retained',Object.keys(state.labels).length===0&&seen[r]>=2);check('no computed final statuses shown',!$('editor').textContent.includes('contradicted_later')&&!$('prefix').textContent.includes('Your frozen tags: claim'));
const resp=await fetch('/provenance.json');check('private provenance not served',resp.status===404);
}catch(e){checks.push({error:e.stack})}
const receipt=document.createElement('pre');receipt.id='UNIT_SELFTEST';receipt.textContent=JSON.stringify(checks);document.body.append(receipt)
})();
