"""Build an offline, blinded human review deck; never calls a model or modifies stores."""
from __future__ import annotations
import argparse, hashlib, json, random, re, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'configs/v2/judge_v2'))
import record as R

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/v2/trace-review')
    p.add_argument('--store', type=Path, action='append', help='Explicit store roots; default discovers v2-screen stores')
    p.add_argument('--all-points', action='store_true', help='Include every 10-step prefix; repeated-run exposure is tracked')
    a = p.parse_args()
    stores = a.store or sorted((Path.home()/'.cache/trajectory-retention/v2-screen').glob('**/journal.sqlite3'))
    stores = [s.parent if s.name == 'journal.sqlite3' else s for s in stores]
    items, inventory, provenance = [], [], {}
    for root in stores:
        entry = {'store': str(root), 'runs': 0, 'available_points': 0, 'included_points': 0, 'excluded': []}
        inventory.append(entry)
        store = R.open_readonly(root)
        try:
            for rid in R.natural_runs(store):
                entry['runs'] += 1
                run = R.RunView(store, rid)
                points = run.points()
                entry['available_points'] += len(points)
                if not points:
                    entry['excluded'].append({'run_id':rid, 'reason':'No complete 10-step point'})
                    continue
                # Outcome-independent deterministic sampling; not a validated representative sample.
                chosen = points if a.all_points else [points[int(digest(str(root)+rid)[:12],16)%len(points)]]
                for k in chosen:
                    ident = digest(str(root)+rid+str(k))[:20]
                    try:
                        rec = run.record(k)
                        if rec.issues:
                            raise ValueError('; '.join(rec.issues))
                        text = rec.render()
                        items.append({'id':ident,'deck':'Existing natural runs','point':k,'text':text,
                                      'run_key':digest(str(root)+rid)[:16], 'sha256':digest(text)})
                        provenance[ident] = {'store':str(root),'run_id':rid,'point':k,'source':rec.source,
                                             'cut_sequence':rec.cut_sequence,'record_version':R.RECORD_VERSION,
                                             'worker_model':run.model,'run_policy':run.policy,
                                             'sha256':digest(text)}
                        entry['included_points'] += 1
                    except (LookupError, ValueError, KeyError) as e:
                        entry['excluded'].append({'run_id':rid,'point':k,'reason':str(e)})
        finally:
            shutil.rmtree(Path(store.database).parent)
    packet = ROOT/'artifacts/calibration/detector-judge-20260926-v1/hand-label/packet-v2'
    for f in sorted(packet.glob('item-*.txt')):
        text = f.read_text()
        # Strip judge's answer instructions, retain exact visible record.
        text = text.split('===== THE RECORD ENDS HERE')[0].rstrip()
        points = re.findall(r'===== STEP (\d+) =====',text)
        if not points:
            raise ValueError(f'No numbered steps in {f}')
        num = f.stem.split('-')[1]
        ident = 'calibration-'+num
        items.append({'id':ident,'deck':'Legacy calibration · 20 items','legacy_item':num,'point':int(points[-1]),
                      'text':text,'run_key':None,'sha256':digest(text)})
        provenance[ident] = {'packet_file':str(f),'legacy_item':num,'point':int(points[-1]),'sha256':digest(text)}
    random.Random(20261026).shuffle(items)
    payload = {'schema':'trace-review-deck/v1','created':datetime.now(timezone.utc).isoformat(),
               'selection':'Every 10-step point' if a.all_points else 'One deterministic, outcome-independent 10-step point per natural run',
               'items':items,'coverage':[ {k:v for k,v in x.items() if k != 'excluded'} | {'excluded_count':len(x['excluded'])} for x in inventory],
               'limitations':['Only discovered v2-screen stores and the existing 20-item v2 calibration packet are loaded. Other post-exp48 stores are not claimed covered.',
                              'These are human calibration examples, not admitted study stalls. Historical protocols and model versions are separate in provenance.',
                              'Sampling is for interface use, not a representative validation set. No judge votes, hidden evaluations or future outcomes are included.',
                              'The legacy packet may repeat runs; run identities are unavailable in its blinded files. Keep its labels separate.']}
    payload['deck_id'] = digest(json.dumps([(i['id'],i['sha256']) for i in items]))[:20]
    a.out.mkdir(parents=True,exist_ok=True)
    data = json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    template = (Path(__file__).parent/'index.html').read_text()
    (a.out/'index.html').write_text(template.replace('/*__DECK__*/',data))
    (a.out/'provenance.json').write_text(json.dumps({'inventory':inventory,'items':provenance},indent=2))
    print(json.dumps({'html':str(a.out/'index.html'),'items':len(items),'stores':inventory},indent=2))
if __name__ == '__main__': main()
