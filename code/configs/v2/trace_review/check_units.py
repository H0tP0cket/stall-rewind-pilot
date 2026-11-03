"""Real Chrome checks of the human unit UI against generated source prefixes; no provider calls."""
import argparse, html, json, re, subprocess, tempfile, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,default=Path('artifacts/v2/unit-review'))
p.add_argument('--chrome',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
p.add_argument('--screenshots',action='store_true')
p.add_argument('--receipt',type=Path,default=Path('artifacts/v2/unit-review/browser-check.json'))
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='unit-review-check-') as tmp:
    tmp=Path(tmp);(tmp/'data').symlink_to(a.directory.resolve()/'data',target_is_directory=True)
    script=(Path(__file__).parent/'unit_browser_check.js').read_text()
    (tmp/'index.html').write_text((a.directory/'index.html').read_text().replace('</html>','<script>'+script+'</script></html>'))
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kw):super().__init__(*args,directory=str(tmp),**kw)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        proc=subprocess.run([a.chrome,'--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={tmp / "profile"}',
                             '--dump-dom','--virtual-time-budget=20000',f'http://127.0.0.1:{server.server_port}/'],
                            capture_output=True,text=True,timeout=90)
        match=re.search(r'<pre id="UNIT_SELFTEST">(.*?)</pre>',proc.stdout,re.S)
        checks=json.loads(html.unescape(match[1])) if match else [{'error':'No browser receipt','stderr':proc.stderr[-1000:]}]
        result={'passed':bool(checks) and all(c.get('passed') for c in checks),'checks':checks,'provider_calls':0,
                'labels':'Scripted UI inputs in a temporary browser profile; not human or study evidence'}
        a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        if a.screenshots:
            for mode in ('tags','rests','claims'):
                setup="(async()=>{startSession('Preview-only');for(let i=0;i<300&&(!prefix||loading);i++)await new Promise(r=>setTimeout(r,20));"
                if mode=='claims':
                    setup+="const u=currentUnits().find(u=>['reasoning','message'].includes(u.kind));tagChange(u.id,'claim');currentDraft.claim_status[u.id]={state:'answered',verdict:'confirms',evidence:null};"
                setup+=f"tab='{mode}';renderEditor();}})();"
                (tmp/'preview.html').write_text((a.directory/'index.html').read_text().replace('</html>','<script>'+setup+'</script></html>'))
                subprocess.run([a.chrome,'--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={tmp / ("visual-"+mode)}',
                                '--window-size=1560,1100','--virtual-time-budget=2000',f'--screenshot={a.directory.resolve() / ("preview-"+mode+".png")}',
                                f'http://127.0.0.1:{server.server_port}/preview.html'],capture_output=True,timeout=30,check=True)

    finally:server.shutdown();server.server_close()
raise SystemExit(0 if result['passed'] else 1)
