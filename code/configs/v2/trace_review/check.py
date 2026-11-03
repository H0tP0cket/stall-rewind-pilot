"""Free end-to-end browser check against a generated deck, isolated from real labels."""
import argparse, html, json, re, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--html',type=Path,default=Path('artifacts/v2/trace-review/index.html'))
p.add_argument('--chrome',default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
p.add_argument('--receipt',type=Path,default=Path('artifacts/v2/trace-review/browser-check.json'))
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='trace-review-check-') as d:
    d=Path(d)
    test=d/'check.html'
    script=(Path(__file__).parent/'browser_check.js').read_text()
    test.write_text(a.html.read_text().replace('</html>','<script>'+script+'</script></html>'))
    proc=subprocess.run([a.chrome,'--headless','--no-sandbox','--disable-gpu',f'--user-data-dir={d / "profile"}',
                         '--dump-dom','--virtual-time-budget=5000',test.as_uri()],capture_output=True,text=True,timeout=60)
    match=re.search(r'<pre id="SELFTEST">(.*?)</pre>',proc.stdout,re.S)
    checks=json.loads(html.unescape(match[1])) if match else [{'error':proc.stderr[-1000:]}]
    result={'passed':bool(checks) and all(c.get('passed') for c in checks),'checks':checks,'provider_calls':0}
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
