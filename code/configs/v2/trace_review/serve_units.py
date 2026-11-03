"""Serve only the unit UI and prefix files on localhost; private provenance is never served."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,default=Path('artifacts/v2/unit-review'))
p.add_argument('--port',type=int,default=8768)
a=p.parse_args()
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kw):super().__init__(*args,directory=str(a.directory.resolve()),**kw)
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path not in ('/','/index.html') and not (path.startswith('/data/') and path.endswith('.json') and '..' not in path and path.count('/')==2):
            self.send_error(404);return
        super().do_GET()
    def do_HEAD(self):
        self.send_error(405)
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
print(f'Unit labeling: http://127.0.0.1:{a.port}/',flush=True)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
