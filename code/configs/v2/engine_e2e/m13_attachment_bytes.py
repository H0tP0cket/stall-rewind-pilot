"""Regression: immutable trace delivery preserves CR, CRLF, Unicode and chunk boundaries. No model calls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from common import ROOT, ARTIFACTS, write


def check(text):
    server = ROOT / 'src/trajectory_retention/observers/record_attachment.py'
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'record.txt'; p.write_bytes(text.encode('utf-8'))
        n = (len(text) + 59999) // 60000
        reqs = [{'jsonrpc': '2.0', 'id': i, 'method': 'tools/call',
                 'params': {'name': 'read_record', 'arguments': {}}} for i in range(n)]
        result = subprocess.run([sys.executable, str(server), str(p)],
            input=''.join(json.dumps(r) + '\n' for r in reqs), text=True, capture_output=True, check=True)
    chunks = [json.loads(json.loads(line)['result']['content'][0]['text']) for line in result.stdout.splitlines()]
    offset = 0
    for c in chunks:
        assert c['offset'] == offset and c['end'] > offset
        assert c['text'] == text[offset:c['end']]
        assert c['total'] == len(text) and c['record_sha256'] == hashlib.sha256(text.encode()).hexdigest()
        assert c['eof'] == (c['end'] == len(text))
        offset = c['end']
    assert ''.join(c['text'] for c in chunks) == text
    return {'chars': len(text), 'chunks': len(chunks), 'exact': True}


def main():
    cases = [check('Progress 1\rProgress 2\r\nUnicode 🐍\n' * 30000), check('a' * 59999 + '\r\n' + 'b' * 60000)]
    if len(sys.argv) > 1:
        d = json.loads(Path(sys.argv[1]).read_text())
        cases.append(check(d['body']['messages'][1]['content']))
    report = {'all_pass': True, 'provider_calls': 0, 'cases': cases}
    write(ARTIFACTS / 'm13-attachment-bytes.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
