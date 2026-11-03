"""Minimal read-only MCP attachment: one immutable record, no paths or queries.

Used only to deliver permitted inputs exceeding Codex's turn/start character
limit. The one exposed tool returns the exact full record and nothing else.
"""
from pathlib import Path
import json
import hashlib
import sys

CHUNK_CHARS = 60000


def main():
    # Universal-newline translation changes recorded terminal CR/CRLF bytes and breaks the input seal.
    text = Path(sys.argv[1]).read_bytes().decode("utf-8")
    offset = 0
    digest = hashlib.sha256(text.encode()).hexdigest()
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        if method == "initialize":
            result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "permitted-record", "version": "2"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "read_record", "description":
                       "Read the NEXT consecutive chunk of one immutable record, including its original "
                       "judging/labeling instructions. Keep calling until eof=true. Chunks are never skipped. "
                       "No other files, paths, queries or external information are available.",
                       "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
                       "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}}]}
        elif method == "tools/call" and request.get("params", {}).get("name") == "read_record" and not request["params"].get("arguments"):
            end = min(len(text), offset + CHUNK_CHARS)
            chunk = {"offset": offset, "end": end, "total": len(text), "eof": end == len(text),
                     "record_sha256": digest, "text": text[offset:end]}
            result = {"content": [{"type": "text", "text": json.dumps(chunk, ensure_ascii=False)}], "isError": False}
            offset = end
        elif method == "ping":
            result = {}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": request["id"],
                              "error": {"code": -32601, "message": "Only the bound read_record tool is available"}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()
