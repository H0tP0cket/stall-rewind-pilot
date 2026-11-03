"""Separate bash output/exit receipts from Docker client transport.

A worker-user supervisor streams raw merged command bytes between nonce-bound
markers, then appends a size/hash-bound exit receipt. It writes no container
files. A missing finish retains partial bytes and remains an uncertain execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

VERSION = 'docker-command-receipt.v1'
TRANSPORT_GRACE_SECONDS = 15.0

SUPERVISOR = r'''
import hashlib, json, os, signal, subprocess, sys, threading, time
spec = json.loads(sys.argv[1])
nonce = spec['receipt_id']
start_marker = ('\n__TR_COMMAND_START__' + nonce + '__\n').encode()
end_marker = ('\n__TR_COMMAND_RECEIPT__' + nonce + '__').encode()
started = time.monotonic()
receipt = {'version': 'docker-command-receipt.v1', 'receipt_id': nonce,
    'command_sha256': hashlib.sha256(spec['command'].encode()).hexdigest(),
    'timeout_seconds': spec['timeout_seconds'], 'command_returncode': None,
    'status': 'supervisor_error', 'error': None}
proc, reader = None, None
state = {'bytes': 0, 'error': None, 'stop': False}
lock = threading.Lock()
digest = hashlib.sha256()
sys.stdout.buffer.write(start_marker)
sys.stdout.buffer.flush()
def stream():
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            with lock:
                if state['stop']:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                digest.update(chunk)
                state['bytes'] += len(chunk)
    except BaseException as error:
        state['error'] = type(error).__name__ + ': ' + str(error)
def kill_group():
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # `timeout N cmd` and similar move children into their own process group; they stay in the
    # command's session (the command is its session leader), so kill every process of that session.
    for _ in range(3):
        victims = []
        try:
            entries = os.listdir('/proc')
        except OSError:
            # No procfs (host-side supervisor tests): the process-group kill above is all there is.
            break
        for entry in entries:
            if not entry.isdigit() or int(entry) == os.getpid():
                continue
            try:
                with open('/proc/' + entry + '/stat') as f:
                    fields = f.read().rsplit(')', 1)[1].split()
                if int(fields[3]) == proc.pid:
                    victims.append(int(entry))
            except (OSError, ValueError, IndexError):
                pass
        if not victims:
            break
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.2)
try:
    proc = subprocess.Popen(['bash', '-lc', spec['command']], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    reader = threading.Thread(target=stream, daemon=True)
    reader.start()
    deadline = started + spec['timeout_seconds']
    try:
        proc.wait(timeout=max(0, deadline - time.monotonic()))
        reader.join(timeout=max(0, deadline - time.monotonic()))
        if reader.is_alive():
            raise subprocess.TimeoutExpired('command output drain', spec['timeout_seconds'])
        receipt['status'] = 'exited'
    except subprocess.TimeoutExpired:
        kill_group()
        proc.wait(timeout=5)
        reader.join(timeout=5)
        if reader.is_alive():
            # A descendant started its own session and still holds the output pipe. The command itself
            # timed out and was killed; stop relaying output so the receipt stays exact, and say so.
            with lock:
                state['stop'] = True
            receipt['cleanup'] = 'output_holder_left_session'
        receipt['status'] = 'timeout'
    receipt['command_returncode'] = proc.returncode
    if state['error']:
        raise RuntimeError(state['error'])
except BaseException as error:
    receipt['status'] = 'supervisor_error'
    receipt['error'] = type(error).__name__ + ': ' + str(error)
    if proc is not None and proc.poll() is None:
        kill_group()
receipt['duration_seconds'] = time.monotonic() - started
receipt['output_bytes'] = state['bytes']
receipt['output_sha256'] = digest.hexdigest()
sys.stdout.buffer.write(end_marker + json.dumps(receipt, allow_nan=False).encode() + b'\n')
sys.stdout.buffer.flush()
'''


class ReceiptError(ValueError):
    def __init__(self, message: str, raw_output: bytes = b'') -> None:
        super().__init__(message)
        self.raw_output = raw_output


def markers(receipt_id: str) -> tuple[bytes, bytes]:
    return (('\n__TR_COMMAND_START__' + receipt_id + '__\n').encode(),
            ('\n__TR_COMMAND_RECEIPT__' + receipt_id + '__').encode())


def decode_receipt(data: bytes, spec: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    start, end = markers(spec['receipt_id'])
    if not data.startswith(start):
        raise ReceiptError('Missing command stream start')
    payload = data[len(start):]
    raw, marker, trailer = payload.rpartition(end)
    if not marker:
        # A truncated marker is transport framing, not proven command bytes.
        trim = next((n for n in range(min(len(payload), len(end)), 0, -1) if payload.endswith(end[:n])), 0)
        raise ReceiptError('Missing command finish receipt', payload[:-trim] if trim else payload)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate command receipt key')
            result[key] = value
        return result

    try:
        value = json.loads(trailer, object_pairs_hook=unique)
        if not isinstance(value, dict) or value.get('version') != VERSION:
            raise ValueError('Missing command receipt version')
        if value.get('receipt_id') != spec['receipt_id'] or value.get('command_sha256') != hashlib.sha256(spec['command'].encode()).hexdigest():
            raise ValueError('Command receipt identity mismatch')
        if type(value.get('timeout_seconds')) not in (int, float) or value['timeout_seconds'] != spec['timeout_seconds']:
            raise ValueError('Command receipt timeout mismatch')
        if (type(value.get('duration_seconds')) not in (int, float)
                or not math.isfinite(value['duration_seconds']) or value['duration_seconds'] < 0):
            raise ValueError('Invalid command duration')
        if value.get('status') not in {'exited', 'timeout', 'supervisor_error'}:
            raise ValueError('Invalid command receipt status')
        if value['status'] != 'supervisor_error' and (type(value.get('command_returncode')) is not int
                or not -255 <= value['command_returncode'] <= 255 or value.get('error') is not None):
            raise ValueError('Invalid command exit receipt')
        if (type(value.get('output_bytes')) is not int or value['output_bytes'] != len(raw)
                or value.get('output_sha256') != hashlib.sha256(raw).hexdigest()):
            raise ValueError('Command output receipt mismatch')
    except (ValueError, TypeError, UnicodeError) as error:
        raise ReceiptError(str(error), raw) from error
    return value, raw
