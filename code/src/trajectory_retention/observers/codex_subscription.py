"""Fresh, tool-free Codex subscription sessions for development-pilot annotations.

This is not the worker transport. No API key or OpenRouter fallback is used. The
caller supplies the complete permitted input; sessions never resume or inherit
the coordinator's context. Dollar cost below means external API charges only:
Codex subscription usage is recorded separately and is not asserted to be free.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

MODEL = "gpt-6.1-terra"
PROVIDER = "Codex subscription"
VERSION = "codex-subscription-terra.v5"
ATTACHMENT_LIMIT = 900000
ATTACHMENT_SYSTEM_EXCEPTION = (
    "Input-delivery exception: for this annotation, any instruction above prohibiting tool use has exactly one "
    "exception. You may and must call permitted_record.read_record to read every consecutive chunk of the "
    "immutable supplied input until eof=true. This tool only delivers the recorded input; it does not execute "
    "the recorded agent's actions. No other tools or evidence are permitted. After reading the entire input, "
    "follow the original labeling/judging instructions and return only the requested JSON object.")


@contextmanager
def helper_slot():
    """At most three helper sessions, including concurrent judge/label processes."""
    root = Path.home() / ".cache/trajectory-retention/codex-helper-slots"
    root.mkdir(parents=True, exist_ok=True)
    for number in range(3):
        handle = (root / str(number)).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        break
    else:
        # Queue in the OS rather than repeatedly losing a polling race to a
        # long-lived producer that immediately starts its next annotation.
        handle = (root / str(os.getpid() % 3)).open("a")
        fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        handle.close()


class CodexSubscription:
    scripted = False

    def __init__(self, root: Path, *, timeout: int = 900):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.env = dict(os.environ)
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL"):
            self.env.pop(key, None)
        # Terminal multiplexers (cmux) put a codex shim first on PATH that injects their own hooks into every
        # session; those hook events are extra context, so resolve the real CLI without shim directories.
        for key in [k for k in self.env if k.startswith("CMUX_CODEX")]:
            self.env.pop(key)
        self.env["PATH"] = os.pathsep.join(d for d in self.env.get("PATH", "").split(os.pathsep)
                                           if "cmux-cli-shims" not in d)
        self.binary = shutil.which("codex", path=self.env["PATH"])
        if self.binary is None:
            raise RuntimeError("codex CLI not found on PATH (outside cmux shims)")
        auth = subprocess.run([self.binary, "login", "status"], capture_output=True, text=True,
                              env=self.env, timeout=30)
        if auth.returncode or "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
            raise RuntimeError("Codex ChatGPT login is required; API-key fallback is disabled")
        self.cli_version = subprocess.check_output([self.binary, "--version"], text=True, env=self.env).strip()

    def post(self, body: dict, ctx: dict) -> dict:
        if body["model"] != MODEL:
            raise ValueError("this pilot transport is pinned to GPT-6.1 Terra")
        messages = body["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("subscription annotations require exactly a system prompt and permitted user input")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        folder = Path(tempfile.mkdtemp(prefix=digest[:16] + "-", dir=self.root))
        # Keep project config/discovery out of the helper's initial context too.
        neutral = Path.home() / ".cache/trajectory-retention/codex-helper-workspaces"
        neutral.mkdir(parents=True, exist_ok=True)
        cwd = Path(tempfile.mkdtemp(prefix="annotation-", dir=neutral))
        system = folder / "system.txt"
        system.write_text(messages[0]["content"])
        (folder / "user.txt").write_text(messages[1]["content"])
        (folder / "request.json").write_text(json.dumps({"body": body, "ctx": ctx}, indent=1))
        output = folder / "answer.txt"
        config = {
            "forced_login_method": "chatgpt", "model_provider": "openai",
            "model_reasoning_effort": body.get("reasoning", {}).get("effort", "max"),
            "model_instructions_file": str(system), "project_doc_max_bytes": 0,
            "features.shell_tool": False, "features.apps": False, "features.plugins": False,
            "features.memories": False, "agents.enabled": False,
            "skills.include_instructions": False, "skills.bundled.enabled": False,
            "web_search": "disabled", "model_context_window": 1050000,
            "model_auto_compact_token_limit": 1000000,
        }
        attachment = len(messages[1]["content"]) > ATTACHMENT_LIMIT
        submitted = messages[1]["content"]
        if attachment:
            # The label system says "never call tools". A lower-priority user instruction cannot authorize
            # attachment delivery; make the previously approved, sole read-only exception explicit here.
            (folder / "original-system.txt").write_text(messages[0]["content"])
            system.write_text(messages[0]["content"] + "\n\n" + ATTACHMENT_SYSTEM_EXCEPTION)
            config.update({
                "mcp_servers.permitted_record.command": sys.executable,
                "mcp_servers.permitted_record.args": [str(Path(__file__).with_name("record_attachment.py")),
                                                       str(folder / "user.txt")],
                "mcp_servers.permitted_record.enabled_tools": ["read_record"],
                "mcp_servers.permitted_record.required": True,
                "mcp_servers.permitted_record.tools.read_record.output_token_limit": 1050000,
                "tool_output_token_limit": 1050000,
            })
            submitted = ("The complete permitted input is supplied as a read-only attachment because it exceeds "
                         "the message character limit. Repeatedly call permitted_record.read_record to read "
                         "EVERY consecutive chunk until eof=true. Read all chunk text in order as one record, "
                         "then follow the original instructions at its end to give your answer. "
                         "Use no other tools or evidence. Do not summarize or skip any portion of the input.")
        (folder / "submitted-user.txt").write_text(submitted)
        cmd = [self.binary, "exec", "--ignore-user-config", "--strict-config", "--skip-git-repo-check",
               "--ephemeral", "--json", "-C", str(cwd), "-s", "read-only", "-m", MODEL,
               "-o", str(output)]
        for key, value in config.items():
            cmd.extend(["-c", key + "=" + json.dumps(value)])
        cmd.append("-")
        meta = {"version": VERSION, "model_requested": MODEL, "cli_version": self.cli_version, "cli_path": self.binary,
                "config": config, "external_api_cost_usd": 0.0,
                "billing": "Codex subscription allowance; dollar equivalent unknown",
                "max_output_tokens": "Codex model default; request max_tokens is not a CLI override",
                "no_inherited_chat": True, "record_delivery": "read-only attachment" if attachment else "inline",
                "system_delivery_exception": ATTACHMENT_SYSTEM_EXCEPTION if attachment else None,
                "original_system_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
                "effective_system_sha256": hashlib.sha256(system.read_bytes()).hexdigest(),
                "attachment_server_sha256": (hashlib.sha256(Path(__file__).with_name("record_attachment.py")
                                                            .read_bytes()).hexdigest() if attachment else None),
                "started": time.time()}
        (folder / "invocation.json").write_text(json.dumps(meta, indent=1))
        try:
            with helper_slot():
                result = subprocess.run(cmd, input=submitted, text=True, capture_output=True,
                                        env=self.env, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            for name, value in (("events.jsonl", exc.stdout), ("stderr.txt", exc.stderr)):
                (folder / name).write_bytes(value if isinstance(value, bytes) else (value or "").encode())
            raise RuntimeError(f"Codex annotation timeout; partial receipt: {folder}") from exc
        (folder / "events.jsonl").write_text(result.stdout)
        (folder / "stderr.txt").write_text(result.stderr)
        meta.update(elapsed=time.time() - meta["started"], returncode=result.returncode)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        # A tool call or compaction would change the permitted evidence. Fail closed.
        def allowed_item(item):
            return item.get("type") in ("agent_message", "reasoning") or (attachment and
                item.get("type") == "mcp_tool_call" and item.get("server") == "permitted_record" and
                item.get("tool") == "read_record" and not item.get("arguments"))
        violations = [e for e in events if (e.get("type", "").startswith("item.") and
                      not allowed_item(e.get("item", {}))) or
                      "compact" in e.get("type", "").lower()]
        reads = [e["item"] for e in events if e.get("type") == "item.completed" and
                 e.get("item", {}).get("type") == "mcp_tool_call"]
        if attachment:
            offset, chunks = 0, []
            for read in reads:
                try:
                    content = read["result"]["content"]
                    assert len(content) == 1 and content[0]["type"] == "text"
                    chunk = json.loads(content[0]["text"])
                    assert chunk["offset"] == offset and chunk["end"] > offset
                    assert chunk["text"] == messages[1]["content"][offset:chunk["end"]]
                    assert chunk["total"] == len(messages[1]["content"])
                    assert chunk["record_sha256"] == hashlib.sha256(messages[1]["content"].encode()).hexdigest()
                    assert chunk["eof"] == (chunk["end"] == chunk["total"])
                    offset = chunk["end"]
                    chunks.append(chunk["text"])
                except (AssertionError, KeyError, TypeError, ValueError):
                    violations.append({"error": "record chunk missing, truncated, duplicated or changed"})
                    break
            if "".join(chunks) != messages[1]["content"]:
                violations.append({"error": "complete record was not delivered in exact consecutive chunks"})
        complete = [e for e in events if e.get("type") == "turn.completed"]
        meta.update(context_violations=violations, completed_turns=len(complete))
        (folder / "invocation.json").write_text(json.dumps(meta, indent=1))
        if result.returncode or violations or len(complete) != 1 or not output.exists():
            raise RuntimeError(f"Codex annotation failed or used extra context; receipt: {folder}")
        usage = complete[0].get("usage") or {}
        return {"provider": PROVIDER, "model": MODEL,
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                             "content": output.read_text()}}],
                "usage": {"prompt_tokens": usage.get("input_tokens"),
                          "completion_tokens": usage.get("output_tokens"), "cost": 0.0,
                          "cost_basis": "external API charges only; subscription cost not estimated",
                          "codex_subscription_usage": usage},
                "codex": {"version": VERSION, "receipt": str(folder), "cli_version": self.cli_version,
                          "reasoning_effort": config["model_reasoning_effort"]}}

    def backoff(self, attempt: int) -> None:
        time.sleep(2 * (attempt + 1))
