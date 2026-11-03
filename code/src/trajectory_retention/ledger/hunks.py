"""Workspace hunks of each step, with the command that produced them (mechanical, no model; units Ledger).

Two sources, chosen per step from what the run recorded (the first that applies):

* ``per_action_archive``: runs recorded with full tool states (the 2026-09 validation runs) keep a workspace tar after
  every action (``tool_state`` events with ``workspace_archive_ref`` and ``tool_index``). The W before action j is the
  W after the previous captured action (or the run's start); a hunk between them is produced by command j exactly.
* ``point_checkpoints``: natural-v2 runs keep a branchable checkpoint at every point (W after step k) and per-action
  change evidence (``mutation.per_action`` paths). The hunk is the difference between points k-1 and k; it is linked
  to the one command whose recorded changed paths contain the hunk's path (``per_action_paths``), to the only command
  of the step (``single_command``), or, when no single command can be named, to the step only (``step_only``, with
  the candidate commands listed).

A step whose captures are missing or failed gets a status saying so and no hunks; nothing is inferred. Only events and
checkpoints up to the cut are read, so the hunks of step k never depend on later steps. Paths under caches and
version-control folders are listed as omitted, not diffed. Binary files are grouped per folder as one hunk.
"""

from __future__ import annotations

import difflib
import io
import tarfile
from typing import Any

from ..contracts import ArtifactRef

HUNKS_VERSION = "ledger-hunks/v1"
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".tox",
              ".venv", "venv", ".eggs", ".cache", ".gocache", "go-build"}
SKIP_SUFFIX = (".pyc", ".pyo", ".o", ".a", ".so", ".class")


def _ref(d: Any) -> ArtifactRef | None:
    if d is None or isinstance(d, ArtifactRef):
        return d
    return ArtifactRef(sha256=d["sha256"], size_bytes=d["size_bytes"], media_type=d.get("media_type", ""))


def _skip(path: str) -> bool:
    return any(p in SKIP_PARTS or p.endswith(".egg-info") for p in path.split("/")) or path.endswith(SKIP_SUFFIX)


def _tar_files(data: bytes) -> dict[str, tuple[str, bytes]]:
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            name = name.lstrip("/")
            if name.startswith("workspace/"):
                name = name[len("workspace/"):]
            if m.issym() or m.islnk():
                out[name] = ("link", m.linkname.encode())
            elif m.isfile():
                out[name] = ("file", tf.extractfile(m).read())
    return out


def _is_text(b: bytes) -> bool:
    if b"\x00" in b[:8192]:
        return False
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def diff_hunks(before: dict, after: dict) -> tuple[list[dict[str, Any]], list[str]]:
    """Hunks between two {path: (type, bytes)} maps, in path order; and the omitted (skipped) changed paths."""
    hunks, omitted, groups = [], [], {}
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path), after.get(path)
        if a == b:
            continue
        if _skip(path):
            omitted.append(path)
            continue
        ab, bb = (a[1] if a else b""), (b[1] if b else b"")
        kind = "added" if a is None else "deleted" if b is None else "modified"
        if not (_is_text(ab) and _is_text(bb)):
            d = path.rsplit("/", 1)[0] + "/" if "/" in path else "./"
            g = groups.get(d)
            if g is None:
                g = groups[d] = {"path": d, "paths": [], "kind": "binary", "binary": True, "header": "binary files",
                                 "lines": []}
                hunks.append(g)
            g["paths"].append(path)
            g["lines"].append(f"(binary {kind}: {path}, {len(ab)} -> {len(bb)} bytes)")
            continue
        diff = list(difflib.unified_diff(ab.decode("utf-8").splitlines(), bb.decode("utf-8").splitlines(),
                                         fromfile=f"a/{path}", tofile=f"b/{path}", n=3, lineterm=""))
        cur = None
        for line in diff[2:]:
            if line.startswith("@@"):
                cur = {"path": path, "paths": [path], "kind": kind, "binary": False, "header": line, "lines": []}
                hunks.append(cur)
            elif cur is not None:
                cur["lines"].append(line)
        if not diff:  # same text, different type (e.g. a file replaced by a link)
            hunks.append({"path": path, "paths": [path], "kind": kind, "binary": False, "header": "type changed",
                          "lines": []})
    for h in hunks:
        h["added"] = sum(1 for x in h["lines"] if x.startswith("+"))
        h["removed"] = sum(1 for x in h["lines"] if x.startswith("-"))
    return hunks, omitted


class _Files:
    def __init__(self, store):
        self.store, self.cache = store, {}

    def get(self, ref) -> dict:
        r = _ref(ref)
        if r.sha256 not in self.cache:
            self.cache[r.sha256] = _tar_files(self.store.get_bytes(r))
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[r.sha256]


def _match(path: str, changed: list[str]) -> bool:
    p = path.rstrip("/").lstrip("./")
    for c in changed:
        c = str(c).lstrip("/").removeprefix("workspace/").lstrip("./")
        if c == p or c.startswith(p + "/") or p.endswith("/" + c) or c.endswith("/" + p):
            return True
    return False


def step_hunks(store, run_id: str, ctx) -> dict[int, dict[str, Any]]:
    """{k: {"status", "source", "hunks": [...], "omitted": [...]}} for every step of the Context (see module doc)."""
    from ..mining.points import checkpoint_step_index

    cut = ctx.cut
    by_seq = {s.response_seq: s for s in ctx.steps if s.response_seq is not None}
    archives: dict[int, list[dict]] = {}
    per_action: dict[int, list[dict]] = {}
    current = None
    for e in store.iter_events(run_id, through_sequence=cut):
        if e.kind == "model_response":
            current = by_seq.get(e.sequence)
            continue
        p = e.payload or {}
        if e.kind == "tool_state" and current is not None and p.get("workspace_archive_ref") is not None:
            archives.setdefault(current.k, []).append({"index": p.get("tool_index"), "status": p.get("status"),
                                                       "ref": p.get("workspace_archive_ref")})
        elif e.kind == "tool_state" and current is not None and p.get("status") == "failed":
            archives.setdefault(current.k, []).append({"index": p.get("tool_index"), "status": "failed", "ref": None})
        elif e.kind == "mutation":
            s = by_seq.get(p.get("response_event_sequence"))
            if s is not None:
                per_action[s.k] = list(p.get("per_action") or [])
    points: dict[int, Any] = {}
    for cp in sorted(store.list_checkpoints(run_id), key=lambda c: c.ordinal):
        if cp.event_sequence <= cut and cp.state.workspace is not None and cp.branchable:
            points[checkpoint_step_index(store, cp)] = cp
    files = _Files(store)
    out: dict[int, dict[str, Any]] = {}
    prev_ref = points[0].state.workspace if 0 in points else None
    for step in ctx.steps:
        k = step.k
        cmds = [c["index"] for c in step.commands]
        arcs = archives.get(k, [])
        rec: dict[str, Any]
        if not step.commands:
            rec = {"status": "no_action", "source": None, "hunks": [], "omitted": []}
        elif arcs:
            rec = {"status": "changed", "source": "per_action_archive", "hunks": [], "omitted": []}
            if any(a["ref"] is None or a["status"] != "captured" for a in arcs) or prev_ref is None:
                rec = {"status": "capture_missing", "source": "per_action_archive", "hunks": [], "omitted": []}
            else:
                before = prev_ref
                for a in sorted(arcs, key=lambda a: a["index"]):
                    if _ref(a["ref"]).sha256 != _ref(before).sha256:
                        hs, om = diff_hunks(files.get(before), files.get(a["ref"]))
                        for h in hs:
                            h.update({"command_index": a["index"], "attribution": "per_action_archive"})
                        rec["hunks"] += hs
                        rec["omitted"] += om
                    before = a["ref"]
            if arcs and arcs[-1]["ref"] is not None:
                prev_ref = arcs[-1]["ref"]
        elif k in points and (k - 1) in points:
            before, after = points[k - 1].state.workspace, points[k].state.workspace
            rec = {"status": "changed", "source": "point_checkpoints", "hunks": [], "omitted": []}
            if _ref(before).sha256 != _ref(after).sha256:
                hs, om = diff_hunks(files.get(before), files.get(after))
                actions = per_action.get(k, [])
                for h in hs:
                    who = sorted({a.get("tool_index") for a in actions
                                  if _match(h["path"], list(a.get("workspace_changed") or []))} - {None})
                    if len(who) == 1:
                        h.update({"command_index": who[0], "attribution": "per_action_paths"})
                    elif len(cmds) == 1:
                        h.update({"command_index": cmds[0], "attribution": "single_command"})
                    else:
                        h.update({"command_index": None, "attribution": "step_only", "candidates": who or cmds})
                rec["hunks"], rec["omitted"] = hs, om
            prev_ref = after
        else:
            rec = {"status": "capture_missing", "source": None, "hunks": [], "omitted": []}
            if k in points:
                prev_ref = points[k].state.workspace
        if rec["status"] == "changed" and not rec["hunks"]:
            rec["status"] = "only_omitted" if rec["omitted"] else "unchanged"
        out[k] = rec
    return out
