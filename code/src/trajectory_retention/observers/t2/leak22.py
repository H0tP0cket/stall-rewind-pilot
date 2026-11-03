"""T2.2 workspace-leak check: does the workspace still carry what a removed approach's commands touched?

Removing an approach's text from T does not remove its effect on the files.
For the commands of a removed approach this reports which paths they named
(path-like tokens in the command text) or wrote (paths the recorded mutation
event of their response batch lists as changed), and whether each path is
still modified at the checkpoint compared with the run's first checkpoint.

Rules:

* Read-only: the source store is only read (use ``store_ro.ReadOnlyStore``).
* Prefix only: tool calls and mutations after the checkpoint cut are ignored.
* Comparison uses the checkpoints' recorded workspace fidelity manifests
  (``validation.workspace_manifest_ref``), or, when a manifest is missing,
  the manifest rebuilt from the checkpoint's workspace tar. Content fields
  only (type, sha256, link target, mode); mtimes and interpreter caches never
  count as modifications. Paths outside ``/workspace`` are compared through the
  checkpoints' recorded runtime (docker diff) records.
* Mutation events are per response batch; when one batch ran several
  commands, written paths are attributed to the batch and flagged so.
* ``availability`` says what could be compared. When either manifest cannot
  be read, ``leak`` is ``None``, never ``False``.
* A source path is dropped when a more specific source path lies under it
  (recorded parents of changed files, ``cd /tmp``). A directory that the
  commands only named, never wrote, is reported but does not set ``leak``:
  naming a directory is not evidence that the commands changed what is in it.
* "Still modified" is state, not attribution: the path differs from the first
  checkpoint, whoever changed it. ``later_writes_by_other_batches`` counts
  later batches that also changed it.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...contracts import ArtifactRef, from_jsonable
from ...execution.environment import LOOP_EXCLUDED_DIRS, fidelity_manifest_from_tar, loop_excluded
from .segment import _PATH

LEAK22_VERSION = "t2.2-leak/v1"
WORKSPACE = "/workspace"
CONTENT_FIELDS = ("type", "sha256", "link", "mode")
MAX_LISTED = 10
_NOT_PATHS = ("//", "/dev", "/proc", "/sys")
DELETED = "<deleted>"


def _ref(raw) -> ArtifactRef:
    return raw if isinstance(raw, ArtifactRef) else from_jsonable(ArtifactRef, raw)


def _manifest(store, checkpoint) -> tuple[dict[str, Any] | None, str]:
    ref = (checkpoint.validation or {}).get("workspace_manifest_ref")
    reason = "manifest_not_recorded"
    if ref:
        try:
            return store.get_json(_ref(ref)), "fidelity_manifest"
        except Exception as exc:  # noqa: BLE001 - missing or corrupt blob: fall back, and say so
            reason = f"manifest_unreadable:{type(exc).__name__}"
    try:
        return fidelity_manifest_from_tar(store.get_bytes(checkpoint.state.workspace)), f"workspace_tar({reason})"
    except Exception as exc:  # noqa: BLE001 - the state cannot be compared
        return None, f"unavailable:{reason};tar:{type(exc).__name__}"


def _runtime(store, checkpoint) -> dict[str, Any] | None:
    """Runtime (docker diff) state as {path: entry | DELETED}; paths absent are as in the image."""
    if checkpoint.state.runtime is None:
        return None
    try:
        record = store.get_json(checkpoint.state.runtime)
    except Exception:  # noqa: BLE001
        return None
    state: dict[str, Any] = {k.lstrip("/"): v for k, v in (record.get("entries") or {}).items()}
    state.update({k.lstrip("/"): DELETED for k in record.get("deleted") or ()})
    return state


def _cache(key: str) -> bool:
    return loop_excluded(key) or key.rsplit("/", 1)[-1] in LOOP_EXCLUDED_DIRS


def named_paths(text: str) -> list[str]:
    found = []
    for m in _PATH.finditer(text):
        p = m.group(1).rstrip(".")
        if p and not p.startswith(_NOT_PATHS) and p not in found:
            found.append(p)
    return found


def _content(entry):
    return entry if entry is None or entry == DELETED else tuple(entry.get(k) for k in CONTENT_FIELDS)


def _compare(before: dict[str, Any], after: dict[str, Any], key: str, *, absent: str) -> tuple[str, list[str]]:
    """Status of one key: a file, or a directory compared over its non-cache entries.

    ``absent`` names what a missing key means: ``not_found`` in a workspace manifest, ``as_image`` in a
    runtime diff."""
    prefix = key + "/"
    children = sorted({k for m in (before, after) for k in m if k.startswith(prefix) and not _cache(k)})
    if children:
        changed = [k for k in children if _content(before.get(k)) != _content(after.get(k))]
        return ("modified" if changed else "unchanged"), changed[:MAX_LISTED]
    b, a = _content(before.get(key)), _content(after.get(key))
    if b is None and a is None:
        return ("not_found" if absent == "not_found" else "not_in_runtime_diff"), []
    if a == DELETED or (a is None and absent == "not_found"):
        return ("unchanged", []) if b == a else ("deleted", [key])
    if b is None and absent == "not_found":
        return "added", [key]
    return ("modified" if b != a else "unchanged"), ([key] if b != a else [])


def _entry_type(before: dict[str, Any], after: dict[str, Any], key: str) -> str:
    entry = after.get(key) if isinstance(after.get(key), dict) else before.get(key)
    if (isinstance(entry, dict) and entry.get("type") == "d") or any(k.startswith(key + "/") for m in (before, after) for k in m):
        return "directory"
    return "file" if isinstance(entry, dict) else "absent"


def _normalize(path: str) -> str:
    if path == WORKSPACE or path.startswith(WORKSPACE + "/"):
        path = path[len(WORKSPACE) + 1:]
    elif path.startswith("/"):
        return "/" + path.strip("/")
    path = path[2:] if path.startswith("./") else path
    return path.rstrip("/")


def _resolve(rel: str, keys: set[str]) -> tuple[str, list[str]]:
    if rel in keys:
        return "exact", [rel]
    hits = sorted(k for k in keys if k.endswith("/" + rel))
    return ("suffix" if len(hits) == 1 else "ambiguous" if hits else "none"), hits


def workspace_leak(source_store, run_id: str, checkpoint_id: str, action_texts: Sequence[str], *,
                   action_seqs: Sequence[int] | None = None) -> dict[str, Any]:
    """Paths named or written by the given commands, and whether each still differs from the first checkpoint.

    ``action_texts`` are worker_action item texts (a prefix of each command). ``action_seqs`` optionally
    restricts matching to those model-response event sequences (the items' ``seq``)."""
    out: dict[str, Any] = {"version": LEAK22_VERSION, "run_id": run_id, "checkpoint_id": checkpoint_id}
    checkpoints = sorted(source_store.list_checkpoints(run_id), key=lambda c: c.ordinal)
    target = next((c for c in checkpoints if c.checkpoint_id == checkpoint_id), None)
    if target is None or not checkpoints:
        return {**out, "availability": "unavailable", "availability_detail": {"checkpoint": "not_found"}, "leak": None}
    first = checkpoints[0]
    cut = target.event_sequence
    before, how_first = _manifest(source_store, first)
    after, how_target = _manifest(source_store, target)
    rt_first, rt_target = _runtime(source_store, first), _runtime(source_store, target)
    events = list(source_store.iter_events(run_id, through_sequence=cut))
    response_seq = {e.call_id: e.sequence for e in events if e.kind == "model_response" and e.call_id}
    n_actions = {e.sequence: (e.payload or {}).get("n_actions") for e in events if e.kind == "model_response"}
    mutations = {(e.payload or {}).get("response_event_sequence"): e for e in events if e.kind == "mutation"}
    calls = [e for e in events if e.kind == "tool_call"]
    wanted = set(action_seqs) if action_seqs is not None else None

    def changed_paths(mutation) -> list[str]:
        p = mutation.payload or {}
        ws = [k for k in p.get("workspace_fidelity_paths_changed") or () if not _cache(k)]
        return ws + ["/" + k.lstrip("/") for k in p.get("runtime_paths_changed") or ()]

    actions, own_batches = [], set()
    sources: dict[str, set[str]] = {}
    for text in action_texts:
        stripped = text.strip()
        matched = [c for c in calls if stripped and stripped in str((c.payload or {}).get("command", ""))
                   and (wanted is None or response_seq.get((c.payload or {}).get("model_call_id")) in wanted)]
        batches = sorted({response_seq[c.payload["model_call_id"]] for c in matched
                          if (c.payload or {}).get("model_call_id") in response_seq})
        own_batches.update(batches)
        for p in named_paths(text):
            sources.setdefault(_normalize(p), set()).add("named")
        for b in batches:
            if b in mutations:
                for p in changed_paths(mutations[b]):
                    sources.setdefault(_normalize(p), set()).add("written")
        actions.append({"text": stripped[:160], "matched_tool_calls": len(matched), "batches": batches,
                        "batch_attribution": any((n_actions.get(b) or 1) > 1 for b in batches),
                        "mutation_recorded": all(b in mutations for b in batches)})
    detail = {"first_checkpoint": how_first, "target_checkpoint": how_target,
              "runtime_records": "present" if rt_first is not None and rt_target is not None else "missing",
              "mutation_events": len(mutations)}
    out.update(first_checkpoint_id=first.checkpoint_id, cut=cut, actions=actions, availability_detail=detail)
    if before is None or after is None:
        return {**out, "availability": "unavailable", "paths": [], "still_modified": [], "leak": None}
    sources.pop("", None)
    dropped = sorted(p for p in sources if any(q.startswith(p.rstrip("/") + "/") for q in sources if q != p))
    for p in dropped:
        del sources[p]
    keys = set(before) | set(after)
    last_own = max(own_batches, default=None)
    merged: dict[str, dict[str, Any]] = {}
    for path, srcs in sorted(sources.items()):
        if path.startswith("/"):
            key, row = path, {"location": "outside_workspace", "match": "runtime", "resolved": [path]}
            if rt_first is None or rt_target is None:
                row.update(status="unknown", changed_entries=[], entry_type="unknown")
            else:
                status, changed = _compare(rt_first, rt_target, path.lstrip("/"), absent="as_image")
                row.update(status=status, changed_entries=["/" + c for c in changed],
                           entry_type=_entry_type(rt_first, rt_target, path.lstrip("/")))
        else:
            match, resolved = _resolve(path, keys)
            key = resolved[0] if match in ("exact", "suffix") else path
            row = {"location": "workspace", "match": match, "resolved": resolved[:MAX_LISTED]}
            if match in ("exact", "suffix"):
                row["status"], row["changed_entries"] = _compare(before, after, key, absent="not_found")
                row["entry_type"] = _entry_type(before, after, key)
            else:
                row["status"], row["changed_entries"] = ("ambiguous" if match == "ambiguous" else "not_found"), []
                row["entry_type"] = "unknown"
        if key in merged:
            merged[key]["named_as"].append(path)
            merged[key]["sources"] = sorted(set(merged[key]["sources"]) | srcs)
            continue
        row.update(path=key, named_as=[path], sources=sorted(srcs), later_writes_by_other_batches=sum(
            1 for b, m in mutations.items() if last_own is not None and b is not None and b > last_own
            and b not in own_batches and any(k == key or k.startswith(key + "/") for k in changed_paths(m))))
        merged[key] = row
    rows = list(merged.values())
    for r in rows:
        r["counts_for_leak"] = (r["status"] in {"modified", "added", "deleted"}
                                and (r["entry_type"] != "directory" or "written" in r["sources"]))
    still = [r["path"] for r in rows if r["counts_for_leak"]]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {**out, "availability": "full" if detail["runtime_records"] == "present" else "workspace_only",
            "dropped_parent_paths": dropped, "paths": rows, "still_modified": still, "status_counts": counts,
            "leak": bool(still)}
