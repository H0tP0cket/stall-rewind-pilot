"""Render the lane README tables from recorded evidence (no execution)."""
from __future__ import annotations

import json

from .pipeline import EVIDENCE, task_id_for


def tables() -> str:
    selection = json.loads((EVIDENCE / "selection.json").read_text())
    lines = ["| # | Family (repo) | Split | Env status | Python (official → base) | Ref. suite (stable pass / flaky) | Ref. suite wall s | Test command |",
             "|---|---|---|---|---|---|---|---|"]
    rows = ["| Task | Upstream instance | Role | Status | F2P / P2P kept (excl.) | Bug files / hunks | Grade s (max) | Partial control score | Reason |",
            "|---|---|---|---|---|---|---|---|---|"]
    for family in selection["families"]:
        key = family["repo_key"]
        path = EVIDENCE / "families" / key / "family.json"
        record = json.loads(path.read_text()) if path.is_file() else {"status": "NOT_RUN"}
        env = record.get("environment") or {}
        runs = record.get("reference_runs") or []
        lines.append(f"| {family['order'] + 1} | {family['upstream']} | {family['split']} | {record['status']}"
                     f"{' — ' + record['reason'][:120] if record.get('reason') else ''} | "
                     f"{env.get('python_official', '-')} → {env.get('python_base', '-')}"
                     f"{' (+build tools)' if env.get('compile_tools') else ''} | "
                     f"{record.get('reference_stable_pass', '-')} / {len(record.get('reference_flaky') or [])} | "
                     f"{max((r['wall_seconds'] for r in runs), default='-')} | `{record.get('native_test_command', '-')}` |")
        ledger_path = EVIDENCE / "candidates" / f"{key}.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else [
            {"instance_id": i, "role": "candidate", "status": "NOT_RUN"} for i in family["candidates"]]
        for row in ledger:
            task_id = row.get("task_id") or task_id_for(row["instance_id"])
            status = row["status"]
            if status == "VALIDATED" and row.get("runner_check") == "passed":
                status = "ADMITTED"
            record_path = EVIDENCE / "tasks" / task_id / "package-record.json"
            package = json.loads(record_path.read_text()) if record_path.is_file() and row["status"] != "NOT_USED" else {}
            counts = package.get("counts") or {}
            shape = package.get("bug_shape") or {}
            validation = package.get("validation") or {}
            partial = next((p.get("score") for p in validation.get("partial_controls", [])
                            if p.get("hunk") == validation.get("partial_control_hunk")), None)
            rows.append(f"| {task_id if status != 'NOT_USED' else '-'} | {row['instance_id']} | {row['role']} | {status} | "
                        f"{counts.get('fail_to_pass', '-')} / {counts.get('pass_to_pass', '-')} ({counts.get('pass_to_pass_excluded', '-')}) | "
                        f"{len(shape.get('files', [])) or '-'} / {shape.get('hunks', '-')} | "
                        f"{(validation.get('grade_seconds') or {}).get('max', '-')} | "
                        f"{round(partial, 3) if isinstance(partial, float) else '-'} | {(row.get('reason') or '')[:160]} |")
    return "\n".join(lines) + "\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    print(tables())
