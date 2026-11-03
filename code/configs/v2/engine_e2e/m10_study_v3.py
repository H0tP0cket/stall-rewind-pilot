"""Current study CLI wiring check, with explicitly synthetic branch outcomes.

Requires the scripted judge/label fixture produced in pilot-integration-20261026.
Tests scheduling/specs/aggregation, not intervention efficacy. No provider calls.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import ROOT, SCRATCH, ARTIFACTS, write
sys.path.insert(0, str(ROOT / "configs/v2"))
from study_common import read_json, INJECTION_CONDITIONS
from trajectory_retention.experiments.branch_v2 import check_spec


def main():
    out = SCRATCH / "m10" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    fixture = Path.home() / ".cache/trajectory-retention/pilot-integration-20261026/operations-check/stalls"
    stalls, study = out / "stalls", out / "study"
    shutil.copytree(fixture, stalls)
    study.mkdir(parents=True)
    (study / "SYNTHETIC").write_text("Scripted study-operations check, not evidence.\n")
    d = next(p for p in stalls.iterdir() if (p / "stall.json").exists())
    st = read_json(d / "stall.json")
    world = {"runs": {st["run_id"]: {"n_tests": 20, "p_solve": {"1_continue": 0, "default": 1},
              "p_reentry": {"default": 0}, "base_failing": 10, "stall_step": st["step_index"],
              "approach_commands": ["echo synthetic"], "fresh_commands": ["echo synthetic"]}},
             "price": {"in": 1e-7, "cached_in": 2e-8, "out": 4e-7}}
    write(out / "world.json", world)
    env = os.environ | {"V2_STUB_WORLD": str(out / "world.json")}
    log, checks = [], {}
    def cli(file, *args):
        cmd = [sys.executable, str(ROOT / "configs/v2" / file), *map(str, args)]
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        log.append({"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-1800:], "stderr": p.stderr[-1800:]})
        if p.returncode:
            write(ARTIFACTS / "m10-study-v3.json", {"all_pass": False, "log": log, "root": str(out)})
            raise RuntimeError(p.stdout[-1800:] + p.stderr[-1800:])
    def specs(cmd, *args):
        cli("arms_v2.py", cmd, "--arm-set", "v3", "--study", study, *args)
    def drive(prefix):
        cli("driver_v2.py", "run", "--specset", study / "specsets" / f"{prefix}.json",
            "--out", study / "runs" / prefix, "--runner", "stub", "--parallel", 2, "--allow-draft")
    specs("screen", "--stalls-root", stalls)
    drive("screen")
    cli("screen_v2.py", "--arm-set", "v3", "--study", study, "--stalls-root", stalls)
    screen = study / "screen/screen.json"
    checks["screen_admitted"] = read_json(screen)["counts"]["admitted"] == 1
    cli("caps.py", "screen-caps", "--arm-set", "v3", "--study", study, "--screen-json", screen)
    cal = study / "calibration.json"
    checks["95_percentile"] = read_json(cal)["cap_rule"]["quantile"] == .95
    specs("tuning", "--stalls-root", stalls, "--stall", st["stall_id"], "--round", 1,
          "--repeats", 2, "--calibration", cal)
    drive("tuning-r1")
    cli("caps.py", "tuning", "--arm-set", "v3", "--study", study)
    cli("caps.py", "freeze", "--arm-set", "v3", "--study", study)
    specs("main", "--stalls-root", stalls, "--screen-json", screen, "--calibration", cal,
          "--repeats", 2, "--allow-draft")
    drive("main")
    main_manifest = read_json(study / "specsets/main.json")
    checks["six_arms_two_repeats"] = main_manifest["branches"] == 12
    # Resume must leave exactly the same completed attempts in place.
    before = sorted(str(p) for p in (study / "runs/main").rglob("result.json"))
    drive("main")
    checks["resume_no_duplicate_attempts"] = before == sorted(str(p) for p in (study / "runs/main").rglob("result.json"))
    near = read_json(screen)
    near["stalls"][0].update(near_cutoff=True, extra_resamples=5)
    write(out / "near.json", near)
    specs("screen-extra", "--stalls-root", stalls, "--screen-json", out / "near.json", "--round", 1)
    checks["topup_five_per_arm"] = read_json(study / "specsets/screen-extra-1.json")["branches"] == 10
    healthy = out / "healthy" / "synthetic-healthy"
    healthy.mkdir(parents=True)
    h = {k: st[k] for k in ("store", "run_id", "task_id", "checkpoint_id", "step_index")}
    h.update(healthy_id="synthetic-healthy", version="healthy.v2", synthetic=True)
    write(healthy / "healthy.json", h)
    for name in ("ledger.json", "review_view_ledger_flat.json", "review_view_context.json"):
        shutil.copy2(d / name, healthy / name)
    specs("harm", "--healthy-root", healthy.parent, "--calibration", cal, "--repeats", 2, "--allow-draft")
    drive("harm")
    checks["harm_two_repeats"] = read_json(study / "specsets/harm.json")["branches"] == 2
    # Only scheduling is tested here. Payload execution is covered by the Docker branch checks.
    write(healthy / "injection_donors.json", {"conditions": {c: {"payload": {"condition": c,
        "synthetic": True}} for c in INJECTION_CONDITIONS}, "draft_rules_used": ["synthetic_payloads"]})
    specs("injection", "--healthy-root", healthy.parent, "--calibration", cal, "--repeats", 2, "--allow-draft")
    drive("injection")
    checks["eight_injections_two_repeats"] = read_json(study / "specsets/injection.json")["branches"] == 16
    for path in (study / "specsets").glob("*/*/*.json"):
        spec = read_json(path)
        if not spec["arm"].startswith("injection:"):
            check_spec(spec)
    checks["arm_specs_validate"] = True
    cli("analyze_v2.py", "--arm-set", "v3", "--study", study)
    report = {"all_pass": all(checks.values()), "synthetic": True, "provider_calls": 0,
              "checks": checks, "root": str(out), "log": log}
    write(ARTIFACTS / "m10-study-v3.json", report)
    print(json.dumps({"all_pass": report["all_pass"], "checks": checks, "root": str(out)}, indent=1))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
