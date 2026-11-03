"""Per-task declarations for terminalbench-v1: public-only partial controls, native-metric regexes, stall rationale.

Everything here is declared from public task material (instruction, worker-visible
files) or published evidence; nothing is derived from reference solutions or
from any grade outcome of this lane.
"""
from __future__ import annotations

# Declared public-only partial controls: commands a worker could run from the natural start using only
# worker-visible material. None of them uses reference/solution/test content.
PARTIALS = {
    "llm-inference-batching-scheduler": {
        "commands": ["cd /app/task_file/scripts && python3 baseline_packer.py"],
        "basis": "the task's own public baseline packer (instruction: 'performs far worse than required thresholds')",
    },
    "path-tracing": {
        "commands": ["cd /app && cat > image.c <<'EOF'\n#include <stdio.h>\nint main(void){FILE*f=fopen(\"reconstructed.ppm\",\"w\");"
                     "fprintf(f,\"P6 %d %d 255\\n\",2400,1800);for(int i=0;i<2400*1800*3;i++)fputc(128,f);fclose(f);return 0;}\nEOF"],
        "basis": "a trivial program writing a uniform grey image of the target size read from the public image.ppm header (compiles, runs, fails similarity)",
    },
    "path-tracing-reverse": {
        "commands": ["cd /app && cat > mystery.c <<'EOF'\n#include <stdio.h>\nint main(void){FILE*f=fopen(\"image.ppm\",\"w\");"
                     "fprintf(f,\"P6 %d %d 255\\n\",800,600);for(int i=0;i<800*600*3;i++)fputc(128,f);fclose(f);return 0;}\nEOF"],
        "basis": "a trivial program writing a uniform grey image (compiles, size limit met; output differs)",
    },
    "tune-mjcf": {
        "commands": ["cp /app/model_ref.xml /app/model.xml"],
        "basis": "the untuned reference model copied as the tuned model (correct but not faster)",
    },
    "make-mips-interpreter": {
        "commands": ["cd /app && printf \"require('fs').writeFileSync('/tmp/frame.bmp', Buffer.alloc(54));\\n"
                     "setTimeout(() => {}, 60000);\\n\" > vm.js"],
        "basis": "a stub vm.js that only writes an empty frame file (no interpreter)",
    },
}

# Native metrics printed by the native tests' own failure messages (annotation only; never scored).
_B = r"([0-9.eE+-]+)"
METRICS = {
    "llm-inference-batching-scheduler": [
        {"name": f"{bucket}_{name}", "check_id": "test_outputs.py::test_performance_thresholds",
         "regex": rf"AssertionError: {bucket} {label} {_B} >"}
        for bucket in ("bucket1", "bucket2")
        for name, label in (("cost", "cost"), ("pad_ratio", "pad_ratio"), ("p95_latency_ms", "p95"),
                            ("sequential_timecost", "sequential_timecost"))
    ],
    "path-tracing": [{"name": "cosine_similarity", "check_id": "test_outputs.py::test_image_similarity",
                      "regex": rf"Image similarity is only {_B}"}],
    "path-tracing-reverse": [{"name": "cosine_similarity", "check_id": "test_outputs.py::test_image_similarity",
                              "regex": rf"Image similarity is only {_B}"}],
}

# Why each task should produce stuck (or over-commit) stalls. Counts are measured from public data by the
# stall-supply scout (benchmark_tasks/admission/stall-supply-v1/scouting/stall-evidence-sources.md and CSVs):
# "stuck-like" = unsolved TB2 trial with >= 40 agent steps and an OpenHands-StuckDetector-style repeat signal
# (yoonholee/terminalbench-trajectories@04e8940f); "strongish" = from agent/model combos with pass rate >= 0.4;
# FaP thrash = human-labelled "keeps repeating the same approach" failures (Failure as a Process, arXiv 2607.09510,
# about 21 runs per task, 7 models x 3 scaffolds); 2026 frontier pass = Harbor TB2 leaderboard 2026 submissions.
_EV = "benchmark_tasks/admission/stall-supply-v1/scouting/stall-evidence-sources.md"
STALL_RATIONALE = {
    "llm-inference-batching-scheduler": {
        "expected_stall": "stuck: offline re-sweeps of the same packing strategy against a self-computable cost model",
        "self_visible_objective": "public cost_model.py and baseline_packer.py; thresholds printed in the instruction",
        "published": {"fap_thrash_of_failures": "15/19 (all 7 models; mini-SWE 6)", "stuck_like_trials": 62,
                      "stuck_like_strongish": 8, "median_steps_failed": 39, "frontier_2026_pass": 0.76},
        "own_history": ("old-era calibration (tb2-batching-scheduler-artifact-v6/v9): 5 genuine stuck stalls in 8 runs "
                        "(4 DeepSeek, 1 Luna), all identical offline strategy re-sweeps (source-history/README.md)"),
        "caveat": "2026 strong-combo pass 0.97: Luna-class workers may solve it; DeepSeek-V3.2 passed 0.8 in 2026 runs",
        "source": _EV},
    "path-tracing": {
        "expected_stall": "stuck: iterating scene parameters against an L2-similarity bar under a 2 KB gzip limit",
        "self_visible_objective": "the target image /app/image.ppm and the compiled /app/orig are in W (native)",
        "published": {"fap_thrash_of_failures": "10/21 (6 models)", "fap_overclaim": 9, "stuck_like_trials": 48,
                      "stuck_like_strongish": 13, "median_steps_failed": 31, "frontier_2026_pass": 0.8,
                      "frontier_2026_timeout_rate": 0.4},
        "caveat": "no network: worker cannot pip-install numpy/PIL for image analysis (python3 stdlib only)",
        "source": _EV},
    "path-tracing-reverse": {
        "expected_stall": "stuck: repeated decompile/reconstruct attempts that miss the 0.995 similarity bar",
        "self_visible_objective": "the /app/mystery binary is runnable, so the worker can compare outputs itself",
        "published": {"fap_thrash_of_failures": "6/21 (5 models)", "fap_overclaim": 13, "stuck_like_trials": 52,
                      "stuck_like_strongish": 13, "median_steps_failed": 38, "frontier_2026_pass": 0.68},
        "caveat": "FaP over-claim (13/21) exceeds thrash: over-commit is also plausible, but the lifecycle has no "
                  "feedback, so over-commit shows only as an early wrong submission",
        "source": _EV},
    "make-mips-interpreter": {
        "expected_stall": "stuck: long debugging of a MIPS emulator (syscalls, instruction coverage) that never boots Doom",
        "self_visible_objective": "node vm.js runs the provided ELF; frames are written to disk",
        "published": {"stuck_like_trials": 88, "stuck_like_strongish": 23, "median_steps_failed": 49,
                      "frontier_2026_pass": 0.48, "strong_pass": 0.3},
        "source": _EV},
    "polyglot-rust-c": {
        "expected_stall": "stuck: alternating fixes that break one of the two compilers",
        "self_visible_objective": "rustc and g++ are both installed; the worker can run both commands",
        "published": {"stuck_like_trials": 68, "stuck_like_strongish": 15, "median_steps_failed": 9,
                      "frontier_2026_pass": 0.6},
        "caveat": "short failed runs (median 9 steps): stalls may be brief",
        "source": _EV},
    "break-filter-js-from-html": {
        "expected_stall": "stuck: cycling XSS payloads against a sanitizer the worker can run locally",
        "self_visible_objective": "filter.py and a copy of test_outputs.py are in /app natively ('OK to include tests')",
        "published": {"stuck_like_trials": 80, "stuck_like_strongish": 13, "median_steps_failed": 16,
                      "frontier_2026_pass": 0.64},
        "source": _EV},
    "schemelike-metacircular-eval": {
        "expected_stall": "stuck: long interpreter debugging against the provided test programs",
        "self_visible_objective": "interp.py and test programs are in /app natively",
        "published": {"stuck_like_trials": 105, "stuck_like_strongish": 23, "median_steps_failed": 50,
                      "frontier_2026_pass": 0.92},
        "caveat": "2026 frontier pass 0.92: likely solved by strong workers",
        "source": _EV},
    "tune-mjcf": {
        "expected_stall": "stuck: repeated parameter tuning against a 60%-time bar measured by eval.py",
        "self_visible_objective": "/app/eval.py times the simulation",
        "published": {"stuck_like_trials": 76, "stuck_like_strongish": 13, "median_steps_failed": 32,
                      "frontier_2026_pass": 0.84},
        "caveat": "timing-based check: grade may be load-sensitive",
        "source": _EV},
    "feal-differential-cryptanalysis": {
        "expected_stall": "stuck: repeated differential attacks that do not recover the key",
        "self_visible_objective": "feal.py is runnable; the attack can be tested locally",
        "published": {"stuck_like_trials": 69, "stuck_like_strongish": 12, "median_steps_failed": 22.5,
                      "frontier_2026_pass": 0.92},
        "caveat": "2026 frontier pass 0.92 and DeepSeek-V3.2 0.8: likely solved",
        "source": _EV},
    "model-extraction-relu-logits": {
        "expected_stall": "stuck: repeated query/fit attempts that do not recover A1 up to permutation and scaling",
        "self_visible_objective": "forward.py is importable, so the worker can query the network and check its fit",
        "published": {"stuck_like_trials": 19, "stuck_like_strongish": 8, "median_steps_failed": 10,
                      "frontier_2026_pass": 0.36, "strong_pass": 0.22},
        "caveat": "short failed runs (median 10 steps); binary single check",
        "source": _EV},
}
