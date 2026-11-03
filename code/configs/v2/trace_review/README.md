# Local trace review

Build without provider calls or source-store writes:

```sh
.venv/bin/python configs/v2/trace_review/build.py
open artifacts/v2/trace-review/index.html
```

The single HTML file works offline. Labels and unfinished drafts persist in browser local storage; **Export labels** downloads JSON and the existing replay scorer's five-column CSV for the legacy 20-item deck. JSON import accepts only the same deck and matching record hashes; it adds missing saved labels without replacing existing labels. Unfinished exported drafts are backups only and are not imported as completed labels. Keep each human's browser/profile and export separate.

By default the builder discovers the five currently available `~/.cache/trajectory-retention/v2-screen/**/journal.sqlite3` stores and chooses one deterministic, outcome-independent scheduled 10-step point per natural run. It also loads the existing blinded **v2** calibration packet, not the v1 packet that contains private grades. This is an exploratory development deck, not a representative validation sample, and does not claim coverage of every post-exp48 artifact. Runs shorter than ten steps and prefixes that the existing worker-visible renderer cannot reconstruct cleanly are recorded as exclusions in `provenance.json`.

Use `--store /path/to/store` (repeatable) to select other stores, or `--all-points --out /separate/output` to build all scheduled points. Labels record when the human has previously opened a later prefix of the same run. Such labels should not be treated as fresh blinded judgements. The legacy packet's blinded files lack run IDs, so repeated-run exposure there cannot be measured by this UI.

`provenance.json` retains source store, run, model, policy, point, cut and rendered-record hash outside the HTML, for joining labels to judge outputs. Do not show it to a blinded labeller. The UI contains only rendered worker-visible prefixes; it uses the existing `judge_v2/record.py` path, not grades, reference fixes or judge votes. No task content is interpreted as HTML.

The legacy CSV can be passed to the existing scorer:

```sh
.venv/bin/python configs/v2/judge_v2/replay_v2.py score --out <replay-directory> --human <downloaded-legacy-calibration-labels.csv>
```

The new deck's JSON uses `stall`, `not_stall`, `unclear`, `onset_step`, and `evidence`, matching the v2 judge answer fields. A null onset for a stalled example means the human could not locate it; it is not an inferred onset. A new-deck scoring/join command is not included; join JSON item IDs through `provenance.json` before comparing to new GLM/Luna judgements.

Run the free browser check (Chrome installed locally; isolated temporary profile, no real labels touched):

```sh
.venv/bin/python configs/v2/trace_review/check.py
```

It checks rendering, onset validation, saving, browser persistence, navigation, JSON/CSV export, JSON import, rejection of another deck, and the future-prefix exposure warning. Its receipt is `artifacts/v2/trace-review/browser-check.json`.

## Human fixed-unit labeling (separate mode)

The existing offline stall deck and its storage keys are unchanged. The unit mode lives separately:

```sh
.venv/bin/python configs/v2/trace_review/build_units.py
.venv/bin/python configs/v2/trace_review/serve_units.py
```

Open **http://127.0.0.1:8768/**. The server binds only to localhost and serves only the UI and prefix JSON files; it refuses private provenance, directory listings and other files. No labels are sent to the server. Keep the same origin/port/browser for local persistence, and export backups.

Default coverage is the five checkpoints listed in the current Terra pilot's `progress.json` selection: full prefixes of 10, 20, 20, 60 and 100 steps, **210 steps / 15,901 taggable units / 237 commands** in the initial build. Evidence workload depends on the human's claim labels. Display names and order are anonymized; selection identities, model, source checkpoint, exact protocol and source/display hashes remain in private `provenance.json`. This is exploratory coverage, not a completed validation gate or representative sample. No Terra labels, judge votes, outcomes, reference solutions or private evaluations are loaded. The builder reads source contexts through `load_context` and uses `segment_context` unchanged.

Each run starts at step 1. Only its task and steps through the current step are fetched and displayed. Types apply only to current reasoning/message units; multiple tags are allowed. Commands choose dependencies from the protocol's permitted task/history/current claim-or-observation units. Current outputs/feedback provide evidence for claims through the current step. Definitions are copied from the current prompt constants. No final computed claim status is shown.

- Blank, explicit none/no-dependencies/says-nothing, and unclear are distinct.
- All required items need an answer before freezing a step. Unclear is a legitimate human answer, retained separately rather than converted into a model label.
- Earlier claim/observation eligibility comes only from this human's frozen types. Unclear earlier types produce a visible incomplete-eligibility warning and make `complete_for_protocol` false. No machine answer fills that gap.
- Saved steps are immutable; opening an earlier step is read-only. A different alias opens a new independent annotation session, but later-prefix exposure is retained across sessions in that browser. Exposure cannot be detected across unrelated browsers/devices; disclose prior exposure when sharing labels.
- JSON export includes frozen answers, drafts, source/display hashes, prior-answer hashes, annotator alias, timestamps and exposure. Import validates the deck, identity, prefix and answer hashes, sequential history, types, eligible dependency IDs, evidence IDs and uncertainty metadata. It rejects conflicts and incomplete prior history. Drafts in an export are backup-only, not silently promoted to answers or imported as completed labels.
- `protocol_answer` is a convenience representation only when all current and earlier relevant types are resolved; otherwise it is null. Exports are **human calibration evidence**, never automatically copied into the study's frozen treatment labels.

For a smaller starter build, use a new output directory:

```sh
.venv/bin/python configs/v2/trace_review/build_units.py --max-steps 10 --out /tmp/unit-review-starter
.venv/bin/python configs/v2/trace_review/serve_units.py --directory /tmp/unit-review-starter --port 8769
```

The builder refuses to replace different generated artifacts; use a new output directory when the selected checkpoints or template change. It never overwrites source stores or user exports. The unit mode needs localhost because later prefixes are separate lazy-loaded files; opening its HTML directly with `file://` is unsupported. The old stall mode remains fully offline.

Checks and visual review:

```sh
.venv/bin/python configs/v2/trace_review/check_units.py --screenshots
.venv/bin/python configs/v2/trace_review/check.py --receipt artifacts/v2/unit-review/old-stall-regression.json
```

Checks use isolated temporary browser profiles and synthetic UI choices, never actual human answers. `preview-claims.png` illustrates the evidence picker with a synthetic claim choice; it is not a human or model label. Receipts and screenshots are saved under the unit-review output directory.
