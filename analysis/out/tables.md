### Main test: solved / branches, per stall and arm

| Stall (type) | Screen: continue | Screen: rewind | Continue | Ledger review | Context review | Forced rewind | Ledger read-only | Context read-only | Any-time review (arm 7, follow-up) | Ledger review, Opus decides (follow-up) |
|---|---|---|---|---|---|---|---|---|---|---|
| production-planning (stuck) | 0/5 | 3/5 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 | 1/2 | 1/2 |
| atrx-vep-crispr (under-commit) | 4/5 | 2/5 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| heat-pump-warranty (over-commit) | 2/5 | 2/5 | 2/2 | 2/2 | 1/2 | 0/2 | 1/2 | 2/2 | 2/2 | 1/2 |
| **All three** | | | 2/6 | 2/6 | 1/6 | 3/6 | 1/6 | 2/6 | 3/6 | 2/6 |

### Continue vs forced rewind, screen and arms pooled

The arms all used the neutral feedback line ("The task is not complete yet. Continue working."). The screens of production-planning and atrx-vep-crispr used the count line ("Hidden tests still failing: N."), so pooling mixes the two lines for those stalls; the screen of heat-pump-warranty used the neutral line.

| Stall (type) | Stall point (step) | Forced rewind restores (step) | Steps undone | Screen feedback line | Continue: screen | Continue: arms (neutral line) | Continue pooled | Forced rewind: screen | Forced rewind: arms (neutral line) | Forced rewind pooled | Fisher exact p, two-sided |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production-planning (stuck) | 20 | 17 | 3 | count line | 0/5 | 0/2 | 0/7 | 3/5 | 2/2 | 5/7 | 0.021 |
| atrx-vep-crispr (under-commit) | 100 | 91 | 9 | count line | 4/5 | 0/2 | 4/7 | 2/5 | 1/2 | 3/7 | 1.000 |
| heat-pump-warranty (over-commit) | 70 | 68 | 2 | neutral line | 2/5 | 2/2 | 4/7 | 2/5 | 0/2 | 2/7 | 0.592 |

### Follow-up: reviews told "This run looks stalled."

| Stall | Ledger review (pilot) | Ledger review (told stalled) | Context review (pilot) | Context review (told stalled) |
|---|---|---|---|---|
| production-planning | 0/2 | 0/2 | 0/2 | 0/2 |
| atrx-vep-crispr | 0/2 | 0/2 | 0/2 | 0/2 |
| heat-pump-warranty | 2/2 | 1/2 | 1/2 | 1/2 |

### What the review turn chose

| Where | Reviews | Changed nothing | Dropped context only | Rewound a channel | Rewound context or files |
|---|---|---|---|---|---|
| Stalls, neutral prompt | 12 | 11 | 1 | 0 | 0 |
| Stalls, told stalled | 12 | 9 | 2 | 1 | 0 |
| Healthy runs (harm test) | 6 | 4 | 2 | 0 | 0 |

Across the 24 one-shot reviews at stalls (12 with the neutral prompt, 12 told the run looks stalled): 20 of 24 changed nothing, 3 dropped context only, 1 rewound the outputs channel only, and 0 of 24 rewound context, files or runtime.

### Follow-up: any-time review (arm 7), every opening of the review

| Stall | Branches solved | Openings | Changed nothing | Trimmed context only | Rewound outputs only | Rewound context, files or runtime |
|---|---|---|---|---|---|---|
| production-planning | 1/2 | 3 | 0 | 1 | 0 | 2 |
| atrx-vep-crispr | 0/2 | 6 | 6 | 0 | 0 | 0 |
| heat-pump-warranty | 2/2 | 4 | 4 | 0 | 0 | 0 |
| **All three** | 3/6 | 13 | 10 | 1 | 0 | 2 |

2 of 13 openings rewound context, files or runtime (production-planning).

### Who rewinds: every answer to the same review requests

28 review requests (25 at stalls, 3 at healthy checkpoints): the 30 one-shot review turns share 15 requests, and each of the 13 any-time openings is its own request.

**All requests**

| Who answers | Answers | Changed nothing | Trimmed context only | Rewound outputs only | Rewound context, files or runtime | No usable answer | Share that rewound | Answers with a call error |
|---|---|---|---|---|---|---|---|---|
| The worker itself (its actual choice) | 43 | 34 | 6 | 1 | 2 | 0 | 5% (2/43) | 0/43 |
| Worker model, asked again with the same request | 84 | 57 | 11 | 0 | 12 | 4 | 14% (12/84) | 17/84 |
| Worker model, asked as an outside observer | 84 | 65 | 4 | 0 | 13 | 2 | 15% (13/84) | 8/84 |
| Opus 5.5, asked as an outside observer | 28 | 0 | 10 | 0 | 18 | 0 | 64% (18/28) | 8/28 |

**Requests at stalls**

| Who answers | Answers | Changed nothing | Trimmed context only | Rewound outputs only | Rewound context, files or runtime | No usable answer | Share that rewound | Answers with a call error |
|---|---|---|---|---|---|---|---|---|
| The worker itself (its actual choice) | 37 | 30 | 4 | 1 | 2 | 0 | 5% (2/37) | 0/37 |
| Worker model, asked again with the same request | 75 | 54 | 7 | 0 | 10 | 4 | 13% (10/75) | 16/75 |
| Worker model, asked as an outside observer | 75 | 60 | 1 | 0 | 12 | 2 | 16% (12/75) | 7/75 |
| Opus 5.5, asked as an outside observer | 25 | 0 | 7 | 0 | 18 | 0 | 72% (18/25) | 7/25 |

**Requests at healthy checkpoints**

| Who answers | Answers | Changed nothing | Trimmed context only | Rewound outputs only | Rewound context, files or runtime | No usable answer | Share that rewound | Answers with a call error |
|---|---|---|---|---|---|---|---|---|
| The worker itself (its actual choice) | 6 | 4 | 2 | 0 | 0 | 0 | 0% (0/6) | 0/6 |
| Worker model, asked again with the same request | 9 | 3 | 4 | 0 | 2 | 0 | 22% (2/9) | 1/9 |
| Worker model, asked as an outside observer | 9 | 5 | 3 | 0 | 1 | 0 | 11% (1/9) | 1/9 |
| Opus 5.5, asked as an outside observer | 3 | 0 | 3 | 0 | 0 | 0 | 0% (0/3) | 1/3 |

Opus changed something at 28 of 28 requests. 11 of its 28 answers were made from a copy of the request in which the word "reasoning" was replaced by "notes" (column input_variant).
The 168 extra worker-model answers cost $3.93; none of these answers was executed.

### Injection test: solved / branches on healthy checkpoints

| Injected content | production p10 | ATRX p60 | mvcc p20 |
|---|---|---|---|
| None (control) | 0/2 | 2/2 | 2/2 |
| Small neutral filler | 1/2 | 2/2 | 2/2 |
| Large neutral filler | 0/2 | -- | -- |
| Stale output | 0/2 | 2/2 | 2/2 |
| False success claim | 0/2 | 1/2 | 2/2 |
| Failed approach, before failure | 0/2 | 1/2 | 2/2 |
| Failed approach, failure shown | 0/2 | 0/2 | 2/2 |
| Leftover files (debris) | 0/2 | 1/2 | -- |

ATRX p60, failed approach with its failure shown against the control: 0/2 vs 2/2, Fisher exact p = 0.33 (two-sided).

### Harm test: Ledger review on healthy checkpoints

| Healthy checkpoint | Ledger review solved | No-injection control solved |
|---|---|---|
| production p10 | 0/2 | 0/2 |
| ATRX p60 | 2/2 | 2/2 |
| mvcc p20 | 2/2 | 2/2 |

Cost: $23.61 for 108 branches (followup_anytime_review $2.32, followup_opus_decides $1.32, followup_stalled_told $2.70, pilot_harm $1.32, pilot_injection $8.26, pilot_main $7.68).

Branches per set: followup_anytime_review 6, followup_opus_decides 6, followup_stalled_told 12, pilot_harm 6, pilot_injection 42, pilot_main 36. 96 branches ran on 2026-11-03 and 12 follow-up branches on 2026-11-04.
