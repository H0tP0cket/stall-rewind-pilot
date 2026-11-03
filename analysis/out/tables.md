### Main test: solved / branches, per stall and arm

| Stall (type) | Screen: continue | Screen: rewind | Continue | Ledger review | Context review | Forced rewind | Ledger read-only | Context read-only |
|---|---|---|---|---|---|---|---|---|
| production-planning (stuck) | 0/5 | 3/5 | 0/2 | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 |
| atrx-vep-crispr (under-commit) | 4/5 | 2/5 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 |
| heat-pump-warranty (over-commit) | 2/5 | 2/5 | 2/2 | 2/2 | 1/2 | 0/2 | 1/2 | 2/2 |
| **All three** | | | 2/6 | 2/6 | 1/6 | 3/6 | 1/6 | 2/6 |

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

### Harm test: Ledger review on healthy checkpoints

| Healthy checkpoint | Ledger review solved | No-injection control solved |
|---|---|---|
| production p10 | 0/2 | 0/2 |
| ATRX p60 | 2/2 | 2/2 |
| mvcc p20 | 2/2 | 2/2 |

Cost: $19.97 for 96 branches (followup_stalled_told $2.70, pilot_harm $1.32, pilot_injection $8.26, pilot_main $7.68).
