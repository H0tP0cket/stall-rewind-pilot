"""Reviewed findings for this new lane; final human validation remains pending."""
AGENT = 'Codex admission review, 2026-10-28; final human validation remains pending'
REVIEWS: dict[str, dict] = {'a0b86f0560763950': {'task': 'pretrain-shard-corruption',
                      'kind': 'contents',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Reviewed all 13 shared lines: the canary warning and unchanged model '
                                'architecture/loss operations already supplied in train_pretrain.py. The '
                                'grader implements that public architecture to load and validate the '
                                'submitted checkpoint. The git finding is the starter copy of the same '
                                'public training code. No corruption repair, private validation data, '
                                'expected score or hidden test assertion appears in these lines.'},
 'f8ea9826c996d8e0': {'task': 'pretrain-shard-corruption',
                      'kind': 'contents',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Reviewed all 13 shared lines: the canary warning and unchanged model '
                                'architecture/loss operations already supplied in train_pretrain.py. The '
                                'grader implements that public architecture to load and validate the '
                                'submitted checkpoint. The git finding is the starter copy of the same '
                                'public training code. No corruption repair, private validation data, '
                                'expected score or hidden test assertion appears in these lines.'},
 'e6516d814ff21ef1': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 '7baa36e965725200': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 'bb0f428cb05f448c': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 '5e4e5a8c544d199c': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 '62fc165c846c3c8b': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 '3a5f79f5e9450cfb': {'task': 'pretrain-shard-corruption',
                      'kind': 'comments',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Exact reviewed comment is only the standard benchmark contamination '
                                'warning. No corruption location, repaired bytes, hidden validation corpus, '
                                'threshold or answer is disclosed.'},
 'c89f28b48faec139': {'task': 'distributed-dedup',
                      'kind': 'contents',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Inspected all three shared lines: the standard contamination warning, the '
                                'public required implementation class declaration, and the public Submission '
                                'provider name already in the throwing stub. No dedup algorithm, test '
                                'expectation, private data, threshold or reference implementation body is '
                                'shared. Preview uses the exact upstream reference source replacement; the '
                                'actual full gate must independently rescan the compiled reference state, '
                                'and unknown findings remain unresolved.'},
 '37bdd0ccbc4d0fb2': {'task': 'distributed-dedup',
                      'kind': 'contents',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Reviewed the complete matching line from actual compiled-reference and '
                                'starter sbt debug logs. It is the generic incremental compiler '
                                'InitialChanges message showing empty change sets, without source code, '
                                'algorithm, task data, test identifiers, thresholds or expected results. '
                                'These generated build-log boilerplate matches do not expose the reference '
                                'solution.'},
 '9170562c01bc4884': {'task': 'distributed-dedup',
                      'kind': 'contents',
                      'verdict': 'not_a_leak',
                      'reviewer': 'Codex admission review, 2026-10-28; final human validation remains '
                                  'pending',
                      'reason': 'Reviewed the complete matching line from actual compiled-reference and '
                                'starter sbt debug logs. It is the generic incremental compiler '
                                'InitialChanges message showing empty change sets, without source code, '
                                'algorithm, task data, test identifiers, thresholds or expected results. '
                                'These generated build-log boilerplate matches do not expose the reference '
                                'solution.'}}

def reviews_for(task: str) -> dict[str, dict]:
    return {k: v for k, v in REVIEWS.items() if v["task"] == task}
