"""T2.2 descriptive question registry ``t2.2-questions/v1``.

T describes; it never judges. Every question asks what a span says, how it is
worded, what kind of output it is, what the later wording explicitly connects
to, or whether a kept sentence says the same thing as removed ones. None asks
whether something is true, useful or worth keeping.

Changes from ``t2.1-questions/v1``:

* ``role`` uses the T2.2 role set (``vocab22.ROLE_OPTIONS``) and asks for the
  single main role of the sentence; the full distribution is kept.
* ``relation`` uses ``vocab22.RELATION_OPTIONS``; the earlier item may now be a
  tool output shown with its producing command.
* ``paraphrase`` is new: used only to make removal views real (a kept sentence
  that restates a removed one goes with it).
* ``same_approach`` is new (v2, the ratified contingency when explicit links are
  too sparse): whether two plan/theory sentences belong to the same approach
  even though neither wording mentions the other. It describes what the two
  sentences are about, never whether the approach is good; links from it form
  their own tier (``LINK_TIER_INFERRED``).

``certainty``, ``specificity``, ``restatement`` and ``output_kind`` are the
T2.1 questions, unchanged (the same objects). ``PROHIBITED_FIELDS`` extends the
T2.1 list; those fields may be read as historical metadata but never used for
T2.2 inclusion, ranking, links, units or filters.
"""

from __future__ import annotations

import json
from typing import Any

from .questions21 import PROHIBITED_FIELDS as PROHIBITED_FIELDS_21
from .questions21 import QUESTIONS as QUESTIONS_21
from .vocab22 import (RELATION_OPTIONS, RELATION_THRESHOLD, ROLE_OPTIONS, ROLE_THRESHOLD, SAME_APPROACH_RELATION,
                      SAME_APPROACH_THRESHOLD)

REGISTRY_VERSION = "t2.2-questions/v2"

PARAPHRASE_OPTIONS = {
    "restates_removed": "The kept sentence says the same thing as at least one removed item, in the same or "
                        "different words.",
    "adds_different_content": "The kept sentence says something different from every removed item "
                              "(sharing a topic, file, function, test or error name is not saying the same thing).",
    "ambiguous": "Cannot tell.",
}
PARAPHRASE_THRESHOLD = 0.7
SAME_APPROACH_OPTIONS = {
    SAME_APPROACH_RELATION: "Both statements are about the same idea of what is wrong or the same plan for fixing it.",
    "different_approach": "The statements are about different ideas of what is wrong or different plans, even if they "
                          "mention the same file, function, test or error.",
    "unclear": "Cannot tell.",
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "role": {"type": "choice", "applies_to": "worker prose (text, replayed reasoning, provider summary, code comment)",
             "instruction": "What is the main communicative role of the exact sentence `{ref}`, read in its local "
                            "context `source_text_excerpt`? Choose the one role that describes the sentence as a "
                            "whole; choose mixed only if it clearly combines two or more roles.",
             "options": ROLE_OPTIONS, "threshold": ROLE_THRESHOLD},
    "certainty": QUESTIONS_21["certainty"],
    "specificity": QUESTIONS_21["specificity"],
    "restatement": QUESTIONS_21["restatement"],
    "output_kind": QUESTIONS_21["output_kind"],
    "relation": {"type": "choice",
                 "applies_to": "ordered pair (earlier item, later worker sentence), same session, earlier available at "
                               "the later item's cut; the earlier item is worker prose or a tool output",
                 "instruction": "Read `later` and `earlier.{ref}`. `earlier.{ref}` is either an earlier worker "
                                "statement or a tool output shown with the command that produced it. Does the wording "
                                "of `later` itself explicitly connect it to `earlier.{ref}`? Choose "
                                "no_explicit_relation if a connection would have to be inferred from shared topic or "
                                "from naming the same file, function, test or error.",
                 "options": RELATION_OPTIONS, "threshold": RELATION_THRESHOLD},
    "paraphrase": {"type": "choice", "applies_to": "kept worker sentence against the numbered list of removed items "
                                                   "of one removal view",
                   "instruction": "Does the kept sentence `{ref}` say the same thing as any of the removed items in "
                                  "`removed`? Judge what is said, not whether it is true.",
                   "options": PARAPHRASE_OPTIONS, "threshold": PARAPHRASE_THRESHOLD},
    "same_approach": {"type": "choice",
                      "applies_to": "ordered pair (earlier, later) of worker sentences whose main role is a theory or "
                                    "plan role (SAME_APPROACH_ROLES), same session, earlier available at the later "
                                    "item's cut",
                      "instruction": "Are the statements `later` and `earlier.{ref}` part of the same approach - the "
                                     "same idea about what is wrong or the same plan for fixing it - even if neither "
                                     "mentions the other? Judge what they are about, not whether the approach is good "
                                     "or correct.",
                      "options": SAME_APPROACH_OPTIONS, "threshold": SAME_APPROACH_THRESHOLD},
}

PROHIBITED_FIELDS = {
    **PROHIBITED_FIELDS_21,
    "keep": "utility judgement ('would this help whoever takes over'); answers the experiment in advance; never asked",
    "grounding": "whether a statement is supported by the record; judging a claim means viewing it from a later point",
    "claim_honesty": "whether a result claim is honest or correct; the grader comparison is analysis-only "
                     "(t22_claim_grader)",
    "source_4way": "4-way attribution of where a statement's content came from; a judgement, excluded by ratification",
}


def registry_record() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "classifier": "TypeSafe jev-1.13 (System One)",
            "questions": QUESTIONS, "prohibited_fields": PROHIBITED_FIELDS,
            "unchanged_from": {"registry": "t2.1-questions/v1",
                               "questions": ["certainty", "specificity", "restatement", "output_kind"]},
            "notes": ["Thresholds are development settings, not validated accuracy targets.",
                      "Answers below threshold are recorded as unknown with the full distribution.",
                      "Classifier-driven exclusion uses the stricter exclusion_threshold (unchanged from T2.1).",
                      "Requests retried with reduced context are labelled with method suffix ':minimal-context'.",
                      "Relation links to a tool output are stored only for reads_output_* relations; "
                      "reads_output_* links are stored only when the earlier item is a tool output.",
                      "same_approach links are their own tier (inferred_same_approach), asked only between theory/plan "
                      "sentences and only where no explicit link exists; results must be reportable without them."]}


if __name__ == "__main__":
    print(json.dumps(registry_record(), indent=1))
