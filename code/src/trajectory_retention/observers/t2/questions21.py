"""T2.1 descriptive question registry (guide §6.2).

Every question describes what a span says, how it is expressed, where it came
from or what it explicitly connects to. None asks whether retaining material
will help recovery. ``PROHIBITED_FIELDS`` lists T2 utility fields that may be
read as historical metadata but never used for T2.1 inclusion, ranking, links
or unit membership.
"""

from __future__ import annotations

import json
from typing import Any

REGISTRY_VERSION = "t2.1-questions/v1"

ROLE_OPTIONS = {
    "observation_report": "Reports something the agent saw or measured: a result, error, test outcome or observed behavior.",
    "factual_assertion": "States a fact about the code, environment, library or requirements as known or settled.",
    "explanation_or_hypothesis": "Offers an explanation or hypothesis for why something happens or what is wrong.",
    "strategy_or_design_choice": "Proposes or adopts an approach, design, or way of implementing or solving something.",
    "intended_action": "States a specific action the agent is about to take on a named target (inspect, edit, run, test, revert).",
    "unresolved_question": "Raises a question or unknown the agent has not answered.",
    "retrospective_takeaway": "Draws a conclusion from something that already happened, framed to guide later work.",
    "working_material": "Intermediate working: calculation, worked example, code fragment, restated input or step-by-step trace.",
    "narration": "Generic process talk with no specific content (e.g. 'let me look at the files', 'now I will implement it').",
    "mixed": "Clearly combines two or more of the above roles in one span.",
}
CERTAINTY_OPTIONS = {
    "asserted": "Stated as true or settled.",
    "tentative": "Hedged or tentative (maybe, might, probably, I think, possibly, not sure).",
    "interrogative": "Phrased as a question.",
    "conditional_or_mixed": "Conditional (if/unless/when) or contains both settled and tentative parts.",
    "not_applicable": "Not a proposition (an action, instruction, heading or fragment).",
}
SPECIFICITY_OPTIONS = {
    "specific": "Names a concrete result, value, file, function, command, behavior, interpretation or attempt about this work.",
    "generic": "Contains no concrete content about this work.",
    "ambiguous": "Cannot tell from the span and its context.",
}
RESTATEMENT_OPTIONS = {
    "restates_requirement": "Only repeats what the task, specification or instructions say, adding nothing of the agent's own.",
    "adds_agent_content": "Adds the agent's own interpretation, observation, decision, attempt or claim (even if it cites the spec).",
    "ambiguous": "Cannot tell.",
}
OUTPUT_KIND_OPTIONS = {
    "execution_result": "Output produced by running something: an error, exit status, test outcome, or printed value.",
    "displayed_file_or_code": "Contents of a file or source code being shown so the agent can read it.",
    "listing_or_progress": "A directory listing, search hit list, progress line or boilerplate.",
    "other": "None of the above.",
}
RELATION_OPTIONS = {
    "tests_or_checks": "The later text explicitly says it tests, checks or verifies the earlier idea or plan.",
    "continues_or_builds_on": "The later text explicitly continues, refines or builds on the earlier idea or plan.",
    "replaces": "The later text explicitly switches from the earlier idea or plan to a different one.",
    "withdraws": "The later text explicitly says the earlier idea or plan is wrong, does not work, or is being dropped.",
    "confirms": "The later text explicitly says the earlier idea turned out to be correct.",
    "restates": "The later text repeats the earlier idea or plan without changing it.",
    "no_explicit_relation": "The later text does not itself state a connection; any link would have to be guessed from topic similarity.",
    "ambiguous": "Cannot tell whether the wording connects them.",
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "role": {"type": "choice", "applies_to": "worker prose (text, replayed reasoning, provider summary, code comment)",
             "instruction": "What communicative role does the exact span `{ref}` express, read in its local context "
                            "`source_text_excerpt`?", "options": ROLE_OPTIONS, "threshold": 0.6},
    "certainty": {"type": "choice", "applies_to": "worker prose",
                  "instruction": "How is the proposition in `{ref}` expressed? Judge the wording, not whether it is true.",
                  "options": CERTAINTY_OPTIONS, "threshold": 0.6},
    "specificity": {"type": "choice", "applies_to": "worker prose and tool-output blocks",
                    "instruction": "Does `{ref}` contain concrete content about this work (a result, value, file, "
                                   "function, command, behavior, interpretation or attempt)? This is not a question "
                                   "about usefulness.", "options": SPECIFICITY_OPTIONS, "threshold": 0.6,
                    "exclusion_threshold": 0.8},
    "restatement": {"type": "choice", "applies_to": "worker prose",
                    "instruction": "Does `{ref}` only repeat a supplied requirement, without adding the agent's own "
                                   "interpretation, observation, decision or attempt?",
                    "options": RESTATEMENT_OPTIONS, "threshold": 0.6, "exclusion_threshold": 0.8},
    "output_kind": {"type": "choice", "applies_to": "tool-output blocks (with the producing command in `command`)",
                    "instruction": "`{ref}` is part of the output of the command in `command`. What kind of output is it?",
                    "options": OUTPUT_KIND_OPTIONS, "threshold": 0.6, "exclusion_threshold": 0.8},
    "relation": {"type": "choice", "applies_to": "ordered pair (earlier worker item, later worker item), same session, "
                                                 "earlier available at the later item's cut",
                 "instruction": "Read `later` and `earlier.{ref}`. Does the wording of `later` itself explicitly "
                                "connect it to `earlier.{ref}`? Choose no_explicit_relation if a connection would "
                                "have to be inferred from shared topic.",
                 "options": RELATION_OPTIONS, "threshold": 0.7},
}

PROHIBITED_FIELDS = {
    "keep": "T2 utility question ('would this help an engineer who takes over'); historical metadata only",
    "refers": "T2 menu-of-positives link question; exploratory/historical only",
    "status": "T2 marker-gated status question with menu-picked target; historical only",
    "spec_statement": "T2 paraphrase question at 0.6 exclusion; superseded by `restatement` with ambiguity option",
}


def registry_record() -> dict[str, Any]:
    return {"version": REGISTRY_VERSION, "classifier": "TypeSafe jev-1.13.0 (System One)",
            "questions": QUESTIONS, "prohibited_fields": PROHIBITED_FIELDS,
            "notes": ["Thresholds are development settings, not validated accuracy targets.",
                      "Answers below threshold are recorded as unknown with the full distribution.",
                      "Classifier-driven exclusion uses the stricter exclusion_threshold.",
                      "Requests retried with reduced context are labelled with method suffix ':minimal-context'."]}


if __name__ == "__main__":
    print(json.dumps(registry_record(), indent=1))
