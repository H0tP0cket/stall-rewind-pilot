"""T2.2 shared vocabulary: role and relation names, filter groups and versions.

Every T2.2 module imports names from here so the labelling, unit, packet and
check layers agree. Nothing here judges truth or usefulness: roles describe
what a sentence says, relations describe what the later wording explicitly
connects to.
"""

from __future__ import annotations

T22_VERSION = "t2.2/dev-2026-09-26"

# ---- pass A: one main role per worker-prose sentence (full distribution kept) ----
ROLE_OPTIONS = {
    "observation_report": "Reports something the agent saw or measured: a result, error, test outcome or observed behavior.",
    "factual_assertion": "States a fact about the code, environment, library or requirements as known or settled.",
    "explanation_or_hypothesis": "Offers an explanation or hypothesis for why something happens or what is wrong.",
    "proposal": "Suggests or considers an approach, fix or design without adopting it (maybe, could, one option is, what if).",
    "commitment": "Adopts or decides on an approach, fix or design that the agent will now pursue (I'll go with, the fix is to, let's do).",
    "intended_action": "States a specific action the agent is about to take on a named target (inspect, edit, run, test, revert).",
    "unresolved_question": "Raises a question or unknown the agent has not answered.",
    "result_claim": "Claims an outcome of the agent's own work: that a fix works, tests pass, a problem is solved, or the task is done.",
    "revision": "Says that an earlier idea, belief, plan or result of the agent's own was wrong, is being dropped, or is changed.",
    "retrospective_takeaway": "Draws a conclusion from something that already happened, framed to guide later work.",
    "working_material": "Intermediate working only: a calculation, worked example or step-by-step trace.",
    "narration": "Generic process talk with no specific content (e.g. 'let me look at the files', 'now I will implement it').",
    "mixed": "Clearly combines two or more of the above roles in one sentence.",
}
ROLE_THRESHOLD = 0.6

# Role groups used by filters and borderline reporting (a sentence's group is its main role's group).
ROLE_GROUPS = {
    "facts": {"observation_report", "factual_assertion"},
    "theories": {"explanation_or_hypothesis", "unresolved_question"},
    "plans": {"proposal", "commitment", "intended_action"},
    "claims": {"result_claim"},
    "revisions": {"revision"},
    "takeaways": {"retrospective_takeaway"},
    "working": {"working_material"},
    "filler": {"narration"},
    "mixed": {"mixed"},
}
APPROACH_ROOT_ROLES = {"proposal", "commitment"}
# Worker-prose roles that can be members of an approach unit.
APPROACH_MEMBER_ROLES = {"explanation_or_hypothesis", "proposal", "commitment", "intended_action",
                         "unresolved_question", "result_claim", "revision", "mixed"}
EVIDENCE_ROLES = {"observation_report", "factual_assertion"}

# ---- pass B: explicit relation from a later worker sentence to an earlier item ----
RELATION_OPTIONS = {
    "tests_or_checks": "The later text explicitly says it tests, checks or verifies the earlier idea or plan.",
    "continues_or_builds_on": "The later text explicitly continues, refines or builds on the earlier idea or plan.",
    "replaces": "The later text explicitly switches from the earlier idea or plan to a different one.",
    "withdraws": "The later text explicitly says the earlier idea or plan is wrong, does not work, or is being dropped.",
    "confirms": "The later text explicitly says the earlier idea turned out to be correct.",
    "restates": "The later text repeats the earlier idea or plan without changing it.",
    "answers": "The later text explicitly answers or resolves the earlier open question.",
    "reads_output_as_supporting": "The earlier item is a tool output, and the later text explicitly reads that output as showing something works, passes or is confirmed.",
    "reads_output_as_refuting": "The earlier item is a tool output, and the later text explicitly reads that output as showing something fails, is wrong or is contradicted.",
    "no_explicit_relation": "The later text does not itself state a connection; any link would have to be guessed from topic similarity.",
    "ambiguous": "Cannot tell whether the wording connects them.",
}
RELATION_THRESHOLD = 0.7
OUTPUT_RELATIONS = {"reads_output_as_supporting", "reads_output_as_refuting"}
# Explicit relations that join two worker sentences into one approach unit.
MEMBER_RELATIONS = {"continues_or_builds_on", "tests_or_checks", "restates", "answers"}
STATUS_RELATIONS = {"withdraws": "withdrawn", "replaces": "replaced", "confirms": "confirmed",
                    "reads_output_as_supporting": "confirmed", "reads_output_as_refuting": "refuted"}
APPROACH_STATUSES = ("open", "confirmed", "refuted", "withdrawn", "replaced")

# ---- link tiers ----
LINK_TIER_EXPLICIT = "explicit_wording"     # Jev relation at or above RELATION_THRESHOLD: may join units
LINK_TIER_BUILTIN = "structural"            # same_turn / result_of from the event log: may join units
LINK_TIER_CANDIDATE = "same_thing_named"    # shared named artifact: retrieval candidate only, NEVER joins units
# Contingency tier (ratified): Jev judges two plan/theory sentences to be the same approach even though neither
# wording mentions the other. Recorded separately so every result can be recomputed on explicit links only.
LINK_TIER_INFERRED = "inferred_same_approach"
SAME_APPROACH_RELATION = "same_approach"
SAME_APPROACH_ROLES = {"explanation_or_hypothesis", "proposal", "commitment", "intended_action", "unresolved_question"}
SAME_APPROACH_THRESHOLD = 0.7

# ---- retrieval (which pairs Jev is asked about) ----
MAX_ARTIFACT_CANDIDATES = 6
RECENT_ALWAYS = 2

# ---- packet ----
BUDGETS = (1500, 3000, 6000)
DEFAULT_BUDGET = 3000
RANDOM_TOLERANCE = 0.10
FILTER_SCOPES = ("current_approach", "global")
