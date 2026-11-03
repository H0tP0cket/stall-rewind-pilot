"""Benchmark task prompts, separate from mini-SWE's tool/termination interface.

Only a stage's already-public instructions enter these builders. They do not
read grades, private tests, other stages, or previous conversations. Requirement
and evaluator identities remain those of the package; the independently bound
prompt version identifies the new worker input.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from ..contracts import StageSpec

VERSION = "benchmark-task-prompt.v2"
# The swe-smith family has a second, later rendering (native instance template,
# swesmith_prompt_v3). Packages keep the exact prompt version they bind; every
# other family renders only VERSION.
SWESMITH_VERSION_V3 = "benchmark-task-prompt.v3"
FAMILY_VERSIONS = {"swe-smith": (VERSION, SWESMITH_VERSION_V3)}
SCB_SOURCE = "https://github.com/SprocketLab/slop-code-bench/blob/06b5c0687d4c05ee502e9696a4d0c22fc1eec5e0/configs/prompts/just-solve.jinja"
COMMIT0_SOURCE = "https://github.com/commit-0/commit0/blob/1123db33219ee1bfcc735304cb4308663e329aeb/agent/cli.py"
SWESMITH_SOURCE = "https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/agent/swesmith_infer.yaml"
# agent.templates.instance_template of SWESMITH_SOURCE, byte-for-byte (a YAML
# `|-` block, so no trailing newline). swesmith_gen_claude.yaml and
# swesmith_gen_gpt.yaml at the same commit carry the identical template.
SWESMITH_INSTANCE_TEMPLATE = (
    "<uploaded_files>\n"
    "{{working_dir}}\n"
    "</uploaded_files>\n"
    "I've uploaded a python code repository in the directory {{working_dir}}. Consider the following PR description:\n"
    "\n"
    "<pr_description>\n"
    "{{problem_statement}}\n"
    "</pr_description>\n"
    "\n"
    "Can you help me implement the necessary changes to the repository so that the requirements specified in the "
    "<pr_description> are met?\n"
    "I've already taken care of all changes to any of the test files described in the <pr_description>. This means "
    "you DON'T have to modify the testing logic or any of the tests in any way!\n"
    "Your task is to make the minimal changes to non-tests files in the {{working_dir}} directory to ensure the "
    "<pr_description> is satisfied.\n"
    "Follow these steps to resolve the issue:\n"
    "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\n"
    "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, "
    "to confirm the error\n"
    "3. Edit the source code of the repo to resolve the issue\n"
    "4. Rerun your reproduce script and confirm that the error is fixed!\n"
    "5. Think about edgecases and make sure your fix handles them as well\n"
    "Your thinking should be thorough and so it's fine if it's very long.")
SWESMITH_INSTANCE_TEMPLATE_SHA256 = "2cc3277023c16ff6e98bccbea7deadbe80d02bd6912dada7817bd8400a388b27"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkerTaskPrompt:
    family: str
    stage_id: str
    target_id: str
    requirement_version: str
    specification_sha256: str
    text: str
    source: str
    adaptations: tuple[str, ...] = ()
    version: str = VERSION

    def binding(self) -> dict:
        return {"version": self.version, "family": self.family,
                "stage_id": self.stage_id, "target_id": self.target_id,
                "requirement_version": self.requirement_version,
                "specification_sha256": self.specification_sha256,
                "rendered_sha256": _sha(self.text), "source": self.source,
                "adaptations": list(self.adaptations)}


def _prompt(stage: StageSpec, family: str, prefix: str, *, source: str,
            suffix: str = "", adaptations: tuple[str, ...] = ()) -> WorkerTaskPrompt:
    return WorkerTaskPrompt(family=family, stage_id=stage.stage_id, target_id=stage.target_id,
        requirement_version=stage.requirement_version, specification_sha256=_sha(stage.instructions),
        text=prefix + stage.instructions + suffix, source=source, adaptations=adaptations)


def scb_prompt(stage: StageSpec) -> WorkerTaskPrompt:
    # just-solve.jinja's virtual-environment / requirements.txt lines are omitted:
    # they address the original Claude Code / Codex runner, and our evaluator
    # invokes the entrypoint with the container interpreter (it never activates
    # a worker-created environment), so obeying them could hide dependencies
    # from grading. The specification itself is delivered byte-for-byte.
    return _prompt(stage, "slop-code-bench",
        "Implement a program that 100% solves the specification. That is all you need to do.\n\n"
        "Your task is:\n", source=SCB_SOURCE,
        adaptations=("Preserve packaged specification bytes, including existing versioned public clarifications.",
                     "Omit the just-solve virtual-environment/requirements.txt instructions; the packaged evaluator runs the "
                     "entrypoint with the container interpreter and does not activate worker-created environments.",
                     "Shared mini-SWE tools; package-native W carry and fresh H/R at stage transitions."))


def commit0_prompt(stage: StageSpec) -> WorkerTaskPrompt:
    return _prompt(stage, "commit0",
        "Here is your task:\n"
        "You need to complete the implementations for all functions (i.e., those with pass statements) and pass the unit tests.\n"
        "Do not change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.\n\n",
        source=COMMIT0_SOURCE,
        adaptations=("Omit Aider-specific search/replace whitespace instructions and its score-zero warning.",
                     "Append the unchanged packaged public specification and evaluator-adaptation disclosures.",
                     "Use the shared mini-SWE worker; original public tests remain available, private grades are not returned."))


def swesmith_prompt(stage: StageSpec) -> WorkerTaskPrompt:
    # The published SWE-smith instance template, with `{{working_dir}}` rendered
    # as the shell's current working directory (the repository), and the
    # packaged issue plus its public compatibility requirements filling the
    # <pr_description> slot. Editor/XML tool mechanics belong to SWE-agent and
    # its closing "thinking should be thorough" line steers the model rather
    # than the task; both are omitted.
    return _prompt(stage, "swe-smith",
        "I've uploaded a python code repository in the current working directory. Consider the following PR description:\n\n"
        "<pr_description>\n",
        suffix=(
            "\n</pr_description>\n\n"
            "Can you help me implement the necessary changes to the repository so that the requirements specified in the "
            "<pr_description> are met?\n"
            "I've already taken care of all changes to any of the test files described in the <pr_description>. This means "
            "you DON'T have to modify the testing logic or any of the tests in any way!\n"
            "Your task is to make the minimal changes to non-tests files in the current working directory to ensure the "
            "<pr_description> is satisfied.\n"
            "Follow these steps to resolve the issue:\n"
            "1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>\n"
            "2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, "
            "to confirm the error\n"
            "3. Edit the source code of the repo to resolve the issue\n"
            "4. Rerun your reproduce script and confirm that the error is fixed!\n"
            "5. Think about edgecases and make sure your fix handles them as well\n"),
        source=SWESMITH_SOURCE,
        adaptations=("Published instance template wording retained; `{{working_dir}}` rendered as the current working directory.",
                     "The packaged issue and its public compatibility requirements fill the <pr_description> slot verbatim.",
                     "Omit SWE-agent editor/XML tool mechanics and the model-steering long-thinking line."))


def swesmith_prompt_v3(stage: StageSpec) -> WorkerTaskPrompt:
    """The native SWE-smith instance template, rendered verbatim (benchmark-task-prompt.v3).

    `{{working_dir}}` is the worker's actual repository directory (the runner's
    W mount, /workspace) and `{{problem_statement}}` is the packaged problem
    statement, byte-for-byte; neither substitution renders the other. Only the
    SWE-agent system template (its XML function-calling format and
    str_replace_editor/submit tools) is not used: every lane shares one mini-SWE
    bash harness, whose tool/submission rules follow this text.
    """
    from ..execution.environment import WORKSPACE
    if _sha(SWESMITH_INSTANCE_TEMPLATE) != SWESMITH_INSTANCE_TEMPLATE_SHA256:
        raise ValueError("SWE-smith instance template differs from the pinned upstream text")
    prefix, marker, suffix = SWESMITH_INSTANCE_TEMPLATE.partition("{{problem_statement}}")
    if not marker or "{{problem_statement}}" in suffix:
        raise ValueError("SWE-smith instance template must hold exactly one problem statement slot")
    render = lambda part: part.replace("{{working_dir}}", WORKSPACE)  # noqa: E731
    prompt = _prompt(stage, "swe-smith", render(prefix), suffix=render(suffix), source=SWESMITH_SOURCE,
        adaptations=("Native instance template (agent/swesmith_infer.yaml; identical in swesmith_gen_claude.yaml and "
                     "swesmith_gen_gpt.yaml) rendered verbatim; {{working_dir}} = " + WORKSPACE +
                     ", the worker's repository directory.",
                     "The packaged problem_statement fills {{problem_statement}} byte-for-byte.",
                     "The SWE-agent system template (XML function calls, str_replace_editor and submit tools) is not "
                     "used; every lane shares the mini-SWE bash harness and its tool/submission rules."))
    return replace(prompt, version=SWESMITH_VERSION_V3)


def terminal_bench_prompt(stage: StageSpec) -> WorkerTaskPrompt:
    # instruction.md already IS the benchmark task prompt. Terminus's JSON,
    # tmux, and confirmation protocol belongs to its agent, not this task.
    return _prompt(stage, "terminal-bench", "", source="package:public/instruction.md",
        adaptations=("Use the original packaged task instruction unchanged with shared mini-SWE tools and submission.",))


def prompt_for_task(task, stage: StageSpec) -> WorkerTaskPrompt:
    """Optional adapter hook; development/custom adapters keep their own text."""
    builder = getattr(task, "worker_prompt", None)
    prompt = (builder(stage.stage_id) if builder is not None else
              _prompt(stage, "adapter-native", "", source="adapter:StageSpec.instructions"))
    if not isinstance(prompt, WorkerTaskPrompt) or prompt.version not in FAMILY_VERSIONS.get(prompt.family, (VERSION,)) or (
        prompt.stage_id, prompt.target_id, prompt.requirement_version, prompt.specification_sha256
    ) != (stage.stage_id, stage.target_id, stage.requirement_version, _sha(stage.instructions)):
        raise ValueError("Worker task prompt differs from its declared stage/specification")
    if not isinstance(prompt.text, str) or not prompt.text.strip():
        raise ValueError("Worker task prompt must be nonempty")
    return prompt
