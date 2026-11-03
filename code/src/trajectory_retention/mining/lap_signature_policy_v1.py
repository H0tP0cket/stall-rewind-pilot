"""V5-only, narrowly scoped fixes demonstrated by the v11 saved grade pairs.

The original task policy, raw checks and every applied before/after substitution
remain in the recorded signature evidence. No installed task seal is rewritten.
"""
from dataclasses import replace
import re

from .signatures import FailureSignaturePolicy, NuisanceRule

VERSION = "v12-demonstrated-nuisance/v1"


def extend_lap_signature_policy(policy: FailureSignaturePolicy | None) -> FailureSignaturePolicy | None:
    if policy is None:
        return None
    rules = []
    if re.fullmatch(r"scb-circuit-eval-adapted-v[1-9][0-9]*", policy.task_id):
        rules.append(NuisanceRule(path=("native_phases", "*", "longrepr"),
            pattern=r"(?m)^(E +\+ +where [^\n]*?<built-in method get of dict object at )0x[0-9a-fA-F]+(>\('exit_code'\))",
            replacement=r"\1<process-allocation>\2",
            reason="Observed Circuit C14/C15 pytest assertion explanation differs only in dict.get receiver allocation. Preserve the compared value, exit_code key and all other addresses."))
        rules.append(NuisanceRule(path=("native_phases", "*", "longrepr"),
            pattern=r"(?m)^(E +\+ +where <built-in method get of dict object at )0x[0-9a-fA-F]+(> = [^\n]*\.get)$",
            replacement=r"\1<process-allocation>\2",
            reason="The same Circuit assertion prints its dict.get receiver again in the next pytest explanation line. Preserve the dictionary and compared values."))
    elif re.fullmatch(r"swesmith-sqlfluff-(?:cli-lint-jtr3hbvg|parse-render-8rm0rrva)-v[1-9][0-9]*", policy.task_id):
        rules.append(NuisanceRule(path=("native_record", "phases", "*", "longrepr"),
            pattern=r"/tmp/pytest-of-root/pytest-[0-9]+(?=/test_(?:_cli__command_(?:lint|parse)|cli_encoding_|_cli__helpers__(?:cli_table|colorize))[A-Za-z0-9_]*(?:/|['\"\s)]|$))",
            replacement="/tmp/pytest-of-root/pytest-<session>",
            reason="Observed SQLFluff pytest CLI-test basetemp session counter; retain test name, parameter/case suffix, files, diagnostics and all other paths/numbers."))
    if not rules:
        return policy
    if policy.allow_status_only:
        raise ValueError("Demonstrated nuisance extension requires actual diagnostics")
    if policy.version.endswith(VERSION):
        return policy
    return replace(policy, policy_id=policy.policy_id + ":" + VERSION,
                   version=policy.version + ":" + VERSION,
                   nuisance_rules=policy.nuisance_rules + tuple(rules))
