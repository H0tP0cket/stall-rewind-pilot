"""Auditable packet-text measurements, separate from provider usage accounting.

Local encodings count exact tokens for the named encoding, not necessarily for
the provider's model. No implementation here claims provider-exact counts.
Chat wrappers, restart wording and request serialization are outside this
measurement scope and remain part of the actual provider request/usage record.
Tolerance, encoding selection and attempt limits are calibration inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ._canonical import content_id, sha256_hex
from .render import approx_tokens

MEASUREMENT_REVISION = "packet-token-measurement.v1"
PACKET_SCOPE = "injected_packet_text_only; excludes restart scaffold and chat wrappers"


@dataclass(frozen=True)
class TokenCounter:
    model: str
    tokenizer_id: str
    tokenizer_revision: str
    implementation_revision: str
    fidelity: str
    count: Callable[[str], int] = field(repr=False, compare=False)
    settings: Mapping[str, Any] = field(default_factory=dict)
    scope: str = PACKET_SCOPE

    def __post_init__(self) -> None:
        if not all((self.model, self.tokenizer_id, self.tokenizer_revision, self.implementation_revision)):
            raise ValueError("token measurements require model, tokenizer and implementation revisions")
        if self.fidelity not in {"approximate_proxy", "local_tokenizer"}:
            raise ValueError("provider-exact fidelity requires separate verified provider evidence")
        if self.scope != PACKET_SCOPE:
            raise ValueError("this counter measures packet text only")

    def as_record(self) -> dict[str, Any]:
        return {"revision": MEASUREMENT_REVISION, "model": self.model,
                "tokenizer_id": self.tokenizer_id, "tokenizer_revision": self.tokenizer_revision,
                "implementation_revision": self.implementation_revision, "fidelity": self.fidelity,
                "provider_exact": False, "settings": dict(self.settings), "scope": self.scope}

    def measure(self, text: str, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = self.count(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token counter must return a nonnegative integer")
        body = {**self.as_record(), "text_sha256": sha256_hex(text), "byte_length": len(text.encode("utf-8")),
                "token_count": value, "context": dict(context or {})}
        return {**body, "measurement_id": content_id(body)}


def byte_proxy_counter(model: str) -> TokenCounter:
    return TokenCounter(model, "utf8-bytes-divided-by-four", "ceil-minimum-one.v1",
                        "trajectory-retention/render.approx_tokens.v1", "approximate_proxy", approx_tokens)


def tiktoken_counter(model: str, encoding_name: str) -> TokenCounter:
    """Use an explicitly selected local encoding; never infer a model mapping.

    The vocabulary, special-token map, pattern and library version are bound.
    tiktoken may populate its normal vocabulary cache on first use.
    """
    from importlib.metadata import version
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)
    vocabulary = {"mergeable_ranks": sorted((k.hex(), v) for k, v in encoding._mergeable_ranks.items()),
                  "special_tokens": encoding._special_tokens, "pattern": encoding._pat_str}
    return TokenCounter(model, f"tiktoken:{encoding_name}", content_id(vocabulary),
                        f"tiktoken:{version('tiktoken')}", "local_tokenizer",
                        lambda text: len(encoding.encode(text, disallowed_special=())),
                        settings={"disallowed_special": [], "model_encoding_mapping": "caller_selected_unverified"})


@dataclass(frozen=True)
class TokenMatchConfig:
    """No matching tolerance is assumed. First within-tolerance attempt wins.

    When both tolerances are supplied, the larger allowance applies. This is a
    declared rule, not a calibrated threshold. Failed/unmatched attempts count
    against max_attempts and all remain in the independent-summary record.
    """

    tolerance_tokens: int | None = None
    tolerance_fraction: float | None = None
    max_attempts: int = 1
    max_words: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if self.tolerance_tokens is not None and (isinstance(self.tolerance_tokens, bool)
                or not isinstance(self.tolerance_tokens, int) or self.tolerance_tokens < 0):
            raise ValueError("tolerance_tokens must be a nonnegative integer")
        if self.tolerance_fraction is not None and (isinstance(self.tolerance_fraction, bool)
                or not math.isfinite(self.tolerance_fraction) or self.tolerance_fraction < 0):
            raise ValueError("tolerance_fraction must be finite and nonnegative")
        if self.max_words is not None and (isinstance(self.max_words, bool)
                or not isinstance(self.max_words, int) or self.max_words < 1):
            raise ValueError("max_words must be a positive integer")
        if self.max_attempts > 1 and not self.configured:
            raise ValueError("multiple attempts require an explicit matching tolerance")

    @property
    def configured(self) -> bool:
        return self.tolerance_tokens is not None or self.tolerance_fraction is not None

    def allowance(self, target: int) -> int | None:
        if not self.configured:
            return None
        return max(self.tolerance_tokens or 0, math.ceil(target * (self.tolerance_fraction or 0)))

    def as_record(self) -> dict[str, Any]:
        return {"revision": "packet-token-matching.v1", "tolerance_tokens": self.tolerance_tokens,
                "tolerance_fraction": self.tolerance_fraction, "max_attempts": self.max_attempts,
                "max_words": self.max_words, "tolerance_rule": "max(absolute, ceil(target*fraction))",
                "selection_rule": "first_within_tolerance; otherwise no matched packet",
                "calibration_status": "parameters_supplied_not_scientifically_validated" if self.configured
                else "pending_no_tolerance_configured"}


def compare_measurements(target: Mapping[str, Any], candidate: Mapping[str, Any],
                         config: TokenMatchConfig) -> dict[str, Any]:
    identity_fields = ("model", "tokenizer_id", "tokenizer_revision", "implementation_revision", "fidelity",
                       "settings", "scope")
    if any(target[k] != candidate[k] for k in identity_fields):
        raise ValueError("token comparison requires the same model, tokenizer and measurement scope")
    target_count, candidate_count = target["token_count"], candidate["token_count"]
    tolerance = config.allowance(target_count)
    difference = candidate_count - target_count
    within = None if tolerance is None else abs(difference) <= tolerance
    return {"target_measurement_id": target["measurement_id"],
            "candidate_measurement_id": candidate["measurement_id"], "target_tokens": target_count,
            "candidate_tokens": candidate_count, "signed_difference": difference,
            "absolute_difference": abs(difference), "relative_difference": None if target_count == 0
            else difference / target_count, "allowed_difference": tolerance, "within_tolerance": within,
            "status": "not_configured" if within is None else "within_tolerance" if within else "outside_tolerance",
            "fidelity": target["fidelity"], "provider_exact": False, "matching_config": config.as_record()}
