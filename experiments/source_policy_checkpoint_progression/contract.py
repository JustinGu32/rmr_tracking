"""Pure classification contract for strict source checkpoint progression."""

from __future__ import annotations

from collections.abc import Mapping

SHORT_0 = "short_model_0"
LONG_0 = "long_model_0"
SHORT_500 = "short_model_500"
LONG_500 = "long_model_500"
SHORT_999 = "short_model_999"
LONG_999 = "long_model_999"
REQUIRED_ARMS = (SHORT_0, LONG_0, SHORT_500, LONG_500, SHORT_999, LONG_999)
EARLY_LONG_ARMS = (LONG_0, LONG_500)
COMPLETE = "short-source-completes"
FAIL = "short-source-fails"
MIXED = "mixed-short-source-competence"
VALID_ARM_OUTCOMES = {COMPLETE, FAIL, MIXED}


def classify_progression(arm_outcomes: Mapping[str, str]) -> dict[str, object]:
    """Classify whether the long run acquired short-prefix competence over time."""
    exact_arms = set(arm_outcomes) == set(REQUIRED_ARMS)
    outcomes_valid = exact_arms and all(
        arm_outcomes[name] in VALID_ARM_OUTCOMES for name in REQUIRED_ARMS
    )
    endpoint_controls_valid = bool(
        outcomes_valid
        and arm_outcomes[SHORT_999] == COMPLETE
        and arm_outcomes[LONG_999] == FAIL
    )

    if not endpoint_controls_valid:
        outcome = "invalid-execution"
    else:
        early_long_outcomes = [arm_outcomes[name] for name in EARLY_LONG_ARMS]
        if MIXED in early_long_outcomes:
            outcome = "mixed-long-temporal-evidence"
        elif COMPLETE in early_long_outcomes:
            outcome = "long-acquires-then-loses-shared-prefix"
        else:
            outcome = "long-never-acquires-shared-prefix"

    return {
        "outcome": outcome,
        "required_arms": list(REQUIRED_ARMS),
        "early_long_arms": list(EARLY_LONG_ARMS),
        "arm_outcomes": {name: arm_outcomes.get(name) for name in REQUIRED_ARMS},
        "exact_arms": exact_arms,
        "arm_outcomes_valid": outcomes_valid,
        "endpoint_controls_valid": endpoint_controls_valid,
    }
