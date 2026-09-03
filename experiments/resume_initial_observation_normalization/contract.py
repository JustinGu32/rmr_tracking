"""Pure outcome contract for the corrected first-resume-observation test."""

from __future__ import annotations

from collections.abc import Mapping

BASELINE_SHORT = "baseline_short"
BASELINE_LONG = "baseline_long"
POST_SHORT = "post_update_1_short"
POST_LONG = "post_update_1_long"
REQUIRED_ARMS = (BASELINE_SHORT, BASELINE_LONG, POST_SHORT, POST_LONG)

SHORT_COMPLETE = "short-source-completes"
SHORT_FAIL = "short-source-fails"
SHORT_MIXED = "mixed-short-source-competence"
LONG_COMPLETE = "source-completes-exact-long"
LONG_FAIL = "source-fails-exact-long"
LONG_MIXED = "mixed-source-competence"

EXPECTED_OUTCOMES = {
    BASELINE_SHORT: {SHORT_COMPLETE, SHORT_FAIL, SHORT_MIXED},
    BASELINE_LONG: {LONG_COMPLETE, LONG_FAIL, LONG_MIXED},
    POST_SHORT: {SHORT_COMPLETE, SHORT_FAIL, SHORT_MIXED},
    POST_LONG: {LONG_COMPLETE, LONG_FAIL, LONG_MIXED},
}


def classify_corrected_resume(arm_outcomes: Mapping[str, str]) -> dict[str, object]:
    """Classify the fixed four-arm test without a survival threshold."""
    exact_arms = set(arm_outcomes) == set(REQUIRED_ARMS)
    labels_valid = exact_arms and all(
        arm_outcomes[name] in EXPECTED_OUTCOMES[name] for name in REQUIRED_ARMS
    )
    baseline_valid = bool(
        labels_valid
        and arm_outcomes[BASELINE_SHORT] == SHORT_COMPLETE
        and arm_outcomes[BASELINE_LONG] == LONG_FAIL
    )
    if not baseline_valid:
        outcome = "invalid-execution"
    elif any(
        arm_outcomes[name] in {SHORT_MIXED, LONG_MIXED} for name in REQUIRED_ARMS
    ):
        outcome = "mixed-corrected-order-evidence"
    elif arm_outcomes[POST_SHORT] == SHORT_FAIL:
        outcome = "corrected-order-still-immediate-loss"
    elif arm_outcomes[POST_LONG] == LONG_COMPLETE:
        outcome = "corrected-order-preserves-short-long-complete"
    else:
        outcome = "corrected-order-preserves-short-long-incomplete"
    return {
        "outcome": outcome,
        "required_arms": list(REQUIRED_ARMS),
        "arm_outcomes": {name: arm_outcomes.get(name) for name in REQUIRED_ARMS},
        "exact_arms": exact_arms,
        "arm_outcomes_valid": labels_valid,
        "baseline_valid": baseline_valid,
    }
