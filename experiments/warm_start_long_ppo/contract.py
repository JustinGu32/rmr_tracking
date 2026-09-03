"""Pure classification contract for the short-to-long PPO warm-start audit."""

from __future__ import annotations

from collections.abc import Mapping

BASELINE_SHORT = "baseline_short"
BASELINE_LONG = "baseline_long"
POST_1_SHORT = "post_update_1_short"
POST_1_LONG = "post_update_1_long"
POST_500_SHORT = "post_update_500_short"
POST_500_LONG = "post_update_500_long"

BASELINE_ARMS = (BASELINE_SHORT, BASELINE_LONG)
POST_TRAINING_ARMS = (
    POST_1_SHORT,
    POST_1_LONG,
    POST_500_SHORT,
    POST_500_LONG,
)
REQUIRED_ARMS = BASELINE_ARMS + POST_TRAINING_ARMS

SHORT_COMPLETE = "short-source-completes"
SHORT_FAIL = "short-source-fails"
SHORT_MIXED = "mixed-short-source-competence"
LONG_COMPLETE = "source-completes-exact-long"
LONG_FAIL = "source-fails-exact-long"
LONG_MIXED = "mixed-source-competence"

SHORT_OUTCOMES = {SHORT_COMPLETE, SHORT_FAIL, SHORT_MIXED}
LONG_OUTCOMES = {LONG_COMPLETE, LONG_FAIL, LONG_MIXED}
EXPECTED_OUTCOMES = {
    BASELINE_SHORT: SHORT_OUTCOMES,
    BASELINE_LONG: LONG_OUTCOMES,
    POST_1_SHORT: SHORT_OUTCOMES,
    POST_1_LONG: LONG_OUTCOMES,
    POST_500_SHORT: SHORT_OUTCOMES,
    POST_500_LONG: LONG_OUTCOMES,
}


def classify_warm_start(
    arm_outcomes: Mapping[str, str], *, training_executed: bool
) -> dict[str, object]:
    """Classify retention and acquisition without a tuned survival threshold."""
    supplied_arms = set(arm_outcomes)
    expected_arms = set(REQUIRED_ARMS if training_executed else BASELINE_ARMS)
    exact_arms = supplied_arms == expected_arms
    labels_valid = exact_arms and all(
        arm_outcomes[name] in EXPECTED_OUTCOMES[name] for name in expected_arms
    )
    baseline_valid = bool(
        labels_valid and arm_outcomes.get(BASELINE_SHORT) == SHORT_COMPLETE
    )

    if not baseline_valid:
        outcome = "invalid-execution"
    elif not training_executed:
        baseline_long = arm_outcomes[BASELINE_LONG]
        if baseline_long == LONG_COMPLETE:
            outcome = "baseline-already-long-complete"
        elif baseline_long == LONG_MIXED:
            outcome = "mixed-warm-start-evidence"
        else:
            outcome = "invalid-execution"
    elif any(arm_outcomes[name] in {SHORT_MIXED, LONG_MIXED} for name in REQUIRED_ARMS):
        outcome = "mixed-warm-start-evidence"
    elif arm_outcomes[BASELINE_LONG] != LONG_FAIL:
        # A complete baseline should have stopped before optimization.
        outcome = "invalid-execution"
    elif arm_outcomes[POST_1_SHORT] == SHORT_FAIL:
        outcome = "immediate-short-retention-loss"
    elif arm_outcomes[POST_500_SHORT] == SHORT_FAIL:
        outcome = "delayed-short-retention-loss"
    elif (
        arm_outcomes[POST_1_LONG] == LONG_COMPLETE
        and arm_outcomes[POST_500_LONG] == LONG_FAIL
    ):
        outcome = "long-competence-lost"
    elif arm_outcomes[POST_500_LONG] == LONG_COMPLETE:
        outcome = "retained-short-long-complete"
    else:
        outcome = "retained-short-long-incomplete"

    return {
        "outcome": outcome,
        "training_executed": training_executed,
        "required_arms": list(REQUIRED_ARMS if training_executed else BASELINE_ARMS),
        "arm_outcomes": {
            name: arm_outcomes.get(name)
            for name in (REQUIRED_ARMS if training_executed else BASELINE_ARMS)
        },
        "exact_arms": exact_arms,
        "arm_outcomes_valid": labels_valid,
        "baseline_valid": baseline_valid,
    }
