"""Pure classification contract for the source policy-component crossover."""

from __future__ import annotations

from collections.abc import Mapping

SHORT_SHORT = "short_actor__short_normalizer"
SHORT_LONG = "short_actor__long_normalizer"
LONG_SHORT = "long_actor__short_normalizer"
LONG_LONG = "long_actor__long_normalizer"
REQUIRED_ARMS = (SHORT_SHORT, SHORT_LONG, LONG_SHORT, LONG_LONG)
COMPLETE = "short-source-completes"
FAIL = "short-source-fails"
MIXED = "mixed-short-source-competence"
VALID_ARM_OUTCOMES = {COMPLETE, FAIL, MIXED}


def classify_crossover(arm_outcomes: Mapping[str, str]) -> dict[str, object]:
    """Classify a fixed 2x2 actor/normalizer crossover without tuned thresholds."""
    exact_arms = set(arm_outcomes) == set(REQUIRED_ARMS)
    outcomes_valid = exact_arms and all(
        arm_outcomes[name] in VALID_ARM_OUTCOMES for name in REQUIRED_ARMS
    )
    controls_valid = bool(
        outcomes_valid
        and arm_outcomes[SHORT_SHORT] == COMPLETE
        and arm_outcomes[LONG_LONG] == FAIL
    )

    if not controls_valid:
        outcome = "invalid-execution"
    else:
        short_actor_long_normalizer = arm_outcomes[SHORT_LONG]
        long_actor_short_normalizer = arm_outcomes[LONG_SHORT]
        if MIXED in (
            short_actor_long_normalizer,
            long_actor_short_normalizer,
        ):
            outcome = "mixed-component-crossover"
        elif (
            short_actor_long_normalizer == COMPLETE
            and long_actor_short_normalizer == FAIL
        ):
            outcome = "actor-component-dominant"
        elif (
            short_actor_long_normalizer == FAIL
            and long_actor_short_normalizer == COMPLETE
        ):
            outcome = "normalizer-component-dominant"
        elif (
            short_actor_long_normalizer == COMPLETE
            and long_actor_short_normalizer == COMPLETE
        ):
            outcome = "either-short-component-rescues"
        else:
            outcome = "both-short-components-required"

    return {
        "outcome": outcome,
        "required_arms": list(REQUIRED_ARMS),
        "arm_outcomes": {
            name: arm_outcomes.get(name) for name in REQUIRED_ARMS
        },
        "exact_arms": exact_arms,
        "arm_outcomes_valid": outcomes_valid,
        "controls_valid": controls_valid,
    }
