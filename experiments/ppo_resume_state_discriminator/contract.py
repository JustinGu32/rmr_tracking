"""Pure categorical contract for the first-step resume-state factorial."""

from __future__ import annotations

from collections.abc import Mapping

try:
    from .design import ARM_NAMES, NATIVE_ARM
except ImportError:  # Direct script/test execution from this experiment directory.
    from design import ARM_NAMES, NATIVE_ARM


def classify_factorial(
    *,
    step0_short_complete: bool,
    short_complete: Mapping[str, bool],
    long_complete: Mapping[str, bool],
) -> dict[str, object]:
    """Classify strict package outcomes without a tuned survival threshold."""
    observed_short = tuple(sorted(short_complete))
    observed_long = tuple(sorted(long_complete))
    expected = tuple(sorted(ARM_NAMES))
    if observed_short != expected or observed_long != expected:
        return {
            "outcome": "invalid-execution",
            "expected_arms": list(expected),
            "observed_short_arms": list(observed_short),
            "observed_long_arms": list(observed_long),
        }
    if not step0_short_complete:
        outcome = "no-update-control-fails"
    elif bool(short_complete[NATIVE_ARM]):
        outcome = "native-loss-not-reproduced"
    elif any(bool(value) for value in long_complete.values()):
        outcome = "unexpected-long-completion"
    else:
        synchronized = bool(short_complete["restored_adam__synced_scheduler"])
        reset = bool(short_complete["reset_adam__fresh_scheduler"])
        combined = bool(short_complete["reset_adam__synced_scheduler"])
        if (synchronized or reset) and not combined:
            outcome = "nonadditive-interaction"
        elif synchronized and reset:
            outcome = "both-single-factor-interventions-preserve"
        elif synchronized:
            outcome = "scheduler-synchronization-preserves"
        elif reset:
            outcome = "adam-reset-preserves"
        elif combined:
            outcome = "combined-intervention-required"
        else:
            outcome = "neither-intervention-preserves"
    return {
        "outcome": outcome,
        "step0_short_complete": bool(step0_short_complete),
        "short_complete": {name: bool(short_complete[name]) for name in ARM_NAMES},
        "long_complete": {name: bool(long_complete[name]) for name in ARM_NAMES},
        "native_arm": NATIVE_ARM,
    }
