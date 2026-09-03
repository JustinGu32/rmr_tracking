"""Pure outcome classification for the fixed first-PPO-update probe."""

from __future__ import annotations

from collections.abc import Mapping

EXPECTED_STEPS = tuple(range(21))


def classify_checkpoint_sweep(
    *,
    short_complete: Mapping[int, bool],
    step0_long_complete: bool,
    step20_long_complete: bool,
) -> dict[str, object]:
    """Classify the categorical checkpoint sweep without a survival threshold."""
    observed_steps = tuple(sorted(short_complete))
    if observed_steps != EXPECTED_STEPS:
        return {
            "outcome": "invalid-execution",
            "expected_steps": list(EXPECTED_STEPS),
            "observed_steps": list(observed_steps),
            "first_loss_step": None,
            "last_complete_step": None,
            "recovery_steps": [],
        }

    complete_steps = [step for step in EXPECTED_STEPS if bool(short_complete[step])]
    failed_steps = [step for step in EXPECTED_STEPS if not bool(short_complete[step])]
    first_loss = min(failed_steps) if failed_steps else None
    last_complete = max(complete_steps) if complete_steps else None
    recovery_steps = (
        [
            step
            for step in complete_steps
            if first_loss is not None and step > first_loss
        ]
        if first_loss is not None
        else []
    )

    if not bool(short_complete[0]):
        outcome = "no-optimizer-control-fails"
    elif step0_long_complete or step20_long_complete:
        outcome = "long-control-drift"
    elif bool(short_complete[20]):
        outcome = "final-loss-not-reproduced"
    elif recovery_steps:
        outcome = "optimizer-step-localized-nonmonotonic-loss"
    else:
        outcome = "optimizer-step-localized-monotonic-loss"

    return {
        "outcome": outcome,
        "expected_steps": list(EXPECTED_STEPS),
        "observed_steps": list(observed_steps),
        "complete_steps": complete_steps,
        "failed_steps": failed_steps,
        "first_loss_step": first_loss,
        "last_complete_step": last_complete,
        "recovery_steps": recovery_steps,
        "step0_long_complete": bool(step0_long_complete),
        "step20_long_complete": bool(step20_long_complete),
    }
