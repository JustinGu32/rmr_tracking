"""Pure categorical contract for synchronized resumed-PPO continuity."""

from __future__ import annotations

from collections.abc import Mapping

CHECKPOINT_STEPS = (0, 1, 2, 4, 8, 12, 16, 20)


def classify_continuity(
    *,
    short_complete: Mapping[int, bool],
    step0_long_complete: bool,
    step20_long_complete: bool,
) -> dict[str, object]:
    """Classify sampled within-update competence without a survival threshold."""
    observed_steps = tuple(sorted(short_complete))
    if observed_steps != CHECKPOINT_STEPS:
        return {
            "outcome": "invalid-execution",
            "expected_steps": list(CHECKPOINT_STEPS),
            "observed_steps": list(observed_steps),
            "first_observed_loss_step": None,
            "recovery_steps": [],
        }

    complete_steps = [step for step in CHECKPOINT_STEPS if bool(short_complete[step])]
    failed_steps = [step for step in CHECKPOINT_STEPS if not bool(short_complete[step])]
    first_loss = min(failed_steps) if failed_steps else None
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
    elif step0_long_complete:
        outcome = "unexpected-step-zero-long-completion"
    elif step20_long_complete:
        outcome = "synchronized-resume-acquires-exact-long"
    elif not failed_steps:
        outcome = "synchronized-resume-preserves-sampled-full-update"
    elif recovery_steps:
        outcome = "synchronized-resume-sampled-nonmonotonic-loss"
    else:
        outcome = "synchronized-resume-delays-loss"

    return {
        "outcome": outcome,
        "expected_steps": list(CHECKPOINT_STEPS),
        "observed_steps": list(observed_steps),
        "complete_steps": complete_steps,
        "failed_steps": failed_steps,
        "first_observed_loss_step": first_loss,
        "last_observed_complete_step": max(complete_steps) if complete_steps else None,
        "recovery_steps": recovery_steps,
        "step0_long_complete": bool(step0_long_complete),
        "step20_long_complete": bool(step20_long_complete),
    }
