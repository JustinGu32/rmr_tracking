"""Exact two-factor interventions for the first resumed PPO Adam step."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArmSpec:
    """One branch of the resume-state factorial."""

    name: str
    synchronize_scheduler: bool
    reset_adam: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# The native arm is deliberately last: the outer runner retains its state as the
# canonical one-step output after all counterfactual branches have been measured.
ARM_SPECS = (
    ArmSpec(
        name="restored_adam__synced_scheduler",
        synchronize_scheduler=True,
        reset_adam=False,
    ),
    ArmSpec(
        name="reset_adam__fresh_scheduler",
        synchronize_scheduler=False,
        reset_adam=True,
    ),
    ArmSpec(
        name="reset_adam__synced_scheduler",
        synchronize_scheduler=True,
        reset_adam=True,
    ),
    ArmSpec(
        name="restored_adam__fresh_scheduler",
        synchronize_scheduler=False,
        reset_adam=False,
    ),
)

ARM_NAMES = tuple(spec.name for spec in ARM_SPECS)
NATIVE_ARM = "restored_adam__fresh_scheduler"


def prepare_arm(
    algorithm: Any,
    spec: ArmSpec,
    *,
    restored_learning_rate: float,
    fresh_scheduler_learning_rate: float,
) -> dict[str, object]:
    """Apply only the registered scheduler and Adam-state interventions.

    The optimizer parameter-group rate is intentionally left untouched here.
    Installed RSL-RL's adaptive-KL block overwrites it from
    ``algorithm.learning_rate`` immediately before the Adam step. This preserves
    the exact native ordering while changing only the selected resume state.
    """
    if restored_learning_rate <= 0.0 or fresh_scheduler_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    groups = algorithm.optimizer.param_groups
    if not groups:
        raise ValueError("optimizer has no parameter groups")
    group_rates = [float(group["lr"]) for group in groups]
    if not all(
        math.isclose(
            rate,
            float(restored_learning_rate),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for rate in group_rates
    ):
        raise RuntimeError(
            f"optimizer was not restored at the expected rate: {sorted(group_rates)}"
        )

    state_entries_before = len(algorithm.optimizer.state)
    if spec.reset_adam:
        algorithm.optimizer.state.clear()
    algorithm.learning_rate = float(
        restored_learning_rate
        if spec.synchronize_scheduler
        else fresh_scheduler_learning_rate
    )
    return {
        "arm": spec.to_dict(),
        "restored_optimizer_learning_rate": float(restored_learning_rate),
        "fresh_scheduler_learning_rate": float(fresh_scheduler_learning_rate),
        "scheduler_learning_rate_after_intervention": float(algorithm.learning_rate),
        "optimizer_group_learning_rates_after_intervention": [
            float(group["lr"]) for group in groups
        ],
        "optimizer_state_entries_before_intervention": state_entries_before,
        "optimizer_state_entries_after_intervention": len(algorithm.optimizer.state),
    }
