"""Single-variable resume-scheduler intervention for the PPO continuity probe."""

from __future__ import annotations

import math
from typing import Any

EXPECTED_FRESH_SCHEDULER_LR = 1.0e-3
EXPECTED_RESTORED_OPTIMIZER_LR = 2.25e-5
EXPECTED_FIRST_APPLIED_LR = 3.375e-5


def synchronize_resume_scheduler(algorithm: Any) -> dict[str, object]:
    """Set only PPO's unsaved scheduler scalar to the restored optimizer rate."""
    scheduler_before = float(algorithm.learning_rate)
    group_rates_before = [
        float(group["lr"]) for group in algorithm.optimizer.param_groups
    ]
    if not math.isclose(
        scheduler_before,
        EXPECTED_FRESH_SCHEDULER_LR,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(
            f"fresh scheduler drift: {scheduler_before} != {EXPECTED_FRESH_SCHEDULER_LR}"
        )
    if not group_rates_before or not all(
        math.isclose(
            rate,
            EXPECTED_RESTORED_OPTIMIZER_LR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for rate in group_rates_before
    ):
        raise RuntimeError(
            "restored optimizer learning rate drift: "
            f"{group_rates_before} != {EXPECTED_RESTORED_OPTIMIZER_LR}"
        )

    state_entries_before = len(algorithm.optimizer.state)
    algorithm.learning_rate = EXPECTED_RESTORED_OPTIMIZER_LR
    group_rates_after = [
        float(group["lr"]) for group in algorithm.optimizer.param_groups
    ]
    if group_rates_after != group_rates_before:
        raise RuntimeError("scheduler synchronization changed optimizer groups")
    if len(algorithm.optimizer.state) != state_entries_before:
        raise RuntimeError("scheduler synchronization changed Adam state")

    return {
        "causal_change": "PPO.learning_rate only",
        "scheduler_learning_rate_before": scheduler_before,
        "scheduler_learning_rate_after": float(algorithm.learning_rate),
        "optimizer_group_learning_rates_before": group_rates_before,
        "optimizer_group_learning_rates_after": group_rates_after,
        "optimizer_state_entries_before": state_entries_before,
        "optimizer_state_entries_after": len(algorithm.optimizer.state),
    }
