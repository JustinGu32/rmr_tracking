"""Pure contract helpers for the exact-long Isaac source evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _zero_ranges(
    ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {name: (0.0, 0.0) for name in ranges}


def apply_nominal_phase_zero_contract(
    env_cfg: Any,
    *,
    motion_file: str,
    num_envs: int,
) -> dict[str, object]:
    """Make a training-task config deterministic without weakening termination gates."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not motion_file:
        raise ValueError("motion_file must be nonempty")

    termination_cfg = env_cfg.terminations
    motion_cfg = env_cfg.commands.motion
    env_cfg.scene.num_envs = num_envs
    motion_cfg.motion_file = motion_file
    motion_cfg.min_sample_idx = 0
    motion_cfg.max_sample_idx = 0
    motion_cfg.pose_range = _zero_ranges(motion_cfg.pose_range)
    motion_cfg.velocity_range = _zero_ranges(motion_cfg.velocity_range)
    motion_cfg.joint_position_range = (0.0, 0.0)
    if hasattr(motion_cfg, "debug_vis"):
        motion_cfg.debug_vis = False

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events = None
    env_cfg.curriculum = None

    return {
        "num_envs": num_envs,
        "motion_file": motion_file,
        "start_phase": 0,
        "motion_reset_noise_zero": True,
        "observation_corruption_disabled": True,
        "all_events_disabled": True,
        "curriculum_disabled": True,
        "hard_terminations_preserved": env_cfg.terminations is termination_cfg,
    }


def classify_episodes(
    episodes: Sequence[dict[str, object]],
    *,
    reference_states: int,
) -> dict[str, object]:
    """Classify repeated phase-zero episodes without tuning a survival threshold."""
    expected_transitions = reference_states - 1
    # Isaac Lab checks ``my_time_out`` before MotionCommand advances its phase.
    # Starting at phase zero therefore takes one policy action at every reference
    # state, including the terminal state, before the native timeout is emitted.
    expected_policy_steps = reference_states
    valid = bool(episodes) and reference_states >= 2
    completion: list[bool] = []
    survival: list[int] = []

    for episode in episodes:
        try:
            steps = int(episode["steps"])
            final_phase = int(episode["final_reference_phase"])
            terminated = bool(episode["terminated"])
            timed_out = bool(episode["timed_out"])
            finite = bool(episode["all_numeric_finite"])
        except (KeyError, TypeError, ValueError):
            valid = False
            continue

        episode_valid = (
            1 <= steps <= expected_policy_steps
            and final_phase == steps - 1
            and (terminated or timed_out)
            and finite
            and (not timed_out or final_phase == expected_transitions)
        )
        valid = valid and episode_valid
        survival.append(steps)
        completion.append(
            episode_valid
            and timed_out
            and not terminated
            and steps == expected_policy_steps
            and final_phase == expected_transitions
        )

    if not valid:
        outcome = "invalid-execution"
    elif all(completion):
        outcome = "source-completes-exact-long"
    elif any(completion):
        outcome = "mixed-source-competence"
    else:
        outcome = "source-fails-exact-long"

    return {
        "outcome": outcome,
        "contract_valid": valid,
        "reference_states": reference_states,
        "expected_transition_count": expected_transitions,
        "expected_source_policy_steps": expected_policy_steps,
        "episode_count": len(episodes),
        "survival_steps": survival,
        "completed_reference": completion,
        "all_episodes_complete": bool(completion) and all(completion),
        "any_episode_complete": any(completion),
    }
