"""Pure contract helpers for the exact-long Isaac source evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

OUTCOME_LABELS = {
    "exact-long": {
        "complete": "source-completes-exact-long",
        "mixed": "mixed-source-competence",
        "fail": "source-fails-exact-long",
    },
    "short-control": {
        "complete": "short-source-completes",
        "mixed": "mixed-short-source-competence",
        "fail": "short-source-fails",
    },
}


def _zero_ranges(
    ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    return {name: (0.0, 0.0) for name in ranges}


def _disable_config_terms(config: Any) -> list[str]:
    """Disable manager terms while retaining the config object Isaac callbacks need."""
    if config is None:
        return []
    disabled: list[str] = []
    for name, value in vars(config).items():
        if name.startswith("_") or value is None:
            continue
        setattr(config, name, None)
        disabled.append(name)
    return disabled


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
    event_cfg = env_cfg.events
    curriculum_cfg = env_cfg.curriculum
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
    disabled_event_terms = _disable_config_terms(event_cfg)
    disabled_curriculum_terms = _disable_config_terms(curriculum_cfg)

    return {
        "num_envs": num_envs,
        "motion_file": motion_file,
        "start_phase": 0,
        "motion_reset_noise_zero": True,
        "observation_corruption_disabled": True,
        "all_events_disabled": True,
        "curriculum_disabled": True,
        "disabled_event_terms": disabled_event_terms,
        "disabled_curriculum_terms": disabled_curriculum_terms,
        "manager_config_objects_preserved": (
            env_cfg.events is event_cfg and env_cfg.curriculum is curriculum_cfg
        ),
        "hard_terminations_preserved": env_cfg.terminations is termination_cfg,
    }


def reset_episode_in_inference_mode(
    env: Any,
    motion_command: Any,
    *,
    seed: int,
    refresh_reference: Any,
) -> Any:
    """Reset tensors in the same inference context used by source rollout steps."""
    import torch

    with torch.inference_mode():
        env.seed(seed)
        obs, _ = env.reset()
        refresh_reference(motion_command)
    return obs


def apply_actor_normalizer_state(
    runner_or_policy: Any,
    state: dict[str, Any],
) -> dict[str, object]:
    """Replace only the actor observation normalizer used for inference."""
    if hasattr(runner_or_policy, "obs_normalizer"):
        normalizer = runner_or_policy.obs_normalizer
    elif hasattr(runner_or_policy, "obs_normalizers"):
        normalizer = runner_or_policy.obs_normalizers["actor"]
    else:
        normalizer = getattr(runner_or_policy, "actor_obs_normalizer", None)
    if normalizer is None:
        raise ValueError("loaded runner or policy has no actor observation normalizer")
    normalizer.load_state_dict(state, strict=True)
    return {
        "normalizer_class": type(normalizer).__name__,
        "state_keys": sorted(state),
    }


def classify_episodes(
    episodes: Sequence[dict[str, object]],
    *,
    reference_states: int,
    outcome_label_set: str = "exact-long",
) -> dict[str, object]:
    """Classify repeated phase-zero episodes without tuning a survival threshold."""
    if outcome_label_set not in OUTCOME_LABELS:
        raise ValueError(f"unknown outcome label set: {outcome_label_set}")
    labels = OUTCOME_LABELS[outcome_label_set]
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
        outcome = labels["complete"]
    elif any(completion):
        outcome = labels["mixed"]
    else:
        outcome = labels["fail"]

    return {
        "outcome": outcome,
        "outcome_label_set": outcome_label_set,
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
