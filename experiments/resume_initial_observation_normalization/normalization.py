"""One-shot observation normalization helpers for a resumed RSL-RL rollout."""

from __future__ import annotations

from typing import Any

import torch


def normalize_without_update(
    normalizer: torch.nn.Module, value: torch.Tensor
) -> torch.Tensor:
    """Apply a normalizer using frozen statistics and restore its prior mode."""
    was_training = normalizer.training
    normalizer.eval()
    try:
        with torch.no_grad():
            result = normalizer(value)
    finally:
        normalizer.train(was_training)
    return result


def normalize_initial_observations(
    actor_observation: torch.Tensor,
    extras: dict[str, Any],
    *,
    actor_normalizer: torch.nn.Module,
    privileged_normalizer: torch.nn.Module,
    privileged_observation_type: str | None,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Normalize the initial actor/critic inputs without incrementing counts."""
    actor = normalize_without_update(actor_normalizer, actor_observation.to(device))
    copied_extras = dict(extras)
    observations = dict(extras.get("observations", {}))
    privileged_applied = False
    privileged_features = None
    if privileged_observation_type is not None and privileged_observation_type in observations:
        privileged = observations[privileged_observation_type].to(device)
        observations[privileged_observation_type] = normalize_without_update(
            privileged_normalizer, privileged
        )
        privileged_applied = True
        privileged_features = int(privileged.shape[-1])
    copied_extras["observations"] = observations
    metadata = {
        "actor_applied": True,
        "privileged_applied": privileged_applied,
        "actor_batch": int(actor.shape[0]),
        "actor_features": int(actor.shape[-1]),
        "privileged_features": privileged_features,
    }
    return actor, copied_extras, metadata
