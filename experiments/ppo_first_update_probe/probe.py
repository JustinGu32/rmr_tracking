"""Semantics-preserving helpers for observing a native RSL-RL PPO update."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BatchEvent:
    """Identity of one yielded PPO batch."""

    epoch: int
    mini_batch: int
    global_step: int
    indices: torch.Tensor


def checkpoint_steps(*, num_epochs: int, num_mini_batches: int) -> tuple[int, ...]:
    """Return the no-update snapshot plus every native optimizer step."""
    if num_epochs < 1 or num_mini_batches < 1:
        raise ValueError("epoch and mini-batch counts must be positive")
    return tuple(range(num_epochs * num_mini_batches + 1))


def phase_bin_indices(
    phases: torch.Tensor, *, reference_states: int, bin_count: int
) -> torch.Tensor:
    """Apply MotionCommand's exact phase-to-adaptive-bin formula."""
    if reference_states < 1 or bin_count < 1:
        raise ValueError("reference_states and bin_count must be positive")
    if phases.numel() and bool(
        ((phases < 0) | (phases >= reference_states)).any().item()
    ):
        raise ValueError("phase lies outside the reference")
    return torch.clamp(
        (phases.to(dtype=torch.long) * bin_count) // reference_states,
        min=0,
        max=bin_count - 1,
    )


def instrumented_mini_batch_generator(
    storage: Any,
    num_mini_batches: int,
    num_epochs: int,
    on_batch: Callable[[BatchEvent], None],
    *,
    expected_indices: torch.Tensor | None = None,
) -> Iterator[tuple[Any, ...]]:
    """Mirror RSL-RL's feed-forward generator while exposing its indices.

    RSL-RL draws one permutation and reuses its partitions for every epoch.  The
    tensor selection and yield order below intentionally match the installed
    implementation exactly.
    """
    if storage.training_type != "rl":
        raise ValueError(
            "This function is only available for reinforcement learning training."
        )
    batch_size = storage.num_envs * storage.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
        num_mini_batches * mini_batch_size,
        requires_grad=False,
        device=storage.device,
    )
    if expected_indices is not None and not torch.equal(
        indices, expected_indices.to(indices.device)
    ):
        raise RuntimeError("native mini-batch permutation drift")

    observations = storage.observations.flatten(0, 1)
    privileged_observations = (
        storage.privileged_observations.flatten(0, 1)
        if storage.privileged_observations is not None
        else observations
    )
    actions = storage.actions.flatten(0, 1)
    values = storage.values.flatten(0, 1)
    returns = storage.returns.flatten(0, 1)
    old_actions_log_prob = storage.actions_log_prob.flatten(0, 1)
    advantages = storage.advantages.flatten(0, 1)
    old_mu = storage.mu.flatten(0, 1)
    old_sigma = storage.sigma.flatten(0, 1)
    rnd_state = (
        storage.rnd_state.flatten(0, 1) if storage.rnd_state_shape is not None else None
    )

    global_step = 0
    for epoch in range(num_epochs):
        for mini_batch in range(num_mini_batches):
            start = mini_batch * mini_batch_size
            end = (mini_batch + 1) * mini_batch_size
            batch_idx = indices[start:end]
            global_step += 1
            on_batch(
                BatchEvent(epoch, mini_batch, global_step, batch_idx.detach().clone())
            )
            rnd_state_batch = rnd_state[batch_idx] if rnd_state is not None else None
            yield (
                observations[batch_idx],
                privileged_observations[batch_idx],
                actions[batch_idx],
                values[batch_idx],
                advantages[batch_idx],
                returns[batch_idx],
                old_actions_log_prob[batch_idx],
                old_mu[batch_idx],
                old_sigma[batch_idx],
                (None, None),
                None,
                rnd_state_batch,
            )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(
        torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    )
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def gradient_geometry(
    *,
    labels: Sequence[str],
    gradients: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> dict[str, object]:
    """Summarize finite gradient vectors, including variance and cancellation."""
    if not labels or len(labels) != len(gradients):
        raise ValueError("labels and gradients must have the same nonzero length")
    flat = [
        gradient.detach().to(device="cpu", dtype=torch.float64).flatten()
        for gradient in gradients
    ]
    width = flat[0].numel()
    if any(gradient.numel() != width for gradient in flat):
        raise ValueError("gradient widths differ")
    if any(not bool(torch.isfinite(gradient).all()) for gradient in flat):
        raise ValueError("nonfinite gradient")
    if weights is None:
        normalized_weights = [1.0 / len(flat)] * len(flat)
    else:
        if len(weights) != len(flat) or any(weight < 0.0 for weight in weights):
            raise ValueError("invalid gradient weights")
        weight_sum = math.fsum(float(weight) for weight in weights)
        if weight_sum <= 0.0:
            raise ValueError("gradient weights must have positive sum")
        normalized_weights = [float(weight) / weight_sum for weight in weights]

    weighted_sum = torch.zeros_like(flat[0])
    for weight, gradient in zip(normalized_weights, flat, strict=True):
        weighted_sum.add_(gradient, alpha=weight)
    norms = [float(torch.linalg.vector_norm(gradient)) for gradient in flat]
    weighted_sum_norm = float(torch.linalg.vector_norm(weighted_sum))
    weighted_component_norm_sum = math.fsum(
        weight * norm for weight, norm in zip(normalized_weights, norms, strict=True)
    )
    variance = math.fsum(
        weight * float(torch.dot(gradient - weighted_sum, gradient - weighted_sum))
        for weight, gradient in zip(normalized_weights, flat, strict=True)
    )
    pairwise = []
    for left_index in range(len(flat)):
        for right_index in range(left_index + 1, len(flat)):
            pairwise.append(
                {
                    "left": str(labels[left_index]),
                    "right": str(labels[right_index]),
                    "cosine": _cosine(flat[left_index], flat[right_index]),
                }
            )

    return {
        "labels": [str(label) for label in labels],
        "weights": normalized_weights,
        "dimensions": width,
        "norms": {str(label): norm for label, norm in zip(labels, norms, strict=True)},
        "weighted_sum_norm": weighted_sum_norm,
        "weighted_component_norm_sum": weighted_component_norm_sum,
        "weighted_cancellation_ratio": (
            weighted_sum_norm / weighted_component_norm_sum
            if weighted_component_norm_sum > 0.0
            else None
        ),
        "mean_squared_distance_from_weighted_sum": variance,
        "gradient_standard_deviation": math.sqrt(max(variance, 0.0)),
        "signal_to_gradient_std": (
            weighted_sum_norm / math.sqrt(variance) if variance > 0.0 else None
        ),
        "cosine_to_weighted_sum": {
            str(label): _cosine(gradient, weighted_sum)
            for label, gradient in zip(labels, flat, strict=True)
        },
        "pairwise_cosine": pairwise,
        "_weighted_sum": weighted_sum,
    }
