"""Experimental runner that observes every seam of one native PPO update."""

from __future__ import annotations

import hashlib
import json
import math
import os
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.resume_initial_observation_normalization.runner import (
    InitialObservationNormalizedMotionOnPolicyRunner,
)

from .probe import (
    BatchEvent,
    checkpoint_steps,
    gradient_geometry,
    instrumented_mini_batch_generator,
    phase_bin_indices,
)

PROBE_DIRECTORY_ENV = "DIFFSIM_FIRST_UPDATE_PROBE_DIR"
EXPECTED_ENVIRONMENTS = 4096
EXPECTED_ROLLOUT_STEPS = 24
EXPECTED_EPOCHS = 5
EXPECTED_MINI_BATCHES = 4
EXPECTED_REFERENCE_STATES = 272


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(_sha256_tensor(value).encode())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, (str, int, float, bool, type(None))):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        digest.update(type(value).__qualname__.encode())
        digest.update(repr(value).encode())


def _state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tensor_stats(value: torch.Tensor) -> dict[str, float | int | None]:
    flat = value.detach().to(dtype=torch.float64).flatten()
    if flat.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> list[int]:
    values: set[int] = set()
    for state in optimizer.state.values():
        if "step" in state:
            step = state["step"]
            values.add(
                int(step.detach().cpu().item())
                if isinstance(step, torch.Tensor)
                else int(step)
            )
    return sorted(values)


def _parameter_group(name: str) -> str:
    if name.startswith("actor."):
        return "actor"
    if name.startswith("critic."):
        return "critic"
    if name in {"std", "log_std"}:
        return "std"
    return "other"


def _norms_from_tensors(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, float]:
    squared = {"actor": 0.0, "critic": 0.0, "std": 0.0, "other": 0.0, "all": 0.0}
    for name, tensor in named_tensors:
        value = float(
            torch.sum(tensor.detach().to(dtype=torch.float64).square()).item()
        )
        group = _parameter_group(name)
        squared[group] += value
        squared["all"] += value
    return {name: math.sqrt(max(value, 0.0)) for name, value in squared.items()}


def _peek_permutation(storage: Any, num_mini_batches: int) -> torch.Tensor:
    count = storage.num_envs * storage.num_transitions_per_env
    count = num_mini_batches * (count // num_mini_batches)
    device = torch.device(storage.device)
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
        try:
            return torch.randperm(count, requires_grad=False, device=device)
        finally:
            torch.cuda.set_rng_state(state, device)
    state = torch.random.get_rng_state()
    try:
        return torch.randperm(count, requires_grad=False, device=device)
    finally:
        torch.random.set_rng_state(state)


class _UpdateProbe:
    def __init__(
        self,
        runner: FirstUpdateProbeMotionOnPolicyRunner,
        *,
        output_dir: Path,
        phases: torch.Tensor,
        timeouts: torch.Tensor,
        reference_states: int,
        bin_count: int,
    ) -> None:
        self.runner = runner
        self.alg = runner.alg
        self.policy = runner.alg.policy
        self.storage = runner.alg.storage
        self.output_dir = output_dir
        self.phases = phases
        self.timeouts = timeouts
        self.reference_states = reference_states
        self.bin_count = bin_count
        self.expected_permutation = _peek_permutation(
            self.storage, self.alg.num_mini_batches
        )
        self.current_event: BatchEvent | None = None
        self.current_log_prob: torch.Tensor | None = None
        self.current_mu: torch.Tensor | None = None
        self.current_sigma: torch.Tensor | None = None
        self.current_entropy: torch.Tensor | None = None
        self.current_value: torch.Tensor | None = None
        self.current_gradient: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self.named_parameters = list(self.policy.named_parameters())
        self.baseline_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters
        }
        self.previous_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters
        }
        self.baseline_norms = _norms_from_tensors(self.named_parameters)
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _rng_digest(self) -> dict[str, Any]:
        return {
            "cpu": _sha256_tensor(torch.random.get_rng_state()),
            "cuda": [_sha256_tensor(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else [],
        }

    def _state_identity(self) -> dict[str, str]:
        return {
            "model": _state_digest(self.policy.state_dict()),
            "optimizer": _state_digest(self.alg.optimizer.state_dict()),
            "actor_normalizer": _state_digest(self.runner.obs_normalizer.state_dict()),
            "critic_normalizer": _state_digest(
                self.runner.privileged_obs_normalizer.state_dict()
            ),
        }

    def _actor_gradient(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        observations = self.storage.observations.flatten(0, 1)[indices]
        actions = self.storage.actions.flatten(0, 1)[indices]
        advantages = self.storage.advantages.flatten(0, 1)[indices].squeeze(-1)
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1)[indices].squeeze(-1)
        actor_parameters = [
            parameter
            for name, parameter in self.named_parameters
            if _parameter_group(name) in {"actor", "std"}
        ]
        self.policy.update_distribution(observations)
        new_log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(new_log_prob - old_log_prob)
        loss = -(advantages * ratio).mean()
        gradients = torch.autograd.grad(loss, actor_parameters, retain_graph=False)
        flat = torch.cat(
            [
                gradient.detach().to(device="cpu", dtype=torch.float32).flatten()
                for gradient in gradients
            ]
        )
        error = (new_log_prob.detach() - old_log_prob).abs()
        metadata = {
            "sample_count": int(indices.numel()),
            "surrogate_loss": float(loss.detach().cpu().item()),
            "old_log_probability_max_abs_error": float(error.max().cpu().item()),
            "old_log_probability_mean_abs_error": float(error.mean().cpu().item()),
        }
        self.policy.zero_grad(set_to_none=True)
        self.policy.distribution = None
        return flat, metadata

    def _frozen_gradient_analysis(self) -> dict[str, Any]:
        state_before = self._state_identity()
        rng_before = self._rng_digest()
        total_samples = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = total_samples // self.alg.num_mini_batches
        partition_gradients: list[torch.Tensor] = []
        partition_metadata: list[dict[str, Any]] = []
        for partition in range(self.alg.num_mini_batches):
            start = partition * mini_batch_size
            stop = (partition + 1) * mini_batch_size
            gradient, metadata = self._actor_gradient(
                self.expected_permutation[start:stop]
            )
            partition_gradients.append(gradient)
            partition_metadata.append({"partition": partition, **metadata})
            print(
                f"[FIRST-UPDATE-PROBE] frozen partition gradient {partition + 1}/{self.alg.num_mini_batches}",
                flush=True,
            )

        flat_phases = self.phases.flatten()
        bins = phase_bin_indices(
            flat_phases,
            reference_states=self.reference_states,
            bin_count=self.bin_count,
        )
        phase_gradients: list[torch.Tensor] = []
        phase_metadata: list[dict[str, Any]] = []
        phase_counts: list[int] = []
        for phase_bin in range(self.bin_count):
            indices = torch.nonzero(bins == phase_bin, as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                raise RuntimeError(
                    f"adaptive phase bin {phase_bin} has no rollout samples"
                )
            gradient, metadata = self._actor_gradient(indices)
            phase_gradients.append(gradient)
            phase_counts.append(int(indices.numel()))
            phase_metadata.append({"phase_bin": phase_bin, **metadata})
            print(
                f"[FIRST-UPDATE-PROBE] frozen phase gradient {phase_bin + 1}/{self.bin_count}",
                flush=True,
            )

        partition_geometry = gradient_geometry(
            labels=[f"partition_{index}" for index in range(len(partition_gradients))],
            gradients=partition_gradients,
        )
        partition_sum = partition_geometry.pop("_weighted_sum")
        phase_geometry = gradient_geometry(
            labels=[f"phase_bin_{index}" for index in range(len(phase_gradients))],
            gradients=phase_gradients,
            weights=phase_counts,
        )
        phase_sum = phase_geometry.pop("_weighted_sum")
        reconstruction_error = float(
            torch.linalg.vector_norm(partition_sum - phase_sum)
        )
        denominator = float(torch.linalg.vector_norm(partition_sum))
        state_after = self._state_identity()
        rng_after = self._rng_digest()
        if state_before != state_after:
            raise RuntimeError(
                "frozen gradient prepass changed model, optimizer, or normalizer state"
            )
        if rng_before != rng_after:
            raise RuntimeError("frozen gradient prepass changed CPU or CUDA RNG state")
        return {
            "definition": (
                "Actor-plus-std unclipped PPO surrogate gradients evaluated at one common pre-update "
                "policy. Four disjoint equal partitions use the exact native permutation; six phase "
                "gradients use MotionCommand's exact adaptive-bin formula."
            ),
            "state_identity_before": state_before,
            "state_identity_after": state_after,
            "rng_identity_before": rng_before,
            "rng_identity_after": rng_after,
            "native_permutation_sha256": _sha256_tensor(self.expected_permutation),
            "partition_metadata": partition_metadata,
            "phase_metadata": phase_metadata,
            "partition_geometry": partition_geometry,
            "phase_geometry": phase_geometry,
            "phase_weighted_sum_reconstruction_l2": reconstruction_error,
            "phase_weighted_sum_reconstruction_relative_l2": (
                reconstruction_error / denominator if denominator > 0.0 else None
            ),
        }

    def _rollout_summary(self) -> dict[str, Any]:
        flat_phases = self.phases.flatten()
        bins = phase_bin_indices(
            flat_phases,
            reference_states=self.reference_states,
            bin_count=self.bin_count,
        )
        advantages = self.storage.advantages.flatten(0, 1).squeeze(-1)
        returns = self.storage.returns.flatten(0, 1).squeeze(-1)
        rewards = self.storage.rewards.flatten(0, 1).squeeze(-1)
        values = self.storage.values.flatten(0, 1).squeeze(-1)
        dones = self.storage.dones.flatten(0, 1).squeeze(-1).bool()
        timeouts = self.timeouts.flatten().bool()
        per_bin = []
        for phase_bin in range(self.bin_count):
            mask = bins == phase_bin
            per_bin.append(
                {
                    "phase_bin": phase_bin,
                    "sample_count": int(mask.sum().item()),
                    "sample_fraction": float(mask.float().mean().item()),
                    "phase": _tensor_stats(flat_phases[mask]),
                    "advantage": _tensor_stats(advantages[mask]),
                    "advantage_positive_fraction": float(
                        (advantages[mask] > 0).float().mean().item()
                    ),
                    "return": _tensor_stats(returns[mask]),
                    "reward": _tensor_stats(rewards[mask]),
                    "old_value": _tensor_stats(values[mask]),
                    "done_count": int(dones[mask].sum().item()),
                    "timeout_count": int(timeouts[mask].sum().item()),
                    "hard_termination_count": int(
                        (dones[mask] & ~timeouts[mask]).sum().item()
                    ),
                }
            )
        initial_histogram = torch.bincount(
            self.phases[0].to(dtype=torch.long), minlength=self.reference_states
        )
        return {
            "samples": int(flat_phases.numel()),
            "rollout_steps": int(self.phases.shape[0]),
            "environments": int(self.phases.shape[1]),
            "reference_states": self.reference_states,
            "adaptive_bin_count": self.bin_count,
            "phase_min": int(flat_phases.min().item()),
            "phase_max": int(flat_phases.max().item()),
            "distinct_phases": int(torch.unique(flat_phases).numel()),
            "initial_distinct_phases": int((initial_histogram > 0).sum().item()),
            "initial_phase_histogram": initial_histogram.detach().cpu().tolist(),
            "done_count": int(dones.sum().item()),
            "timeout_count": int(timeouts.sum().item()),
            "hard_termination_count": int((dones & ~timeouts).sum().item()),
            "advantage": _tensor_stats(advantages),
            "return": _tensor_stats(returns),
            "reward": _tensor_stats(rewards),
            "old_value": _tensor_stats(values),
            "per_phase_bin": per_bin,
        }

    def _save_rollout_tensors(self) -> dict[str, Any]:
        path = self.output_dir / "actor_rollout_tensors.pt"
        payload = {
            "phases": self.phases.detach().cpu(),
            "timeouts": self.timeouts.detach().cpu(),
            "observations": self.storage.observations.detach().cpu(),
            "actions": self.storage.actions.detach().cpu(),
            "rewards": self.storage.rewards.detach().cpu(),
            "dones": self.storage.dones.detach().cpu(),
            "values": self.storage.values.detach().cpu(),
            "returns": self.storage.returns.detach().cpu(),
            "advantages": self.storage.advantages.detach().cpu(),
            "actions_log_prob": self.storage.actions_log_prob.detach().cpu(),
            "mu": self.storage.mu.detach().cpu(),
            "sigma": self.storage.sigma.detach().cpu(),
        }
        torch.save(payload, path)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    def _save_checkpoint(self, step: int) -> dict[str, Any]:
        path = self.checkpoint_dir / f"model_step_{step:02d}.pt"
        if path.exists():
            raise RuntimeError(f"refusing to overwrite probe checkpoint: {path}")
        self.runner.save(str(path))
        return {
            "step": step,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "optimizer_steps": _optimizer_steps(self.alg.optimizer),
        }

    def _on_batch(self, event: BatchEvent) -> None:
        self.current_event = event
        self.current_log_prob = None
        self.current_mu = None
        self.current_sigma = None
        self.current_entropy = None
        self.current_value = None
        self.current_gradient = None

    def _gradient_norms(self) -> dict[str, float]:
        return _norms_from_tensors(
            [
                (name, parameter.grad)
                for name, parameter in self.named_parameters
                if parameter.grad is not None
            ]
        )

    def _drift(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cumulative = []
        incremental = []
        for name, parameter in self.named_parameters:
            cumulative.append(
                (name, parameter.detach() - self.baseline_parameters[name])
            )
            incremental.append(
                (name, parameter.detach() - self.previous_parameters[name])
            )
        cumulative_norms = _norms_from_tensors(cumulative)
        incremental_norms = _norms_from_tensors(incremental)
        relative = {
            group: (
                cumulative_norms[group] / self.baseline_norms[group]
                if self.baseline_norms[group] > 0.0
                else None
            )
            for group in cumulative_norms
        }
        for name, parameter in self.named_parameters:
            self.previous_parameters[name].copy_(parameter.detach())
        return (
            {"l2": cumulative_norms, "relative_l2": relative},
            {"l2": incremental_norms},
        )

    def _step_metrics(self) -> dict[str, Any]:
        if self.current_event is None:
            raise RuntimeError("optimizer step has no current mini-batch event")
        if any(
            value is None
            for value in (
                self.current_log_prob,
                self.current_mu,
                self.current_sigma,
                self.current_entropy,
                self.current_value,
                self.current_gradient,
            )
        ):
            raise RuntimeError("optimizer step instrumentation is incomplete")
        event = self.current_event
        indices = event.indices
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1)[indices].squeeze(-1)
        old_mu = self.storage.mu.flatten(0, 1)[indices]
        old_sigma = self.storage.sigma.flatten(0, 1)[indices]
        target_values = self.storage.values.flatten(0, 1)[indices]
        advantages = self.storage.advantages.flatten(0, 1)[indices].squeeze(-1)
        returns = self.storage.returns.flatten(0, 1)[indices]
        phases = self.phases.flatten()[indices]
        log_ratio = self.current_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        value_clipped = target_values + (self.current_value - target_values).clamp(
            -self.alg.clip_param, self.alg.clip_param
        )
        value_loss = torch.max(
            (self.current_value - returns).pow(2),
            (value_clipped - returns).pow(2),
        ).mean()
        entropy = self.current_entropy.mean()
        analytic_kl = torch.sum(
            torch.log(self.current_sigma / old_sigma + 1.0e-5)
            + (old_sigma.square() + (old_mu - self.current_mu).square())
            / (2.0 * self.current_sigma.square())
            - 0.5,
            dim=-1,
        )
        total_loss = (
            surrogate_loss
            + self.alg.value_loss_coef * value_loss
            - self.alg.entropy_coef * entropy
        )
        batch_bins = phase_bin_indices(
            phases,
            reference_states=self.reference_states,
            bin_count=self.bin_count,
        )
        return {
            "optimizer_step": event.global_step,
            "epoch": event.epoch,
            "mini_batch": event.mini_batch,
            "sample_count": int(indices.numel()),
            "indices_sha256": _sha256_tensor(indices),
            "phase_bin_counts": torch.bincount(batch_bins, minlength=self.bin_count)
            .detach()
            .cpu()
            .tolist(),
            "phase": _tensor_stats(phases),
            "advantage": _tensor_stats(advantages),
            "return": _tensor_stats(returns),
            "ratio": _tensor_stats(ratio),
            "log_ratio": _tensor_stats(log_ratio),
            "approximate_kl": float(((ratio - 1.0) - log_ratio).mean().item()),
            "analytic_kl": _tensor_stats(analytic_kl),
            "clipped_fraction": float(
                ((ratio - 1.0).abs() > self.alg.clip_param).float().mean().item()
            ),
            "loss": {
                "surrogate": float(surrogate_loss.item()),
                "value": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "total": float(total_loss.item()),
            },
            "learning_rate": float(self.alg.optimizer.param_groups[0]["lr"]),
            "optimizer_state_steps_before": _optimizer_steps(self.alg.optimizer),
            "gradient": self.current_gradient,
        }

    def run(self, original_update: Any) -> dict[str, Any]:
        expected_steps = checkpoint_steps(
            num_epochs=self.alg.num_learning_epochs,
            num_mini_batches=self.alg.num_mini_batches,
        )
        if expected_steps != tuple(range(21)):
            raise RuntimeError(f"unexpected PPO update shape: {expected_steps}")

        rollout_summary = self._rollout_summary()
        rollout_tensors = self._save_rollout_tensors()
        checkpoint_records = [self._save_checkpoint(0)]
        frozen_gradients = self._frozen_gradient_analysis()

        original_generator = self.storage.mini_batch_generator
        original_log_prob = self.policy.get_actions_log_prob
        original_evaluate = self.policy.evaluate
        original_clip = torch.nn.utils.clip_grad_norm_
        original_optimizer_step = self.alg.optimizer.step

        def generator(storage: Any, num_mini_batches: int, num_epochs: int):
            if (
                num_mini_batches != EXPECTED_MINI_BATCHES
                or num_epochs != EXPECTED_EPOCHS
            ):
                raise RuntimeError("native PPO requested an unexpected batch schedule")
            yield from instrumented_mini_batch_generator(
                storage,
                num_mini_batches,
                num_epochs,
                self._on_batch,
                expected_indices=self.expected_permutation,
            )

        def get_actions_log_prob(actions: torch.Tensor) -> torch.Tensor:
            result = original_log_prob(actions)
            if self.current_event is not None:
                self.current_log_prob = result.detach()
                self.current_mu = self.policy.action_mean.detach()
                self.current_sigma = self.policy.action_std.detach()
                self.current_entropy = self.policy.entropy.detach()
            return result

        def evaluate(observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            result = original_evaluate(observations, **kwargs)
            if self.current_event is not None:
                self.current_value = result.detach()
            return result

        def clip_grad_norm_(
            parameters: Any, max_norm: float, norm_type: float = 2.0, **kwargs: Any
        ) -> torch.Tensor:
            materialized = list(parameters)
            pre = self._gradient_norms()
            returned = original_clip(
                materialized, max_norm, norm_type=norm_type, **kwargs
            )
            post = self._gradient_norms()
            self.current_gradient = {
                "pre_clip_l2": pre,
                "post_clip_l2": post,
                "native_returned_total_norm": float(returned.detach().cpu().item()),
                "max_grad_norm": float(max_norm),
                "clip_scale": (post["all"] / pre["all"] if pre["all"] > 0.0 else None),
            }
            return returned

        def optimizer_step(*args: Any, **kwargs: Any) -> Any:
            record = self._step_metrics()
            result = original_optimizer_step(*args, **kwargs)
            record["optimizer_state_steps_after"] = _optimizer_steps(self.alg.optimizer)
            cumulative, incremental = self._drift()
            record["parameter_drift_from_step0"] = cumulative
            record["parameter_update_from_previous_step"] = incremental
            checkpoint = self._save_checkpoint(record["optimizer_step"])
            record["checkpoint"] = checkpoint
            checkpoint_records.append(checkpoint)
            self.trace.append(record)
            _write_json(
                self.output_dir / "optimizer_trace.partial.json",
                {"complete": False, "optimizer_trace": self.trace},
            )
            print(
                "[FIRST-UPDATE-PROBE] "
                f"step={record['optimizer_step']:02d} "
                f"kl={record['analytic_kl']['mean']:.8g} "
                f"clip={record['clipped_fraction']:.6f} "
                f"actor_drift={cumulative['relative_l2']['actor']:.8g}",
                flush=True,
            )
            return result

        self.storage.mini_batch_generator = types.MethodType(generator, self.storage)
        self.policy.get_actions_log_prob = get_actions_log_prob
        self.policy.evaluate = evaluate
        torch.nn.utils.clip_grad_norm_ = clip_grad_norm_
        self.alg.optimizer.step = optimizer_step
        try:
            loss_dict = original_update()
        finally:
            self.storage.mini_batch_generator = original_generator
            self.policy.get_actions_log_prob = original_log_prob
            self.policy.evaluate = original_evaluate
            torch.nn.utils.clip_grad_norm_ = original_clip
            self.alg.optimizer.step = original_optimizer_step

        if len(self.trace) != 20 or [
            item["optimizer_step"] for item in self.trace
        ] != list(range(1, 21)):
            raise RuntimeError("did not observe exactly 20 ordered optimizer steps")
        result = {
            "schema_version": 1,
            "complete": True,
            "measurement_only": True,
            "rollout": rollout_summary,
            "rollout_tensors": rollout_tensors,
            "frozen_gradient_analysis": frozen_gradients,
            "optimizer_trace": self.trace,
            "checkpoints": checkpoint_records,
            "native_loss_dict": {
                name: float(value) for name, value in loss_dict.items()
            },
            "final_optimizer_steps": _optimizer_steps(self.alg.optimizer),
            "final_state_identity": self._state_identity(),
        }
        _write_json(self.output_dir / "probe_result.json", result)
        _write_json(
            self.output_dir / "optimizer_trace.partial.json",
            {"complete": True, "optimizer_trace": self.trace},
        )
        return loss_dict


class FirstUpdateProbeMotionOnPolicyRunner(
    InitialObservationNormalizedMotionOnPolicyRunner
):
    """Correct the resume ordering, then observe one otherwise-native update."""

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> Any:
        if num_learning_iterations != 1:
            raise RuntimeError(
                "first-update probe requires exactly one learning iteration"
            )
        if self.current_learning_iteration != 500:
            raise RuntimeError(
                "first-update probe requires the loaded iteration-500 source"
            )
        if (
            self.env.num_envs != EXPECTED_ENVIRONMENTS
            or self.num_steps_per_env != EXPECTED_ROLLOUT_STEPS
        ):
            raise RuntimeError("first-update probe requires 4096 environments and H=24")
        if (
            self.alg.num_learning_epochs != EXPECTED_EPOCHS
            or self.alg.num_mini_batches != EXPECTED_MINI_BATCHES
            or self.alg.policy.is_recurrent
            or self.alg.rnd is not None
            or self.alg.symmetry is not None
            or self.alg.normalize_advantage_per_mini_batch
        ):
            raise RuntimeError(
                "PPO structure differs from the registered feed-forward 5x4 update"
            )

        output_value = os.environ.get(PROBE_DIRECTORY_ENV)
        if not output_value:
            raise RuntimeError(f"{PROBE_DIRECTORY_ENV} is required")
        output_dir = Path(output_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "probe_result.json").exists():
            raise RuntimeError("refusing to overwrite a completed first-update probe")

        raw_env = self.env.unwrapped
        motion_command = raw_env.command_manager.get_term("motion")
        reference_states = int(motion_command.motion.time_step_total)
        bin_count = int(motion_command.bin_count)
        if reference_states != EXPECTED_REFERENCE_STATES or bin_count != 6:
            raise RuntimeError(
                f"expected 272 reference states and six bins, got {reference_states} and {bin_count}"
            )

        phases: list[torch.Tensor] = []
        timeouts: list[torch.Tensor] = []
        returns_ready = False
        original_env_step = self.env.step
        original_compute_returns = self.alg.compute_returns
        original_update = self.alg.update

        def observed_env_step(actions: torch.Tensor):
            phases.append(motion_command.time_steps.detach().clone())
            result = original_env_step(actions)
            infos = result[3]
            timeout = infos.get("time_outs")
            if timeout is None:
                timeout = torch.zeros_like(result[2], dtype=torch.bool)
            timeouts.append(timeout.detach().clone().reshape(-1))
            return result

        def observed_compute_returns(last_critic_obs: torch.Tensor) -> Any:
            nonlocal returns_ready
            result = original_compute_returns(last_critic_obs)
            returns_ready = True
            return result

        def observed_update() -> dict[str, Any]:
            if (
                not returns_ready
                or len(phases) != EXPECTED_ROLLOUT_STEPS
                or len(timeouts) != EXPECTED_ROLLOUT_STEPS
            ):
                raise RuntimeError("rollout capture is incomplete before PPO update")
            probe = _UpdateProbe(
                self,
                output_dir=output_dir,
                phases=torch.stack(phases),
                timeouts=torch.stack(timeouts),
                reference_states=reference_states,
                bin_count=bin_count,
            )
            return probe.run(original_update)

        self.env.step = observed_env_step
        self.alg.compute_returns = observed_compute_returns
        self.alg.update = observed_update
        try:
            result = super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.env.step = original_env_step
            self.alg.compute_returns = original_compute_returns
            self.alg.update = original_update

        result_path = output_dir / "probe_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["runner_completed"] = True
        payload["current_learning_iteration"] = self.current_learning_iteration
        payload["normalizer_counts"] = {
            "actor": int(self.obs_normalizer.count.detach().cpu().item()),
            "critic": int(self.privileged_obs_normalizer.count.detach().cpu().item()),
        }
        _write_json(result_path, payload)
        print(
            f"[FIRST-UPDATE-PROBE] complete result={result_path}",
            flush=True,
        )
        return result
