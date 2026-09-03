"""Branch the exact first resumed PPO minibatch across optimizer state factors."""

from __future__ import annotations

import copy
import json
import math
import os
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.ppo_first_update_probe.probe import (
    BatchEvent,
    instrumented_mini_batch_generator,
    phase_bin_indices,
)
from experiments.ppo_first_update_probe.runner import (
    _norms_from_tensors,
    _optimizer_steps,
    _peek_permutation,
    _sha256_file,
    _sha256_tensor,
    _state_digest,
    _tensor_stats,
    _write_json,
)
from experiments.resume_initial_observation_normalization.runner import (
    InitialObservationNormalizedMotionOnPolicyRunner,
)

from .design import ARM_NAMES, ARM_SPECS, NATIVE_ARM, ArmSpec, prepare_arm

OUTPUT_DIRECTORY_ENV = "DIFFSIM_RESUME_STATE_DISCRIMINATOR_DIR"
EXPECTED_ENVIRONMENTS = 4096
EXPECTED_ROLLOUT_STEPS = 24
EXPECTED_EPOCHS = 5
EXPECTED_MINI_BATCHES = 4
EXPECTED_REFERENCE_STATES = 272
EXPECTED_FRESH_SCHEDULER_LR = 1.0e-3
EXPECTED_RESTORED_OPTIMIZER_LR = 2.25e-5
EXPECTED_RESTORED_OPTIMIZER_STEP = 10020
EXPECTED_DESIRED_KL = 0.01
EXPECTED_CLIP_PARAM = 0.2
EXPECTED_MAX_GRAD_NORM = 1.0


def _capture_rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_digest(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cpu": _sha256_tensor(state["cpu"]),
        "cuda": [_sha256_tensor(value) for value in state["cuda"]],
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


class _FirstStepFactorial:
    """Execute four counterfactual first steps from one immutable snapshot."""

    def __init__(
        self,
        runner: ResumeStateDiscriminatorMotionOnPolicyRunner,
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
        self.named_parameters = list(self.policy.named_parameters())
        self.baseline_parameters = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters
        }
        self.baseline_parameter_norms = _norms_from_tensors(self.named_parameters)
        self.baseline_model = copy.deepcopy(self.policy.state_dict())
        self.baseline_optimizer = copy.deepcopy(self.alg.optimizer.state_dict())
        self.baseline_fresh_scheduler_lr = float(self.alg.learning_rate)
        self.baseline_restored_optimizer_lr = self._single_optimizer_lr()
        self.baseline_storage_step = int(self.storage.step)
        self.baseline_rng = _capture_rng_state()
        self.baseline_identity = self._state_identity()
        self.current_event: BatchEvent | None = None
        self.current_log_prob: torch.Tensor | None = None
        self.current_mu: torch.Tensor | None = None
        self.current_sigma: torch.Tensor | None = None
        self.current_entropy: torch.Tensor | None = None
        self.current_value: torch.Tensor | None = None
        self.current_gradient: dict[str, Any] | None = None
        self.optimizer_step_count = 0
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _single_optimizer_lr(self) -> float:
        rates = {float(group["lr"]) for group in self.alg.optimizer.param_groups}
        if len(rates) != 1:
            raise RuntimeError(f"optimizer learning rates differ: {sorted(rates)}")
        return rates.pop()

    def _state_identity(self) -> dict[str, str]:
        return {
            "model": _state_digest(self.policy.state_dict()),
            "optimizer": _state_digest(self.alg.optimizer.state_dict()),
            "actor_normalizer": _state_digest(self.runner.obs_normalizer.state_dict()),
            "critic_normalizer": _state_digest(
                self.runner.privileged_obs_normalizer.state_dict()
            ),
        }

    def _restore_baseline(self) -> None:
        self.policy.load_state_dict(copy.deepcopy(self.baseline_model))
        self.alg.optimizer.load_state_dict(copy.deepcopy(self.baseline_optimizer))
        self.alg.learning_rate = self.baseline_fresh_scheduler_lr
        self.storage.step = self.baseline_storage_step
        self.policy.zero_grad(set_to_none=True)
        self.policy.distribution = None
        _restore_rng_state(self.baseline_rng)
        if self._state_identity() != self.baseline_identity:
            raise RuntimeError("counterfactual branch did not restore baseline state")

    def _rollout_summary(self) -> dict[str, Any]:
        flat_phases = self.phases.flatten()
        bins = phase_bin_indices(
            flat_phases,
            reference_states=self.reference_states,
            bin_count=self.bin_count,
        )
        dones = self.storage.dones.flatten(0, 1).squeeze(-1).bool()
        timeouts = self.timeouts.flatten().bool()
        advantages = self.storage.advantages.flatten(0, 1).squeeze(-1)
        initial_histogram = torch.bincount(
            self.phases[0].to(dtype=torch.long), minlength=self.reference_states
        )
        return {
            "samples": int(flat_phases.numel()),
            "rollout_steps": int(self.phases.shape[0]),
            "environments": int(self.phases.shape[1]),
            "reference_states": self.reference_states,
            "adaptive_bin_count": self.bin_count,
            "distinct_phases": int(torch.unique(flat_phases).numel()),
            "initial_distinct_phases": int((initial_histogram > 0).sum().item()),
            "initial_phase_histogram": initial_histogram.detach().cpu().tolist(),
            "phase_bin_counts": torch.bincount(bins, minlength=self.bin_count)
            .detach()
            .cpu()
            .tolist(),
            "done_count": int(dones.sum().item()),
            "timeout_count": int(timeouts.sum().item()),
            "hard_termination_count": int((dones & ~timeouts).sum().item()),
            "advantage": _tensor_stats(advantages),
            "advantage_by_phase_bin": [
                {
                    "phase_bin": phase_bin,
                    "statistics": _tensor_stats(advantages[bins == phase_bin]),
                }
                for phase_bin in range(self.bin_count)
            ],
        }

    def _save_rollout_tensors(self) -> dict[str, Any]:
        path = self.output_dir / "factorial_rollout_tensors.pt"
        payload = {
            "phases": self.phases.detach().cpu(),
            "timeouts": self.timeouts.detach().cpu(),
            "native_permutation": self.expected_permutation.detach().cpu(),
            "observations": self.storage.observations.detach().cpu(),
            "privileged_observations": self.storage.privileged_observations.detach().cpu(),
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

    def _save_checkpoint(self, name: str) -> dict[str, Any]:
        path = self.checkpoint_dir / name
        if path.exists():
            raise RuntimeError(f"refusing to overwrite checkpoint: {path}")
        self.runner.save(str(path))
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "optimizer_steps": _optimizer_steps(self.alg.optimizer),
            "optimizer_learning_rates": [
                float(group["lr"]) for group in self.alg.optimizer.param_groups
            ],
        }

    def _parameter_drift(self) -> dict[str, Any]:
        differences = [
            (name, parameter.detach() - self.baseline_parameters[name])
            for name, parameter in self.named_parameters
        ]
        norms = _norms_from_tensors(differences)
        return {
            "l2": norms,
            "relative_l2": {
                group: norms[group] / self.baseline_parameter_norms[group]
                if self.baseline_parameter_norms[group] > 0.0
                else None
                for group in norms
            },
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

    def _gradient_digest(self) -> str:
        gradients = {
            name: parameter.grad.detach()
            for name, parameter in self.named_parameters
            if parameter.grad is not None
        }
        return _state_digest(gradients)

    def _pre_step_metrics(self) -> dict[str, Any]:
        if self.current_event is None or self.current_gradient is None:
            raise RuntimeError("optimizer step is missing batch or gradient metadata")
        tensors = (
            self.current_log_prob,
            self.current_mu,
            self.current_sigma,
            self.current_entropy,
            self.current_value,
        )
        if any(value is None for value in tensors):
            raise RuntimeError("optimizer step instrumentation is incomplete")
        event = self.current_event
        indices = event.indices
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1)[indices].squeeze(-1)
        old_mu = self.storage.mu.flatten(0, 1)[indices]
        old_sigma = self.storage.sigma.flatten(0, 1)[indices]
        advantages = self.storage.advantages.flatten(0, 1)[indices].squeeze(-1)
        phases = self.phases.flatten()[indices]
        assert self.current_log_prob is not None
        assert self.current_mu is not None
        assert self.current_sigma is not None
        assert self.current_entropy is not None
        assert self.current_value is not None
        log_ratio = self.current_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.alg.clip_param, 1.0 + self.alg.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        target_values = self.storage.values.flatten(0, 1)[indices]
        returns = self.storage.returns.flatten(0, 1)[indices]
        value_clipped = target_values + (self.current_value - target_values).clamp(
            -self.alg.clip_param, self.alg.clip_param
        )
        value_loss = torch.max(
            (self.current_value - returns).pow(2),
            (value_clipped - returns).pow(2),
        ).mean()
        entropy = self.current_entropy.mean()
        total_loss = (
            surrogate_loss
            + self.alg.value_loss_coef * value_loss
            - self.alg.entropy_coef * entropy
        )
        analytic_kl = torch.sum(
            torch.log(self.current_sigma / old_sigma + 1.0e-5)
            + (old_sigma.square() + (old_mu - self.current_mu).square())
            / (2.0 * self.current_sigma.square())
            - 0.5,
            dim=-1,
        )
        bins = phase_bin_indices(
            phases,
            reference_states=self.reference_states,
            bin_count=self.bin_count,
        )
        return {
            "epoch": event.epoch,
            "mini_batch": event.mini_batch,
            "global_step": event.global_step,
            "indices_sha256": _sha256_tensor(indices),
            "sample_count": int(indices.numel()),
            "phase_bin_counts": torch.bincount(bins, minlength=self.bin_count)
            .detach()
            .cpu()
            .tolist(),
            "advantage": _tensor_stats(advantages),
            "ratio": _tensor_stats(ratio),
            "log_ratio": _tensor_stats(log_ratio),
            "analytic_kl": _tensor_stats(analytic_kl),
            "approximate_kl": float(((ratio - 1.0) - log_ratio).mean().item()),
            "clipped_fraction": float(
                ((ratio - 1.0).abs() > self.alg.clip_param).float().mean().item()
            ),
            "forward_sha256": {
                "log_probability": _sha256_tensor(self.current_log_prob),
                "mean": _sha256_tensor(self.current_mu),
                "sigma": _sha256_tensor(self.current_sigma),
                "entropy": _sha256_tensor(self.current_entropy),
                "value": _sha256_tensor(self.current_value),
            },
            "loss": {
                "surrogate": float(surrogate_loss.item()),
                "value": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "total": float(total_loss.item()),
            },
            "applied_learning_rate": self._single_optimizer_lr(),
            "scheduler_learning_rate": float(self.alg.learning_rate),
            "optimizer_state_steps_before": _optimizer_steps(self.alg.optimizer),
            "gradient": self.current_gradient,
        }

    def _run_arm(self, spec: ArmSpec, original_update: Any) -> dict[str, Any]:
        self._restore_baseline()
        pre_intervention_identity = self._state_identity()
        intervention = prepare_arm(
            self.alg,
            spec,
            restored_learning_rate=self.baseline_restored_optimizer_lr,
            fresh_scheduler_learning_rate=self.baseline_fresh_scheduler_lr,
        )
        post_intervention_identity = self._state_identity()
        self.optimizer_step_count = 0
        record: dict[str, Any] | None = None

        original_generator = self.storage.mini_batch_generator
        original_log_prob = self.policy.get_actions_log_prob
        original_evaluate = self.policy.evaluate
        original_clip = torch.nn.utils.clip_grad_norm_
        original_optimizer_step = self.alg.optimizer.step

        def one_batch_generator(storage: Any, num_mini_batches: int, num_epochs: int):
            if (
                num_mini_batches != EXPECTED_MINI_BATCHES
                or num_epochs != EXPECTED_EPOCHS
            ):
                raise RuntimeError("native PPO requested an unexpected batch schedule")
            generator = instrumented_mini_batch_generator(
                storage,
                num_mini_batches,
                num_epochs,
                self._on_batch,
                expected_indices=self.expected_permutation,
            )
            yield next(generator)

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
                "post_clip_sha256": self._gradient_digest(),
                "native_returned_total_norm": float(returned.detach().cpu().item()),
                "max_grad_norm": float(max_norm),
                "clip_scale": post["all"] / pre["all"] if pre["all"] > 0.0 else None,
            }
            return returned

        def optimizer_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal record
            self.optimizer_step_count += 1
            if self.optimizer_step_count != 1:
                raise RuntimeError(
                    "counterfactual branch attempted multiple Adam steps"
                )
            record = self._pre_step_metrics()
            result = original_optimizer_step(*args, **kwargs)
            record["optimizer_state_steps_after"] = _optimizer_steps(self.alg.optimizer)
            record["parameter_drift"] = self._parameter_drift()
            return result

        self.storage.mini_batch_generator = types.MethodType(
            one_batch_generator, self.storage
        )
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

        if self.optimizer_step_count != 1 or record is None:
            raise RuntimeError(
                "counterfactual branch did not execute exactly one Adam step"
            )
        checkpoint = self._save_checkpoint(f"model_{spec.name}_step_01.pt")
        result = {
            "arm": spec.to_dict(),
            "intervention": intervention,
            "pre_intervention_identity": pre_intervention_identity,
            "post_intervention_identity": post_intervention_identity,
            "pre_step": record,
            "post_step_identity": self._state_identity(),
            "post_step_rng": _rng_digest(_capture_rng_state()),
            "checkpoint": checkpoint,
            "native_codepath_loss_dict_divided_by_configured_20": {
                name: float(value) for name, value in loss_dict.items()
            },
        }
        if not _all_finite(result):
            raise RuntimeError(f"nonfinite branch result: {spec.name}")
        print(
            "[RESUME-STATE-DISCRIMINATOR] "
            f"arm={spec.name} lr={record['applied_learning_rate']:.9g} "
            f"actor_drift={record['parameter_drift']['relative_l2']['actor']:.9g}",
            flush=True,
        )
        return result

    def run(self, original_update: Any) -> dict[str, Any]:
        if not math.isclose(
            self.baseline_fresh_scheduler_lr,
            EXPECTED_FRESH_SCHEDULER_LR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "fresh scheduler learning rate differs from the registered 1e-3"
            )
        if not math.isclose(
            self.baseline_restored_optimizer_lr,
            EXPECTED_RESTORED_OPTIMIZER_LR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError(
                "restored optimizer learning rate differs from the registered 2.25e-5"
            )
        if _optimizer_steps(self.alg.optimizer) != [EXPECTED_RESTORED_OPTIMIZER_STEP]:
            raise RuntimeError("restored Adam step differs from the registered 10020")
        rollout_summary = self._rollout_summary()
        rollout_tensors = self._save_rollout_tensors()
        step0 = self._save_checkpoint("model_step_00.pt")
        branch_results = [self._run_arm(spec, original_update) for spec in ARM_SPECS]
        if tuple(item["arm"]["name"] for item in branch_results) != ARM_NAMES:
            raise RuntimeError("counterfactual arm order drift")
        if ARM_SPECS[-1].name != NATIVE_ARM:
            raise RuntimeError("native branch must remain last")

        common_keys = (
            "indices_sha256",
            "sample_count",
            "phase_bin_counts",
            "advantage",
            "ratio",
            "log_ratio",
            "analytic_kl",
            "approximate_kl",
            "clipped_fraction",
            "forward_sha256",
            "loss",
            "gradient",
        )
        reference = branch_results[0]["pre_step"]
        for branch in branch_results:
            candidate = branch["pre_step"]
            if any(candidate[key] != reference[key] for key in common_keys):
                raise RuntimeError(
                    f"pre-Adam batch or gradient differs for {branch['arm']['name']}"
                )
            if branch["pre_intervention_identity"] != self.baseline_identity:
                raise RuntimeError(
                    f"baseline identity differs for {branch['arm']['name']}"
                )
            name = branch["arm"]["name"]
            synchronize = bool(branch["arm"]["synchronize_scheduler"])
            reset_adam = bool(branch["arm"]["reset_adam"])
            expected_scheduler_start = (
                self.baseline_restored_optimizer_lr
                if synchronize
                else self.baseline_fresh_scheduler_lr
            )
            expected_applied_rate = expected_scheduler_start * 1.5
            if not math.isclose(
                float(candidate["applied_learning_rate"]),
                expected_applied_rate,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ) or not math.isclose(
                float(candidate["scheduler_learning_rate"]),
                expected_applied_rate,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise RuntimeError(f"adaptive learning-rate application drift: {name}")
            expected_before = [] if reset_adam else [EXPECTED_RESTORED_OPTIMIZER_STEP]
            expected_after = (
                [1] if reset_adam else [EXPECTED_RESTORED_OPTIMIZER_STEP + 1]
            )
            if (
                candidate["optimizer_state_steps_before"] != expected_before
                or candidate["optimizer_state_steps_after"] != expected_after
            ):
                raise RuntimeError(f"Adam state accounting drift: {name}")
            intervention = branch["intervention"]
            expected_entries = (
                0
                if reset_adam
                else intervention["optimizer_state_entries_before_intervention"]
            )
            if (
                intervention["optimizer_state_entries_after_intervention"]
                != expected_entries
            ):
                raise RuntimeError(f"Adam state intervention drift: {name}")
            for component in ("model", "actor_normalizer", "critic_normalizer"):
                if (
                    branch["post_intervention_identity"][component]
                    != self.baseline_identity[component]
                ):
                    raise RuntimeError(
                        f"non-optimizer state changed before Adam for {name}: {component}"
                    )

        kl_mean = float(reference["analytic_kl"]["mean"])
        if not 0.0 < kl_mean < float(self.alg.desired_kl) / 2.0:
            raise RuntimeError(
                f"first-batch KL did not select the registered low-positive branch: {kl_mean}"
            )
        if float(reference["clipped_fraction"]) != 0.0:
            raise RuntimeError("unchanged first-batch policy was unexpectedly clipped")
        if (
            len(
                {
                    json.dumps(branch["post_step_rng"], sort_keys=True)
                    for branch in branch_results
                }
            )
            != 1
        ):
            raise RuntimeError("post-step RNG differs across counterfactual branches")
        if (
            len(
                {
                    json.dumps(
                        branch["native_codepath_loss_dict_divided_by_configured_20"],
                        sort_keys=True,
                    )
                    for branch in branch_results
                }
            )
            != 1
        ):
            raise RuntimeError("native loss dictionary differs across branches")
        final_rng = _rng_digest(_capture_rng_state())
        native_branch = branch_results[-1]
        if self._state_identity() != native_branch["post_step_identity"]:
            raise RuntimeError("retained outer state differs from native branch")
        if final_rng != native_branch["post_step_rng"]:
            raise RuntimeError("retained outer RNG differs from native branch")

        result = {
            "schema_version": 1,
            "complete": True,
            "factorial": True,
            "single_shared_rollout": True,
            "single_native_partition_per_arm": True,
            "baseline_state_identity": self.baseline_identity,
            "baseline_rng": _rng_digest(self.baseline_rng),
            "fresh_scheduler_learning_rate": self.baseline_fresh_scheduler_lr,
            "restored_optimizer_learning_rate": self.baseline_restored_optimizer_lr,
            "native_permutation_sha256": _sha256_tensor(self.expected_permutation),
            "rollout": rollout_summary,
            "rollout_tensors": rollout_tensors,
            "step0_checkpoint": step0,
            "branches": branch_results,
            "retained_outer_state_arm": NATIVE_ARM,
            "final_state_identity": self._state_identity(),
            "retained_outer_scheduler_learning_rate": float(self.alg.learning_rate),
            "retained_outer_optimizer_learning_rates": [
                float(group["lr"]) for group in self.alg.optimizer.param_groups
            ],
            "retained_outer_rng": final_rng,
        }
        _write_json(self.output_dir / "factorial_result.json", result)
        return branch_results[-1]["native_codepath_loss_dict_divided_by_configured_20"]


class ResumeStateDiscriminatorMotionOnPolicyRunner(
    InitialObservationNormalizedMotionOnPolicyRunner
):
    """Correct resume ordering, then branch only scheduler and Adam state."""

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> Any:
        if num_learning_iterations != 1 or self.current_learning_iteration != 500:
            raise RuntimeError(
                "resume-state discriminator requires one update from iteration 500"
            )
        if (
            self.env.num_envs != EXPECTED_ENVIRONMENTS
            or self.num_steps_per_env != EXPECTED_ROLLOUT_STEPS
        ):
            raise RuntimeError(
                "resume-state discriminator requires 4096 environments and H=24"
            )
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
        if (
            self.alg.schedule != "adaptive"
            or self.alg.desired_kl is None
            or not math.isclose(
                float(self.alg.desired_kl),
                EXPECTED_DESIRED_KL,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not self.alg.use_clipped_value_loss
            or not math.isclose(
                float(self.alg.clip_param),
                EXPECTED_CLIP_PARAM,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                float(self.alg.max_grad_norm),
                EXPECTED_MAX_GRAD_NORM,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise RuntimeError(
                "PPO trust-region settings differ from the registered adaptive-KL update"
            )

        output_value = os.environ.get(OUTPUT_DIRECTORY_ENV)
        if not output_value:
            raise RuntimeError(f"{OUTPUT_DIRECTORY_ENV} is required")
        output_dir = Path(output_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "factorial_result.json"
        if result_path.exists():
            raise RuntimeError("refusing to overwrite a completed discriminator")

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
            timeout = result[3].get("time_outs")
            if timeout is None:
                timeout = torch.zeros_like(result[2], dtype=torch.bool)
            timeouts.append(timeout.detach().clone().reshape(-1))
            return result

        def observed_compute_returns(last_critic_obs: torch.Tensor) -> Any:
            nonlocal returns_ready
            result = original_compute_returns(last_critic_obs)
            returns_ready = True
            return result

        def factorial_update() -> dict[str, Any]:
            if (
                not returns_ready
                or len(phases) != EXPECTED_ROLLOUT_STEPS
                or len(timeouts) != EXPECTED_ROLLOUT_STEPS
            ):
                raise RuntimeError("rollout capture is incomplete before PPO update")
            discriminator = _FirstStepFactorial(
                self,
                output_dir=output_dir,
                phases=torch.stack(phases),
                timeouts=torch.stack(timeouts),
                reference_states=reference_states,
                bin_count=bin_count,
            )
            return discriminator.run(original_update)

        self.env.step = observed_env_step
        self.alg.compute_returns = observed_compute_returns
        self.alg.update = factorial_update
        try:
            result = super().learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            self.env.step = original_env_step
            self.alg.compute_returns = original_compute_returns
            self.alg.update = original_update

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["runner_completed"] = True
        payload["current_learning_iteration"] = self.current_learning_iteration
        payload["normalizer_counts"] = {
            "actor": int(self.obs_normalizer.count.detach().cpu().item()),
            "critic": int(self.privileged_obs_normalizer.count.detach().cpu().item()),
        }
        payload["outer_state_after_super"] = {
            "model": _state_digest(self.alg.policy.state_dict()),
            "optimizer": _state_digest(self.alg.optimizer.state_dict()),
            "actor_normalizer": _state_digest(self.obs_normalizer.state_dict()),
            "critic_normalizer": _state_digest(
                self.privileged_obs_normalizer.state_dict()
            ),
            "scheduler_learning_rate": float(self.alg.learning_rate),
            "optimizer_learning_rates": [
                float(group["lr"]) for group in self.alg.optimizer.param_groups
            ],
            "optimizer_steps": _optimizer_steps(self.alg.optimizer),
            "rng": _rng_digest(_capture_rng_state()),
        }
        if {
            key: payload["outer_state_after_super"][key]
            for key in (
                "model",
                "optimizer",
                "actor_normalizer",
                "critic_normalizer",
            )
        } != payload["final_state_identity"]:
            raise RuntimeError(
                "outer runner changed native state after factorial update"
            )
        if not math.isclose(
            payload["outer_state_after_super"]["scheduler_learning_rate"],
            payload["retained_outer_scheduler_learning_rate"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError("outer runner changed native scheduler state")
        _write_json(result_path, payload)
        print(
            f"[RESUME-STATE-DISCRIMINATOR] complete result={result_path}",
            flush=True,
        )
        return result
