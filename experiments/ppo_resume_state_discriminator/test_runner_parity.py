import copy
import json
import sys
import types
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

fake_parent = types.ModuleType(
    "experiments.resume_initial_observation_normalization.runner"
)
fake_parent.InitialObservationNormalizedMotionOnPolicyRunner = object
sys.modules[fake_parent.__name__] = fake_parent

from experiments.ppo_resume_state_discriminator.design import NATIVE_ARM
from experiments.ppo_resume_state_discriminator.runner import _FirstStepFactorial


class _FakeRunner:
    def __init__(self, algorithm: PPO):
        self.alg = algorithm
        self.obs_normalizer = torch.nn.Identity()
        self.privileged_obs_normalizer = torch.nn.Identity()

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "obs_norm_state_dict": self.obs_normalizer.state_dict(),
                "privileged_obs_norm_state_dict": self.privileged_obs_normalizer.state_dict(),
                "iter": 500,
            },
            path,
        )


def _algorithm() -> PPO:
    policy = ActorCritic(
        2,
        3,
        1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        activation="elu",
        init_noise_std=0.7,
    )
    algorithm = PPO(
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        entropy_coef=0.005,
    )
    algorithm.storage = RolloutStorage("rl", 4, 3, [2], [3], [1], device="cpu")
    storage = algorithm.storage
    storage.observations.copy_(
        torch.linspace(-1.0, 1.0, storage.observations.numel()).reshape_as(
            storage.observations
        )
    )
    storage.privileged_observations.copy_(
        torch.linspace(0.5, -0.5, storage.privileged_observations.numel()).reshape_as(
            storage.privileged_observations
        )
    )
    with torch.no_grad():
        observations = storage.observations.flatten(0, 1)
        policy.update_distribution(observations)
        actions = policy.action_mean + 0.125
        storage.actions.copy_(actions.reshape_as(storage.actions))
        storage.actions_log_prob.copy_(
            policy.get_actions_log_prob(actions).reshape_as(storage.actions_log_prob)
        )
        storage.mu.copy_(policy.action_mean.reshape_as(storage.mu))
        storage.sigma.copy_(policy.action_std.reshape_as(storage.sigma))
        values = policy.evaluate(storage.privileged_observations.flatten(0, 1))
        storage.values.copy_(values.reshape_as(storage.values))
    storage.returns.copy_(storage.values + 0.2)
    advantages = torch.linspace(-1.0, 1.0, storage.advantages.numel()).reshape_as(
        storage.advantages
    )
    storage.advantages.copy_((advantages - advantages.mean()) / advantages.std())
    storage.rewards.copy_(
        torch.linspace(0.0, 0.2, storage.rewards.numel()).reshape_as(storage.rewards)
    )
    storage.dones.zero_()

    algorithm.optimizer.zero_grad(set_to_none=True)
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, 0.01)
    algorithm.optimizer.step()
    for state in algorithm.optimizer.state.values():
        state["step"].fill_(10020)
    for group in algorithm.optimizer.param_groups:
        group["lr"] = 2.25e-5
    algorithm.learning_rate = 1.0e-3
    policy.zero_grad(set_to_none=True)
    return algorithm


def _clone_algorithm(source: PPO) -> PPO:
    clone = _algorithm()
    clone.policy.load_state_dict(copy.deepcopy(source.policy.state_dict()))
    clone.optimizer.load_state_dict(copy.deepcopy(source.optimizer.state_dict()))
    clone.learning_rate = source.learning_rate
    for name in (
        "observations",
        "privileged_observations",
        "actions",
        "values",
        "returns",
        "actions_log_prob",
        "advantages",
        "mu",
        "sigma",
        "rewards",
        "dones",
    ):
        getattr(clone.storage, name).copy_(getattr(source.storage, name))
    clone.storage.step = source.storage.step
    return clone


def _run_native_first_batch(algorithm: PPO) -> None:
    original = algorithm.storage.mini_batch_generator

    def one_batch(storage, num_mini_batches, num_epochs):
        generator = original(num_mini_batches, num_epochs)
        yield next(generator)

    algorithm.storage.mini_batch_generator = types.MethodType(
        one_batch, algorithm.storage
    )
    try:
        algorithm.update()
    finally:
        algorithm.storage.mini_batch_generator = original


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_factorial_uses_identical_batch_and_retains_exact_native_first_step(
    tmp_path: Path,
):
    torch.manual_seed(123)
    source = _algorithm()
    expected = _clone_algorithm(source)
    observed = _clone_algorithm(source)
    update_rng = torch.random.get_rng_state().clone()

    torch.random.set_rng_state(update_rng)
    _run_native_first_batch(expected)
    expected_rng = torch.random.get_rng_state().clone()

    phases = torch.arange(12).reshape(3, 4)
    torch.random.set_rng_state(update_rng)
    factorial = _FirstStepFactorial(
        _FakeRunner(observed),
        output_dir=tmp_path,
        phases=phases,
        timeouts=torch.zeros_like(phases, dtype=torch.bool),
        reference_states=12,
        bin_count=6,
    )
    factorial.run(observed.update)
    observed_rng = torch.random.get_rng_state().clone()

    assert torch.equal(expected_rng, observed_rng)
    _assert_nested_equal(expected.policy.state_dict(), observed.policy.state_dict())
    _assert_nested_equal(
        expected.optimizer.state_dict(), observed.optimizer.state_dict()
    )

    result = json.loads((tmp_path / "factorial_result.json").read_text())
    assert result["retained_outer_state_arm"] == NATIVE_ARM
    assert len(result["branches"]) == 4
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 5
    gradients = {
        branch["pre_step"]["gradient"]["post_clip_sha256"]
        for branch in result["branches"]
    }
    indices = {branch["pre_step"]["indices_sha256"] for branch in result["branches"]}
    assert len(gradients) == len(indices) == 1
    by_name = {branch["arm"]["name"]: branch for branch in result["branches"]}
    assert (
        by_name["restored_adam__synced_scheduler"]["pre_step"]["applied_learning_rate"]
        == 3.375e-5
    )
    assert (
        by_name["restored_adam__fresh_scheduler"]["pre_step"]["applied_learning_rate"]
        == 1.5e-3
    )
    assert (
        by_name["reset_adam__fresh_scheduler"]["pre_step"][
            "optimizer_state_steps_before"
        ]
        == []
    )
    assert by_name["reset_adam__fresh_scheduler"]["pre_step"][
        "optimizer_state_steps_after"
    ] == [1]
    assert by_name["restored_adam__fresh_scheduler"]["pre_step"][
        "optimizer_state_steps_after"
    ] == [10021]
