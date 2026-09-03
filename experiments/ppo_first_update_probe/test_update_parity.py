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

from experiments.ppo_first_update_probe.runner import _UpdateProbe


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
        learning_rate=1e-3,
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
        flat_obs = storage.observations.flatten(0, 1)
        policy.update_distribution(flat_obs)
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
    return algorithm


def _clone_algorithm(source: PPO) -> PPO:
    clone = _algorithm()
    clone.policy.load_state_dict(copy.deepcopy(source.policy.state_dict()))
    clone.optimizer.load_state_dict(copy.deepcopy(source.optimizer.state_dict()))
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
    return clone


def test_measurement_probe_is_state_loss_and_rng_identical_to_native_update(
    tmp_path: Path,
):
    torch.manual_seed(123)
    source = _algorithm()
    native = _clone_algorithm(source)
    observed = _clone_algorithm(source)

    update_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(update_rng)
    native_loss = native.update()
    native_rng = torch.random.get_rng_state().clone()

    phases = torch.arange(12).reshape(3, 4)
    torch.random.set_rng_state(update_rng)
    probe = _UpdateProbe(
        _FakeRunner(observed),
        output_dir=tmp_path,
        phases=phases,
        timeouts=torch.zeros_like(phases, dtype=torch.bool),
        reference_states=12,
        bin_count=6,
    )
    observed_loss = probe.run(observed.update)
    observed_rng = torch.random.get_rng_state().clone()

    assert observed_loss == native_loss
    assert torch.equal(observed_rng, native_rng)
    for name, value in native.policy.state_dict().items():
        assert torch.equal(value, observed.policy.state_dict()[name]), name
    native_optimizer = native.optimizer.state_dict()
    observed_optimizer = observed.optimizer.state_dict()
    assert native_optimizer["param_groups"] == observed_optimizer["param_groups"]
    for key, state in native_optimizer["state"].items():
        for name, value in state.items():
            other = observed_optimizer["state"][key][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), (key, name)
            else:
                assert value == other
    result = json.loads((tmp_path / "probe_result.json").read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert len(result["optimizer_trace"]) == 20
    assert len(list((tmp_path / "checkpoints").glob("model_step_*.pt"))) == 21
