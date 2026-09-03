import torch
from normalization import normalize_initial_observations, normalize_without_update


class UpdatingNormalizer(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.register_buffer("count", torch.tensor(10))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += value.shape[0]
        return value + self.offset


def test_normalize_without_update_preserves_count_and_mode():
    normalizer = UpdatingNormalizer(2.0)
    normalizer.train()
    result = normalize_without_update(normalizer, torch.tensor([[1.0, 3.0]]))
    assert torch.equal(result, torch.tensor([[3.0, 5.0]]))
    assert int(normalizer.count) == 10
    assert normalizer.training is True


def test_initial_actor_and_privileged_observations_are_normalized_once():
    actor = UpdatingNormalizer(2.0)
    critic = UpdatingNormalizer(-1.0)
    raw_actor = torch.tensor([[1.0, 3.0]])
    raw_critic = torch.tensor([[5.0, 7.0, 9.0]])
    extras = {"observations": {"critic": raw_critic, "other": torch.tensor([[4.0]])}, "token": 3}

    normalized_actor, normalized_extras, metadata = normalize_initial_observations(
        raw_actor,
        extras,
        actor_normalizer=actor,
        privileged_normalizer=critic,
        privileged_observation_type="critic",
        device="cpu",
    )

    assert torch.equal(normalized_actor, torch.tensor([[3.0, 5.0]]))
    assert torch.equal(normalized_extras["observations"]["critic"], torch.tensor([[4.0, 6.0, 8.0]]))
    assert normalized_extras["observations"]["other"] is extras["observations"]["other"]
    assert normalized_extras is not extras
    assert normalized_extras["observations"] is not extras["observations"]
    assert metadata == {
        "actor_applied": True,
        "privileged_applied": True,
        "actor_batch": 1,
        "actor_features": 2,
        "privileged_features": 3,
    }
    assert int(actor.count) == int(critic.count) == 10


def test_missing_privileged_group_uses_normalized_actor_fallback():
    actor = UpdatingNormalizer(2.0)
    critic = UpdatingNormalizer(-1.0)
    result, extras, metadata = normalize_initial_observations(
        torch.tensor([[1.0, 3.0]]),
        {"observations": {}},
        actor_normalizer=actor,
        privileged_normalizer=critic,
        privileged_observation_type="critic",
        device="cpu",
    )
    assert torch.equal(result, torch.tensor([[3.0, 5.0]]))
    assert extras["observations"] == {}
    assert metadata["privileged_applied"] is False
