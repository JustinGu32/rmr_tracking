from types import SimpleNamespace

import pytest
import torch
from probe import (
    checkpoint_steps,
    gradient_geometry,
    instrumented_mini_batch_generator,
    phase_bin_indices,
)
from rsl_rl.storage import RolloutStorage


def _storage() -> RolloutStorage:
    storage = RolloutStorage("rl", 4, 3, [2], [3], [1], device="cpu")
    tensors = (
        storage.observations,
        storage.privileged_observations,
        storage.actions,
        storage.values,
        storage.returns,
        storage.actions_log_prob,
        storage.advantages,
        storage.mu,
        storage.sigma,
    )
    for offset, tensor in enumerate(tensors):
        values = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape)
        tensor.copy_(values + 1000 * offset)
    return storage


def test_all_native_optimizer_steps_are_materialized():
    assert checkpoint_steps(num_epochs=5, num_mini_batches=4) == tuple(range(21))
    with pytest.raises(ValueError, match="positive"):
        checkpoint_steps(num_epochs=0, num_mini_batches=4)


def test_phase_bins_match_native_motion_command_formula():
    phases = torch.tensor([0, 45, 46, 90, 91, 135, 136, 181, 182, 226, 227, 271])
    assert phase_bin_indices(phases, reference_states=272, bin_count=6).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]
    with pytest.raises(ValueError, match="outside"):
        phase_bin_indices(torch.tensor([272]), reference_states=272, bin_count=6)


def test_instrumented_generator_is_tensor_and_rng_identical_to_rsl_rl():
    storage = _storage()
    torch.manual_seed(17)
    expected = list(storage.mini_batch_generator(3, 2))
    expected_rng = torch.random.get_rng_state().clone()

    events: list[SimpleNamespace] = []
    torch.manual_seed(17)
    actual = list(instrumented_mini_batch_generator(storage, 3, 2, events.append))
    actual_rng = torch.random.get_rng_state().clone()

    assert len(actual) == len(expected) == 6
    assert torch.equal(actual_rng, expected_rng)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        for actual_item, expected_item in zip(
            actual_batch[:9], expected_batch[:9], strict=True
        ):
            assert torch.equal(actual_item, expected_item)
    assert [(event.epoch, event.mini_batch) for event in events] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert all(torch.equal(events[i].indices, events[i + 3].indices) for i in range(3))


def test_generator_refuses_expected_permutation_drift():
    storage = _storage()
    wrong = torch.arange(12)
    torch.manual_seed(17)
    with pytest.raises(RuntimeError, match="permutation drift"):
        list(
            instrumented_mini_batch_generator(
                storage, 3, 1, lambda _: None, expected_indices=wrong
            )
        )


def test_gradient_geometry_reports_variance_alignment_and_cancellation():
    summary = gradient_geometry(
        labels=["a", "b"],
        gradients=[torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
        weights=[0.5, 0.5],
    )
    assert summary["pairwise_cosine"] == [{"left": "a", "right": "b", "cosine": -1.0}]
    assert summary["weighted_sum_norm"] == 0.0
    assert summary["weighted_cancellation_ratio"] == 0.0
    assert summary["mean_squared_distance_from_weighted_sum"] == pytest.approx(1.0)
