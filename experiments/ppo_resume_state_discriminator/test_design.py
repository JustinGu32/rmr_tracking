from types import SimpleNamespace

import pytest
import torch
from design import ARM_SPECS, prepare_arm


def _algorithm() -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=2.25e-5)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    return SimpleNamespace(
        optimizer=optimizer,
        learning_rate=1.0e-3,
    )


@pytest.mark.parametrize("spec", ARM_SPECS, ids=lambda spec: spec.name)
def test_prepare_arm_changes_only_selected_resume_state(spec):
    algorithm = _algorithm()
    state_entries = len(algorithm.optimizer.state)
    record = prepare_arm(
        algorithm,
        spec,
        restored_learning_rate=2.25e-5,
        fresh_scheduler_learning_rate=1.0e-3,
    )

    expected_scheduler = 2.25e-5 if spec.synchronize_scheduler else 1.0e-3
    assert algorithm.learning_rate == expected_scheduler
    assert [group["lr"] for group in algorithm.optimizer.param_groups] == [2.25e-5]
    assert len(algorithm.optimizer.state) == (0 if spec.reset_adam else state_entries)
    assert record["optimizer_state_entries_before_intervention"] == state_entries


def test_prepare_arm_rejects_optimizer_rate_drift():
    algorithm = _algorithm()
    with pytest.raises(RuntimeError, match="expected rate"):
        prepare_arm(
            algorithm,
            ARM_SPECS[0],
            restored_learning_rate=1.0e-5,
            fresh_scheduler_learning_rate=1.0e-3,
        )
