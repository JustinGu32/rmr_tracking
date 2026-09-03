from types import SimpleNamespace

import pytest
from design import (
    EXPECTED_FRESH_SCHEDULER_LR,
    EXPECTED_RESTORED_OPTIMIZER_LR,
    synchronize_resume_scheduler,
)


def _algorithm(*, scheduler=EXPECTED_FRESH_SCHEDULER_LR, group=None):
    rate = EXPECTED_RESTORED_OPTIMIZER_LR if group is None else group
    optimizer = SimpleNamespace(
        param_groups=[{"lr": rate}],
        state={"parameter": {"step": 10020}},
    )
    return SimpleNamespace(learning_rate=scheduler, optimizer=optimizer)


def test_synchronization_changes_only_scheduler_scalar():
    algorithm = _algorithm()
    state = algorithm.optimizer.state
    record = synchronize_resume_scheduler(algorithm)
    assert algorithm.learning_rate == EXPECTED_RESTORED_OPTIMIZER_LR
    assert algorithm.optimizer.param_groups == [{"lr": EXPECTED_RESTORED_OPTIMIZER_LR}]
    assert algorithm.optimizer.state is state
    assert record["causal_change"] == "PPO.learning_rate only"


def test_synchronization_copies_exact_group_float_instead_of_nominal_constant():
    exact_group_rate = 2.2500000000000008e-5
    algorithm = _algorithm(group=exact_group_rate)
    record = synchronize_resume_scheduler(algorithm)
    assert algorithm.learning_rate == exact_group_rate
    assert record["scheduler_learning_rate_after"] == exact_group_rate
    assert record["expected_first_applied_learning_rate"] == (exact_group_rate * 1.5)


@pytest.mark.parametrize(
    "algorithm",
    [_algorithm(scheduler=5.0e-4), _algorithm(group=1.0e-4)],
)
def test_synchronization_rejects_resume_state_drift(algorithm):
    with pytest.raises(RuntimeError, match="drift"):
        synchronize_resume_scheduler(algorithm)
