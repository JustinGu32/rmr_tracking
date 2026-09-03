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


@pytest.mark.parametrize(
    "algorithm",
    [_algorithm(scheduler=5.0e-4), _algorithm(group=1.0e-4)],
)
def test_synchronization_rejects_resume_state_drift(algorithm):
    with pytest.raises(RuntimeError, match="drift"):
        synchronize_resume_scheduler(algorithm)
