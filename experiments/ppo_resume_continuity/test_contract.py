import pytest
from contract import CHECKPOINT_STEPS, classify_continuity


def _short(value: bool = True) -> dict[int, bool]:
    return {step: value for step in CHECKPOINT_STEPS}


def test_full_update_preservation_is_categorical():
    result = classify_continuity(
        short_complete=_short(),
        step0_long_complete=False,
        step20_long_complete=False,
    )
    assert result["outcome"] == "synchronized-resume-preserves-sampled-full-update"
    assert result["failed_steps"] == []


def test_exact_long_acquisition_takes_precedence():
    result = classify_continuity(
        short_complete=_short(),
        step0_long_complete=False,
        step20_long_complete=True,
    )
    assert result["outcome"] == "synchronized-resume-acquires-exact-long"


@pytest.mark.parametrize(
    ("failed", "expected", "recoveries"),
    [
        ({8, 12, 16, 20}, "synchronized-resume-delays-loss", []),
        ({8, 12}, "synchronized-resume-sampled-nonmonotonic-loss", [16, 20]),
    ],
)
def test_loss_reports_first_sampled_failure_and_recovery(failed, expected, recoveries):
    short = {step: step not in failed for step in CHECKPOINT_STEPS}
    result = classify_continuity(
        short_complete=short,
        step0_long_complete=False,
        step20_long_complete=False,
    )
    assert result["outcome"] == expected
    assert result["first_observed_loss_step"] == 8
    assert result["recovery_steps"] == recoveries


def test_missing_checkpoint_is_invalid():
    short = _short()
    short.pop(12)
    result = classify_continuity(
        short_complete=short,
        step0_long_complete=False,
        step20_long_complete=False,
    )
    assert result["outcome"] == "invalid-execution"
