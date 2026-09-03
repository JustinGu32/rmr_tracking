import pytest
from contract import classify_factorial
from design import ARM_NAMES, NATIVE_ARM


def _mapping(*complete: str) -> dict[str, bool]:
    return {name: name in complete for name in ARM_NAMES}


@pytest.mark.parametrize(
    ("complete", "outcome"),
    [
        (
            {
                "restored_adam__synced_scheduler",
                "reset_adam__synced_scheduler",
            },
            "scheduler-synchronization-preserves",
        ),
        (
            {"reset_adam__fresh_scheduler", "reset_adam__synced_scheduler"},
            "adam-reset-preserves",
        ),
        (
            {
                "restored_adam__synced_scheduler",
                "reset_adam__fresh_scheduler",
                "reset_adam__synced_scheduler",
            },
            "both-single-factor-interventions-preserve",
        ),
        ({"reset_adam__synced_scheduler"}, "combined-intervention-required"),
        (set(), "neither-intervention-preserves"),
        ({"restored_adam__synced_scheduler"}, "nonadditive-interaction"),
    ],
)
def test_classifies_factorial_outcomes(complete, outcome):
    result = classify_factorial(
        step0_short_complete=True,
        short_complete=_mapping(*complete),
        long_complete=_mapping(),
    )
    assert result["outcome"] == outcome


def test_rejects_failed_controls_and_incomplete_arm_sets():
    assert (
        classify_factorial(
            step0_short_complete=False,
            short_complete=_mapping(),
            long_complete=_mapping(),
        )["outcome"]
        == "no-update-control-fails"
    )
    assert (
        classify_factorial(
            step0_short_complete=True,
            short_complete=_mapping(NATIVE_ARM),
            long_complete=_mapping(),
        )["outcome"]
        == "native-loss-not-reproduced"
    )
    assert (
        classify_factorial(
            step0_short_complete=True,
            short_complete=_mapping(),
            long_complete=_mapping(ARM_NAMES[0]),
        )["outcome"]
        == "unexpected-long-completion"
    )
    assert (
        classify_factorial(
            step0_short_complete=True,
            short_complete={ARM_NAMES[0]: False},
            long_complete=_mapping(),
        )["outcome"]
        == "invalid-execution"
    )
