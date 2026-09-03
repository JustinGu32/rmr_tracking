from contract import classify_checkpoint_sweep


def test_classifies_monotonic_first_loss_without_survival_threshold():
    result = classify_checkpoint_sweep(
        short_complete={step: step < 3 for step in range(21)},
        step0_long_complete=False,
        step20_long_complete=False,
    )
    assert result["outcome"] == "optimizer-step-localized-monotonic-loss"
    assert result["first_loss_step"] == 3
    assert result["last_complete_step"] == 2
    assert result["recovery_steps"] == []


def test_classifies_nonmonotonic_recovery_explicitly():
    complete = {step: step < 2 for step in range(21)}
    complete[7] = True
    result = classify_checkpoint_sweep(
        short_complete=complete,
        step0_long_complete=False,
        step20_long_complete=False,
    )
    assert result["outcome"] == "optimizer-step-localized-nonmonotonic-loss"
    assert result["first_loss_step"] == 2
    assert result["recovery_steps"] == [7]


def test_rejects_failed_no_optimizer_control_and_missing_steps():
    failed_zero = {step: False for step in range(21)}
    assert (
        classify_checkpoint_sweep(
            short_complete=failed_zero,
            step0_long_complete=False,
            step20_long_complete=False,
        )["outcome"]
        == "no-optimizer-control-fails"
    )

    missing = {step: True for step in range(20)}
    assert (
        classify_checkpoint_sweep(
            short_complete=missing,
            step0_long_complete=False,
            step20_long_complete=False,
        )["outcome"]
        == "invalid-execution"
    )


def test_rejects_nonfailing_step20_and_long_control_drift():
    all_complete = {step: True for step in range(21)}
    assert (
        classify_checkpoint_sweep(
            short_complete=all_complete,
            step0_long_complete=False,
            step20_long_complete=False,
        )["outcome"]
        == "final-loss-not-reproduced"
    )

    expected_short = {step: step == 0 for step in range(21)}
    assert (
        classify_checkpoint_sweep(
            short_complete=expected_short,
            step0_long_complete=True,
            step20_long_complete=False,
        )["outcome"]
        == "long-control-drift"
    )
