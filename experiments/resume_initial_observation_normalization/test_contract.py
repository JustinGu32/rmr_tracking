from contract import (
    BASELINE_LONG,
    BASELINE_SHORT,
    LONG_COMPLETE,
    LONG_FAIL,
    LONG_MIXED,
    POST_LONG,
    POST_SHORT,
    SHORT_COMPLETE,
    SHORT_FAIL,
    SHORT_MIXED,
    classify_corrected_resume,
)


def _arms(post_short: str, post_long: str = LONG_FAIL) -> dict[str, str]:
    return {
        BASELINE_SHORT: SHORT_COMPLETE,
        BASELINE_LONG: LONG_FAIL,
        POST_SHORT: post_short,
        POST_LONG: post_long,
    }


def test_corrected_resume_preserves_short_and_long_still_fails():
    result = classify_corrected_resume(_arms(SHORT_COMPLETE))
    assert result["outcome"] == "corrected-order-preserves-short-long-incomplete"
    assert result["exact_arms"] is True
    assert result["arm_outcomes_valid"] is True


def test_corrected_resume_preserves_short_and_completes_long():
    result = classify_corrected_resume(_arms(SHORT_COMPLETE, LONG_COMPLETE))
    assert result["outcome"] == "corrected-order-preserves-short-long-complete"


def test_corrected_resume_still_loses_short():
    result = classify_corrected_resume(_arms(SHORT_FAIL))
    assert result["outcome"] == "corrected-order-still-immediate-loss"


def test_any_mixed_scientific_arm_is_mixed():
    assert classify_corrected_resume(_arms(SHORT_MIXED))["outcome"] == "mixed-corrected-order-evidence"
    assert classify_corrected_resume(_arms(SHORT_COMPLETE, LONG_MIXED))["outcome"] == "mixed-corrected-order-evidence"


def test_invalid_baseline_or_arm_set_is_invalid():
    bad_baseline = _arms(SHORT_COMPLETE)
    bad_baseline[BASELINE_SHORT] = SHORT_FAIL
    assert classify_corrected_resume(bad_baseline)["outcome"] == "invalid-execution"
    missing = _arms(SHORT_COMPLETE)
    del missing[POST_LONG]
    assert classify_corrected_resume(missing)["outcome"] == "invalid-execution"
