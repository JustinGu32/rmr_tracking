from __future__ import annotations

import unittest

from contract import (
    BASELINE_LONG,
    BASELINE_SHORT,
    LONG_COMPLETE,
    LONG_FAIL,
    LONG_MIXED,
    POST_1_LONG,
    POST_1_SHORT,
    POST_500_LONG,
    POST_500_SHORT,
    SHORT_COMPLETE,
    SHORT_FAIL,
    SHORT_MIXED,
    classify_warm_start,
)


def _normal_arms(
    *,
    post_1_short: str = SHORT_COMPLETE,
    post_500_short: str = SHORT_COMPLETE,
    post_1_long: str = LONG_FAIL,
    post_500_long: str = LONG_FAIL,
) -> dict[str, str]:
    return {
        BASELINE_SHORT: SHORT_COMPLETE,
        BASELINE_LONG: LONG_FAIL,
        POST_1_SHORT: post_1_short,
        POST_1_LONG: post_1_long,
        POST_500_SHORT: post_500_short,
        POST_500_LONG: post_500_long,
    }


class WarmStartClassificationTest(unittest.TestCase):
    def test_baseline_already_long_complete_stops_before_training(self) -> None:
        result = classify_warm_start(
            {
                BASELINE_SHORT: SHORT_COMPLETE,
                BASELINE_LONG: LONG_COMPLETE,
            },
            training_executed=False,
        )
        self.assertEqual(result["outcome"], "baseline-already-long-complete")
        self.assertTrue(result["baseline_valid"])

    def test_immediate_short_retention_loss_has_precedence(self) -> None:
        result = classify_warm_start(
            _normal_arms(
                post_1_short=SHORT_FAIL,
                post_500_short=SHORT_COMPLETE,
                post_500_long=LONG_COMPLETE,
            ),
            training_executed=True,
        )
        self.assertEqual(result["outcome"], "immediate-short-retention-loss")

    def test_delayed_short_retention_loss(self) -> None:
        result = classify_warm_start(
            _normal_arms(post_500_short=SHORT_FAIL),
            training_executed=True,
        )
        self.assertEqual(result["outcome"], "delayed-short-retention-loss")

    def test_long_competence_can_be_acquired_then_lost(self) -> None:
        result = classify_warm_start(
            _normal_arms(post_1_long=LONG_COMPLETE, post_500_long=LONG_FAIL),
            training_executed=True,
        )
        self.assertEqual(result["outcome"], "long-competence-lost")

    def test_retained_short_and_long_complete(self) -> None:
        result = classify_warm_start(
            _normal_arms(post_500_long=LONG_COMPLETE),
            training_executed=True,
        )
        self.assertEqual(result["outcome"], "retained-short-long-complete")

    def test_retained_short_but_long_incomplete(self) -> None:
        result = classify_warm_start(
            _normal_arms(),
            training_executed=True,
        )
        self.assertEqual(result["outcome"], "retained-short-long-incomplete")

    def test_any_mixed_scientific_arm_is_preserved(self) -> None:
        for arm, value in (
            (BASELINE_LONG, LONG_MIXED),
            (POST_1_SHORT, SHORT_MIXED),
            (POST_1_LONG, LONG_MIXED),
            (POST_500_SHORT, SHORT_MIXED),
            (POST_500_LONG, LONG_MIXED),
        ):
            with self.subTest(arm=arm):
                arms = _normal_arms()
                arms[arm] = value
                result = classify_warm_start(arms, training_executed=True)
                self.assertEqual(result["outcome"], "mixed-warm-start-evidence")

    def test_invalid_baseline_or_arm_set_is_invalid(self) -> None:
        bad_baseline = _normal_arms()
        bad_baseline[BASELINE_SHORT] = SHORT_FAIL
        self.assertEqual(
            classify_warm_start(bad_baseline, training_executed=True)["outcome"],
            "invalid-execution",
        )
        missing = _normal_arms()
        del missing[POST_500_LONG]
        self.assertEqual(
            classify_warm_start(missing, training_executed=True)["outcome"],
            "invalid-execution",
        )
        wrong_label = _normal_arms()
        wrong_label[POST_1_LONG] = SHORT_COMPLETE
        self.assertEqual(
            classify_warm_start(wrong_label, training_executed=True)["outcome"],
            "invalid-execution",
        )
        self.assertEqual(
            classify_warm_start(
                {
                    BASELINE_SHORT: SHORT_COMPLETE,
                    BASELINE_LONG: LONG_FAIL,
                },
                training_executed=False,
            )["outcome"],
            "invalid-execution",
        )


if __name__ == "__main__":
    unittest.main()
