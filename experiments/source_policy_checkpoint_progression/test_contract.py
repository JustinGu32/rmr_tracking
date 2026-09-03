from __future__ import annotations

import unittest

from contract import classify_progression

SHORT_0 = "short_model_0"
LONG_0 = "long_model_0"
SHORT_500 = "short_model_500"
LONG_500 = "long_model_500"
SHORT_999 = "short_model_999"
LONG_999 = "long_model_999"
COMPLETE = "short-source-completes"
FAIL = "short-source-fails"
MIXED = "mixed-short-source-competence"


def _arms(long_0: str, long_500: str) -> dict[str, str]:
    return {
        SHORT_0: FAIL,
        LONG_0: long_0,
        SHORT_500: COMPLETE,
        LONG_500: long_500,
        SHORT_999: COMPLETE,
        LONG_999: FAIL,
    }


class CheckpointProgressionClassificationTest(unittest.TestCase):
    def test_long_run_never_acquires_shared_prefix(self) -> None:
        result = classify_progression(_arms(FAIL, FAIL))
        self.assertEqual(result["outcome"], "long-never-acquires-shared-prefix")
        self.assertTrue(result["endpoint_controls_valid"])

    def test_long_run_acquires_then_loses_shared_prefix(self) -> None:
        for early in ((COMPLETE, FAIL), (FAIL, COMPLETE), (COMPLETE, COMPLETE)):
            with self.subTest(early=early):
                result = classify_progression(_arms(*early))
                self.assertEqual(
                    result["outcome"],
                    "long-acquires-then-loses-shared-prefix",
                )

    def test_mixed_early_checkpoint_is_preserved(self) -> None:
        result = classify_progression(_arms(FAIL, MIXED))
        self.assertEqual(result["outcome"], "mixed-long-temporal-evidence")

    def test_wrong_or_missing_endpoint_controls_are_invalid(self) -> None:
        wrong_short = _arms(FAIL, FAIL)
        wrong_short[SHORT_999] = FAIL
        self.assertEqual(
            classify_progression(wrong_short)["outcome"],
            "invalid-execution",
        )
        wrong_long = _arms(FAIL, FAIL)
        wrong_long[LONG_999] = COMPLETE
        self.assertEqual(
            classify_progression(wrong_long)["outcome"],
            "invalid-execution",
        )
        self.assertEqual(
            classify_progression({SHORT_999: COMPLETE, LONG_999: FAIL})["outcome"],
            "invalid-execution",
        )


if __name__ == "__main__":
    unittest.main()
