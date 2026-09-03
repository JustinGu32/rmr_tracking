from __future__ import annotations

import unittest

from contract import classify_crossover

SHORT_SHORT = "short_actor__short_normalizer"
SHORT_LONG = "short_actor__long_normalizer"
LONG_SHORT = "long_actor__short_normalizer"
LONG_LONG = "long_actor__long_normalizer"
COMPLETE = "short-source-completes"
FAIL = "short-source-fails"
MIXED = "mixed-short-source-competence"


def _arms(short_long: str, long_short: str) -> dict[str, str]:
    return {
        SHORT_SHORT: COMPLETE,
        SHORT_LONG: short_long,
        LONG_SHORT: long_short,
        LONG_LONG: FAIL,
    }


class CrossoverClassificationTest(unittest.TestCase):
    def test_actor_identity_controls_competence(self) -> None:
        result = classify_crossover(_arms(COMPLETE, FAIL))
        self.assertEqual(result["outcome"], "actor-component-dominant")
        self.assertTrue(result["controls_valid"])

    def test_normalizer_identity_controls_competence(self) -> None:
        result = classify_crossover(_arms(FAIL, COMPLETE))
        self.assertEqual(result["outcome"], "normalizer-component-dominant")

    def test_either_short_component_rescues(self) -> None:
        result = classify_crossover(_arms(COMPLETE, COMPLETE))
        self.assertEqual(result["outcome"], "either-short-component-rescues")

    def test_both_short_components_are_required(self) -> None:
        result = classify_crossover(_arms(FAIL, FAIL))
        self.assertEqual(result["outcome"], "both-short-components-required")

    def test_mixed_cross_arm_is_preserved(self) -> None:
        result = classify_crossover(_arms(MIXED, FAIL))
        self.assertEqual(result["outcome"], "mixed-component-crossover")

    def test_invalid_or_wrong_controls_are_invalid(self) -> None:
        invalid = _arms(COMPLETE, FAIL)
        invalid[SHORT_SHORT] = FAIL
        self.assertEqual(
            classify_crossover(invalid)["outcome"],
            "invalid-execution",
        )
        self.assertEqual(
            classify_crossover({SHORT_SHORT: COMPLETE})["outcome"],
            "invalid-execution",
        )


if __name__ == "__main__":
    unittest.main()
