from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run.py"


FAKE_EVALUATOR = r"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--normalizer-checkpoint-path", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--episodes", type=int, required=True)
args, _ = parser.parse_known_args()

actor = Path(args.checkpoint_path).resolve()
normalizer = Path(args.normalizer_checkpoint_path).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

regime = actor.parent.name
iteration = int(actor.stem.split("_")[-1])
outcome = (
    "short-source-completes"
    if regime == "short" and iteration in (500, 999)
    else "short-source-fails"
)
episodes = [
    {
        "initial_qpos_sha256": "same-qpos",
        "initial_qvel_sha256": "same-qvel",
        "initial_policy_observation_sha256": "same-observation",
    }
    for _ in range(args.episodes)
]
result = {
    "outcome": outcome,
    "inputs": {
        "checkpoint": {"path": str(actor), "sha256": sha256(actor)},
        "normalizer_checkpoint": {
            "path": str(normalizer),
            "sha256": sha256(normalizer),
        },
    },
    "evaluation_contract": {"normalizer_override_applied": actor != normalizer},
    "episodes": episodes,
    "all_numeric_finite": True,
}
(output / "result.json").write_text(json.dumps(result), encoding="utf-8")
"""


class CheckpointProgressionRunnerTest(unittest.TestCase):
    def test_fake_fixed_six_arm_run_selects_never_acquires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            short_run = root / "short"
            long_run = root / "long"
            short_run.mkdir()
            long_run.mkdir()
            for run_directory, prefix in ((short_run, b"short"), (long_run, b"long")):
                for iteration in (0, 500, 999):
                    (run_directory / f"model_{iteration}.pt").write_bytes(
                        prefix + str(iteration).encode()
                    )
            motion = root / "motion.npz"
            evaluator = root / "fake_evaluator.py"
            output = root / "output"
            motion.write_bytes(b"motion")
            evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--short-run-dir",
                    str(short_run),
                    "--long-run-dir",
                    str(long_run),
                    "--motion-file",
                    str(motion),
                    "--output-dir",
                    str(output),
                    "--evaluator",
                    str(evaluator),
                    "--episodes",
                    "3",
                    "--arm-timeout-seconds",
                    "5",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(
                result["outcome"],
                "long-never-acquires-shared-prefix",
            )
            self.assertEqual(
                [arm["arm"] for arm in result["arms"]],
                [
                    "short_model_0",
                    "long_model_0",
                    "short_model_500",
                    "long_model_500",
                    "short_model_999",
                    "long_model_999",
                ],
            )
            self.assertTrue(result["classification"]["endpoint_controls_valid"])
            self.assertTrue(
                result["evaluation_contract"][
                    "identical_initial_raw_observation_across_all_episodes_and_arms"
                ]
            )
            self.assertTrue((output / "arms.json").is_file())


if __name__ == "__main__":
    unittest.main()
