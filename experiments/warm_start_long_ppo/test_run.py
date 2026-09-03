from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run.py"


FAKE_TRAINER = r"""
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
args, _ = parser.parse_known_args()
root = Path(os.environ["DIFFSIM_WARMSTART_LOG_ROOT"])
run = root / f"2099-01-01_00-00-00_{args.run_name}"
(run / "params").mkdir(parents=True)
(run / "params" / "env.yaml").write_text("motion: long\n", encoding="utf-8")
(run / "params" / "agent.yaml").write_text("resume: true\n", encoding="utf-8")
(run / "model_500.pt").write_text("post1", encoding="utf-8")
(run / "model_999.pt").write_text("post500", encoding="utf-8")
print("fake training update 1", flush=True)
print("fake training update 500", flush=True)
"""


FAKE_EVALUATOR = r"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--motion-file", required=True)
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--normalizer-checkpoint-path", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--outcome-label-set", required=True)
args, _ = parser.parse_known_args()

checkpoint = Path(args.checkpoint_path).resolve()
normalizer = Path(args.normalizer_checkpoint_path).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

is_short = args.outcome_label_set == "short-control"
content = checkpoint.read_text(encoding="utf-8")
if is_short:
    outcome = "short-source-completes"
else:
    outcome = (
        "source-completes-exact-long"
        if content == "baseline-long-complete"
        else "source-fails-exact-long"
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
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "normalizer_checkpoint": {
            "path": str(normalizer),
            "sha256": sha256(normalizer),
        },
    },
    "evaluation_contract": {
        "normalizer_override_applied": checkpoint != normalizer,
    },
    "episodes": episodes,
    "all_numeric_finite": True,
    "fake_checkpoint_content": content,
}
(output / "result.json").write_text(json.dumps(result), encoding="utf-8")
"""


class WarmStartRunnerTest(unittest.TestCase):
    def test_fake_run_preserves_baseline_and_uses_exact_resume_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_logs = root / "training_logs"
            training_logs.mkdir()
            source_run = training_logs / "source_run"
            source_run.mkdir()
            source_checkpoint = source_run / "model_500.pt"
            source_checkpoint.write_text("baseline", encoding="utf-8")
            source_sha = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
            short_motion = root / "short_motion.npz"
            long_motion = root / "long_motion.npz"
            short_motion.write_bytes(b"short")
            long_motion.write_bytes(b"long")
            trainer = root / "fake_trainer.py"
            evaluator = root / "fake_evaluator.py"
            trainer.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
            evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")
            output = root / "output"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--source-checkpoint",
                    str(source_checkpoint),
                    "--short-motion-file",
                    str(short_motion),
                    "--long-motion-file",
                    str(long_motion),
                    "--output-dir",
                    str(output),
                    "--trainer",
                    str(trainer),
                    "--evaluator",
                    str(evaluator),
                    "--training-log-root",
                    str(training_logs),
                    "--run-name",
                    "fake_warm_start",
                    "--training-updates",
                    "500",
                    "--num-envs",
                    "4096",
                    "--train-seed",
                    "42",
                    "--eval-episodes",
                    "3",
                    "--eval-seed",
                    "0",
                    "--training-timeout-seconds",
                    "10",
                    "--evaluation-timeout-seconds",
                    "10",
                    "--render-evaluations",
                    "--headless",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["outcome"], "retained-short-long-incomplete")
            self.assertEqual(
                [arm["arm"] for arm in result["arms"]],
                [
                    "baseline_short",
                    "baseline_long",
                    "post_update_1_short",
                    "post_update_1_long",
                    "post_update_500_short",
                    "post_update_500_long",
                ],
            )
            command = result["training"]["command"]
            self.assertIn("--resume", command)
            self.assertIn("True", command)
            self.assertIn("--load_run", command)
            self.assertIn(source_run.name, command)
            self.assertIn("--checkpoint", command)
            self.assertIn(source_checkpoint.name, command)
            self.assertIn("--motion_file", command)
            self.assertIn(str(long_motion.resolve()), command)
            self.assertIn("--max_iterations", command)
            self.assertIn("500", command)
            self.assertIn("--num_envs", command)
            self.assertIn("4096", command)
            self.assertEqual(
                source_sha, hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
            )
            self.assertTrue(result["training"]["source_checkpoint_unchanged"])
            self.assertTrue(
                result["evaluation_contract"]["all_initial_hashes_identical"]
            )
            self.assertTrue((output / "training" / "combined.log").is_file())
            self.assertTrue((output / "arms.json").is_file())

    def test_long_complete_baseline_stops_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_logs = root / "training_logs"
            source_run = training_logs / "source_run"
            source_run.mkdir(parents=True)
            source_checkpoint = source_run / "model_500.pt"
            source_checkpoint.write_text("baseline-long-complete", encoding="utf-8")
            short_motion = root / "short_motion.npz"
            long_motion = root / "long_motion.npz"
            short_motion.write_bytes(b"short")
            long_motion.write_bytes(b"long")
            trainer = root / "fake_trainer.py"
            evaluator = root / "fake_evaluator.py"
            trainer.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
            evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")
            output = root / "output"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--source-checkpoint",
                    str(source_checkpoint),
                    "--short-motion-file",
                    str(short_motion),
                    "--long-motion-file",
                    str(long_motion),
                    "--output-dir",
                    str(output),
                    "--trainer",
                    str(trainer),
                    "--evaluator",
                    str(evaluator),
                    "--training-log-root",
                    str(training_logs),
                    "--run-name",
                    "must_not_train",
                    "--evaluation-timeout-seconds",
                    "10",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["outcome"], "baseline-already-long-complete")
            self.assertFalse(result["training"]["executed"])
            self.assertEqual(
                [arm["arm"] for arm in result["arms"]],
                ["baseline_short", "baseline_long"],
            )
            self.assertFalse((output / "training" / "combined.log").exists())


if __name__ == "__main__":
    unittest.main()
