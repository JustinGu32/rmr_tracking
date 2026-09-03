import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from run import _training_command, _validate_training_log_iterations


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        trainer=Path("/repo/experiments/resume_initial_observation_normalization/train.py"),
        task="Tracking-Flat-G1-v0",
        long_motion_file=Path("/motions/long.npz"),
        run_name="E009_test",
        num_envs=4096,
        training_updates=1,
        train_seed=42,
        ppo_output="delta-all",
        device="cuda:0",
        source_checkpoint=Path("/logs/source/model_500.pt"),
        headless=True,
    )


def test_training_command_is_exact_one_update_resume():
    command = _training_command(_args())
    assert command[0].endswith("python") or command[0].endswith("python3")
    assert command[1] == "/repo/experiments/resume_initial_observation_normalization/train.py"
    assert command[command.index("--max_iterations") + 1] == "1"
    assert command[command.index("--num_envs") + 1] == "4096"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--load_run") + 1] == "source"
    assert command[command.index("--checkpoint") + 1] == "model_500.pt"
    assert command[-1] == "--headless"


def test_training_log_requires_only_native_iteration_500():
    good = "header\n Learning iteration 500/501 \nfooter\n"
    assert _validate_training_log_iterations(good) == []
    assert _validate_training_log_iterations(
        "Learning iteration 500/501\nLearning iteration 501/501\n"
    ) == ["expected exactly native iteration 500 once, observed [500, 501]"]
    assert _validate_training_log_iterations("header only") == [
        "expected exactly native iteration 500 once, observed []"
    ]


FAKE_TRAINER = r"""
import argparse
import os
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
args, _ = parser.parse_known_args()
root = Path(os.environ["DIFFSIM_CORRECTED_RESUME_LOG_ROOT"])
source = torch.load(root / args.load_run / args.checkpoint, map_location="cpu", weights_only=False)
run = root / f"2099-01-01_00-00-00_{args.run_name}"
(run / "params").mkdir(parents=True)
(run / "params" / "env.yaml").write_text("motion: long\n", encoding="utf-8")
(run / "params" / "agent.yaml").write_text("resume: true\n", encoding="utf-8")
payload = {
    "iter": 500,
    "infos": None,
    "model_state_dict": {"weight": torch.ones(2)},
    "obs_norm_state_dict": {
        "_mean": torch.zeros(2), "_std": torch.ones(2), "_var": torch.ones(2),
        "count": source["obs_norm_state_dict"]["count"] + 4096 * 24,
    },
    "privileged_obs_norm_state_dict": {
        "_mean": torch.zeros(3), "_std": torch.ones(3), "_var": torch.ones(3),
        "count": source["privileged_obs_norm_state_dict"]["count"] + 4096 * 24,
    },
    "optimizer_state_dict": {"state": {0: {"step": torch.tensor(120)}}, "param_groups": [{}]},
}
torch.save(payload, run / "model_500.pt")
print("[RESUME-INITIAL-NORMALIZATION] metadata=fake", flush=True)
print("Learning iteration 500/501", flush=True)
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
motion = Path(args.motion_file).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

short = args.outcome_label_set == "short-control"
outcome = "short-source-completes" if short else "source-fails-exact-long"
episodes = [{
    "initial_qpos_sha256": "same-qpos",
    "initial_qvel_sha256": "same-qvel",
    "initial_policy_observation_sha256": "same-observation",
} for _ in range(args.episodes)]
result = {
    "outcome": outcome,
    "inputs": {
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "normalizer_checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "motion": {"path": str(motion), "sha256": sha(motion)},
    },
    "evaluation_contract": {"normalizer_override_applied": False},
    "classification": {"contract_valid": True},
    "episodes": episodes,
    "all_numeric_finite": True,
}
(output / "result.json").write_text(json.dumps(result), encoding="utf-8")
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_process_level_fake_executes_exact_four_arm_protocol(tmp_path: Path):
    training_logs = tmp_path / "training_logs"
    source_run = training_logs / "source"
    source_run.mkdir(parents=True)
    source_checkpoint = source_run / "model_500.pt"
    source_payload = {
        "iter": 500,
        "infos": None,
        "model_state_dict": {"weight": torch.zeros(2)},
        "obs_norm_state_dict": {
            "_mean": torch.zeros(2), "_std": torch.ones(2), "_var": torch.ones(2),
            "count": torch.tensor(10),
        },
        "privileged_obs_norm_state_dict": {
            "_mean": torch.zeros(3), "_std": torch.ones(3), "_var": torch.ones(3),
            "count": torch.tensor(10),
        },
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(100)}}, "param_groups": [{}]},
    }
    torch.save(source_payload, source_checkpoint)
    short_motion = tmp_path / "short.npz"
    long_motion = tmp_path / "long.npz"
    short_motion.write_bytes(b"short")
    long_motion.write_bytes(b"long")
    control_checkpoint = tmp_path / "control.pt"
    control_checkpoint.write_bytes(b"unmodified")
    control_result = tmp_path / "control.json"
    control_result.write_text(json.dumps({
        "verdict": "pass", "selected_outcome": "immediate-short-retention-loss"
    }), encoding="utf-8")
    trainer = tmp_path / "trainer.py"
    evaluator = tmp_path / "evaluator.py"
    trainer.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
    evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")
    output = tmp_path / "output"
    runner = Path(__file__).resolve().parent / "run.py"
    completed = subprocess.run([
        sys.executable, str(runner),
        "--source-checkpoint", str(source_checkpoint),
        "--source-checkpoint-sha256", _sha(source_checkpoint),
        "--short-motion-file", str(short_motion), "--short-motion-sha256", _sha(short_motion),
        "--long-motion-file", str(long_motion), "--long-motion-sha256", _sha(long_motion),
        "--unmodified-control-checkpoint", str(control_checkpoint),
        "--unmodified-control-checkpoint-sha256", _sha(control_checkpoint),
        "--unmodified-control-result", str(control_result),
        "--unmodified-control-result-sha256", _sha(control_result),
        "--output-dir", str(output), "--trainer", str(trainer),
        "--evaluator", str(evaluator), "--training-log-root", str(training_logs),
        "--run-name", "fake_corrected", "--training-updates", "1", "--num-envs", "4096",
        "--train-seed", "42", "--eval-episodes", "3", "--eval-seed", "0",
        "--training-timeout-seconds", "15", "--evaluation-timeout-seconds", "15", "--headless",
    ], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "corrected-order-preserves-short-long-incomplete"
    assert [record["arm"] for record in result["arms"]] == [
        "baseline_short", "baseline_long", "post_update_1_short", "post_update_1_long"
    ]
    assert result["training"]["package_metadata"] == {
        "iter": 500,
        "actor_normalizer_count": 98314,
        "privileged_normalizer_count": 98314,
        "optimizer_steps": [120],
        "model_entries": 1,
    }
    assert result["evaluation_contract"]["identical_initial_raw_observation"] is True
