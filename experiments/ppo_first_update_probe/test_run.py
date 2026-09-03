import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from run import _training_command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_command_is_one_corrected_native_resume():
    args = argparse.Namespace(
        trainer=Path("/repo/experiments/ppo_first_update_probe/train.py"),
        task="Tracking-Flat-G1-v0",
        long_motion_file=Path("/motions/long.npz"),
        run_name="E010_test",
        num_envs=4096,
        training_updates=1,
        train_seed=42,
        ppo_output="delta-all",
        device="cuda:0",
        source_checkpoint=Path("/logs/source/model_500.pt"),
        headless=True,
    )
    command = _training_command(args)
    assert command[command.index("--max_iterations") + 1] == "1"
    assert command[command.index("--num_envs") + 1] == "4096"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--sampling") + 1] == "adaptive"
    assert command[command.index("--load_run") + 1] == "source"
    assert command[command.index("--checkpoint") + 1] == "model_500.pt"
    assert command[-1] == "--headless"


FAKE_TRAINER = r"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
args, _ = parser.parse_known_args()
root = Path(os.environ["DIFFSIM_FIRST_UPDATE_LOG_ROOT"])
probe = Path(os.environ["DIFFSIM_FIRST_UPDATE_PROBE_DIR"])
probe.mkdir(parents=True, exist_ok=True)
source = torch.load(root / args.load_run / args.checkpoint, map_location="cpu", weights_only=False)
run = root / f"2099-01-01_00-00-00_{args.run_name}"
run.mkdir(parents=True)
checkpoint_dir = probe / "checkpoints"
checkpoint_dir.mkdir()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records = []
for step in range(21):
    package = {
        "iter": 500,
        "model_state_dict": {"weight": torch.tensor([float(step)])},
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(10020 + step)}},
            "param_groups": [{"lr": 1e-5}],
        },
        "obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
        "privileged_obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
    }
    path = checkpoint_dir / f"model_step_{step:02d}.pt"
    torch.save(package, path)
    records.append({
        "step": step,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
        "optimizer_steps": [10020 + step],
    })

probe_result = {
    "schema_version": 1,
    "complete": True,
    "runner_completed": True,
    "measurement_only": True,
    "optimizer_trace": [{"optimizer_step": step} for step in range(1, 21)],
    "frozen_gradient_analysis": {
        "state_identity_before": {"same": "yes"},
        "state_identity_after": {"same": "yes"},
        "rng_identity_before": {"same": "yes"},
        "rng_identity_after": {"same": "yes"},
    },
    "checkpoints": records,
}
(probe / "probe_result.json").write_text(json.dumps(probe_result), encoding="utf-8")
final = torch.load(checkpoint_dir / "model_step_20.pt", map_location="cpu", weights_only=False)
torch.save(final, run / "model_500.pt")
print("[RESUME-INITIAL-NORMALIZATION] metadata=fake", flush=True)
for step in range(1, 21):
    print(f"[FIRST-UPDATE-PROBE] step={step:02d} fake", flush=True)
print("Learning iteration 500/501", flush=True)
"""


FAKE_EVALUATOR = r"""
import argparse
import hashlib
import json
import re
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
motion = Path(args.motion_file).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True)
step = int(re.search(r"step_(\d+)", checkpoint.name).group(1))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

short = args.outcome_label_set == "short-control"
complete = short and step == 0
if complete:
    outcome = "short-source-completes"
    survival = 125
elif short:
    outcome = "short-source-fails"
    survival = 37
else:
    outcome = "source-fails-exact-long"
    survival = 126 if step == 0 else 37
episodes = [{
    "initial_qpos_sha256": "same-qpos",
    "initial_qvel_sha256": "same-qvel",
    "initial_policy_observation_sha256": "same-observation",
} for _ in range(args.episodes)]
payload = {
    "outcome": outcome,
    "inputs": {
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "normalizer_checkpoint": {"path": str(normalizer), "sha256": sha(normalizer)},
        "motion": {"path": str(motion), "sha256": sha(motion)},
    },
    "classification": {
        "contract_valid": True,
        "survival_steps": [survival] * args.episodes,
        "all_episodes_complete": complete,
    },
    "evaluation_contract": {"normalizer_override_applied": False},
    "episodes": episodes,
    "all_numeric_finite": True,
}
(output / "result.json").write_text(json.dumps(payload), encoding="utf-8")
"""


def test_fake_process_executes_all_21_checkpoints_and_two_long_endpoints(
    tmp_path: Path,
):
    logs = tmp_path / "logs"
    source_dir = logs / "source"
    source_dir.mkdir(parents=True)
    source = source_dir / "model_500.pt"
    torch.save(
        {
            "iter": 500,
            "model_state_dict": {"weight": torch.tensor([0.0])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(10020)}},
                "param_groups": [{"lr": 1e-3}],
            },
            "obs_norm_state_dict": {"count": torch.tensor(10)},
            "privileged_obs_norm_state_dict": {"count": torch.tensor(10)},
        },
        source,
    )
    short_motion = tmp_path / "short.npz"
    long_motion = tmp_path / "long.npz"
    short_motion.write_bytes(b"short")
    long_motion.write_bytes(b"long")
    e009 = tmp_path / "e009.json"
    e009.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "selected_outcome": "corrected-order-still-immediate-loss",
            }
        ),
        encoding="utf-8",
    )
    trainer = tmp_path / "trainer.py"
    evaluator = tmp_path / "evaluator.py"
    trainer.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
    evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")
    output = tmp_path / "result"
    runner = Path(__file__).resolve().parent / "run.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--source-checkpoint",
            str(source),
            "--source-checkpoint-sha256",
            _sha256(source),
            "--short-motion-file",
            str(short_motion),
            "--short-motion-sha256",
            _sha256(short_motion),
            "--long-motion-file",
            str(long_motion),
            "--long-motion-sha256",
            _sha256(long_motion),
            "--e009-audit",
            str(e009),
            "--e009-audit-sha256",
            _sha256(e009),
            "--output-dir",
            str(output),
            "--trainer",
            str(trainer),
            "--evaluator",
            str(evaluator),
            "--training-log-root",
            str(logs),
            "--run-name",
            "fake_probe",
            "--training-timeout-seconds",
            "15",
            "--evaluation-timeout-seconds",
            "15",
            "--headless",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "optimizer-step-localized-monotonic-loss"
    assert result["classification"]["first_loss_step"] == 1
    assert len(result["arms"]) == 23
    assert result["evaluation_contract"]["initial_reset_identity"] is True
    assert (
        result["training"]["package_metadata"][
            "step20_model_optimizer_normalizers_equal_native_final"
        ]
        is True
    )
