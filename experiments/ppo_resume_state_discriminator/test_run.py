import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from design import ARM_NAMES
from run import _training_command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_command_is_one_exact_adaptive_resume():
    args = argparse.Namespace(
        trainer=Path("/repo/experiments/ppo_resume_state_discriminator/train.py"),
        task="Tracking-Flat-G1-v0",
        long_motion_file=Path("/motions/long.npz"),
        run_name="E011_test",
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

ARMS = (
    "restored_adam__synced_scheduler",
    "reset_adam__fresh_scheduler",
    "reset_adam__synced_scheduler",
    "restored_adam__fresh_scheduler",
)
parser = argparse.ArgumentParser()
parser.add_argument("--run_name", required=True)
parser.add_argument("--load_run", required=True)
parser.add_argument("--checkpoint", required=True)
args, _ = parser.parse_known_args()
root = Path(os.environ["DIFFSIM_FIRST_UPDATE_LOG_ROOT"])
output = Path(os.environ["DIFFSIM_RESUME_STATE_DISCRIMINATOR_DIR"])
output.mkdir(parents=True, exist_ok=True)
run = root / f"2099-01-01_00-00-00_{args.run_name}"
run.mkdir(parents=True)
checkpoint_dir = output / "checkpoints"
checkpoint_dir.mkdir()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def package(weight, step, lr):
    return {
        "iter": 500,
        "model_state_dict": {"weight": torch.tensor([weight])},
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(step)}},
            "param_groups": [{"lr": lr}],
        },
        "obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
        "privileged_obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
    }

step0_path = checkpoint_dir / "model_step_00.pt"
torch.save(package(0.0, 10020, 2.25e-5), step0_path)
step0 = {
    "path": str(step0_path),
    "bytes": step0_path.stat().st_size,
    "sha256": sha(step0_path),
    "optimizer_steps": [10020],
}
branches = []
for index, arm in enumerate(ARMS, start=1):
    reset = arm.startswith("reset_adam__")
    synced = arm.endswith("synced_scheduler")
    step = 1 if reset else 10021
    lr = 3.375e-5 if synced else 1.5e-3
    path = checkpoint_dir / f"model_{arm}_step_01.pt"
    torch.save(package(float(index), step, lr), path)
    branches.append({
        "arm": {
            "name": arm,
            "reset_adam": reset,
            "synchronize_scheduler": synced,
        },
        "pre_step": {
            "indices_sha256": "same-indices",
            "gradient": {"post_clip_sha256": "same-gradient"},
            "optimizer_state_steps_before": [] if reset else [10020],
            "optimizer_state_steps_after": [step],
        },
        "checkpoint": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "optimizer_steps": [step],
        },
    })
factorial = {
    "schema_version": 1,
    "complete": True,
    "runner_completed": True,
    "step0_checkpoint": step0,
    "branches": branches,
}
(output / "factorial_result.json").write_text(json.dumps(factorial), encoding="utf-8")
native = next(branch for branch in branches if branch["arm"]["name"] == ARMS[-1])
native_package = torch.load(native["checkpoint"]["path"], map_location="cpu", weights_only=False)
torch.save(native_package, run / "model_500.pt")
print("[RESUME-INITIAL-NORMALIZATION] metadata=fake", flush=True)
for arm in ARMS:
    print(f"[RESUME-STATE-DISCRIMINATOR] arm={arm} lr=fake", flush=True)
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
parser.add_argument("--render", action="store_true")
args, _ = parser.parse_known_args()
checkpoint = Path(args.checkpoint_path).resolve()
normalizer = Path(args.normalizer_checkpoint_path).resolve()
motion = Path(args.motion_file).resolve()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

name = checkpoint.name
short = args.outcome_label_set == "short-control"
step0 = name == "model_step_00.pt"
scheduler_safe = (
    "restored_adam__synced_scheduler" in name
    or "reset_adam__synced_scheduler" in name
)
complete = short and (step0 or scheduler_safe)
if complete:
    outcome = "short-source-completes"
    survival = 125
elif short:
    outcome = "short-source-fails"
    survival = 43 if "restored_adam__fresh_scheduler" in name else 50
else:
    outcome = "source-fails-exact-long"
    survival = 126 if step0 or scheduler_safe else 43
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
if args.render:
    (output / "source_rollout_episode0.mp4").write_bytes(b"fake-mp4")
    (output / "source_rollout_episode0_contact_sheet.png").write_bytes(b"fake-png")
"""


def test_fake_process_executes_complete_factorial_and_strict_evaluations(
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
                "param_groups": [{"lr": 2.25e-5}],
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
    e010 = tmp_path / "e010.json"
    e010.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "selected_outcome": "optimizer-step-localized-monotonic-loss",
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
            "--e010-audit",
            str(e010),
            "--e010-audit-sha256",
            _sha256(e010),
            "--output-dir",
            str(output),
            "--trainer",
            str(trainer),
            "--evaluator",
            str(evaluator),
            "--training-log-root",
            str(logs),
            "--run-name",
            "fake_factorial",
            "--training-timeout-seconds",
            "15",
            "--evaluation-timeout-seconds",
            "15",
            "--render-short-arms",
            "--headless",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "scheduler-synchronization-preserves"
    assert len(result["evaluations"]) == 10
    assert result["evaluation_contract"]["initial_reset_identity"] is True
    assert tuple(result["training_contract"]["factorial_arms"]) == ARM_NAMES
    assert len(list((output / "evaluations").glob("*/source_eval/*.mp4"))) == 5
