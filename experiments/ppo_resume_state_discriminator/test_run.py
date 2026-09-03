import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from design import ARM_NAMES
from run import (
    DEFAULT_EVALUATOR,
    DEFAULT_TRAINING_LOG_ROOT,
    LOCAL_WHOLE_BODY_SOURCE,
    _artifact_errors,
    _child_environment,
    _git_output,
    _integrity_errors,
    _reported_source_checkpoint,
    _scientific_preflight_errors,
    _training_command,
)


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
    assert command[command.index("--load_run") + 1] == "^source$"
    assert command[command.index("--checkpoint") + 1] == r"^model_500\.pt$"
    assert command[-1] == "--headless"


def test_child_environment_forces_pinned_worktree_source_first(tmp_path: Path):
    environment, _ = _child_environment(
        training_log_root=tmp_path / "logs",
        discriminator_dir=tmp_path / "discriminator",
    )
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(
        LOCAL_WHOLE_BODY_SOURCE.resolve()
    )


FAKE_TRAINER = r"""
import argparse
import copy
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

def sha_tensor(value):
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def update_digest(digest, value):
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(sha_tensor(value).encode())
    elif isinstance(value, dict):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            update_digest(digest, key)
            update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            update_digest(digest, item)
    elif isinstance(value, (str, int, float, bool, type(None))):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        digest.update(type(value).__qualname__.encode())
        digest.update(repr(value).encode())

def state_digest(value):
    digest = hashlib.sha256()
    update_digest(digest, value)
    return digest.hexdigest()

def identity(value):
    return {
        "model": state_digest(value["model_state_dict"]),
        "optimizer": state_digest(value["optimizer_state_dict"]),
        "actor_normalizer": state_digest(value["obs_norm_state_dict"]),
        "critic_normalizer": state_digest(value["privileged_obs_norm_state_dict"]),
    }

def artifact(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}

def package(weight, step, lr):
    return {
        "iter": 500,
        "infos": None,
        "model_state_dict": {"weight": torch.tensor([weight])},
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(step)}},
            "param_groups": [{"lr": lr}],
        },
        "obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
        "privileged_obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
    }

step0_path = checkpoint_dir / "model_step_00.pt"
step0_package = package(0.0, 10020, 2.25e-5)
torch.save(step0_package, step0_path)
step0 = {**artifact(step0_path), "optimizer_steps": [10020], "optimizer_learning_rates": [2.25e-5]}
baseline_identity = identity(step0_package)
reset_optimizer = copy.deepcopy(step0_package["optimizer_state_dict"])
reset_optimizer["state"] = {}

phases = (torch.arange(4096).remainder(272)).repeat(24, 1)
timeouts = torch.zeros((24, 4096), dtype=torch.bool)
rollout = {
    "phases": phases,
    "timeouts": timeouts,
    "native_permutation": torch.arange(4096 * 24),
    "observations": torch.zeros((24, 4096, 1)),
    "privileged_observations": torch.zeros((24, 4096, 1)),
    "actions": torch.zeros((24, 4096, 1)),
    "rewards": torch.zeros((24, 4096, 1)),
    "dones": torch.zeros((24, 4096, 1), dtype=torch.bool),
    "values": torch.zeros((24, 4096, 1)),
    "returns": torch.zeros((24, 4096, 1)),
    "advantages": torch.zeros((24, 4096, 1)),
    "actions_log_prob": torch.zeros((24, 4096, 1)),
    "mu": torch.zeros((24, 4096, 1)),
    "sigma": torch.ones((24, 4096, 1)),
}
rollout_path = output / "factorial_rollout_tensors.pt"
torch.save(rollout, rollout_path)
initial_histogram = torch.bincount(phases[0], minlength=272)
bins = torch.clamp((phases.flatten() * 6) // 272, min=0, max=5)
phase_bin_counts = torch.bincount(bins, minlength=6)
first_indices = rollout["native_permutation"][: 4096 * 24 // 4]
first_bins = torch.clamp((phases.flatten()[first_indices] * 6) // 272, min=0, max=5)
common_pre_step = {
    "epoch": 0,
    "mini_batch": 0,
    "global_step": 1,
    "indices_sha256": sha_tensor(first_indices),
    "sample_count": 4096 * 24 // 4,
    "phase_bin_counts": torch.bincount(first_bins, minlength=6).tolist(),
    "advantage": {"mean": 0.0},
    "ratio": {"mean": 1.0},
    "log_ratio": {"mean": 0.0},
    "analytic_kl": {"mean": 0.00029},
    "approximate_kl": 0.0,
    "clipped_fraction": 0.0,
    "forward_sha256": {
        "log_probability": "2" * 64,
        "mean": "3" * 64,
        "sigma": "4" * 64,
        "entropy": "5" * 64,
        "value": "6" * 64,
    },
    "loss": {"surrogate": 0.2, "value": 0.4, "entropy": 0.6, "total": 0.597},
    "gradient": {"post_clip_sha256": "7" * 64},
}
common_rng = {"cpu": "8" * 64, "cuda": []}
native_losses = {"value_function": 0.02, "surrogate": 0.01, "entropy": 0.03}
branches = []
for index, arm in enumerate(ARMS, start=1):
    reset = arm.startswith("reset_adam__")
    synced = arm.endswith("synced_scheduler")
    step = 1 if reset else 10021
    lr = 3.375e-5 if synced else 1.5e-3
    path = checkpoint_dir / f"model_{arm}_step_01.pt"
    branch_package = package(float(index), step, lr)
    torch.save(branch_package, path)
    post_intervention_identity = dict(baseline_identity)
    if reset:
        post_intervention_identity["optimizer"] = state_digest(reset_optimizer)
    pre_step = {
        **common_pre_step,
        "applied_learning_rate": lr,
        "scheduler_learning_rate": lr,
        "optimizer_state_steps_before": [] if reset else [10020],
        "optimizer_state_steps_after": [step],
    }
    branches.append({
        "arm": {
            "name": arm,
            "reset_adam": reset,
            "synchronize_scheduler": synced,
        },
        "intervention": {
            "arm": {
                "name": arm,
                "reset_adam": reset,
                "synchronize_scheduler": synced,
            },
            "restored_optimizer_learning_rate": 2.25e-5,
            "fresh_scheduler_learning_rate": 1.0e-3,
            "scheduler_learning_rate_after_intervention": 2.25e-5 if synced else 1.0e-3,
            "optimizer_group_learning_rates_after_intervention": [2.25e-5],
            "optimizer_state_entries_before_intervention": 1,
            "optimizer_state_entries_after_intervention": 0 if reset else 1,
        },
        "pre_intervention_identity": baseline_identity,
        "post_intervention_identity": post_intervention_identity,
        "pre_step": pre_step,
        "post_step_identity": identity(branch_package),
        "post_step_rng": common_rng,
        "checkpoint": {
            **artifact(path),
            "optimizer_steps": [step],
            "optimizer_learning_rates": [lr],
        },
        "native_codepath_loss_dict_divided_by_configured_20": native_losses,
    })
native = next(branch for branch in branches if branch["arm"]["name"] == ARMS[-1])
factorial = {
    "schema_version": 1,
    "complete": True,
    "factorial": True,
    "single_shared_rollout": True,
    "single_native_partition_per_arm": True,
    "runner_completed": True,
    "current_learning_iteration": 500,
    "normalizer_counts": {"actor": 10 + 4096 * 24, "critic": 10 + 4096 * 24},
    "baseline_state_identity": baseline_identity,
    "baseline_rng": common_rng,
    "fresh_scheduler_learning_rate": 1.0e-3,
    "restored_optimizer_learning_rate": 2.25e-5,
    "native_permutation_sha256": sha_tensor(rollout["native_permutation"]),
    "rollout": {
        "samples": 4096 * 24,
        "rollout_steps": 24,
        "environments": 4096,
        "reference_states": 272,
        "adaptive_bin_count": 6,
        "distinct_phases": int(torch.unique(phases).numel()),
        "initial_distinct_phases": int((initial_histogram > 0).sum().item()),
        "initial_phase_histogram": initial_histogram.tolist(),
        "phase_bin_counts": phase_bin_counts.tolist(),
        "done_count": 0,
        "timeout_count": 0,
        "hard_termination_count": 0,
        "advantage": {"mean": 0.0},
        "advantage_by_phase_bin": [],
    },
    "rollout_tensors": artifact(rollout_path),
    "step0_checkpoint": step0,
    "branches": branches,
    "retained_outer_state_arm": ARMS[-1],
    "final_state_identity": native["post_step_identity"],
    "retained_outer_scheduler_learning_rate": 1.5e-3,
    "retained_outer_optimizer_learning_rates": [1.5e-3],
    "retained_outer_rng": common_rng,
    "outer_state_after_super": {
        **native["post_step_identity"],
        "scheduler_learning_rate": 1.5e-3,
        "optimizer_learning_rates": [1.5e-3],
        "optimizer_steps": [10021],
        "rng": common_rng,
    },
}
(output / "factorial_result.json").write_text(json.dumps(factorial), encoding="utf-8")
native_package = torch.load(native["checkpoint"]["path"], map_location="cpu", weights_only=False)
torch.save(native_package, run / "model_500.pt")
source = next(path for path in root.glob("*/model_500.pt") if path.parent != run)
print(f"[INFO]: Loading model checkpoint from: {source.resolve()}", flush=True)
whole_body_source = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])
print(
    f"[RESUME-STATE-CODE] whole_body_tracking={(whole_body_source / 'whole_body_tracking' / '__init__.py').resolve()}",
    flush=True,
)
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
import numpy as np

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
episodes_path = output / "episodes.json"
trajectory_path = output / "trajectory.npz"
episodes_path.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
np.savez_compressed(trajectory_path, step=np.arange(args.episodes))
artifact_paths = [episodes_path, trajectory_path]
if args.render:
    video = output / "source_rollout_episode0.mp4"
    contact = output / "source_rollout_episode0_contact_sheet.png"
    video.write_bytes(b"fake-mp4")
    contact.write_bytes(b"fake-png")
    artifact_paths.extend((video, contact))

def artifact(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }

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
    "artifacts": {path.name: artifact(path) for path in artifact_paths},
}
(output / "result.json").write_text(json.dumps(payload), encoding="utf-8")
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
            "infos": None,
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
            "--test-mode",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "test-only-contract-pass"
    assert result["scientific_valid"] is False
    assert (
        result["classification"]["synthetic_factorial_classification"]["outcome"]
        == "scheduler-synchronization-preserves"
    )
    assert len(result["evaluations"]) == 10
    assert result["evaluation_contract"]["initial_reset_identity"] is True
    assert tuple(result["training_contract"]["factorial_arms"]) == ARM_NAMES
    assert len(list((output / "evaluations").glob("*/source_eval/*.mp4"))) == 5


def test_artifact_manifest_detects_content_tampering(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"before")
    record = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": _sha256(artifact),
    }
    assert _artifact_errors(record, artifact) == []
    artifact.write_bytes(b"after!")
    assert "artifact SHA-256 mismatch" in " ".join(_artifact_errors(record, artifact))


def test_reported_source_checkpoint_rejects_wrong_or_ambiguous_paths(tmp_path: Path):
    expected = (tmp_path / "source" / "model_500.pt").resolve()
    wrong = (tmp_path / "other" / "model_500.pt").resolve()
    marker = f"[INFO]: Loading model checkpoint from: {expected}\n"
    assert _reported_source_checkpoint(marker, expected) == (expected, [])
    loaded, errors = _reported_source_checkpoint(
        f"[INFO]: Loading model checkpoint from: {wrong}\n", expected
    )
    assert loaded == wrong
    assert "trainer loaded" in errors[0]
    assert _reported_source_checkpoint(marker + marker, expected)[1] == [
        "expected exactly one trainer-reported source checkpoint, observed 2"
    ]


def test_scientific_preflight_forbids_alternate_trainer(tmp_path: Path):
    args = argparse.Namespace(
        test_mode=False,
        trainer=(tmp_path / "trainer.py").resolve(),
        evaluator=DEFAULT_EVALUATOR.resolve(),
        training_log_root=DEFAULT_TRAINING_LOG_ROOT.resolve(),
        device="cuda:0",
        render_short_arms=True,
        headless=True,
        code_commit=_git_output("rev-parse", "HEAD"),
        code_manifest_sha256="registered",
    )
    errors = _scientific_preflight_errors(args, {"sha256": "registered"})
    assert "scientific mode forbids trainer overrides" in errors


def test_pinned_input_mutation_is_detected(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"registered")
    pinned = {"source": (source, _sha256(source))}
    assert _integrity_errors(pinned, None) == []
    source.write_bytes(b"mutated")
    assert _integrity_errors(pinned, None) == ["pinned input changed: source"]
