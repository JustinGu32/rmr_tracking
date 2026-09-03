import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from run import _bridge_to_e011, _evaluation_plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(path: Path, value: float) -> None:
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([value])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(10021)}},
                "param_groups": [{"lr": 3.375e-5}],
            },
            "obs_norm_state_dict": {"count": torch.tensor(98304)},
            "privileged_obs_norm_state_dict": {"count": torch.tensor(98304)},
        },
        path,
    )


def test_evaluation_plan_is_the_registered_thirty_episodes():
    plan = _evaluation_plan()
    assert [row[:2] for row in plan] == [
        (0, "short"),
        (0, "long"),
        (1, "short"),
        (2, "short"),
        (4, "short"),
        (8, "short"),
        (12, "short"),
        (16, "short"),
        (20, "short"),
        (20, "long"),
    ]
    assert sum(row[2] for row in plan) == 30


def test_step_one_bridge_requires_equal_package_and_telemetry(tmp_path: Path):
    observed = tmp_path / "model_step_01.pt"
    expected = tmp_path / "e011.pt"
    _package(observed, 1.0)
    _package(expected, 1.0)
    common = {
        "indices_sha256": "indices",
        "phase_bin_counts": [1, 2, 3, 4, 5, 6],
        "loss": {"total": 1.0},
        "analytic_kl": {"mean": 0.00029},
    }
    probe = {
        "checkpoints": [{"step": 1, "path": str(observed)}],
        "optimizer_trace": [
            {
                **common,
                "learning_rate": 3.375e-5,
                "parameter_drift_from_step0": {"relative_l2": {"actor": 0.000276866}},
            }
        ],
    }
    branch = {
        "pre_step": {
            **common,
            "parameter_drift": {"relative_l2": {"actor": 0.000276866}},
        }
    }
    bridge, errors = _bridge_to_e011(probe, branch, expected)
    assert errors == []
    assert bridge["equal"] is True

    _package(expected, 2.0)
    bridge, errors = _bridge_to_e011(probe, branch, expected)
    assert errors == ["E012 step one differs from E011 synchronized package"]
    assert bridge["equal"] is False


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

common = {
    "indices_sha256": "fixed-indices",
    "phase_bin_counts": [6625, 8012, 7343, 911, 848, 837],
    "loss": {"entropy": 1.0, "surrogate": 2.0, "total": 3.0, "value": 4.0},
    "analytic_kl": {"count": 24576, "max": 0.00029, "mean": 0.00029, "min": 0.00029, "std": 0.0},
}
records = []
trace = []
for step in range(21):
    package = {
        "iter": 500,
        "model_state_dict": {"weight": torch.tensor([float(step)])},
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(10020 + step)}},
            "param_groups": [{"lr": 3.375e-5}],
        },
        "obs_norm_state_dict": {"count": source["obs_norm_state_dict"]["count"] + 4096 * 24},
        "privileged_obs_norm_state_dict": {"count": source["privileged_obs_norm_state_dict"]["count"] + 4096 * 24},
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
    if step:
        trace.append({
            "optimizer_step": step,
            **common,
            "learning_rate": 3.375e-5,
            "parameter_drift_from_step0": {"relative_l2": {"actor": 0.000276866}},
        })

probe_result = {
    "schema_version": 1,
    "complete": True,
    "runner_completed": True,
    "continuity_runner_completed": True,
    "measurement_only": True,
    "optimizer_trace": trace,
    "frozen_gradient_analysis": {
        "state_identity_before": {"same": "yes"},
        "state_identity_after": {"same": "yes"},
        "rng_identity_before": {"same": "yes"},
        "rng_identity_after": {"same": "yes"},
    },
    "checkpoints": records,
    "resume_scheduler_intervention": {
        "causal_change": "PPO.learning_rate only",
        "scheduler_learning_rate_before": 1e-3,
        "scheduler_learning_rate_after": 2.25e-5,
        "optimizer_group_learning_rates_before": [2.25e-5],
        "optimizer_group_learning_rates_after": [2.25e-5],
        "optimizer_state_entries_before": 2,
        "optimizer_state_entries_after": 2,
    },
}
(probe / "probe_result.json").write_text(json.dumps(probe_result), encoding="utf-8")
final = torch.load(checkpoint_dir / "model_step_20.pt", map_location="cpu", weights_only=False)
torch.save(final, run / "model_500.pt")
print("[RESUME-CONTINUITY-CODE] whole_body_tracking=fake", flush=True)
print("[RESUME-INITIAL-NORMALIZATION] metadata=fake", flush=True)
print("[RESUME-SCHEDULER-CONTINUITY] scheduler=2.25e-5 adam_entries=2", flush=True)
for step in range(1, 21):
    print(f"[FIRST-UPDATE-PROBE] step={step:02d} fake", flush=True)
print("[RESUME-SCHEDULER-CONTINUITY] complete result=fake", flush=True)
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

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

short = args.outcome_label_set == "short-control"
outcome = "short-source-completes" if short else "source-fails-exact-long"
survival = 125 if short else 126
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
        "all_episodes_complete": short,
    },
    "evaluation_contract": {"normalizer_override_applied": False},
    "episodes": episodes,
    "all_numeric_finite": True,
}
(output / "result.json").write_text(json.dumps(payload), encoding="utf-8")
"""


def test_fake_end_to_end_preserves_all_registered_checkpoints(tmp_path: Path):
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
    expected_step1 = tmp_path / "e011_step1.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(10021)}},
                "param_groups": [{"lr": 3.375e-5}],
            },
            "obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
            "privileged_obs_norm_state_dict": {"count": torch.tensor(10 + 4096 * 24)},
        },
        expected_step1,
    )
    common = {
        "indices_sha256": "fixed-indices",
        "phase_bin_counts": [6625, 8012, 7343, 911, 848, 837],
        "loss": {"entropy": 1.0, "surrogate": 2.0, "total": 3.0, "value": 4.0},
        "analytic_kl": {
            "count": 24576,
            "max": 0.00029,
            "mean": 0.00029,
            "min": 0.00029,
            "std": 0.0,
        },
    }
    factorial = tmp_path / "factorial.json"
    factorial.write_text(
        json.dumps(
            {
                "branches": [
                    {
                        "arm": {"name": "restored_adam__synced_scheduler"},
                        "checkpoint": {
                            "path": str(expected_step1),
                            "sha256": _sha256(expected_step1),
                        },
                        "pre_step": {
                            **common,
                            "parameter_drift": {"relative_l2": {"actor": 0.000276866}},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "experiment_id": "E-20260903-011",
                "selected_outcome": "scheduler-synchronization-preserves",
                "failures": [],
                "artifacts": {
                    "factorial_result": {
                        "path": str(factorial),
                        "sha256": _sha256(factorial),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    trainer = tmp_path / "trainer.py"
    evaluator = tmp_path / "evaluator.py"
    trainer.write_text(textwrap.dedent(FAKE_TRAINER), encoding="utf-8")
    evaluator.write_text(textwrap.dedent(FAKE_EVALUATOR), encoding="utf-8")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "run.py"),
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
            "--e011-audit",
            str(audit),
            "--e011-audit-sha256",
            _sha256(audit),
            "--output-dir",
            str(output),
            "--trainer",
            str(trainer),
            "--evaluator",
            str(evaluator),
            "--training-log-root",
            str(logs),
            "--run-name",
            "E012_fake",
            "--headless",
            "--test-mode",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["outcome"] == "synchronized-resume-preserves-sampled-full-update"
    assert result["scientific_valid"] is False
    assert result["e011_step_one_bridge"]["equal"] is True
    assert len(result["evaluations"]) == 10
