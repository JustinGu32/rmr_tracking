"""Run the bounded synchronized-scheduler PPO continuity control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ppo_first_update_probe import run as base_run

try:
    from .contract import CHECKPOINT_STEPS, classify_continuity
    from .design import (
        EXPECTED_FIRST_APPLIED_LR,
        EXPECTED_FRESH_SCHEDULER_LR,
        EXPECTED_RESTORED_OPTIMIZER_LR,
    )
except ImportError:  # Direct script execution.
    from contract import CHECKPOINT_STEPS, classify_continuity
    from design import (
        EXPECTED_FIRST_APPLIED_LR,
        EXPECTED_FRESH_SCHEDULER_LR,
        EXPECTED_RESTORED_OPTIMIZER_LR,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER = SCRIPT_DIR / "train.py"
DEFAULT_EVALUATOR = REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
DEFAULT_TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
LOCAL_WHOLE_BODY_SOURCE = REPO_ROOT / "source" / "whole_body_tracking"
ISAACLAB_ROOT = Path("/home/ubuntu/IsaacLab")
DEPENDENCY_FILES = {
    "first_update_run.py": REPO_ROOT
    / "experiments"
    / "ppo_first_update_probe"
    / "run.py",
    "first_update_runner.py": REPO_ROOT
    / "experiments"
    / "ppo_first_update_probe"
    / "runner.py",
    "first_update_probe.py": REPO_ROOT
    / "experiments"
    / "ppo_first_update_probe"
    / "probe.py",
    "resume_runner.py": REPO_ROOT
    / "experiments"
    / "resume_initial_observation_normalization"
    / "runner.py",
    "resume_normalization.py": REPO_ROOT
    / "experiments"
    / "resume_initial_observation_normalization"
    / "normalization.py",
    "motion_on_policy_runner.py": LOCAL_WHOLE_BODY_SOURCE
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py",
    "stable_trainer.py": REPO_ROOT / "scripts" / "rsl_rl" / "train.py",
    "source_evaluator.py": DEFAULT_EVALUATOR,
    "source_evaluator_contract.py": REPO_ROOT
    / "experiments"
    / "exact_long_source_eval"
    / "contract.py",
}
INSTALLED_MODULES = (
    "rsl_rl.algorithms.ppo",
    "rsl_rl.storage.rollout_storage",
    "rsl_rl.runners.on_policy_runner",
)
INSTALLED_DISTRIBUTIONS = (
    "rsl-rl-lib",
    "isaaclab",
    "isaaclab-rl",
    "isaaclab-tasks",
    "isaacsim",
    "torch",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue E011's synchronized resumed PPO state through one exact "
            "native 20-minibatch update."
        )
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--short-motion-file", required=True)
    parser.add_argument("--short-motion-sha256", required=True)
    parser.add_argument("--long-motion-file", required=True)
    parser.add_argument("--long-motion-sha256", required=True)
    parser.add_argument("--e011-audit", required=True)
    parser.add_argument("--e011-audit-sha256", required=True)
    parser.add_argument("--code-commit")
    parser.add_argument("--code-manifest-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainer", default=str(DEFAULT_TRAINER))
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR))
    parser.add_argument("--training-log-root", default=str(DEFAULT_TRAINING_LOG_ROOT))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--training-updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--ppo-output", default="delta-all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timeout-seconds", type=int, default=1200)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=300)
    parser.add_argument("--render-endpoints", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="allow fake executables while marking the result non-scientific",
    )
    args = parser.parse_args()
    if args.task != "Tracking-Flat-G1-v0" or args.ppo_output != "delta-all":
        parser.error("this control requires Tracking-Flat-G1-v0 with delta-all")
    if args.training_updates != 1 or args.num_envs != 4096 or args.train_seed != 42:
        parser.error("this control requires one update, 4096 environments, and seed 42")
    if args.eval_seed != 0:
        parser.error("this control requires deterministic evaluation seed zero")
    if min(args.training_timeout_seconds, args.evaluation_timeout_seconds) < 1:
        parser.error("timeouts must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("invalid run name")
    if not args.test_mode and (not args.code_commit or not args.code_manifest_sha256):
        parser.error("scientific execution requires pinned code commit and manifest")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_output(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_manifest() -> dict[str, Any]:
    experiment_files = {
        name: {
            "path": str((SCRIPT_DIR / name).resolve()),
            "sha256": _sha256_file(SCRIPT_DIR / name),
        }
        for name in (
            "run.py",
            "contract.py",
            "design.py",
            "train.py",
            "runner.py",
            "SEARCH_RECEIPT.md",
        )
    }
    dependency_files = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in DEPENDENCY_FILES.items()
    }
    installed_modules: dict[str, dict[str, str]] = {}
    for module_name in INSTALLED_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"installed module has no source: {module_name}")
        path = Path(spec.origin).resolve()
        installed_modules[module_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    versions = {
        name: importlib.metadata.version(name) for name in INSTALLED_DISTRIBUTIONS
    }
    versions.update(
        {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "torch_runtime": torch.__version__,
            "torch_cuda_runtime": str(torch.version.cuda),
        }
    )
    isaaclab_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ISAACLAB_ROOT, text=True
    ).strip()
    isaaclab_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ISAACLAB_ROOT,
        text=True,
    ).strip()
    isaaclab_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=ISAACLAB_ROOT
    )
    payload = {
        "experiment_files": experiment_files,
        "dependency_files": dependency_files,
        "installed_modules": installed_modules,
        "versions": versions,
        "external_repositories": {
            "IsaacLab": {
                "path": str(ISAACLAB_ROOT.resolve()),
                "git_commit": isaaclab_commit,
                "git_status_porcelain": isaaclab_status,
                "git_diff_binary_sha256": hashlib.sha256(isaaclab_diff).hexdigest(),
            }
        },
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _integrity_errors(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.test_mode:
        return []
    errors: list[str] = []
    if _git_output("rev-parse", "HEAD") != args.code_commit:
        errors.append("repository commit differs from registration")
    status = _git_output("status", "--short", "--untracked-files=all")
    if status != "?? uv.lock":
        errors.append(
            f"worktree state differs from registered uv.lock-only state: {status!r}"
        )
    if manifest["sha256"] != args.code_manifest_sha256:
        errors.append("code/environment manifest differs from registration")
    isaaclab = manifest["external_repositories"]["IsaacLab"]
    if isaaclab["git_status_porcelain"]:
        errors.append("editable IsaacLab checkout is dirty")
    return errors


def _prior_e011(audit: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    if (
        audit.get("verdict") != "pass"
        or audit.get("experiment_id") != "E-20260903-011"
        or audit.get("selected_outcome") != "scheduler-synchronization-preserves"
        or audit.get("failures") != []
    ):
        raise RuntimeError("E011 audit is not the passed scheduler discriminator")
    artifact = audit.get("artifacts", {}).get("factorial_result", {})
    factorial_path = Path(str(artifact.get("path", ""))).expanduser().resolve()
    factorial_sha = str(artifact.get("sha256", ""))
    if not factorial_path.is_file() or _sha256_file(factorial_path) != factorial_sha:
        raise RuntimeError("E011 factorial artifact is missing or hash-mismatched")
    factorial = json.loads(factorial_path.read_text(encoding="utf-8"))
    matches = [
        branch
        for branch in factorial.get("branches", [])
        if branch.get("arm", {}).get("name") == "restored_adam__synced_scheduler"
    ]
    if len(matches) != 1:
        raise RuntimeError("E011 synchronized restored-Adam branch is ambiguous")
    checkpoint = Path(matches[0]["checkpoint"]["path"]).expanduser().resolve()
    checkpoint_sha = str(matches[0]["checkpoint"]["sha256"])
    if not checkpoint.is_file() or _sha256_file(checkpoint) != checkpoint_sha:
        raise RuntimeError("E011 synchronized checkpoint is missing or hash-mismatched")
    return matches[0], checkpoint, factorial_sha


def _evaluation_plan() -> tuple[tuple[int, str, int], ...]:
    plan: list[tuple[int, str, int]] = [(0, "short", 3), (0, "long", 3)]
    plan.extend((step, "short", 3) for step in CHECKPOINT_STEPS if step != 0)
    plan.append((20, "long", 3))
    return tuple(plan)


def _bridge_to_e011(
    probe_result: dict[str, Any],
    e011_branch: dict[str, Any],
    e011_checkpoint: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    records = {
        int(record["step"]): record for record in probe_result.get("checkpoints", [])
    }
    step1_path = Path(str(records.get(1, {}).get("path", ""))).resolve()
    if not step1_path.is_file():
        return {"equal": False}, ["continuity step-one checkpoint is missing"]
    observed = torch.load(step1_path, map_location="cpu", weights_only=False)
    expected = torch.load(e011_checkpoint, map_location="cpu", weights_only=False)
    fields = (
        "model_state_dict",
        "optimizer_state_dict",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    )
    field_equal = {
        field: base_run._nested_tensor_equal(observed[field], expected[field])
        for field in fields
    }
    if not all(field_equal.values()):
        errors.append("continuity step one differs from E011 synchronized package")

    trace = probe_result.get("optimizer_trace", [])
    first = trace[0] if trace else {}
    expected_pre = e011_branch.get("pre_step", {})
    comparisons = {
        "indices_sha256": first.get("indices_sha256")
        == expected_pre.get("indices_sha256"),
        "phase_bin_counts": first.get("phase_bin_counts")
        == expected_pre.get("phase_bin_counts"),
        "loss": first.get("loss") == expected_pre.get("loss"),
        "analytic_kl": first.get("analytic_kl") == expected_pre.get("analytic_kl"),
        "actor_relative_l2_drift": math.isclose(
            float(
                first.get("parameter_drift_from_step0", {})
                .get("relative_l2", {})
                .get("actor", math.nan)
            ),
            float(
                expected_pre.get("parameter_drift", {})
                .get("relative_l2", {})
                .get("actor", math.nan)
            ),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "applied_learning_rate": math.isclose(
            float(first.get("learning_rate", math.nan)),
            float(expected_pre.get("applied_learning_rate", math.nan)),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
    }
    if not all(comparisons.values()):
        errors.append(
            "continuity first update telemetry differs from E011 synchronized branch"
        )
    return {
        "equal": not errors,
        "e012_checkpoint": {
            "path": str(step1_path),
            "sha256": _sha256_file(step1_path),
        },
        "e011_checkpoint": {
            "path": str(e011_checkpoint),
            "sha256": _sha256_file(e011_checkpoint),
        },
        "package_field_equal": field_equal,
        "telemetry_equal": comparisons,
    }, errors


def _survival(arm: dict[str, Any]) -> list[int]:
    return [
        int(value) for value in arm.get("classification", {}).get("survival_steps", [])
    ]


def _finalize(
    args: argparse.Namespace,
    *,
    started_at: str,
    source_sha: str,
    audit_sha: str,
    audit: dict[str, Any],
    manifest: dict[str, Any],
    integrity_before: list[str],
    integrity_after: list[str],
    training: dict[str, Any],
    probe_result: dict[str, Any] | None,
    bridge: dict[str, Any] | None,
    bridge_errors: list[str],
    arms: list[dict[str, Any]],
    reset_hashes: tuple[list[str], list[str], list[str]],
) -> int:
    errors = list(integrity_before) + list(integrity_after)
    errors.extend(training.get("contract_errors", []))
    errors.extend(bridge_errors)
    errors.extend(error for arm in arms for error in arm.get("contract_errors", []))
    qpos, qvel, observations = reset_hashes
    expected_episodes = sum(arm.get("episodes", 0) for arm in arms)
    reset_identity = (
        len(qpos) == len(qvel) == len(observations) == expected_episodes
        and expected_episodes == 30
        and len(set(qpos)) == len(set(qvel)) == len(set(observations)) == 1
    )
    if not reset_identity:
        errors.append("strict evaluation reset identities differ or are incomplete")

    short_arms = {
        int(arm["checkpoint_step"]): arm
        for arm in arms
        if arm.get("outcome_label_set") == "short-control"
    }
    long_arms = {
        int(arm["checkpoint_step"]): arm
        for arm in arms
        if arm.get("outcome_label_set") == "exact-long"
    }
    short_complete = {
        step: arm.get("outcome") == "short-source-completes"
        for step, arm in short_arms.items()
    }
    if tuple(sorted(short_complete)) != CHECKPOINT_STEPS or set(long_arms) != {0, 20}:
        errors.append("evaluation plan is incomplete")
    classification = classify_continuity(
        short_complete=short_complete,
        step0_long_complete=long_arms.get(0, {}).get("outcome")
        == "source-completes-exact-long",
        step20_long_complete=long_arms.get(20, {}).get("outcome")
        == "source-completes-exact-long",
    )
    if _survival(short_arms.get(0, {})) != [125, 125, 125]:
        errors.append("step-zero short control did not reproduce 125/125/125")
    if _survival(long_arms.get(0, {})) != [126, 126, 126]:
        errors.append("step-zero exact-long control did not reproduce 126/126/126")
    if _survival(short_arms.get(1, {})) != [125, 125, 125]:
        errors.append("step-one short bridge did not reproduce E011's 125/125/125")
    if (
        probe_result is None
        or probe_result.get("continuity_runner_completed") is not True
    ):
        errors.append("continuity runner did not complete")
    intervention = (probe_result or {}).get("resume_scheduler_intervention", {})
    scheduler_before = float(
        intervention.get("scheduler_learning_rate_before", math.nan)
    )
    scheduler_after = float(intervention.get("scheduler_learning_rate_after", math.nan))
    groups_before = intervention.get("optimizer_group_learning_rates_before", [])
    groups_after = intervention.get("optimizer_group_learning_rates_after", [])
    if (
        not math.isclose(
            scheduler_before,
            EXPECTED_FRESH_SCHEDULER_LR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            scheduler_after,
            EXPECTED_RESTORED_OPTIMIZER_LR,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not groups_before
        or groups_before != groups_after
        or scheduler_after != groups_before[0]
        or intervention.get("expected_first_applied_learning_rate")
        != scheduler_after * 1.5
        or intervention.get("optimizer_state_entries_before")
        != intervention.get("optimizer_state_entries_after")
    ):
        errors.append("scheduler-only intervention contract failed")
    if errors:
        classification = {
            **classification,
            "outcome": "invalid-execution",
            "execution_errors": errors,
        }

    result = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "outcome": classification["outcome"],
        "scientific_valid": not errors and not args.test_mode,
        "classification": classification,
        "inputs": {
            "source_checkpoint": {
                "path": str(args.source_checkpoint),
                "sha256": source_sha,
            },
            "short_motion": {
                "path": str(args.short_motion_file),
                "sha256": _sha256_file(args.short_motion_file),
            },
            "long_motion": {
                "path": str(args.long_motion_file),
                "sha256": _sha256_file(args.long_motion_file),
            },
            "e011_audit": {
                "path": str(args.e011_audit),
                "sha256": audit_sha,
                "verdict": audit.get("verdict"),
                "selected_outcome": audit.get("selected_outcome"),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short_at_end": _git_output(
                    "status", "--short", "--untracked-files=all"
                ),
                "registered_commit": args.code_commit,
                "registered_manifest_sha256": args.code_manifest_sha256,
                "manifest": manifest,
            },
        },
        "causal_change": (
            "Before E010's otherwise-identical corrected-order rollout and native PPO update, "
            "set only the unsaved PPO.learning_rate scalar from 1e-3 to the optimizer's restored "
            "2.25e-5 rate. Preserve Adam state, rollout, permutation, losses, clipping, and all task semantics."
        ),
        "training_contract": {
            "task": args.task,
            "updates": 1,
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "transitions": 98304,
            "epochs": 5,
            "mini_batches": 4,
            "optimizer_steps": 20,
            "sampling": "adaptive",
            "train_seed": 42,
            "scheduler_learning_rate_before": EXPECTED_FRESH_SCHEDULER_LR,
            "scheduler_learning_rate_after_sync": EXPECTED_RESTORED_OPTIMIZER_LR,
            "first_expected_applied_learning_rate": EXPECTED_FIRST_APPLIED_LR,
        },
        "evaluation_contract": {
            "short_checkpoint_steps": list(CHECKPOINT_STEPS),
            "short_episodes_per_checkpoint": 3,
            "long_checkpoint_steps": [0, 20],
            "long_episodes_per_checkpoint": 3,
            "strict_phase_zero_native_package": True,
            "initial_reset_identity": reset_identity,
            "rendered_endpoint_steps": [20] if args.render_endpoints else [],
        },
        "integrity_errors_before": integrity_before,
        "integrity_errors_after": integrity_after,
        "training": training,
        "probe_result": probe_result,
        "e011_step_one_bridge": bridge,
        "evaluations": arms,
        "claim_boundary": (
            "This is the final bounded PPO mechanism control. It tests one synchronized resumed "
            "20-minibatch update, not continued PPO convergence, AHAC correctness, differentiable-physics "
            "credit, or sim-to-real transfer. No generated PPO checkpoint is retained as a policy."
        ),
        "test_mode": bool(args.test_mode),
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    print(
        f"[RESUME-CONTINUITY-RUN] outcome={result['outcome']} result={result_path}",
        flush=True,
    )
    return 1 if result["outcome"] == "invalid-execution" else 0


def main() -> int:
    args = _parse_args()
    for name in (
        "source_checkpoint",
        "short_motion_file",
        "long_motion_file",
        "e011_audit",
        "trainer",
        "evaluator",
        "training_log_root",
        "output_dir",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    for path, expected, label in (
        (args.source_checkpoint, args.source_checkpoint_sha256, "source checkpoint"),
        (args.short_motion_file, args.short_motion_sha256, "short motion"),
        (args.long_motion_file, args.long_motion_sha256, "long motion"),
        (args.e011_audit, args.e011_audit_sha256, "E011 audit"),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"{label} missing or SHA-256 mismatched: {path}")
    if not args.trainer.is_file() or not args.evaluator.is_file():
        raise RuntimeError("trainer or evaluator missing")
    if not args.training_log_root.is_dir():
        raise RuntimeError("training log root missing")
    if (args.output_dir / "result.json").exists():
        raise RuntimeError("refusing to overwrite completed result")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_sha = _sha256_file(args.e011_audit)
    audit = json.loads(args.e011_audit.read_text(encoding="utf-8"))
    e011_branch, e011_checkpoint, e011_factorial_sha = _prior_e011(audit)
    manifest = _code_manifest() if not args.test_mode else {"sha256": "test-mode"}
    integrity_before = _integrity_errors(args, manifest)
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    source_sha = _sha256_file(args.source_checkpoint)
    started_at = _utc_now()
    probe_dir = args.output_dir / "training" / "probe"
    environment, removed = base_run._child_environment(
        training_log_root=args.training_log_root, probe_dir=probe_dir
    )

    training, _, probe_result = base_run._run_training(
        args,
        environment=environment,
        removed_toggles=removed,
        source_sha_before=source_sha,
        source_payload=source_payload,
    )
    training["e011_factorial_sha256"] = e011_factorial_sha
    log_path = Path(training["combined_log_path"])
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if log_text.count("[RESUME-SCHEDULER-CONTINUITY]") != 2:
            training["contract_errors"].append(
                "resume-scheduler continuity marker count mismatch"
            )
        if log_text.count("[RESUME-CONTINUITY-CODE]") != 1:
            training["contract_errors"].append("worktree-local code marker missing")

    bridge = None
    bridge_errors: list[str] = []
    if probe_result is not None:
        bridge, bridge_errors = _bridge_to_e011(
            probe_result, e011_branch, e011_checkpoint
        )
    else:
        bridge_errors.append("probe result unavailable for E011 bridge")
    _write_json(args.output_dir / "training.json", training)

    arms: list[dict[str, Any]] = []
    qpos: list[str] = []
    qvel: list[str] = []
    observations: list[str] = []
    plan = _evaluation_plan()
    if (
        not training["contract_errors"]
        and not bridge_errors
        and probe_result is not None
    ):
        checkpoints = {
            int(record["step"]): Path(record["path"]).resolve()
            for record in probe_result["checkpoints"]
        }
        for index, (step, regime, episodes) in enumerate(plan):
            arm, arm_qpos, arm_qvel, arm_observations = base_run._evaluate(
                args,
                arm=f"step_{step:02d}_{regime}",
                checkpoint=checkpoints[step],
                motion=args.short_motion_file
                if regime == "short"
                else args.long_motion_file,
                label_set="short-control" if regime == "short" else "exact-long",
                episodes=episodes,
                render=args.render_endpoints and step == 20,
                environment=environment,
            )
            arms.append(arm)
            qpos.extend(arm_qpos)
            qvel.extend(arm_qvel)
            observations.extend(arm_observations)
            _write_json(
                args.output_dir / "progress.json",
                {
                    "complete": False,
                    "completed_arms": index + 1,
                    "total_arms": len(plan),
                    "arms": arms,
                },
            )
            if arm["contract_errors"]:
                break
            if (
                arm["arm"] == "step_00_short"
                and arm["outcome"] != "short-source-completes"
            ):
                break
            if (
                arm["arm"] == "step_00_long"
                and arm["outcome"] != "source-fails-exact-long"
            ):
                break

    integrity_after = (
        _integrity_errors(args, _code_manifest()) if not args.test_mode else []
    )
    return _finalize(
        args,
        started_at=started_at,
        source_sha=source_sha,
        audit_sha=audit_sha,
        audit=audit,
        manifest=manifest,
        integrity_before=integrity_before,
        integrity_after=integrity_after,
        training=training,
        probe_result=probe_result,
        bridge=bridge,
        bridge_errors=bridge_errors,
        arms=arms,
        reset_hashes=(qpos, qvel, observations),
    )


if __name__ == "__main__":
    raise SystemExit(main())
