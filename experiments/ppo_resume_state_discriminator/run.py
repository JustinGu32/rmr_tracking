"""Run the fixed-rollout scheduler-state versus Adam-state discriminator."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from contract import classify_factorial
from design import ARM_NAMES, NATIVE_ARM

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER = SCRIPT_DIR / "train.py"
DEFAULT_EVALUATOR = REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
DEFAULT_TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
LOCAL_WHOLE_BODY_SOURCE = REPO_ROOT / "source" / "whole_body_tracking"
ISAACLAB_ROOT = Path("/home/ubuntu/IsaacLab")
DEPENDENCY_FILES = {
    "first_update_probe_runner.py": REPO_ROOT
    / "experiments"
    / "ppo_first_update_probe"
    / "runner.py",
    "first_update_probe_probe.py": REPO_ROOT
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
EDITABLE_PACKAGE_MODULES = ("isaaclab", "isaaclab_rl", "isaaclab_tasks")
EDITABLE_DISTRIBUTION_ROOTS = {
    "isaaclab": ISAACLAB_ROOT / "source" / "isaaclab",
    "isaaclab-rl": ISAACLAB_ROOT / "source" / "isaaclab_rl",
    "isaaclab-tasks": ISAACLAB_ROOT / "source" / "isaaclab_tasks",
}
AMBIENT_TOGGLES = (
    "ENABLE_CAMERAS",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "WBT_CURRICULUM",
    "WBT_DEPTH_DEBUG_MAX_FRAMES",
    "WBT_DEPTH_SAVE_FRAMES",
    "WBT_DOUBLE_STEP",
    "WBT_MOTION_JOINT_POS",
    "WBT_STAIR_PHASE_GRACE",
    "WBT_STAIR_PHASE_MIN_STEPS",
    "WBT_STAIR_PHASE_TERM",
    "WBT_USE_DEPTH_OBS",
    "WBT_VIDEO",
    "WBT_VIDEO_INTERVAL",
    "WBT_VIDEO_LENGTH",
    "BONES_DOUBLE_STEP",
    "BONES_GRAVITY_CURRICULUM",
    "BONES_GRAVITY_RAMP_STEPS",
    "BONES_START_GRAVITY",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate scheduler-resume state from restored Adam state on one fixed PPO minibatch."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--short-motion-file", required=True)
    parser.add_argument("--short-motion-sha256", required=True)
    parser.add_argument("--long-motion-file", required=True)
    parser.add_argument("--long-motion-sha256", required=True)
    parser.add_argument("--e010-audit", required=True)
    parser.add_argument("--e010-audit-sha256", required=True)
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
    parser.add_argument("--render-short-arms", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="allow fake executable overrides while producing non-scientific output",
    )
    args = parser.parse_args()
    if args.task != "Tracking-Flat-G1-v0" or args.ppo_output != "delta-all":
        parser.error("this discriminator requires Tracking-Flat-G1-v0 with delta-all")
    if args.training_updates != 1 or args.num_envs != 4096 or args.train_seed != 42:
        parser.error(
            "this discriminator requires one update, 4096 environments, and train seed 42"
        )
    if args.eval_seed != 0:
        parser.error("this discriminator requires deterministic evaluation seed zero")
    if min(args.training_timeout_seconds, args.evaluation_timeout_seconds) < 1:
        parser.error("timeouts must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("invalid run name")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_manifest() -> dict[str, Any]:
    experiment_files = {
        name: {
            "path": str((SCRIPT_DIR / name).resolve()),
            "sha256": _sha256_file(SCRIPT_DIR / name),
        }
        for name in ("run.py", "contract.py", "design.py", "train.py", "runner.py")
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
    versions: dict[str, str] = {}
    distribution_metadata: dict[str, dict[str, Any]] = {}
    for name in INSTALLED_DISTRIBUTIONS:
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        direct_url_text = distribution.read_text("direct_url.json")
        distribution_metadata[name] = {
            "version": distribution.version,
            "direct_url": (
                json.loads(direct_url_text) if direct_url_text is not None else None
            ),
        }
    versions.update(
        {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "torch_runtime": torch.__version__,
            "torch_cuda_runtime": str(torch.version.cuda),
        }
    )
    editable_package_origins: dict[str, dict[str, str]] = {}
    for module_name in EDITABLE_PACKAGE_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"editable package has no import origin: {module_name}")
        path = Path(spec.origin).resolve()
        editable_package_origins[module_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    try:
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
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"unable to pin editable IsaacLab checkout: {error}"
        ) from error
    payload = {
        "experiment_files": experiment_files,
        "dependency_files": dependency_files,
        "installed_modules": installed_modules,
        "versions": versions,
        "distribution_metadata": distribution_metadata,
        "editable_package_origins": editable_package_origins,
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


def _artifact_errors(record: Any, expected_path: Path) -> list[str]:
    if not isinstance(record, dict):
        return [f"artifact record missing: {expected_path}"]
    expected_path = expected_path.resolve()
    try:
        recorded_path = Path(str(record["path"])).resolve()
        recorded_bytes = int(record["bytes"])
        recorded_sha = str(record["sha256"])
    except (KeyError, TypeError, ValueError):
        return [f"artifact record malformed: {expected_path}"]
    errors: list[str] = []
    if recorded_path != expected_path:
        errors.append(f"artifact path mismatch: {recorded_path} != {expected_path}")
    if not expected_path.is_file():
        errors.append(f"artifact missing: {expected_path}")
        return errors
    if expected_path.stat().st_size != recorded_bytes:
        errors.append(f"artifact byte count mismatch: {expected_path}")
    if _sha256_file(expected_path) != recorded_sha:
        errors.append(f"artifact SHA-256 mismatch: {expected_path}")
    return errors


def _integrity_errors(
    pinned_files: dict[str, tuple[Path, str]],
    expected_manifest_sha256: str | None,
    expected_commit: str | None = None,
) -> list[str]:
    errors = []
    for label, (path, expected_sha) in pinned_files.items():
        if not path.is_file() or _sha256_file(path) != expected_sha:
            errors.append(f"pinned input changed: {label}")
    if expected_manifest_sha256 is not None:
        observed = _code_manifest()["sha256"]
        if observed != expected_manifest_sha256:
            errors.append("code/environment manifest changed")
    if (
        expected_commit is not None
        and _git_output("rev-parse", "HEAD") != expected_commit
    ):
        errors.append("repository commit changed")
    if expected_commit is not None:
        status = _git_output("status", "--porcelain", "--untracked-files=all")
        if status is None or any(
            line != "?? uv.lock" for line in (status or "").splitlines()
        ):
            errors.append("repository worktree changed")
    return errors


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


def _scientific_preflight_errors(
    args: argparse.Namespace, code_manifest: dict[str, Any]
) -> list[str]:
    if args.test_mode:
        return []
    errors: list[str] = []
    if args.trainer != DEFAULT_TRAINER.resolve():
        errors.append("scientific mode forbids trainer overrides")
    if args.evaluator != DEFAULT_EVALUATOR.resolve():
        errors.append("scientific mode forbids evaluator overrides")
    if args.training_log_root != DEFAULT_TRAINING_LOG_ROOT.resolve():
        errors.append("scientific mode requires the canonical training log root")
    if args.device != "cuda:0":
        errors.append("scientific mode requires the registered cuda:0 child device")
    if not args.render_short_arms:
        errors.append("scientific mode requires rendered short-arm audits")
    if not args.headless:
        errors.append("scientific mode requires headless simulator launches")
    head = _git_output("rev-parse", "HEAD")
    if not args.code_commit or args.code_commit != head:
        errors.append(
            f"registered code commit {args.code_commit!r} differs from HEAD {head!r}"
        )
    if not args.code_manifest_sha256 or args.code_manifest_sha256 != code_manifest.get(
        "sha256"
    ):
        errors.append("registered code/environment manifest SHA-256 mismatch")
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status is None:
        errors.append("unable to inspect repository status")
    else:
        disallowed = [line for line in status.splitlines() if line != "?? uv.lock"]
        if disallowed:
            errors.append(
                "scientific mode requires committed code; disallowed status: "
                + ", ".join(disallowed)
            )
    isaaclab_repository = code_manifest.get("external_repositories", {}).get(
        "IsaacLab", {}
    )
    if (
        Path(str(isaaclab_repository.get("path", "/"))).resolve()
        != ISAACLAB_ROOT.resolve()
        or isaaclab_repository.get("git_status_porcelain") != ""
        or isaaclab_repository.get("git_diff_binary_sha256")
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ):
        errors.append("scientific mode requires the clean pinned IsaacLab checkout")
    for distribution_name, expected_root in EDITABLE_DISTRIBUTION_ROOTS.items():
        direct_url = (
            code_manifest.get("distribution_metadata", {})
            .get(distribution_name, {})
            .get("direct_url")
        )
        if (
            not isinstance(direct_url, dict)
            or direct_url.get("url") != expected_root.resolve().as_uri()
            or direct_url.get("dir_info", {}).get("editable") is not True
        ):
            errors.append(f"editable install provenance mismatch: {distribution_name}")
    return errors


def _optimizer_steps(payload: dict[str, Any]) -> list[int]:
    values: set[int] = set()
    for state in payload["optimizer_state_dict"]["state"].values():
        if "step" in state:
            step = state["step"]
            values.add(
                int(step.item()) if isinstance(step, torch.Tensor) else int(step)
            )
    return sorted(values)


def _nested_tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_tensor_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _nested_tensor_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        )
    return left == right


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(_sha256_tensor(value).encode())
    elif isinstance(value, dict):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _update_state_digest(digest, item)
    elif isinstance(value, (str, int, float, bool, type(None))):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        digest.update(type(value).__qualname__.encode())
        digest.update(repr(value).encode())


def _state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value)
    return digest.hexdigest()


def _package_identity(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "model": _state_digest(payload["model_state_dict"]),
        "optimizer": _state_digest(payload["optimizer_state_dict"]),
        "actor_normalizer": _state_digest(payload["obs_norm_state_dict"]),
        "critic_normalizer": _state_digest(payload["privileged_obs_norm_state_dict"]),
    }


def _normalizer_count(payload: dict[str, Any], key: str) -> int:
    count = payload[key]["count"]
    if isinstance(count, torch.Tensor):
        if count.numel() != 1:
            raise ValueError(f"{key}.count is not scalar")
        return int(count.detach().cpu().item())
    return int(count)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _close(left: Any, right: float, *, tolerance: float = 1.0e-12) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _expected_package_path(discriminator_dir: Path, arm: str | None) -> Path:
    name = "model_step_00.pt" if arm is None else f"model_{arm}_step_01.pt"
    return (discriminator_dir / "checkpoints" / name).resolve()


def _reported_source_checkpoint(
    log_text: str, expected: Path
) -> tuple[Path | None, list[str]]:
    matches = re.findall(
        r"^\[INFO\]: Loading model checkpoint from: (.+?)\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        return None, [
            f"expected exactly one trainer-reported source checkpoint, observed {len(matches)}"
        ]
    candidate = Path(matches[0]).expanduser()
    loaded = (candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve()
    if loaded != expected.resolve():
        return loaded, [f"trainer loaded {loaded}, expected {expected.resolve()}"]
    return loaded, []


def _rollout_tensor_errors(
    factorial: dict[str, Any], discriminator_dir: Path
) -> list[str]:
    expected_path = (discriminator_dir / "factorial_rollout_tensors.pt").resolve()
    errors = _artifact_errors(factorial.get("rollout_tensors"), expected_path)
    if errors:
        return errors
    try:
        payload = torch.load(expected_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        return [f"rollout tensor package is unreadable: {error}"]
    expected_keys = {
        "phases",
        "timeouts",
        "native_permutation",
        "observations",
        "privileged_observations",
        "actions",
        "rewards",
        "dones",
        "values",
        "returns",
        "advantages",
        "actions_log_prob",
        "mu",
        "sigma",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        return ["rollout tensor package keys differ from the registered schema"]
    for name, tensor in payload.items():
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"rollout tensor entry is not a tensor: {name}")
            continue
        if name == "native_permutation":
            if tuple(tensor.shape) != (4096 * 24,):
                errors.append("native permutation shape mismatch")
        elif tuple(tensor.shape[:2]) != (24, 4096):
            errors.append(f"rollout tensor leading shape mismatch: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            errors.append(f"rollout tensor contains nonfinite values: {name}")
    if errors:
        return errors
    phases = payload["phases"].to(dtype=torch.long)
    timeouts = payload["timeouts"].bool()
    permutation = payload["native_permutation"].to(dtype=torch.long)
    if bool(((phases < 0) | (phases >= 272)).any().item()):
        errors.append("rollout phase lies outside the 272-state reference")
    if not torch.equal(torch.sort(permutation).values, torch.arange(4096 * 24)):
        errors.append("native permutation is not a bijection over 98,304 samples")
    if factorial.get("native_permutation_sha256") != _sha256_tensor(
        payload["native_permutation"]
    ):
        errors.append("native permutation digest mismatch")
    first_indices = permutation[: 4096 * 24 // 4]
    first_indices_digest = _sha256_tensor(first_indices)
    first_bins = torch.clamp((phases.flatten()[first_indices] * 6) // 272, min=0, max=5)
    first_bin_counts = torch.bincount(first_bins, minlength=6).tolist()
    for branch in factorial.get("branches", []):
        pre_step = branch.get("pre_step", {})
        if pre_step.get("indices_sha256") != first_indices_digest:
            errors.append(
                "branch indices do not identify the first saved permutation partition"
            )
            break
        if pre_step.get("phase_bin_counts") != first_bin_counts:
            errors.append(
                "branch phase-bin counts differ from the saved first partition"
            )
            break
    rollout = factorial.get("rollout", {})
    expected_scalars = {
        "samples": 4096 * 24,
        "rollout_steps": 24,
        "environments": 4096,
        "reference_states": 272,
        "adaptive_bin_count": 6,
    }
    for key, expected in expected_scalars.items():
        if rollout.get(key) != expected:
            errors.append(f"rollout summary mismatch: {key}")
    initial_histogram = torch.bincount(phases[0], minlength=272).tolist()
    bins = torch.clamp((phases.flatten() * 6) // 272, min=0, max=5)
    phase_bin_counts = torch.bincount(bins, minlength=6).tolist()
    dones = payload["dones"].flatten(0, 1).squeeze(-1).bool()
    if rollout.get("initial_phase_histogram") != initial_histogram:
        errors.append("initial phase histogram differs from saved rollout")
    if rollout.get("phase_bin_counts") != phase_bin_counts:
        errors.append("phase-bin counts differ from saved rollout")
    if rollout.get("distinct_phases") != int(torch.unique(phases).numel()):
        errors.append("distinct rollout phase count mismatch")
    if rollout.get("initial_distinct_phases") != int(
        (torch.bincount(phases[0], minlength=272) > 0).sum().item()
    ):
        errors.append("distinct initial phase count mismatch")
    if rollout.get("done_count") != int(dones.sum().item()):
        errors.append("rollout done count mismatch")
    if rollout.get("timeout_count") != int(timeouts.sum().item()):
        errors.append("rollout timeout count mismatch")
    if rollout.get("hard_termination_count") != int(
        (dones & ~timeouts.flatten()).sum().item()
    ):
        errors.append("rollout hard-termination count mismatch")
    return errors


def _factorial_errors(
    factorial: dict[str, Any],
    *,
    discriminator_dir: Path,
    source_payload: dict[str, Any],
    final_checkpoint: Path | None,
) -> list[str]:
    errors: list[str] = []
    required_true = (
        "complete",
        "factorial",
        "single_shared_rollout",
        "single_native_partition_per_arm",
        "runner_completed",
    )
    if factorial.get("schema_version") != 1:
        errors.append("factorial schema version mismatch")
    for key in required_true:
        if factorial.get(key) is not True:
            errors.append(f"factorial invariant is not true: {key}")
    if factorial.get("current_learning_iteration") != 500:
        errors.append("factorial current learning iteration mismatch")
    if factorial.get("retained_outer_state_arm") != NATIVE_ARM:
        errors.append("factorial did not retain the native arm")
    if not _close(factorial.get("fresh_scheduler_learning_rate"), 1.0e-3):
        errors.append("fresh scheduler scalar mismatch")
    if not _close(factorial.get("restored_optimizer_learning_rate"), 2.25e-5):
        errors.append("restored optimizer learning rate mismatch")
    errors.extend(_rollout_tensor_errors(factorial, discriminator_dir))

    branches = factorial.get("branches")
    if not isinstance(branches, list):
        return errors + ["factorial branches are missing"]
    if tuple(branch.get("arm", {}).get("name") for branch in branches) != ARM_NAMES:
        return errors + ["factorial result arm order mismatch"]

    step0_path = _expected_package_path(discriminator_dir, None)
    step0_record = factorial.get("step0_checkpoint")
    step0_artifact_errors = _artifact_errors(step0_record, step0_path)
    errors.extend(step0_artifact_errors)
    packages: dict[str, dict[str, Any]] = {}
    if not step0_artifact_errors:
        try:
            packages["step0"] = torch.load(
                step0_path, map_location="cpu", weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"step-zero checkpoint is unreadable: {error}")
    for name, branch in zip(ARM_NAMES, branches, strict=True):
        expected_path = _expected_package_path(discriminator_dir, name)
        artifact_errors = _artifact_errors(branch.get("checkpoint"), expected_path)
        errors.extend(artifact_errors)
        if not artifact_errors:
            try:
                packages[name] = torch.load(
                    expected_path, map_location="cpu", weights_only=False
                )
            except (OSError, RuntimeError, ValueError) as error:
                errors.append(f"branch checkpoint is unreadable for {name}: {error}")

    required_package_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    }
    if (
        not isinstance(source_payload, dict)
        or set(source_payload) != required_package_keys
    ):
        errors.append("source package keys differ from the registered RSL-RL package")
    step0 = packages.get("step0")
    baseline_identity: dict[str, str] | None = None
    if isinstance(step0, dict):
        if set(step0) != required_package_keys:
            errors.append("step-zero package keys differ from the registered schema")
        elif set(source_payload) == required_package_keys:
            for key in ("model_state_dict", "optimizer_state_dict", "iter", "infos"):
                if not _nested_tensor_equal(step0[key], source_payload[key]):
                    errors.append(f"step-zero package changed source {key}")
            try:
                expected_count = (
                    _normalizer_count(source_payload, "obs_norm_state_dict") + 4096 * 24
                )
                for key in (
                    "obs_norm_state_dict",
                    "privileged_obs_norm_state_dict",
                ):
                    if _normalizer_count(step0, key) != expected_count:
                        errors.append(f"step-zero normalizer count mismatch: {key}")
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"normalizer count validation failed: {error}")
            baseline_identity = _package_identity(step0)
            if factorial.get("baseline_state_identity") != baseline_identity:
                errors.append(
                    "factorial baseline identity differs from step-zero package"
                )
            if step0_record.get("optimizer_steps") != _optimizer_steps(step0):
                errors.append("step-zero optimizer-step manifest mismatch")
            step0_rates = [
                float(group["lr"])
                for group in step0["optimizer_state_dict"]["param_groups"]
            ]
            if step0_record.get("optimizer_learning_rates") != step0_rates:
                errors.append("step-zero optimizer-rate manifest mismatch")

    common_keys = (
        "indices_sha256",
        "sample_count",
        "phase_bin_counts",
        "advantage",
        "ratio",
        "log_ratio",
        "analytic_kl",
        "approximate_kl",
        "clipped_fraction",
        "forward_sha256",
        "loss",
        "gradient",
    )
    reference_pre_step = branches[0].get("pre_step", {})
    reference_rng = branches[0].get("post_step_rng")
    reference_losses = branches[0].get(
        "native_codepath_loss_dict_divided_by_configured_20"
    )
    state_entries = len(source_payload.get("optimizer_state_dict", {}).get("state", {}))
    reset_optimizer = copy.deepcopy(source_payload.get("optimizer_state_dict", {}))
    if isinstance(reset_optimizer, dict):
        reset_optimizer["state"] = {}
    reset_optimizer_digest = _state_digest(reset_optimizer)

    for name, branch in zip(ARM_NAMES, branches, strict=True):
        arm = branch.get("arm", {})
        reset_adam = name.startswith("reset_adam__")
        synchronized = name.endswith("synced_scheduler")
        if arm != {
            "name": name,
            "synchronize_scheduler": synchronized,
            "reset_adam": reset_adam,
        }:
            errors.append(f"arm specification mismatch: {name}")
        if (
            baseline_identity is not None
            and branch.get("pre_intervention_identity") != baseline_identity
        ):
            errors.append(f"pre-intervention identity mismatch: {name}")
        post_intervention = branch.get("post_intervention_identity", {})
        if baseline_identity is not None:
            for key in ("model", "actor_normalizer", "critic_normalizer"):
                if post_intervention.get(key) != baseline_identity[key]:
                    errors.append(
                        f"non-optimizer intervention detected for {name}: {key}"
                    )
            expected_optimizer_digest = (
                reset_optimizer_digest if reset_adam else baseline_identity["optimizer"]
            )
            if post_intervention.get("optimizer") != expected_optimizer_digest:
                errors.append(f"post-intervention optimizer identity mismatch: {name}")

        scheduler_start = 2.25e-5 if synchronized else 1.0e-3
        applied_rate = scheduler_start * 1.5
        intervention = branch.get("intervention", {})
        expected_entries_after = 0 if reset_adam else state_entries
        if (
            intervention.get("arm") != arm
            or not _close(intervention.get("restored_optimizer_learning_rate"), 2.25e-5)
            or not _close(intervention.get("fresh_scheduler_learning_rate"), 1.0e-3)
            or not _close(
                intervention.get("scheduler_learning_rate_after_intervention"),
                scheduler_start,
            )
            or any(
                not _close(rate, 2.25e-5)
                for rate in intervention.get(
                    "optimizer_group_learning_rates_after_intervention", []
                )
            )
            or not intervention.get("optimizer_group_learning_rates_after_intervention")
            or intervention.get("optimizer_state_entries_before_intervention")
            != state_entries
            or intervention.get("optimizer_state_entries_after_intervention")
            != expected_entries_after
        ):
            errors.append(f"intervention accounting mismatch: {name}")

        pre_step = branch.get("pre_step", {})
        if any(pre_step.get(key) != reference_pre_step.get(key) for key in common_keys):
            errors.append(
                f"pre-Adam minibatch, forward pass, loss, or gradient differs: {name}"
            )
        if (
            pre_step.get("epoch") != 0
            or pre_step.get("mini_batch") != 0
            or pre_step.get("global_step") != 1
            or pre_step.get("sample_count") != 4096 * 24 // 4
            or len(pre_step.get("phase_bin_counts", [])) != 6
            or sum(pre_step.get("phase_bin_counts", [])) != 4096 * 24 // 4
        ):
            errors.append(f"first native partition accounting mismatch: {name}")
        if not _is_sha256(pre_step.get("indices_sha256")):
            errors.append(f"first partition digest missing: {name}")
        forward_hashes = pre_step.get("forward_sha256", {})
        if set(forward_hashes) != {
            "log_probability",
            "mean",
            "sigma",
            "entropy",
            "value",
        } or not all(_is_sha256(value) for value in forward_hashes.values()):
            errors.append(f"forward-pass digests malformed: {name}")
        gradient = pre_step.get("gradient", {})
        if not _is_sha256(gradient.get("post_clip_sha256")):
            errors.append(f"clipped-gradient digest malformed: {name}")
        loss = pre_step.get("loss", {})
        if set(loss) != {"surrogate", "value", "entropy", "total"} or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in loss.values()
        ):
            errors.append(f"loss record malformed: {name}")
        if (
            not _close(pre_step.get("applied_learning_rate"), applied_rate)
            or not _close(pre_step.get("scheduler_learning_rate"), applied_rate)
            or pre_step.get("optimizer_state_steps_before")
            != ([] if reset_adam else [10020])
            or pre_step.get("optimizer_state_steps_after")
            != ([1] if reset_adam else [10021])
        ):
            errors.append(f"applied optimizer state mismatch: {name}")
        kl_mean = pre_step.get("analytic_kl", {}).get("mean")
        if not (
            isinstance(kl_mean, (int, float))
            and 0.0 < float(kl_mean) < 0.005
            and _close(pre_step.get("clipped_fraction"), 0.0)
        ):
            errors.append(f"adaptive-KL branch precondition mismatch: {name}")

        native_losses = branch.get(
            "native_codepath_loss_dict_divided_by_configured_20", {}
        )
        if native_losses != reference_losses or set(native_losses) != {
            "value_function",
            "surrogate",
            "entropy",
        }:
            errors.append(f"native returned loss dictionary mismatch: {name}")
        elif set(loss) == {
            "surrogate",
            "value",
            "entropy",
            "total",
        } and not (
            _close(native_losses["value_function"], float(loss["value"]) / 20.0)
            and _close(native_losses["surrogate"], float(loss["surrogate"]) / 20.0)
            and _close(native_losses["entropy"], float(loss["entropy"]) / 20.0)
        ):
            errors.append(f"native returned loss scaling mismatch: {name}")
        if branch.get("post_step_rng") != reference_rng:
            errors.append(f"post-step RNG mismatch: {name}")

        package = packages.get(name)
        if isinstance(package, dict) and isinstance(step0, dict):
            if set(package) != required_package_keys:
                errors.append(f"branch package keys differ from schema: {name}")
            else:
                for key in (
                    "obs_norm_state_dict",
                    "privileged_obs_norm_state_dict",
                    "iter",
                    "infos",
                ):
                    if not _nested_tensor_equal(package[key], step0[key]):
                        errors.append(
                            f"branch changed non-update package field {key}: {name}"
                        )
                expected_steps = [1] if reset_adam else [10021]
                actual_steps = _optimizer_steps(package)
                actual_rates = [
                    float(group["lr"])
                    for group in package["optimizer_state_dict"]["param_groups"]
                ]
                checkpoint_record = branch.get("checkpoint", {})
                if actual_steps != expected_steps:
                    errors.append(f"branch package optimizer-step mismatch: {name}")
                if not actual_rates or any(
                    not _close(rate, applied_rate) for rate in actual_rates
                ):
                    errors.append(f"branch package optimizer-rate mismatch: {name}")
                if checkpoint_record.get("optimizer_steps") != actual_steps:
                    errors.append(f"branch checkpoint step manifest mismatch: {name}")
                if checkpoint_record.get("optimizer_learning_rates") != actual_rates:
                    errors.append(f"branch checkpoint rate manifest mismatch: {name}")
                if branch.get("post_step_identity") != _package_identity(package):
                    errors.append(f"branch package identity mismatch: {name}")

    if factorial.get("retained_outer_rng") != reference_rng:
        errors.append("retained outer RNG differs from the counterfactual branches")
    native_branch = branches[-1]
    if factorial.get("final_state_identity") != native_branch.get("post_step_identity"):
        errors.append("retained outer state differs from native branch")
    if not _close(factorial.get("retained_outer_scheduler_learning_rate"), 1.5e-3):
        errors.append("retained native scheduler scalar mismatch")
    retained_rates = factorial.get("retained_outer_optimizer_learning_rates", [])
    if not retained_rates or any(not _close(rate, 1.5e-3) for rate in retained_rates):
        errors.append("retained native optimizer rates mismatch")
    outer = factorial.get("outer_state_after_super", {})
    if (
        {
            key: outer.get(key)
            for key in ("model", "optimizer", "actor_normalizer", "critic_normalizer")
        }
        != factorial.get("final_state_identity")
        or not _close(outer.get("scheduler_learning_rate"), 1.5e-3)
        or outer.get("optimizer_learning_rates") != retained_rates
        or outer.get("optimizer_steps") != [10021]
        or outer.get("rng") != reference_rng
    ):
        errors.append("outer runner state differs from retained native state")
    if isinstance(step0, dict):
        try:
            expected_count = _normalizer_count(step0, "obs_norm_state_dict")
            if factorial.get("normalizer_counts") != {
                "actor": expected_count,
                "critic": expected_count,
            }:
                errors.append("reported final normalizer counts mismatch")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"final normalizer count validation failed: {error}")

    native_package = packages.get(NATIVE_ARM)
    if final_checkpoint is None or not final_checkpoint.is_file():
        errors.append("native final checkpoint missing")
    elif isinstance(native_package, dict):
        try:
            final_package = torch.load(
                final_checkpoint, map_location="cpu", weights_only=False
            )
            if not _nested_tensor_equal(native_package, final_package):
                errors.append(
                    "trainer final package is not fully identical to native branch"
                )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"trainer final checkpoint is unreadable: {error}")
    return errors


def _child_environment(
    *, training_log_root: Path, discriminator_dir: Path
) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(name for name in AMBIENT_TOGGLES if name in environment)
    for name in AMBIENT_TOGGLES:
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    inherited_pythonpath = [
        str(Path(value).expanduser().resolve())
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value
    ]
    local_source = str(LOCAL_WHOLE_BODY_SOURCE.resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        [local_source]
        + [value for value in inherited_pythonpath if value != local_source]
    )
    environment["DIFFSIM_RESUME_STATE_DISCRIMINATOR_DIR"] = str(discriminator_dir)
    environment["DIFFSIM_FIRST_UPDATE_LOG_ROOT"] = str(training_log_root)
    return environment, removed


def _training_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(args.trainer),
        "--task",
        args.task,
        "--motion_file",
        str(args.long_motion_file),
        "--logger",
        "tensorboard",
        "--run_name",
        args.run_name,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.training_updates),
        "--seed",
        str(args.train_seed),
        "--ppo_output",
        args.ppo_output,
        "--sampling",
        "adaptive",
        "--device",
        args.device,
        "--resume",
        "True",
        "--load_run",
        f"^{re.escape(args.source_checkpoint.parent.name)}$",
        "--checkpoint",
        f"^{re.escape(args.source_checkpoint.name)}$",
    ]
    if args.headless:
        command.append("--headless")
    return command


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait()


def _stream_process(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool, str | None, float]:
    started = time.monotonic()
    timed_out = False
    launch_error = None
    return_code = 127
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            launch_error = str(error)
        else:
            assert process.stdout is not None
            lines: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            deadline = started + timeout_seconds
            reader_done = False
            while not (reader_done and process.poll() is not None):
                try:
                    line = lines.get(timeout=0.5)
                    if line is None:
                        reader_done = True
                    else:
                        log.write(line)
                        print(line, end="", flush=True)
                except queue.Empty:
                    pass
                if process.poll() is None and time.monotonic() >= deadline:
                    timed_out = True
                    _terminate_process_group(process)
            reader.join(timeout=1)
            return_code = process.wait()
    return return_code, timed_out, launch_error, time.monotonic() - started


def _run_training(
    args: argparse.Namespace,
    *,
    environment: dict[str, str],
    removed_toggles: list[str],
    source_sha_before: str,
    source_payload: dict[str, Any],
    pinned_files: dict[str, tuple[Path, str]],
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    training_dir = args.output_dir / "training"
    discriminator_dir = training_dir / "discriminator"
    discriminator_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "combined.log"
    integrity_before = _integrity_errors(
        pinned_files,
        expected_manifest_sha256,
        None if args.test_mode else args.code_commit,
    )
    if integrity_before:
        record = {
            "executed": False,
            "command": _training_command(args),
            "started_at": _utc_now(),
            "finished_at": _utc_now(),
            "duration_seconds": 0.0,
            "return_code": None,
            "timed_out": False,
            "combined_log_path": str(log_path),
            "removed_ambient_environment_toggles": removed_toggles,
            "training_log_root": str(args.training_log_root),
            "new_run_candidates": [],
            "run_dir": None,
            "factorial_result": None,
            "native_final_checkpoint": None,
            "loaded_source_checkpoint": None,
            "source_checkpoint_sha256_before": source_sha_before,
            "source_checkpoint_sha256_after": None,
            "integrity_errors_before": integrity_before,
            "integrity_errors_after": [],
            "contract_errors": integrity_before,
        }
        _write_json(args.output_dir / "training.json", record)
        return record, None
    before = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    command = _training_command(args)
    started_at = _utc_now()
    return_code, timed_out, launch_error, duration = _stream_process(
        command,
        environment=environment,
        log_path=log_path,
        timeout_seconds=args.training_timeout_seconds,
    )
    after = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    candidates = sorted(
        path for path in after - before if path.name.endswith(f"_{args.run_name}")
    )
    run_dir = candidates[0] if len(candidates) == 1 else None
    final_checkpoint = run_dir / "model_500.pt" if run_dir else None
    result_path = discriminator_dir / "factorial_result.json"
    errors: list[str] = []
    if launch_error:
        errors.append(f"trainer launch failed: {launch_error}")
    if return_code != 0:
        errors.append(f"trainer return code {return_code}")
    if timed_out:
        errors.append("trainer timed out")
    if len(candidates) != 1:
        errors.append(f"expected one new run directory, observed {len(candidates)}")
    if final_checkpoint is None or not final_checkpoint.is_file():
        errors.append("native final checkpoint missing")
    if not result_path.is_file():
        errors.append("factorial result missing")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    iterations = [
        int(value) for value in re.findall(r"Learning iteration (\d+)/\d+", log_text)
    ]
    if iterations != [500]:
        errors.append(f"expected native iteration 500 only, observed {iterations}")
    if log_text.count("[RESUME-INITIAL-NORMALIZATION]") != 1:
        errors.append("initial-normalization marker count mismatch")
    arm_markers = re.findall(r"\[RESUME-STATE-DISCRIMINATOR\] arm=([^ ]+)", log_text)
    if tuple(arm_markers) != ARM_NAMES:
        errors.append(f"factorial marker order mismatch: {arm_markers}")
    whole_body_origins = re.findall(
        r"^\[RESUME-STATE-CODE\] whole_body_tracking=(.+?)\s*$",
        log_text,
        flags=re.MULTILINE,
    )
    expected_whole_body_origin = (
        LOCAL_WHOLE_BODY_SOURCE / "whole_body_tracking" / "__init__.py"
    ).resolve()
    if (
        len(whole_body_origins) != 1
        or Path(whole_body_origins[0]).resolve() != expected_whole_body_origin
    ):
        errors.append(
            "trainer did not prove the pinned worktree whole_body_tracking import"
        )
    loaded_source, source_path_errors = _reported_source_checkpoint(
        log_text, args.source_checkpoint
    )
    errors.extend(source_path_errors)

    factorial = None
    if result_path.is_file():
        try:
            factorial = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid factorial result: {error}")
    if factorial is not None:
        try:
            errors.extend(
                _factorial_errors(
                    factorial,
                    discriminator_dir=discriminator_dir,
                    source_payload=source_payload,
                    final_checkpoint=final_checkpoint,
                )
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"factorial schema validation failed: {error}")

    integrity_after = _integrity_errors(
        pinned_files,
        expected_manifest_sha256,
        None if args.test_mode else args.code_commit,
    )
    errors.extend(integrity_after)

    record = {
        "executed": True,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": duration,
        "return_code": return_code,
        "timed_out": timed_out,
        "combined_log_path": str(log_path),
        "removed_ambient_environment_toggles": removed_toggles,
        "child_pythonpath": environment.get("PYTHONPATH"),
        "whole_body_tracking_origin": (
            whole_body_origins[0] if len(whole_body_origins) == 1 else None
        ),
        "training_log_root": str(args.training_log_root),
        "new_run_candidates": [str(path) for path in candidates],
        "run_dir": str(run_dir) if run_dir else None,
        "factorial_result": (
            {
                "path": str(result_path.resolve()),
                "bytes": result_path.stat().st_size,
                "sha256": _sha256_file(result_path),
            }
            if result_path.is_file()
            else None
        ),
        "native_final_checkpoint": (
            {
                "path": str(final_checkpoint.resolve()),
                "bytes": final_checkpoint.stat().st_size,
                "sha256": _sha256_file(final_checkpoint),
            }
            if final_checkpoint is not None and final_checkpoint.is_file()
            else None
        ),
        "loaded_source_checkpoint": str(loaded_source) if loaded_source else None,
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": (
            _sha256_file(args.source_checkpoint)
            if args.source_checkpoint.is_file()
            else None
        ),
        "integrity_errors_before": integrity_before,
        "integrity_errors_after": integrity_after,
        "contract_errors": errors,
    }
    _write_json(args.output_dir / "training.json", record)
    print(f"[RESUME-STATE-RUN] training rc={return_code} errors={errors}", flush=True)
    return record, factorial


def _evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    motion: Path,
    label_set: str,
    output_dir: Path,
    render: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator),
        "--task",
        args.task,
        "--motion-file",
        str(motion),
        "--checkpoint-path",
        str(checkpoint),
        "--normalizer-checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(output_dir / "source_eval"),
        "--episodes",
        "3",
        "--eval-seed",
        str(args.eval_seed),
        "--outcome-label-set",
        label_set,
        "--ppo-output",
        args.ppo_output,
        "--device",
        args.device,
    ]
    if render:
        command.append("--render")
    if args.headless:
        command.append("--headless")
    return command


def _evaluate(
    args: argparse.Namespace,
    *,
    name: str,
    package_arm: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    motion: Path,
    label_set: str,
    render: bool,
    environment: dict[str, str],
    pinned_files: dict[str, tuple[Path, str]],
    expected_manifest_sha256: str | None,
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    output_dir = args.output_dir / "evaluations" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_eval_dir = output_dir / "source_eval"
    command = _evaluation_command(
        args,
        checkpoint=checkpoint,
        motion=motion,
        label_set=label_set,
        output_dir=output_dir,
        render=render,
    )
    stdout_path = output_dir / "launcher_stdout.log"
    stderr_path = output_dir / "launcher_stderr.log"
    started = time.monotonic()
    timed_out = False
    checkpoint_sha = _sha256_file(checkpoint) if checkpoint.is_file() else None
    motion_sha = _sha256_file(motion) if motion.is_file() else None
    integrity_before = _integrity_errors(
        pinned_files,
        expected_manifest_sha256,
        None if args.test_mode else args.code_commit,
    )
    errors: list[str] = list(integrity_before)
    if checkpoint_sha is None:
        errors.append("evaluation checkpoint missing before launch")
    elif checkpoint_sha != checkpoint_sha256:
        errors.append("evaluation checkpoint differs from its training manifest")
    if motion_sha is None:
        errors.append("evaluation motion missing before launch")
    return_code: int | None = None
    if not errors:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=args.evaluation_timeout_seconds,
                    check=False,
                    start_new_session=True,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                return_code = 124
                timed_out = True
    result_path = source_eval_dir / "result.json"
    payload = None
    if return_code is not None and return_code != 0:
        errors.append(f"child return code {return_code}")
    if timed_out:
        errors.append("child timed out")
    if not integrity_before and result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid child result: {error}")
    elif not integrity_before:
        errors.append("child result missing")

    identities: list[tuple[str, str, str]] = []
    outcome = "invalid-execution"
    classification: dict[str, Any] = {}
    if payload is not None:
        outcome = str(payload.get("outcome", "invalid-execution"))
        classification = dict(payload.get("classification", {}))
        allowed = (
            {
                "short-source-completes",
                "short-source-fails",
                "mixed-short-source-competence",
            }
            if label_set == "short-control"
            else {
                "source-completes-exact-long",
                "source-fails-exact-long",
                "mixed-source-competence",
            }
        )
        if outcome not in allowed:
            errors.append(f"unexpected evaluator outcome {outcome}")
        if classification.get("contract_valid") is not True:
            errors.append("child evaluator contract invalid")
        inputs = payload.get("inputs", {})
        for key in ("checkpoint", "normalizer_checkpoint"):
            record = inputs.get(key, {})
            if (
                Path(str(record.get("path", "/"))).resolve() != checkpoint.resolve()
                or record.get("sha256") != checkpoint_sha
            ):
                errors.append(f"{key} path or hash mismatch")
        motion_record = inputs.get("motion", {})
        if (
            Path(str(motion_record.get("path", "/"))).resolve() != motion.resolve()
            or motion_record.get("sha256") != motion_sha
        ):
            errors.append("motion path or hash mismatch")
        episodes = payload.get("episodes", [])
        if not isinstance(episodes, list) or len(episodes) != 3:
            errors.append("episode count mismatch")
            episodes = []
        for episode in episodes:
            identities.append(
                (
                    str(episode.get("initial_qpos_sha256", "")),
                    str(episode.get("initial_qvel_sha256", "")),
                    str(episode.get("initial_policy_observation_sha256", "")),
                )
            )
        if not identities or any(not all(identity) for identity in identities):
            errors.append("reset identity missing")
        if payload.get("all_numeric_finite") is not True:
            errors.append("numeric telemetry nonfinite")
        if (
            payload.get("evaluation_contract", {}).get("normalizer_override_applied")
            is not False
        ):
            errors.append("normalizer override detected")
        expected_artifacts = {
            "episodes.json": source_eval_dir / "episodes.json",
            "trajectory.npz": source_eval_dir / "trajectory.npz",
        }
        if render:
            expected_artifacts.update(
                {
                    "source_rollout_episode0.mp4": source_eval_dir
                    / "source_rollout_episode0.mp4",
                    "source_rollout_episode0_contact_sheet.png": source_eval_dir
                    / "source_rollout_episode0_contact_sheet.png",
                }
            )
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
            errors.append("evaluator artifact manifest set mismatch")
        else:
            for artifact_name, artifact_path in expected_artifacts.items():
                errors.extend(
                    _artifact_errors(artifacts.get(artifact_name), artifact_path)
                )
    if not checkpoint.is_file() or _sha256_file(checkpoint) != checkpoint_sha256:
        errors.append("evaluation checkpoint changed during launch")
    if motion.is_file() and motion_sha != _sha256_file(motion):
        errors.append("evaluation motion changed during launch")
    integrity_after = _integrity_errors(
        pinned_files,
        expected_manifest_sha256,
        None if args.test_mode else args.code_commit,
    )
    errors.extend(integrity_after)
    if errors:
        outcome = "invalid-execution"
    record = {
        "name": name,
        "package_arm": package_arm,
        "regime": "short" if label_set == "short-control" else "long",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "observed_sha256_before": checkpoint_sha,
        },
        "motion": {"path": str(motion), "sha256": motion_sha},
        "episodes": 3,
        "render": render,
        "command": command,
        "duration_seconds": time.monotonic() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "outcome": outcome,
        "classification": classification,
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path) if result_path.is_file() else None,
        "result_bytes": result_path.stat().st_size if result_path.is_file() else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "integrity_errors_before": integrity_before,
        "integrity_errors_after": integrity_after,
        "contract_errors": errors,
    }
    print(
        f"[RESUME-STATE-RUN] arm={name} outcome={outcome} "
        f"survival={classification.get('survival_steps')} rc={return_code}",
        flush=True,
    )
    return record, identities


def _finalize(
    args: argparse.Namespace,
    *,
    started_at: str,
    e010_audit: dict[str, Any],
    training: dict[str, Any],
    factorial: dict[str, Any] | None,
    evaluations: list[dict[str, Any]],
    identities: list[tuple[str, str, str]],
    pinned_files: dict[str, tuple[Path, str]],
    code_manifest: dict[str, Any],
    expected_manifest_sha256: str | None,
    git_status_at_start: str | None,
) -> int:
    errors = list(training.get("contract_errors", []))
    errors.extend(
        error for evaluation in evaluations for error in evaluation["contract_errors"]
    )
    reset_identity = (
        len(identities) == 30
        and len({identity[0] for identity in identities}) == 1
        and len({identity[1] for identity in identities}) == 1
        and len({identity[2] for identity in identities}) == 1
    )
    if not reset_identity:
        errors.append("thirty initial reset identities differ or are incomplete")
    errors.extend(
        _integrity_errors(
            pinned_files,
            expected_manifest_sha256,
            None if args.test_mode else args.code_commit,
        )
    )

    by_name = {evaluation["name"]: evaluation for evaluation in evaluations}
    required = {"step_00_short", "step_00_long"}
    required.update(
        f"{arm}_{regime}" for arm in ARM_NAMES for regime in ("short", "long")
    )
    if set(by_name) != required:
        errors.append("evaluation arm set is incomplete")

    mixed = any(
        evaluation["outcome"]
        in {"mixed-short-source-competence", "mixed-source-competence"}
        for evaluation in evaluations
    )
    if mixed:
        classification: dict[str, Any] = {"outcome": "mixed-factorial-competence"}
    elif required.issubset(by_name):
        classification = classify_factorial(
            step0_short_complete=by_name["step_00_short"]["outcome"]
            == "short-source-completes",
            short_complete={
                arm: by_name[f"{arm}_short"]["outcome"] == "short-source-completes"
                for arm in ARM_NAMES
            },
            long_complete={
                arm: by_name[f"{arm}_long"]["outcome"] == "source-completes-exact-long"
                for arm in ARM_NAMES
            },
        )
    else:
        classification = {"outcome": "invalid-execution"}

    if required.issubset(by_name):
        if by_name["step_00_short"]["classification"].get("survival_steps") != [
            125,
            125,
            125,
        ]:
            errors.append("step-zero short control did not reproduce 125/125/125")
        if by_name["step_00_long"]["classification"].get("survival_steps") != [
            126,
            126,
            126,
        ]:
            errors.append("step-zero long control did not reproduce 126/126/126")
        if by_name[f"{NATIVE_ARM}_short"]["classification"].get("survival_steps") != [
            43,
            43,
            43,
        ]:
            errors.append("native first-step control did not reproduce 43/43/43")
    errors = list(dict.fromkeys(errors))
    if errors:
        classification = {
            **classification,
            "outcome": "invalid-execution",
            "execution_errors": errors,
        }

    scientific_valid = not args.test_mode and not errors
    if args.test_mode and not errors:
        classification = {
            "outcome": "test-only-contract-pass",
            "scientific_valid": False,
            "synthetic_factorial_classification": classification,
        }
    else:
        classification = {**classification, "scientific_valid": scientific_valid}

    source_path, source_sha = pinned_files["source checkpoint"]
    short_path, short_sha = pinned_files["short motion"]
    long_path, long_sha = pinned_files["long motion"]
    audit_path, audit_sha = pinned_files["E010 audit"]
    trainer_path, trainer_sha = pinned_files["trainer"]
    evaluator_path, evaluator_sha = pinned_files["evaluator"]

    result = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "outcome": classification["outcome"],
        "scientific_valid": scientific_valid,
        "test_mode": args.test_mode,
        "classification": classification,
        "inputs": {
            "source_checkpoint": {
                "path": str(source_path),
                "sha256": source_sha,
            },
            "short_motion": {
                "path": str(short_path),
                "sha256": short_sha,
            },
            "long_motion": {
                "path": str(long_path),
                "sha256": long_sha,
            },
            "e010_audit": {
                "path": str(audit_path),
                "sha256": audit_sha,
                "verdict": e010_audit.get("verdict"),
                "selected_outcome": e010_audit.get("selected_outcome"),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "registered_git_commit": args.code_commit,
                "git_status_short_at_start": git_status_at_start,
                "manifest": code_manifest,
                "registered_manifest_sha256": args.code_manifest_sha256,
                "trainer": {
                    "path": str(trainer_path),
                    "sha256": trainer_sha,
                },
                "evaluator": {
                    "path": str(evaluator_path),
                    "sha256": evaluator_sha,
                },
            },
        },
        "causal_change": (
            "From one byte-identical post-rollout state, branch only whether PPO.learning_rate is synchronized "
            "to the restored optimizer rate and whether restored Adam state is cleared. Each arm consumes the "
            "same first native permutation partition and identical clipped gradient exactly once."
        ),
        "training_contract": {
            "task": args.task,
            "updates": 1,
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "transitions": 98304,
            "train_seed": 42,
            "sampling": "adaptive",
            "configured_epochs": 5,
            "configured_mini_batches": 4,
            "executed_optimizer_steps_per_arm": 1,
            "factorial_arms": list(ARM_NAMES),
            "fresh_scheduler_learning_rate": 1.0e-3,
            "restored_optimizer_learning_rate": 2.25e-5,
        },
        "evaluation_contract": {
            "step0_short_and_long_episodes": 3,
            "each_arm_short_and_long_episodes": 3,
            "short_arms_rendered": args.render_short_arms,
            "strict_phase_zero_native_package": True,
            "initial_reset_identity": reset_identity,
        },
        "training": training,
        "factorial_result": factorial,
        "evaluations": evaluations,
        "claim_boundary": (
            "This is a one-minibatch PPO mechanism discriminator. It does not establish stable long-reference "
            "learning, AHAC correctness, differentiable-physics credit, or sim-to-real transfer. No branch "
            "package is retained as a locomotion policy. Synthetic test-mode executions are explicitly "
            "non-scientific and cannot support the causal classification."
        ),
    }
    result_path = args.output_dir / "result.json"
    _write_json(result_path, result)
    print(
        f"[RESUME-STATE-RUN] outcome={result['outcome']} result={result_path}",
        flush=True,
    )
    return 1 if result["outcome"] == "invalid-execution" else 0


def main() -> int:
    args = _parse_args()
    for name in (
        "source_checkpoint",
        "short_motion_file",
        "long_motion_file",
        "e010_audit",
        "trainer",
        "evaluator",
        "training_log_root",
        "output_dir",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    code_manifest = _code_manifest()
    preflight_errors = _scientific_preflight_errors(args, code_manifest)
    if preflight_errors:
        raise RuntimeError("; ".join(preflight_errors))
    if args.source_checkpoint.parent.parent != args.training_log_root:
        raise RuntimeError(
            "source checkpoint must be directly inside one canonical training run"
        )
    for path, expected, label in (
        (args.source_checkpoint, args.source_checkpoint_sha256, "source checkpoint"),
        (args.short_motion_file, args.short_motion_sha256, "short motion"),
        (args.long_motion_file, args.long_motion_sha256, "long motion"),
        (args.e010_audit, args.e010_audit_sha256, "E010 audit"),
    ):
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"{label} missing or SHA-256 mismatched: {path}")
    if not args.trainer.is_file() or not args.evaluator.is_file():
        raise RuntimeError("trainer or evaluator missing")
    if not args.training_log_root.is_dir():
        raise RuntimeError("training log root missing")
    if (args.output_dir / "result.json").exists():
        raise RuntimeError("refusing to overwrite completed result")

    e010_audit = json.loads(args.e010_audit.read_text(encoding="utf-8"))
    if (
        e010_audit.get("verdict") != "pass"
        or e010_audit.get("selected_outcome")
        != "optimizer-step-localized-monotonic-loss"
    ):
        raise RuntimeError("E010 audit is not the passed first-update boundary")
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    required_source_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    }
    if (
        not isinstance(source_payload, dict)
        or set(source_payload) != required_source_keys
    ):
        raise RuntimeError("source checkpoint package schema mismatch")
    source_rates = [
        float(group["lr"])
        for group in source_payload["optimizer_state_dict"]["param_groups"]
    ]
    source_boundary_matches = (
        source_payload.get("iter") == 500
        and _optimizer_steps(source_payload) == [10020]
        and bool(source_rates)
        and all(_close(rate, 2.25e-5) for rate in source_rates)
    )
    if not args.test_mode:
        source_boundary_matches = source_boundary_matches and (
            _normalizer_count(source_payload, "obs_norm_state_dict") == 49_250_304
            and _normalizer_count(source_payload, "privileged_obs_norm_state_dict")
            == 49_250_304
        )
    if not source_boundary_matches:
        raise RuntimeError(
            "source checkpoint does not match the registered E010 boundary"
        )

    pinned_files = {
        "source checkpoint": (
            args.source_checkpoint,
            args.source_checkpoint_sha256,
        ),
        "short motion": (args.short_motion_file, args.short_motion_sha256),
        "long motion": (args.long_motion_file, args.long_motion_sha256),
        "E010 audit": (args.e010_audit, args.e010_audit_sha256),
        "trainer": (args.trainer, _sha256_file(args.trainer)),
        "evaluator": (args.evaluator, _sha256_file(args.evaluator)),
    }
    expected_manifest_sha256 = None if args.test_mode else args.code_manifest_sha256
    initial_integrity_errors = _integrity_errors(
        pinned_files,
        expected_manifest_sha256,
        None if args.test_mode else args.code_commit,
    )
    if initial_integrity_errors:
        raise RuntimeError("; ".join(initial_integrity_errors))
    git_status_at_start = _git_output("status", "--short")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    discriminator_dir = args.output_dir / "training" / "discriminator"
    environment, removed = _child_environment(
        training_log_root=args.training_log_root,
        discriminator_dir=discriminator_dir,
    )
    training, factorial = _run_training(
        args,
        environment=environment,
        removed_toggles=removed,
        source_sha_before=args.source_checkpoint_sha256,
        source_payload=source_payload,
        pinned_files=pinned_files,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    evaluations: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str]] = []
    if not training["contract_errors"] and factorial is not None:
        step0 = Path(factorial["step0_checkpoint"]["path"])
        step0_sha = str(factorial["step0_checkpoint"]["sha256"])
        checkpoints = {
            branch["arm"]["name"]: Path(branch["checkpoint"]["path"])
            for branch in factorial["branches"]
        }
        checkpoint_hashes = {
            branch["arm"]["name"]: str(branch["checkpoint"]["sha256"])
            for branch in factorial["branches"]
        }
        plan = [
            (
                "step_00_short",
                "step_00",
                step0,
                step0_sha,
                args.short_motion_file,
                "short-control",
                args.render_short_arms,
            ),
            (
                "step_00_long",
                "step_00",
                step0,
                step0_sha,
                args.long_motion_file,
                "exact-long",
                False,
            ),
        ]
        for arm in ARM_NAMES:
            plan.extend(
                (
                    (
                        f"{arm}_short",
                        arm,
                        checkpoints[arm],
                        checkpoint_hashes[arm],
                        args.short_motion_file,
                        "short-control",
                        args.render_short_arms,
                    ),
                    (
                        f"{arm}_long",
                        arm,
                        checkpoints[arm],
                        checkpoint_hashes[arm],
                        args.long_motion_file,
                        "exact-long",
                        False,
                    ),
                )
            )
        for index, (
            name,
            package_arm,
            checkpoint,
            checkpoint_sha256,
            motion,
            label_set,
            render,
        ) in enumerate(plan):
            evaluation, arm_identities = _evaluate(
                args,
                name=name,
                package_arm=package_arm,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                motion=motion,
                label_set=label_set,
                render=render,
                environment=environment,
                pinned_files=pinned_files,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            evaluations.append(evaluation)
            identities.extend(arm_identities)
            _write_json(
                args.output_dir / "progress.json",
                {
                    "complete": False,
                    "completed_evaluations": index + 1,
                    "total_evaluations": len(plan),
                    "evaluations": evaluations,
                },
            )
            if evaluation["contract_errors"]:
                break

    return _finalize(
        args,
        started_at=started_at,
        e010_audit=e010_audit,
        training=training,
        factorial=factorial,
        evaluations=evaluations,
        identities=identities,
        pinned_files=pinned_files,
        code_manifest=code_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        git_status_at_start=git_status_at_start,
    )


if __name__ == "__main__":
    raise SystemExit(main())
