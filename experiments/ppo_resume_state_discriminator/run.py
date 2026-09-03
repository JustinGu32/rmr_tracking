"""Run the fixed-rollout scheduler-state versus Adam-state discriminator."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    "stable_trainer.py": REPO_ROOT / "scripts" / "rsl_rl" / "train.py",
    "source_evaluator.py": DEFAULT_EVALUATOR,
    "source_evaluator_contract.py": REPO_ROOT
    / "experiments"
    / "exact_long_source_eval"
    / "contract.py",
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


def _child_environment(
    *, training_log_root: Path, discriminator_dir: Path
) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(name for name in AMBIENT_TOGGLES if name in environment)
    for name in AMBIENT_TOGGLES:
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
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
        args.source_checkpoint.parent.name,
        "--checkpoint",
        args.source_checkpoint.name,
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
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    training_dir = args.output_dir / "training"
    discriminator_dir = training_dir / "discriminator"
    discriminator_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "combined.log"
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
    if _sha256_file(args.source_checkpoint) != source_sha_before:
        errors.append("source checkpoint changed")
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

    factorial = None
    if result_path.is_file():
        try:
            factorial = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid factorial result: {error}")
    if factorial is not None:
        if (
            factorial.get("complete") is not True
            or factorial.get("runner_completed") is not True
        ):
            errors.append("factorial runner did not report complete execution")
        branches = factorial.get("branches", [])
        if tuple(branch.get("arm", {}).get("name") for branch in branches) != ARM_NAMES:
            errors.append("factorial result arm order mismatch")
        common_indices = {
            branch.get("pre_step", {}).get("indices_sha256") for branch in branches
        }
        common_gradients = {
            branch.get("pre_step", {}).get("gradient", {}).get("post_clip_sha256")
            for branch in branches
        }
        if len(common_indices) != 1 or None in common_indices:
            errors.append("branches do not share one first minibatch")
        if len(common_gradients) != 1 or None in common_gradients:
            errors.append("branches do not share one pre-Adam gradient")
        checkpoints = [factorial.get("step0_checkpoint")]
        checkpoints.extend(branch.get("checkpoint") for branch in branches)
        for record in checkpoints:
            if not isinstance(record, dict):
                errors.append("checkpoint manifest entry missing")
                continue
            path = Path(str(record.get("path", "/")))
            if not path.is_file() or _sha256_file(path) != record.get("sha256"):
                errors.append(f"checkpoint missing or hash-mismatched: {path}")

        restored_steps = _optimizer_steps(source_payload)
        if restored_steps != [10020]:
            errors.append(f"source optimizer step mismatch: {restored_steps}")
        for branch in branches:
            name = str(branch.get("arm", {}).get("name"))
            before_steps = branch.get("pre_step", {}).get(
                "optimizer_state_steps_before"
            )
            after_steps = branch.get("pre_step", {}).get("optimizer_state_steps_after")
            expected_before = [] if name.startswith("reset_adam__") else [10020]
            expected_after = [1] if name.startswith("reset_adam__") else [10021]
            if before_steps != expected_before or after_steps != expected_after:
                errors.append(f"optimizer accounting mismatch for {name}")

        if final_checkpoint is not None and final_checkpoint.is_file() and branches:
            try:
                native_path = Path(
                    next(
                        branch
                        for branch in branches
                        if branch["arm"]["name"] == NATIVE_ARM
                    )["checkpoint"]["path"]
                )
                native = torch.load(native_path, map_location="cpu", weights_only=False)
                final = torch.load(
                    final_checkpoint, map_location="cpu", weights_only=False
                )
                names = (
                    "model_state_dict",
                    "optimizer_state_dict",
                    "obs_norm_state_dict",
                    "privileged_obs_norm_state_dict",
                )
                if not all(
                    _nested_tensor_equal(native[name], final[name]) for name in names
                ):
                    errors.append(
                        "retained native branch differs from trainer final package"
                    )
            except (KeyError, OSError, RuntimeError, StopIteration) as error:
                errors.append(f"native final package validation failed: {error}")

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
        "training_log_root": str(args.training_log_root),
        "new_run_candidates": [str(path) for path in candidates],
        "run_dir": str(run_dir) if run_dir else None,
        "factorial_result": {
            "path": str(result_path),
            "sha256": _sha256_file(result_path) if result_path.is_file() else None,
        },
        "native_final_checkpoint": (
            {
                "path": str(final_checkpoint),
                "sha256": _sha256_file(final_checkpoint),
            }
            if final_checkpoint is not None and final_checkpoint.is_file()
            else None
        ),
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": _sha256_file(args.source_checkpoint),
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
    motion: Path,
    label_set: str,
    render: bool,
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    output_dir = args.output_dir / "evaluations" / name
    output_dir.mkdir(parents=True, exist_ok=True)
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
    result_path = output_dir / "source_eval" / "result.json"
    errors: list[str] = []
    payload = None
    if return_code != 0:
        errors.append(f"child return code {return_code}")
    if timed_out:
        errors.append("child timed out")
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid child result: {error}")
    else:
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
        checkpoint_sha = _sha256_file(checkpoint)
        for key in ("checkpoint", "normalizer_checkpoint"):
            if inputs.get(key, {}).get("sha256") != checkpoint_sha:
                errors.append(f"{key} hash mismatch")
        if inputs.get("motion", {}).get("sha256") != _sha256_file(motion):
            errors.append("motion hash mismatch")
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
        if render:
            video = output_dir / "source_eval" / "source_rollout_episode0.mp4"
            contact_sheet = (
                output_dir / "source_eval" / "source_rollout_episode0_contact_sheet.png"
            )
            if not video.is_file() or not contact_sheet.is_file():
                errors.append("rendered video or contact sheet missing")
    if errors:
        outcome = "invalid-execution"
    record = {
        "name": name,
        "package_arm": package_arm,
        "regime": "short" if label_set == "short-control" else "long",
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256_file(checkpoint)},
        "motion": {"path": str(motion), "sha256": _sha256_file(motion)},
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
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
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
    source_sha: str,
    e010_audit: dict[str, Any],
    training: dict[str, Any],
    factorial: dict[str, Any] | None,
    evaluations: list[dict[str, Any]],
    identities: list[tuple[str, str, str]],
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
            "e010_audit": {
                "path": str(args.e010_audit),
                "sha256": _sha256_file(args.e010_audit),
                "verdict": e010_audit.get("verdict"),
                "selected_outcome": e010_audit.get("selected_outcome"),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "experiment_files": {
                    name: _sha256_file(SCRIPT_DIR / name)
                    for name in (
                        "run.py",
                        "contract.py",
                        "design.py",
                        "train.py",
                        "runner.py",
                    )
                },
                "dependency_files": {
                    name: _sha256_file(path) for name, path in DEPENDENCY_FILES.items()
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
            "package is retained as a locomotion policy."
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    source_sha = _sha256_file(args.source_checkpoint)
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
        source_sha_before=source_sha,
        source_payload=source_payload,
    )

    evaluations: list[dict[str, Any]] = []
    identities: list[tuple[str, str, str]] = []
    if not training["contract_errors"] and factorial is not None:
        step0 = Path(factorial["step0_checkpoint"]["path"])
        checkpoints = {
            branch["arm"]["name"]: Path(branch["checkpoint"]["path"])
            for branch in factorial["branches"]
        }
        plan = [
            (
                "step_00_short",
                "step_00",
                step0,
                args.short_motion_file,
                "short-control",
                args.render_short_arms,
            ),
            (
                "step_00_long",
                "step_00",
                step0,
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
                        args.short_motion_file,
                        "short-control",
                        args.render_short_arms,
                    ),
                    (
                        f"{arm}_long",
                        arm,
                        checkpoints[arm],
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
            motion,
            label_set,
            render,
        ) in enumerate(plan):
            evaluation, arm_identities = _evaluate(
                args,
                name=name,
                package_arm=package_arm,
                checkpoint=checkpoint,
                motion=motion,
                label_set=label_set,
                render=render,
                environment=environment,
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
        source_sha=source_sha,
        e010_audit=e010_audit,
        training=training,
        factorial=factorial,
        evaluations=evaluations,
        identities=identities,
    )


if __name__ == "__main__":
    raise SystemExit(main())
