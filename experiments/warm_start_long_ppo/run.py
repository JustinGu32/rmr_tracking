"""Resume a competent short-reference PPO package on the exact long reference.

The runner keeps the stable trainer unchanged.  It evaluates the immutable
source package on both strict gates, performs one fixed 500-update resume whose
only configured task change is the motion file, and evaluates the first and
final saved packages on the same gates.
"""

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

from contract import (
    BASELINE_LONG,
    BASELINE_SHORT,
    EXPECTED_OUTCOMES,
    LONG_COMPLETE,
    LONG_FAIL,
    LONG_MIXED,
    POST_1_LONG,
    POST_1_SHORT,
    POST_500_LONG,
    POST_500_SHORT,
    SHORT_COMPLETE,
    classify_warm_start,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER = REPO_ROOT / "scripts" / "rsl_rl" / "train.py"
DEFAULT_EVALUATOR = REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
DEFAULT_TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
FIXED_ARM_ORDER = (
    BASELINE_SHORT,
    BASELINE_LONG,
    POST_1_SHORT,
    POST_1_LONG,
    POST_500_SHORT,
    POST_500_LONG,
)
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
        description="Fixed strict-gate audit of short-to-long PPO warm starting."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--short-motion-file", required=True)
    parser.add_argument("--long-motion-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainer", default=str(DEFAULT_TRAINER))
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR))
    parser.add_argument("--training-log-root", default=str(DEFAULT_TRAINING_LOG_ROOT))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--training-updates", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--ppo-output", default="delta-all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timeout-seconds", type=int, default=7200)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=300)
    parser.add_argument("--render-evaluations", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.task != "Tracking-Flat-G1-v0":
        parser.error("the warm-start audit requires Tracking-Flat-G1-v0")
    if args.ppo_output != "delta-all":
        parser.error("the warm-start audit requires the delta-all action contract")
    if args.training_updates != 500:
        parser.error("the predeclared audit requires exactly 500 PPO updates")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if args.training_timeout_seconds < 1:
        parser.error("--training-timeout-seconds must be positive")
    if args.evaluation_timeout_seconds < 1:
        parser.error("--evaluation-timeout-seconds must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name):
        parser.error("--run-name must contain only letters, digits, '.', '_', or '-'")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _artifact_record(path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _child_environment(training_log_root: Path) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(name for name in AMBIENT_TOGGLES if name in environment)
    for name in AMBIENT_TOGGLES:
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    # Used by the process-level fake trainer; ignored by the stable trainer.
    environment["DIFFSIM_WARMSTART_LOG_ROOT"] = str(training_log_root)
    return environment, removed


def _evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    motion_file: Path,
    outcome_label_set: str,
    arm_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator),
        "--task",
        args.task,
        "--motion-file",
        str(motion_file),
        "--checkpoint-path",
        str(checkpoint),
        "--normalizer-checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str(arm_dir / "source_eval"),
        "--episodes",
        str(args.eval_episodes),
        "--eval-seed",
        str(args.eval_seed),
        "--outcome-label-set",
        outcome_label_set,
        "--ppo-output",
        args.ppo_output,
        "--device",
        args.device,
    ]
    if args.render_evaluations:
        command.append("--render")
    if args.headless:
        command.append("--headless")
    return command


def _evaluate_arm(
    args: argparse.Namespace,
    *,
    arm_name: str,
    checkpoint: Path,
    motion_file: Path,
    outcome_label_set: str,
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    arm_dir = args.output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = _evaluation_command(
        args,
        checkpoint=checkpoint,
        motion_file=motion_file,
        outcome_label_set=outcome_label_set,
        arm_dir=arm_dir,
    )
    stdout_path = arm_dir / "launcher_stdout.log"
    stderr_path = arm_dir / "launcher_stderr.log"
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    timed_out = False
    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=args.evaluation_timeout_seconds,
                check=False,
                start_new_session=True,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
    duration_seconds = time.monotonic() - monotonic_start

    checkpoint_sha256 = _sha256_file(checkpoint)
    motion_sha256 = _sha256_file(motion_file)
    result_path = arm_dir / "source_eval" / "result.json"
    arm_result: dict[str, Any] | None = None
    contract_errors: list[str] = []
    if return_code != 0:
        contract_errors.append(f"child return code {return_code}")
    if timed_out:
        contract_errors.append("child timed out")
    if result_path.is_file():
        try:
            arm_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            contract_errors.append(f"invalid arm result: {error}")
    else:
        contract_errors.append("arm result.json missing")

    outcome = "invalid-execution"
    qpos_hashes: list[str] = []
    qvel_hashes: list[str] = []
    observation_hashes: list[str] = []
    if arm_result is not None:
        outcome = str(arm_result.get("outcome", "invalid-execution"))
        if outcome not in EXPECTED_OUTCOMES[arm_name]:
            contract_errors.append(f"unexpected outcome label {outcome!r}")
        inputs = arm_result.get("inputs", {})
        actor_record = inputs.get("checkpoint", {})
        normalizer_record = inputs.get("normalizer_checkpoint", {})
        if actor_record.get("sha256") != checkpoint_sha256:
            contract_errors.append("actor checkpoint identity mismatch")
        if normalizer_record.get("sha256") != checkpoint_sha256:
            contract_errors.append("normalizer checkpoint identity mismatch")
        for record_name, record in (
            ("actor", actor_record),
            ("normalizer", normalizer_record),
        ):
            try:
                recorded_path = Path(str(record.get("path"))).resolve()
            except (OSError, TypeError):
                recorded_path = Path("/")
            if recorded_path != checkpoint:
                contract_errors.append(f"{record_name} checkpoint path mismatch")
        motion_record = inputs.get("motion")
        if motion_record is not None and motion_record.get("sha256") != motion_sha256:
            contract_errors.append("motion identity mismatch")
        evaluation_contract = arm_result.get("evaluation_contract", {})
        if evaluation_contract.get("normalizer_override_applied") is not False:
            contract_errors.append("native package unexpectedly overrides normalizer")
        recorded_label_set = evaluation_contract.get("outcome_label_set")
        if recorded_label_set is not None and recorded_label_set != outcome_label_set:
            contract_errors.append("outcome label set mismatch")
        classification = arm_result.get("classification")
        if (
            classification is not None
            and classification.get("contract_valid") is not True
        ):
            contract_errors.append("child scientific contract invalid")
        episodes = arm_result.get("episodes", [])
        if not isinstance(episodes, list) or len(episodes) != args.eval_episodes:
            contract_errors.append("episode count mismatch")
            episodes = []
        for episode in episodes:
            for key, destination in (
                ("initial_qpos_sha256", qpos_hashes),
                ("initial_qvel_sha256", qvel_hashes),
                ("initial_policy_observation_sha256", observation_hashes),
            ):
                value = episode.get(key)
                if not isinstance(value, str) or not value:
                    contract_errors.append(f"missing episode {key}")
                else:
                    destination.append(value)
        if arm_result.get("all_numeric_finite") is not True:
            contract_errors.append("nonfinite arm telemetry")
    if contract_errors:
        outcome = "invalid-execution"

    record = {
        "arm": arm_name,
        "motion_regime": "short" if outcome_label_set == "short-control" else "long",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
        "motion": {"path": str(motion_file), "sha256": motion_sha256},
        "outcome_label_set": outcome_label_set,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": duration_seconds,
        "return_code": return_code,
        "timed_out": timed_out,
        "outcome": outcome,
        "contract_errors": contract_errors,
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path) if result_path.is_file() else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    print(
        f"[WARM-START] arm={arm_name} outcome={outcome} "
        f"return_code={return_code} duration={duration_seconds:.3f}s",
        flush=True,
    )
    return record, qpos_hashes, qvel_hashes, observation_hashes


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


def _run_training(
    args: argparse.Namespace,
    *,
    environment: dict[str, str],
    removed_environment_toggles: list[str],
    source_sha256_before: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    training_dir = args.output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path = training_dir / "combined.log"
    command = _training_command(args)
    before_dirs = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    timed_out = False
    launch_error: str | None = None
    return_code = 127

    with combined_log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
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
            process = None
            launch_error = str(error)
            log_handle.write(f"launcher error: {error}\n")

        if process is not None:
            lines: queue.Queue[str | None] = queue.Queue()

            def _read_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()
            deadline = monotonic_start + args.training_timeout_seconds
            reader_finished = False
            while not (reader_finished and process.poll() is not None):
                try:
                    line = lines.get(timeout=0.5)
                    if line is None:
                        reader_finished = True
                    else:
                        log_handle.write(line)
                        print(line, end="", flush=True)
                except queue.Empty:
                    pass
                if process.poll() is None and time.monotonic() >= deadline:
                    timed_out = True
                    _terminate_process_group(process)
            reader.join(timeout=1)
            return_code = process.wait()

    duration_seconds = time.monotonic() - monotonic_start
    source_sha256_after = _sha256_file(args.source_checkpoint)
    source_unchanged = source_sha256_after == source_sha256_before
    suffix = f"_{args.run_name}"
    after_dirs = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    candidates = sorted(
        path for path in after_dirs - before_dirs if path.name.endswith(suffix)
    )
    contract_errors: list[str] = []
    if launch_error is not None:
        contract_errors.append(f"trainer launch failed: {launch_error}")
    if return_code != 0:
        contract_errors.append(f"trainer return code {return_code}")
    if timed_out:
        contract_errors.append("trainer timed out")
    if not source_unchanged:
        contract_errors.append("immutable source checkpoint changed")
    if len(candidates) != 1:
        contract_errors.append(
            f"expected one new namespaced training directory, found {len(candidates)}"
        )

    checkpoints: dict[str, Path] = {}
    run_dir = candidates[0] if len(candidates) == 1 else None
    if run_dir is not None:
        expected_paths = {
            "post_update_1": run_dir / "model_500.pt",
            "post_update_500": run_dir / "model_999.pt",
            "env_config": run_dir / "params" / "env.yaml",
            "agent_config": run_dir / "params" / "agent.yaml",
        }
        for name, path in expected_paths.items():
            if not path.is_file():
                contract_errors.append(f"missing {name}: {path}")
        checkpoints = {
            name: path
            for name, path in expected_paths.items()
            if name.startswith("post_update") and path.is_file()
        }
    record = {
        "executed": True,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": duration_seconds,
        "return_code": return_code,
        "timed_out": timed_out,
        "combined_log_path": str(combined_log_path),
        "training_log_root": str(args.training_log_root),
        "new_run_candidates": [str(path) for path in candidates],
        "run_dir": str(run_dir) if run_dir is not None else None,
        "source_checkpoint_sha256_before": source_sha256_before,
        "source_checkpoint_sha256_after": source_sha256_after,
        "source_checkpoint_unchanged": source_unchanged,
        "removed_ambient_environment_toggles": removed_environment_toggles,
        "checkpoint_semantics": {
            "model_500.pt": "first saved package after resumed PPO update 1",
            "model_999.pt": "final package after resumed PPO update 500",
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in checkpoints.items()
        },
        "contract_errors": contract_errors,
    }
    print(
        f"[WARM-START] training return_code={return_code} timed_out={timed_out} "
        f"duration={duration_seconds:.3f}s run_dir={record['run_dir']}",
        flush=True,
    )
    return record, checkpoints


def _finalize(
    args: argparse.Namespace,
    *,
    started_at: str,
    arm_records: list[dict[str, Any]],
    arm_outcomes: dict[str, str],
    qpos_hashes: list[str],
    qvel_hashes: list[str],
    observation_hashes: list[str],
    training_record: dict[str, Any],
    source_sha256_before: str,
    removed_environment_toggles: list[str],
) -> int:
    training_executed = bool(training_record["executed"])
    expected_arm_count = 6 if training_executed else 2
    expected_hash_count = expected_arm_count * args.eval_episodes
    identical_qpos = (
        len(qpos_hashes) == expected_hash_count and len(set(qpos_hashes)) == 1
    )
    identical_qvel = (
        len(qvel_hashes) == expected_hash_count and len(set(qvel_hashes)) == 1
    )
    identical_observation = (
        len(observation_hashes) == expected_hash_count
        and len(set(observation_hashes)) == 1
    )
    all_initial_hashes_identical = (
        identical_qpos and identical_qvel and identical_observation
    )
    classification = classify_warm_start(
        arm_outcomes, training_executed=training_executed
    )
    execution_errors = list(training_record.get("contract_errors", []))
    if not all_initial_hashes_identical:
        execution_errors.append("initial reset identity mismatch or missing hashes")
    if training_record.get("source_checkpoint_unchanged") is not True:
        execution_errors.append("immutable source checkpoint identity mismatch")
    if execution_errors:
        classification = {
            **classification,
            "outcome": "invalid-execution",
            "execution_errors": execution_errors,
        }

    arms_path = args.output_dir / "arms.json"
    training_path = args.output_dir / "training.json"
    _write_json(arms_path, {"arms": arm_records})
    _write_json(training_path, training_record)
    result_path = args.output_dir / "result.json"
    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path != result_path and not path.name.endswith(".tmp")
    )
    result = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "outcome": classification["outcome"],
        "classification": classification,
        "inputs": {
            "source_checkpoint": {
                "path": str(args.source_checkpoint),
                "sha256": source_sha256_before,
            },
            "short_motion": {
                "path": str(args.short_motion_file),
                "sha256": _sha256_file(args.short_motion_file),
            },
            "long_motion": {
                "path": str(args.long_motion_file),
                "sha256": _sha256_file(args.long_motion_file),
            },
            "trainer": {
                "path": str(args.trainer),
                "sha256": _sha256_file(args.trainer),
            },
            "evaluator": {
                "path": str(args.evaluator),
                "sha256": _sha256_file(args.evaluator),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "run_py_sha256": _sha256_file(Path(__file__).resolve()),
                "contract_py_sha256": _sha256_file(SCRIPT_DIR / "contract.py"),
            },
        },
        "training_contract": {
            "task": args.task,
            "configured_task_change_only": "motion_file",
            "long_motion_file": str(args.long_motion_file),
            "resume_native_checkpoint_package": True,
            "restored_checkpoint_state": [
                "actor_and_critic_weights",
                "action_standard_deviation",
                "actor_observation_normalizer",
                "privileged_observation_normalizer",
                "optimizer",
                "learning_iteration",
            ],
            "reinitialized_runtime_state": [
                "environment",
                "rollout_storage",
                "pseudorandom_generators_from_train_seed",
            ],
            "optimizer_state_loaded_by_native_runner": True,
            "updates": args.training_updates,
            "num_envs": args.num_envs,
            "train_seed": args.train_seed,
            "ppo_output": args.ppo_output,
            "sampling": "adaptive",
            "device": args.device,
            "logger": "tensorboard",
            "training_timeout_seconds": args.training_timeout_seconds,
            "removed_ambient_environment_toggles": removed_environment_toggles,
        },
        "evaluation_contract": {
            "fixed_arm_order": list(FIXED_ARM_ORDER),
            "executed_arm_order": [record["arm"] for record in arm_records],
            "episodes_per_arm": args.eval_episodes,
            "eval_seed_reapplied_per_episode": args.eval_seed,
            "strict_native_training_task": args.task,
            "native_actor_normalizer_package_per_arm": True,
            "render": args.render_evaluations,
            "headless": args.headless,
            "evaluation_timeout_seconds": args.evaluation_timeout_seconds,
            "identical_initial_qpos": identical_qpos,
            "identical_initial_qvel": identical_qvel,
            "identical_initial_raw_observation": identical_observation,
            "all_initial_hashes_identical": all_initial_hashes_identical,
        },
        "training": training_record,
        "arms": arm_records,
        "artifacts": {
            str(path.relative_to(args.output_dir)): _artifact_record(
                path, args.output_dir
            )
            for path in artifact_paths
        },
        "claim_boundary": (
            "This fixed native-source audit tests whether switching a verified "
            "short-reference PPO package to the exact long reference preserves "
            "strict phase-zero short competence while learning the long task. It "
            "recreates the environment and seeded runtime rather than resuming an "
            "unserialized rollout or random-number-generator state. It "
            "does not isolate which PPO gradient component causes any measured "
            "change, establish sim-to-real transfer, or retain a deployment policy."
        ),
    }
    _write_json(result_path, result)
    print(f"[WARM-START] outcome={result['outcome']}", flush=True)
    print(f"[WARM-START] result={result_path}", flush=True)
    return 1 if result["outcome"] == "invalid-execution" else 0


def main() -> int:
    args = _parse_args()
    for name in (
        "source_checkpoint",
        "short_motion_file",
        "long_motion_file",
        "trainer",
        "evaluator",
        "training_log_root",
        "output_dir",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())

    for name in (
        "source_checkpoint",
        "short_motion_file",
        "long_motion_file",
        "trainer",
        "evaluator",
    ):
        if not getattr(args, name).is_file():
            raise FileNotFoundError(f"{name} does not exist: {getattr(args, name)}")
    if not args.training_log_root.is_dir():
        raise FileNotFoundError(
            f"training_log_root does not exist: {args.training_log_root}"
        )
    resolved_resume_path = (
        args.training_log_root
        / args.source_checkpoint.parent.name
        / args.source_checkpoint.name
    ).resolve()
    if resolved_resume_path != args.source_checkpoint:
        raise ValueError(
            "source checkpoint is not addressable by the trainer's load_run/checkpoint "
            f"contract: expected {resolved_resume_path}, got {args.source_checkpoint}"
        )
    if args.trainer == DEFAULT_TRAINER.resolve() and (
        args.training_log_root != DEFAULT_TRAINING_LOG_ROOT.resolve()
    ):
        raise ValueError(
            "the stable trainer derives its log root from the repository; "
            f"expected {DEFAULT_TRAINING_LOG_ROOT.resolve()}"
        )
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed result: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    source_sha256_before = _sha256_file(args.source_checkpoint)
    environment, removed_toggles = _child_environment(args.training_log_root)
    arm_records: list[dict[str, Any]] = []
    arm_outcomes: dict[str, str] = {}
    qpos_hashes: list[str] = []
    qvel_hashes: list[str] = []
    observation_hashes: list[str] = []

    def evaluate(
        arm_name: str,
        checkpoint: Path,
        motion_file: Path,
        outcome_label_set: str,
    ) -> None:
        record, qpos, qvel, observation = _evaluate_arm(
            args,
            arm_name=arm_name,
            checkpoint=checkpoint,
            motion_file=motion_file,
            outcome_label_set=outcome_label_set,
            environment=environment,
        )
        arm_records.append(record)
        arm_outcomes[arm_name] = record["outcome"]
        qpos_hashes.extend(qpos)
        qvel_hashes.extend(qvel)
        observation_hashes.extend(observation)

    evaluate(
        BASELINE_SHORT,
        args.source_checkpoint,
        args.short_motion_file,
        "short-control",
    )
    evaluate(
        BASELINE_LONG,
        args.source_checkpoint,
        args.long_motion_file,
        "exact-long",
    )

    source_sha256_after_baseline = _sha256_file(args.source_checkpoint)
    empty_training_record: dict[str, Any] = {
        "executed": False,
        "command": None,
        "source_checkpoint_sha256_before": source_sha256_before,
        "source_checkpoint_sha256_after": source_sha256_after_baseline,
        "source_checkpoint_unchanged": (
            source_sha256_after_baseline == source_sha256_before
        ),
        "removed_ambient_environment_toggles": removed_toggles,
        "contract_errors": [],
    }
    baseline_short = arm_outcomes[BASELINE_SHORT]
    baseline_long = arm_outcomes[BASELINE_LONG]
    if baseline_short != SHORT_COMPLETE or baseline_long in {
        LONG_COMPLETE,
        LONG_MIXED,
    }:
        return _finalize(
            args,
            started_at=started_at,
            arm_records=arm_records,
            arm_outcomes=arm_outcomes,
            qpos_hashes=qpos_hashes,
            qvel_hashes=qvel_hashes,
            observation_hashes=observation_hashes,
            training_record=empty_training_record,
            source_sha256_before=source_sha256_before,
            removed_environment_toggles=removed_toggles,
        )
    if baseline_long != LONG_FAIL:
        empty_training_record["contract_errors"] = [
            f"invalid baseline long outcome: {baseline_long}"
        ]
        return _finalize(
            args,
            started_at=started_at,
            arm_records=arm_records,
            arm_outcomes=arm_outcomes,
            qpos_hashes=qpos_hashes,
            qvel_hashes=qvel_hashes,
            observation_hashes=observation_hashes,
            training_record=empty_training_record,
            source_sha256_before=source_sha256_before,
            removed_environment_toggles=removed_toggles,
        )

    training_record, checkpoints = _run_training(
        args,
        environment=environment,
        removed_environment_toggles=removed_toggles,
        source_sha256_before=source_sha256_before,
    )
    if training_record["contract_errors"]:
        return _finalize(
            args,
            started_at=started_at,
            arm_records=arm_records,
            arm_outcomes=arm_outcomes,
            qpos_hashes=qpos_hashes,
            qvel_hashes=qvel_hashes,
            observation_hashes=observation_hashes,
            training_record=training_record,
            source_sha256_before=source_sha256_before,
            removed_environment_toggles=removed_toggles,
        )

    post_1 = checkpoints["post_update_1"]
    post_500 = checkpoints["post_update_500"]
    for arm_name, checkpoint, motion_file, label_set in (
        (POST_1_SHORT, post_1, args.short_motion_file, "short-control"),
        (POST_1_LONG, post_1, args.long_motion_file, "exact-long"),
        (POST_500_SHORT, post_500, args.short_motion_file, "short-control"),
        (POST_500_LONG, post_500, args.long_motion_file, "exact-long"),
    ):
        evaluate(arm_name, checkpoint, motion_file, label_set)

    return _finalize(
        args,
        started_at=started_at,
        arm_records=arm_records,
        arm_outcomes=arm_outcomes,
        qpos_hashes=qpos_hashes,
        qvel_hashes=qvel_hashes,
        observation_hashes=observation_hashes,
        training_record=training_record,
        source_sha256_before=source_sha256_before,
        removed_environment_toggles=removed_toggles,
    )


if __name__ == "__main__":
    raise SystemExit(main())
