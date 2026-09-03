"""Run the fixed one-update corrected initial-observation resume test."""

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
from contract import (
    BASELINE_LONG,
    BASELINE_SHORT,
    EXPECTED_OUTCOMES,
    LONG_FAIL,
    POST_LONG,
    POST_SHORT,
    SHORT_COMPLETE,
    classify_corrected_resume,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER = SCRIPT_DIR / "train.py"
DEFAULT_EVALUATOR = REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
DEFAULT_TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
FIXED_ARM_ORDER = (BASELINE_SHORT, BASELINE_LONG, POST_SHORT, POST_LONG)
AMBIENT_TOGGLES = (
    "ENABLE_CAMERAS", "LOCAL_RANK", "RANK", "WORLD_SIZE", "WBT_CURRICULUM",
    "WBT_DEPTH_DEBUG_MAX_FRAMES", "WBT_DEPTH_SAVE_FRAMES", "WBT_DOUBLE_STEP",
    "WBT_MOTION_JOINT_POS", "WBT_STAIR_PHASE_GRACE", "WBT_STAIR_PHASE_MIN_STEPS",
    "WBT_STAIR_PHASE_TERM", "WBT_USE_DEPTH_OBS", "WBT_VIDEO", "WBT_VIDEO_INTERVAL",
    "WBT_VIDEO_LENGTH", "BONES_DOUBLE_STEP", "BONES_GRAVITY_CURRICULUM",
    "BONES_GRAVITY_RAMP_STEPS", "BONES_START_GRAVITY",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct only the first resumed actor/critic observation, then run one PPO update."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--short-motion-file", required=True)
    parser.add_argument("--short-motion-sha256", required=True)
    parser.add_argument("--long-motion-file", required=True)
    parser.add_argument("--long-motion-sha256", required=True)
    parser.add_argument("--unmodified-control-checkpoint", required=True)
    parser.add_argument("--unmodified-control-checkpoint-sha256", required=True)
    parser.add_argument("--unmodified-control-result", required=True)
    parser.add_argument("--unmodified-control-result-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trainer", default=str(DEFAULT_TRAINER))
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR))
    parser.add_argument("--training-log-root", default=str(DEFAULT_TRAINING_LOG_ROOT))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--training-updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--ppo-output", default="delta-all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-timeout-seconds", type=int, default=900)
    parser.add_argument("--evaluation-timeout-seconds", type=int, default=300)
    parser.add_argument("--render-evaluations", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.task != "Tracking-Flat-G1-v0":
        parser.error("this discriminator requires Tracking-Flat-G1-v0")
    if args.ppo_output != "delta-all":
        parser.error("this discriminator requires delta-all")
    if args.training_updates != 1:
        parser.error("this discriminator requires exactly one PPO update")
    if args.num_envs != 4096:
        parser.error("this discriminator requires exactly 4096 environments")
    if args.train_seed != 42 or args.eval_seed != 0 or args.eval_episodes != 3:
        parser.error("this discriminator requires train seed 42 and three eval seed-zero episodes")
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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_record(path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "path": str(path), "relative_path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size, "sha256": _sha256_file(path),
    }


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _child_environment(training_log_root: Path) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(name for name in AMBIENT_TOGGLES if name in environment)
    for name in AMBIENT_TOGGLES:
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["DIFFSIM_CORRECTED_RESUME_LOG_ROOT"] = str(training_log_root)
    return environment, removed


def _evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    motion_file: Path,
    label_set: str,
    arm_dir: Path,
) -> list[str]:
    command = [
        sys.executable, str(args.evaluator), "--task", args.task,
        "--motion-file", str(motion_file), "--checkpoint-path", str(checkpoint),
        "--normalizer-checkpoint-path", str(checkpoint), "--output-dir",
        str(arm_dir / "source_eval"), "--episodes", str(args.eval_episodes),
        "--eval-seed", str(args.eval_seed), "--outcome-label-set", label_set,
        "--ppo-output", args.ppo_output, "--device", args.device,
    ]
    if args.render_evaluations:
        command.append("--render")
    if args.headless:
        command.append("--headless")
    return command


def _evaluate_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    checkpoint: Path,
    motion_file: Path,
    label_set: str,
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    arm_dir = args.output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = _evaluation_command(
        args, checkpoint=checkpoint, motion_file=motion_file, label_set=label_set, arm_dir=arm_dir
    )
    stdout_path = arm_dir / "launcher_stdout.log"
    stderr_path = arm_dir / "launcher_stderr.log"
    started = _utc_now()
    monotonic_start = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, stdout=stdout, stderr=stderr,
                timeout=args.evaluation_timeout_seconds, check=False, start_new_session=True,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            timed_out = True
    result_path = arm_dir / "source_eval" / "result.json"
    errors: list[str] = []
    if return_code != 0:
        errors.append(f"child return code {return_code}")
    if timed_out:
        errors.append("child timed out")
    result = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid child result: {error}")
    else:
        errors.append("child result missing")
    outcome = "invalid-execution"
    qpos: list[str] = []
    qvel: list[str] = []
    observations: list[str] = []
    checkpoint_sha = _sha256_file(checkpoint)
    motion_sha = _sha256_file(motion_file)
    if result is not None:
        outcome = str(result.get("outcome", "invalid-execution"))
        if outcome not in EXPECTED_OUTCOMES[arm]:
            errors.append(f"unexpected outcome {outcome}")
        inputs = result.get("inputs", {})
        for key in ("checkpoint", "normalizer_checkpoint"):
            if inputs.get(key, {}).get("sha256") != checkpoint_sha:
                errors.append(f"{key} hash mismatch")
            if Path(str(inputs.get(key, {}).get("path", "/"))).resolve() != checkpoint:
                errors.append(f"{key} path mismatch")
        if inputs.get("motion", {}).get("sha256") != motion_sha:
            errors.append("motion hash mismatch")
        child_contract = result.get("evaluation_contract", {})
        if child_contract.get("normalizer_override_applied") is not False:
            errors.append("normalizer override detected")
        if result.get("classification", {}).get("contract_valid") is not True:
            errors.append("child scientific contract invalid")
        episodes = result.get("episodes", [])
        if not isinstance(episodes, list) or len(episodes) != args.eval_episodes:
            errors.append("episode count mismatch")
            episodes = []
        for episode in episodes:
            qpos.append(str(episode.get("initial_qpos_sha256", "")))
            qvel.append(str(episode.get("initial_qvel_sha256", "")))
            observations.append(str(episode.get("initial_policy_observation_sha256", "")))
        if not all(qpos + qvel + observations) or result.get("all_numeric_finite") is not True:
            errors.append("missing reset identity or nonfinite telemetry")
    if errors:
        outcome = "invalid-execution"
    record = {
        "arm": arm, "motion_regime": "short" if label_set == "short-control" else "long",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "motion": {"path": str(motion_file), "sha256": motion_sha},
        "outcome_label_set": label_set, "command": command, "started_at": started,
        "finished_at": _utc_now(), "duration_seconds": time.monotonic() - monotonic_start,
        "return_code": return_code, "timed_out": timed_out, "outcome": outcome,
        "contract_errors": errors, "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path) if result_path.is_file() else None,
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
    }
    print(f"[CORRECTED-RESUME] arm={arm} outcome={outcome} rc={return_code}", flush=True)
    return record, qpos, qvel, observations


def _training_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(args.trainer), "--task", args.task, "--motion_file",
        str(args.long_motion_file), "--logger", "tensorboard", "--run_name", args.run_name,
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.training_updates),
        "--seed", str(args.train_seed), "--ppo_output", args.ppo_output,
        "--sampling", "adaptive", "--device", args.device, "--resume", "True",
        "--load_run", args.source_checkpoint.parent.name, "--checkpoint",
        args.source_checkpoint.name,
    ]
    if args.headless:
        command.append("--headless")
    return command


def _validate_training_log_iterations(log_text: str) -> list[str]:
    iterations = [int(value) for value in re.findall(r"Learning iteration (\d+)/\d+", log_text)]
    if iterations != [500]:
        return [f"expected exactly native iteration 500 once, observed {iterations}"]
    return []


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


def _optimizer_steps(payload: dict[str, Any]) -> list[int]:
    return sorted({int(item["step"]) for item in payload["optimizer_state_dict"]["state"].values()})


def _run_training(
    args: argparse.Namespace,
    *,
    environment: dict[str, str],
    removed_toggles: list[str],
    source_sha_before: str,
    source_payload: dict[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    training_dir = args.output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "combined.log"
    command = _training_command(args)
    before = {path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()}
    started = _utc_now()
    monotonic_start = time.monotonic()
    timed_out = False
    launch_error = None
    return_code = 127
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        try:
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            )
        except OSError as error:
            process = None
            launch_error = str(error)
            log.write(f"launcher error: {error}\n")
        if process is not None:
            lines: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    lines.put(line)
                lines.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            deadline = monotonic_start + args.training_timeout_seconds
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
    source_sha_after = _sha256_file(args.source_checkpoint)
    after = {path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()}
    candidates = sorted(path for path in after - before if path.name.endswith(f"_{args.run_name}"))
    errors: list[str] = []
    if launch_error:
        errors.append(f"trainer launch failed: {launch_error}")
    if return_code != 0:
        errors.append(f"trainer return code {return_code}")
    if timed_out:
        errors.append("trainer timed out")
    if source_sha_after != source_sha_before:
        errors.append("source checkpoint changed")
    if len(candidates) != 1:
        errors.append(f"expected one new run directory, observed {len(candidates)}")
    run_dir = candidates[0] if len(candidates) == 1 else None
    checkpoint = run_dir / "model_500.pt" if run_dir is not None else None
    for path_name, path in {
        "checkpoint": checkpoint,
        "env config": run_dir / "params" / "env.yaml" if run_dir else None,
        "agent config": run_dir / "params" / "agent.yaml" if run_dir else None,
    }.items():
        if path is None or not path.is_file():
            errors.append(f"missing {path_name}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    errors.extend(_validate_training_log_iterations(log_text))
    if log_text.count("[RESUME-INITIAL-NORMALIZATION]") != 1:
        errors.append("corrected initial-normalization marker not observed exactly once")
    package_metadata = None
    if checkpoint is not None and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        actor_count = int(payload["obs_norm_state_dict"]["count"])
        privileged_count = int(payload["privileged_obs_norm_state_dict"]["count"])
        source_count = int(source_payload["obs_norm_state_dict"]["count"])
        source_steps = _optimizer_steps(source_payload)
        steps = _optimizer_steps(payload)
        package_metadata = {
            "iter": payload.get("iter"), "actor_normalizer_count": actor_count,
            "privileged_normalizer_count": privileged_count, "optimizer_steps": steps,
            "model_entries": len(payload["model_state_dict"]),
        }
        if payload.get("iter") != 500:
            errors.append(f"checkpoint iter mismatch: {payload.get('iter')}")
        if actor_count != source_count + 4096 * 24 or privileged_count != source_count + 4096 * 24:
            errors.append("normalizer transition count mismatch")
        if len(source_steps) != 1 or steps != [source_steps[0] + 20]:
            errors.append(f"optimizer step mismatch: source={source_steps}, new={steps}")
        tensors = [value for mapping in (
            payload["model_state_dict"], payload["obs_norm_state_dict"],
            payload["privileged_obs_norm_state_dict"],
        ) for value in mapping.values() if isinstance(value, torch.Tensor)]
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            errors.append("nonfinite checkpoint tensor")
    record = {
        "executed": True, "command": command, "started_at": started, "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - monotonic_start, "return_code": return_code,
        "timed_out": timed_out, "combined_log_path": str(log_path),
        "training_log_root": str(args.training_log_root),
        "new_run_candidates": [str(path) for path in candidates],
        "run_dir": str(run_dir) if run_dir else None,
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "source_checkpoint_unchanged": source_sha_after == source_sha_before,
        "removed_ambient_environment_toggles": removed_toggles,
        "checkpoint_semantics": {"model_500.pt": "package after corrected resumed PPO update 1"},
        "checkpoint": ({"path": str(checkpoint), "sha256": _sha256_file(checkpoint)}
                       if checkpoint is not None and checkpoint.is_file() else None),
        "package_metadata": package_metadata, "contract_errors": errors,
    }
    print(f"[CORRECTED-RESUME] training rc={return_code} errors={errors}", flush=True)
    return record, checkpoint


def _finalize(
    args: argparse.Namespace,
    *,
    started_at: str,
    arm_records: list[dict[str, Any]],
    arm_outcomes: dict[str, str],
    qpos_hashes: list[str],
    qvel_hashes: list[str],
    observation_hashes: list[str],
    training: dict[str, Any],
    source_sha: str,
    control_result: dict[str, Any],
) -> int:
    classification = classify_corrected_resume(arm_outcomes)
    identity = (
        len(qpos_hashes) == len(qvel_hashes) == len(observation_hashes) == 12
        and len(set(qpos_hashes)) == len(set(qvel_hashes)) == len(set(observation_hashes)) == 1
    )
    execution_errors = list(training.get("contract_errors", []))
    execution_errors.extend(error for record in arm_records for error in record["contract_errors"])
    if not identity:
        execution_errors.append("initial reset identities differ or are incomplete")
    if execution_errors:
        classification = {**classification, "outcome": "invalid-execution", "execution_errors": execution_errors}
    arms_path = args.output_dir / "arms.json"
    training_path = args.output_dir / "training.json"
    _write_json(arms_path, {"arms": arm_records})
    _write_json(training_path, training)
    result_path = args.output_dir / "result.json"
    artifact_paths = sorted(
        path for path in args.output_dir.rglob("*")
        if path.is_file() and path != result_path and not path.name.endswith(".tmp")
    )
    code_files = ["run.py", "contract.py", "train.py", "runner.py", "normalization.py"]
    result = {
        "schema_version": 1, "started_at": started_at, "completed_at": _utc_now(),
        "outcome": classification["outcome"], "classification": classification,
        "inputs": {
            "source_checkpoint": {"path": str(args.source_checkpoint), "sha256": source_sha},
            "short_motion": {"path": str(args.short_motion_file), "sha256": _sha256_file(args.short_motion_file)},
            "long_motion": {"path": str(args.long_motion_file), "sha256": _sha256_file(args.long_motion_file)},
            "unmodified_control_checkpoint": {
                "path": str(args.unmodified_control_checkpoint),
                "sha256": _sha256_file(args.unmodified_control_checkpoint),
            },
            "unmodified_control_result": {
                "path": str(args.unmodified_control_result),
                "sha256": _sha256_file(args.unmodified_control_result),
                "verdict": control_result.get("verdict"),
                "selected_outcome": control_result.get("selected_outcome"),
            },
            "evaluator": {"path": str(args.evaluator), "sha256": _sha256_file(args.evaluator)},
            "code": {
                "repository": str(REPO_ROOT), "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "files": {name: _sha256_file(SCRIPT_DIR / name) for name in code_files},
            },
        },
        "causal_change": (
            "Only the first actor and privileged observations returned to the loaded runner are passed through "
            "their restored empirical normalizers, in eval mode so counts remain unchanged; all subsequent loop semantics use the pinned stable trainer."
        ),
        "training_contract": {
            "task": args.task, "updates": 1, "num_envs": 4096, "rollout_steps_per_env": 24,
            "new_transitions": 98_304, "train_seed": 42, "ppo_output": "delta-all",
            "sampling": "adaptive", "device": args.device, "resume_native_package": True,
            "initial_normalization_updates_statistics": False,
        },
        "evaluation_contract": {
            "fixed_arm_order": list(FIXED_ARM_ORDER),
            "executed_arm_order": [record["arm"] for record in arm_records],
            "episodes_per_arm": 3, "eval_seed_reapplied_per_episode": 0,
            "native_actor_normalizer_package_per_arm": True,
            "identical_initial_qpos": identity, "identical_initial_qvel": identity,
            "identical_initial_raw_observation": identity,
            "render": args.render_evaluations, "headless": args.headless,
        },
        "unmodified_control": {
            "experiment_id": "E-20260903-008", "post_update_1_short_steps": [41, 41, 41],
            "post_update_1_long_steps": [41, 41, 41],
            "selected_outcome": "immediate-short-retention-loss",
        },
        "training": training, "arms": arm_records,
        "artifacts": {str(path.relative_to(args.output_dir)): _artifact_record(path, args.output_dir)
                      for path in artifact_paths},
        "claim_boundary": (
            "This one-update PPO control can attribute a changed first-update outcome to the corrected initial-observation path "
            "under the pinned seed/configuration. It does not test continued PPO learning, AHAC, differentiable-physics training, or sim-to-real transfer, and retains no policy."
        ),
    }
    _write_json(result_path, result)
    print(f"[CORRECTED-RESUME] outcome={result['outcome']} result={result_path}", flush=True)
    return 1 if result["outcome"] == "invalid-execution" else 0


def main() -> int:
    args = _parse_args()
    path_names = (
        "source_checkpoint", "short_motion_file", "long_motion_file",
        "unmodified_control_checkpoint", "unmodified_control_result", "output_dir",
        "trainer", "evaluator", "training_log_root",
    )
    for name in path_names:
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    for name in path_names:
        if name in {"output_dir", "training_log_root"}:
            continue
        if not getattr(args, name).is_file():
            raise FileNotFoundError(f"{name} missing: {getattr(args, name)}")
    if not args.training_log_root.is_dir():
        raise FileNotFoundError(f"training log root missing: {args.training_log_root}")
    expected_hashes = {
        "source_checkpoint": args.source_checkpoint_sha256,
        "short_motion_file": args.short_motion_sha256,
        "long_motion_file": args.long_motion_sha256,
        "unmodified_control_checkpoint": args.unmodified_control_checkpoint_sha256,
        "unmodified_control_result": args.unmodified_control_result_sha256,
    }
    for name, expected in expected_hashes.items():
        observed = _sha256_file(getattr(args, name))
        if observed != expected:
            raise ValueError(f"{name} hash mismatch: expected {expected}, observed {observed}")
    control_result = json.loads(args.unmodified_control_result.read_text(encoding="utf-8"))
    if control_result.get("verdict") != "pass" or control_result.get("selected_outcome") != "immediate-short-retention-loss":
        raise ValueError("unmodified E008 control is not a passed immediate-loss result")
    if (args.training_log_root / args.source_checkpoint.parent.name / args.source_checkpoint.name).resolve() != args.source_checkpoint:
        raise ValueError("source checkpoint is not addressable by load_run/checkpoint")
    if (args.output_dir / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite completed result: {args.output_dir / 'result.json'}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256_file(args.source_checkpoint)
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    environment, removed = _child_environment(args.training_log_root)
    started = _utc_now()
    records: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    qpos: list[str] = []
    qvel: list[str] = []
    observations: list[str] = []

    def evaluate(arm: str, checkpoint: Path, motion: Path, label: str) -> None:
        record, arm_qpos, arm_qvel, arm_obs = _evaluate_arm(
            args, arm=arm, checkpoint=checkpoint, motion_file=motion,
            label_set=label, environment=environment,
        )
        records.append(record)
        outcomes[arm] = record["outcome"]
        qpos.extend(arm_qpos)
        qvel.extend(arm_qvel)
        observations.extend(arm_obs)

    evaluate(BASELINE_SHORT, args.source_checkpoint, args.short_motion_file, "short-control")
    evaluate(BASELINE_LONG, args.source_checkpoint, args.long_motion_file, "exact-long")
    empty_training: dict[str, Any] = {
        "executed": False, "contract_errors": ["baseline did not authorize training"],
        "source_checkpoint_sha256_before": source_sha,
        "source_checkpoint_sha256_after": _sha256_file(args.source_checkpoint),
        "source_checkpoint_unchanged": _sha256_file(args.source_checkpoint) == source_sha,
    }
    if outcomes.get(BASELINE_SHORT) != SHORT_COMPLETE or outcomes.get(BASELINE_LONG) != LONG_FAIL:
        return _finalize(
            args, started_at=started, arm_records=records, arm_outcomes=outcomes,
            qpos_hashes=qpos, qvel_hashes=qvel, observation_hashes=observations,
            training=empty_training, source_sha=source_sha, control_result=control_result,
        )
    training, checkpoint = _run_training(
        args, environment=environment, removed_toggles=removed,
        source_sha_before=source_sha, source_payload=source_payload,
    )
    if checkpoint is not None and not training["contract_errors"]:
        evaluate(POST_SHORT, checkpoint, args.short_motion_file, "short-control")
        evaluate(POST_LONG, checkpoint, args.long_motion_file, "exact-long")
    return _finalize(
        args, started_at=started, arm_records=records, arm_outcomes=outcomes,
        qpos_hashes=qpos, qvel_hashes=qvel, observation_hashes=observations,
        training=training, source_sha=source_sha, control_result=control_result,
    )


if __name__ == "__main__":
    raise SystemExit(main())
