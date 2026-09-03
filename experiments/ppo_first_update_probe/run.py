"""Run and evaluate the fixed measurement-only first-PPO-update probe."""

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
from contract import classify_checkpoint_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINER = SCRIPT_DIR / "train.py"
DEFAULT_EVALUATOR = REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
DEFAULT_TRAINING_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "g1_flat"
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
        description="Measure every batch of one corrected-order native PPO update."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--short-motion-file", required=True)
    parser.add_argument("--short-motion-sha256", required=True)
    parser.add_argument("--long-motion-file", required=True)
    parser.add_argument("--long-motion-sha256", required=True)
    parser.add_argument("--e009-audit", required=True)
    parser.add_argument("--e009-audit-sha256", required=True)
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
    parser.add_argument("--render-selected-failure", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.task != "Tracking-Flat-G1-v0" or args.ppo_output != "delta-all":
        parser.error("this probe requires Tracking-Flat-G1-v0 with delta-all")
    if args.training_updates != 1 or args.num_envs != 4096 or args.train_seed != 42:
        parser.error(
            "this probe requires one update, 4096 environments, and train seed 42"
        )
    if args.eval_seed != 0:
        parser.error("this probe requires deterministic evaluation seed zero")
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


def _package_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "iter": payload.get("iter"),
        "optimizer_steps": _optimizer_steps(payload),
        "actor_normalizer_count": int(payload["obs_norm_state_dict"]["count"]),
        "critic_normalizer_count": int(
            payload["privileged_obs_norm_state_dict"]["count"]
        ),
        "model_entries": len(payload["model_state_dict"]),
        "optimizer_entries": len(payload["optimizer_state_dict"]["state"]),
    }


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
    *, training_log_root: Path, probe_dir: Path
) -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = sorted(name for name in AMBIENT_TOGGLES if name in environment)
    for name in AMBIENT_TOGGLES:
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["DIFFSIM_FIRST_UPDATE_PROBE_DIR"] = str(probe_dir)
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


def _run_training(
    args: argparse.Namespace,
    *,
    environment: dict[str, str],
    removed_toggles: list[str],
    source_sha_before: str,
    source_payload: dict[str, Any],
) -> tuple[dict[str, Any], Path | None, dict[str, Any] | None]:
    training_dir = args.output_dir / "training"
    probe_dir = training_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    log_path = training_dir / "combined.log"
    command = _training_command(args)
    before = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    started_at = _utc_now()
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
            deadline = started + args.training_timeout_seconds
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
    after = {
        path.resolve() for path in args.training_log_root.iterdir() if path.is_dir()
    }
    candidates = sorted(
        path for path in after - before if path.name.endswith(f"_{args.run_name}")
    )
    run_dir = candidates[0] if len(candidates) == 1 else None
    final_checkpoint = run_dir / "model_500.pt" if run_dir else None
    probe_result_path = probe_dir / "probe_result.json"
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
    if final_checkpoint is None or not final_checkpoint.is_file():
        errors.append("native final checkpoint missing")
    if not probe_result_path.is_file():
        errors.append("probe result missing")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    iterations = [
        int(value) for value in re.findall(r"Learning iteration (\d+)/\d+", log_text)
    ]
    if iterations != [500]:
        errors.append(
            f"expected only native learning iteration 500, observed {iterations}"
        )
    if log_text.count("[RESUME-INITIAL-NORMALIZATION]") != 1:
        errors.append("initial-normalization marker count mismatch")
    if len(re.findall(r"\[FIRST-UPDATE-PROBE\] step=\d+", log_text)) != 20:
        errors.append("optimizer-step marker count mismatch")

    probe_result = None
    if probe_result_path.is_file():
        try:
            probe_result = json.loads(probe_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid probe result: {error}")
    if probe_result is not None:
        if (
            probe_result.get("complete") is not True
            or probe_result.get("runner_completed") is not True
        ):
            errors.append("probe did not report complete runner execution")
        if probe_result.get("measurement_only") is not True:
            errors.append("probe did not report measurement-only semantics")
        trace = probe_result.get("optimizer_trace", [])
        if [item.get("optimizer_step") for item in trace] != list(range(1, 21)):
            errors.append("probe optimizer trace is incomplete or unordered")
        frozen = probe_result.get("frozen_gradient_analysis", {})
        if frozen.get("state_identity_before") != frozen.get("state_identity_after"):
            errors.append("gradient prepass state identity changed")
        if frozen.get("rng_identity_before") != frozen.get("rng_identity_after"):
            errors.append("gradient prepass RNG identity changed")
        checkpoint_records = probe_result.get("checkpoints", [])
        if [item.get("step") for item in checkpoint_records] != list(range(21)):
            errors.append("probe checkpoint manifest is incomplete or unordered")
        source_steps = _optimizer_steps(source_payload)
        if source_steps != [10020]:
            errors.append(f"source optimizer step changed: {source_steps}")
        for record in checkpoint_records:
            step = int(record.get("step", -1))
            path = Path(str(record.get("path", "/")))
            if not path.is_file() or _sha256_file(path) != record.get("sha256"):
                errors.append(f"checkpoint step {step} missing or hash-mismatched")
            if record.get("optimizer_steps") != [10020 + step]:
                errors.append(f"checkpoint step {step} optimizer count mismatch")

    package_metadata = None
    if (
        probe_result is not None
        and final_checkpoint is not None
        and final_checkpoint.is_file()
    ):
        try:
            step0_path = Path(probe_result["checkpoints"][0]["path"])
            step20_path = Path(probe_result["checkpoints"][20]["path"])
            step0 = torch.load(step0_path, map_location="cpu", weights_only=False)
            step20 = torch.load(step20_path, map_location="cpu", weights_only=False)
            final = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
            step0_meta = _package_metadata(step0)
            step20_meta = _package_metadata(step20)
            final_meta = _package_metadata(final)
            package_metadata = {
                "step0": step0_meta,
                "step20": step20_meta,
                "native_final": final_meta,
                "step20_model_optimizer_normalizers_equal_native_final": all(
                    _nested_tensor_equal(step20[name], final[name])
                    for name in (
                        "model_state_dict",
                        "optimizer_state_dict",
                        "obs_norm_state_dict",
                        "privileged_obs_norm_state_dict",
                    )
                ),
            }
            expected_count = (
                int(source_payload["obs_norm_state_dict"]["count"]) + 4096 * 24
            )
            if step0_meta["optimizer_steps"] != [10020] or step20_meta[
                "optimizer_steps"
            ] != [10040]:
                errors.append("endpoint optimizer package accounting mismatch")
            if any(
                metadata[normalizer] != expected_count
                for metadata in (step0_meta, step20_meta, final_meta)
                for normalizer in ("actor_normalizer_count", "critic_normalizer_count")
            ):
                errors.append("endpoint normalizer count mismatch")
            if not package_metadata[
                "step20_model_optimizer_normalizers_equal_native_final"
            ]:
                errors.append("step-20 snapshot differs from native final package")
        except (KeyError, IndexError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"endpoint package validation failed: {error}")

    record = {
        "executed": True,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "combined_log_path": str(log_path),
        "removed_ambient_environment_toggles": removed_toggles,
        "training_log_root": str(args.training_log_root),
        "new_run_candidates": [str(path) for path in candidates],
        "run_dir": str(run_dir) if run_dir else None,
        "probe_dir": str(probe_dir),
        "probe_result": {
            "path": str(probe_result_path),
            "sha256": _sha256_file(probe_result_path)
            if probe_result_path.is_file()
            else None,
        },
        "final_checkpoint": (
            {
                "path": str(final_checkpoint),
                "sha256": _sha256_file(final_checkpoint),
            }
            if final_checkpoint is not None and final_checkpoint.is_file()
            else None
        ),
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "package_metadata": package_metadata,
        "contract_errors": errors,
    }
    _write_json(args.output_dir / "training.json", record)
    print(f"[FIRST-UPDATE-RUN] training rc={return_code} errors={errors}", flush=True)
    return record, final_checkpoint, probe_result


def _evaluation_command(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    motion: Path,
    label_set: str,
    output_dir: Path,
    episodes: int,
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
        str(episodes),
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
    arm: str,
    checkpoint: Path,
    motion: Path,
    label_set: str,
    episodes: int,
    render: bool,
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    arm_dir = args.output_dir / "evaluations" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = _evaluation_command(
        args,
        checkpoint=checkpoint,
        motion=motion,
        label_set=label_set,
        output_dir=arm_dir,
        episodes=episodes,
        render=render,
    )
    stdout_path = arm_dir / "launcher_stdout.log"
    stderr_path = arm_dir / "launcher_stderr.log"
    started_at = _utc_now()
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
    result_path = arm_dir / "source_eval" / "result.json"
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

    qpos: list[str] = []
    qvel: list[str] = []
    observations: list[str] = []
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
        motion_sha = _sha256_file(motion)
        for key in ("checkpoint", "normalizer_checkpoint"):
            if inputs.get(key, {}).get("sha256") != checkpoint_sha:
                errors.append(f"{key} hash mismatch")
        if inputs.get("motion", {}).get("sha256") != motion_sha:
            errors.append("motion hash mismatch")
        episode_rows = payload.get("episodes", [])
        if not isinstance(episode_rows, list) or len(episode_rows) != episodes:
            errors.append("episode count mismatch")
            episode_rows = []
        for episode in episode_rows:
            qpos.append(str(episode.get("initial_qpos_sha256", "")))
            qvel.append(str(episode.get("initial_qvel_sha256", "")))
            observations.append(
                str(episode.get("initial_policy_observation_sha256", ""))
            )
        if (
            not all(qpos + qvel + observations)
            or payload.get("all_numeric_finite") is not True
        ):
            errors.append("reset identity missing or numeric telemetry nonfinite")
        if (
            payload.get("evaluation_contract", {}).get("normalizer_override_applied")
            is not False
        ):
            errors.append("normalizer override detected")
    if errors:
        outcome = "invalid-execution"
    record = {
        "arm": arm,
        "checkpoint_step": int(re.search(r"step_(\d+)", checkpoint.name).group(1)),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256_file(checkpoint)},
        "motion": {"path": str(motion), "sha256": _sha256_file(motion)},
        "outcome_label_set": label_set,
        "episodes": episodes,
        "render": render,
        "command": command,
        "started_at": started_at,
        "finished_at": _utc_now(),
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
        f"[FIRST-UPDATE-RUN] arm={arm} outcome={outcome} "
        f"survival={classification.get('survival_steps')} rc={return_code}",
        flush=True,
    )
    return record, qpos, qvel, observations


def _finalize(
    args: argparse.Namespace,
    *,
    started_at: str,
    source_sha: str,
    e009_audit: dict[str, Any],
    training: dict[str, Any],
    probe_result: dict[str, Any] | None,
    arms: list[dict[str, Any]],
    reset_hashes: tuple[list[str], list[str], list[str]],
    selected_render: dict[str, Any] | None,
) -> int:
    qpos, qvel, observations = reset_hashes
    scientific_arms = [arm for arm in arms if not arm["arm"].startswith("selected_")]
    errors = list(training.get("contract_errors", []))
    errors.extend(error for arm in arms for error in arm["contract_errors"])
    expected_identity_count = sum(arm["episodes"] for arm in arms)
    identity = (
        len(qpos) == len(qvel) == len(observations) == expected_identity_count
        and len(set(qpos)) == len(set(qvel)) == len(set(observations)) == 1
    )
    if not identity:
        errors.append("initial reset identities differ or are incomplete")

    short_complete: dict[int, bool] = {}
    for arm in scientific_arms:
        if arm["outcome_label_set"] == "short-control":
            short_complete[int(arm["checkpoint_step"])] = (
                arm["outcome"] == "short-source-completes"
            )
    long_by_step = {
        int(arm["checkpoint_step"]): arm
        for arm in scientific_arms
        if arm["outcome_label_set"] == "exact-long"
    }
    if 0 not in long_by_step or 20 not in long_by_step:
        errors.append("endpoint long evaluations missing")
        classification = {"outcome": "invalid-execution"}
    else:
        classification = classify_checkpoint_sweep(
            short_complete=short_complete,
            step0_long_complete=long_by_step[0]["outcome"]
            == "source-completes-exact-long",
            step20_long_complete=long_by_step[20]["outcome"]
            == "source-completes-exact-long",
        )

    step0_short = next(
        (arm for arm in scientific_arms if arm["arm"] == "step_00_short"), None
    )
    step20_short = next(
        (arm for arm in scientific_arms if arm["arm"] == "step_20_short"), None
    )
    if step0_short is not None and step0_short["classification"].get(
        "survival_steps"
    ) != [125, 125, 125]:
        errors.append("no-update short control did not reproduce 125/125/125")
    if long_by_step.get(0, {}).get("classification", {}).get("survival_steps") != [
        126,
        126,
        126,
    ]:
        errors.append("no-update long control did not reproduce 126/126/126")
    if step20_short is not None and step20_short["classification"].get(
        "survival_steps"
    ) != [37, 37, 37]:
        errors.append("step-20 short endpoint did not reproduce E009's 37/37/37")
    if long_by_step.get(20, {}).get("classification", {}).get("survival_steps") != [
        37,
        37,
        37,
    ]:
        errors.append("step-20 long endpoint did not reproduce E009's 37/37/37")
    if errors:
        classification = {
            **classification,
            "outcome": "invalid-execution",
            "execution_errors": errors,
        }

    result_path = args.output_dir / "result.json"
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
            "e009_audit": {
                "path": str(args.e009_audit),
                "sha256": _sha256_file(args.e009_audit),
                "verdict": e009_audit.get("verdict"),
                "selected_outcome": e009_audit.get("selected_outcome"),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "files": {
                    name: _sha256_file(SCRIPT_DIR / name)
                    for name in (
                        "run.py",
                        "contract.py",
                        "train.py",
                        "runner.py",
                        "probe.py",
                    )
                },
            },
        },
        "causal_change": (
            "No optimizer or rollout semantics change. The corrected first-observation ordering from E009 is retained; "
            "the runner records the rollout, computes read-only frozen gradients, exposes the exact native permutation, "
            "and saves the native package after optimizer steps zero through twenty."
        ),
        "training_contract": {
            "task": args.task,
            "updates": 1,
            "num_envs": 4096,
            "rollout_steps_per_env": 24,
            "transitions": 98304,
            "train_seed": 42,
            "epochs": 5,
            "mini_batches": 4,
            "optimizer_steps": 20,
            "sampling": "adaptive",
            "measurement_only": True,
        },
        "evaluation_contract": {
            "short_checkpoint_steps": list(range(21)),
            "short_episodes": {
                str(step): 3 if step in {0, 20} else 1 for step in range(21)
            },
            "long_checkpoint_steps": [0, 20],
            "long_episodes": 3,
            "strict_phase_zero_native_package": True,
            "initial_reset_identity": identity,
            "selected_failure_rendered": selected_render is not None,
        },
        "training": training,
        "probe_result": probe_result,
        "arms": arms,
        "selected_failure_render": selected_render,
        "claim_boundary": (
            "This probe localizes competence loss within one diagnostic PPO optimization and quantifies its fixed-rollout "
            "gradient geometry. It does not test PPO continuation, AHAC, differentiable physics, or sim-to-real transfer, "
            "and none of its PPO checkpoints is a retained policy."
        ),
    }
    _write_json(result_path, result)
    print(
        f"[FIRST-UPDATE-RUN] outcome={result['outcome']} result={result_path}",
        flush=True,
    )
    return 1 if result["outcome"] == "invalid-execution" else 0


def main() -> int:
    args = _parse_args()
    for name in (
        "source_checkpoint",
        "short_motion_file",
        "long_motion_file",
        "e009_audit",
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
        (args.e009_audit, args.e009_audit_sha256, "E009 audit"),
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

    e009_audit = json.loads(args.e009_audit.read_text(encoding="utf-8"))
    if (
        e009_audit.get("verdict") != "pass"
        or e009_audit.get("selected_outcome") != "corrected-order-still-immediate-loss"
    ):
        raise RuntimeError("E009 audit is not the passed corrected-order control")
    source_payload = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    source_sha = _sha256_file(args.source_checkpoint)
    started_at = _utc_now()
    probe_dir = args.output_dir / "training" / "probe"
    environment, removed = _child_environment(
        training_log_root=args.training_log_root, probe_dir=probe_dir
    )

    training, _, probe_result = _run_training(
        args,
        environment=environment,
        removed_toggles=removed,
        source_sha_before=source_sha,
        source_payload=source_payload,
    )
    arms: list[dict[str, Any]] = []
    qpos: list[str] = []
    qvel: list[str] = []
    observations: list[str] = []
    selected_render = None

    if not training["contract_errors"] and probe_result is not None:
        checkpoints = {
            int(record["step"]): Path(record["path"]).resolve()
            for record in probe_result["checkpoints"]
        }
        plan = [(0, "short", 3), (0, "long", 3)]
        plan.extend((step, "short", 3 if step == 20 else 1) for step in range(1, 21))
        plan.append((20, "long", 3))
        for index, (step, regime, episodes) in enumerate(plan):
            arm, arm_qpos, arm_qvel, arm_observations = _evaluate(
                args,
                arm=f"step_{step:02d}_{regime}",
                checkpoint=checkpoints[step],
                motion=args.short_motion_file
                if regime == "short"
                else args.long_motion_file,
                label_set="short-control" if regime == "short" else "exact-long",
                episodes=episodes,
                render=False,
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

        if len(arms) == len(plan) and args.render_selected_failure:
            first_failure = next(
                (
                    arm["checkpoint_step"]
                    for arm in arms
                    if arm["outcome_label_set"] == "short-control"
                    and arm["outcome"] == "short-source-fails"
                ),
                None,
            )
            if first_failure is not None:
                selected_render, render_qpos, render_qvel, render_observations = (
                    _evaluate(
                        args,
                        arm=f"selected_first_failure_step_{first_failure:02d}",
                        checkpoint=checkpoints[first_failure],
                        motion=args.short_motion_file,
                        label_set="short-control",
                        episodes=1,
                        render=True,
                        environment=environment,
                    )
                )
                arms.append(selected_render)
                qpos.extend(render_qpos)
                qvel.extend(render_qvel)
                observations.extend(render_observations)

    return _finalize(
        args,
        started_at=started_at,
        source_sha=source_sha,
        e009_audit=e009_audit,
        training=training,
        probe_result=probe_result,
        arms=arms,
        reset_hashes=(qpos, qvel, observations),
        selected_render=selected_render,
    )


if __name__ == "__main__":
    raise SystemExit(main())
