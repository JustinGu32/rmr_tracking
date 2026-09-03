"""Evaluate one exact-long PPO checkpoint in its native Isaac/PhysX task.

This evaluator intentionally lives in an experiment namespace.  It uses the
training task (not the permissive Play task), starts every episode at exact
reference phase zero, disables all stochastic reset/event/observation terms,
and preserves every training-task termination gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Deterministic phase-zero source-simulator competence gate."
)
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--motion-file", required=True)
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--eval-seed", type=int, default=0)
parser.add_argument(
    "--ppo-output",
    default="delta-all",
    choices=("target", "delta-pseudotarget", "delta-all"),
)
parser.add_argument("--render", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task != "Tracking-Flat-G1-v0":
    parser.error(
        "this evaluator requires the original Tracking-Flat-G1-v0 training task"
    )
if args_cli.episodes < 1:
    parser.error("--episodes must be positive")

motion_path_cli = Path(args_cli.motion_file).expanduser().resolve()
checkpoint_path_cli = Path(args_cli.checkpoint_path).expanduser().resolve()
output_dir_cli = Path(args_cli.output_dir).expanduser().resolve()
if not motion_path_cli.is_file():
    parser.error(f"--motion-file does not exist: {motion_path_cli}")
if not checkpoint_path_cli.is_file():
    parser.error(f"--checkpoint-path does not exist: {checkpoint_path_cli}")
if (output_dir_cli / "result.json").exists():
    parser.error(
        f"refusing to overwrite completed result: {output_dir_cli / 'result.json'}"
    )

# Configuration modules read these variables at import time.  Remove ambient
# toggles that could silently change observation/reward shape, then set the one
# action contract that belongs to the checkpoint under test.
for env_name in (
    "ENABLE_CAMERAS",
    "WBT_CURRICULUM",
    "WBT_DOUBLE_STEP",
    "WBT_MOTION_JOINT_POS",
    "WBT_STAIR_PHASE_TERM",
    "WBT_USE_DEPTH_OBS",
):
    os.environ.pop(env_name, None)
os.environ["WBT_PPO_OUTPUT"] = args_cli.ppo_output
if args_cli.render:
    args_cli.enable_cameras = True

# Leave only Hydra overrides for the decorator.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import whole_body_tracking.tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from PIL import Image, ImageDraw
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from contract import (
    apply_nominal_phase_zero_contract,
    classify_episodes,
    reset_episode_in_inference_mode,
)

EXPECTED_TERMINATIONS = [
    "time_out",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
    "bad_anchor_pos_xy",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _all_finite(*values: Any) -> bool:
    for value in values:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all().item()):
                return False
        elif isinstance(value, np.ndarray):
            if not bool(np.isfinite(value).all()):
                return False
        elif isinstance(value, dict):
            if not _all_finite(*value.values()):
                return False
        elif isinstance(value, (list, tuple)):
            if not _all_finite(*value):
                return False
        elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return False
    return True


def _refresh_relative_reference(motion_command: Any) -> None:
    """Refresh derived reference state after reset without advancing its phase."""
    body_count = len(motion_command.cfg.body_names)
    anchor_pos = motion_command.anchor_pos_w[:, None, :].expand(-1, body_count, -1)
    anchor_quat = motion_command.anchor_quat_w[:, None, :].expand(-1, body_count, -1)
    robot_anchor_quat = motion_command.robot_anchor_quat_w[:, None, :].expand(
        -1, body_count, -1
    )
    delta_pos = (
        motion_command.robot_anchor_pos_w[:, None, :].expand(-1, body_count, -1).clone()
    )
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
    motion_command.body_quat_relative_w = quat_mul(
        delta_ori, motion_command.body_quat_w
    )
    motion_command.body_pos_relative_w = delta_pos + quat_apply(
        delta_ori, motion_command.body_pos_w - anchor_pos
    )


def _state_snapshot(raw_env: Any, motion_command: Any) -> dict[str, np.ndarray]:
    robot = motion_command.robot
    root_pos = _tensor_numpy(robot.data.root_pos_w[0])
    root_quat = _tensor_numpy(robot.data.root_quat_w[0])
    root_lin_vel = _tensor_numpy(robot.data.root_lin_vel_w[0])
    root_ang_vel = _tensor_numpy(robot.data.root_ang_vel_w[0])
    joint_pos = _tensor_numpy(robot.data.joint_pos[0])
    joint_vel = _tensor_numpy(robot.data.joint_vel[0])
    return {
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_w": root_lin_vel,
        "root_ang_vel_w": root_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "qpos": np.concatenate((root_pos, root_quat, joint_pos)),
        "qvel": np.concatenate((root_lin_vel, root_ang_vel, joint_vel)),
        "env_origin": _tensor_numpy(raw_env.scene.env_origins[0]),
    }


def _metric_snapshot(motion_command: Any) -> dict[str, float]:
    return {
        name: float(value[0].detach().cpu().item())
        for name, value in motion_command.metrics.items()
    }


def _reward_snapshot(raw_env: Any) -> dict[str, float]:
    step_reward = raw_env.reward_manager._step_reward[0]
    return {
        name: float(step_reward[index].detach().cpu().item())
        for index, name in enumerate(raw_env.reward_manager.active_terms)
    }


def _processed_target(
    action_term: Any, motion_command: Any, action: torch.Tensor
) -> torch.Tensor:
    reference = motion_command.joint_pos
    if not isinstance(action_term._joint_ids, slice):
        reference = reference[:, action_term._joint_ids]
    target = reference + action.to(reference.device) * action_term._scale
    if action_term.cfg.clip is not None:
        target = torch.clamp(
            target, min=action_term._clip[:, :, 0], max=action_term._clip[:, :, 1]
        )
    return target


def _save_contact_sheet(
    frames: list[np.ndarray], phases: list[int], path: Path
) -> None:
    if not frames:
        raise RuntimeError("rendering was requested but produced no frames")
    sample_count = min(12, len(frames))
    sample_ids = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
    tile_width = 320
    tile_height = 180
    label_height = 24
    columns = 4
    rows = (sample_count + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for slot, frame_index in enumerate(sample_ids.tolist()):
        frame = frames[frame_index][..., :3]
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        image.thumbnail((tile_width, tile_height))
        x = (slot % columns) * tile_width + (tile_width - image.width) // 2
        y0 = (slot // columns) * (tile_height + label_height)
        y = y0 + (tile_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (x + 4, y0 + tile_height + 4), f"phase {phases[frame_index]}", fill="black"
        )
    sheet.save(path)


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    output_dir_cli.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()

    random.seed(args_cli.eval_seed)
    np.random.seed(args_cli.eval_seed)
    torch.manual_seed(args_cli.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.eval_seed)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.eval_seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.eval_seed

    config_audit = apply_nominal_phase_zero_contract(
        env_cfg,
        motion_file=str(motion_path_cli),
        num_envs=1,
    )

    with np.load(motion_path_cli) as motion_npz:
        reference_shapes = {
            name: list(value.shape) for name, value in motion_npz.items()
        }
        reference_states = int(motion_npz["joint_pos"].shape[0])
        reference_fps = float(np.asarray(motion_npz["fps"]).reshape(-1)[0])

    gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.render else None,
    )
    env = RslRlVecEnvWrapper(gym_env)
    raw_env = env.unwrapped
    motion_command = raw_env.command_manager.get_term("motion")
    termination_manager = raw_env.termination_manager
    action_term = raw_env.action_manager.get_term("joint_pos")

    active_terminations = list(termination_manager.active_terms)
    event_modes = list(raw_env.event_manager.available_modes)
    curriculum_terms = list(raw_env.curriculum_manager.active_terms)
    if active_terminations != EXPECTED_TERMINATIONS:
        raise RuntimeError(
            f"termination contract changed: expected {EXPECTED_TERMINATIONS}, got {active_terminations}"
        )
    if event_modes:
        raise RuntimeError(
            f"nominal evaluator still has active event modes: {event_modes}"
        )
    if curriculum_terms:
        raise RuntimeError(
            f"nominal evaluator still has curriculum terms: {curriculum_terms}"
        )
    if type(action_term).__name__ != "ReferenceJointPositionAction":
        raise RuntimeError(f"wrong action contract: {type(action_term).__name__}")
    if int(motion_command.motion.time_step_total) != reference_states:
        raise RuntimeError("loaded source motion length differs from inspected NPZ")

    runner = MotionOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(str(checkpoint_path_cli))
    policy = runner.get_inference_policy(device=raw_env.device)

    # Capture the real post-physics terminal state inside the reset seam.  Isaac
    # resets done environments before env.step returns, so reading state only
    # after step would silently record the next episode's phase-one state.
    capture_context: dict[str, Any] = {"active": False, "snapshot": None}
    original_reset_idx = raw_env._reset_idx

    def _capture_then_reset(self: Any, env_ids: torch.Tensor) -> None:
        if capture_context["active"] and bool((env_ids == 0).any().item()):
            motion_command._update_metrics()
            capture_context["snapshot"] = {
                "state": _state_snapshot(self, motion_command),
                "metrics": _metric_snapshot(motion_command),
                "reward_terms": _reward_snapshot(self),
                "reference_phase": int(motion_command.time_steps[0].item()),
                "termination_terms": [
                    name
                    for name in active_terminations
                    if bool(termination_manager.get_term(name)[0].item())
                ],
            }
        original_reset_idx(env_ids)

    raw_env._reset_idx = types.MethodType(_capture_then_reset, raw_env)

    frames: list[np.ndarray] = []
    frame_phases: list[int] = []
    episode_records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    initial_qpos_hashes: list[str] = []
    initial_qvel_hashes: list[str] = []
    initial_obs_hashes: list[str] = []
    max_steps = reference_states + 1

    for episode_index in range(args_cli.episodes):
        obs = reset_episode_in_inference_mode(
            env,
            motion_command,
            seed=args_cli.eval_seed,
            refresh_reference=_refresh_relative_reference,
        )
        if int(motion_command.time_steps[0].item()) != 0:
            raise RuntimeError(
                "manual episode reset did not produce reference phase zero"
            )

        initial_state = _state_snapshot(raw_env, motion_command)
        initial_qpos_hashes.append(_sha256_array(initial_state["qpos"]))
        initial_qvel_hashes.append(_sha256_array(initial_state["qvel"]))
        initial_obs_hashes.append(_sha256_array(_tensor_numpy(obs[0])))

        if args_cli.render and episode_index == 0:
            with torch.inference_mode():
                for _ in range(6):
                    raw_env.render()

        episode_rows: list[dict[str, Any]] = []
        return_sum = 0.0
        all_numeric_finite = True
        terminal_snapshot: dict[str, Any] | None = None
        terminal_phase = -1
        terminated = False
        timed_out = False
        fired_terms: list[str] = []

        for step_index in range(max_steps):
            phase = int(motion_command.time_steps[0].item())
            if phase != step_index:
                break

            if args_cli.render and episode_index == 0:
                frame = raw_env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8)[..., :3].copy())
                    frame_phases.append(phase)

            before = _state_snapshot(raw_env, motion_command)
            reference_joint_pos = _tensor_numpy(motion_command.joint_pos[0])
            reference_joint_vel = _tensor_numpy(motion_command.joint_vel[0])
            with torch.inference_mode():
                actor_action = policy(obs)
                processed_target = _processed_target(
                    action_term, motion_command, actor_action
                )

            capture_context["active"] = True
            capture_context["snapshot"] = None
            with torch.inference_mode():
                next_obs, reward, done, _ = env.step(actor_action)
            capture_context["active"] = False

            terminated = bool(raw_env.reset_terminated[0].item())
            timed_out = bool(raw_env.reset_time_outs[0].item())
            done_flag = bool(done[0].item())
            if done_flag != (terminated or timed_out):
                raise RuntimeError(
                    "RSL wrapper done flag disagrees with raw Isaac termination flags"
                )

            if done_flag:
                terminal_snapshot = capture_context["snapshot"]
                if terminal_snapshot is None:
                    raise RuntimeError(
                        "done transition bypassed terminal-state capture"
                    )
                after = terminal_snapshot["state"]
                metrics = terminal_snapshot["metrics"]
                reward_terms = terminal_snapshot["reward_terms"]
                fired_terms = list(terminal_snapshot["termination_terms"])
                terminal_phase = int(terminal_snapshot["reference_phase"])
            else:
                after = _state_snapshot(raw_env, motion_command)
                metrics = _metric_snapshot(motion_command)
                reward_terms = _reward_snapshot(raw_env)

            reward_value = float(reward[0].detach().cpu().item())
            row_finite = _all_finite(
                obs,
                actor_action,
                processed_target,
                before,
                after,
                reference_joint_pos,
                reference_joint_vel,
                reward_value,
                reward_terms,
                metrics,
            )
            all_numeric_finite = all_numeric_finite and row_finite
            return_sum += reward_value
            row = {
                "episode_index": episode_index,
                "step_index": step_index,
                "reference_phase": phase,
                "obs": _tensor_numpy(obs[0]),
                "actor_action": _tensor_numpy(actor_action[0]),
                "processed_joint_target": _tensor_numpy(processed_target[0]),
                "reference_joint_pos": reference_joint_pos,
                "reference_joint_vel": reference_joint_vel,
                "qpos_before": before["qpos"],
                "qvel_before": before["qvel"],
                "qpos_after": after["qpos"],
                "qvel_after": after["qvel"],
                "reward": reward_value,
                "reward_terms": reward_terms,
                "metrics": metrics,
                "terminated": terminated,
                "timed_out": timed_out,
                "all_numeric_finite": row_finite,
            }
            episode_rows.append(row)
            trajectory_rows.append(row)
            obs = next_obs
            if done_flag:
                break

        if episode_rows and terminal_phase < 0:
            terminal_phase = int(episode_rows[-1]["reference_phase"])
        metric_names = sorted(motion_command.metrics)
        episode_metric_mean = {
            name: float(np.mean([row["metrics"][name] for row in episode_rows]))
            for name in metric_names
        }
        episode_metric_max = {
            name: float(np.max([row["metrics"][name] for row in episode_rows]))
            for name in metric_names
        }
        episode_records.append(
            {
                "episode_index": episode_index,
                "steps": len(episode_rows),
                "initial_reference_phase": 0,
                "final_reference_phase": terminal_phase,
                "terminated": terminated,
                "timed_out": timed_out,
                "termination_terms": fired_terms,
                "return": return_sum,
                "all_numeric_finite": all_numeric_finite and bool(episode_rows),
                "metric_mean": episode_metric_mean,
                "metric_max": episode_metric_max,
                "initial_qpos_sha256": initial_qpos_hashes[-1],
                "initial_qvel_sha256": initial_qvel_hashes[-1],
                "initial_policy_observation_sha256": initial_obs_hashes[-1],
            }
        )
        print(
            f"[SOURCE-EVAL] episode={episode_index} steps={len(episode_rows)} "
            f"phase={terminal_phase} terminated={terminated} timed_out={timed_out} "
            f"terms={fired_terms}"
        )

    classification = classify_episodes(
        episode_records, reference_states=reference_states
    )

    episodes_path = output_dir_cli / "episodes.json"
    trajectory_path = output_dir_cli / "trajectory.npz"
    _write_json(episodes_path, {"episodes": episode_records})

    reward_names = list(raw_env.reward_manager.active_terms)
    metric_names = sorted(motion_command.metrics)
    np.savez_compressed(
        trajectory_path,
        episode_index=np.asarray(
            [row["episode_index"] for row in trajectory_rows], dtype=np.int64
        ),
        step_index=np.asarray(
            [row["step_index"] for row in trajectory_rows], dtype=np.int64
        ),
        reference_phase=np.asarray(
            [row["reference_phase"] for row in trajectory_rows], dtype=np.int64
        ),
        observation=np.stack([row["obs"] for row in trajectory_rows]),
        actor_action=np.stack([row["actor_action"] for row in trajectory_rows]),
        processed_joint_target=np.stack(
            [row["processed_joint_target"] for row in trajectory_rows]
        ),
        reference_joint_pos=np.stack(
            [row["reference_joint_pos"] for row in trajectory_rows]
        ),
        reference_joint_vel=np.stack(
            [row["reference_joint_vel"] for row in trajectory_rows]
        ),
        qpos_before=np.stack([row["qpos_before"] for row in trajectory_rows]),
        qvel_before=np.stack([row["qvel_before"] for row in trajectory_rows]),
        qpos_after=np.stack([row["qpos_after"] for row in trajectory_rows]),
        qvel_after=np.stack([row["qvel_after"] for row in trajectory_rows]),
        reward=np.asarray([row["reward"] for row in trajectory_rows], dtype=np.float64),
        reward_terms=np.asarray(
            [
                [row["reward_terms"][name] for name in reward_names]
                for row in trajectory_rows
            ],
            dtype=np.float64,
        ),
        reward_term_names=np.asarray(reward_names),
        metrics=np.asarray(
            [
                [row["metrics"][name] for name in metric_names]
                for row in trajectory_rows
            ],
            dtype=np.float64,
        ),
        metric_names=np.asarray(metric_names),
        terminated=np.asarray(
            [row["terminated"] for row in trajectory_rows], dtype=np.bool_
        ),
        timed_out=np.asarray(
            [row["timed_out"] for row in trajectory_rows], dtype=np.bool_
        ),
        all_numeric_finite=np.asarray(
            [row["all_numeric_finite"] for row in trajectory_rows], dtype=np.bool_
        ),
    )

    artifact_paths = [episodes_path, trajectory_path]
    if args_cli.render:
        video_path = output_dir_cli / "source_rollout_episode0.mp4"
        contact_sheet_path = (
            output_dir_cli / "source_rollout_episode0_contact_sheet.png"
        )
        imageio.mimsave(
            video_path,
            frames,
            fps=round(1.0 / raw_env.step_dt),
            macro_block_size=1,
        )
        _save_contact_sheet(frames, frame_phases, contact_sheet_path)
        artifact_paths.extend((video_path, contact_sheet_path))

    termination_cfg = {
        name: {
            "time_out": bool(termination_manager.get_term_cfg(name).time_out),
            "params": termination_manager.get_term_cfg(name).params,
            "function": (
                f"{termination_manager.get_term_cfg(name).func.__module__}."
                f"{termination_manager.get_term_cfg(name).func.__name__}"
            ),
        }
        for name in active_terminations
    }
    global_metric_mean = {
        name: float(np.mean([row["metrics"][name] for row in trajectory_rows]))
        for name in metric_names
    }
    global_metric_max = {
        name: float(np.max([row["metrics"][name] for row in trajectory_rows]))
        for name in metric_names
    }
    per_term_counts = {
        name: sum(name in episode["termination_terms"] for episode in episode_records)
        for name in active_terminations
    }

    result = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "outcome": classification["outcome"],
        "classification": classification,
        "task": args_cli.task,
        "evaluation_contract": {
            **config_audit,
            "episodes_requested": args_cli.episodes,
            "eval_seed_reapplied_before_each_episode": args_cli.eval_seed,
            "ppo_output": args_cli.ppo_output,
            "source_backend": "Isaac Lab / PhysX GPU",
            "physics_dt_seconds": float(raw_env.physics_dt),
            "control_dt_seconds": float(raw_env.step_dt),
            "rendered_episode_index": 0 if args_cli.render else None,
            "active_terminations": active_terminations,
            "termination_configuration": termination_cfg,
            "event_modes": event_modes,
            "curriculum_terms": curriculum_terms,
            "action_term_class": type(action_term).__name__,
            "wrapper_action_clip": env.clip_actions,
            "processed_action_clip": action_term.cfg.clip,
            "initial_qpos_identical": len(set(initial_qpos_hashes)) == 1,
            "initial_qvel_identical": len(set(initial_qvel_hashes)) == 1,
            "initial_policy_observation_identical": len(set(initial_obs_hashes)) == 1,
        },
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint_path_cli),
                "sha256": _sha256_file(checkpoint_path_cli),
            },
            "motion": {
                "path": str(motion_path_cli),
                "sha256": _sha256_file(motion_path_cli),
                "reference_states": reference_states,
                "fps": reference_fps,
                "array_shapes": reference_shapes,
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "evaluate_py_sha256": _sha256_file(Path(__file__).resolve()),
                "contract_py_sha256": _sha256_file(SCRIPT_DIR / "contract.py"),
            },
        },
        "episodes": episode_records,
        "per_termination_counts": per_term_counts,
        "metrics": {
            "mean_over_policy_steps": global_metric_mean,
            "max_over_policy_steps": global_metric_max,
        },
        "all_numeric_finite": all(
            episode["all_numeric_finite"] for episode in episode_records
        ),
        "artifacts": {path.name: _artifact_record(path) for path in artifact_paths},
        "claim_boundary": (
            "Measures phase-zero competence of the selected PPO checkpoint in the original "
            "training task under nominal source-simulator conditions. It does not measure "
            "robustness to training randomization or transfer to MJX/real hardware."
        ),
        "capture_note": (
            "Trajectory qpos_after/qvel_after and terminal metrics are captured inside "
            "Isaac's pre-reset seam. Video frames show each pre-action state, because Isaac "
            "automatically resets a terminal state before env.step returns."
        ),
    }

    # The result is written last so its presence is the completion signal.
    result_tmp = output_dir_cli / "result.json.tmp"
    result_path = output_dir_cli / "result.json"
    _write_json(result_tmp, result)
    os.replace(result_tmp, result_path)
    print(f"[SOURCE-EVAL] outcome={classification['outcome']}")
    print(f"[SOURCE-EVAL] result={result_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
