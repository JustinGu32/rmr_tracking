"""Play a checkpoint and record an mp4 to disk (manual render -> imageio, no RecordVideo).

Renders env 0 (follow-cam viewport) for one full motion clip from frame 0 and writes an mp4.
Used to compare policies at the SAME checkpoint number.

Usage:
  python scripts/rsl_rl/record_play.py --task=Staircase-G1-ObsAug-v0 \
    --motion_file /home/ubuntu/Downloads/walk_up_33.npz_v0/motion.npz \
    --load_run <run_dir> --checkpoint model_5000.pt --out /tmp/videos/B_5000.mp4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--out", type=str, required=True, help="Output mp4 path.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--clips", type=int, default=1, help="How many full motion clips to record.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # need app-level rendering for the viewport render product
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cmd = u.command_manager.get_term("motion")
    T = int(cmd.motion.time_step_total)
    fps = int(round(1.0 / u.step_dt))

    # Force the sampler to ALWAYS pick frame 0 so a real env.reset() writes the robot's
    # physical pose to the start-of-clip (bottom of the staircase). With these window values
    # _uniform_sampling computes sampling_range=(1-1)-0=0 -> frame 0, clipped to [0,0].
    cmd.min_sample_idx = 0
    cmd.max_sample_idx = 1
    cmd.steps_collect = 1
    obs, _ = env.reset()  # resamples (frame 0) AND writes robot state to that frame
    # warm the renderer (first frames are black until the RTX buffer fills)
    with torch.inference_mode():
        for _ in range(6):
            u.render()

    frames = []
    with torch.inference_mode():
        for _ in range(args_cli.clips * (T + 2)):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            fr = u.render()
            if fr is not None:
                frames.append(np.asarray(fr, dtype=np.uint8))
            if int(cmd.time_steps[0]) >= T - 1:
                cmd.time_steps[0] = 0  # loop the clip if recording multiple

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    import imageio
    imageio.mimsave(args_cli.out, frames, fps=fps, macro_block_size=1)
    print(f"[INFO] wrote {len(frames)} frames @ {fps}fps -> {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
