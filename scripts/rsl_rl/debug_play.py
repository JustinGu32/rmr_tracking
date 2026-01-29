"""Debug script: create env + print contact sensor data (no checkpoint needed)."""

import argparse
import sys
from isaaclab.app import AppLauncher

# local imports (safe: doesn't import pxr)
import cli_args  # isort: skip


# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motion_file", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra (same trick as play.py)
sys.argv = [sys.argv[0]] + hydra_args

# ---- Launch Isaac Sim FIRST ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Only now import anything that may touch pxr / Omniverse ----
import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to register your tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    # set env count
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
    else:
        # fallback to something sane so it doesn't stay MISSING
        env_cfg.commands.motion.motion_file = "/path/to/your/motion.npz"


    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    base_env = env.unwrapped
    print("SENSORS:", list(base_env.scene.sensors.keys()))

    # sanity: check your object exists
    print("ASSETS:", list(base_env.scene.keys()))

    # step a few frames with zero actions
    act_dim = base_env.action_manager.action_dim
    for i in range(5):
        actions = torch.zeros((base_env.num_envs, act_dim), device=base_env.device)
        obs, _, _, _ = env.step(actions)

        cs = base_env.scene.sensors["contact_forces"]   # <-- your robot contact sensor
        print(f"\n--- step {i} ---")
        print(cs.data)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
