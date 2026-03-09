"""Test script to demonstrate the spring-force curriculum in isolation.

Creates the Staircase env with 2 environments:
  - Env 0: full spring force (curriculum_factor=1.0)  -> robot should track the reference motion
  - Env 1: no spring force   (curriculum_factor=0.0)  -> robot should ragdoll and fall

Both environments receive ZERO actions (default-pose PD target), so the *only*
difference between them is whether the spring force is present.

Usage
-----
cd /move/u/karenvo/Projects/rmr_tracking

# Interactive (GUI)
python scripts/rsl_rl/test_curriculum.py \
    --task Staircase-G1-v0 --num_envs 2 \
    --motion_file artifacts/staircase_final_v3:v0/motion.npz

# Record a comparison video (headless)
python scripts/rsl_rl/test_curriculum.py \
    --task Staircase-G1-v0 --num_envs 2 \
    --motion_file artifacts/staircase_final_v3:v0/motion.npz \
    --video --video_length 300 --headless
"""

# ---------------------------------------------------------------------------
# 1. Parse CLI args & launch Omniverse  (must happen before any other imports)
# ---------------------------------------------------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spring-force curriculum isolation test.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=300, help="Video length in env steps.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs (use 2 for comparison).")
parser.add_argument("--task", type=str, default="Staircase-G1-v0", help="Gym task id.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the motion .npz file.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv so Hydra only sees its own overrides (same pattern as play.py)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2. Imports that need the simulator running
# ---------------------------------------------------------------------------
import gymnasium as gym
import os
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

# Register staircase task
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # ------------------------------------------------------------------
    # 3. Build env config with modifications for the test
    # ------------------------------------------------------------------
    # Force exactly the requested number of envs
    env_cfg.scene.num_envs = args_cli.num_envs

    # Point to the motion file
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Disable terminations so we get a long, uninterrupted run
    env_cfg.terminations.anchor_pos_z = None
    env_cfg.terminations.anchor_pos_xy = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None

    # Make episode very long so the robot doesn't get reset
    env_cfg.episode_length_s = 120.0

    # Disable the curriculum scheduler so the force stays constant
    # (we manually control curriculum_factor via env override below)
    env_cfg.curriculum.adr = None
    env_cfg.curriculum.spring_force_adr = None

    # Extract spring force params from the event config before disabling it
    spring_params = dict(env_cfg.events.assistive_spring_force.params)
    # Override ang_stiffness for testing (default 20.0 is too low to keep pelvis upright)
    spring_params["ang_stiffness"] = 150.0
    print(f"[INFO] Spring force params (ang_stiffness bumped): {spring_params}")

    # Disable the built-in spring force event so we can manually control per-env
    env_cfg.events.assistive_spring_force = None

    # ------------------------------------------------------------------
    # 4. Create environment
    # ------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "curriculum_test", "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video for spring-force curriculum isolation test.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ------------------------------------------------------------------
    # 5. Reset + prepare for per-env force control
    # ------------------------------------------------------------------
    obs, _ = env.reset()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    print("\n" + "=" * 60)
    print("SPRING-FORCE CURRICULUM ISOLATION TEST")
    print("  Env 0 (left) : FULL spring force  -> should TRACK reference")
    print("  Env 1 (right): ZERO spring force  -> should FALL")
    print("  Actions      : ZERO (default-pose PD target)")
    print("  Spring params:", spring_params)
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 6. Run with zero actions
    #    We manually apply spring force before each step, then zero env 1.
    # ------------------------------------------------------------------
    action_dim = base_env.action_manager.total_action_dim
    zero_actions = torch.zeros(args_cli.num_envs, action_dim, device=base_env.device)

    # Import spring force function
    from whole_body_tracking.tasks.staircase.mdp.curriculum import apply_spring_force

    CUTOFF_STEP = 10000

    timestep = 0
    force_active = True
    while simulation_app.is_running():
        with torch.inference_mode():
            # After cutoff, stop applying spring force to env 0 too
            if force_active and timestep >= CUTOFF_STEP:
                force_active = False
                # Clear any persistent forces by applying zero
                zero_force = torch.zeros(1, 1, 3, device=base_env.device)
                robot.set_external_force_and_torque(
                    zero_force, zero_force,
                    body_ids=[base_env.command_manager.get_term("motion").robot_anchor_body_index],
                    env_ids=torch.tensor([0], device=base_env.device),
                    is_global=True,
                )
                print(f"\n{'='*60}")
                print(f"[Step {timestep}] CUTOFF: Spring force on env 0 set to ZERO")
                print(f"{'='*60}\n")

            if force_active:
                # Apply spring force only to env 0
                total_force = apply_spring_force(
                    env=base_env,
                    command_name=spring_params["command_name"],
                    asset_name=spring_params["asset_name"],
                    stiffness=spring_params["stiffness"],
                    ang_stiffness=spring_params.get("ang_stiffness", 100.0),
                    damping=spring_params["damping"],
                    axis_weights=tuple(spring_params["axis_weights"]),
                    gravity_comp=1.0,
                    curriculum_factor=1.0,
                    env_ids=torch.tensor([0], device=base_env.device),
                )

            # Step the env
            obs, _, _, _, _ = env.step(zero_actions)


            # Print debug info periodically
            if timestep % 50 == 0:
                if force_active:
                    print(f"[Step {timestep:4d}]  "
                          f"Env 0 spring force: [{total_force[0, 0]:.1f}, {total_force[0, 1]:.1f}, {total_force[0, 2]:.1f}] N  "
                          f"(|F|={torch.norm(total_force[0]):.1f} N)")
                else:
                    print(f"[Step {timestep:4d}]  Env 0 spring force: OFF (cutoff)")

                # Print position tracking
                cmd = base_env.command_manager.get_term("motion")
                ref_pos = cmd.anchor_pos_w
                cur_pos = robot.data.body_pos_w[:, cmd.robot_anchor_body_index, :]
                pos_err = torch.norm(ref_pos - cur_pos, dim=-1)
                print(f"           Pos error: Env0={pos_err[0]:.3f}m  Env1={pos_err[1]:.3f}m")



        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

    env.close()
    print(f"\n[INFO] Test complete after {timestep} steps.")
    if args_cli.video:
        print(f"[INFO] Video saved to: logs/curriculum_test/videos/")



if __name__ == "__main__":
    main()
    simulation_app.close()
