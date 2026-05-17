"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""
python scripts/rsl_rl/play.py --task=Chair-Step-G1-v0 --num_envs=2 --wandb_path=robot-mcrobotface/chair_step/q4yp8tny
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr store for multi-clip tasks.")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation (physics steps per policy step).")
parser.add_argument("--max_clips", type=int, default=None, help="Max clips to load from Zarr (for smaller GPUs).")
parser.add_argument("--heightmap", action="store_true", default=False,
                    help="Enable task-configured height-map observations.")
parser.add_argument("--heightmap_debug_vis", action="store_true", default=False,
                    help="Show height-map raycaster debug visualization.")
parser.add_argument("--depth_obs", action="store_true", default=False, help="Enable optional RGB-D camera depth observations.")
parser.add_argument("--depth_debug_save_frames", action="store_true", default=False, help="Save a few normalized depth frames during rollout.")
parser.add_argument("--depth_debug_max_frames", type=int, default=4, help="Maximum number of depth debug frames to save.")
parser.add_argument("--ppo_output", type=str, default="target", choices=["target", "delta-pseudotarget", "delta-all"],
                    help="PPO output mode: must match the checkpoint's training setting.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.depth_obs:
    args_cli.enable_cameras = True

# Mirror env vars that __post_init__ reads and that affect obs/action shape.
# These must be set before whole_body_tracking.tasks is imported.
os.environ["WBT_PPO_OUTPUT"] = args_cli.ppo_output
if args_cli.depth_obs:
    os.environ["WBT_USE_DEPTH_OBS"] = "1"
if args_cli.depth_debug_save_frames:
    os.environ["WBT_DEPTH_SAVE_FRAMES"] = "1"
os.environ["WBT_DEPTH_DEBUG_MAX_FRAMES"] = str(args_cli.depth_debug_max_frames)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

def configure_height_map_obs(env_cfg, enabled: bool):
    """Enable or remove task-provided height-map sensor and observation terms."""
    height_scanner_cfg = getattr(env_cfg.scene, "height_scanner", None)
    has_height_scan_term = False
    for group_name in ("policy", "critic", "diffusion_collect"):
        group_cfg = getattr(env_cfg.observations, group_name, None)
        if group_cfg is not None and getattr(group_cfg, "height_scan", None) is not None:
            has_height_scan_term = True

    if enabled:
        if height_scanner_cfg is None or not has_height_scan_term:
            raise ValueError(
                "--heightmap was passed, but this task does not define both scene.height_scanner "
                "and observations.*.height_scan."
            )
        return

    if height_scanner_cfg is not None:
        env_cfg.scene.height_scanner = None
    for group_name in ("policy", "critic", "diffusion_collect"):
        group_cfg = getattr(env_cfg.observations, group_name, None)
        if group_cfg is not None and getattr(group_cfg, "height_scan", None) is not None:
            group_cfg.height_scan = None


def print_height_map_obs_debug(env, env_cfg):
    """Print enough state to verify height-map observations reached the built env."""
    height_scanner_cfg = getattr(env_cfg.scene, "height_scanner", None)
    print(f"[HEIGHT_MAP_DEBUG] scene.height_scanner configured: {height_scanner_cfg is not None}")
    if height_scanner_cfg is not None:
        print(f"[HEIGHT_MAP_DEBUG] height_scanner prim_path: {height_scanner_cfg.prim_path}")
        print(f"[HEIGHT_MAP_DEBUG] height_scanner pattern: {height_scanner_cfg.pattern_cfg}")
        print(f"[HEIGHT_MAP_DEBUG] height_scanner mesh_prim_paths: {height_scanner_cfg.mesh_prim_paths}")
        print(f"[HEIGHT_MAP_DEBUG] height_scanner update_period: {height_scanner_cfg.update_period}")
    for group_name in ("policy", "critic", "diffusion_collect"):
        group_cfg = getattr(env_cfg.observations, group_name, None)
        has_term = group_cfg is not None and getattr(group_cfg, "height_scan", None) is not None
        print(f"[HEIGHT_MAP_DEBUG] cfg observations.{group_name}.height_scan: {has_term}")

    unwrapped = env.unwrapped
    sensor_names = sorted(getattr(unwrapped.scene, "sensors", {}).keys())
    print(f"[HEIGHT_MAP_DEBUG] built scene sensors: {sensor_names}")
    obs_manager = getattr(unwrapped, "observation_manager", None)
    if obs_manager is not None:
        print(f"[HEIGHT_MAP_DEBUG] active observation terms: {obs_manager.active_terms}")
        print(f"[HEIGHT_MAP_DEBUG] observation term dims: {obs_manager.group_obs_term_dim}")
    obs_space = getattr(unwrapped, "observation_space", None)
    if hasattr(obs_space, "spaces"):
        for group_name, space in obs_space.spaces.items():
            print(f"[HEIGHT_MAP_DEBUG] observation_space[{group_name}]: {space}")
    else:
        print(f"[HEIGHT_MAP_DEBUG] observation_space: {obs_space}")


def print_depth_obs_debug(env, env_cfg):
    """Print enough state to verify depth observations reached the built env."""
    depth_cfg = getattr(env_cfg.scene, "depth_camera", None)
    depth_term = getattr(getattr(env_cfg.observations, "policy", None), "depth_image", None)
    print(f"[DEPTH_OBS_DEBUG] scene.depth_camera configured: {depth_cfg is not None}")
    if depth_cfg is not None:
        print(f"[DEPTH_OBS_DEBUG] depth_camera prim_path: {depth_cfg.prim_path}")
        print(f"[DEPTH_OBS_DEBUG] depth_camera resolution: ({depth_cfg.height}, {depth_cfg.width})")
        print(f"[DEPTH_OBS_DEBUG] depth_camera data_types: {depth_cfg.data_types}")
    print(f"[DEPTH_OBS_DEBUG] cfg observations.policy.depth_image: {depth_term is not None}")

    unwrapped = env.unwrapped
    sensor_names = sorted(getattr(unwrapped.scene, "sensors", {}).keys())
    print(f"[DEPTH_OBS_DEBUG] built scene sensors: {sensor_names}")
    obs_manager = getattr(unwrapped, "observation_manager", None)
    if obs_manager is not None:
        print(f"[DEPTH_OBS_DEBUG] active observation terms: {obs_manager.active_terms}")
        print(f"[DEPTH_OBS_DEBUG] observation term dims: {obs_manager.group_obs_term_dim}")
    obs_space = getattr(unwrapped, "observation_space", None)
    if hasattr(obs_space, "spaces"):
        for group_name, space in obs_space.spaces.items():
            print(f"[DEPTH_OBS_DEBUG] observation_space[{group_name}]: {space}")
    else:
        print(f"[DEPTH_OBS_DEBUG] observation_space: {obs_space}")


def validate_checkpoint_obs_shape(ppo_runner: MotionOnPolicyRunner, resume_path: str, *, heightmap_enabled: bool):
    """Fail early with a useful hint when the checkpoint and play obs shapes differ."""
    loaded_dict = torch.load(resume_path, map_location="cpu")
    model_state_dict = loaded_dict.get("model_state_dict", loaded_dict)

    checkpoint_actor = model_state_dict.get("actor.0.weight")
    checkpoint_critic = model_state_dict.get("critic.0.weight")
    if checkpoint_actor is None and checkpoint_critic is None:
        return

    policy = ppo_runner.alg.policy
    current_actor_in = getattr(policy.actor[0], "in_features", None) if hasattr(policy, "actor") else None
    current_critic_in = getattr(policy.critic[0], "in_features", None) if hasattr(policy, "critic") else None

    mismatches = []
    if checkpoint_actor is not None and current_actor_in is not None:
        checkpoint_actor_in = checkpoint_actor.shape[1]
        if checkpoint_actor_in != current_actor_in:
            mismatches.append(("actor", checkpoint_actor_in, current_actor_in))
    if checkpoint_critic is not None and current_critic_in is not None:
        checkpoint_critic_in = checkpoint_critic.shape[1]
        if checkpoint_critic_in != current_critic_in:
            mismatches.append(("critic", checkpoint_critic_in, current_critic_in))

    if not mismatches:
        return

    lines = [
        "Checkpoint observation shape does not match the play environment:",
        *[
            f"  {name}: checkpoint expects {checkpoint_in}, play env built {current_in}"
            for name, checkpoint_in, current_in in mismatches
        ],
    ]
    if not heightmap_enabled and any(checkpoint_in - current_in == 20 for _, checkpoint_in, current_in in mismatches):
        lines.append(
            "The checkpoint expects 20 extra observation features, which matches the height-map term. "
            "Re-run play with --heightmap."
        )
    elif not args_cli.depth_obs and any(checkpoint_in - current_in == 768 for _, checkpoint_in, current_in in mismatches):
        lines.append(
            "The checkpoint expects 768 extra observation features, which matches the default flattened depth term. "
            "Re-run play with --depth_obs."
        )
    else:
        lines.append("Use the same observation-affecting flags for play that were used during training.")
    raise RuntimeError("\n".join(lines))


# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No model artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Set zarr_path for multi-clip tasks
    if args_cli.zarr_path is not None:
        if not hasattr(env_cfg.commands.motion, "zarr_path"):
            raise ValueError("--zarr_path was provided, but this task does not use a zarr motion command.")
        env_cfg.commands.motion.zarr_path = args_cli.zarr_path

    # Override decimation if provided
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    # Limit clips for smaller GPUs
    if args_cli.max_clips is not None and hasattr(env_cfg.commands.motion, 'max_clips'):
        env_cfg.commands.motion.max_clips = args_cli.max_clips

    configure_height_map_obs(env_cfg, args_cli.heightmap)
    if getattr(env_cfg.scene, "height_scanner", None) is not None and args_cli.heightmap_debug_vis:
        env_cfg.scene.height_scanner.debug_vis = True

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.heightmap_debug_vis:
        print_height_map_obs_debug(env, env_cfg)
    if args_cli.depth_obs:
        print_depth_obs_debug(env, env_cfg)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play", wandb_run.id if args_cli.wandb_path else "local"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    validate_checkpoint_obs_shape(ppo_runner, resume_path, heightmap_enabled=args_cli.heightmap)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    # export_motion_policy_as_onnx(
    #     env.unwrapped,
    #     ppo_runner.alg.policy,
    #     normalizer=ppo_runner.obs_normalizer,
    #     path=export_model_dir,
    #     filename="policy.onnx",
    # )
    # attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
