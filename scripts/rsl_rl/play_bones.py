"""Script to play a checkpoint of an RL agent from RSL-RL (Bones tasks with MultiMotionCommand)."""

"""
python scripts/rsl_rl/play_bones.py --task=Bones-Flat-chip-G1-Play-v0 --num_envs=2 --wandb_path=robot-mcrobotface/bones_crane_ablation/ke5triph
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (Bones).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_folder", type=str, default=None, help="Directory to save recorded videos.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr store for multi-clip tasks.")
parser.add_argument("--max_clips", type=int, default=None, help="Max clips to load from Zarr (for smaller GPUs).")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (must match training).")
parser.add_argument("--start_from_beginning", action="store_true", default=False,
                    help="Start each episode from the beginning of the clip instead of adaptive sampling.")
parser.add_argument("--decimation", type=int, default=None, help="Override decimation (e.g., 6 for 33hz, 4 for 50hz).")
parser.add_argument("--bones_popart", action="store_true", default=False, help="Use PopArt multi-head critic (must match training).")
parser.add_argument("--popart_head_mode", type=str, default="grouped", choices=["per_term", "grouped"],
                    help="PopArt critic head layout for inference (default: per_term).")
parser.add_argument("--popart_group_preset", type=str, default="upper_lower", choices=["upper_lower", "motion_tracking", "actual_individual", "limb_tracking", "limb_tracking_ul", "limb_tracking_ul_individual"],
                    help="Default grouped PopArt preset when --popart_head_mode grouped and no explicit popart_groups are provided.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.start_from_beginning:
    os.environ["BONES_START_FROM_BEGINNING"] = "1"
if args_cli.bones_popart:
    os.environ["BONES_POPART"] = "1"
    os.environ["BONES_POPART_HEADS"] = args_cli.popart_head_mode

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

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

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.bones.popart_reward_manager import install_bones_per_term_reward_manager
from whole_body_tracking.utils.bones_popart import BonesPopArtOnPolicyRunner
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    if args_cli.bones_popart_balanced and args_cli.bones_popart_individual:
        raise ValueError("--bones_popart_balanced and --bones_popart_individual are mutually exclusive.")
    if hasattr(agent_cfg, "bones_popart"):
        agent_cfg.bones_popart = dict(agent_cfg.bones_popart)
        if args_cli.bones_popart or args_cli.bones_popart_balanced or args_cli.bones_popart_individual or args_cli.bones_no_popart_stats:
            agent_cfg.bones_popart["enabled"] = True
        if args_cli.bones_no_popart_stats:
            agent_cfg.bones_popart["use_popart"] = False
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

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
            motion_path = args_cli.motion_file
            print(f"[INFO]: Using motion file from CLI: {motion_path}")
        else:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print(f"[WARN] No motion artifact found in the run.")
                motion_path = None
            else:
                motion_path = str(pathlib.Path(art.download()) / "motion.npz")

        if motion_path is not None:
            if hasattr(env_cfg.commands.motion, "motion_files"):
                env_cfg.commands.motion.motion_files = [motion_path]
            else:
                env_cfg.commands.motion.motion_file = motion_path

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    checkpoint = torch.load(resume_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    checkpoint_uses_popart = any(key.startswith("critic_trunk.") for key in model_state_dict)
    use_popart_runner = args_cli.bones_popart or checkpoint_uses_popart
    if checkpoint_uses_popart:
        print("[INFO] Detected PopArt checkpoint architecture from saved model state.")
    if use_popart_runner:
        agent_cfg.algorithm.use_popart_multihead = True
        agent_cfg.algorithm.popart_head_mode = args_cli.popart_head_mode
        agent_cfg.algorithm.popart_group_preset = args_cli.popart_group_preset

    # Set zarr_path for multi-clip tasks
    if args_cli.zarr_path is not None:
        env_cfg.commands.motion.zarr_path = args_cli.zarr_path

    # Limit clips for smaller GPUs
    if args_cli.max_clips is not None and hasattr(env_cfg.commands.motion, 'max_clips'):
        env_cfg.commands.motion.max_clips = args_cli.max_clips

    # Disable contradictory reward terms depending on the populated head setting
    if hasattr(env_cfg, "rewards"):
        if getattr(args_cli, "bones_popart_balanced", False):
            if hasattr(env_cfg.rewards, "vr_position_upper"):
                env_cfg.rewards.vr_position_upper.weight = 0.0
            if hasattr(env_cfg.rewards, "vr_position_lower"):
                env_cfg.rewards.vr_position_lower.weight = 0.0
            agent_cfg.bones_popart["reward_heads"] = list(BALANCED_REWARD_HEADS)
        elif getattr(args_cli, "bones_popart_individual", False):
            if hasattr(env_cfg.rewards, "vr_position_upper"): env_cfg.rewards.vr_position_upper.weight = 0.0
            if hasattr(env_cfg.rewards, "vr_position_lower"): env_cfg.rewards.vr_position_lower.weight = 0.0
            agent_cfg.bones_popart["reward_heads"] = list(INDIVIDUAL_REWARD_HEADS)
        else:
            for limb in ["left_arm", "right_arm", "torso", "left_leg", "right_leg", "pelvis"]:
                term_name = f"vr_position_{limb}"
                if hasattr(env_cfg.rewards, term_name):
                    getattr(env_cfg.rewards, term_name).weight = 0.0
            agent_cfg.bones_popart["reward_heads"] = list(DEFAULT_REWARD_HEADS)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": args_cli.video_folder if args_cli.video_folder else os.path.join(log_dir, "videos", "play", wandb_run.id if args_cli.wandb_path else "local"),
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

    if use_popart_runner:
        install_bones_per_term_reward_manager(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    runner_cls = BonesPopArtOnPolicyRunner if use_popart_runner else OnPolicyRunner
    train_cfg = agent_cfg.to_dict()
    if not use_popart_runner:
        # Strip PopArt-specific keys that the stock PPO doesn't accept
        for key in [
            "use_popart_multihead", "popart_head_mode", "popart_groups",
            "popart_group_preset",
            "popart_grouped_actor_weight_mode",
            "popart_momentum", "popart_epsilon", "popart_normalize_actor_weights",
            "popart_actor_advantage_scaling",
        ]:
            train_cfg.get("algorithm", {}).pop(key, None)
    ppo_runner = runner_cls(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.alg.policy.actor_obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # upload updated onnx to wandb
    if args_cli.wandb_path:
        onnx_path = os.path.join(export_model_dir, "policy.onnx")
        wandb_run.upload_file(onnx_path)
        print(f"[INFO]: Uploaded {onnx_path} to wandb run: {run_path}")

    # Track termination counts
    term_manager = env.unwrapped.termination_manager
    term_counts = {name: 0 for name in term_manager.active_terms}
    total_episodes = 0

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
            obs, _, dones, _ = env.step(actions)

        # count terminations per term
        for name in term_manager.active_terms:
            fired = term_manager.get_term(name)
            term_counts[name] += int(fired.sum().item())
        total_episodes += int(dones.sum().item())

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # Save termination stats
    term_stats = {
        "total_episodes": total_episodes,
        "total_steps": timestep,
        "termination_counts": term_counts,
    }
    video_folder = args_cli.video_folder if args_cli.video_folder else os.path.join(log_dir, "videos", "play", wandb_run.id if args_cli.wandb_path else "local")
    os.makedirs(video_folder, exist_ok=True)
    stats_path = os.path.join(video_folder, "termination_stats.json")
    with open(stats_path, "w") as f:
        json.dump(term_stats, f, indent=2)
    print(f"[INFO] Termination stats saved to: {stats_path}")
    print(json.dumps(term_stats, indent=2))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
