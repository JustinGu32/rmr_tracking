# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr motion store (for multi-clip training).")
parser.add_argument("--include_objects", action="store_true", default=False, help="Include motions with object manipulation (excluded by default).")
parser.add_argument("--double_step", action="store_true", default=False, help="Enable double-step penalty reward.")
parser.add_argument("--ppo_output", type=str, default="target", choices=["target", "delta-pseudotarget", "delta-all"],
                    help="PPO output mode: 'target' for absolute joint pos, 'delta-pseudotarget' for pseudo-target ONNX output, 'delta-all' for raw delta output.")
parser.add_argument("--push", type=str, default="normal", choices=["normal", "none", "soft"],
                    help="Push perturbation: 'normal' (default), 'soft' (reduced range), 'none' (disabled).")
parser.add_argument("--no_command_obs", action="store_true", default=False, help="Remove generated_commands observation from policy.")
parser.add_argument("--crane", action="store_true", default=False, help="Add foot contact state penalty and tight foot tracking for single-leg stance motions.")
parser.add_argument("--decimation", type=int, default=None, help="Override decimation (e.g., 6 for 33hz, 4 for 50hz).")
parser.add_argument("--curriculum", action="store_true", default=False, help="Enable assistive spring force curriculum.")
parser.add_argument("--assist_mode", type=str, default=None, choices=["both", "gravity_only", "spring_only", "none"], help="Assistive force mode for staircase training.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Export CLI flags as env vars so __post_init__ in env configs can read them
if args_cli.double_step:
    os.environ["BONES_DOUBLE_STEP"] = "1"
os.environ["BONES_PPO_OUTPUT"] = args_cli.ppo_output
os.environ["BONES_PUSH"] = args_cli.push
if args_cli.no_command_obs:
    os.environ["BONES_NO_COMMAND_OBS"] = "1"
if args_cli.crane:
    os.environ["BONES_CRANE"] = "1"
if args_cli.curriculum:
    os.environ["BONES_CURRICULUM"] = "1"
if args_cli.assist_mode is not None:
    os.environ["BONES_ASSIST_MODE"] = args_cli.assist_mode

# Auto-detect distributed training (torchrun sets LOCAL_RANK)
if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config



import pickle
import yaml


from collections.abc import Iterable, Mapping
from typing import Any

# from .array import TENSOR_TYPE_CONVERSIONS, TENSOR_TYPES
# from .string import callable_to_string, string_to_callable, string_to_slice

"""
Dictionary <-> Class operations.
"""
from collections.abc import Callable, Sequence

def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        A string representation of the callable object.
    """
    # check if callable
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        # we resolve the lambda expression by checking the source code and extracting the line with lambda expression
        # we also remove any comments from the line
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"
    else:
        # get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        # return the string
        return f"{module_name}:{function_name}"

def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        # We have to treat torch tensors specially because `torch.tensor.__dict__` returns an empty
        # dict, which would mean that a torch.tensor would be stored as an empty dict. Instead we
        # want to store it directly as the tensor.
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # check if attribute is a list or tuple
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def dump_pickle(filename: str, data):
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)



# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Auto-generate run name suffix from CLI flags
    flag_suffix = f"{args_cli.ppo_output}_push-{args_cli.push}"
    if args_cli.no_command_obs:
        flag_suffix += "_no-cmd-obs"
    if args_cli.double_step:
        flag_suffix += "_dstep"
    if args_cli.crane:
        flag_suffix += "_crane"
    if args_cli.curriculum:
        flag_suffix += "_curriculum"
    if agent_cfg.run_name:
        agent_cfg.run_name = f"{agent_cfg.run_name}_{flag_suffix}"
    else:
        agent_cfg.run_name = flag_suffix
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    # In multi-GPU mode, AppLauncher sets device based on LOCAL_RANK — don't override it
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        correct_device = f"cuda:{local_rank}"
        env_cfg.sim.device = correct_device
        agent_cfg.device = correct_device
    else:
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # load the motion file from zarr path or wandb registry
    import pathlib
    if args_cli.zarr_path is not None:
        # Multi-clip training from local Zarr store
        print(f"[INFO] Loading motion from Zarr: {args_cli.zarr_path}")
        env_cfg.commands.motion.zarr_path = args_cli.zarr_path
        env_cfg.commands.motion.exclude_objects = not args_cli.include_objects
        registry_name = f"zarr:{args_cli.zarr_path}"
    elif args_cli.registry_name is not None:
        # Single-clip training from wandb registry (original path)
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        print(f"DEBUG: registry_name is {registry_name}")
        import wandb
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_path = str(pathlib.Path(artifact.download()) / "motion.npz")
        if hasattr(env_cfg.commands.motion, 'motion_files'):
            env_cfg.commands.motion.motion_files = [motion_path]
        else:
            env_cfg.commands.motion.motion_file = motion_path
    else:
        raise ValueError("Either --zarr_path or --registry_name must be provided.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # log observation dimensions to wandb config
    import wandb
    if wandb.run is not None:
        obs_mgr = env.unwrapped.observation_manager
        for group_name in obs_mgr.active_terms:
            term_names = obs_mgr.active_terms[group_name]
            term_dims = obs_mgr.group_obs_term_dim[group_name]
            for name, dim in zip(term_names, term_dims):
                wandb.config[f"obs_dim/{group_name}/{name}"] = dim

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    wandb.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
