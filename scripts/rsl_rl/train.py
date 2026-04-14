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
parser.add_argument("--curriculum", action="store_true", default=False, help="Enable assistive spring force curriculum.")
parser.add_argument("--double_step", action="store_true", default=False, help="Enable double-step penalty reward.")
parser.add_argument("--motion_joint_pos", action="store_true", default=False, help="Enable motion joint position reward.")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation (physics steps per policy step).")
parser.add_argument("--future_steps", type=str, default=None, help="Comma-separated future timestep offsets for ref observations (e.g., '5,10,15').")
parser.add_argument("--wandb_resume", type=str, default=None, help="Wandb run path to resume from (e.g., 'user/project/run_id'). Downloads latest checkpoint.")
parser.add_argument("--num_steps_per_env", type=int, default=None, help="Override num rollout steps per env per iteration.")
parser.add_argument("--layer_norm", action="store_true", default=False, help="Insert LayerNorm after each hidden activation in actor/critic MLPs.")
parser.add_argument("--ppo_output", type=str, default="target", choices=["target", "delta-pseudotarget", "delta-all"],
                    help="PPO output mode: 'target' for absolute joint pos, 'delta-pseudotarget' for pseudo-target ONNX output, 'delta-all' for raw delta output.")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (default: elu).")
# parser.add_argument("--assist_mode", type=str, default=None, choices=["both", "gravity_only", "spring_only", "none"], help="Assistive force mode for staircase training.")
parser.add_argument("--gravity_curriculum", action="store_true", default=False, help="Enable gravity curriculum (ramp from reduced to full gravity).")
parser.add_argument("--start_gravity", type=float, default=-2.0, help="Starting Z gravity for gravity curriculum (default: -2.0).")
parser.add_argument("--gravity_ramp_steps", type=int, default=5000, help="Steps to ramp from start to full gravity (default: 5000).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Export CLI flags as env vars so __post_init__ in env configs can read them
if args_cli.curriculum:
    os.environ["WBT_CURRICULUM"] = "1"
if args_cli.double_step:
    os.environ["WBT_DOUBLE_STEP"] = "1"
    os.environ["BONES_DOUBLE_STEP"] = "1"
if args_cli.motion_joint_pos:
    os.environ["WBT_MOTION_JOINT_POS"] = "1"
os.environ["WBT_PPO_OUTPUT"] = args_cli.ppo_output
# if args_cli.assist_mode is not None:
#     os.environ["WBT_ASSIST_MODE"] = args_cli.assist_mode
if args_cli.gravity_curriculum:
    os.environ["BONES_GRAVITY_CURRICULUM"] = "1"
    os.environ["BONES_START_GRAVITY"] = str(args_cli.start_gravity)
    os.environ["BONES_GRAVITY_RAMP_STEPS"] = str(args_cli.gravity_ramp_steps)

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
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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

    # Override decimation if provided
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    # Override num_steps_per_env if provided
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env

    # Configure future reference motion observations
    if args_cli.future_steps is not None:
        steps = [int(s.strip()) for s in args_cli.future_steps.split(",")]
        if hasattr(env_cfg.commands.motion, 'future_steps'):
            env_cfg.commands.motion.future_steps = steps
            print(f"[INFO] Future ref steps: {steps}")

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
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")
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

    # ── Wandb resume: download checkpoint from a previous run ──
    _wandb_resume_path = None
    if args_cli.wandb_resume is not None:
        import wandb as _wandb
        api = _wandb.Api()
        run_path = args_cli.wandb_resume
        wandb_run = api.run(run_path)
        model_files = [f for f in wandb_run.files() if "model" in f.name and f.name.endswith(".pt")]
        if not model_files:
            raise RuntimeError(f"No model checkpoints found in wandb run: {run_path}")
        latest_file = max(model_files, key=lambda x: int(x.name.split("_")[1].split(".")[0]))
        dl_dir = os.path.join("logs", "rsl_rl", "wandb_resume")
        latest_file.download(dl_dir, replace=True)
        _wandb_resume_path = os.path.join(dl_dir, latest_file.name)
        print(f"[INFO]: Resuming from wandb run {wandb_run.id}, checkpoint: {latest_file.name}")

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

    # Load wandb checkpoint into runner (after runner is created)
    if _wandb_resume_path is not None:
        print(f"[INFO]: Loading wandb checkpoint: {_wandb_resume_path}")
        runner.load(_wandb_resume_path)

    # Insert LayerNorm into actor/critic MLPs if requested
    if args_cli.layer_norm:
        import torch.nn as _nn
        def _insert_layer_norm(mlp: _nn.Sequential):
            """Insert LayerNorm after each activation in an MLP Sequential."""
            new_layers = []
            for layer in mlp:
                new_layers.append(layer)
                if isinstance(layer, (_nn.SiLU, _nn.ELU, _nn.ReLU, _nn.LeakyReLU, _nn.GELU, _nn.Mish)):
                    # Get the output dim from the preceding Linear layer
                    for prev in reversed(new_layers[:-1]):
                        if isinstance(prev, _nn.Linear):
                            new_layers.append(_nn.LayerNorm(prev.out_features))
                            break
            # Rebuild the Sequential
            mlp._modules.clear()
            for idx, layer in enumerate(new_layers):
                mlp.add_module(str(idx), layer)

        policy = runner.alg.get_policy() if hasattr(runner.alg, 'get_policy') else runner.alg.policy
        if hasattr(policy, 'mlp'):
            _insert_layer_norm(policy.mlp)
            print(f"[INFO] LayerNorm inserted into actor MLP: {policy.mlp}")
        # Also apply to critic if it has a separate mlp
        critic = getattr(runner.alg, 'critic', None) or getattr(runner.alg, 'value_function', None)
        if critic is not None and hasattr(critic, 'mlp'):
            _insert_layer_norm(critic.mlp)
            print(f"[INFO] LayerNorm inserted into critic MLP")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    try:
        wandb.finish()
    except (NameError, Exception):
        pass

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
