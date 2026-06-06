# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train staircase RL agents with RSL-RL."""

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
parser.add_argument("--double_step", action="store_true", default=False, help="Enable original velocity-mismatch double-step penalty (WBT_DOUBLE_STEP).")
parser.add_argument("--double_step_contact", action="store_true", default=False, help="Enable stance_contact_penalty: penalize missing contact during reference stance.")
parser.add_argument("--double_step_slide", action="store_true", default=False, help="Enable stance_slide_penalty: penalize lateral sliding during reference stance.")
parser.add_argument("--double_step_drift", action="store_true", default=False, help="Enable stance_drift_penalty: penalize stance foot drifting from initial contact position.")
parser.add_argument("--double_step_all", action="store_true", default=False, help="Enable all four double-step terms (original + contact + slide + drift).")
parser.add_argument("--motion_joint_pos", action="store_true", default=False, help="Enable motion joint position reward.")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation (physics steps per policy step).")
parser.add_argument("--future_steps", type=str, default=None, help="Comma-separated future timestep offsets for ref observations (e.g., '5,10,15').")
parser.add_argument("--wandb_resume", type=str, default=None, help="Wandb run path to resume from (e.g., 'user/project/run_id'). Downloads latest checkpoint.")
parser.add_argument("--num_steps_per_env", type=int, default=None, help="Override num rollout steps per env per iteration.")
parser.add_argument("--layer_norm", action="store_true", default=False, help="Insert LayerNorm after each hidden activation in actor/critic MLPs.")
parser.add_argument("--heightmap", action="store_true", default=False, help="Enable task-configured height-map observations during training.")
parser.add_argument("--heightmap_debug_vis", action="store_true", default=False, help="Show height-map raycaster debug visualization.")
parser.add_argument("--depth_obs", action="store_true", default=False, help="Enable optional RGB-D camera depth observations.")
parser.add_argument("--use_depth", dest="depth_obs", action="store_true", help="Alias for --depth_obs.")
parser.add_argument(
    "--staircase_multiclip_debug",
    action="store_true",
    default=False,
    help="Print multiclip staircase clip/stair metadata and keep command debug visualization enabled.",
)
parser.add_argument("--depth_debug_save_frames", action="store_true", default=False, help="Save a few normalized depth frames during rollout.")
parser.add_argument("--depth_debug_max_frames", type=int, default=4, help="Maximum number of depth debug frames to save.")
parser.add_argument("--global_critic_obs", action="store_true", default=False, help="Use world-frame root-relative body positions in the critic (default: base-frame). Swaps robot_body_pos_b -> robot_body_pos_w_rootrel.")
parser.add_argument("--depth_encoder", action="store_true", default=False, help="Encode the flat depth slice with a small MLP before the actor/critic (requires --depth_obs).")
parser.add_argument("--depth_latent_dim", type=int, default=64, help="Output latent dim for --depth_encoder MLP (default: 64).")
parser.add_argument("--depth_encoder_hidden_dims", type=str, default="256,128", help="Hidden dims for --depth_encoder MLP (default: '256,128').")
parser.add_argument("--depth_cnn", action="store_true", default=False, help="Encode the flat depth slice with a 3-layer stride-2 CNN before the actor/critic (requires --depth_obs, mutually exclusive with --depth_encoder).")
parser.add_argument("--cnn_depth_latent_dim", type=int, default=64, help="Output latent dim for --depth_cnn (default: 64).")
parser.add_argument("--ppo_output", type=str, default="target", choices=["target", "delta-pseudotarget", "delta-all"],
                    help="PPO output mode: 'target' for absolute joint pos, 'delta-pseudotarget' for pseudo-target ONNX output, 'delta-all' for raw delta output.")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (default: elu).")
parser.add_argument("--gravity_curriculum", action="store_true", default=False, help="Enable gravity curriculum (ramp from reduced to full gravity).")
parser.add_argument("--start_gravity", type=float, default=-2.0, help="Starting Z gravity for gravity curriculum (default: -2.0).")
parser.add_argument("--gravity_ramp_steps", type=int, default=5000, help="Steps to ramp from start to full gravity (default: 5000).")
parser.add_argument(
    "--sampling",
    type=str,
    default=None,
    choices=["adaptive", "uniform"],
    help="Motion clip sampling strategy. Defaults to uniform for Staircase-MultiClip tasks and adaptive otherwise.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def resolve_sampling_mode(args: argparse.Namespace) -> str:
    """Resolve sampling mode with a task-specific default for staircase multiclip."""
    if args.sampling is not None:
        return args.sampling

    task_name = args.task or ""
    if task_name.startswith("Staircase-MultiClip-"):
        return "uniform"

    return "adaptive"


args_cli.sampling = resolve_sampling_mode(args_cli)

# always enable cameras to record video
if args_cli.video or args_cli.depth_obs:
    args_cli.enable_cameras = True

# Export CLI flags as env vars so __post_init__ in env configs can read them
if args_cli.curriculum:
    os.environ["WBT_CURRICULUM"] = "1"
if args_cli.double_step_all:
    os.environ["WBT_DOUBLE_STEP"] = "1"
    os.environ["WBT_DS_CONTACT"] = "1"
    os.environ["WBT_DS_SLIDE"] = "1"
    os.environ["WBT_DS_DRIFT"] = "1"
if args_cli.double_step:
    os.environ["WBT_DOUBLE_STEP"] = "1"
    os.environ["BONES_DOUBLE_STEP"] = "1"
if args_cli.double_step_contact:
    os.environ["WBT_DS_CONTACT"] = "1"
if args_cli.double_step_slide:
    os.environ["WBT_DS_SLIDE"] = "1"
if args_cli.double_step_drift:
    os.environ["WBT_DS_DRIFT"] = "1"
if args_cli.motion_joint_pos:
    os.environ["WBT_MOTION_JOINT_POS"] = "1"
if args_cli.depth_obs:
    os.environ["WBT_USE_DEPTH_OBS"] = "1"
if args_cli.global_critic_obs:
    os.environ["WBT_GLOBAL_CRITIC_OBS"] = "1"
if args_cli.depth_encoder and args_cli.depth_cnn:
    raise ValueError("--depth_encoder and --depth_cnn are mutually exclusive.")
if args_cli.depth_encoder:
    if not args_cli.depth_obs:
        raise ValueError("--depth_encoder requires --depth_obs.")
    os.environ["WBT_USE_DEPTH_ENCODER"] = "1"
if args_cli.depth_cnn:
    if not args_cli.depth_obs:
        raise ValueError("--depth_cnn requires --depth_obs.")
    os.environ["WBT_USE_DEPTH_CNN"] = "1"
if args_cli.depth_debug_save_frames:
    os.environ["WBT_DEPTH_SAVE_FRAMES"] = "1"
os.environ["WBT_DEPTH_DEBUG_MAX_FRAMES"] = str(args_cli.depth_debug_max_frames)
os.environ["WBT_PPO_OUTPUT"] = args_cli.ppo_output
if args_cli.staircase_multiclip_debug:
    os.environ["WBT_STAIRCASE_MULTICLIP_DEBUG"] = "1"
if args_cli.gravity_curriculum:
    os.environ["BONES_GRAVITY_CURRICULUM"] = "1"
    os.environ["BONES_START_GRAVITY"] = str(args_cli.start_gravity)
    os.environ["BONES_GRAVITY_RAMP_STEPS"] = str(args_cli.gravity_ramp_steps)
os.environ["BONES_SAMPLING"] = args_cli.sampling

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import pickle
import yaml

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string."""
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    if value.__name__ == "<lambda>":
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"
    module_name = value.__module__
    function_name = value.__name__
    return f"{module_name}:{function_name}"


def class_to_dict(obj: object) -> dict[Any, Any]:
    """Convert an object into dictionary recursively."""
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    data = dict()
    for key, value in obj_dict.items():
        if isinstance(key, str) and key.startswith("__"):
            continue
        if callable(value):
            data[key] = callable_to_string(value)
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def dump_pickle(filename: str, data):
    """Saves data into a pickle file safely."""
    if not filename.endswith("pkl"):
        filename += ".pkl"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely."""
    if not filename.endswith("yaml"):
        filename += ".yaml"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(data, dict):
        data = class_to_dict(data)
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)


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
        print("[HEIGHT_MAP_DEBUG] Enabled task-configured height-map observations.")
        return

    if height_scanner_cfg is not None:
        env_cfg.scene.height_scanner = None
    for group_name in ("policy", "critic", "diffusion_collect"):
        group_cfg = getattr(env_cfg.observations, group_name, None)
        if group_cfg is not None and getattr(group_cfg, "height_scan", None) is not None:
            group_cfg.height_scan = None
    if height_scanner_cfg is not None or has_height_scan_term:
        print("[HEIGHT_MAP_DEBUG] Disabled task-configured height-map observations. Pass --heightmap to train with them.")


def _decode_zarr_strings(values) -> list[str]:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_staircase_multiclip_scene_specs(zarr_path: str) -> list[dict]:
    import zarr

    store = zarr.open(zarr_path, mode="r")
    required = ["staircase_id", "staircase_asset_path", "staircase_usd_dir"]
    missing = [key for key in required if key not in store]
    if missing:
        raise KeyError(
            f"Staircase multiclip zarr requires datasets {required}. Missing {missing} in {zarr_path}."
        )

    staircase_ids = store["staircase_id"][:]
    asset_paths = _decode_zarr_strings(store["staircase_asset_path"][:])
    usd_dirs = _decode_zarr_strings(store["staircase_usd_dir"][:])
    if not (len(staircase_ids) == len(asset_paths) == len(usd_dirs)):
        raise ValueError(
            "staircase_id, staircase_asset_path, and staircase_usd_dir must have one value per clip."
        )

    specs_by_id: dict[int, dict] = {}
    for staircase_id_raw, asset_path, usd_dir in zip(staircase_ids, asset_paths, usd_dirs):
        staircase_id = int(staircase_id_raw)
        spec = {"staircase_id": staircase_id, "asset_path": asset_path, "usd_dir": usd_dir}
        if staircase_id in specs_by_id:
            prev = specs_by_id[staircase_id]
            if prev["asset_path"] != asset_path or prev["usd_dir"] != usd_dir:
                raise ValueError(
                    f"Inconsistent asset mapping for staircase_id={staircase_id}: "
                    f"{prev} vs {spec}"
                )
        else:
            specs_by_id[staircase_id] = spec
    return [specs_by_id[key] for key in sorted(specs_by_id)]


import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        correct_device = f"cuda:{local_rank}"
        env_cfg.sim.device = correct_device
        agent_cfg.device = correct_device
    else:
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
    configure_height_map_obs(env_cfg, args_cli.heightmap)
    if getattr(env_cfg.scene, "height_scanner", None) is not None and args_cli.heightmap_debug_vis:
        env_cfg.scene.height_scanner.debug_vis = True
        print("[HEIGHT_MAP_DEBUG] Enabled height_scanner debug visualization.")
    elif args_cli.heightmap_debug_vis:
        print("[HEIGHT_MAP_DEBUG] Ignoring --heightmap_debug_vis because height-map observations are disabled.")

    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env

    if args_cli.future_steps is not None:
        steps = [int(s.strip()) for s in args_cli.future_steps.split(",")]
        if hasattr(env_cfg.commands.motion, "future_steps"):
            env_cfg.commands.motion.future_steps = steps
            print(f"[INFO] Future ref steps: {steps}")

    import pathlib
    if args_cli.zarr_path is not None:
        print(f"[INFO] Loading motion from Zarr: {args_cli.zarr_path}")
        if args_cli.task == "Staircase-G1-v0":
            raise ValueError(
                "--zarr_path is not supported on Staircase-G1-v0. "
                "Use --task Staircase-MultiClip-G1-v0 for multiclip staircase training."
            )
        if args_cli.task == "Staircase-G1-Play-v0":
            raise ValueError(
                "--zarr_path is not supported on Staircase-G1-Play-v0. "
                "Use --task Staircase-MultiClip-G1-Play-v0 for multiclip staircase playback."
            )
        if "Staircase-MultiClip" in (args_cli.task or ""):
            from whole_body_tracking.tasks.staircase.staircase_env_cfg import configure_multiclip_staircase_scene

            staircase_specs = load_staircase_multiclip_scene_specs(args_cli.zarr_path)
            staircase_variant_names = configure_multiclip_staircase_scene(env_cfg.scene, staircase_specs)
            env_cfg.commands.motion.zarr_path = args_cli.zarr_path
            env_cfg.commands.motion.staircase_variant_names = staircase_variant_names
            if args_cli.staircase_multiclip_debug:
                env_cfg.commands.motion.debug_vis = True
                if getattr(env_cfg.scene, "height_scanner", None) is not None:
                    env_cfg.scene.height_scanner.debug_vis = True
            print(
                "[INFO] Configured staircase multiclip variants: "
                + ", ".join(
                    f"id={spec['staircase_id']}:{pathlib.Path(spec['asset_path']).name}"
                    for spec in staircase_specs
                )
            )
        else:
            env_cfg.commands.motion.zarr_path = args_cli.zarr_path
            if hasattr(env_cfg.commands.motion, "exclude_objects"):
                env_cfg.commands.motion.exclude_objects = not args_cli.include_objects
        registry_name = f"zarr:{args_cli.zarr_path}"
    elif args_cli.registry_name is not None:
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

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print_height_map_obs_debug(env, env_cfg)
    if args_cli.depth_obs:
        print_depth_obs_debug(env, env_cfg)

    env = RslRlVecEnvWrapper(env)

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

    train_cfg = agent_cfg.to_dict()
    if args_cli.depth_encoder:
        hidden_dims = [int(x) for x in args_cli.depth_encoder_hidden_dims.split(",")]
        train_cfg["policy"]["depth_latent_dim"] = args_cli.depth_latent_dim
        train_cfg["policy"]["depth_encoder_hidden_dims"] = hidden_dims
        print(
            f"[DepthEncoderActorCritic] config: latent_dim={args_cli.depth_latent_dim} "
            f"hidden_dims={hidden_dims}"
        )
    if args_cli.depth_cnn:
        train_cfg["policy"]["depth_latent_dim"] = args_cli.cnn_depth_latent_dim
        train_cfg["policy"]["depth_height"] = 24   # DEPTH_OBS_HEIGHT
        train_cfg["policy"]["depth_width"] = 32    # DEPTH_OBS_WIDTH
        train_cfg["policy"]["depth_channels"] = 1
        print(f"[DepthCNNActorCritic] config: latent_dim={args_cli.cnn_depth_latent_dim} input=(1,24,32)")

    runner = OnPolicyRunner(
        env, train_cfg, log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    if _wandb_resume_path is not None:
        print(f"[INFO]: Loading wandb checkpoint: {_wandb_resume_path}")
        runner.load(_wandb_resume_path)

    if args_cli.layer_norm:
        import torch.nn as _nn

        def _insert_layer_norm(mlp: _nn.Sequential):
            """Insert LayerNorm after each activation in an MLP Sequential."""
            new_layers = []
            for layer in mlp:
                new_layers.append(layer)
                if isinstance(layer, (_nn.SiLU, _nn.ELU, _nn.ReLU, _nn.LeakyReLU, _nn.GELU, _nn.Mish)):
                    for prev in reversed(new_layers[:-1]):
                        if isinstance(prev, _nn.Linear):
                            new_layers.append(_nn.LayerNorm(prev.out_features))
                            break
            mlp._modules.clear()
            for idx, layer in enumerate(new_layers):
                mlp.add_module(str(idx), layer)

        policy = runner.alg.get_policy() if hasattr(runner.alg, "get_policy") else runner.alg.policy
        if hasattr(policy, "mlp"):
            _insert_layer_norm(policy.mlp)
            print(f"[INFO] LayerNorm inserted into actor MLP: {policy.mlp}")
        critic = getattr(runner.alg, "critic", None) or getattr(runner.alg, "value_function", None)
        if critic is not None and hasattr(critic, "mlp"):
            _insert_layer_norm(critic.mlp)
            print("[INFO] LayerNorm inserted into critic MLP")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    try:
        wandb.finish()
    except (NameError, Exception):
        pass

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
