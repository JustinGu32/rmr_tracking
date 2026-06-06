"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""example command: 
python scripts/rsl_rl/collect_dataset.py \
    --task Tracking-Flat-G1-Collect-v0   \
    --num_envs 2 \
    --wandb_path robot-mcrobotface/takara_walk_isaac/nxotitq9 \
    --num_steps_collect 60 \
    --num_eps_collect 10000 \
    --episode_collect_length 4 \
    --min_delay 0 \
    --max_delay 0 \
    --min_sample_idx 0 \
    --max_sample_idx 1000 \
    --num_obstacles 4 \
    --video

python scripts/rsl_rl/collect_dataset.py \
    --task Chair-Step-G1-Collect-v0   \
    --num_envs 2 \
    --wandb_path robot-mcrobotface/chair_step/q4yp8tny \
    --num_steps_collect 60 \
    --num_eps_collect 10000 \
    --episode_collect_length 4 \
    --min_delay 0 \
    --max_delay 0 \
    --min_sample_idx 0 \
    --max_sample_idx 1000 \
    --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from tensordict import TensorDict

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import pathlib 
from pathlib import Path
import matplotlib.pyplot as plt
import traceback
import random

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument("--episode_collect_length", type=str, default=None, help="how long to run the episode in total.")
parser.add_argument("--num_steps_collect", type=int, default=200, help="num of steps to store as an episode (smaller than episode_length).")
parser.add_argument("--policy_frequency", type=int, default=50, help="num of steps to store as an episode (smaller than episode_length).")

parser.add_argument("--num_eps_collect", type=int, default=500, help="num of episodes to collect.")

parser.add_argument("--motion_file", type=str, default=None, help="Motion File")
parser.add_argument("--save_folder", type=str, default=None, help="save folder")
parser.add_argument(
    "--dump_raw_depth_dir",
    type=str,
    default=None,
    help="Optional directory to save raw pre-encoding depth frames as .npy files.",
)
parser.add_argument(
    "--dump_raw_depth_limit",
    type=int,
    default=1,
    help="Maximum number of raw depth frames to dump when --dump_raw_depth_dir is set.",
)

parser.add_argument("--min_delay", type=int, default=0, help="actuator delay.")
parser.add_argument("--max_delay", type=int, default=0, help="actuator delay.")
parser.add_argument("--num_obstacles", type=int, default=0, help="Number of obstacles to generate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the experiment.")
parser.add_argument(
    "--disable_rgb",
    action="store_true",
    default=False,
    help="Skip the SigLIP RGB encoder and store depth-only embeddings (no rgb_embed keys). "
    "Matches the sim2sim --disable_rgb model and the existing STAIRCASE_DATA_COLLECTED schema.",
)

def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

parser.add_argument("--min_sample_idx", type=none_or_int, default=None, help="actuator delay.")
parser.add_argument("--max_sample_idx", type=none_or_int, default=None, help="actuator delay.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import zarr
import time
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from replay_buffer import ReplayBuffer
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from torch.distributions import Normal


# Import extensions to set up environment tasks
# Set ENABLE_CAMERAS before importing tasks so config files pick it up
os.environ["ENABLE_CAMERAS"] = "1"
import whole_body_tracking.tasks  # noqa: F401
from transformers import AutoImageProcessor, SiglipModel
from transformers import AutoImageProcessor, SiglipModel
import torch.nn.functional as F

from whole_body_tracking.utils.defm_utils import preprocess_depth_batch
from PIL import Image


def _parse_model_checkpoint_iteration(filename: str) -> int | None:
    """Return the numeric step from `model_<step>.pt`, or None if it does not match."""
    if not filename.startswith("model_") or not filename.endswith(".pt"):
        return None
    try:
        return int(filename.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _resolve_wandb_checkpoint(api, wandb_path: str):
    """Resolve a wandb path to a run, checkpoint filename, and downloadable file."""
    requested_file = None
    run_path = wandb_path
    maybe_file = wandb_path.rsplit("/", 1)[-1]
    if _parse_model_checkpoint_iteration(maybe_file) is not None:
        requested_file = maybe_file
        run_path = wandb_path.rsplit("/", 1)[0]

    wandb_run = api.run(run_path)

    if requested_file is not None:
        wandb_file = wandb_run.file(requested_file)
        if wandb_file is None:
            raise RuntimeError(
                f"Requested wandb checkpoint '{requested_file}' was not found in run '{run_path}'."
            )
        return wandb_run, run_path, requested_file, wandb_file

    try:
        run_files = list(wandb_run.files())
    except TypeError as exc:
        raise RuntimeError(
            f"W&B returned an empty file listing for run '{run_path}'. "
            "Pass an explicit checkpoint path like 'entity/project/run_id/model_500.pt', "
            "or verify that the run uploaded checkpoint files."
        ) from exc

    model_files = []
    for wandb_file in run_files:
        step = _parse_model_checkpoint_iteration(wandb_file.name)
        if step is not None:
            model_files.append((step, wandb_file))

    if not model_files:
        raise RuntimeError(
            f"No uploaded checkpoint files matching 'model_<step>.pt' were found in run '{run_path}'."
        )

    _, latest_file = max(model_files, key=lambda item: item[0])
    return wandb_run, run_path, latest_file.name, latest_file


def main():
    """Play with RSL-RL agent."""
    # parse configuration


    # Set environment variable for dynamic obstacles BEFORE parsing config
    os.environ["NUM_OBSTACLES"] = str(args_cli.num_obstacles)

    # Set random seed if provided
    if args_cli.seed is not None:
        print(f"[INFO] Setting random seed to {args_cli.seed}")
        random.seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args_cli.seed)
            torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )    

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        api = wandb.Api()
        wandb_run, run_path, file, wandb_file = _resolve_wandb_checkpoint(api, args_cli.wandb_path)

        wandb_file.download(f"./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
        
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        art_combined = next((a for a in wandb_run.used_artifacts() if a.type == "combined_motions"), None)
        
        # import ipdb; ipdb.set_trace() 
        # motion_file_name = 'motion'
        # import ipdb; ipdb.set_trace()
        if art is not None:
            print(f"[INFO]: Downloading motion file")
            # motion_file_name = art.name.split(':')[0] +'.npz'
            # motion_file_name = 'motion.npz'
            motion_file_name = art.file().split('/')[-1]
            # import ipdb; ipdb.set_trace()
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / motion_file_name)

        if art_combined is not None:
            motion_file_name = art_combined.file().split('/')[-1]
            # motion_file_name = 'motion.npz'

            env_cfg.commands.motion.motion_file = str(pathlib.Path(art_combined.download()) / motion_file_name)
        if art is None and art_combined is None:
            print('[INFO] No motion file found')

            
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Set parameters from cli 
    env_cfg.episode_length_s = float(args_cli.episode_collect_length) if args_cli.episode_collect_length is not None else float(env_cfg.episode_length_s) 
    # import ipdb; ipdb.set_trace()

    for actuator_name in ['legs', 'feet', 'waist', 'waist_yaw', 'arms']:
        env_cfg.scene.robot.actuators[actuator_name].min_delay = args_cli.min_delay
        env_cfg.scene.robot.actuators[actuator_name].max_delay = args_cli.max_delay
        # env_cfg.scene.robot.actuators[actuator_name].delay_change_interval = your_interval_value

    env_cfg.commands.motion.min_sample_idx = args_cli.min_sample_idx
    env_cfg.commands.motion.max_sample_idx = args_cli.max_sample_idx
    env_cfg.commands.motion.steps_collect = args_cli.num_steps_collect

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", args_cli.task),
            "step_trigger": lambda step: step % args_cli.video_length == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "inclusion",
        }
        print((f"[INFO] Recording videos to: {video_kwargs['video_folder']}"))
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # env_cfg 
    # recorded_obs = []
    # recorded_acs = []
    # recorded_rgb_embed = []
    # recorded_depth_embed = []
    episode_ends = []
    
    # Initialize ReplayBuffer to save directly to disk
    COLLECT_STEPS = int(args_cli.num_steps_collect)
    NUM_EPISODE = int(args_cli.num_eps_collect)
    
    # Noise params for filename
    # noise_level = .05
    # hip_noise = .05
    # knee_noise = .05
    # ankle_noise = .1

    # noise_level = .3
    # hip_noise = .3
    # knee_noise = .3
    # ankle_noise = .5

    noise_level = .15
    hip_noise = .15
    knee_noise = .15
    ankle_noise = .2

    hip_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if 'hip' in name ]
    knee_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if 'knee_joint' in name ]
    ankle_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if  'ankle' in name]

    
    from datetime import datetime
    timestamp = datetime.now().strftime("%d_%H%M")
    base_filename = f'{timestamp}_seed-{args_cli.seed}_{wandb_run.name}_ep-{NUM_EPISODE}_steps-{COLLECT_STEPS}_dec-{env_cfg.decimation}_delay-{args_cli.min_delay}-{args_cli.max_delay}_noise-{noise_level}_hip-{hip_noise}_knee-{knee_noise}_ankle-{ankle_noise}.zarr'
    save_path = Path(args_cli.save_folder) if args_cli.save_folder else Path.cwd()
    save_path.mkdir(parents=True, exist_ok=True)
    SAVE_FILE_NAME = str(save_path / base_filename)
    
    print(f"[INFO] Initializing ReplayBuffer at: {SAVE_FILE_NAME}")
    buff = ReplayBuffer.create_empty_zarr(
        storage=zarr.DirectoryStore(SAVE_FILE_NAME)
    )

    metadata = {
        'wandb_run': wandb_run.name,
        'num_episodes': NUM_EPISODE,
        'collect_steps': COLLECT_STEPS,
        'decimation': env_cfg.decimation,
        'min_delay': args_cli.min_delay,
        'max_delay': args_cli.max_delay,
        'noise_level': noise_level,
        'hip_noise': hip_noise,
        'knee_noise': knee_noise,
        'ankle_noise': ankle_noise,
        'timestamp': timestamp,
    }
    buff.update_meta(metadata)
    
    num_bodies = 30
    num_joints = 29
    

    num_envs = env.unwrapped.num_envs # type: ignore
    recorded_obs_episode = np.zeros((num_envs, 2000, env.unwrapped.observation_space['diffusion_collect'].shape[-1])) # type: ignore
    recorded_acs_episode = np.zeros((num_envs, 2000, env.unwrapped.action_space.shape[-1])) # type: ignore
    
    # Initialize image buffers (assuming 480x848 resolution from config)
    # RGB: (num_envs, max_steps, H, W, 3), Depth: (num_envs, max_steps, H, W)
    img_h, img_w = 480, 848
    # recorded_rgb_episode = np.zeros((num_envs, 2000, img_h, img_w, 3), dtype=np.uint8)
    # recorded_depth_episode = np.zeros((num_envs, 2000, img_h, img_w), dtype=np.float32)

    # Initialize Siglip2 Model (RGB encoder). Skipped entirely with --disable_rgb so the
    # dataset stores depth-only embeddings (matches sim2sim --disable_rgb + existing data).
    collect_rgb = not args_cli.disable_rgb
    processor = None
    siglip_model = None
    recorded_rgb_embed_episode = None
    recorded_rgb_embed_flipped_episode = None
    if collect_rgb:
        print("[INFO] Initializing Siglip2 Vision Model...")
        model_id = "google/siglip2-so400m-patch14-384"
        # model_id = "google/siglip2-base-patch16-384"

        # Use AutoImageProcessor to avoid tokenizer issues
        processor = AutoImageProcessor.from_pretrained(model_id)
        # Try Flash Attention 2 first, then fall back if the package is unavailable.
        try:
            siglip_model = SiglipModel.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                dtype=torch.float16,
            )
        except (ImportError, ValueError):
            print("[WARN] flash_attention_2 not available, using default attention")
            siglip_model = SiglipModel.from_pretrained(
                model_id,
                dtype=torch.float16,
            )
        siglip_model.to(device=args_cli.device).eval()

        # Siglip2 Embeddings Storage
        # Get hidden size from the vision component properly
        rgb_embed_dim = siglip_model.vision_model.config.hidden_size
        recorded_rgb_embed_episode = np.zeros((num_envs, 2000, rgb_embed_dim), dtype=np.float32)
        recorded_rgb_embed_flipped_episode = np.zeros((num_envs, 2000, rgb_embed_dim), dtype=np.float32)
    else:
        print("[INFO] --disable_rgb set: skipping SigLIP RGB encoder (depth-only collection).")

    # DeFM Embeddings Storage
    # Initialize DeFM Model
    print("[INFO] Initializing DeFM Depth Model...")
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    defm_model = torch.hub.load(
        "leggedrobotics/defm:main",
        "defm_vit_l14",
        pretrained=True,
        trust_repo=True,
    )
    defm_model = defm_model.eval().to(device=args_cli.device)
    depth_embed_dim = 1024 # DeFM ViT-L14 class token size
    recorded_depth_embed_episode = np.zeros((num_envs, 2000, depth_embed_dim), dtype=np.float32)
    recorded_depth_embed_flipped_episode = np.zeros((num_envs, 2000, depth_embed_dim), dtype=np.float32)
    raw_depth_dump_dir = None
    dumped_raw_depth_count = 0
    if args_cli.dump_raw_depth_dir is not None:
        raw_depth_dump_dir = Path(args_cli.dump_raw_depth_dir)
        raw_depth_dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Dumping up to {args_cli.dump_raw_depth_limit} raw depth frames to: {raw_depth_dump_dir}")

    # OU parameters
    theta = .8 # 0 #0.4  # mean reversion rate
    mu = 0.0      # long-term mean
    dt = 1.0      # time step
    sqrt_dt = torch.sqrt(torch.tensor(dt))  # compute once for efficiency
    

    device = env.unwrapped.device # type: ignore

    saved_idx = 0
    saved_epi = 0
    step = 0

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # import ipdb; ipdb.set_trace()
    # reset environment
    obs, _ = env.reset()

    # obs = obs['diffusion_collect']
    # obs_diffusion = env.unwrapped.observations['diffusion_collect']
    obs_diffusion = obs['diffusion_collect']

    timestep = 0
    # simulate environment

    # noise_level = .05
    # hip_noise = .2 # *0
    # knee_noise = .3 # *0
    # ankle_noise =.4
    
    # noise_level = .8
    # hip_noise =   .8 # *0
    # knee_noise =  .8 # *0
    # ankle_noise = .8
    
    # noise_level = .5
    # hip_noise = .5 # *0
    # knee_noise = .5 # *0
    # ankle_noise =.7

    # print('total tuples collected', COLLECT_STEPS*NUM_EPISODE)
    # print('baseline:', 100*20/.02) # 100k 

    print('total tuples collected', COLLECT_STEPS*NUM_EPISODE)
    print('baseline:', 100*20/.02) # 100k 


    num_actions = env.unwrapped.action_space.shape[-1]
    noise_state = torch.zeros((num_envs, num_actions), device=device)

    num_actions = env.unwrapped.action_space.shape[-1]
    noise_state = torch.zeros((num_envs, num_actions), device=device)

    # OU parameters
    theta = .8 # 0 #0.4  # mean reversion rate
    mu = 0.0      # long-term mean
    dt = 1.0      # time step
    sqrt_dt = torch.sqrt(torch.tensor(dt))  # compute once for efficiency
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # import ipdb; ipdb.set_trace()
            actions = policy(obs)
            # actions = policy(env.get_observations()['policy'])
            clean_actions = actions.clone()
            
            # Generate random noise
            random_noise = torch.randn_like(noise_state)
            
            # Apply base sigma to all joints, then override only ankle joints
            # sigma_noise = 0.2 * random_noise  # base sigma = 0.5 for all joints (hip, knee, others)
            # sigma_noise[:, hip_idxs] = 0.3 * random_noise[:, hip_idxs]      # hip sigma = 0.5 (explicit)
            # sigma_noise[:, knee_idxs] = 0.3 * random_noise[:, knee_idxs]    # knee sigma = 0.5 (explicit)  
            # sigma_noise[:, ankle_idxs] = 0.4 * random_noise[:, ankle_idxs]  # ankle sigma = 0.7 (higher)
            # import ipdb; ipdb.set_trace() 
            

            sigma_noise = torch.normal(0, noise_level, size=actions.shape).to(device )
            sigma_noise[:,hip_idxs]   = torch.normal(0, hip_noise, size=actions[:,hip_idxs].shape).to(device )
            sigma_noise[:,knee_idxs]  = torch.normal(0, knee_noise, size=actions[:,knee_idxs].shape).to(device )
            sigma_noise[:,ankle_idxs] = torch.normal(0, ankle_noise, size=actions[:,ankle_idxs].shape).to(device )

            # Apply OU update: X_{t+1} = X_t + θ(μ - X_t)dt + σ√(dt)ε
            noise_state = noise_state + theta * (mu - noise_state) * dt + sigma_noise * sqrt_dt
            
            # COMMENT OUT NOISE FOR NOW TO DEBUG
            actions = actions + noise_state
            # import ipdb; ipdb.set_trace() 
            curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
            
            recorded_obs_episode[np.arange(num_envs), curr_idx, :] = (obs['diffusion_collect'].to("cpu").detach().numpy())
            recorded_acs_episode[np.arange(num_envs), curr_idx, :] = (clean_actions.to("cpu").detach().numpy())
            # TODO: JUSTIN HELP


            camera_sensor = env.unwrapped.scene.sensors["depth_camera"]
            depth_image = camera_sensor.data.output["depth"]

            # --- Process and Embed RGB Images with Siglip2 (skipped with --disable_rgb) ---
            if collect_rgb:
                rgb_image = camera_sensor.data.output["rgb"]
                # Check for alpha channel in RGB
                if rgb_image.shape[-1] == 4:
                    rgb_to_save = rgb_image[..., :3]
                else:
                    rgb_to_save = rgb_image

                # Convert tensors to list of PIL images for the processor
                # rgb_to_save: (B, H, W, 3)
                rgb_np = rgb_to_save.cpu().numpy().astype(np.uint8)
                rgb_images = [Image.fromarray(img) for img in rgb_np]
                rgb_np_flipped = np.flip(rgb_np, axis=2).copy()
                rgb_images_flipped = [Image.fromarray(img) for img in rgb_np_flipped]

            # depth_image: (B, H, W, 1) -> convert to 3 channel for processor
            depth_np = depth_image.cpu().numpy().squeeze(-1) # (B, H, W)
            depth_np_flipped = np.flip(depth_np, axis=2).copy()
            if raw_depth_dump_dir is not None and dumped_raw_depth_count < args_cli.dump_raw_depth_limit:
                dump_count = min(num_envs, args_cli.dump_raw_depth_limit - dumped_raw_depth_count)
                for env_offset in range(dump_count):
                    dump_path = raw_depth_dump_dir / f"depth_ep{int(num_epi[env_offset]):05d}_step{curr_idx:04d}_env{env_offset}.npy"
                    np.save(dump_path, depth_np[env_offset].astype(np.float32))
                    print(f"[INFO] Saved raw depth frame to {dump_path}")
                    dumped_raw_depth_count += 1
                    if dumped_raw_depth_count >= args_cli.dump_raw_depth_limit:
                        break
            # Normalize depth for visualization-like input if needed, or just replicate channels
            # Here we replicate channels to make it (H, W, 3) grayscale-like
            # depth_images = []


            try:
                # Batch processing
                with torch.no_grad():
                    VISION_BATCH_SIZE = 10 # Process in batches to avoid OOM
                    
                    # --- Process RGB in batches (skipped with --disable_rgb) ---
                    if collect_rgb:
                        rgb_embeds_list = []
                        rgb_embeds_flipped_list = []
                        for i in range(0, len(rgb_images), VISION_BATCH_SIZE):
                            batch_imgs = rgb_images[i : i + VISION_BATCH_SIZE]
                            inputs_rgb = processor(images=batch_imgs, return_tensors="pt").to(device)
                            inputs_rgb["pixel_values"] = inputs_rgb["pixel_values"].to(dtype=torch.float16)

                            outputs_rgb = siglip_model.vision_model(**inputs_rgb)
                            rgb_embeds_list.append(outputs_rgb.pooler_output.cpu()) # Move to CPU immediately

                            batch_imgs_flipped = rgb_images_flipped[i : i + VISION_BATCH_SIZE]
                            inputs_rgb_flipped = processor(images=batch_imgs_flipped, return_tensors="pt").to(device)
                            inputs_rgb_flipped["pixel_values"] = inputs_rgb_flipped["pixel_values"].to(dtype=torch.float16)

                            outputs_rgb_flipped = siglip_model.vision_model(**inputs_rgb_flipped)
                            rgb_embeds_flipped_list.append(outputs_rgb_flipped.pooler_output.cpu())

                            # Cleanup
                            del inputs_rgb, outputs_rgb, inputs_rgb_flipped, outputs_rgb_flipped
                            # torch.cuda.empty_cache() # Optional: helps if fragmentation is high

                        rgb_embeds = torch.cat(rgb_embeds_list, dim=0) # Concatenate on CPU, then move to GPU if needed or keep on CPU
                        rgb_embeds_flipped = torch.cat(rgb_embeds_flipped_list, dim=0)

                    # For saving, we want numpy anyway, so keeping on CPU is perfect.
                    # But the code below expects `rgb_embeds` to have a .cpu() method or be a tensor.
                    # rgb_embeds_list contains cpu tensors now.
                    # rgb_embeds will be a CPU tensor.

                    # --- Process Depth in batches ---
                    # depth_image shape is (B, H, W, 1) -> squeeze to (B, H, W)
                    batch_depth = depth_image.squeeze(-1).float() 
                    batch_depth_np = batch_depth.cpu().numpy()
                    
                    depth_embeds_list = []
                    depth_embeds_flipped_list = []
                    for i in range(0, len(batch_depth_np), VISION_BATCH_SIZE):
                        # 1. Get chunk of raw depth (numpy, CPU)
                        batch_depth_chunk_np = batch_depth_np[i : i + VISION_BATCH_SIZE]
                        batch_depth_chunk_flipped_np = depth_np_flipped[i : i + VISION_BATCH_SIZE]
                        
                        # 2. Preprocess just this chunk (moves to GPU inside function)
                        normalized_depth_chunk = preprocess_depth_batch(
                            batch_depth_chunk_np,
                            target_size=518, 
                            patch_size=14,
                            device=device
                        )
                        normalized_depth_chunk = normalized_depth_chunk.float()
                        normalized_depth_chunk_flipped = preprocess_depth_batch(
                            batch_depth_chunk_flipped_np,
                            target_size=518,
                            patch_size=14,
                            device=device
                        )
                        normalized_depth_chunk_flipped = normalized_depth_chunk_flipped.float()

                        # 3. Run Inference
                        output = defm_model.get_intermediate_layers(
                            normalized_depth_chunk, n=1, reshape=True, return_class_token=True
                        )
                        depth_embeds_list.append(output[0][1].cpu()) # Move result to CPU immediately
                        output_flipped = defm_model.get_intermediate_layers(
                            normalized_depth_chunk_flipped, n=1, reshape=True, return_class_token=True
                        )
                        depth_embeds_flipped_list.append(output_flipped[0][1].cpu())
                        
                        # 4. Cleanup GPU tensors for this chunk
                        del normalized_depth_chunk, output, normalized_depth_chunk_flipped, output_flipped
                        
                    depth_embeds = torch.cat(depth_embeds_list, dim=0)
                    depth_embeds_flipped = torch.cat(depth_embeds_flipped_list, dim=0)
                
                # Arrays are already on CPU if we used .cpu() above, but .cpu() is safe to call on CPU tensors too.
                if collect_rgb:
                    recorded_rgb_embed_episode[np.arange(num_envs), curr_idx] = rgb_embeds.cpu().numpy()
                    recorded_rgb_embed_flipped_episode[np.arange(num_envs), curr_idx] = rgb_embeds_flipped.cpu().numpy()
                recorded_depth_embed_episode[np.arange(num_envs), curr_idx] = depth_embeds.cpu().numpy()
                recorded_depth_embed_flipped_episode[np.arange(num_envs), curr_idx] = depth_embeds_flipped.cpu().numpy()
            except Exception as e:
                print(f"Error in Siglip2 embedding: {e}")
                traceback.print_exc()

            # ---------------------------------------------


            # # Visualize the first environment's image every step (as requested)
            # if step % 1 == 0:
            #     try:
            #         # Get image from first env, remove batch dim
            #         img_tensor = rgb_image[0] 
            #         # Convert to numpy, ensure it's on CPU
            #         img_np = img_tensor.cpu().numpy()
                    
            #         # Handle alpha channel if present (RGBA -> RGB)
            #         if img_np.shape[-1] == 4:
            #             img_np = img_np[:, :, :3]
                    
            #         # Use matplotlib for visualization (robust to headless OpenCV)
            #         plt.imshow(img_np)
            #         plt.axis('off')
            #         plt.title("Camera View")
            #         plt.pause(0.001)
            #         plt.clf()
            #     except Exception as e:
            #         # If even matplotlib fails (e.g. completely headless), catch and print once
            #         if step == 0:
            #             print(f"Error visualizing image with matplotlib: {e}") 


            # get image from env
            # pass through encoder
            # recorded_img_episode[np.arange(num_envs), curr_idx, :] = encoded_image
            # import ipdb; ipdb.set_trace()
            # env stepping
            obs, rews, dones, extra = env.step(actions)
            
            # Reconstruct what the wrapper would have returned
            # dones = (terminated | truncated).to(dtype=torch.long)
            
            # Explicitly add time_outs to extra
            # extra["time_outs"] = truncated
            
            # obs = TensorDict(obs_dict, batch_size=[env.num_envs])
            
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            
            step += 1

            if step % 10 == 0:
                print("[DEBUG] sim step", step, "saved_epi", saved_epi, flush=True)

            if len(done_indices) > 0:
                env_ids = done_indices.to("cpu").detach().numpy()
                # import ipdb; ipdb.set_trace()

                for i in range(len(env_ids)):
                    # check if successful      
                    
                    if extra['time_outs'][env_ids[i]]:
                        epi_len = (np.all(recorded_obs_episode[env_ids[i]] == 0, axis=-1).argmax(axis=-1)) #if COLLECT_STEPS is None else COLLECT_STEPS

                        # import ipdb; ipdb.set_trace() 

                        if epi_len >= COLLECT_STEPS:

                            epi_len = min(epi_len, COLLECT_STEPS)

                            if np.any(np.isnan( recorded_obs_episode[env_ids[i], :epi_len])):
                                import ipdb; ipdb.set_trace() 

                            if np.any(np.all(recorded_obs_episode[env_ids[i], :epi_len] == 0, axis=1)):
                                print(f"Warning: All-zero rows found in obs_slice for env {env_ids[i]}")
                                import ipdb; ipdb.set_trace() 
                            

                            # Extract data for this episode
                            ep_obs = np.copy(recorded_obs_episode[env_ids[i], :epi_len])
                            ep_acs = np.copy(recorded_acs_episode[env_ids[i], :epi_len])
                            ep_depth_embed = np.copy(recorded_depth_embed_episode[env_ids[i], :epi_len])
                            ep_depth_embed_flipped = np.copy(recorded_depth_embed_flipped_episode[env_ids[i], :epi_len])
                            # Save to Zarr immediately
                            print(f"[INFO] Saving episode for env {env_ids[i]} to ReplayBuffer...")
                            episode_data = {
                                "body_pos": ep_obs[:,: num_bodies * 3],
                                "body_rot": ep_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4],
                                "body_lin_vel": ep_obs[:, num_bodies * 7 : num_bodies * 10],
                                "body_ang_vel": ep_obs[:, num_bodies * 10 : num_bodies * 13],
                                "joint_pos": ep_obs[:, num_bodies * 13 : num_bodies * 13 + num_joints],
                                "joint_vel": ep_obs[:,num_bodies * 13 + num_joints : num_bodies * 13 + num_joints * 2,],
                                "root_pos": (ep_obs[:, : num_bodies * 3].reshape(-1, num_bodies, 3)[:, 0, :].reshape(-1, 3)),
                                "root_rot": (ep_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4].reshape(-1, num_bodies, 4)[:, 0, :].reshape(-1, 4)),
                                "act": ep_acs[:],
                                "depth_embed": ep_depth_embed,
                                "depth_embed_flipped": ep_depth_embed_flipped,
                            }
                            # Only store RGB embeddings when the SigLIP encoder is enabled.
                            if collect_rgb:
                                episode_data["rgb_embed"] = np.copy(recorded_rgb_embed_episode[env_ids[i], :epi_len])
                                episode_data["rgb_embed_flipped"] = np.copy(recorded_rgb_embed_flipped_episode[env_ids[i], :epi_len])
                            buff.add_episode(episode_data)

                            saved_idx += epi_len
                            saved_epi += 1
                            episode_ends.append(saved_idx)

                            print("SAVED: ", env_ids[i], "LEN: ", epi_len, "EPISODES: ",saved_epi,)
                        else:
                            print("SKIP DONE: ", env_ids[i], "DUE TO NOT LONG ENOUGH EPISODE", epi_len)
                            
                    else:
                        print("SKIP DONE: ", env_ids[i], "DUE TO BAD REWARD")
                    
                    recorded_obs_episode[env_ids[i]] = 0
                    recorded_acs_episode[env_ids[i]] = 0
                    if collect_rgb:
                        recorded_rgb_embed_episode[env_ids[i]] = 0
                        recorded_rgb_embed_flipped_episode[env_ids[i]] = 0
                    recorded_depth_embed_episode[env_ids[i]] = 0
                    recorded_depth_embed_flipped_episode[env_ids[i]] = 0

                    if saved_epi >= NUM_EPISODE:
                        print(f"Collected {saved_epi} episodes. Usage limit reached.")
                        print('saved to:', SAVE_FILE_NAME)
                        env.close()
                        simulation_app.close()
                        exit()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
