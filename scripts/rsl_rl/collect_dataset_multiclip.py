"""Multiclip variant of collect_dataset.py.

Collects rollouts from a multiclip policy checkpoint using a Zarr motion store
(no wandb motion artifact). Expects a Collect-capable multiclip task that
exposes the `diffusion_collect` obs group and a `depth_camera` scene sensor.

TODO: register e.g. `Bones-MultiClip-Compliance-G1-Collect-v0` (extends
G1BonesMultiClipComplianceEnvCfg, re-enables diffusion_collect, adds depth_camera).
Otherwise obs['diffusion_collect'] and scene.sensors['depth_camera'] below will fail.

example:
  python scripts/rsl_rl/collect_dataset_multiclip.py \
      --task=Bones-MultiClip-Compliance-G1-Collect-v0 \
      --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
      --include_motion_types walk,jog \
      --wandb_path=robot-mcrobotface/multiclip_bones/c15qko8c \
      --activation swish \
      --num_envs 750 --num_steps_collect 60 --num_eps_collect 500 \
      --episode_collect_length 4 --headless
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

parser.add_argument("--min_delay", type=int, default=0, help="actuator delay.")
parser.add_argument("--max_delay", type=int, default=0, help="actuator delay.")
parser.add_argument("--num_obstacles", type=int, default=0, help="Number of obstacles to generate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the experiment.")

def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

parser.add_argument("--min_sample_idx", type=none_or_int, default=None, help="actuator delay.")
parser.add_argument("--max_sample_idx", type=none_or_int, default=None, help="actuator delay.")

# --- Multiclip-specific ---
parser.add_argument("--zarr_path", type=str, default=None,
                    help="Path to Zarr motion store (required for multiclip tasks).")
parser.add_argument("--include_motion_types", type=str, default=None,
                    help="Comma-separated keywords restricting clips by clip_name (case-insensitive OR).")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Actor/critic activation (must match training).")
parser.add_argument("--decimation", type=int, default=None,
                    help="Override env decimation (physics steps per policy step).")

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
    agent_cfg.policy.activation = args_cli.activation

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb
        run_path = args_cli.wandb_path

        api = wandb.Api()
        if 'model' in args_cli.wandb_path:
            run_path = '/'.join(args_cli.wandb_path.split('/')[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if 'model' in file.name]
        # files are all model_xxx.pt find the largest filename
        
        if 'model' in args_cli.wandb_path:
            file = args_cli.wandb_path.split('/')[-1]
        else:
            file = max(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download(f"./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        # --- Multiclip: plumb zarr_path + include_motion_types (no wandb motion artifact) ---
        assert args_cli.zarr_path is not None, "--zarr_path is required for multiclip collect"
        assert os.path.isdir(args_cli.zarr_path), f"zarr_path does not exist: {args_cli.zarr_path}"
        env_cfg.commands.motion.zarr_path = args_cli.zarr_path
        print(f"[INFO] Zarr motion store: {args_cli.zarr_path}")

        if args_cli.include_motion_types is not None:
            kws = [s.strip() for s in args_cli.include_motion_types.split(",") if s.strip()]
            env_cfg.commands.motion.include_motion_types = kws
            print(f"[INFO] Restricting clips by motion-type keywords: {kws}")

            
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Set parameters from cli
    env_cfg.episode_length_s = float(args_cli.episode_collect_length) if args_cli.episode_collect_length is not None else float(env_cfg.episode_length_s)
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation
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
    noise_level = .3
    hip_noise = .3
    knee_noise = .3 
    ankle_noise =.5

    # noise_level = .15
    # hip_noise = .15
    # knee_noise = .15
    # ankle_noise =.2

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
    img_h, img_w = 480, 640
    # recorded_rgb_episode = np.zeros((num_envs, 2000, img_h, img_w, 3), dtype=np.uint8)
    # recorded_depth_episode = np.zeros((num_envs, 2000, img_h, img_w), dtype=np.float32)

    # Initialize Siglip2 Model
    print("[INFO] Initializing Siglip2 Vision Model...")
    # model_id = "google/siglip2-so400m-patch14-384" 
    model_id = "google/siglip2-base-patch16-384" 
    
    # Use AutoImageProcessor to avoid tokenizer issues
    processor = AutoImageProcessor.from_pretrained(model_id)
    # Use SiglipModel (explicit) with Flash Attention 2 and Float16
    siglip_model = SiglipModel.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    siglip_model.to(device=args_cli.device).eval()
    
    # Siglip2 Embeddings Storage
    # Get hidden size from the vision component properly
    rgb_embed_dim = siglip_model.vision_model.config.hidden_size
    recorded_rgb_embed_episode = np.zeros((num_envs, 2000, rgb_embed_dim), dtype=np.float32)

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
            rgb_image = camera_sensor.data.output["rgb"] 
            depth_image = camera_sensor.data.output["depth"] 

            # Save images to buffer
            # Check for alpha channel in RGB
            if rgb_image.shape[-1] == 4:
                rgb_to_save = rgb_image[..., :3]
            else:
                rgb_to_save = rgb_image

            # recorded_rgb_episode[np.arange(num_envs), curr_idx] = rgb_to_save.cpu().numpy().astype(np.uint8)
            # recorded_depth_episode[np.arange(num_envs), curr_idx] = depth_image.squeeze(-1).cpu().numpy().astype(np.float32)

            # --- Process and Embed Images with Siglip2 ---
            
            # Convert tensors to list of PIL images for the processor
            # rgb_to_save: (B, H, W, 3) 
            rgb_np = rgb_to_save.cpu().numpy().astype(np.uint8)
            rgb_images = [Image.fromarray(img) for img in rgb_np]

            # depth_image: (B, H, W, 1) -> convert to 3 channel for processor
            depth_np = depth_image.cpu().numpy().squeeze(-1) # (B, H, W)
            # Normalize depth for visualization-like input if needed, or just replicate channels
            # Here we replicate channels to make it (H, W, 3) grayscale-like
            # depth_images = []


            try:
                # Batch processing
                with torch.no_grad():
                    VISION_BATCH_SIZE = 10 # Process in batches to avoid OOM
                    
                    # --- Process RGB in batches ---
                    rgb_embeds_list = []
                    for i in range(0, len(rgb_images), VISION_BATCH_SIZE):
                        batch_imgs = rgb_images[i : i + VISION_BATCH_SIZE]
                        inputs_rgb = processor(images=batch_imgs, return_tensors="pt").to(device)
                        inputs_rgb["pixel_values"] = inputs_rgb["pixel_values"].to(dtype=torch.float16)
                        
                        outputs_rgb = siglip_model.vision_model(**inputs_rgb)
                        rgb_embeds_list.append(outputs_rgb.pooler_output.cpu()) # Move to CPU immediately
                        
                        # Cleanup
                        del inputs_rgb, outputs_rgb
                        # torch.cuda.empty_cache() # Optional: helps if fragmentation is high
                    
                    rgb_embeds = torch.cat(rgb_embeds_list, dim=0) # Concatenate on CPU, then move to GPU if needed or keep on CPU
                    
                    # For saving, we want numpy anyway, so keeping on CPU is perfect.
                    # But the code below expects `rgb_embeds` to have a .cpu() method or be a tensor.
                    # rgb_embeds_list contains cpu tensors now.
                    # rgb_embeds will be a CPU tensor.

                    # --- Process Depth in batches ---
                    # depth_image shape is (B, H, W, 1) -> squeeze to (B, H, W)
                    batch_depth = depth_image.squeeze(-1).float() 
                    batch_depth_np = batch_depth.cpu().numpy()
                    
                    depth_embeds_list = []
                    for i in range(0, len(batch_depth_np), VISION_BATCH_SIZE):
                        # 1. Get chunk of raw depth (numpy, CPU)
                        batch_depth_chunk_np = batch_depth_np[i : i + VISION_BATCH_SIZE]
                        
                        # 2. Preprocess just this chunk (moves to GPU inside function)
                        normalized_depth_chunk = preprocess_depth_batch(
                            batch_depth_chunk_np,
                            target_size=518, 
                            patch_size=14,
                            device=device
                        )
                        normalized_depth_chunk = normalized_depth_chunk.float()

                        # 3. Run Inference
                        output = defm_model.get_intermediate_layers(
                            normalized_depth_chunk, n=1, reshape=True, return_class_token=True
                        )
                        depth_embeds_list.append(output[0][1].cpu()) # Move result to CPU immediately
                        
                        # 4. Cleanup GPU tensors for this chunk
                        del normalized_depth_chunk, output
                        
                    depth_embeds = torch.cat(depth_embeds_list, dim=0)
                
                # Arrays are already on CPU if we used .cpu() above, but .cpu() is safe to call on CPU tensors too.
                recorded_rgb_embed_episode[np.arange(num_envs), curr_idx] = rgb_embeds.cpu().numpy()
                recorded_depth_embed_episode[np.arange(num_envs), curr_idx] = depth_embeds.cpu().numpy()
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
                            ep_rgb_embed = np.copy(recorded_rgb_embed_episode[env_ids[i], :epi_len])
                            ep_depth_embed = np.copy(recorded_depth_embed_episode[env_ids[i], :epi_len])

                            # Save to Zarr immediately
                            print(f"[INFO] Saving episode for env {env_ids[i]} to ReplayBuffer...")
                            buff.add_episode({
                                "body_pos": ep_obs[:,: num_bodies * 3],
                                "body_rot": ep_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4],
                                "body_lin_vel": ep_obs[:, num_bodies * 7 : num_bodies * 10],
                                "body_ang_vel": ep_obs[:, num_bodies * 10 : num_bodies * 13],
                                "joint_pos": ep_obs[:, num_bodies * 13 : num_bodies * 13 + num_joints],
                                "joint_vel": ep_obs[:,num_bodies * 13 + num_joints : num_bodies * 13 + num_joints * 2,],
                                "root_pos": (ep_obs[:, : num_bodies * 3].reshape(-1, num_bodies, 3)[:, 0, :].reshape(-1, 3)),
                                "root_rot": (ep_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4].reshape(-1, num_bodies, 4)[:, 0, :].reshape(-1, 4)),
                                "act": ep_acs[:],
                                "rgb_embed": ep_rgb_embed,
                                "depth_embed": ep_depth_embed,
                            })

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
                    recorded_rgb_embed_episode[env_ids[i]] = 0
                    recorded_depth_embed_episode[env_ids[i]] = 0

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
