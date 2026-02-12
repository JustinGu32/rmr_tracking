import argparse
import subprocess
import os
import torch
import onnx
import collections
import numpy as np
import wandb
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play diffusion policy in Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Tracking-Flat-G1-v0', help="Name of the task.")
# parser.add_argument("--motion_file", type=str, default='/home/takaraet/Downloads/cartwheel.npz', help="Motion file path.")
parser.add_argument("--motion_file", type=str, default='/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz', help="Motion file.")

parser.add_argument("--checkpoint_tag", type=str, default='latest.ckpt', help="Checkpoint filename.")
# parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")

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
import hydra
from omegaconf import OmegaConf

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.math import matrix_from_quat

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401



def load_diffusion_policy(checkpoint_path):
    """Load diffusion policy from checkpoint.
    
    Only works with NEW refactored checkpoints.
    """
    print(f"[INFO]: Loading checkpoint from: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if payload has cfg
    if 'cfg' not in payload:
        raise ValueError(
            "\n" + "="*80 + "\n"
            "ERROR: Invalid checkpoint format - missing 'cfg' key\n"
            "="*80
        )
    
    cfg = payload['cfg']
    
    # Check if this is a NEW refactored checkpoint
    try:
        # Try to access cfg.model - will fail if old checkpoint
        _ = cfg.model
        print("[INFO]: Detected NEW refactored checkpoint ✓")
    except (AttributeError, KeyError):
        # This is an old checkpoint
        raise ValueError(
            "\n" + "="*80 + "\n"
            "ERROR: This checkpoint is from the OLD codebase (before refactoring)\n\n"
            "The refactored code has a completely different architecture and\n"
            "cannot load old checkpoints.\n\n"
            "SOLUTION:\n"
            "  1. Train a new checkpoint with the refactored code:\n"
            "     python train_simplified.py\n\n"
            "  2. Use the NEW wandb run ID to load the checkpoint:\n"
            "     python play_sequential_new.py --wandb_path entity/project/NEW_RUN_ID\n\n"
            "OLD checkpoints have: cfg.policy.actor (nested structure)\n"
            "NEW checkpoints have: cfg.model (flat DiffusionActor)\n"
            "="*80
        )
    
    # Load model from new checkpoint
    model = hydra.utils.instantiate(cfg.model)
    
    # Load weights (handle both 'model' and 'ema_model' keys)
    # Use strict=False to allow missing/unexpected keys (buffers may differ)
    if 'ema_model' in payload and cfg.training.use_ema:
        print("[INFO]: Loading EMA model weights")
        model.load_state_dict(payload['ema_model'], strict=False)
    else:
        model.load_state_dict(payload['model'], strict=False)
    
    # Load normalizer from checkpoint (NO DATASET NEEDED!)
    from diffusion_policy.utils.normalizer import LinearNormalizer  # FIXED: correct import path
    
    if 'normalizer' in payload:
        print("[INFO]: Loading normalizer from checkpoint ✓")
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(payload['normalizer'])
    else:
        print("[WARNING]: Normalizer not found in checkpoint - using identity normalizer")
        print("[WARNING]: You may need to retrain to get a checkpoint with normalizer")
        normalizer = LinearNormalizer()
    
    # Get dataset class for the transformation functions (state_normalize, ee_idxs, etc.)
    model.normalizer = normalizer
    model.dataset_class = hydra.utils.get_class(cfg.dataset._target_)
    
    model.eval()

    model.to('cuda')
    model.compile_for_inference()

    print("Warming up model...")

    B = 1
    device = 'cuda'
    H = model.n_past_steps
    J = 30  # Number of bodies for G1 humanoid
    num_joints = 29  # Number of joints for G1

    dummy_obs_dict = {
        'body_pos': torch.randn(B, H, J, 3, device=device),
        'body_rot': torch.randn(B, H, J, 4, device=device),
        'body_lin_vel': torch.randn(B, H, J, 3, device=device),
        'body_ang_vel': torch.randn(B, H, J, 3, device=device),
        'joint_pos': torch.randn(B, H, num_joints, device=device),
        'joint_vel': torch.randn(B, H, num_joints, device=device),
    }

    nominal_frame_idx = model.n_past_steps - 1

    with torch.no_grad():
        for _ in range(3):
            model.act(obs_dict=dummy_obs_dict, nom_frame_idx=nominal_frame_idx)

            
    return model, cfg


def load_wandb(wandb_path, checkpoint_file="latest.ckpt"):
    """Load checkpoint from wandb."""
    
    run_path = wandb_path
    
    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    
    wandb_run = api.run(run_path)
    
    print(f"[INFO]: Downloading WandB Checkpoint from {run_path}")
    
    wandb_file = wandb_run.file(str('checkpoints/' + checkpoint_file))
    wandb_file.download(f"./logs/rsl_rl/temp", replace=True)
    
    print(f"[INFO]: Loading model checkpoint from: {run_path}/checkpoints/{checkpoint_file}")
    resume_path = f"./logs/rsl_rl/temp/checkpoints/{checkpoint_file}"
    
    # Load policy
    policy, cfg = load_diffusion_policy(resume_path)
    dataset_class = policy.dataset_class
    
    policy = policy.to('cuda')
    # import ipdb; ipdb.set_trace() 

    # Check if model parameters contain NaN
    for name, param in policy.named_parameters():
        if torch.isnan(param).any():
            print(f"[WARNING] NaN found in parameter: {name}")
    
    return policy, cfg, dataset_class, resume_path, wandb_run.name


def global_to_characterFrame(obs, dataset_class, nom_frame_idx):
    """Convert global observations to character frame."""
    B, H = obs.shape[:2]
    J = 30
    
    body_pos = obs[:, :, : J * 3].view(B, H, J, 3).clone()
    body_rot = obs[:, :, J * 3 : J * 7].view(B, H, J, 4).clone()
    root_pos = body_pos[:, :, 0, :].clone()
    root_rot = body_rot[:, :, 0, :].clone()
    
    body_lin_vel = obs[:, :, J * 7 : J * 10].view(B, H, J, 3).clone()
    body_ang_vel = obs[:, :, J * 10 : J * 13].view(B, H, J, 3).clone()
    joint_pos = obs[:, :, J * 13 : J * 14 - 1].view(B, H, -1).clone()
    joint_vel = obs[:, :, J * 14 - 1 : J * 15 - 2].view(B, H, -1).clone()
    
    handsfeet_idx = dataset_class.ee_idxs()
    
    diff_obs = dataset_class.state_normalize(
        root_pos, root_rot, body_pos, body_rot,
        body_lin_vel, body_ang_vel, nom_frame_idx,
        handsfeet_idx, joint_pos, joint_vel
    )
    diff_state = joint_pos.reshape(B, H, -1)
    return diff_obs, diff_state


def main():
    """Play with diffusion policy agent."""
    
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)

    # Load motion file
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Load policy - either from wandb or local
    if args_cli.wandb_path:
        # Load from wandb
        policy, cfg, dataset_class, resume_path, run_name = load_wandb(
            args_cli.wandb_path, 
            args_cli.checkpoint_tag
        )
        print(f"[INFO]: Loaded wandb run: {run_name}")
    else:
        # Load from local path
        # print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        
        policy, cfg = load_diffusion_policy(args_cli.checkpoint)
        dataset_class = policy.dataset_class
        policy = policy.to('cuda')

    # Set actuator delays to zero
    for actuator_name in ['legs', 'feet', 'waist', 'waist_yaw', 'arms']:
        if actuator_name in env_cfg.scene.robot.actuators:
            env_cfg.scene.robot.actuators[actuator_name].min_delay = 0
            env_cfg.scene.robot.actuators[actuator_name].max_delay = 0

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapper(env)

    # Reset environment
    obs, extra = env.get_observations()

    # Initialize observation and action deques
    obs_deque = collections.deque(
        [extra["observations"]["obs_all"].clone()] * policy.n_past_steps,
        maxlen=policy.n_past_steps,
    )
    action_deque = collections.deque(
        [torch.zeros(1, env.unwrapped.action_space.shape[-1], device=env.unwrapped.device)] 
        * (policy.n_past_steps - 1),
        maxlen=policy.n_past_steps - 1,
    )

    nominal_frame_idx = policy.n_past_steps - 1
    print(f"[INFO] Nominal frame index: {nominal_frame_idx}")
    
    timestep = 0
    past_actions = None

    from diffusion_policy.inference.guidance import GuidanceFunctions
    # waypoints_tensor = torch.zeros(1, 3, device=env.unwrapped.device)  # Dummy waypoints
    # target_velocity = torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
    target_velocity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=env.unwrapped.device)

    # Simulation loop
    while simulation_app.is_running():
        with torch.no_grad():
            # Update observation deque
            obs_deque.append(extra["observations"]["obs_all"].clone())
            
            global_obs = torch.stack(list(obs_deque)).permute((1, 0, 2))
            
            # Convert to character frame
            diff_obs, diff_state = global_to_characterFrame(global_obs, dataset_class, nominal_frame_idx)
            
            # Prepare past actions
            if nominal_frame_idx > 0:
                past_actions = torch.stack(list(action_deque)).permute((1, 0, 2))
            if past_actions is None:
                past_actions = torch.zeros(1, env.unwrapped.action_space.shape[-1], device=env.unwrapped.device)
           
            global_obs = torch.stack(list(obs_deque)).permute((1, 0, 2))

            # Parse into components
            B, H = global_obs.shape[:2]
            J = 30
            obs_dict = {
                'body_pos': global_obs[:, :, :J*3].view(B, H, J, 3),
                'body_rot': global_obs[:, :, J*3:J*7].view(B, H, J, 4),
                'body_lin_vel': global_obs[:, :, J*7:J*10].view(B, H, J, 3),
                'body_ang_vel': global_obs[:, :, J*10:J*13].view(B, H, J, 3),
                'joint_pos': global_obs[:, :, J*13:J*14-1].view(B, H, -1),
                'joint_vel': global_obs[:, :, J*14-1:J*15-2].view(B, H, -1),
            }
            # import ipdb; ipdb.set_trace() 
            
            import time 
            # torch.cuda.synchronize()
            start_time = time.time()

            # Call act() - character frame conversion + normalization happen inside!
            policy.guidance_mode = 'epsilon'
            action_traj, state_traj = policy.act(
                obs_dict=obs_dict,
                nom_frame_idx=nominal_frame_idx,
                guidance_fn=GuidanceFunctions.velocity_constraint,
                guidance_kwargs={'target_velocity': target_velocity},
                guidance_scale=.1
            )

            # torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"[INFO] Inference time: {elapsed*1000:.2f} ms")

            # Extract action at nominal frame
            if nominal_frame_idx > 0:
                action = action_traj[:, nominal_frame_idx, :]
                action_deque.append(action.clone())
            else:
                action = action_traj[:, nominal_frame_idx, :]
                past_actions = action.clone()
        

        # Step environment
        action_denormalized = policy.normalizer.unnormalize({'action': action})['action']

        # import ipdb; ipdb.set_trace()     

        obs, r, dones, extra = env.step(action_denormalized)

        # Handle episode terminations
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            print(f"[INFO] Episode done at indices: {done_indices}")
            
            # Allow loading new checkpoint (wandb or local)
            response = input("\n[PROMPT] Enter wandb_path and checkpoint_file (or just press Enter to continue): ")
            
            if response.strip():
                while True:
                    try:
                        # Parse input - expect format: "bay-research/diffuse_cloc/381msv4x latest.ckpt"
                        parts = response.strip().split(maxsplit=1)
                        wandb_path = parts[0]
                        checkpoint_file = parts[1] if len(parts) > 1 else "latest.ckpt"
                        
                        print(f"[INFO] Loading {wandb_path} {checkpoint_file}")
                        policy, cfg, dataset_class, resume_path, run_name = load_wandb(
                            wandb_path, 
                            checkpoint_file
                        )
                        print(f"[INFO] Loaded! Run: {run_name}")
      
                        break  # Success - exit the loop
                    except Exception as e:
                        print(f"[ERROR] Failed to load: {e}")
                        response = input("[PROMPT] Enter 'wandb_path checkpoint_file' or press Enter to skip: ")
                        if not response.strip():
                            break  # User wants to skip
            
            # Reset observation and action deques
            obs_deque = collections.deque(
                [extra["observations"]["obs_all"].clone()] * policy.n_past_steps,
                maxlen=policy.n_past_steps,
            )
            action_deque = collections.deque(
                [torch.zeros(1, env.unwrapped.action_space.shape[-1], device=env.unwrapped.device)]
                * (policy.n_past_steps - 1),
                maxlen=policy.n_past_steps - 1,
            )
            nominal_frame_idx = policy.n_past_steps - 1
        
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()