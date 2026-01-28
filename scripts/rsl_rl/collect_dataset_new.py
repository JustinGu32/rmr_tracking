"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from tensordict import TensorDict

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import pathlib 
from pathlib import Path

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
import whole_body_tracking.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration


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
    
    # env_cfg 
    recorded_obs = []
    recorded_acs = []
    episode_ends = []

    num_envs = env.unwrapped.num_envs # type: ignore
    recorded_obs_episode = np.zeros((num_envs, 2000, env.unwrapped.observation_space['diffusion_collect'].shape[-1])) # type: ignore
    recorded_acs_episode = np.zeros((num_envs, 2000, env.unwrapped.action_space.shape[-1])) # type: ignore
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
    
    noise_level = .5
    hip_noise = .5 # *0
    knee_noise = .5 # *0
    ankle_noise =.7

    hip_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if 'hip' in name ]
    knee_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if 'knee_joint' in name ]
    ankle_idxs = [i for i, name in enumerate(env.unwrapped.command_manager.get_term('motion').robot.joint_names) if  'ankle' in name]


    COLLECT_STEPS = int(args_cli.num_steps_collect)
    NUM_EPISODE = int(args_cli.num_eps_collect)
    from datetime import datetime

    # Day, hours, and minutes
    timestamp = datetime.now().strftime("%d_%H%M")

    base_filename = f'{wandb_run.name}_ep-{NUM_EPISODE}_steps-{COLLECT_STEPS}_delay-{args_cli.min_delay}-{args_cli.max_delay}_noise-{noise_level}_hip-{hip_noise}_knee-{knee_noise}_ankle-{ankle_noise}_{timestamp}.zarr'
    
    # base_filename = f'{wandb_run.name}_ep-{NUM_EPISODE}_steps-{COLLECT_STEPS}_delay-{args_cli.min_delay}-{args_cli.max_delay}_noise-{noise_level}_hip-{hip_noise}_knee-{knee_noise}_ankle-{ankle_roll_noise}.zarr'
    save_path = Path(args_cli.save_folder) if args_cli.save_folder else Path.cwd()
    save_path.mkdir(parents=True, exist_ok=True)
    SAVE_FILE_NAME = str(save_path / base_filename)

    print('total tuples collected', COLLECT_STEPS*NUM_EPISODE)
    print('baseline:', 100*20/.02) # 100k 


    num_actions = env.unwrapped.action_space.shape[-1]
    noise_state = torch.zeros((num_envs, num_actions), device=device)

    num_actions = env.unwrapped.action_space.shape[-1]
    noise_state = torch.zeros((num_envs, num_actions), device=device)

    # OU parameters
    theta = 0 #0.4  # mean reversion rate
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
            actions = actions #+ noise_state
            # import ipdb; ipdb.set_trace() 
            curr_idx = np.all(recorded_obs_episode == 0, axis=-1).argmax(axis=-1)
            
            recorded_obs_episode[np.arange(num_envs), curr_idx, :] = (obs['diffusion_collect'].to("cpu").detach().numpy())
            recorded_acs_episode[np.arange(num_envs), curr_idx, :] = (clean_actions.to("cpu").detach().numpy())
            #JUSTIN HELP
            # get image from env
            # pass through encoder
            # recorded_img_episode[np.arange(num_envs), curr_idx, :] = encoded_image


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
                            

                            recorded_obs.append(np.copy(recorded_obs_episode[env_ids[i], :epi_len]))
                            recorded_acs.append(np.copy(recorded_acs_episode[env_ids[i], :epi_len]))

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
                    
                    
                    if saved_epi > NUM_EPISODE:
                        buff = ReplayBuffer.create_empty_zarr()
                        num_bodies = 30
                        num_joints = 29
                         
                        for i in range(min(len(recorded_obs), NUM_EPISODE)):
                            buff.add_episode({
                                "body_pos": recorded_obs[i][:,: num_bodies * 3],
                                "body_rot": recorded_obs[i][:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4],
                                "body_lin_vel": recorded_obs[i][:, num_bodies * 7 : num_bodies * 10],
                                "body_ang_vel": recorded_obs[i][:, num_bodies * 10 : num_bodies * 13],
                                "joint_pos": recorded_obs[i][:, num_bodies * 13 : num_bodies * 13 + num_joints],
                                "joint_vel": recorded_obs[i][:,num_bodies * 13 + num_joints : num_bodies * 13 + num_joints * 2,],
                                "root_pos": (recorded_obs[i][:, : num_bodies * 3].reshape(-1, num_bodies, 3)[:, 0, :].reshape(-1, 3)),
                                "root_rot": (recorded_obs[i][:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4].reshape(-1, num_bodies, 4)[:, 0, :].reshape(-1, 4)),
                                "act": recorded_acs[i][:],
                                # "img": recorded_img[i][:],
                            })
                        
                        buff.save_to_path(SAVE_FILE_NAME)
                        print('saved to:', SAVE_FILE_NAME)
                        env.close()
                        simulation_app.close()
                        exit()
                        
                        # recorded_obs = np.concatenate(recorded_obs[:NUM_EPISODE])
                        # recorded_acs = np.concatenate(recorded_acs[:NUM_EPISODE])
                        # episode_ends = np.array(episode_ends[:NUM_EPISODE])

                        # num_bodies = 30
                        # num_joints = 29
                        # zdata["body_pos"] = recorded_obs[:, : num_bodies * 3]
                        # zdata["body_rot"] = recorded_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4]
                        # zdata["body_lin_vel"] = recorded_obs[:, num_bodies * 7 : num_bodies * 10]
                        # zdata["body_ang_vel"] = recorded_obs[:, num_bodies * 10 : num_bodies * 13]
                        # zdata["joint_pos"] = recorded_obs[:, num_bodies * 13 : num_bodies * 13 + num_joints]
                        # zdata["joint_vel"] = recorded_obs[:,num_bodies * 13 + num_joints : num_bodies * 13 + num_joints * 2,]
                        # zdata["root_pos"] = (recorded_obs[:, : num_bodies * 3].reshape(-1, num_bodies, 3)[:, 0, :].reshape(-1, 3))
                        # zdata["root_rot"] = (recorded_obs[:, num_bodies * 3 : num_bodies * 3 + num_bodies * 4].reshape(-1, num_bodies, 4)[:, 0, :].reshape(-1, 4))
                        # zdata["act"] = recorded_acs
                        # zmeta["episode_ends"] = episode_ends
                        
                        # print(zroot.tree())
                        # print('saved to:', SAVE_FILE_NAME)
                        # exit()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
