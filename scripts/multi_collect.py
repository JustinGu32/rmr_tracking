import os
import sys
import subprocess
from dataclasses import dataclass, replace
from typing import Optional, List
import numpy as np

@dataclass
class ExperimentConfig:
    episode_collect_length_s: float | None = None  
    num_steps_collect: int | None= None
    num_eps_collect: int = 500
    wandb_path: str | None= None
    min_delay: int = 0 
    max_delay: int = 0 
    save_folder: str | None = None
    delays: List[int] | None = None  # New field for delay values 
    min_sample_idx: int | None = None
    max_sample_idx: int | None = None
    no_action_noise: bool = False
    num_obstacles: int = 0
    disable_rgb: bool = False  # depth-only collection (skip SigLIP, no rgb_embed keys)

def run_experiment(config: ExperimentConfig, task: str = "Tracking-Flat-G1-Collect-v0", num_envs: int = 10, seed: int | None = None):
    """Run a single experiment with the given configuration."""
    
    # Construct the base command
    command = [
        sys.executable,
        "scripts/rsl_rl/collect_dataset.py",
        f"--task={task}",
        f"--num_envs={num_envs}",
        f"--wandb_path={config.wandb_path}",
        f"--num_steps_collect={config.num_steps_collect}",
        f"--num_eps_collect={config.num_eps_collect}",
        f"--episode_collect_length={config.episode_collect_length_s}",
        f"--min_delay={config.min_delay}", 
        f"--max_delay={config.max_delay}", 

        f"--min_sample_idx={config.min_sample_idx}", 
        f"--max_sample_idx={config.max_sample_idx}", 
        f"--save_folder={config.save_folder}",
        f"--seed={seed}",
        f"--num_obstacles={config.num_obstacles}",
        f"--headless",
        ]
    # --disable_rgb is a store_true flag, so only append it when enabled.
    if config.disable_rgb:
        command.append("--disable_rgb")
    # import ipdb; ipdb.set_trace() 

    print(f"Running command: {' '.join(command)}")
    
    # Run the command
    try:
        # Get the repository root directory (assuming this script is in scripts/collect/)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Copopy current environment and set ENABLE_CAMERAS
        env = os.environ.copy()
        env["ENABLE_CAMERAS"] = "1"

        # result = subprocess.run( command,  cwd="/move/u/takaraet/whole_body_tracking", check=True, capture_output=False, text=True)
        # result = subprocess.run( command,  cwd="/move/u/karenvo/Projects/rmr_tracking",check=True, capture_output=False, text=True, env=env)
        result = subprocess.run( command,  cwd="/move/u/chrzhang/rmr_tracking", check=True, capture_output=False, text=True, env=env)
        print(f"Experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def expand_config_delays(config: ExperimentConfig) -> List[ExperimentConfig]:
    """Expand a config with delays into multiple configs, one for each delay value."""
    if config.delays is None:
        # If no delays specified, return the config as-is
        return [config]
    
    expanded_configs = []
    for delay in config.delays:
        expanded_config = replace(config, min_delay=delay, max_delay=delay, delays=None)
        expanded_configs.append(expanded_config)
    
    return expanded_configs

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--disable_rgb",
        action="store_true",
        default=False,
        help="Depth-only collection: skip the SigLIP RGB encoder so episodes store no rgb_embed keys.",
    )
    args = parser.parse_args()

    # Define your experiment configurations here
    #=========== 25 HZ Teleport Policy Ablation Test ===================================    
    
    experiment_configs = [
    ExperimentConfig(
            wandb_path= 'robot-mcrobotface/new_staircase/njj02br2/model_9500.pt',
            # wandb_path = 'justingu-stanfo`rd-university/takara_walk_isaac/p45lz75q',
# 'takaraet/tracking/5xjdxvln', #justingu-stanford-university/takara_rumba_isaac/up5d790d',
            episode_collect_length_s=8.0, # 8 for walk and stairs, 3.4 for staircase
            num_steps_collect=250,  # 250 for walk and stairs, 120 for staircase
            num_eps_collect= 20, # 2000 #10000, #8000
            min_sample_idx = 0,  # 0 for walk and stairs, 120 for staircase
            max_sample_idx = 20,  # 20 for walk and stairs, 140 for staircase
            save_folder='STAIRCASE_DATA_new_walk_and_stairs', #_OU
            delays=[0],  
            # num_obstacles=0,
    ),
    # 5333
    ]


    # Expand all configs based on their delay lists
    all_experiments = []
    for config in experiment_configs:
        all_experiments.extend(expand_config_delays(config))

    # Apply the CLI --disable_rgb flag to every experiment.
    for config in all_experiments:
        config.disable_rgb = args.disable_rgb

    # Run all experiments
    successful_experiments = 0
    total_experiments = len(all_experiments)
    print(total_experiments)
    for i, config in enumerate(all_experiments):
        print(f"\n{'='*50}")
        print(f"Running Experiment {i+1}/{total_experiments} (delay={config.min_delay})")
        print(f"{'='*50}")
        
        success = run_experiment(config, task="Staircase-G1-Collect-v0", num_envs=8, seed=args.seed) # 750
        if success:
            successful_experiments += 1
        
        print(f"Experiment {i} {'PASSED' if success else 'FAILED'}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {successful_experiments}/{total_experiments} experiments completed successfully")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
