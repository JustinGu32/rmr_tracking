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
        result = subprocess.run( command,  cwd="/move/u/karenvo/Projects/rmr_tracking",check=True, capture_output=False, text=True, env=env)
        # result = subprocess.run( command,  cwd=repo_root, check=True, capture_output=False, text=True)
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
    args = parser.parse_args()

    # Define your experiment configurations here
    #=========== 25 HZ Teleport Policy Ablation Test ===================================    
    
    # 25 hz walk fast
    experiment_configs = [
    ExperimentConfig(
            wandb_path= 'robot-mcrobotface/new_staircase/zvhor4js',
            # wandb_path = 'justingu-stanfo`rd-university/takara_walk_isaac/p45lz75q',
# 'takaraet/tracking/5xjdxvln', #justingu-stanford-university/takara_rumba_isaac/up5d790d',
            episode_collect_length_s=5,
            num_steps_collect=80,  # 60 -> 1.8 sec, 80 -> 2.4 sec
            num_eps_collect= 2000, #10000, #8000
            min_sample_idx = 0,
            max_sample_idx = 91, #16000, # 313 walk up, 171 walk down, 141 up, 91 down
            save_folder='down_continuous_v2_karen_stairs_collection_v2', #_OU
            delays=[0],  
            # num_obstacles=0,
    ),
    # 5333
    ]


    # Expand all configs based on their delay lists
    all_experiments = []
    for config in experiment_configs:
        all_experiments.extend(expand_config_delays(config))
    
    # Run all experiments
    successful_experiments = 0
    total_experiments = len(all_experiments)
    print(total_experiments)
    for i, config in enumerate(all_experiments):
        print(f"\n{'='*50}")
        print(f"Running Experiment {i+1}/{total_experiments} (delay={config.min_delay})")
        print(f"{'='*50}")
        
        success = run_experiment(config, task="Staircase-G1-Collect-v0", num_envs=175, seed=args.seed) # 750
        if success:
            successful_experiments += 1
        
        print(f"Experiment {i} {'PASSED' if success else 'FAILED'}")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {successful_experiments}/{total_experiments} experiments completed successfully")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
