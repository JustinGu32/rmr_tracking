"""Evaluate a trained RL policy and report success/failure rates.

Usage:
    python scripts/rsl_rl/eval.py --task=Staircase-G1-v0 --num_envs=10 --num_episodes=100 \
        --wandb_path=robot-mcrobotface/staircase_final/RUN_ID --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RL policy.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--push", action="store_true", default=False, help="Enable push perturbations during evaluation.")
parser.add_argument("--push_feet", action="store_true", default=False, help="Enable force push perturbations on feet/pelvis during evaluation.")
parser.add_argument("--depth_obs", action="store_true", default=False, help="Enable optional RGB-D camera depth observations.")
parser.add_argument("--results_dir", type=str, default="eval_results", help="Directory to save result JSON files.")
parser.add_argument("--results_name", type=str, default=None, help="Filename for the result JSON (without .json extension).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.push:
    os.environ["WBT_PUSH"] = "1"
if args_cli.push_feet:
    os.environ["WBT_PUSH_FEET"] = "1"
if args_cli.depth_obs:
    os.environ["WBT_USE_DEPTH_OBS"] = "1"
    args_cli.enable_cameras = True

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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate policy and report termination statistics."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

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
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name
        # find the best/latest model checkpoint
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            print("[WARN] No motion artifact found in the run.")
        else:
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # --- Evaluation loop ---
    raw_env = env.unwrapped
    term_manager = raw_env.termination_manager
    term_names = term_manager._term_names

    # Get the motion command term to access metrics
    motion_command = raw_env.command_manager.get_term("motion")
    metric_names = list(motion_command.metrics.keys())

    # Per-termination counters
    term_counts = {name: 0 for name in term_names}
    # Per-metric accumulators (sum and count for computing means)
    metric_sums = {name: 0.0 for name in metric_names}
    metric_step_count = 0
    total_episodes = 0
    total_failures = 0
    target_episodes = args_cli.num_episodes

    print(f"\n[EVAL] Running {target_episodes} episodes across {args_cli.num_envs} parallel envs...")
    print(f"[EVAL] Termination terms: {term_names}")
    print(f"[EVAL] Tracking metrics: {metric_names}\n")

    obs, _ = env.reset()

    while total_episodes < target_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Accumulate metrics every step across all envs
        for name in metric_names:
            metric_sums[name] += motion_command.metrics[name].sum().item()
        metric_step_count += args_cli.num_envs

        # Check which envs just finished
        terminated = raw_env.reset_terminated  # (num_envs,) — failure
        timed_out = raw_env.reset_time_outs  # (num_envs,) — success (completed motion)
        done = terminated | timed_out

        if not done.any():
            continue

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        num_done = done_ids.numel()
        num_failed = terminated.sum().item()

        # Count per-termination breakdowns for done envs
        per_term = term_manager._term_dones[done_ids]  # (num_done, num_terms)
        for i, name in enumerate(term_names):
            term_counts[name] += per_term[:, i].sum().item()

        total_episodes += num_done
        total_failures += num_failed

        if total_episodes % 10 < num_done:
            print(f"  Progress: {total_episodes}/{target_episodes} episodes, "
                  f"failures so far: {total_failures}")

    # --- Print results ---
    total_successes = total_episodes - total_failures
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({total_episodes} episodes)")
    print("=" * 60)
    print(f"  Success (time_out):  {total_successes}/{total_episodes} ({success_rate:.1%})")
    print(f"  Failures:            {total_failures}/{total_episodes} ({1 - success_rate:.1%})")
    print()
    print("Per-termination breakdown:")
    print("-" * 40)
    for name in term_names:
        count = int(term_counts[name])
        pct = count / total_episodes * 100 if total_episodes > 0 else 0.0
        label = "(timeout/success)" if name == "time_out" else "(failure)"
        print(f"  {name:30s} {label:20s} {count:5d}  ({pct:.1f}%)")
    print()
    print("Tracking metrics (mean across all steps):")
    print("-" * 40)
    metric_means = {}
    for name in metric_names:
        mean_val = metric_sums[name] / metric_step_count if metric_step_count > 0 else 0.0
        metric_means[name] = mean_val
        print(f"  {name:30s} {mean_val:.6f}")
    print("=" * 60)

    # --- Save results to JSON ---
    results = {
        "task": args_cli.task,
        "wandb_path": args_cli.wandb_path,
        "wandb_run_name": wandb_run_name if args_cli.wandb_path else None,
        "num_envs": args_cli.num_envs,
        "num_episodes": int(total_episodes),
        "push": args_cli.push,
        "push_feet": args_cli.push_feet,
        "success_rate": success_rate,
        "total_successes": int(total_successes),
        "total_failures": int(total_failures),
        "per_termination": {name: int(term_counts[name]) for name in term_names},
        "metrics": metric_means,
    }

    results_dir = args_cli.results_dir
    os.makedirs(results_dir, exist_ok=True)

    if args_cli.results_name:
        filename = f"{args_cli.results_name}.json"
    else:
        push_suffix = "_push" if args_cli.push else ("_push_feet" if args_cli.push_feet else "")
        if args_cli.wandb_path:
            filename = f"{wandb_run_id}_{wandb_run_name}{push_suffix}.json"
        else:
            filename = f"{args_cli.task}{push_suffix}.json"

    results_path = os.path.join(results_dir, filename)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Results saved to: {results_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
