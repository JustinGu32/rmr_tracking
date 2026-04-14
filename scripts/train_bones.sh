#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --job-name=bones_crane
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


set -uo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# --- Crane ablations (12 variants) ---
# --- push normal (1-4) ---
# # 1. delta + push-normal
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_rtx \
#    --ppo_output delta \
#    --push normal \
#    --crane \
#    --video --video_interval 10000

# # 2. delta + push-normal + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_rtx \
#    --ppo_output delta \
#    --push normal \
#    --no_command_obs \
#    --crane \
#    --video --video_interval 10000

# # 3. target + push-normal
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_rtx \
#    --ppo_output target \
#    --push normal \
#    --crane \
#    --video --video_interval 10000

# # 4. target + push-normal + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_rtx \
#    --ppo_output target \
#    --push normal \
#    --no_command_obs \
#    --crane \
#    --video --video_interval 10000

# # --- push soft (5-8) ---
# # 5. delta + push-soft
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_l40s \
#    --ppo_output delta \
#    --push soft \
#    --crane \
#    --video --video_interval 10000

# # 6. delta + push-soft + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_l40s \
#    --ppo_output delta \
#    --push soft \
#    --no_command_obs \
#    --crane \
#    --video --video_interval 10000

# # 7. target + push-soft
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_l40s \
#    --ppo_output target \
#    --push soft \
#    --crane \
#    --video --video_interval 10000

# # 8. target + push-soft + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_l40s \
#    --ppo_output target \
#    --push soft \
#    --no_command_obs \
#    --crane \
#    --video --video_interval 10000

# # --- push none (9-12) ---
# # 9. delta + push-none
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_a5000 \
#    --ppo_output delta \
#    --push none \
#    --crane 

# # 10. delta + push-none + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_a5000 \
#    --ppo_output delta \
#    --push none \
#    --no_command_obs \
#    --crane 

# # 11. target + push-none
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_a5000 \
#    --ppo_output target \
#    --push none \
#    --crane 

# # 12. target + push-none + no-cmd-obs
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_a5000 \
#    --ppo_output target \
#    --push none \
#    --no_command_obs \
#    --crane 

# --- curriculum ablations (13-16) ---
# # 13. delta + push-none + curriculum
python scripts/rsl_rl/train_bones.py \
   --task=Bones-Flat-chip-G1-v0 \
   --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
   --headless \
   --logger wandb \
   --log_project_name bones_crane_ablation \
   --run_name crane_a5000_both_all \
   --ppo_output target \
   --push none \
   --crane \
   --curriculum \
   --assist_mode both_all \
   --video --video_interval 10000

# 14. delta + push-none + no curriculum (baseline, same as #9)
# python scripts/rsl_rl/train_bones.py \
#   --task=Bones-Flat-chip-G1-v0 \
#   --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#   --headless \
#   --logger wandb \
#   --log_project_name bones_crane_ablation \
#   --run_name DELTA-ALL_crane_a5000 \
#   --ppo_output delta \
#   --push none

# # 15. target + push-none + curriculum
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name crane_a5000 \
#    --ppo_output target \
#    --push none \
#    --crane \
#    --curriculum

# # 16. target + push-none + no curriculum (baseline, same as #11)
# python scripts/rsl_rl/train_bones.py \
#    --task=Bones-Flat-chip-G1-v0 \
#    --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane:v0 \
#    --headless \
#    --logger wandb \
#    --log_project_name bones_crane_ablation \
#    --run_name MAIN-URDF_crane_a5000 \
#    --ppo_output target \
#    --push none

