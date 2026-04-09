#!/bin/bash
#SBATCH --job-name=multiclip_train
#SBATCH --partition=humanoid  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/multiclip_%j.out
#SBATCH --error=logs/slurm/multiclip_%j.err

# Create log directory
mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# BeyondMimic baseline (Tracking task) — 4 configs
# ============================================================

# 1. BM, delta-all, 50hz (decimation 4, default)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_delta-all_50hz \
#     --ppo_output delta-all \
#     --double_step

# 2. BM, target, 50hz (decimation 4, default)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_target_50hz \
#     --ppo_output target \
#     --double_step

# 3. BM, delta-all, 33hz (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_delta-all_33hz \
#     --ppo_output delta-all \
#     --decimation 6 \
#     --double_step

# 4. BM, target, 33hz (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_target_33hz \
#     --ppo_output target \
#     --decimation 6 \
#     --double_step

# ============================================================
# BONES Compliance task — 4 configs
# ============================================================

# 5. BONES, delta-all, 50hz (decimation 4, default)
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_delta-all_50hz \
#     --ppo_output delta-all \
#     --double_step

# 6. BONES, target, 50hz (decimation 4, default)
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_target_50hz \
#     --ppo_output target \
#     --double_step

# 7. BONES, delta-all, 33hz (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_delta-all_33hz \
#     --ppo_output delta-all \
#     --decimation 6 \
#     --double_step

# 8. BONES, target, 33hz (decimation 6)
python scripts/rsl_rl/train.py \
    --task=Bones-MultiClip-Compliance-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_bones \
    --run_name bones_loco_target_33hz \
    --ppo_output target \
    --decimation 6 \
    --double_step

# ============================================================
# 16k envs / 6 steps variants (same configs, larger batch)
# ============================================================

# ---- BeyondMimic baseline ----

# 9. BM, delta-all, 50hz, 16k envs
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_delta-all_50hz_16k \
#     --ppo_output delta-all \
#     --double_step

# 10. BM, target, 50hz, 16k envs
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_target_50hz_16k \
#     --ppo_output target \
#     --double_step

# 11. BM, delta-all, 33hz, 16k envs (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_delta-all_33hz_16k \
#     --ppo_output delta-all \
#     --decimation 6 \
#     --double_step

# 12. BM, target, 33hz, 16k envs (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_target_33hz_16k \
#     --ppo_output target \
#     --decimation 6 \
#     --double_step

# ---- BONES Compliance ----

# 13. BONES, delta-all, 50hz, 16k envs
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_delta-all_50hz_16k \
#     --ppo_output delta-all \
#     --double_step

# 14. BONES, target, 50hz, 16k envs
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_target_50hz_16k \
#     --ppo_output target \
#     --double_step

# 15. BONES, delta-all, 33hz, 16k envs (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_delta-all_33hz_16k \
#     --ppo_output delta-all \
#     --decimation 6 \
#     --double_step

# 16. BONES, target, 33hz, 16k envs (decimation 6)
# python scripts/rsl_rl/train.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_loco_target_33hz_16k \
#     --ppo_output target \
#     --decimation 6 \
#     --double_step
