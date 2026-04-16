#!/bin/bash
#SBATCH --job-name=multiclip_bones
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/multiclip_bones_%j.out
#SBATCH --error=logs/slurm/multiclip_bones_%j.err

# Create log directory
mkdir -p logs/slurm

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# BONES MultiClip Compliance ablations
# ============================================================

# # 1. Baseline: target, elu, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_gravcurr \
#     --ppo_output target \
#     --activation elu \
#     --double_step \
#     --gravity_curriculum --start_gravity -2.0 --gravity_ramp_steps 5000

# # 2. delta-all, elu, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_delta-all_50hz_loose-terms \
#     --ppo_output delta-all \
#     --activation elu \
#     --double_step

# # 3. target, swish, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_swish_loose-terms \
#     --ppo_output target \
#     --activation swish \
#     --double_step

# # 4. target, elu, 50hz, 16k envs / 6 steps
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_16k_loose-terms \
#     --ppo_output target \
#     --activation elu \
#     --double_step

# # 5. target, elu, 33hz, decimation 6, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_33hz_loose-terms \
#     --ppo_output target \
#     --activation elu \
#     --decimation 6 \
#     --double_step

# # 6. kitchen sink: target, swish, 50hz, 16k envs / 6 steps
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_swish_16k_loose-terms \
#     --ppo_output target \
#     --activation swish \
#     --double_step

# # 7. kitchen sink: delta-all, swish, 50hz, 16k envs / 6 steps
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=16384 \
#     --num_steps_per_env 6 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_delta-all_50hz_swish_16k_loose-terms \
#     --ppo_output delta-all \
#     --activation swish \
#     --double_step

# # 8. BASELINE: standard tracking task (no BONES/compliance), target, elu, 50hz, 4096 envs
<<<<<<< HEAD
python scripts/rsl_rl/train_bones.py \
    --task=Tracking-MultiClip-Flat-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_bones \
    --run_name tracking_baseline_target_50hz \
    --ppo_output target \
    --activation elu \
    --double_step \
    --bones_popart_balanced
=======
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name tracking_baseline_target_50hz \
#     --ppo_output target \
#     --activation elu \
#     --double_step

# -- tryna get stuff to work --

# # 1. Base CHIP: target, elu, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_elu_gravcurr6.7_uniform \
#     --ppo_output target \
#     --activation elu \
#     --double_step \
#     --gravity_curriculum --start_gravity -6.7 --gravity_ramp_steps 120000 \
#     --sampling uniform

# # 2. Swish CHIP: target, swish, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_50hz_swish_gravcurr12.7_uniform \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --gravity_curriculum --start_gravity -12.7 --gravity_ramp_steps 120000 \
#     --sampling uniform

# # 3. BASELINE BeyondMimic: standard tracking task (no BONES/compliance), target, elu, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name tracking_baseline_target_50hz_elu \
#     --ppo_output target \
#     --activation elu \
#     --double_step


# # 4. Swish BeyondMimic: standard tracking task (no BONES/compliance), target, swish, 50hz, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name tracking_baseline_target_50hz_swish \
#     --ppo_output target \
#     --activation swish \
#     --double_step
>>>>>>> abc6371ded8678e97eb1502bc835e708780c4b0e
