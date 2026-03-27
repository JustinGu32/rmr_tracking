#!/bin/bash
#SBATCH --job-name=multiclip_train
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/multiclip_%j.out
#SBATCH --error=logs/slurm/multiclip_%j.err

# # Original bones_multi config
# mkdir -p logs/slurm
# conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_tracking \
#     --run_name locomotion_33hz

# Create log directory
mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# # ============================================================
# # BeyondMimic baseline (Tracking task) — 5 configs
# # ============================================================

# # 1. Full dataset, target mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/motions_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_full_target_50hz \
#     --ppo_output target

# # 2. Full dataset, delta mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/motions_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_full_delta_50hz \
#     --ppo_output delta

# # 3. Locomotion dataset, target mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name MAIN-URDF_bm_loco_target_50hz \
#     --ppo_output target

# # 4. Locomotion dataset, delta mode, 33hz (decimation 6)
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bm_loco_delta_33hz \
#     --ppo_output delta \
#     --decimation 6

# # 5. Locomotion dataset, target mode, 50hz (decimation 4)
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name MAIN-URDF_bm_loco_target_50hz \
#     --ppo_output target

# 5. Locomotion dataset, target mode, 50hz (decimation 4)
python scripts/rsl_rl/train.py \
    --task=Tracking-MultiClip-Flat-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_bones \
    --run_name MAIN-URDF_bm_loco_target_50hz \
    --ppo_output target

# # 10. Locomotion dataset, delta mode, 50hz (decimation 4)
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name MAIN-URDF_compliance_loco_delta_50hz \
#     --ppo_output delta

# # ============================================================
# # CHIP Compliance (Bones task) — 5 configs
# # ============================================================

# # 6. Full dataset, target mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/motions_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name compliance_full_target_50hz \
#     --ppo_output target

# # 7. Full dataset, delta mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/motions_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name compliance_full_delta_50hz \
#     --ppo_output delta

# # 8. Locomotion dataset, target mode, 50hz
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name compliance_loco_target_50hz \
#     --ppo_output target

# # 9. Locomotion dataset, delta mode, 33hz (decimation 6)
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name compliance_loco_delta_33hz \
#     --ppo_output delta \
#     --decimation 6

# # 10. Locomotion dataset, delta mode, 50hz (decimation 4)
# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name compliance_loco_delta_50hz \
#     --ppo_output delta
