#!/bin/bash
#SBATCH --job-name=eval_multiclip
#SBATCH --partition=humanoid  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/eval_multiclip_%j.out
#SBATCH --error=logs/slurm/eval_multiclip_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/eval_multiclip.py \
    --task=Bones-MultiClip-Compliance-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
    --wandb_path=robot-mcrobotface/multiclip_bones/orj1uirj \
    --num_envs=16384 \
    --headless \
    --results_dir=eval_results/multiclip_pt1 \
    --results_name=bones_target_50hz_elu_orj1uirj

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/yp2xkmja \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_pt1 \
#     --results_name=tracking_baseline_target_50hz_elu_yp2xkmja
    