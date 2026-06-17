#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=hier_popart
#SBATCH --output=logs/slurm/hier_popart_%j.out
#SBATCH --error=logs/slurm/hier_popart_%j.err

set -euo pipefail
mkdir -p logs/slurm
cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export WANDB_ENTITY="karenvo-stanford-university"

# Hierarchical (motion-category x reward-head) PopArt + category-balanced sampling.
#
# Target a ZARR multi-clip task (CHIP multi-clip command exposes clip_names, which
# the categorizer needs): Bones-MultiClip-Compliance-G1-v0 or Bones-MultiClip-G1-v0.
#
# Key flags:
#   --popart_hierarchical          enable (category x head) PopArt
#   --popart_categorizer balanced  7-way motion categorizer (balanced/mid/fine)
#   --popart_balanced_sampling     uniform category -> clip -> frame sampling
#   --popart_group_preset          reward-head grouping (must cover the env's reward terms;
#                                   use all_rewards if unsure)
#   --popart_actor_advantage_scaling raw   recommended (whitening over-amplifies penalties)

python scripts/rsl_rl/train_bones.py \
  --task=Bones-MultiClip-Flat-G1-v0 \
  --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
  --num_envs=4096 \
  --headless \
  --logger wandb \
  --log_project_name multiclip_bones_popart \
  --run_name multiclip_flat_hier_popart_jump_head_weights_mom0.001_bs \
  --popart_hierarchical \
  --popart_group_preset upper_lower \
  --popart_actor_advantage_scaling raw \
  --popart_momentum 0.001 \
  --popart_categorizer balanced \
  --popart_balanced_sampling \
  --popart_category_head_weights '{"jump": [1.0, 3.0, 3.0, 0.5, 0.5]}'

# heads for upper_lower preset (in order):
#   0: global_pose_tracking
#   1: lower_limb_tracking      ← 3x for jump (foot z)
#   2: motion_dynamics           ← 3x for jump (vertical velocity)
#   3: regularization_constraints ← 0.5x (allow more aggressive joint motion)
#   4: upper_limb_tracking       ← 0.5x (arms matter less during jump)

#   --warmstart_actor_from karenvo-stanford-university/multiclip_bones_popart/6wwgw3ig

# Ablation B (balanced sampling, NO PopArt) — the key baseline for "B vs E":
#   drop --popart_hierarchical and the popart_* flags, keep balanced sampling by
#   instead enabling it on the command (currently only wired through hierarchical).
