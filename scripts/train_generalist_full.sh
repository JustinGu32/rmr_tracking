#!/bin/bash
#SBATCH --job-name=generalist
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/generalist/generalist_%j.out
#SBATCH --error=logs/slurm/generalist/generalist_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# # ============================================================
# # Reference invocation: train the generalist with everything turned on.
# # Uses the CLEAN zarr + symmetric augmentation + three-level adaptive
# # sampling + history-10 proprioceptive obs. Edit the LOG_PROJECT / RUN_NAME
# # to your run names.
# # ============================================================

# ZARR=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr
# LOG_PROJECT=generalist
# RUN_NAME=generalist_v1_full_catRatio0.5_clipRatio0.6_allMotions

# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name ${RUN_NAME} \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --symmetric_augment \
#     --history_length 10 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# ZARR=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr
# LOG_PROJECT=BONES_walk_turn
# RUN_NAME=bones_walk_turn

# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 8192 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name ${RUN_NAME} \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories walk,turn --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --symmetric_augment \
#     --history_length 10 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

ZARR=/move/data/bones/g1/zarr/locomotion_33hz.zarr
LOG_PROJECT=BONES_walk_turn
RUN_NAME=bones_walk_turn_GlobalObs

python scripts/rsl_rl/train_bones.py \
    --task=Generalist-GlobalObs-Flat-G1-v0 \
    --zarr_path=${ZARR} \
    --num_envs 8192 --headless \
    --logger wandb --log_project_name ${LOG_PROJECT} \
    --run_name ${RUN_NAME} \
    --decimation 6 --sampling uniform --activation swish \
    --categories walk,turn --popart off \
    --sampling_mode cat_blend_clip_uniform \
    --cat_adaptive_uniform_ratio 0.5 \
    --history_length 10 \
    --terrain_noise \
    --jump_tighten_anchor_z


# ============================================================
# Alternative: VAE latent-K-means categorization instead of keyword
# categories. Drops `--categories` so ALL clips in the zarr are loaded,
# and replaces the keyword categorizer with a clusters JSON produced by
# scripts/cluster_motion_latents.py — clip→category is inferred from
# per-clip VAE latents rather than name keywords. num_categories is
# auto-derived from the JSON's `k` field (no need to pass it).
#
# Requires having already run, in order:
#   sbatch scripts/train_motion_vae.sh                    # train VAE
#   python  scripts/dump_motion_latents.py ...            # dump latents
#   python  scripts/cluster_motion_latents.py --k 8 ...   # cluster -> JSON
# ============================================================

# VAE_RUN_NAME=generalist_v1_full_vaeK8_catRatio0.5_clipRatio0.6_allMotions
# CENTROIDS_PATH=${CENTROIDS_PATH:-/move/u/justingu/rmr_tracking/logs/motion_vae/v1/clip_clusters_k8.json}

# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name ${VAE_RUN_NAME} \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path ${CENTROIDS_PATH} \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --symmetric_augment \
#     --history_length 10 \
#     --terrain_noise \
#     --jump_tighten_anchor_z





# # 1. all motions. full dataset (with mirrored), no sym aug.
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # 2. standup,walk,jump. full dataset (with mirrored), no sym aug.
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # 3. same as 2 but with sym aug.
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # # 4. cat_adaptive_clip_adaptive
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_adaptive_clip_adaptive-catRatio0.5-clipRatio0.6-stand_up,walk,jump-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # 5. VAE
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name vae-cat_blend_clip_uniform-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path logs/motion_vae/v1/clip_clusters_k8.json \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --symmetric_augment \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # 6. all motions. sym aug.
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# -- WITH UPDATED CAT_UNIFORM_PROB --
#
# Copies of 1-6 above, with the renamed probability-based flags:
#   --cat_adaptive_uniform_ratio  ->  --cat_uniform_prob
#   --clip_adaptive_uniform_ratio ->  --clip_uniform_prob
# NOTE: these values are now PROBABILITIES of uniform sampling in [0,1], NOT the
# old additive floors. 0.5 here means 50% uniform at the category stage — much
# more uniform than the old additive 0.5 (which was ~6% uniform at K=8, ~14% at
# K=3). To instead REPRODUCE an old run's behavior, set the prob to ratio/(K+ratio)
# e.g. old cat 0.5 at K=8 -> --cat_uniform_prob 0.0588; at K=3 -> 0.143.
# (clip_uniform_prob is unused by cat_blend_clip_uniform — its clip stage is uniform.)

# # # 1. cat_adaptive_clip_adaptive
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --clip_uniform_prob 0.75 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # # 2. cat_blend
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # # 3. cat_adaptive_clip_adaptive, VAE (12 clusters)
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name VAE-cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path logs/motion_vae/v1/clip_clusters_k12.json \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --clip_uniform_prob 0.75 \
#     --terrain_noise \
#     --jump_tighten_anchor_z

# # # 4. cat_blend, VAE (12 clusters)
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name generalist \
#     --run_name VAE-cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path logs/motion_vae/v1/clip_clusters_k12.json \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --terrain_noise \
#     --jump_tighten_anchor_z