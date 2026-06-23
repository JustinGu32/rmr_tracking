#!/bin/bash
#SBATCH --job-name=resume_generalist
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/resume_generalist/resume_generalist_%j.out
#SBATCH --error=logs/slurm/resume_generalist/resume_generalist_%j.err

mkdir -p logs/slurm/resume_generalist

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# ============================================================
# Resume the 10 generalist runs from the 2026-05-31 adaptive-sampling sweep
# by reloading their latest wandb checkpoints (via --wandb_resume).
#
# Categorizer / sampling-mode / sym_aug / category set / zarr must match the
# original training command in train_generalist_full.sh; --wandb_resume only
# restores the policy+critic weights.
#
# Sampling flag conventions:
#   * Runs with "catProb" / "clipProb" in their name were trained with the
#     NEW probability flags (--cat_uniform_prob / --clip_uniform_prob in [0,1])
#     → we pass those values directly.
#   * Runs with "catRatio" / "clipRatio" in their name were trained with the
#     OLD additive-floor flags (--cat_adaptive_uniform_ratio /
#     --clip_adaptive_uniform_ratio). These were briefly removed during the
#     rename but have been RESTORED in train_bones.py / generalist commands.py
#     to support exact-reproduction resumes. We pass the original ratios
#     verbatim; the env_cfg's `*_adaptive_uniform_ratio` field, when non-None,
#     overrides the corresponding `*_uniform_prob` and selects the old
#     additive-floor sampling formula.
#
# Uncomment the run(s) you want to resume.
# ============================================================

ZARR=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr
# Older zarr used by n5an93th and 8pmudt9g (pre-migration). Their training
# commands in train_generalist_full.sh (#1 and #2 in the old-flag section)
# passed this path explicitly, so we mirror it here.
OLD_ZARR=/move/data/bones/g1/zarr/locomotion_33hz.zarr
WANDB_ENTITY=robot-mcrobotface
LOG_PROJECT=generalist
VAE_K12_JSON=logs/motion_vae/v1/clip_clusters_k12.json
# NOTE: ki4gtu0j was trained with clip_clusters_k8.json when that file CONTAINED
# K=16 clusters (the cluster script was invoked with --k 16). The file has
# since been overwritten with K=8 content, so to match the trained policy's
# 16-head critic we must point at clip_clusters_k16.json now.
VAE_K16_JSON=logs/motion_vae/v1/clip_clusters_k16.json


# ── VAE-categorized (k=12) ────────────────────────────────────────────────

# # 1. raykysnl  —  VAE k=12, cat_blend_clip_uniform, catProb0.5, allMotions, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name raykysnl_VAE-cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/raykysnl \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path ${VAE_K12_JSON} \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 2. klkm63s6  —  VAE k=12, cat_adaptive_clip_adaptive, catProb0.5/clipProb0.75, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name klkm63s6_VAE-cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/klkm63s6 \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path ${VAE_K12_JSON} \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --clip_uniform_prob 0.75 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# ── VAE-categorized (older k=16) ──────────────────────────────────────────

# # 3. ki4gtu0j  —  VAE k=16, cat_blend_clip_uniform, OLD catRatio0.5, sym_aug
# #     Original training (train_generalist_full.sh #5 in old section) did NOT
# #     pass --clip_adaptive_uniform_ratio (cat_blend mode ignores it anyway).
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name ki4gtu0j_vae-cat_blend_clip_uniform-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/ki4gtu0j \
#     --decimation 6 --sampling uniform --activation swish \
#     --popart off \
#     --categorizer_mode latent_kmeans \
#     --latent_centroids_path ${VAE_K16_JSON} \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# ── Keyword 8-cat (allMotions) ────────────────────────────────────────────

# # 4. 329tiqlx  —  cat_blend_clip_uniform, catProb0.5, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name 329tiqlx_cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/329tiqlx \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 5. 8i82390u  —  cat_adaptive_clip_adaptive, catProb0.5/clipProb0.75, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name 8i82390u_cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/8i82390u \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_uniform_prob 0.5 \
#     --clip_uniform_prob 0.75 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 6. yr974gsu  —  cat_blend_clip_uniform, OLD catRatio0.5/clipRatio0.6, sym_aug
# #     (clipRatio is ignored by cat_blend mode but we pass it to match the
# #     original CLI verbatim; it has no functional effect.)
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name yr974gsu_cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/yr974gsu \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 7. n5an93th  —  cat_blend_clip_uniform, OLD catRatio0.5/clipRatio0.6, NO sym_aug
# #     ⚠ trained with OLD_ZARR (/move/data/bones/g1/zarr/locomotion_33hz.zarr)
python scripts/rsl_rl/train_bones.py \
    --task=Generalist-Flat-G1-v0 \
    --zarr_path=${OLD_ZARR} \
    --num_envs 4096 --headless \
    --logger wandb --log_project_name ${LOG_PROJECT} \
    --run_name n5an93th_cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions_resume \
    --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/n5an93th \
    --decimation 6 --sampling uniform --activation swish \
    --categories stand_up,walk,jump,run,jog,crouch,turn,idle --popart off \
    --sampling_mode cat_blend_clip_uniform \
    --cat_adaptive_uniform_ratio 0.5 \
    --clip_adaptive_uniform_ratio 0.6 \
    --terrain_noise \
    --jump_tighten_anchor_z


# ── Keyword 3-cat (stand_up,walk,jump) ────────────────────────────────────

# # 8. 8pmudt9g  —  cat_blend_clip_uniform, OLD catRatio0.5/clipRatio0.6, NO sym_aug
# #     ⚠ trained with OLD_ZARR (/move/data/bones/g1/zarr/locomotion_33hz.zarr)
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${OLD_ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name 8pmudt9g_cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/8pmudt9g \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 9. qq7w57b5  —  cat_adaptive_clip_adaptive, OLD catRatio0.5/clipRatio0.6, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name qq7w57b5_cat_adaptive_clip_adaptive-catRatio0.5-clipRatio0.6-stand_up,walk,jump-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/qq7w57b5 \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_adaptive_clip_adaptive \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z


# # 10. n1p6pgca  —  cat_blend_clip_uniform, OLD catRatio0.5/clipRatio0.6, sym_aug
# python scripts/rsl_rl/train_bones.py \
#     --task=Generalist-Flat-G1-v0 \
#     --zarr_path=${ZARR} \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${LOG_PROJECT} \
#     --run_name n1p6pgca_cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump-sym_aug_resume \
#     --wandb_resume ${WANDB_ENTITY}/${LOG_PROJECT}/n1p6pgca \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump --popart off \
#     --sampling_mode cat_blend_clip_uniform \
#     --symmetric_augment \
#     --cat_adaptive_uniform_ratio 0.5 \
#     --clip_adaptive_uniform_ratio 0.6 \
#     --terrain_noise \
#     --jump_tighten_anchor_z
