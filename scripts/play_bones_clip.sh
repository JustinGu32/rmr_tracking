#!/bin/bash
#SBATCH --job-name=play_clip
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/play_clip_%j.out
#SBATCH --error=logs/slurm/play_clip_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ===================== Configuration =====================
ZARR_PATH="/move/data/bones/g1/zarr/locomotion_50hz.zarr"
VIDEO_DIR="eval_results/clip_videos/multiclip_gravity"

# Clip selection (pick ONE)
# CLIP_FLAG="--clip_name=jump_and_land_light_003__A001"
CLIP_FLAG="--clip_name=Jump_002__A017_M"
# CLIP_FLAG="--clip_name=sitting_idle_R_001__A459_M"
# CLIP_FLAG="--clip_name=walk_forward_normal_001__A006"
# CLIP_FLAG="--clip_name=walk_arc_cw_loop_R_very_slow_001__A444_M"
# CLIP_FLAG="--clip_name=Turn_Start_Jog_0360_001__A019"

# CLIP_FLAG="--clip_id=86"

# Policy checkpoints
BONES_WANDB="robot-mcrobotface/multiclip_bones/x6i28m58"
TRACKING_WANDB="robot-mcrobotface/multiclip_bones/3dk71nvh"
# ==========================================================

# # --- Bones compliance policy ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${BONES_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR} \
#     --num_envs=1 \
#     --headless \
#     --video

# # --- Tracking baseline policy ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${TRACKING_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR} \
#     --num_envs=1 \
#     --headless \
#     --video


GRAVITY_12_WANDB="robot-mcrobotface/multiclip_bones/u90b2ybr"
GRAVITY_6_WANDB="robot-mcrobotface/multiclip_bones/tdhhyrl8"
TRACKING_GRAVITY_12_WANDB="robot-mcrobotface/multiclip_bones/etrf9b1c"
TRACKING_TIMETOLIVE="robot-mcrobotface/multiclip_bones/3oftz6dh"

# --- Bones compliance policy (gravity 12) ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${GRAVITY_12_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR}/gravity_12_pt2 \
#     --activation=swish \
#     --num_envs=1 \
#     --headless \
#     --video

# # --- Bones compliance policy (gravity 6) ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${GRAVITY_6_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR}/gravity_6_pt2 \
#     --activation=swish \
#     --num_envs=1 \
#     --headless \
#     --video

# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${TRACKING_GRAVITY_12_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR}/tracking_gravity_12_pt2 \
#     --activation=swish \
#     --num_envs=1 \
#     --headless \
#     --video

MOTIONS=(
    walk_arc_cw_loop_R_very_slow_001__A444_M
    # Jump_002__A017_M
    # jump_and_land_light_003__A001
    walk_forward_normal_001__A006
    Turn_Start_Jog_0360_001__A019
    # stand_up_lying_R_002__A475_M
    # stand_up_lying_stomach_R_002__A472
    # stand_up_lying_side_R_002__A475
)

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Tracking-MultiClip-Flat-G1-v0 \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${TRACKING_TIMETOLIVE} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/tracking_timetolive \
#         --activation=swish \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=robot-mcrobotface/multiclip_bones/tg334ct9 \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/gravity_6_pt3 \
#         --activation=swish \
#         --num_envs=1 \
#         --headless \
#         --video
# done    

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=robot-mcrobotface/multiclip_bones/mch9kxtr \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/gravity_12_pt3 \
#         --activation=swish \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# --- Walk+jog 33hz swish uniform (eval run c15qko8c) ---
WALK_JOG_MOTIONS=(
    walk_forward_normal_001__A006
    walk_arc_cw_loop_R_very_slow_001__A444_M
    Turn_Start_Jog_0360_001__A019
)
for motion in "${WALK_JOG_MOTIONS[@]}"; do
    python scripts/rsl_rl/play_bones_clip.py \
        --task=Bones-MultiClip-Compliance-G1-v0 \
        --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
        --wandb_path=robot-mcrobotface/multiclip_bones/3j0fwfyc \
        --clip_name="${motion}" \
        --video_dir=${VIDEO_DIR}/walk_jog_33hz_3j0fwfyc \
        --activation=swish \
        --num_envs=1 \
        --headless \
        --video
done
