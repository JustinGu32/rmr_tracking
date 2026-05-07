#!/bin/bash
#SBATCH --job-name=play_clip
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
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
VIDEO_DIR="clip_videos/multiclip_gravity"

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


# GRAVITY_12_WANDB="robot-mcrobotface/multiclip_bones/u90b2ybr"
# GRAVITY_6_WANDB="robot-mcrobotface/multiclip_bones/tdhhyrl8"
# TRACKING_GRAVITY_12_WANDB="robot-mcrobotface/multiclip_bones/etrf9b1c"
# TRACKING_TIMETOLIVE="robot-mcrobotface/multiclip_bones/3oftz6dh"

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

# MOTIONS=(
#     walk_arc_cw_loop_R_very_slow_001__A444_M
#     Jump_002__A017_M
#     jump_and_land_light_003__A001
#     walk_forward_normal_001__A006
#     Turn_Start_Jog_0360_001__A019
#     stand_up_lying_R_002__A475_M
#     stand_up_lying_stomach_R_002__A472
#     stand_up_lying_side_R_002__A475
# )

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
#         --wandb_path=robot-mcrobotface/multiclip_bones/p2m0jbp3 \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/gravity_4 \
#         --activation=swish \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=robot-mcrobotface/multiclip_bones/6q0ka6qq \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/gravity_16 \
#         --activation=swish \
#         --num_envs=1 \
#         --headless \
#         --video
# done


# --- play evals ---
WALK_JOG_MOTIONS=(
    walk_forward_normal_001__A006
    walk_arc_cw_loop_R_very_slow_001__A444_M
    Turn_Start_Jog_0360_001__A019
)

WALK_JOG_JUMP_MOTIONS=(
    walk_forward_normal_001__A006
    walk_arc_cw_loop_R_very_slow_001__A444_M
    Turn_Start_Jog_0360_001__A019
    Jump_002__A017_M
    jump_and_land_light_003__A001
)

GENERAL_MOTIONS=(
    walk_arc_cw_loop_R_very_slow_001__A444_M
    Jump_002__A017_M
    jump_and_land_light_003__A001
    walk_forward_normal_001__A006
    Turn_Start_Jog_0360_001__A019
    stand_up_lying_R_002__A475_M
    stand_up_lying_stomach_R_002__A472
    stand_up_lying_side_R_002__A475
)


# for motion in "${WALK_JOG_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/dpm67awe \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/walk_jog_FINETUNED_33hz_dpm67awe \
#         --activation=swish \
#         --decimation 6 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# run_id | policy_name | dataset_path | decimation
# qqerhnkp | bones_100hz_gravity12.81_decimation2_finetune33hz_dec6 | /move/data/bones/g1/zarr/locomotion_33hz.zarr | 6
# awvfb9ba | tracking_100hz_gravity12.81_decimation2_finetune33hz_dec6 | /move/data/bones/g1/zarr/locomotion_33hz.zarr | 6
# 6k3qmjbm | bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_finetune33hz_dec6 | /move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr | 6
# bw8lhrfu | bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_resumed_dec2 | /move/u/justingu/rmr_tracking/motions/locomotion_100hz_walk_jog_jump_all.zarr | 2
# 8fvblkk8 | bones_100hz_gravity12.81_decimation2_resumed | /move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr | 2
# 2lo1gj66 | tracking_100hz_gravity12.81_decimation2_resumed | /move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr | 2
# 7p3hop2s | bones_target_33hz_swish_uniform_walk-jog_resumed | /move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr | 6
# mch9kxtr | bones_target_50hz_swish_gravcurr12.7_uniform_resumed2 | /move/data/bones/g1/zarr/locomotion_50hz.zarr | 4
# 3oftz6dh | tracking_baseline_target_50hz_swish_uniform_xyterm_timetolive | /move/data/bones/g1/zarr/locomotion_50hz.zarr | 4


# # 1. qqerhnkp -- bones_100hz_gravity12.81_decimation2_finetune33hz_dec6
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/qqerhnkp \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/bones_100hz_gravity12.81_decimation2_finetune33hz_dec6 \
#         --activation=swish \
#         --decimation 6 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 2. awvfb9ba -- tracking_100hz_gravity12.81_decimation2_finetune33hz_dec6
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Tracking-MultiClip-Flat-G1-Play-v0 \
#         --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/awvfb9ba \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/tracking_100hz_gravity12.81_decimation2_finetune33hz_dec6 \
#         --activation=swish \
#         --decimation 6 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 3. 6k3qmjbm -- bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_finetune33hz_dec6
# for motion in "${WALK_JOG_JUMP_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/6k3qmjbm \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_finetune33hz_dec6 \
#         --activation=swish \
#         --decimation 6 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 4. bw8lhrfu -- bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_resumed_dec2
# for motion in "${WALK_JOG_JUMP_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz_walk_jog_jump_all.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/bw8lhrfu \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_resumed_dec2 \
#         --activation=swish \
#         --decimation 2 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 5. 8fvblkk8 -- bones_100hz_gravity12.81_decimation2_resumed
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/8fvblkk8 \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/bones_100hz_gravity12.81_decimation2_resumed \
#         --activation=swish \
#         --decimation 2 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 6. 2lo1gj66 -- tracking_100hz_gravity12.81_decimation2_resumed
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Tracking-MultiClip-Flat-G1-Play-v0 \
#         --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/2lo1gj66 \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/tracking_100hz_gravity12.81_decimation2_resumed \
#         --activation=swish \
#         --decimation 2 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# 7. 7p3hop2s -- bones_target_33hz_swish_uniform_walk-jog_resumed
for motion in "${WALK_JOG_MOTIONS[@]}"; do
    python scripts/rsl_rl/play_bones_clip.py \
        --task=Bones-MultiClip-Compliance-G1-v0 \
        --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr \
        --wandb_path=robot-mcrobotface/multiclip_bones/7p3hop2s \
        --clip_name="${motion}" \
        --video_dir=${VIDEO_DIR}/bones_target_33hz_swish_uniform_walk-jog_resumed \
        --activation=swish \
        --decimation 6 \
        --num_envs=1 \
        --headless \
        --video
done

# # 8. mch9kxtr -- bones_target_50hz_swish_gravcurr12.7_uniform_resumed2
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Bones-MultiClip-Compliance-G1-v0 \
#         --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/mch9kxtr \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/bones_target_50hz_swish_gravcurr12.7_uniform_resumed2 \
#         --activation=swish \
#         --decimation 4 \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # 9. 3oftz6dh -- tracking_baseline_target_50hz_swish_uniform_xyterm_timetolive
# for motion in "${GENERAL_MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=Tracking-MultiClip-Flat-G1-Play-v0 \
#         --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#         --wandb_path=robot-mcrobotface/multiclip_bones/3oftz6dh \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR}/tracking_baseline_target_50hz_swish_uniform_xyterm_timetolive \
#         --activation=swish \
#         --decimation 4 \
#         --num_envs=1 \
#         --headless \
#         --video
# done