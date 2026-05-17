#!/bin/bash
#SBATCH --job-name=play_standup
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/play_standup_%j.out
#SBATCH --error=logs/slurm/play_standup_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_stand_up_all.zarr"
# WANDB_PATH="robot-mcrobotface/multiclip_bones_standup/oyqt3l22"
# RUN_NAME="2026-05-07_13-15-51_tracking_33hz_standup_decimation6"
# VIDEO_DIR="clip_videos/standup33hz/${RUN_NAME}"
# TASK="Tracking-MultiClip-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # Enumerate all clip names in the zarr at runtime.
# mapfile -t MOTIONS < <(python -c "
# import zarr
# z = zarr.open('${ZARR_PATH}', mode='r')
# for n in z['clip_names'][:]:
#     print(n)
# ")

# echo "Found ${#MOTIONS[@]} clips in ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video
# done

# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_small.zarr"
# WANDB_PATH="robot-mcrobotface/justin_popart/f0b7yb5p"
# RUN_NAME="2026-05-10_01-12-52_popart_standupwalk_small"
# VIDEO_DIR="clip_videos/popart_standupwalk_small_f0b7yb5p/${RUN_NAME}"
# TASK="Popart-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # # Enumerate all clip names in the zarr at runtime.
# # mapfile -t MOTIONS < <(python -c "
# # import zarr
# # z = zarr.open('${ZARR_PATH}', mode='r')
# # for n in z['clip_names'][:]:
# #     print(n)
# # ")

# # Hand-picked subset: 1 stand_up + diverse walk variants.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done


# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_all.zarr"
# WANDB_PATH="robot-mcrobotface/justin_popart/xzvm6ej6"
# RUN_NAME="2026-05-10_01-12-52_popart_standupwalk_all"
# VIDEO_DIR="clip_videos/popart_standupwalk_all_xzvm6ej6/${RUN_NAME}"
# TASK="Popart-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # # Enumerate all clip names in the zarr at runtime.
# # mapfile -t MOTIONS < <(python -c "
# # import zarr
# # z = zarr.open('${ZARR_PATH}', mode='r')
# # for n in z['clip_names'][:]:
# #     print(n)
# # ")

# # Hand-picked subset: 3 distinct stand_up variants (back/side/stomach) + diverse walks.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "stand_up_lying_side_R_002__A472"
#     "stand_up_lying_stomach_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done



# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_small.zarr"
# WANDB_PATH="robot-mcrobotface/justin_popart/57ac4z1z"
# RUN_NAME="2026-05-10_17-22-45_balanced_popart_standup_walk_small"
# VIDEO_DIR="clip_videos/balanced_popart_standupwalk_small_57ac4z1z/${RUN_NAME}"
# TASK="Popart-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # # Enumerate all clip names in the zarr at runtime.
# # mapfile -t MOTIONS < <(python -c "
# # import zarr
# # z = zarr.open('${ZARR_PATH}', mode='r')
# # for n in z['clip_names'][:]:
# #     print(n)
# # ")

# # Hand-picked subset: 1 stand_up + diverse walk variants.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done


# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_small.zarr"
# WANDB_PATH="robot-mcrobotface/justin_popart/jb8bhnws"
# RUN_NAME="2026-05-10_17-21-47_balanced_vanilla_standup_walk_small"
# VIDEO_DIR="clip_videos/balanced_vanilla_standupwalk_small_jb8bhnws/${RUN_NAME}"
# TASK="Popart-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # # Enumerate all clip names in the zarr at runtime.
# # mapfile -t MOTIONS < <(python -c "
# # import zarr
# # z = zarr.open('${ZARR_PATH}', mode='r')
# # for n in z['clip_names'][:]:
# #     print(n)
# # ")

# # Hand-picked subset: 1 stand_up + diverse walk variants.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done








# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_all.zarr"
# WANDB_PATH="robot-mcrobotface/multiclip_bones_standup/bxtqge09"
# RUN_NAME="2026-05-10_17-33-46_tracking_33hz_standup_decimation6_finetuneWalk"
# VIDEO_DIR="clip_videos/finetuneWalk_bxtqge09/${RUN_NAME}"
# TASK="Tracking-MultiClip-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # Hand-picked subset: 1 stand_up + diverse walk variants.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done



# # ===================== Configuration =====================
# ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_all.zarr"
# WANDB_PATH="robot-mcrobotface/multiclip_bones_standup/qo2uubzn"
# RUN_NAME="2026-05-10_17-33-51_tracking_33hz_standup_decimation6_finetuneWalkStandup"
# VIDEO_DIR="clip_videos/finetuneWalkStandup_qo2uubzn/${RUN_NAME}"
# TASK="Tracking-MultiClip-Flat-G1-Play-v0"
# DECIMATION=6
# ACTIVATION="swish"
# # ==========================================================

# # Hand-picked subset: 1 stand_up + diverse walk variants.
# MOTIONS=(
#     "stand_up_lying_R_002__A472"
#     "walk_ff_loop_360_005__A059"
#     "walk_ff_loop_180_R_very_fast_001__A448"
#     "walk_ff_stop_270_R_very_slow_001__A444"
#     "walk_sideway_090_loop_001__A024"
#     "walk_arc_cw_stop_R_slow_002__A446"
#     "walk_backward_stop_004__A022"
#     "walk_180_R_002__A349"
#     "walk_hands_on_back_loop_002__A063"
#     "turn_start_walk_0090_005__A024"
# )

# echo "Playing ${#MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

# for motion in "${MOTIONS[@]}"; do
#     python scripts/rsl_rl/play_bones_clip.py \
#         --task=${TASK} \
#         --zarr_path=${ZARR_PATH} \
#         --wandb_path=${WANDB_PATH} \
#         --clip_name="${motion}" \
#         --video_dir=${VIDEO_DIR} \
#         --activation=${ACTIVATION} \
#         --decimation ${DECIMATION} \
#         --num_envs=1 \
#         --headless \
#         --video \
#         --categories stand_up,walk
# done



===================== Configuration =====================
ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
WANDB_PATH="robot-mcrobotface/balanced_sampling/iryct29h"
RUN_NAME="2026-05-11_17-22-51_balanced_vanilla_jump"
VIDEO_DIR="clip_videos/balanced_vanilla_jump_iryct29h/${RUN_NAME}"
TASK="Popart-Flat-G1-Play-v0"
DECIMATION=6
ACTIVATION="swish"
# ==========================================================

# Hand-picked subset: 1 stand_up + diverse walk variants.
JUMP_MOTIONS=(
    # "Jump_002__A017"
    # "Jump_forward_Left_001__A017"
    # "jump_backward_002__A021"
    # "jump_left_001__A021"
    # "jump_sideway_045_001__A021"
    # "jump_sideway_090_001__A021"
    # "jump_sideway_135_001__A021"
    # "turn_jump_090_001__A029"
    # "turn_jump_270_001__A046"
    # "turn_jump_360_001__A046"
    "high_jump_R_opt_2_001__A476"
    "high_jump_ff_180_R_opt_1_001__A476"
    # "jump_around_001__A492"
    # "jump_to_see_001__A350"
    # "jump_and_sit_R_001__A533"
    # "jump_and_land_heavy_001__A001"
    # "jump_and_land_light_001__A001"
    # "avoid_obstacle_jump_run_ff_180_R_002__A500"
)

echo "Playing ${#JUMP_MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

for motion in "${JUMP_MOTIONS[@]}"; do
    python scripts/rsl_rl/play_bones_clip.py \
        --task=${TASK} \
        --zarr_path=${ZARR_PATH} \
        --wandb_path=${WANDB_PATH} \
        --clip_name="${motion}" \
        --video_dir=${VIDEO_DIR} \
        --activation=${ACTIVATION} \
        --decimation ${DECIMATION} \
        --num_envs=1 \
        --headless \
        --video \
        --categories jump
done


===================== Configuration =====================
ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
WANDB_PATH="robot-mcrobotface/balanced_sampling/oy7awwpz"
RUN_NAME="2026-05-11_17-22-51_balanced_vanilla_jump"
VIDEO_DIR="clip_videos/balanced_vanilla_standup_walk_jump_oy7awwpz/${RUN_NAME}"
TASK="Popart-Flat-G1-Play-v0"
DECIMATION=6
ACTIVATION="swish"
# ==========================================================


echo "Playing ${#JUMP_MOTIONS[@]} hand-picked clips from ${ZARR_PATH}"

for motion in "${JUMP_MOTIONS[@]}"; do
    python scripts/rsl_rl/play_bones_clip.py \
        --task=${TASK} \
        --zarr_path=${ZARR_PATH} \
        --wandb_path=${WANDB_PATH} \
        --clip_name="${motion}" \
        --video_dir=${VIDEO_DIR} \
        --activation=${ACTIVATION} \
        --decimation ${DECIMATION} \
        --num_envs=1 \
        --headless \
        --video \
        --categories stand_up,walk,jump
done

