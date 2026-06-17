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

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ===================== Configuration =====================
ZARR_PATH="/move/data/bones/g1/zarr/locomotion_50hz.zarr"
VIDEO_DIR="eval_results/clip_videos/multiclip_flat_hier_popart_jump_head_weights_mom0.3_bs_uniform_hierpopart_bal/turn"

# crane_new_tracking_baseline"
# crane_new_baseline_UL
# crane_upperlower_raw_uniform_mom0.0005_v2_popart"

WANDB_RUN="karenvo-stanford-university/multiclip_bones_popart/cgwt6x25"

# Clip selection (pick ONE)
CLIP_FLAG_jump="--clip_name=jump_and_land_light_003__A001"
# CLIP_FLAG="--clip_id=0"

# Locomotion - forward walk
CLIP_FLAG_forward_walk="--clip_name=walk_forward_shoulder_amplified_002__A001"

# Walk backward
CLIP_FLAG_backward_walk="--clip_name=walk_backward_loop_001__A022"

# Turn
CLIP_FLAG_turn="--clip_name=Turn_Start_Jog_0000_001__A017"
# ==========================================================

python scripts/rsl_rl/play_bones_clip.py \
    --task=Bones-MultiClip-Flat-G1-v0 \
    --zarr_path=${ZARR_PATH} \
    --wandb_path=${WANDB_RUN} \
    ${CLIP_FLAG_turn} \
    --video_dir=${VIDEO_DIR} \
    --num_envs=1 \
    --headless \
    --video \
    --popart_head_mode=grouped \
    --popart_group_preset=upper_lower

# python scripts/rsl_rl/play.py \
#     --task=Tracking-Flat-G1-Play-v0 \
#     --wandb_path=robot-mcrobotface/multiclip_bones_popart/gerdxo4n \
#     --num_envs=1 \
#     --headless \
#     --video

# cnbj48t9 â Bones-Flat-chip-G1-v0, no PopArt
# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --wandb_path=robot-mcrobotface/multiclip_bones_popart/cnbj48t9 \
#     --num_envs=1 --headless --video --video_length=500 \
#     --video_folder=${VIDEO_DIR}

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-G1-v0 \
#     --wandb_path=robot-mcrobotface/multiclip_bones_popart/e5bufwyf \
#     --num_envs=1 --headless --video \
#     --video_folder=${VIDEO_DIR}

# python scripts/rsl_rl/play_bones.py \
#     --task=Tracking-Flat-G1-Play-v0 \
#     --wandb_path=robot-mcrobotface/multiclip_bones_popart/gerdxo4n \
#     --num_envs=1 \
#     --headless \
#     --video \
#     --video_length=500 \
#     --video_folder=${VIDEO_DIR}