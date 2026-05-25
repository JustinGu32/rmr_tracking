#!/bin/bash
#SBATCH --job-name=eval_multiclip
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/eval_multiclip_%j.out
#SBATCH --error=logs/slurm/eval_multiclip_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# Failed-clip eval for the 5 sampling-mode PopArt runs trained
# on standup_walk_jump_all.zarr (project: balanced_sampling).
# Mirrors the eval_specialist_pool.py docstring defaults:
#   3 deterministic passes per clip, uniform start frame,
#   4096 envs, swish, decimation 6, categories=stand_up,walk,jump.
# Each run writes failed_clip_ids.json + eval_summary.csv into
# eval_results/balanced_sampling/<run_name>/.
# ============================================================

ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
TASK="Popart-Flat-G1-Play-v0"
WANDB_ENTITY="robot-mcrobotface"
WANDB_PROJECT="balanced_sampling"

eval_run() {
    local run_id="$1"
    local run_name="$2"
    local wandb_path="${WANDB_ENTITY}/${WANDB_PROJECT}/${run_id}"
    local output_dir="eval_results/balanced_sampling/${run_name}_${run_id}"

    echo "=========================================="
    echo "Run:     ${run_name}"
    echo "Wandb:   ${wandb_path}"
    echo "Output:  ${output_dir}"
    echo "=========================================="

    python scripts/eval_specialist_pool.py \
        --task=${TASK} \
        --wandb_path=${wandb_path} \
        --zarr_path=${ZARR_PATH} \
        --categories stand_up,walk,jump \
        --num_passes 3 \
        --start_frame_mode uniform \
        --decimation 6 \
        --activation swish \
        --num_envs 4096 \
        --output_dir ${output_dir} \
        --headless
}

# # (1) balanced — uniform cat → uniform clip → uniform frame.
# eval_run "ia5mxune" "balanced_standup_walk_jump"

# # (2) frame_uniform — global frame timeline, no category structure.
# eval_run "p6ztldda" "frame_uniform_standup_walk_jump"

# # (3) clip_adaptive — clipped-adaptive over all clips, uniform frame.
# eval_run "y253ux77" "clip_adaptive_standup_walk_jump"

# # (4) cat_adaptive_clip_uniform — adaptive cat → uniform clip in cat → uniform frame.
# eval_run "kzbl56fi" "cat_adaptive_clip_uniform_standup_walk_jump"

# (5) cat_uniform_clip_adaptive — uniform cat → adaptive clip in cat → uniform frame.
eval_run "etigz6mp" "cat_uniform_clip_adaptive_standup_walk_jump"

# ----- previous (commented-out) eval_multiclip.py invocations below -----

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/tg334ct9 \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_gravity_swish_uniform \
#     --results_name=bones_target_50hz_swish_gravcur6.7_uniform_pt3 \
#     --activation swish

# # Walk+jog eval — mirror of training run #3 (33hz, swish, uniform, no gravity curriculum)
# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/c15qko8c \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_walk_jog_33hz \
#     --results_name=bones_target_33hz_swish_uniform_walk-jog \
#     --activation swish \
#     --include_motion_types walk,jog

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/3oftz6dh \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_tracking_timetolive \
#     --results_name=tracking_target_50hz_swish_uniform_timetolive \
#     --activation swish

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/etrf9b1c \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_tracking_gravity \
#     --results_name=tracking_target_50hz_swish_uniform_gravity_12_pt2 \
#     --activation swish