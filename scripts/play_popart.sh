#!/bin/bash
#SBATCH --job-name=play_popart
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/play_popart_%j.out
#SBATCH --error=logs/slurm/play_popart_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# Roll out the 5 sampling-mode runs trained from
# scripts/train_multiclip_popart.sh on a diverse hand-picked
# subset of standup_walk_jump_all.zarr.
#
# All 5 runs share the same dataset, task, categories, decimation,
# and activation — only the trained policy weights differ. Each run
# writes its videos into clip_videos/<run_subdir>/<run_name>/.
#
# Comment out any of the 5 blocks at the bottom to skip that run.
# ============================================================

# ===================== Shared configuration =====================
ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
TASK="Popart-Flat-G1-Play-v0"
DECIMATION=6
ACTIVATION="swish"
CATEGORIES="stand_up,walk,jump"
WANDB_ENTITY="robot-mcrobotface"
WANDB_PROJECT="balanced_sampling"
# ================================================================

# Hand-picked diverse subset across the three categories.
# Includes jump_and_sit_R_001__A533 per request.
MOTIONS=(
    # stand_up variants — different starting postures
    "stand_up_lying_R_002__A472"
    "stand_up_lying_side_R_002__A472"
    "stand_up_lying_stomach_R_002__A472"
    # walk variants — forward loop, fast turn, sideways, backward, turn-then-walk
    "walk_ff_loop_360_005__A059"
    "walk_ff_loop_180_R_very_fast_001__A448"
    "walk_sideway_090_loop_001__A024"
    "walk_backward_stop_004__A022"
    "turn_start_walk_0090_005__A024"
    # jump variants — basic, backward, high, jump+sit (requested), heavy landing
    "Jump_002__A017"
    "jump_backward_002__A021"
    "high_jump_R_opt_2_001__A476"
    "jump_and_sit_R_001__A533"
    "jump_and_land_heavy_001__A001"
)

# Helper: run play on every motion in MOTIONS for a single trained policy.
play_run() {
    local run_id="$1"
    local run_name="$2"
    local video_subdir="$3"
    local wandb_path="${WANDB_ENTITY}/${WANDB_PROJECT}/${run_id}"
    local video_dir="clip_videos/${video_subdir}/${run_name}"

    echo "=========================================="
    echo "Run:     ${run_name}"
    echo "Wandb:   ${wandb_path}"
    echo "Output:  ${video_dir}"
    echo "Motions: ${#MOTIONS[@]}"
    echo "=========================================="

    for motion in "${MOTIONS[@]}"; do
        python scripts/rsl_rl/play_bones_clip.py \
            --task=${TASK} \
            --zarr_path=${ZARR_PATH} \
            --wandb_path=${wandb_path} \
            --clip_name="${motion}" \
            --video_dir=${video_dir} \
            --activation=${ACTIVATION} \
            --decimation ${DECIMATION} \
            --num_envs=1 \
            --headless \
            --video \
            --categories ${CATEGORIES}
    done
}

# (1) frame_uniform — global frame timeline, no category structure.
play_run "p6ztldda" \
    "2026-05-15_17-22-00_frame_uniform_standup_walk_jump" \
    "frame_uniform_standup_walk_jump_p6ztldda"

# (2) balanced — uniform cat → uniform clip → uniform frame.
play_run "ia5mxune" \
    "2026-05-15_17-22-00_balanced_standup_walk_jump" \
    "balanced_standup_walk_jump_ia5mxune"

# (3) clip_adaptive — clipped-adaptive over all clips, uniform frame.
play_run "y253ux77" \
    "2026-05-15_17-28-52_clip_adaptive_standup_walk_jump" \
    "clip_adaptive_standup_walk_jump_y253ux77"

# (4) cat_adaptive_clip_uniform — adaptive cat → uniform clip in cat → uniform frame.
play_run "kzbl56fi" \
    "2026-05-15_17-28-52_cat_adaptive_clip_uniform_standup_walk_jump" \
    "cat_adaptive_clip_uniform_standup_walk_jump_kzbl56fi"

# (5) cat_uniform_clip_adaptive — uniform cat → adaptive clip in cat → uniform frame.
play_run "etigz6mp" \
    "2026-05-15_17-28-57_cat_uniform_clip_adaptive_standup_walk_jump" \
    "cat_uniform_clip_adaptive_standup_walk_jump_etigz6mp"
