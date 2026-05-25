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

# Diverse subset of clips that the baseline (ia5mxune) FAILED on, sampled
# across stand_up / walk / jump / turn / "painful" buckets from
# eval_results/dagger/balanced_ia5mxune/failed_clip_ids.json. Use these to
# visually compare baseline vs DAgger student on the actual failure set.
FAILED_MOTIONS=(
    # stand_up failures (only 2 unique in the failure set)
    "stand_up_lying_stomach_R_002__A472"
    # walk failures — mirrored + non-mirrored, hands-on-back, loop variants
    "walk_hands_on_back_start_002__A033_M"
    "walk_hands_on_back_start_002__A030_M"
    "walk_ff_loop_225_R_001__A268_M"
    # jump failures — sideways, high-jump-with-turn
    "jump_sideway_135_001__A024_M"
    "high_jump_full_turn_R_opt_1_001__A479_M"
    # turn failures — combined turn + jump motions
    "turn_high_jump_270_R_opt_1_001__A477"
    "turn_jump_360_R_001__A422_M"
    # "painful" prefix — non-canonical postures
    "painful_stand_on_walk_ff_270_R_001__A460"
)

# Helper: run play on every motion in $4..$N (or MOTIONS if none provided).
# Lets the DAgger block reuse the same policy for two different clip sets
# (diverse hand-picked + baseline-failure) without duplicating the function.
play_run() {
    local run_id="$1"
    local run_name="$2"
    local video_subdir="$3"
    shift 3
    local -a motions=("$@")
    if [ ${#motions[@]} -eq 0 ]; then
        motions=("${MOTIONS[@]}")
    fi
    local wandb_path="${WANDB_ENTITY}/${WANDB_PROJECT}/${run_id}"
    local video_dir="clip_videos/${video_subdir}/${run_name}"

    echo "=========================================="
    echo "Run:     ${run_name}"
    echo "Wandb:   ${wandb_path}"
    echo "Output:  ${video_dir}"
    echo "Motions: ${#motions[@]}"
    echo "=========================================="

    for motion in "${motions[@]}"; do
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

# # (1) frame_uniform — global frame timeline, no category structure.
# play_run "p6ztldda" \
#     "2026-05-15_17-22-00_frame_uniform_standup_walk_jump" \
#     "frame_uniform_standup_walk_jump_p6ztldda"

# # (2) balanced — uniform cat → uniform clip → uniform frame.
# play_run "ia5mxune" \
#     "2026-05-15_17-22-00_balanced_standup_walk_jump" \
#     "balanced_standup_walk_jump_ia5mxune"

# # (3) clip_adaptive — clipped-adaptive over all clips, uniform frame.
# play_run "y253ux77" \
#     "2026-05-15_17-28-52_clip_adaptive_standup_walk_jump" \
#     "clip_adaptive_standup_walk_jump_y253ux77"

# # (4) cat_adaptive_clip_uniform — adaptive cat → uniform clip in cat → uniform frame.
# play_run "kzbl56fi" \
#     "2026-05-15_17-28-52_cat_adaptive_clip_uniform_standup_walk_jump" \
#     "cat_adaptive_clip_uniform_standup_walk_jump_kzbl56fi"

# # (5) cat_uniform_clip_adaptive — uniform cat → adaptive clip in cat → uniform frame.
# play_run "etigz6mp" \
#     "2026-05-15_17-28-57_cat_uniform_clip_adaptive_standup_walk_jump" \
#     "cat_uniform_clip_adaptive_standup_walk_jump_etigz6mp"

# (6) DAgger student v1 — baseline (ia5mxune) BC'd toward specialist (7pddrm5x)
# across 100 iters × 4096 envs. Cut baseline failed-clip count from 239 → 81
# on the failure set; see eval_results/dagger_student_u8h1t7gz_vs_failed/.
# play_run "u8h1t7gz" \
#     "dagger_student_v1_ia5mxune_u8h1t7gz" \
#     "dagger_student_v1_u8h1t7gz"

# (7) DAgger student v2 — baseline (ia5mxune) BC'd toward specialist (7pddrm5x)
# across 30 iters × 4096 envs. See dagger.sh.
#
# Run twice for the DAgger student: once on the diverse hand-picked set
# (same as the 5 baseline runs above, for apples-to-apples comparison), and
# once on the baseline-failure subset (where DAgger should visibly improve).
play_run "9nit9er3" \
    "dagger_student_ia5mxune_30iter_umqm6zt1" \
    "dagger_student_30iter_umqm6zt1"

play_run "9nit9er3" \
    "dagger_student_ia5mxune_30iter_umqm6zt1_baseline_failures" \
    "dagger_student_30iter_umqm6zt1_baseline_failures" \
    "${FAILED_MOTIONS[@]}"
