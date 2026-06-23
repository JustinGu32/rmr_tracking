#!/bin/bash
#SBATCH --job-name=play_jumps
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/play_jumps_%j.out
#SBATCH --error=logs/slurm/play_jumps_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# Roll out the jump-exploration runs (scripts/train_multiclip_popart_jumps.sh)
# on a diverse hand-picked subset of standup_walk_jump_all.zarr.
#
# All runs share the same dataset, task, categories, decimation, and
# activation — only the trained policy weights differ. Each run writes its
# videos into clip_videos/jumps/<run_name>/ (run_name = wandb run name minus
# the leading date-time stamp).
#
# NOTE on the 4h limit: each play_run restarts the sim once PER motion
# (~13 sim starts/run). Running all 13 arms in one job will exceed
# --time=4:00:00 — comment out the arms you don't need per submission, or
# split across several sbatch jobs. Arms are ordered high→low priority so the
# most informative videos render first.
# ============================================================

# ===================== Shared configuration =====================
ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
TASK="Popart-Flat-G1-Play-v0"
DECIMATION=6
ACTIVATION="swish"
CATEGORIES="stand_up,walk,jump"
WANDB_ENTITY="robot-mcrobotface"
# WANDB_PROJECT="jump_exploration"
WANDB_PROJECT="balanced_sampling"
VIDEO_SUBDIR="jumps"
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
    "walk_backward_stop_004__A022"
    "turn_start_walk_0090_005__A024"
    "walk_ff_stop_270_R_very_slow_001__A444"
    "walk_arc_cw_loop_R_very_slow_001__A444"
    # jump variants — basic, backward, high, jump+sit (requested), heavy landing
    "Jump_002__A017"
    "jump_backward_002__A021"
    "high_jump_R_opt_2_001__A476"
    "jump_and_sit_R_001__A533"
    "jump_and_land_heavy_001__A001"
)

# Helper: run play on every motion in $4..$N (or MOTIONS if none provided).
play_run() {
    local run_id="$1"
    local run_name="$2"
    local video_subdir="$3"
    shift 3
    local -a motions=("$@")
    if [ ${#motions[@]} -eq 0 ]; then
        motions=("${MOTIONS[@]}")
    fi
    # Set TERRAIN_NOISE=1 before calling play_run to roll out with the matching
    # random-bump terrain (must match the run's training-time --terrain_noise).
    local terrain_noise_flag=""
    if [ "${TERRAIN_NOISE:-0}" = "1" ]; then
        terrain_noise_flag="--terrain_noise"
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
            --categories ${CATEGORIES} \
            ${terrain_noise_flag}
    done
}

# ── Combined arms ────────────────────────────────────────────────────────────
# play_run "po5zy237" "combo_blend_sampling_jump_term_airborne_bonus_rsi" "${VIDEO_SUBDIR}"
# play_run "3gu7le42" "combo_all_reward_shaping_plus_rsi" "${VIDEO_SUBDIR}"

# # ── Reward / RSI arms ─────────────────────────────────────────────────────────
# play_run "0wzcfzxv" "rsi_start_airborne_30pct" "${VIDEO_SUBDIR}"
# play_run "dk4pulzh" "rew_airborne_penalty_plus_flight_bonus" "${VIDEO_SUBDIR}"
# play_run "xnvsc5dk" "rew_airborne_contact_penalty" "${VIDEO_SUBDIR}"
# play_run "31psdhuv" "rew_contact_phase_match" "${VIDEO_SUBDIR}"
# play_run "x654mdmv" "rew_pelvis_below_reference_z_penalty" "${VIDEO_SUBDIR}"
# play_run "sd3lvqav" "rew_foot_below_reference_penalty" "${VIDEO_SUBDIR}"

# # ── Adaptive-sampling arms (2x2 + blend) ──────────────────────────────────────
# play_run "gea4n32f" "adap_fail_error_blend_with_jump_termination" "${VIDEO_SUBDIR}"
# play_run "lmhcxqha" "adap_catfail_with_jump_termination" "${VIDEO_SUBDIR}"
# play_run "bv6jm7nc" "adap_trackerror_with_jump_termination" "${VIDEO_SUBDIR}"
# play_run "mgobz5o0" "adap_trackerror_only" "${VIDEO_SUBDIR}"
# play_run "kb8ytbig" "adap_catfail_only" "${VIDEO_SUBDIR}"

TERRAIN_NOISE=1 play_run "w68bc8aj" "terrainNoise" "terrainNoiseEnv"