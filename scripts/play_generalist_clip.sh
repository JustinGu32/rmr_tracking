#!/bin/bash
#SBATCH --job-name=play_generalist
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/play_generalist/play_generalist_%j.out
#SBATCH --error=logs/slurm/play_generalist/play_generalist_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# ============================================================
# Roll out trained generalist policies on a hand-picked subset of clips.
#
# Uses the Generalist-Flat-G1-Play-v0 task. Just run it:
#     bash scripts/play_generalist_clip.sh
# It loops over the RUNS table below (run id | categories | run name); each
# run's --categories MUST match how it was trained. Edit the constants below
# to change history, decimation, motions, etc.
# ============================================================

ZARR_PATH=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr
TASK=Generalist-Flat-G1-Play-v0
DECIMATION=6
ACTIVATION=swish
# Must match the runs' training --history_length (0 = none). The 2026-05-31
# adaptive-sampling sweep used the default (0); older generalist_v1 runs used 10.
HISTORY_LENGTH=0
WANDB_ENTITY=robot-mcrobotface
WANDB_PROJECT=generalist
VIDEO_SUBDIR=generalist

# Path to the K=12 VAE clusters JSON, used by the VAE-categorized runs.
VAE_K12_JSON=logs/motion_vae/v1/clip_clusters_k12.json
# Path to the K=16 VAE clusters JSON, used by the (older) k=16 VAE run (ki4gtu0j).
VAE_K16_JSON=logs/motion_vae/v1/clip_clusters_k16.json

# Runs to roll out: "RUN_ID | EXTRA_ARGS | RUN_NAME".
# EXTRA_ARGS is the categorizer flag(s) for this run; it MUST match the run's
# training-time config so the checkpoint loads against the same K-head critic
# and category-conditioned policy. Two flavors:
#   keyword:  --categories stand_up,walk,jump,...
#   VAE:      --categorizer_mode latent_kmeans --latent_centroids_path <json>
RUNS=(
    # # historical reference (history_length=10; will mismatch if HISTORY_LENGTH=0 above)
    # "2r802tqg | --categories stand_up,walk,jump,run,jog,crouch,turn,idle | 2r802tqg_allMotions"
    # "7bfrr5y7 | --categories stand_up,walk,jump | 7bfrr5y7_weak"

    # ── 2026-05-31 adaptive-sampling sweep (history_length=0) ──────────────
    # VAE-categorized (k=12)
    "raykysnl | --categorizer_mode latent_kmeans --latent_centroids_path ${VAE_K12_JSON} | raykysnl_VAE-cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug"
    "klkm63s6 | --categorizer_mode latent_kmeans --latent_centroids_path ${VAE_K12_JSON} | klkm63s6_VAE-cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug"
    # VAE-categorized (older k=16)
    "ki4gtu0j | --categorizer_mode latent_kmeans --latent_centroids_path ${VAE_K16_JSON}  | ki4gtu0j_vae-cat_blend_clip_uniform-sym_aug"
    # Keyword 8-cat (allMotions)
    "329tiqlx | --categories stand_up,walk,jump,run,jog,crouch,turn,idle | 329tiqlx_cat_blend_clip_uniform-catProb0.5-allMotions-sym_aug"
    "8i82390u | --categories stand_up,walk,jump,run,jog,crouch,turn,idle | 8i82390u_cat_adaptive_clip_adaptive-catProb0.5-clipProb0.75-allMotions-sym_aug"
    "yr974gsu | --categories stand_up,walk,jump,run,jog,crouch,turn,idle | yr974gsu_cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions-sym_aug"
    "n5an93th | --categories stand_up,walk,jump,run,jog,crouch,turn,idle | n5an93th_cat_blend_clip_uniform-catRatio0.5-clipRatio0.6-allMotions"
    
    # Keyword 3-cat (stand_up,walk,jump)
    "8pmudt9g | --categories stand_up,walk,jump | 8pmudt9g_cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump"
    "qq7w57b5 | --categories stand_up,walk,jump | qq7w57b5_cat_adaptive_clip_adaptive-catRatio0.5-clipRatio0.6-stand_up,walk,jump-sym_aug"
    "n1p6pgca | --categories stand_up,walk,jump | n1p6pgca_cat_blend_clip_uniform-catRatio0.5-stand_up,walk,jump-sym_aug"
)

MOTIONS=(
    "stand_up_lying_R_002__A472"
    "stand_up_lying_side_R_002__A472"
    "stand_up_lying_stomach_R_002__A472"
    "walk_ff_loop_360_005__A059"
    "walk_ff_loop_180_R_very_fast_001__A448"
    "walk_backward_stop_004__A022"
    "turn_start_walk_0090_005__A024"
    "walk_ff_stop_270_R_very_slow_001__A444"
    "walk_arc_cw_loop_R_very_slow_001__A444"
    "Jump_002__A017"
    "jump_backward_002__A021"
    "high_jump_R_opt_2_001__A476"
    "jump_and_sit_R_001__A533"
    "jump_and_land_heavy_001__A001"

    # "run_loop_180_R_003__A326"

    # "jog_arc_cw_loop_001__A047"
    # "jog_ff_stop_360_001__A050"
    # "jog_sideway_090_start_001__A022"
    # "jog_backward_stop_001__A022"

    # "crouch_ff_loop_180_R_101__A125"
    # "crouch_ff_loop_270_R_002__A194"

    # "turn_jog_360_001__A049"
    # "turn_walk_270_001__A047"
    # "turn_jump_360_002__A046"

    # "idle_loop_001__A021"
    # "idle_hands_on_back_loop_001__A031"
    # "idle_turn_360_001__A047"

    # "sit_on_heels_start_004__A021"
    # "kneeling_start_001__A021"
    # "sitting_legs_bend_arms_back_loop_001__A030"

    # "mohak_forward_loop_001__A033"
    # "mohak_backward_loop_001__A030"

    # "walk_sideway_045_loop_001__A022"
    # "jog_sideway_135_stop_001__A022"

    # "on_the_edge_001__A097"
)

for run in "${RUNS[@]}"; do
    IFS='|' read -r RUN_ID EXTRA_ARGS RUN_NAME <<< "${run}"
    # trim surrounding whitespace from the split fields
    RUN_ID="$(echo "${RUN_ID}" | xargs)"
    EXTRA_ARGS="$(echo "${EXTRA_ARGS}" | xargs)"
    RUN_NAME="$(echo "${RUN_NAME}" | xargs)"

    WANDB_PATH="${WANDB_ENTITY}/${WANDB_PROJECT}/${RUN_ID}"
    VIDEO_DIR="clip_videos/${VIDEO_SUBDIR}/${RUN_NAME}"

    echo "=========================================="
    echo "Run:        ${RUN_NAME}"
    echo "Wandb:      ${WANDB_PATH}"
    echo "Extra args: ${EXTRA_ARGS}"
    echo "History:    ${HISTORY_LENGTH}"
    echo "Output:     ${VIDEO_DIR}"
    echo "=========================================="

    for motion in "${MOTIONS[@]}"; do
        python scripts/rsl_rl/play_bones_clip.py \
            --task=${TASK} \
            --zarr_path=${ZARR_PATH} \
            --wandb_path=${WANDB_PATH} \
            --clip_name="${motion}" \
            --video_dir=${VIDEO_DIR} \
            --activation=${ACTIVATION} \
            --decimation ${DECIMATION} \
            --history_length ${HISTORY_LENGTH} \
            --num_envs=1 \
            --headless \
            --video \
            ${EXTRA_ARGS}
    done
done

echo "[play_generalist] DONE."
