#!/bin/bash
#SBATCH --job-name=dagger
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/dagger_%j.out
#SBATCH --error=logs/slurm/dagger/dagger_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Unbuffered stdout/stderr — SLURM block-buffers Python prints to file mode
# (4-8KB), which makes progress invisible until the buffer fills. Force flush
# every line so we can see [ZarrMotionLoader] / [EVAL] / per-clip logs live.
export PYTHONUNBUFFERED=1

# ============================================================
# DAgger pipeline (Track B in plan)
#
# Three sequential phases. UNCOMMENT EXACTLY ONE block at a time
# (same pattern as train_multiclip_popart.sh):
#   (1) eval baseline -> failed_clip_ids.json
#   (2) train specialist on those clips (standard PPO)
#   (2.5) sanity-check specialist on failed clips (re-run block 1 with
#         --include_clip_names_file + specialist wandb_path)
#   (3) DAgger fine-tune the student using the specialist as expert
# ============================================================

# Shared
ZARR_PATH="/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr"
CATEGORIES="stand_up,walk,jump"
WANDB_ENTITY="robot-mcrobotface"
WANDB_PROJECT="balanced_sampling"

# Pick whichever baseline wins play_popart.sh. Defaults to balanced
# (ia5mxune) as the strongest baseline so far. To swap: edit BASELINE_RUN_ID
# and BASELINE_NAME below.
BASELINE_RUN_ID="ia5mxune"
BASELINE_NAME="balanced_${BASELINE_RUN_ID}"
EVAL_OUT_DIR="eval_results/dagger/${BASELINE_NAME}"
FAILED_CLIPS_JSON="${EVAL_OUT_DIR}/failed_clip_ids.json"

# Set this AFTER step 2 finishes — copy the specialist's run id from wandb.
SPECIALIST_RUN_ID="7pddrm5x" # 2026-05-17_15-19-14_specialist_failed_clips_ia5mxune_balanced_standup_walk_jump

# Set this AFTER step 3 finishes — copy the DAgger student's wandb run id.
DAGGER_RUN_ID="9nit9er3" # e.g. 1ylhof7f (v0_midexpert), or v1 once expert converges

# ============================================================
# (1) Identify failed clips by running 3 deterministic passes per clip.
#     Outputs:
#       eval_results/<baseline>/failed_clip_ids.json
#       eval_results/<baseline>/eval_summary.csv
# ============================================================
# python scripts/eval_specialist_pool.py \
#     --task=Popart-Flat-G1-Play-v0 \
#     --wandb_path=${WANDB_ENTITY}/${WANDB_PROJECT}/${BASELINE_RUN_ID} \
#     --zarr_path=${ZARR_PATH} \
#     --categories ${CATEGORIES} \
#     --num_passes 3 \
#     --start_frame_mode uniform \
#     --max_steps_per_pass 500 \
#     --decimation 6 --activation swish \
#     --popart off \
#     --num_envs 4096 \
#     --output_dir ${EVAL_OUT_DIR}
#
# NOTE: the "_all" zarr has 14,454 clips → 14,454 × 3 = 43,362 (clip, pass)
# units. The eval is now BATCHED across num_envs envs — each env is pinned to
# a distinct unit via a per-env clip-id tensor, and we track per-env first
# failure (auto-reset after termination is ignored). At num_envs=4096, that
# packs into ~11 batches; with the 500-step cap each batch is bounded so
# wall-clock is on the order of tens of minutes rather than days.
# If 4096 OOMs on the GPU, drop to 2048 — same code path, just more batches.

# ============================================================
# (2) Train specialist on failed clips. Uses existing train_bones.py with
#     the new --include_clip_names_file flag. frame_uniform sampling is
#     safest when the include filter leaves some categories empty (balanced
#     requires every cat to have >=1 clip).
# ============================================================
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --num_envs 2048 --headless \
#     --logger wandb --log_project_name ${WANDB_PROJECT} \
#     --run_name specialist_failed_clips_${BASELINE_RUN_ID} \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories ${CATEGORIES} \
#     --sampling_mode frame_uniform \
#     --popart off \
#     --include_clip_names_file ${FAILED_CLIPS_JSON}

# ============================================================
# (2.5) Sanity check: specialist's termination rate on failed clips MUST
#       be meaningfully lower than baseline's, else DAgger has nothing to
#       distill. Re-run step 1 against the specialist, restricted to the
#       failed-clip set. Compare the two eval_summary.csv files.
# ============================================================
# python scripts/eval_specialist_pool.py \
#     --task=Popart-Flat-G1-Play-v0 \
#     --wandb_path=${WANDB_ENTITY}/${WANDB_PROJECT}/${SPECIALIST_RUN_ID} \
#     --zarr_path=${ZARR_PATH} \
#     --categories ${CATEGORIES} \
#     --include_clip_names_file ${FAILED_CLIPS_JSON} \
#     --num_passes 3 \
#     --start_frame_mode uniform \
#     --max_steps_per_pass 500 \
#     --decimation 6 --activation swish \
#     --popart off \
#     --num_envs 4096 \
#     --output_dir eval_results/specialist_${SPECIALIST_RUN_ID}_vs_failed/

# OLD ============================================================
# (3) DAgger fine-tune. Student = baseline; expert = specialist. Both
#     loaded via wandb. Rolls out the student on the failed-clip env,
#     labels visited states with the expert, BC-trains the student.
# ============================================================
# python scripts/rsl_rl/dagger.py \
#     --task=Popart-Flat-G1-v0 \
#     --student_wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${BASELINE_RUN_ID} \
#     --expert_wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${SPECIALIST_RUN_ID} \
#     --zarr_path=${ZARR_PATH} \
#     --include_clip_names_file ${FAILED_CLIPS_JSON} \
#     --categories ${CATEGORIES} \
#     --decimation 6 --activation swish \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${WANDB_PROJECT} \
#     --run_name dagger_student_v1_${BASELINE_RUN_ID}_30iter \
#     --sampling_mode frame_uniform \
#     --popart off \
#     --n_iters 30 --rollout_steps 200 --bc_epochs 5 \
#     --lr 1e-4 --batch_size 4096 --buffer_cap 1000000 \
#     --save_every 1

# NEW ============================================================
# TWO-POOL DAgger — anti-forgetting variant.
#     Loads FULL zarr (no include_clip_names restriction on the env).
#     30% of envs are pinned to sample from the failed-clip set, labeled by
#     the specialist; 70% sample from easy clips, labeled by a frozen copy
#     of the baseline (= the student's own init weights). Goal: preserve
#     baseline behavior on easy clips while distilling specialist wins on
#     failed clips.
# ============================================================
# python scripts/rsl_rl/dagger.py \
#     --task=Popart-Flat-G1-v0 \
#     --student_wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${BASELINE_RUN_ID} \
#     --expert_wandb=${WANDB_ENTITY}/${WANDB_PROJECT}/${SPECIALIST_RUN_ID} \
#     --zarr_path=${ZARR_PATH} \
#     --include_clip_names_file ${FAILED_CLIPS_JSON} \
#     --two_pool --failed_pool_frac 0.3 \
#     --categories ${CATEGORIES} \
#     --decimation 6 --activation swish \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name ${WANDB_PROJECT} \
#     --run_name dagger_student_${BASELINE_RUN_ID}_twopool_${SPECIALIST_RUN_ID} \
#     --sampling_mode frame_uniform \
#     --popart off \
#     --n_iters 30 --rollout_steps 200 --bc_epochs 5 \
#     --lr 1e-4 --batch_size 4096 --buffer_cap 1000000 \
#     --save_every 1





# ============================================================
# (4) Final eval: the DAgger student vs the baseline failure set.
#     This is the number that decides whether DAgger worked.
#     Diff this eval_summary.csv against:
#       - eval_results/dagger/${BASELINE_NAME}/eval_summary.csv (baseline)
#       - eval_results/specialist_${SPECIALIST_RUN_ID}_vs_failed/eval_summary.csv
#     A successful DAgger run: termination rate on failed clips drops below
#     baseline's. A great DAgger run: also matches specialist's rate.
#     Forgetting check: separately eval on the full zarr (no
#     --include_clip_names_file) and confirm overall failed-clip count is
#     not much higher than baseline's 239.
# ============================================================
python scripts/eval_specialist_pool.py \
    --task=Popart-Flat-G1-Play-v0 \
    --wandb_path=${WANDB_ENTITY}/${WANDB_PROJECT}/${DAGGER_RUN_ID} \
    --zarr_path=${ZARR_PATH} \
    --categories ${CATEGORIES} \
    --include_clip_names_file ${FAILED_CLIPS_JSON} \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation 6 --activation swish \
    --popart off \
    --num_envs 4096 \
    --output_dir eval_results/dagger_student_${DAGGER_RUN_ID}_vs_failed/


# ============================================================
# (5) FORGETTING CHECK: same as block 4 but WITHOUT
#     --include_clip_names_file, so the DAgger student is evaluated on
#     ALL clips (not just the baseline failure set). The DAgger student's
#     failed-clip count here should be in the same ballpark as baseline's
#     239. Much higher = policy "forgot" easy clips while fitting the
#     failure set, and v2 should mix easy clips into the rollout dist.
# ============================================================
python scripts/eval_specialist_pool.py \
    --task=Popart-Flat-G1-Play-v0 \
    --wandb_path=${WANDB_ENTITY}/${WANDB_PROJECT}/${DAGGER_RUN_ID} \
    --zarr_path=${ZARR_PATH} \
    --categories ${CATEGORIES} \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation 6 --activation swish \
    --popart off \
    --num_envs 4096 \
    --output_dir eval_results/dagger_student_${DAGGER_RUN_ID}_full_zarr/