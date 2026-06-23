#!/bin/bash
#SBATCH --job-name=hnm_round
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/hnm/round_%j.out
#SBATCH --error=logs/slurm/hnm/round_%j.err

# ===========================================================================
# Hard-negative mining — one round (Phase 3).
#
# Runs five phases SEQUENTIALLY in one sbatch job (so wandb run-id lookups
# happen in-process):
#   1. eval current generalist on full zarr → failed_clip_ids.json
#   2. train PRIVILEGED EXPERT (--expert_mode) on the failed-clip subset
#   3. DAgger distill: privileged expert → generalist (two-pool, anti-forget)
#   4. eval distilled generalist (full zarr + failed-set, two evals)
#   5. write per-round summary to eval_results/hnm/history.csv; if the failed
#      count dropped by >=50% relative to last round, submit the NEXT round
#      via sbatch with the new baseline run-id; else stop.
#
# Usage:
#   sbatch scripts/run_hnm_round.sh
#   (with env vars set in scripts/run_hnm_loop.sh, or via --export=)
#
# Required env vars:
#   HNM_ROUND               int, 1-indexed round number
#   HNM_BASELINE_RUN_ID     wandb run id of the current best generalist
#   HNM_ZARR_PATH           absolute path to zarr motion store
#   HNM_CATEGORIES          comma-separated categories (e.g. "stand_up,walk,jump")
#   HNM_WANDB_PROJECT       wandb project name (e.g. "balanced_sampling")
#   HNM_WANDB_ENTITY        wandb entity (e.g. "robot-mcrobotface")
#
# Optional env vars (with defaults):
#   HNM_OUT_BASE_DIR        default: eval_results/hnm
#   HNM_NUM_ENVS            default: 4096
#   HNM_DECIMATION          default: 6
#   HNM_ACTIVATION          default: swish
#   HNM_EXPERT_ITERS        default: 5000
#   HNM_DAGGER_ITERS        default: 30
#   HNM_DAGGER_ROLLOUT_STEPS default: 200
#   HNM_RELATIVE_STOP_THRESH default: 0.5  (stop when new/old > 0.5)
#   HNM_SAMPLING_MODE       default: cat_adaptive_clip_adaptive  (generalist task)
#   HNM_SYM_AUG             default: 1  (0 to disable)
#   HNM_HISTORY_LENGTH      default: 10
#   HNM_MAX_ROUNDS          default: 6
# ===========================================================================

set -euo pipefail

mkdir -p logs/slurm/hnm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# ── Required ────────────────────────────────────────────────────────────────
: "${HNM_ROUND:?must set HNM_ROUND}"
: "${HNM_BASELINE_RUN_ID:?must set HNM_BASELINE_RUN_ID}"
: "${HNM_ZARR_PATH:?must set HNM_ZARR_PATH}"
: "${HNM_CATEGORIES:?must set HNM_CATEGORIES}"
: "${HNM_WANDB_PROJECT:?must set HNM_WANDB_PROJECT}"
: "${HNM_WANDB_ENTITY:?must set HNM_WANDB_ENTITY}"

# ── Defaults ────────────────────────────────────────────────────────────────
HNM_OUT_BASE_DIR="${HNM_OUT_BASE_DIR:-eval_results/hnm}"
HNM_NUM_ENVS="${HNM_NUM_ENVS:-4096}"
HNM_DECIMATION="${HNM_DECIMATION:-6}"
HNM_ACTIVATION="${HNM_ACTIVATION:-swish}"
HNM_EXPERT_ITERS="${HNM_EXPERT_ITERS:-5000}"
HNM_DAGGER_ITERS="${HNM_DAGGER_ITERS:-30}"
HNM_DAGGER_ROLLOUT_STEPS="${HNM_DAGGER_ROLLOUT_STEPS:-200}"
HNM_RELATIVE_STOP_THRESH="${HNM_RELATIVE_STOP_THRESH:-0.5}"
HNM_SAMPLING_MODE="${HNM_SAMPLING_MODE:-cat_adaptive_clip_adaptive}"
HNM_SYM_AUG="${HNM_SYM_AUG:-1}"
HNM_HISTORY_LENGTH="${HNM_HISTORY_LENGTH:-10}"
HNM_MAX_ROUNDS="${HNM_MAX_ROUNDS:-6}"

ROUND_DIR="${HNM_OUT_BASE_DIR}/round${HNM_ROUND}"
mkdir -p "${ROUND_DIR}"
HISTORY_CSV="${HNM_OUT_BASE_DIR}/history.csv"
[ -f "${HISTORY_CSV}" ] || echo "round,baseline_run_id,expert_run_id,distilled_run_id,failed_count_pre,failed_count_post,relative_reduction,decision" > "${HISTORY_CSV}"

# Sym-aug flag passthrough.
SYM_AUG_FLAG=""
if [ "${HNM_SYM_AUG}" = "1" ]; then
    SYM_AUG_FLAG="--symmetric_augment"
fi

# Wandb run name conventions for this round (used both as --run_name AND as
# the lookup key when fetching the auto-assigned run_id afterward).
EXPERT_RUN_NAME="hnm_r${HNM_ROUND}_expert_${HNM_BASELINE_RUN_ID}"
DAGGER_RUN_NAME="hnm_r${HNM_ROUND}_dagger_${HNM_BASELINE_RUN_ID}"

echo "================================================================"
echo "HNM round ${HNM_ROUND} | baseline=${HNM_BASELINE_RUN_ID}"
echo "  output dir:  ${ROUND_DIR}"
echo "  zarr:        ${HNM_ZARR_PATH}"
echo "  categories:  ${HNM_CATEGORIES}"
echo "  sampling:    ${HNM_SAMPLING_MODE}"
echo "  sym aug:     ${HNM_SYM_AUG} (${SYM_AUG_FLAG})"
echo "  history len: ${HNM_HISTORY_LENGTH}"
echo "================================================================"

# ── 1. Eval baseline → failed_clip_ids.json ──────────────────────────────────
EVAL_BASELINE_DIR="${ROUND_DIR}/eval_baseline"
echo "[hnm r${HNM_ROUND}] STEP 1: eval baseline ${HNM_BASELINE_RUN_ID}"
python scripts/eval_specialist_pool.py \
    --task=Generalist-Flat-G1-Play-v0 \
    --wandb_path="${HNM_WANDB_ENTITY}/${HNM_WANDB_PROJECT}/${HNM_BASELINE_RUN_ID}" \
    --zarr_path="${HNM_ZARR_PATH}" \
    --categories "${HNM_CATEGORIES}" \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation "${HNM_DECIMATION}" --activation "${HNM_ACTIVATION}" \
    --popart off \
    --num_envs "${HNM_NUM_ENVS}" \
    --output_dir "${EVAL_BASELINE_DIR}"

FAILED_JSON="${EVAL_BASELINE_DIR}/failed_clip_ids.json"
if [ ! -f "${FAILED_JSON}" ]; then
    echo "[hnm r${HNM_ROUND}] ERROR: baseline eval did not produce ${FAILED_JSON}"
    exit 1
fi
FAILED_COUNT_PRE=$(python -c "import json; print(len(json.load(open('${FAILED_JSON}'))))")
echo "[hnm r${HNM_ROUND}] baseline failed clips: ${FAILED_COUNT_PRE}"

# ── 2. Train privileged expert on failed clips ─────────────────────────────
echo "[hnm r${HNM_ROUND}] STEP 2: train expert (${HNM_EXPERT_ITERS} iters) on ${FAILED_COUNT_PRE} clips"
python scripts/rsl_rl/train_bones.py \
    --task=Generalist-Flat-G1-v0 \
    --zarr_path="${HNM_ZARR_PATH}" \
    --num_envs "${HNM_NUM_ENVS}" --headless \
    --logger wandb --log_project_name "${HNM_WANDB_PROJECT}" \
    --run_name "${EXPERT_RUN_NAME}" \
    --decimation "${HNM_DECIMATION}" --sampling uniform --activation "${HNM_ACTIVATION}" \
    --categories "${HNM_CATEGORIES}" --popart off \
    --sampling_mode frame_uniform \
    --expert_mode \
    --include_clip_names_file "${FAILED_JSON}" \
    --max_iterations "${HNM_EXPERT_ITERS}" \
    ${SYM_AUG_FLAG}

EXPERT_RUN_ID=$(python scripts/hnm_lookup_wandb_run.py "${HNM_WANDB_ENTITY}" "${HNM_WANDB_PROJECT}" "${EXPERT_RUN_NAME}")
echo "[hnm r${HNM_ROUND}] expert run_id: ${EXPERT_RUN_ID}"
echo "${EXPERT_RUN_ID}" > "${ROUND_DIR}/expert_run_id.txt"

# ── 3. DAgger distill expert → generalist (privileged, two-pool) ───────────
echo "[hnm r${HNM_ROUND}] STEP 3: DAgger distill expert → student"
python scripts/rsl_rl/dagger.py \
    --task=Generalist-Flat-G1-v0 \
    --student_wandb="${HNM_WANDB_ENTITY}/${HNM_WANDB_PROJECT}/${HNM_BASELINE_RUN_ID}" \
    --expert_wandb="${HNM_WANDB_ENTITY}/${HNM_WANDB_PROJECT}/${EXPERT_RUN_ID}" \
    --zarr_path="${HNM_ZARR_PATH}" \
    --include_clip_names_file "${FAILED_JSON}" \
    --two_pool --failed_pool_frac 0.3 \
    --categories "${HNM_CATEGORIES}" \
    --expert_obs_group expert \
    --decimation "${HNM_DECIMATION}" --activation "${HNM_ACTIVATION}" \
    --num_envs "${HNM_NUM_ENVS}" --headless \
    --logger wandb --log_project_name "${HNM_WANDB_PROJECT}" \
    --run_name "${DAGGER_RUN_NAME}" \
    --sampling_mode frame_uniform \
    --popart off \
    --n_iters "${HNM_DAGGER_ITERS}" --rollout_steps "${HNM_DAGGER_ROLLOUT_STEPS}" --bc_epochs 5 \
    --lr 1e-4 --batch_size 4096 --buffer_cap 1000000 \
    --save_every 1 \
    ${SYM_AUG_FLAG}

DISTILLED_RUN_ID=$(python scripts/hnm_lookup_wandb_run.py "${HNM_WANDB_ENTITY}" "${HNM_WANDB_PROJECT}" "${DAGGER_RUN_NAME}")
echo "[hnm r${HNM_ROUND}] distilled run_id: ${DISTILLED_RUN_ID}"
echo "${DISTILLED_RUN_ID}" > "${ROUND_DIR}/distilled_run_id.txt"

# ── 4. Eval distilled generalist (full zarr + failed set) ──────────────────
EVAL_DISTILLED_DIR_FULL="${ROUND_DIR}/eval_distilled_full"
EVAL_DISTILLED_DIR_FAILED="${ROUND_DIR}/eval_distilled_failed"

echo "[hnm r${HNM_ROUND}] STEP 4a: eval distilled (FULL zarr)"
python scripts/eval_specialist_pool.py \
    --task=Generalist-Flat-G1-Play-v0 \
    --wandb_path="${HNM_WANDB_ENTITY}/${HNM_WANDB_PROJECT}/${DISTILLED_RUN_ID}" \
    --zarr_path="${HNM_ZARR_PATH}" \
    --categories "${HNM_CATEGORIES}" \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation "${HNM_DECIMATION}" --activation "${HNM_ACTIVATION}" \
    --popart off \
    --num_envs "${HNM_NUM_ENVS}" \
    --output_dir "${EVAL_DISTILLED_DIR_FULL}"

echo "[hnm r${HNM_ROUND}] STEP 4b: eval distilled (restricted to baseline failed set)"
python scripts/eval_specialist_pool.py \
    --task=Generalist-Flat-G1-Play-v0 \
    --wandb_path="${HNM_WANDB_ENTITY}/${HNM_WANDB_PROJECT}/${DISTILLED_RUN_ID}" \
    --zarr_path="${HNM_ZARR_PATH}" \
    --categories "${HNM_CATEGORIES}" \
    --include_clip_names_file "${FAILED_JSON}" \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation "${HNM_DECIMATION}" --activation "${HNM_ACTIVATION}" \
    --popart off \
    --num_envs "${HNM_NUM_ENVS}" \
    --output_dir "${EVAL_DISTILLED_DIR_FAILED}"

FAILED_COUNT_POST=$(python -c "import json; print(len(json.load(open('${EVAL_DISTILLED_DIR_FULL}/failed_clip_ids.json'))))")
echo "[hnm r${HNM_ROUND}] post-DAgger failed clips (full zarr): ${FAILED_COUNT_POST}"

# ── 5. Record + maybe submit next round ────────────────────────────────────
RELATIVE_REDUCTION="0.0"
DECISION="CONTINUE"
if [ "${FAILED_COUNT_PRE}" -gt 0 ]; then
    RELATIVE_REDUCTION=$(python -c "print(round(1.0 - ${FAILED_COUNT_POST}/${FAILED_COUNT_PRE}, 4))")
fi
RATIO=$(python -c "print(${FAILED_COUNT_POST}/${FAILED_COUNT_PRE} if ${FAILED_COUNT_PRE}>0 else 1.0)")
# Stop if (1) we hit the max-round budget, or (2) the relative reduction is
# below the configured threshold (i.e. new/old > thresh → not enough progress).
if [ "${HNM_ROUND}" -ge "${HNM_MAX_ROUNDS}" ]; then
    DECISION="STOP_MAX_ROUNDS"
elif python -c "import sys; sys.exit(0 if ${RATIO} > ${HNM_RELATIVE_STOP_THRESH} else 1)"; then
    DECISION="STOP_NO_PROGRESS"
fi

echo "${HNM_ROUND},${HNM_BASELINE_RUN_ID},${EXPERT_RUN_ID},${DISTILLED_RUN_ID},${FAILED_COUNT_PRE},${FAILED_COUNT_POST},${RELATIVE_REDUCTION},${DECISION}" >> "${HISTORY_CSV}"
echo "[hnm r${HNM_ROUND}] history.csv row appended: ${DECISION}"

if [ "${DECISION}" = "CONTINUE" ]; then
    NEXT_ROUND=$(( HNM_ROUND + 1 ))
    echo "[hnm r${HNM_ROUND}] submitting next round (r${NEXT_ROUND}) with baseline=${DISTILLED_RUN_ID}"
    # Pass everything through with --export so the next round inherits the same config.
    sbatch --export=ALL,HNM_ROUND="${NEXT_ROUND}",HNM_BASELINE_RUN_ID="${DISTILLED_RUN_ID}" \
        scripts/run_hnm_round.sh
fi

echo "[hnm r${HNM_ROUND}] DONE."
