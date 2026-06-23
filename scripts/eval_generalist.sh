#!/bin/bash
#SBATCH --job-name=eval_generalist
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/eval_generalist_%j.out
#SBATCH --error=logs/slurm/eval_generalist_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# ============================================================
# Run eval_specialist_pool on a trained generalist policy across every clip
# in the cleaner zarr. Produces failed_clip_ids.json and eval_summary.csv.
# Use this output as the baseline failed list for HNM round 1.
# ============================================================

ZARR_PATH=${ZARR_PATH:-/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr}
CATEGORIES=${CATEGORIES:-stand_up,walk,jump}
WANDB_ENTITY=${WANDB_ENTITY:-robot-mcrobotface}
WANDB_PROJECT=${WANDB_PROJECT:-balanced_sampling}
RUN_ID=${RUN_ID:?must set RUN_ID (wandb run id of the generalist)}

OUT_DIR=${OUT_DIR:-eval_results/generalist_${RUN_ID}}

python scripts/eval_specialist_pool.py \
    --task=Generalist-Flat-G1-Play-v0 \
    --wandb_path=${WANDB_ENTITY}/${WANDB_PROJECT}/${RUN_ID} \
    --zarr_path=${ZARR_PATH} \
    --categories ${CATEGORIES} \
    --num_passes 3 \
    --start_frame_mode uniform \
    --max_steps_per_pass 500 \
    --decimation 6 --activation swish \
    --popart off \
    --num_envs 4096 \
    --output_dir ${OUT_DIR}

echo "[eval_generalist] outputs:"
echo "  ${OUT_DIR}/failed_clip_ids.json"
echo "  ${OUT_DIR}/eval_summary.csv"
