#!/bin/bash
# Submit resume jobs for multiclip_bones swish gravcurr runs.
# Each job picks up from the wandb checkpoint of a previous run.
# Gravity curriculum args are intentionally omitted: both source runs
# passed ~13k iterations, well beyond the 5000-iter gravity ramp, so
# gravity is already back to normal.

set -euo pipefail
cd /move/u/justingu/rmr_tracking/

ENTITY="robot-mcrobotface"
PROJECT="multiclip_bones"
ZARR="/move/data/bones/g1/zarr/locomotion_50hz.zarr"

# ── Run definitions ──
# Format: TASK  RUN_NAME  ACTIVATION  WANDB_RUN_ID  [ZARR_OVERRIDE]  [INCLUDE_MOTION_TYPES]
# WANDB_RUN_IDs point at the *previous resumed* runs (the _resumed ones at ~27k iters),
# not the original pre-resume runs. This is the 2nd resume for each of these.
RUNS=(
  # "Bones-MultiClip-Compliance-G1-v0  bones_target_50hz_swish_gravcurr12.7_uniform     swish  wil1p51t"
  # "Bones-MultiClip-Compliance-G1-v0  bones_target_50hz_swish_gravcurr6.7_uniform      swish  p32yz9xv"
  # "Tracking-MultiClip-Flat-G1-v0     tracking_baseline_target_50hz_swish_gravcurr12.7_uniform  swish  povzou5t"
  # "Bones-MultiClip-Compliance-G1-v0  bones_target_33hz_swish_uniform_walk-jog  swish  c15qko8c  /move/data/bones/g1/zarr/locomotion_33hz.zarr  walk,jog"
  "Bones-MultiClip-Compliance-G1-v0  bones_target_50hz_swish_gravcurr12.7_uniform_resumed2_walk-jog_finetune33hz  swish  mch9kxtr  /move/data/bones/g1/zarr/locomotion_33hz.zarr  walk,jog"
)

for entry in "${RUNS[@]}"; do
  read -r TASK RUN_NAME ACTIVATION WANDB_ID ZARR_OVERRIDE INCLUDE_TYPES <<< "$entry"
  ZARR_USE="${ZARR_OVERRIDE:-$ZARR}"
  INCLUDE_FLAG=""
  if [[ -n "${INCLUDE_TYPES:-}" ]]; then
    INCLUDE_FLAG="--include_motion_types ${INCLUDE_TYPES}"
  fi

  sbatch --job-name="resume_${RUN_NAME}" \
         --partition=move --account=move \
         --gres=gpu:l40s:1 \
         --cpus-per-task=4 \
         --mem=48G \
         --time=24:00:00 \
         --output="logs/slurm/resume_${RUN_NAME}_%j.out" \
         --error="logs/slurm/resume_${RUN_NAME}_%j.err" \
         --export=ALL \
         --wrap="bash -c '
source /move/u/justingu/miniconda3/etc/profile.d/conda.sh &&
conda activate env_isaaclab &&
mkdir -p logs/slurm &&
cd /move/u/justingu/rmr_tracking/ &&
python scripts/rsl_rl/train_bones.py \
    --task=${TASK} \
    --zarr_path=${ZARR_USE} \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name ${PROJECT} \
    --run_name ${RUN_NAME}_resumed_33hz \
    --ppo_output target \
    --activation ${ACTIVATION} \
    --double_step \
    --sampling uniform \
    --decimation 6 \
    ${INCLUDE_FLAG} \
    --wandb_resume ${ENTITY}/${PROJECT}/${WANDB_ID}
'"

  echo "Submitted resume job for ${RUN_NAME} (wandb: ${WANDB_ID})"
done
