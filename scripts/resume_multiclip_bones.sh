#!/bin/bash
# Submit 4 resume jobs for the multiclip_bones experiment.
# Each job picks up from the wandb checkpoint of a previous run.
# Safe to submit while originals are still running — these will
# queue and start once GPUs free up.

set -euo pipefail
cd /move/u/justingu/rmr_tracking/

ENTITY="robot-mcrobotface"
PROJECT="multiclip_bones"
ZARR="/move/data/bones/g1/zarr/locomotion_50hz.zarr"

# ── Run definitions ──
# Format: TASK  RUN_NAME  ACTIVATION  WANDB_RUN_ID
RUNS=(
  "Bones-MultiClip-Compliance-G1-v0  bones_target_50hz_elu          elu    orj1uirj"
  "Bones-MultiClip-Compliance-G1-v0  bones_target_50hz_swish        swish  n94pye30"
  "Tracking-MultiClip-Flat-G1-v0     tracking_baseline_target_50hz_elu    elu    yp2xkmja"
  "Tracking-MultiClip-Flat-G1-v0     tracking_baseline_target_50hz_swish  swish  cjyxqo05"
)

for entry in "${RUNS[@]}"; do
  read -r TASK RUN_NAME ACTIVATION WANDB_ID <<< "$entry"

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
python scripts/rsl_rl/train.py \
    --task=${TASK} \
    --zarr_path=${ZARR} \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name ${PROJECT} \
    --run_name ${RUN_NAME}_resumed \
    --ppo_output target \
    --activation ${ACTIVATION} \
    --double_step \
    --wandb_resume ${ENTITY}/${PROJECT}/${WANDB_ID}
'"

  echo "Submitted resume job for ${RUN_NAME} (wandb: ${WANDB_ID})"
done
