#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:a5000:1 
#SBATCH --job-name=collect_dataset

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

cd /move/u/justingu/Projects/whole_body_tracking/

export HOME=/move/u/justingu

source .venv/bin/activate
conda deactivate >/dev/null 2>&1 || true

echo "Python: $(which python)"
python --version
nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

ENABLE_CAMERAS=1 python scripts/collect_dataset.py \
  --task=Tracking-Flat-G1-v0 \
  --wandb_path=justingu-stanford-university/stair_climbing/rmj7wc1b \
  --num_eps_collect=2 \
  --num_steps_collect=10 \
  --noise_level=0.5 \
  --save_folder=./datasets \
  --headless \
  --num_envs 1

echo "Job finished at: $(date)"
