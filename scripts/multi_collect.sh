#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtxpro6000:1 
#SBATCH --job-name=up_continuous_v2_karen_stairs_collection
#SBATCH --output=slurm_outputs/slurm-%A_%a.out
#SBATCH --array=1-3

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
export OMNI_KIT_ACCEPT_EULA=y
conda activate env_isaaclab


python scripts/multi_collect.py --seed $SLURM_ARRAY_TASK_ID