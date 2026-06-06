#!/bin/bash
#SBATCH --account=move
#SBATCH --partition=move --qos=normal
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:rtxpro6000:1

#SBATCH --job-name="debug_staircases"
#SBATCH --output=logs/debug_staircases-%j.out
#SBATCH --error=logs/debug_staircases-%j.err

# Submit a SMALL debug staircase collection to SLURM (few envs/eps, --video) so the
# new 1/2/3-step staircase + cropped-motion pairs can be eyeballed before a full run.
#
# Usage (modeled on run_dataset.sh):
#   sbatch run_debug_staircases.sh            # debug all of 1/2/3-step
#   sbatch run_debug_staircases.sh 1          # just the 1-step staircase
#   sbatch run_debug_staircases.sh 2 3        # the 2- and 3-step staircases
#   sbatch --export=ALL,DEBUG_ENVS=4,DEBUG_EPS=10 run_debug_staircases.sh 1
#
# Any positional args (K values) are forwarded to build_step_datasets.sh --debug.
# DEBUG_ENVS / DEBUG_EPS / WANDB_PATH / NUM_STEPS are read from the environment by
# that script (sbatch passes the environment through with the default --export=ALL).

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "working directory = "$SLURM_SUBMIT_DIR

# not needed if already in the conda environment when running this script
source /nlp/scr/chrzhang/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

# Builds the assets (meshes + cropped motions) then runs the small per-K debug collect.
bash scripts/build_step_datasets.sh --debug "$@"

# done
echo "Done"
