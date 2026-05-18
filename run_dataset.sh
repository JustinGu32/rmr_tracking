#!/bin/bash
#SBATCH --account=move
#SBATCH --partition=move --qos=normal
####SBATCH --dependency=afterok:15258806
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:rtxpro6000:1

#SBATCH --job-name="collect_dataset"
#SBATCH --output=logs/collect_dataset-%j.out
#SBATCH --error=logs/collect_dataset-%j.err

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# not needed if already in the conda environment when running this script
source /nlp/scr/chrzhang/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

python scripts/multi_collect.py --seed 0

# done
echo "Done"
