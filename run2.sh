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

#SBATCH --job-name="vision_sim"
#SBATCH --output=logs/vision_sim-%j.out
#SBATCH --error=logs/vision_sim-%j.err

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

CHECKPOINT_NAME=relabeled_staircase_dataset_again2
CHECKPOINT=/move/u/chrzhang/outputs/diffuse_cloc/${CHECKPOINT_NAME}/checkpoints/latest.ckpt

# python scripts/sim2sim_isaaclab_vision_carrot.py --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/root_extrareduced_refactored_state_less_noisy_gravityrootrot/checkpoints/latest.ckpt --carrot_distance_m 0.75 --headless --video
# python scripts/sim2sim_isaaclab_vision_carrot.py --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/combined_datasets_irl_vision_fix/checkpoints/latest.ckpt --carrot_distance_m 0.75 --headless --video --guidance_type velocity --guidance_scale 1 --forward_speed 1.0 --lateral_speed 0 --spin_speed 0

python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --no_vision
# python scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --no_vision
# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_lateral_m 0.2
# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_lateral_m -0.2
# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_yaw_bias_deg 60

# CHECKPOINT_NAME=relabel_with_stairs_vision_flipped
# CHECKPOINT=/move/u/chrzhang/outputs/diffuse_cloc/${CHECKPOINT_NAME}/checkpoints/latest.ckpt
# python scripts/sim2sim_isaaclab_vision_stairs.py --headless --video --checkpoint ${CHECKPOINT}

# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_yaw_bias_deg 20
# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_yaw_bias_deg 40
# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video --staircase_yaw_bias_deg 60

# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint ${CHECKPOINT} --video_folder videos/tri_experiments --guidance_type velocity --guidance_scale 0.5 --forward_speed 1.0 --lateral_speed 0 --spin_speed 0
# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint ${CHECKPOINT} --video_folder videos/tri_experiments --guidance_type velocity --guidance_scale 0.5 --forward_speed -1.0 --lateral_speed 0 --spin_speed 0
# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint ${CHECKPOINT} --video_folder videos/tri_experiments --guidance_type velocity --guidance_scale 0.5 --forward_speed 0 --lateral_speed 1.0 --spin_speed 0
# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint ${CHECKPOINT} --video_folder videos/tri_experiments --guidance_type velocity --guidance_scale 0.5 --forward_speed 0 --lateral_speed 0 --spin_speed 1.0

# python scripts/sim2sim_isaaclab_vision_rgb_pillars.py --headless --video --checkpoint ${CHECKPOINT}
# python scripts/sim2sim_isaaclab_vision_rgb_pillars.py --headless --video --checkpoint ${CHECKPOINT} --vision_inpaint_image /move/u/chrzhang/rmr_tracking/videos/test/vision_debug/step_00200_rgb.png --vision_inpaint_depth /move/u/chrzhang/rmr_tracking/videos/test/vision_debug/step_00200_depth.npy --debug_vision

# python scripts/sim2sim_isaaclab.py --headless --video --checkpoint ${CHECKPOINT}
# python scripts/sim2sim_isaaclab.py --headless --video --checkpoint ${CHECKPOINT} --guidance_type velocity --guidance_scale 0.5 --forward_speed 1.0 --lateral_speed 0 --spin_speed 0
# python scripts/sim2sim_isaaclab.py --headless --video --checkpoint ${CHECKPOINT} --guidance_type velocity --guidance_scale 0.5 --forward_speed -1.0 --lateral_speed 0 --spin_speed 0
# python scripts/sim2sim_isaaclab.py --headless --video --checkpoint ${CHECKPOINT} --guidance_type velocity --guidance_scale 0.5 --forward_speed 0.0 --lateral_speed 0 --spin_speed 1.0

# done
echo "Done"
