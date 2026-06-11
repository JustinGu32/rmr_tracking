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

CHECKPOINT_NAME=my_stairs_0.15_noise_no_vision
CHECKPOINT=/move/u/chrzhang/outputs/diffuse_cloc/${CHECKPOINT_NAME}/checkpoints/latest.ckpt
# Staircase selection. Do NOT `export` this at the top level: an exported var
# persists in your shell and is inherited by later runs (and by sbatch via the
# default --export=ALL), so commenting it out does NOT revert to 3-step — you
# have to `unset WBT_STAIRCASE_DIR` or restart the shell. Instead prefix it
# directly onto the python command (process-scoped) — see below.
STAIRCASE_6STEP=/move/u/chrzhang/rmr_tracking/artifacts/walk_up_karen_stairs_6step

# python scripts/sim2sim_isaaclab_vision_carrot.py --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/root_extrareduced_refactored_state_less_noisy_gravityrootrot/checkpoints/latest.ckpt --carrot_distance_m 0.75 --headless --video
# python scripts/sim2sim_isaaclab_vision_carrot.py --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/combined_datasets_irl_vision_fix/checkpoints/latest.ckpt --carrot_distance_m 0.75 --headless --video --guidance_type velocity --guidance_scale 1 --forward_speed 1.0 --lateral_speed 0 --spin_speed 0

# -u (unbuffered): IsaacLab's simulation_app.close() tears the process down hard
# enough that buffered stdout is lost. Without this, in-loop diagnostics (action
# stats, per-step pelvis height, final result) never reach the .out file.
# ROOT CAUSE: the eval was resetting to the WRONG reference motion. The staircase
# mesh is walk_up_karen_stairs (STAIRCASE_POSITION=[3.2,-0.2,0]); the data was
# collected on the walk_up_karen_stairs motion (robot starts at the foot of the
# stairs, pelvis ~(3.6,0.79), and climbs). The old default --motion_file
# staircase_final_v3 starts the robot at ~(0.07,-1.06) — 3.4m AWAY from the
# staircase — so the camera sees open floor, not stairs, and the policy collapses.
# Fix: use the matching motion so the robot resets at the stairs it trained on.
# MOTION=/move/u/karenvo/Projects/rmr_tracking/artifacts/walk_up_karen_stairs:v0/motion.npz
min_sample_idx=130
max_sample_idx=150
# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --disable_rgb --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "my_stairs_0.15_noise_no_vision_start_${min_sample_idx}" --no_vision

# use shifted action with --action_shift 1
# 6-STEP: the WBT_STAIRCASE_DIR prefix exists only for this python process.
# To run the 3-STEP default instead, comment THIS line and uncomment the one below.
# WBT_STAIRCASE_DIR=${STAIRCASE_6STEP} python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --disable_rgb --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "${CHECKPOINT_NAME}_6step_start_${min_sample_idx}" # --debug_vision
WBT_STAIRCASE_DIR=${STAIRCASE_6STEP} python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --no_vision --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "${CHECKPOINT_NAME}_6step_start_${min_sample_idx}"
# 3-STEP default (no prefix -> cfg uses the original staircase):
# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --disable_rgb --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "${CHECKPOINT_NAME}_3step_start_${min_sample_idx}" # --debug_vision --video_folder videos/vision_stair_climbing_debug

# min_sample_idx=120
# max_sample_idx=140
# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --disable_rgb --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "my_stairs_0.15_noise_start_${min_sample_idx}"

# min_sample_idx=130
# max_sample_idx=150
# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --headless --video --disable_rgb --min_sample_idx ${min_sample_idx} --max_sample_idx ${max_sample_idx} --video_name "my_stairs_0.15_noise_start_${min_sample_idx}"

# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --motion_file ${MOTION} --headless --video --min_sample_idx 0 --max_sample_idx 0 --disable_rgb

# CHECKPOINT_NAME=combined_datasets_stairs_vision_mix_0.5_0.5
# CHECKPOINT=/move/u/chrzhang/outputs/diffuse_cloc/${CHECKPOINT_NAME}/checkpoints/latest.ckpt
# python -u scripts/sim2sim_isaaclab_vision_stair_climbing.py --checkpoint ${CHECKPOINT} --motion_file ${MOTION} --headless --video --min_sample_idx 0 --max_sample_idx 0

# python scripts/sim2sim_isaaclab_vision_stairs.py --checkpoint ${CHECKPOINT} --headless --video
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
