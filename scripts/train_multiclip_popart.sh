#!/bin/bash
#SBATCH --job-name=popart
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/popart_%j.out
#SBATCH --error=logs/slurm/popart_%j.err

# Create log directory
mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ============================================================
# PopArt task ablations (multi-head value normalization)
#
# The Popart-Flat-G1-v0 task has these defaults baked into its cfg:
#   include_motion_types=["walk", "jog"]   (zarr filter)
#   categorizer="walk_vs_jog"              (clip name → category int)
#   num_categories=2
#   popart_momentum=0.1
#   use_clipped_value_loss=False           (forced by PopArt path)
#
# To switch to walk vs stand_up:
#   1. Edit popart/config/g1/flat_env_cfg.py (categorizer + include list)
#   2. Pass --include_motion_types=walk,stand_up to override the filter
#      (also matches the new categorizer's expected vocabulary)
# ============================================================

# # 1. Vanilla full-scale walk vs jog, 4096 envs
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk100_jog100.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger=wandb \
#     --log_project_name=popart_smoke \
#     --run_name=popart_walkjog_full \
#     --decimation=6

# # 2. Stage A comparison: same dataset/scale, no PopArt path
# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk100_jog100.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger=wandb \
#     --log_project_name=popart_smoke \
#     --run_name=tracking_walkjog_full \
#     --decimation=6 \
#     --include_motion_types=walk,jog \
#     --seed=42

# # 3. Walk vs jog with gravity curriculum (matches your bones_target_50hz_gravcurr pattern)
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk100_jog100.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger=wandb \
#     --log_project_name=popart_smoke \
#     --run_name=popart_walkjog_gravcurr \
#     --decimation=6 \
#     --gravity_curriculum --start_gravity -2.0 --gravity_ramp_steps 5000

# # 4. Resume from a previous popart run by wandb path
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk100_jog100.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger=wandb \
#     --log_project_name=popart_smoke \
#     --run_name=popart_walkjog_resume \
#     --decimation=6 \
#     --wandb_resume=robot-mcrobotface/popart_smoke/<RUN_ID>

# Default active run: full-scale vanilla popart on walk + jog.
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_small.zarr \
#     --num_envs 4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name justin_popart \
#     --run_name popart_standupwalk_small \
#     --decimation 6 \
#     --sampling uniform \
#     --activation swish \
#     --categories stand_up,walk

# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_all.zarr \
#     --num_envs 4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name justin_popart \
#     --run_name popart_standupwalk_all \
#     --decimation 6 \
#     --sampling uniform \
#     --activation swish \
#     --categories stand_up,walk


# # vanilla (no popart), balanced
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_small.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name justin_popart \
#     --run_name balanced_vanilla_standup_walk_small \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk \
#     --sampling_mode balanced

# popart off, balanced
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name balanced_sampling \
#     --run_name balanced_vanilla_standup_walk_jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump \
#     --sampling_mode balanced \
#     --popart off \

# ============================================================
# Sampling-mode matrix on standup_walk_jump
# Run ONE block at a time (uncomment exactly one of the python invocations).
# All blocks share: same task, dataset, num_envs, --popart off, --sampling
# uniform (the --sampling flag is overridden by sampling_mode for any mode
# other than frame_uniform — it's left in so frame_uniform runs are valid too).
# ============================================================

# (1) frame_uniform — global frame timeline, no category structure.
#     This is the parent MultiClipMotionCommand default. With --sampling
#     uniform = pure uniform over all frames; with --sampling adaptive =
#     the bin-based EMA curriculum.
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name balanced_sampling \
#     --run_name frame_uniform_standup_walk_jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump \
#     --sampling_mode frame_uniform \
#     --popart off

# (2) balanced — uniform cat → uniform clip → uniform frame.
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name balanced_sampling \
#     --run_name balanced_standup_walk_jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump \
#     --sampling_mode balanced \
#     --popart off

# (3) clip_adaptive — clipped-adaptive over all clips, ignoring cats; uniform frame.
python scripts/rsl_rl/train_bones.py \
    --task=Popart-Flat-G1-v0 \
    --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
    --num_envs 4096 --headless \
    --logger wandb --log_project_name balanced_sampling \
    --run_name clip_adaptive_standup_walk_jump \
    --decimation 6 --sampling uniform --activation swish \
    --categories stand_up,walk,jump \
    --sampling_mode clip_adaptive \
    --popart off

# (4) cat_uniform_clip_adaptive — uniform cat → adaptive clip in cat → uniform frame.
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name balanced_sampling \
#     --run_name cat_uniform_clip_adaptive_standup_walk_jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump \
#     --sampling_mode cat_uniform_clip_adaptive \
#     --popart off

# # (5) cat_adaptive_clip_uniform — adaptive cat → uniform clip in cat → uniform frame.
# python scripts/rsl_rl/train_bones.py \
#     --task=Popart-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \
#     --num_envs 4096 --headless \
#     --logger wandb --log_project_name balanced_sampling \
#     --run_name cat_adaptive_clip_uniform_standup_walk_jump \
#     --decimation 6 --sampling uniform --activation swish \
#     --categories stand_up,walk,jump \
#     --sampling_mode cat_adaptive_clip_uniform \
#     --popart off \
