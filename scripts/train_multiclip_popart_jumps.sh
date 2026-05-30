#!/bin/bash
#SBATCH --job-name=jumping
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/jumps/jump_%j.out
#SBATCH --error=logs/slurm/jumps/jump_%j.err

# ============================================================================
# Jump-exploration ablations on the multi-clip popart task (standup/walk/jump).
#
# Usage: this is NOT an sbatch array — it's a menu. Uncomment exactly ONE
# command and run it, choosing the GPU yourself, e.g.:
#
#     CUDA_VISIBLE_DEVICES=0 bash scripts/train_multiclip_popart_jumps.sh
#
# (or paste a single line into your shell). Run several concurrently by
# launching the script multiple times with different CUDA_VISIBLE_DEVICES.
#
# Each arm is documented in source/.../popart/mdp/jumps.py. All jump terms are
# OFF unless their flag is passed, so the first arm is a clean baseline.
# Terrain noise is intentionally NOT included here (still under separate test).
# ============================================================================

cd /move/u/justingu/rmr_tracking/
source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

ZARR=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr

# Shared flags for every arm (run_name + sampling_mode + arm flags are appended).
BASE="python scripts/rsl_rl/train_bones.py \
  --task=Popart-Flat-G1-v0 \
  --zarr_path=$ZARR \
  --num_envs 4096 --headless \
  --logger wandb --log_project_name jump_exploration \
  --decimation 6 --sampling uniform --activation swish \
  --categories stand_up,walk,jump --popart off"

# ── Baseline ────────────────────────────────────────────────────────────────
# $BASE --sampling_mode balanced --run_name baseline_balanced_no_jump_methods

# ── Sampling 2x2 (orthogonal; no reward shaping) ──────────────────────────────
#   rows: category sampling by termination-failures (beta 0) vs tracking-error (beta 1)
#   cols: no jump termination vs + below-ref-during-flight termination (T3)
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 0                          --run_name adap_catfail_only
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 0   --jump_tighten_anchor_z --run_name adap_catfail_with_jump_termination
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 1                          --run_name adap_trackerror_only
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 1   --jump_tighten_anchor_z --run_name adap_trackerror_with_jump_termination
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 0.5 --jump_tighten_anchor_z --run_name adap_fail_error_blend_with_jump_termination

# ── Reward / RSI arms (on top of balanced sampling) ──────────────────────────
# $BASE --sampling_mode balanced --jump_airborne_penalty                       --run_name rew_airborne_contact_penalty
# $BASE --sampling_mode balanced --jump_airborne_penalty --jump_flight_bonus   --run_name rew_airborne_penalty_plus_flight_bonus
# $BASE --sampling_mode balanced --jump_below_z_penalty                        --run_name rew_pelvis_below_reference_z_penalty
# $BASE --sampling_mode balanced --jump_foot_z_penalty                         --run_name rew_foot_below_reference_penalty
# $BASE --sampling_mode balanced --jump_contact_phase                          --run_name rew_contact_phase_match
# $BASE --sampling_mode balanced --flight_rsi_ratio 0.3                        --run_name rsi_start_airborne_30pct

# ── Combined arms ────────────────────────────────────────────────────────────
# $BASE --sampling_mode balanced --jump_airborne_penalty --jump_flight_bonus --jump_contact_phase --flight_rsi_ratio 0.3 --run_name combo_all_reward_shaping_plus_rsi
# $BASE --sampling_mode cat_blend_clip_uniform --error_blend_beta 0.5 --jump_tighten_anchor_z --jump_airborne_penalty --jump_flight_bonus --flight_rsi_ratio 0.3 --run_name combo_blend_sampling_jump_term_airborne_bonus_rsi
