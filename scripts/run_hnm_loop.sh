#!/bin/bash
# ===========================================================================
# Hard-negative mining — start the loop.
#
# Just kicks off round 1; each round's sbatch script (run_hnm_round.sh)
# decides whether to submit the next round (when failed-clip count drops
# by >= HNM_RELATIVE_STOP_THRESH relative to the previous round and we're
# under HNM_MAX_ROUNDS).
#
# All config is via env vars. The minimum is:
#
#   HNM_BASELINE_RUN_ID=ia5mxune                                            \
#   HNM_ZARR_PATH=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
#   HNM_CATEGORIES=stand_up,walk,jump                                       \
#   HNM_WANDB_PROJECT=balanced_sampling                                     \
#   HNM_WANDB_ENTITY=robot-mcrobotface                                      \
#   bash scripts/run_hnm_loop.sh
#
# Or set them in the environment of the calling shell and just run the
# script. See scripts/run_hnm_round.sh for the full list of supported
# env vars and their defaults.
# ===========================================================================

set -euo pipefail

: "${HNM_BASELINE_RUN_ID:?must set HNM_BASELINE_RUN_ID (wandb run id of starting baseline)}"
: "${HNM_ZARR_PATH:?must set HNM_ZARR_PATH}"
: "${HNM_CATEGORIES:?must set HNM_CATEGORIES}"
: "${HNM_WANDB_PROJECT:?must set HNM_WANDB_PROJECT}"
: "${HNM_WANDB_ENTITY:?must set HNM_WANDB_ENTITY}"

export HNM_ROUND="${HNM_ROUND:-1}"

# Wipe any stale history.csv from a previous loop with the same baseline?
# No — we APPEND, never truncate. The header line is only written if the
# file doesn't already exist.

# Submit round 1. Subsequent rounds are submitted by run_hnm_round.sh's
# finalize step. Job id of round 1 printed for visibility.
mkdir -p logs/slurm/hnm
JOBID=$(sbatch --parsable --export=ALL scripts/run_hnm_round.sh)
echo "[hnm loop] submitted round ${HNM_ROUND} as sbatch job ${JOBID}"
echo "[hnm loop] subsequent rounds will self-submit as long as failed-clip count"
echo "[hnm loop] drops by >= ${HNM_RELATIVE_STOP_THRESH:-0.5} per round and we're"
echo "[hnm loop] under HNM_MAX_ROUNDS=${HNM_MAX_ROUNDS:-6}."
echo "[hnm loop] watch progress in: ${HNM_OUT_BASE_DIR:-eval_results/hnm}/history.csv"
