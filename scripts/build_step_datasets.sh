#!/bin/bash
# Build matched (staircase mesh, episode length) settings for collecting 1-, 2-, and
# 3-step staircase datasets -- WITHOUT touching the motion.
#
# Approach (matches the "just_stairs" 3-step collection you already use):
#   * SAME policy, SAME original motion.npz, SAME spawn window (min/max_sample_idx).
#   * Only two things change per K:
#       - the rendered staircase mesh: K boxes (make_n_step_staircase.py), boxes 1..K
#         are byte-identical to the original, so the staircase POSITION is unchanged.
#       - num_steps_collect: stop the episode after the robot finishes climbing step K.
#   No motion cropping / freezing -- the robot starts exactly where it does in the
#   3-step run (spawn frame ~120, at the base on box 1) and climbs as far as the
#   shortened episode allows.
#
# Per-K cut points come from the real just_stairs data (steps-120, dec-6, 33.3 Hz),
# measured as the control-step where BOTH feet first reach box K:
#       both on box1 ~ctrl-step 0-10   (robot starts here)
#       both on box2 ~ctrl-step 46-55
#       both on box3 ~ctrl-step 90-99  (then settles until step 120)
# so cutting K=2 at ~70 and K=1 at ~30 lands the robot settled on box K, before it
# reaches for box K+1. Tune from the debug video; override with CUT1/CUT2/CUT3.
#
# Default mode: generates the K-box meshes and PRINTS the collect commands.
# Debug mode (--debug [K ...]): RUNS a small collect (few envs/eps, --video) so you can
# eyeball each staircase length. Launches Isaac (GPU).
#   bash scripts/build_step_datasets.sh --debug 1      # just the 1-step
#   bash scripts/build_step_datasets.sh --debug        # all three
#   DEBUG_ENVS=4 DEBUG_EPS=3 bash scripts/build_step_datasets.sh --debug 2
#
# The staircase mesh is selected by the process-scoped WBT_STAIRCASE_DIR prefix
# (NEVER `export` it -- env-var gotcha).
set -euo pipefail
cd /move/u/chrzhang/rmr_tracking

# ---- arg parsing ----
DEBUG=0
DEBUG_KS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) DEBUG=1 ;;
    1|2|3)   DEBUG_KS+=("$1") ;;
    *) echo "unknown arg: $1 (usage: [--debug [1] [2] [3]])"; exit 1 ;;
  esac
  shift
done
[[ ${#DEBUG_KS[@]} -eq 0 ]] && DEBUG_KS=(1 2 3)
DEBUG_ENVS="${DEBUG_ENVS:-8}"    # small env count for a quick debug rollout
DEBUG_EPS="${DEBUG_EPS:-20}"     # small episode count for a quick debug rollout

ORIG_STAIR=/move/u/karenvo/Projects/rmr_tracking/artifacts/walk_up_karen_stairs
ORIG_MOTION=artifacts/walk_up_karen_stairs:v0/motion.npz

echo "============================================================"
echo "[1/1] Generating staircase meshes (boxes 1..K identical -> same position)"
echo "============================================================"
python scripts/make_n_step_staircase.py --num_boxes 1 --out_dir artifacts/walk_up_karen_stairs_1step
python scripts/make_n_step_staircase.py --num_boxes 2 --out_dir artifacts/walk_up_karen_stairs_2step
echo "3-step: using the original mesh as-is -> $ORIG_STAIR"

# Per-K staircase mesh. The MOTION is the original for every K (no crop).
declare -A STAIR=( [1]=artifacts/walk_up_karen_stairs_1step
                   [2]=artifacts/walk_up_karen_stairs_2step
                   [3]="$ORIG_STAIR" )

# Per-K episode length (control steps): where the robot has settled on box K.
declare -A CUT=( [1]="${CUT1:-30}" [2]="${CUT2:-70}" [3]="${CUT3:-120}" )

# Same policy + spawn-at-the-stairs as the just_stairs 3-step collection.
# In the FULL original motion.npz the flat-ground approach is frames 0-214 and the
# climb is frames ~215-362 (lower foot lands box1@226, box2@295, box3@362). The
# just_stairs collection spawned at raw frame ~205-227 (confirmed by pose-matching its
# saved episodes back to this motion) -- i.e. min_sample_idx 120 on a motion that had
# its ~100 leading flat frames trimmed. On the full motion that window is ~205-227,
# NOT 120 (120 is flat ground; the robot would just walk and never reach the stairs).
WANDB_PATH="${WANDB_PATH:-robot-mcrobotface/new_staircase/hfv4s1wg}"
MIN_IDX="${MIN_IDX:-205}"
MAX_IDX="${MAX_IDX:-227}"
NUM_EPS="${NUM_EPS:-2000}"

# episode_collect_length (s) must outlast num_steps_collect at 33.3 Hz (+margin).
eplen() { python -c "print(round($1/33.3 + 1.5, 1))"; }

# ---- debug mode: small collect per requested K (launches Isaac) ----
if [[ $DEBUG -eq 1 ]]; then
  echo
  echo "============================================================"
  echo "DEBUG: small collect for K=${DEBUG_KS[*]}  (num_envs=$DEBUG_ENVS, num_eps=$DEBUG_EPS, --video)"
  echo "============================================================"
  for K in "${DEBUG_KS[@]}"; do
    steps="${CUT[$K]}"; ep_len=$(eplen "$steps")
    echo
    echo ">>> ${K}-step  mesh=${STAIR[$K]}  num_steps=$steps  spawn[$MIN_IDX,$MAX_IDX]  ep_len=${ep_len}s"
    WBT_STAIRCASE_DIR="${STAIR[$K]}" WBT_USE_DEPTH_OBS=0 ENABLE_CAMERAS=1 \
      python scripts/rsl_rl/collect_dataset.py --task=Staircase-G1-Collect-v0 --num_envs "$DEBUG_ENVS" \
      --wandb_path="$WANDB_PATH" --motion_file="$ORIG_MOTION" \
      --num_eps_collect="$DEBUG_EPS" --num_steps_collect="$steps" \
      --episode_collect_length "$ep_len" --min_sample_idx "$MIN_IDX" --max_sample_idx "$MAX_IDX" \
      --save_folder="./STAIRCASE_DATA_STEPS_DEBUG/${K}step" --disable_rgb --video --headless
  done
  echo
  echo "DEBUG done. Videos: videos/Staircase-G1-Collect-v0/   zarrs: STAIRCASE_DATA_STEPS_DEBUG/"
  exit 0
fi

cat <<EOF

============================================================
NEXT STEPS  (run these yourself; they launch Isaac)
============================================================
Meshes are ready. For each K, the ONLY changes vs the 3-step collection are the mesh
(WBT_STAIRCASE_DIR) and num_steps_collect. Same policy, same motion, same spawn.
Eyeball a debug video first:  bash scripts/build_step_datasets.sh --debug K

EOF

for K in 1 2 3; do
  steps="${CUT[$K]}"; ep_len=$(eplen "$steps")
  echo "------------------------------------------------------------"
  echo "  ${K}-STEP   mesh=${STAIR[$K]}"
  echo "           num_steps=${steps}  spawn frames [${MIN_IDX}, ${MAX_IDX}]  ep_len=${ep_len}s"
  echo "------------------------------------------------------------"
  echo "  # collect (WBT_USE_DEPTH_OBS must match your teacher; 0 for 154-dim proprio):"
  echo "  WBT_STAIRCASE_DIR=${STAIR[$K]} WBT_USE_DEPTH_OBS=0 ENABLE_CAMERAS=1 \\"
  echo "    python scripts/rsl_rl/collect_dataset.py --task=Staircase-G1-Collect-v0 --num_envs 750 \\"
  echo "    --wandb_path=$WANDB_PATH --motion_file=$ORIG_MOTION \\"
  echo "    --num_eps_collect=$NUM_EPS --num_steps_collect=$steps \\"
  echo "    --episode_collect_length $ep_len --min_sample_idx $MIN_IDX --max_sample_idx $MAX_IDX \\"
  echo "    --save_folder=./STAIRCASE_DATA_STEPS/${K}step --disable_rgb --headless"
  echo
done

cat <<EOF
Notes:
 - No motion cropping. The robot spawns at frame [$MIN_IDX, $MAX_IDX] of the original
   motion -- the base of the stairs, where the climb begins (frame ~215). It then
   climbs; the shorter num_steps just ends the episode once it has settled on box K.
   (Spawn ~120 would be flat-ground approach -- the robot never reaches the stairs.)
 - Cut points (CUT1=$CUT1.. defaults 30/70/120 control-steps) come from the just_stairs
   data: both feet on box2 ~ctrl-step 46-55, box3 ~90-99 (motion advances ~1.5 frame
   per control step). Tune via CUT1/CUT2/CUT3 after watching the debug video; override
   spawn with MIN_IDX/MAX_IDX, policy with WANDB_PATH.
EOF
