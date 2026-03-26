#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1 
#SBATCH --job-name=bones_crane
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

set -uo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab


WANDB_PROJECT="robot-mcrobotface/bones_crane_ablation"
TASK="Bones-Flat-chip-G1-Play-v0"
VIDEO_LENGTH=1000
NUM_ENVS=1

play_run() {
    local run_id="$1"
    local run_name="$2"
    local ppo_output="$3"
    local no_cmd_obs="$4"

    echo "=========================================="
    echo "Playing: $run_name ($run_id)"
    echo "  ppo_output=$ppo_output, no_cmd_obs=$no_cmd_obs"
    echo "=========================================="

    export BONES_PPO_OUTPUT="$ppo_output"
    export BONES_PUSH="none"
    if [ "$no_cmd_obs" = "1" ]; then
        export BONES_NO_COMMAND_OBS="1"
    else
        unset BONES_NO_COMMAND_OBS
    fi

    python scripts/rsl_rl/play_bones.py \
        --task="$TASK" \
        --num_envs="$NUM_ENVS" \
        --wandb_path="${WANDB_PROJECT}/${run_id}" \
        --video \
        --video_length="$VIDEO_LENGTH" \
        --headless
}

# # 1. delta + push-normal
# play_run "g0i84946" "2026-03-25_03-53-35_crane_rtx_delta_push-normal" "delta" "0"

# # 2. delta + push-normal + no-cmd-obs
# play_run "xccg4odh" "2026-03-25_03-54-37_crane_rtx_delta_push-normal_no-cmd-obs" "delta" "1"

# # 3. target + push-normal
# play_run "4u1sgrtr" "2026-03-25_03-55-01_crane_rtx_target_push-normal" "target" "0"

# # 4. target + push-normal + no-cmd-obs
# play_run "kf4w23fn" "2026-03-25_03-55-34_crane_rtx_target_push-normal_no-cmd-obs" "target" "1"

# # 5. delta + push-soft
# play_run "5rc6mscf" "2026-03-25_03-56-38_crane_l40s_delta_push-soft" "delta" "0"

# # 6. delta + push-soft + no-cmd-obs
# play_run "8ciqbws1" "2026-03-25_03-57-37_crane_l40s_delta_push-soft_no-cmd-obs" "delta" "1"

# # 7. target + push-soft
# play_run "u62b28gt" "2026-03-25_03-58-36_crane_l40s_target_push-soft" "target" "0"

# # 8. target + push-soft + no-cmd-obs
# play_run "m97lawzi" "2026-03-25_03-59-38_crane_l40s_target_push-soft_no-cmd-obs" "target" "1"

# 9. delta + push-none
play_run "7qr0d4dy" "2026-03-25_04-00-14_crane_a5000_delta_push-none" "delta" "0"

# # 10. delta + push-none + no-cmd-obs
# play_run "zikhwu99" "2026-03-25_04-00-37_crane_a5000_delta_push-none_no-cmd-obs" "delta" "1"

# # 11. target + push-none
# play_run "ke5triph" "2026-03-25_04-01-38_crane_a5000_target_push-none" "target" "0"

# # 12. target + push-none + no-cmd-obs
# play_run "bbjay056" "2026-03-25_04-02-37_crane_a5000_target_push-none_no-cmd-obs" "target" "1"
