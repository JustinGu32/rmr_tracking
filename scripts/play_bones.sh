#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:titanrtx:1 
#SBATCH --job-name=play_bones
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

set -uo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab


# WANDB_PROJECT="robot-mcrobotface/bones_crane_ablation"
# TASK="Bones-Flat-chip-G1-Play-v0"
# VIDEO_LENGTH=3000
# NUM_ENVS=1

# play_run() {
#     local run_id="$1"
#     local run_name="$2"
#     local ppo_output="$3"
#     local no_cmd_obs="$4"

#     echo "=========================================="
#     echo "Playing: $run_name ($run_id)"
#     echo "  ppo_output=$ppo_output, no_cmd_obs=$no_cmd_obs"
#     echo "=========================================="

#     export BONES_PPO_OUTPUT="$ppo_output"
#     export BONES_PUSH="none"
#     if [ "$no_cmd_obs" = "1" ]; then
#         export BONES_NO_COMMAND_OBS="1"
#     else
#         unset BONES_NO_COMMAND_OBS
#     fi

#     python scripts/rsl_rl/play_bones.py \
#         --task="$TASK" \
#         --num_envs="$NUM_ENVS" \
#         --wandb_path="${WANDB_PROJECT}/${run_id}" \
#         --video \
#         --video_length="$VIDEO_LENGTH" \
#         --video_folder="videos/${run_name}" \
#         --headless
# }

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


# # 10. MAIN-URDF + delta-all + push-none
# play_run "zgct3u7r" "2026-03-27_03-39-09_MAIN-URDF_DELTA-ALL_crane_a5000_delta_push-none" "delta-all" "0"

# # 11. MAIN-URDF + target + push-none
# play_run "zm1yov1z" "2026-03-27_02-36-45_MAIN-URDF_crane_a5000_target_push-none" "target" "0"

# # 12. MAIN-URDF + delta + push-none
# play_run "5mp8bzwk" "2026-03-27_02-36-42_MAIN-URDF_crane_a5000_delta_push-none" "delta" "0"

# # 13. delta + push-none
# play_run "2qkabw5i" "2026-03-26_20-52-39_crane_a5000_delta_push-none" "delta" "0"

# # 14. target + push-none
# play_run "31jta9bw" "2026-03-26_20-52-39_crane_a5000_target_push-none" "target" "0"

# # 9. delta-all + push-none
# play_run "vzrv1g2j" "2026-03-27_04-09-52_DELTA-ALL_crane_a5000_delta_push-none" "delta-all" "0"


# ============================================================
# MultiClip BONES play commands
# ============================================================

# WANDB_PROJECT_MC="robot-mcrobotface/multiclip_bones"
# TASK_MC="Bones-MultiClip-Compliance-G1-v0"
# ZARR_PATH="/move/data/bones/g1/zarr/locomotion_50hz.zarr"

# play_multiclip_run() {
#     local run_id="$1"
#     local run_name="$2"
#     local ppo_output="$3"
#     local activation="${4:-elu}"
#     local zarr_path="${5:-$ZARR_PATH}"
#     local decimation="${6:-}"

#     echo "=========================================="
#     echo "Playing MultiClip: $run_name ($run_id)"
#     echo "  ppo_output=$ppo_output, activation=$activation, zarr_path=$zarr_path, decimation=${decimation:-default}"
#     echo "=========================================="

#     export BONES_PPO_OUTPUT="$ppo_output"
#     export BONES_PUSH="none"

#     local decimation_flag=""
#     if [ -n "$decimation" ]; then
#         decimation_flag="--decimation=$decimation"
#     fi

#     python scripts/rsl_rl/play_bones.py \
#         --task="$TASK_MC" \
#         --num_envs="$NUM_ENVS" \
#         --zarr_path="$zarr_path" \
#         --max_clips=10 \
#         --wandb_path="${WANDB_PROJECT_MC}/${run_id}" \
#         --activation="$activation" \
#         --start_from_beginning \
#         $decimation_flag \
#         --video \
#         --video_length="$VIDEO_LENGTH" \
#         --video_folder="videos/multiclip/wider_terms/${run_name}" \
#         --headless
# }

# # 1. bones_delta-all_50hz_swish_8k
# play_multiclip_run "6sferz1a" "2026-04-06_09-15-11_bones_delta-all_50hz_swish_8k" "delta-all" "swish"

# # 2. bones_target_50hz_swish_8k
# play_multiclip_run "eeo5xira" "2026-04-06_09-14-07_bones_target_50hz_swish_8k" "target" "swish"

# # 3. bones_target_50hz_8k
# play_multiclip_run "e9ubd36w" "2026-04-06_09-11-56_bones_target_50hz_8k" "target"

# # 4. bones_delta-all_50hz
# play_multiclip_run "gait2u9c" "2026-04-06_08-54-14_bones_delta-all_50hz" "delta-all"

# # 5. bones_target_33hz_loose-terms
# play_multiclip_run "sqznpjgk" "2026-04-07_02-44-12_bones_target_33hz_loose-terms" "target" "elu" "/move/data/bones/g1/zarr/locomotion_33hz.zarr" "6"

# # 6. bones_target_50hz_swish_16k_loose-terms
# play_multiclip_run "xf2vwse4" "2026-04-07_02-44-14_bones_target_50hz_swish_16k_loose-terms" "target" "swish"

# # 7. bones_delta-all_50hz_swish_16k_loose-terms
# play_multiclip_run "4v04yhka" "2026-04-07_02-44-10_bones_delta-all_50hz_swish_16k_loose-terms" "delta-all" "swish"

# # 8. bones_target_50hz_16k_loose-terms
# play_multiclip_run "1wd1fh8b" "2026-04-07_02-42-14_bones_target_50hz_16k_loose-terms" "target"

# # 9. bones_delta-all_50hz_loose-terms
# play_multiclip_run "pkpsylra" "2026-04-07_02-41-13_bones_delta-all_50hz_loose-terms" "delta-all"

# # 10. bones_target_50hz_swish_loose-terms
# play_multiclip_run "fudnqofd" "2026-04-07_02-42-14_bones_target_50hz_swish_loose-terms" "target" "swish"

# # 11. bones_target_50hz_loose-terms
# play_multiclip_run "3tultbot" "2026-04-07_02-40-14_bones_target_50hz_loose-terms" "target"


# # ============================================================
# # Single-clip BONES play: stand_up_lying_R_002 (run eozs32pw)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/eozs32pw \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/37815_stand_up_lying_R_002__A475_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/bones_stand_up_lying_R_002_37815_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_002_131_M_33hz (run hvgzk08y)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/hvgzk08y \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/131_Jump_002__A017_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_002_131_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_and_land_light_005_M_33hz (run 27v9xoyy)
# # NOTE: only candidate in bones_33hz/ is 5_jump_and_land_light_003__A001_M.npz
# #       (run name's "_005" likely came from the file's "5_" prefix).
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/27v9xoyy \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/5_jump_and_land_light_003__A001_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_and_land_light_005_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: 125_Jump_002__A017_M_50hz (run el7o911j)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/el7o911j \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/125_Jump_002__A017_M.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/125_Jump_002__A017_M_50hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: stand_up_lying_R_002__A475_M_50hz (run pbwpcori)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/pbwpcori \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/37810_stand_up_lying_R_002__A475_M.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/stand_up_lying_R_002__A475_M_50hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_and_land_light_003_50hz (run c1mcnvoc)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/c1mcnvoc \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/3_jump_and_land_light_003__A001.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_and_land_light_003_50hz" \
#     --headless


# ============================================================
# Single-clip TRACKING play: tracking_jump_and_land_light_003_50hz (run vwy5ns2k)
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/vwy5ns2k \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/3_jump_and_land_light_003__A001.npz \
    --activation=swish \
    --decimation=4 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_jump_and_land_light_003_50hz" \
    --headless


# ============================================================
# Single-clip TRACKING play: tracking_stand_up_lying_R_002__A475_M_50hz (run g1oa1vb4)
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/g1oa1vb4 \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/37810_stand_up_lying_R_002__A475_M.npz \
    --activation=swish \
    --decimation=4 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_stand_up_lying_R_002__A475_M_50hz" \
    --headless


# ============================================================
# Single-clip TRACKING play: tracking_125_Jump_002__A017_M_50hz (run 9jhcjxr3)
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/9jhcjxr3 \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/125_Jump_002__A017_M.npz \
    --activation=swish \
    --decimation=4 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_125_Jump_002__A017_M_50hz" \
    --headless


# ============================================================
# Single-clip TRACKING play: tracking_jump_and_land_light_005_M_33hz (run uaetnfwg)
# NOTE: only candidate in bones_33hz/ is 5_jump_and_land_light_003__A001_M.npz
#       (run name's "_005" likely came from the file's "5_" prefix).
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/uaetnfwg \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/5_jump_and_land_light_003__A001_M.npz \
    --activation=swish \
    --decimation=6 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_jump_and_land_light_005_M_33hz" \
    --headless


# ============================================================
# Single-clip TRACKING play: tracking_stand_up_lying_R_002_37815_M_33hz (run y17mhuei)
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/y17mhuei \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/37815_stand_up_lying_R_002__A475_M.npz \
    --activation=swish \
    --decimation=6 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_stand_up_lying_R_002_37815_M_33hz" \
    --headless


# ============================================================
# Single-clip TRACKING play: tracking_jump_002_131_M_33hz (run 7uo3dtyw)
# ============================================================
export BONES_PPO_OUTPUT="target"
export WBT_PPO_OUTPUT="target"
export BONES_DOUBLE_STEP="1"

python scripts/rsl_rl/play_bones.py \
    --task=Tracking-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/bones_singleclip/7uo3dtyw \
    --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/131_Jump_002__A017_M.npz \
    --activation=swish \
    --decimation=6 \
    --video \
    --video_length=3000 \
    --video_folder="eval_results/clip_videos/bones_singleclip/tracking_jump_002_131_M_33hz" \
    --headless


# # ============================================================
# # Single-clip BONES play: stand_up_lying_R_002_37815_M_33hz (run eozs32pw)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/eozs32pw \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/37815_stand_up_lying_R_002__A475_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/stand_up_lying_R_002_37815_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_002_131_M_33hz (run hvgzk08y)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/hvgzk08y \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/131_Jump_002__A017_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_002_131_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_and_land_light_005_M_33hz (run 27v9xoyy)
# # NOTE: only candidate in bones_33hz/ is 5_jump_and_land_light_003__A001_M.npz
# #       (run name's "_005" likely came from the file's "5_" prefix).
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/27v9xoyy \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_33hz/5_jump_and_land_light_003__A001_M.npz \
#     --activation=swish \
#     --decimation=6 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_and_land_light_005_M_33hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: 125_Jump_002__A017_M_50hz (run el7o911j)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/el7o911j \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/125_Jump_002__A017_M.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/125_Jump_002__A017_M_50hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: stand_up_lying_R_002__A475_M_50hz (run pbwpcori)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/pbwpcori \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/37810_stand_up_lying_R_002__A475_M.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/stand_up_lying_R_002__A475_M_50hz" \
#     --headless


# # ============================================================
# # Single-clip BONES play: jump_and_land_light_003_50hz (run c1mcnvoc)
# # ============================================================
# export BONES_PPO_OUTPUT="target"
# export WBT_PPO_OUTPUT="target"
# export BONES_DOUBLE_STEP="1"

# python scripts/rsl_rl/play_bones.py \
#     --task=Bones-Flat-chip-G1-Play-v0 \
#     --num_envs=1 \
#     --wandb_path=robot-mcrobotface/bones_singleclip/c1mcnvoc \
#     --motion_file=/move/u/justingu/rmr_tracking/motions/bones_50hz/3_jump_and_land_light_003__A001.npz \
#     --activation=swish \
#     --decimation=4 \
#     --video \
#     --video_length=3000 \
#     --video_folder="eval_results/clip_videos/bones_singleclip/jump_and_land_light_003_50hz" \
#     --headless