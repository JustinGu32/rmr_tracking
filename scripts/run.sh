# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/vision_tri_limited_cloned_fixedmask_bs256/checkpoints/latest.ckpt
# python scripts/sim2sim_isaaclab_vision.py --headless --video --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/vision_tri_limited_cloned_fixedmask_visionproj512_layers14/checkpoints/latest.ckpt
# OBSTACLES_ON_PATH=1 places obstacles along the reference path.
# NUM_OBSTACLES=5 OBSTACLES_ON_PATH=1 python scripts/sim2sim_isaaclab_vision.py --headless --video --task Tracking-Flat-G1-Collect-v0 --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/vision_root_extrareduced_refactored_state_less_noisy_gravityrootrot/checkpoints/latest.ckpt
# NUM_OBSTACLES=3 OBSTACLES_ON_PATH=1 python scripts/sim2sim_isaaclab_vision.py --headless --video --task Tracking-Flat-G1-Collect-v0 --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/vision_alternate_limited_cloned_h8_obstacles/checkpoints/latest.ckpt

# NUM_OBSTACLES=5 OBSTACLES_ON_PATH=1 python scripts/sim2sim_isaaclab_vision.py --headless --video --task Tracking-Flat-G1-Collect-v0 --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/vision_root_reduced_fix_obstacles/checkpoints/latest.ckpt --guidance_type combined --guidance_scale 1

python scripts/sim2sim_isaaclab.py --headless --video --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/kimodo_fullstate_bodyplusaction_codiff/checkpoints/latest.ckpt --guidance_type kimodo_velocity --guidance_scale 10 --target_speed 0.8 --target_lateral_speed 0
python scripts/sim2sim_isaaclab.py --headless --video --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/kimodo_fullstate_bodyplusaction_codiff/checkpoints/latest.ckpt --guidance_type kimodo_velocity --guidance_scale 10 --target_speed -0.8 --target_lateral_speed 0
python scripts/sim2sim_isaaclab.py --headless --video --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/kimodo_fullstate_bodyplusaction_codiff/checkpoints/latest.ckpt --guidance_type kimodo_velocity --guidance_scale 10 --target_speed 0 --target_lateral_speed 0.8
