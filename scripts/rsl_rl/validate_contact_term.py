"""Validation harness for the contact-phase termination idea (READ-ONLY; no task changes).

Runs the current trained policy from motion frame 0 and, each step, compares:
  - reference stance: motion foot speed < SPEED_THR  -> that foot SHOULD be planted
  - robot contact:    contact-sensor force > FORCE_THR -> that foot IS planted
A failure (the missed-step cheat) is: reference says STANCE but the robot foot has been
airborne for > GRACE consecutive steps (timing tolerance). We log per-step and report
whether/when the detector fires, to confirm it catches the known skipped first step
BEFORE wiring it into training.

Usage:
  ENABLE_CAMERAS=0 python scripts/rsl_rl/validate_contact_term.py \
    --task=Staircase-G1-ObsAug-v0 \
    --motion_file /home/ubuntu/Downloads/walk_up_33.npz_v0/motion.npz \
    --load_run <run_dir> --checkpoint model_XXXX.pt \
    --speed_thr 0.15 --force_thr 10.0 --grace 5
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Validate contact-phase termination detector.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--speed_thr", type=float, default=0.15, help="Ref foot speed below this = stance.")
parser.add_argument("--force_thr", type=float, default=10.0, help="Contact force above this = in contact.")
parser.add_argument("--grace", type=int, default=5, help="Timing tolerance: airborne steps allowed while ref=stance.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cmd = u.command_manager.get_term("motion")
    T = int(cmd.motion.time_step_total)

    # Tracked-body ankle indices (positions in cfg.body_names): left=3, right=6.
    names = cmd.cfg.body_names
    L_track = names.index("left_ankle_roll_link")
    R_track = names.index("right_ankle_roll_link")

    # Contact-sensor body ids for the two ankles.
    cs = u.scene.sensors["contact_forces"]
    L_cs = cs.body_names.index("left_ankle_roll_link")
    R_cs = cs.body_names.index("right_ankle_roll_link")

    SPEED, FORCE, GRACE = args_cli.speed_thr, args_cli.force_thr, args_cli.grace

    obs, _ = env.get_observations()
    cmd.time_steps[0] = 0
    # ── Build the REFERENCE per-frame foot->stair schedule from the motion + stair geometry ──
    # "Which foot should be on which stair at each frame" — membership in the stair's box
    # (anywhere on the stair is fine), gated by stance (low ref foot speed).
    import json
    import numpy as np
    import math
    from whole_body_tracking.tasks.staircase.staircase_env_cfg import (
        STAIRCASE_POSITION, STAIRCASE_ROTATION, STAIRCASE_DIR,
    )
    meta = json.load(open(os.path.join(STAIRCASE_DIR, "staircase_metadata.json")))
    stairs = meta["stairs"]
    stair_pos = np.array(STAIRCASE_POSITION)
    w, zq = STAIRCASE_ROTATION[0], STAIRCASE_ROTATION[3]
    yaw = 2.0 * math.atan2(zq, w)
    XY_SLACK, Z_SLACK = 0.05, 0.08

    def stair_membership(p_world):
        """Return stair index (1-based) the world-point sits on, else 0 (none)."""
        d = p_world - stair_pos
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = c * d[0] - s * d[1]
        ly = s * d[0] + c * d[1]
        lz = d[2]
        for i, st in enumerate(stairs):
            lo, hi = st["bounds_min_m"], st["bounds_max_m"]
            if (lo[0] - XY_SLACK <= lx <= hi[0] + XY_SLACK and
                    lo[1] - XY_SLACK <= ly <= hi[1] + XY_SLACK and
                    abs(lz - hi[2]) < Z_SLACK):
                return i + 1
        return 0

    # Reference schedule: for each frame, which stair each foot should be on (stance only).
    ref_bp = cmd.motion._body_pos_w.cpu().numpy()  # (T, num_all_bodies, 3) world (motion frame == env-origin frame for env 0 at origin)
    ref_v = cmd.motion._body_lin_vel_w.cpu().numpy()
    # motion body indices for the two feet (the ankle tracked bodies map back to absolute indices)
    L_abs = int(cmd.body_indexes[L_track].item())
    R_abs = int(cmd.body_indexes[R_track].item())
    ref_sched = {}  # frame -> [L_stair, R_stair]  (0 = none expected)
    for fr in range(T):
        out = []
        for ab in (L_abs, R_abs):
            sp = float(np.linalg.norm(ref_v[fr, ab]))
            out.append(stair_membership(ref_bp[fr, ab]) if sp < SPEED else 0)
        ref_sched[fr] = out

    # env-origin offset for env 0 (robot world pos -> motion/stair frame)
    origin0 = u.scene.env_origins[0].cpu().numpy()

    off_streak = torch.zeros(2, device=u.device)  # consecutive steps foot is OFF its expected stair
    fired_at = None
    rows = []

    with torch.inference_mode():
        for _ in range(T + 2):
            t = int(cmd.time_steps[0])
            exp = ref_sched.get(t, [0, 0])  # [L_stair, R_stair] expected this frame

            # robot foot world positions for the two ankles
            rb = u.scene["robot"].data.body_pos_w[0].cpu().numpy()
            L_abs_robot = int(cmd.body_indexes[L_track].item())
            R_abs_robot = int(cmd.body_indexes[R_track].item())
            robot_stair = [
                stair_membership(rb[L_abs_robot] - origin0),
                stair_membership(rb[R_abs_robot] - origin0),
            ]

            # violation: ref expects foot on stair S (>0) but robot foot is NOT on that stair
            viol = torch.tensor(
                [(exp[i] > 0) and (robot_stair[i] != exp[i]) for i in range(2)],
                device=u.device, dtype=torch.bool,
            )
            off_streak = torch.where(viol, off_streak + 1, torch.zeros_like(off_streak))
            fire = bool((off_streak > GRACE).any())
            if fire and fired_at is None:
                fired_at = t

            rows.append((t, exp, robot_stair, off_streak.tolist(), fire))

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if int(cmd.time_steps[0]) >= T - 1:
                break

    print("\n=== per-frame trace (frame | expStair[L,R] | robotStair[L,R] | offStreak[L,R] | FIRE) ===")
    for t, exp, rsr, ost, fr in rows:
        mark = "  <-- FIRE" if fr else ""
        # only print frames where something is expected or a streak is building (keep it readable)
        if exp != [0, 0] or ost[0] > 0 or ost[1] > 0 or fr:
            print(f"f{t:3d} | exp {exp[0]}{exp[1]} | robot {rsr[0]}{rsr[1]} | streak {ost[0]:.0f},{ost[1]:.0f}{mark}")
    print("\n=== RESULT ===")
    if fired_at is not None:
        print(f">>> DETECTOR FIRES at frame {fired_at} (grace={GRACE}, speed_thr={SPEED})")
    else:
        print(f">>> detector never fired (grace={GRACE}) — robot kept feet on the correct stairs per schedule")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
