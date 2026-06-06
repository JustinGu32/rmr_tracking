"""
Offline teacher-forcing diagnostic for the staircase diffusion policy.

Why: in sim2sim the robot starts upright and collapses within ~2 s, while the
emitted actions look in-distribution. That can mean two very different things:
  (a) the policy is underfit / has an obs-preprocessing mismatch -> its actions
      are wrong even on the exact states it trained on, or
  (b) the policy reproduces training actions fine, and the fall is closed-loop
      divergence (compounding error / dynamics) instead.

This script settles it WITHOUT IsaacLab: it feeds the model the *recorded*
observations from the training zarr (including the stored depth_embed, so the
camera + DeFM encoder are bypassed entirely) through the exact same
DiffusionAgentIsaac.get_action() path used in sim2sim, then compares the
predicted action to the recorded action frame-by-frame.

  - Low error + high correlation  -> policy learned the mapping; the sim2sim
    fall is closed-loop divergence (or an eval-time vision/dynamics gap).
  - High error / low correlation  -> the policy itself is the problem
    (underfit, normalizer, or an obs layout mismatch). No point chasing sim.

Usage:
  python scripts/teacher_forcing_check.py \
      --checkpoint /move/u/chrzhang/outputs/diffuse_cloc/my_stair_climbing_dataset/checkpoints/latest.ckpt \
      --zarr /move/u/chrzhang/rmr_tracking/STAIRCASE_DATA_COLLECTED/merged_dataset.zarr \
      --episodes 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

TML_ROOT = Path(__file__).resolve().parent.parent.parent / "TML-BeyondMimic"
sys.path.insert(0, str(TML_ROOT))
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zarr", default="/move/u/chrzhang/rmr_tracking/STAIRCASE_DATA_COLLECTED/merged_dataset.zarr")
    ap.add_argument("--episodes", type=int, default=3, help="How many episodes to evaluate.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--stochastic", dest="deterministic", action="store_false",
                    help="Use stochastic sampling instead of deterministic DDIM.")
    args = ap.parse_args()

    z = zarr.open(args.zarr, "r")
    ee = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ee[:-1]])

    # Recorded streams. Shapes: body_* (N,90)/(N,120); joint_* (N,29); act (N,29).
    act_all = z["data/act"]
    body_pos_all = z["data/body_pos"]
    body_rot_all = z["data/body_rot"]
    body_lv_all = z["data/body_lin_vel"]
    body_av_all = z["data/body_ang_vel"]
    jpos_all = z["data/joint_pos"]
    jvel_all = z["data/joint_vel"]
    depth_all = z["data/depth_embed"]

    print(f"[INFO] Loading policy from {args.checkpoint}")
    policy = DiffusionAgentIsaac(
        checkpoint_path=args.checkpoint,
        device=args.device,
        compile=False,
        warmup=False,
        deterministic=args.deterministic,
    )
    print(f"[INFO] deterministic={args.deterministic}")

    all_pred, all_true = [], []
    for ep in range(min(args.episodes, len(ee))):
        s, e = int(starts[ep]), int(ee[ep])
        policy.reset()
        preds = []
        for t in range(s, e):
            body_pos = body_pos_all[t].reshape(30, 3).astype(np.float32)
            body_quat = body_rot_all[t].reshape(30, 4).astype(np.float32)
            body_lv = body_lv_all[t].reshape(30, 3).astype(np.float32)
            body_av = body_av_all[t].reshape(30, 3).astype(np.float32)
            jpos = jpos_all[t].astype(np.float32)          # already relative (joint_pos_rel)
            jvel = jvel_all[t].astype(np.float32)
            depth = depth_all[t].astype(np.float32)        # recorded DeFM embed (1024,)
            a = policy.get_action(
                body_pos, body_quat, body_lv, body_av, jpos, jvel,
                vision_embeds=depth,
            )
            preds.append(a)
        preds = np.stack(preds, axis=0)
        truth = act_all[s:e].astype(np.float32)
        all_pred.append(preds)
        all_true.append(truth)

        err = preds - truth
        rmse = np.sqrt((err ** 2).mean())
        # normalize error by the action std so it's interpretable
        rel = rmse / (truth.std() + 1e-8)
        print(f"[EP {ep}] frames={e-s} | RMSE={rmse:.3f} (={rel*100:.0f}% of action std) "
              f"| pred absmax={np.abs(preds).max():.2f} true absmax={np.abs(truth).max():.2f}")

    pred = np.concatenate(all_pred, 0)
    true = np.concatenate(all_true, 0)
    err = pred - true
    rmse = np.sqrt((err ** 2).mean())
    # per-joint Pearson correlation between predicted and recorded action
    corr = np.array([
        np.corrcoef(pred[:, j], true[:, j])[0, 1] if true[:, j].std() > 1e-6 else np.nan
        for j in range(pred.shape[1])
    ])
    print("\n=== OVERALL (teacher-forced on recorded obs+depth) ===")
    print(f"RMSE={rmse:.3f}  | action std={true.std():.3f}  | RMSE/std={rmse/(true.std()+1e-8)*100:.0f}%")
    print(f"per-joint corr: mean={np.nanmean(corr):.3f} median={np.nanmedian(corr):.3f} "
          f"min={np.nanmin(corr):.3f} (n_joints<0.5corr={int((corr<0.5).sum())}/29)")
    print(f"pred mean={pred.mean():.3f} std={pred.std():.3f} | true mean={true.mean():.3f} std={true.std():.3f}")
    worst = np.argsort(corr)[:6]
    print("worst joints (idx, corr):", [(int(j), round(float(corr[j]), 2)) for j in worst])
    print("\nInterpretation:")
    print("  corr>0.9, RMSE/std<30%  -> policy reproduces training actions; fall is closed-loop.")
    print("  corr<0.5 or RMSE/std>60% -> policy/obs mismatch; the model itself is wrong.")


if __name__ == "__main__":
    main()
