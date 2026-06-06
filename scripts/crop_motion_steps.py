#!/usr/bin/env python3
"""
Crop the 3-step `walk_up_karen_stairs` reference motion into shorter motions that
climb only the first K steps, so each one can be paired with a K-step staircase
mesh (scripts/make_n_step_staircase.py --num_boxes K) for matched-geometry data
collection.

Why crop is needed (not just shorten the episode at collect time):
  The robot in the staircase task is teleported onto the reference motion at reset
  and then *tracks* it. The staircase mesh is a fixed, matched pair with the motion
  (feet were retargeted to land on the boxes). If we render only K boxes but keep
  the full motion, the frames after step K command the swing foot UP toward box K+1,
  which no longer exists -> tracking error -> the episode terminates (fall) instead
  of timing out, and `collect_dataset.py` only saves time_out episodes. So we must
  cut each motion *before* the robot starts reaching for the (now absent) next box.

What this writes (one .npz per K, same key/layout as the source so MotionLoader
reads it unchanged):
    walk_up_<K>step.npz   for K in --steps   (e.g. 1 2)
The 3-step case is just the original motion -- pass --steps 1 2 and keep using the
original motion.npz for the 3-step dataset (or --steps 1 2 3 to emit a copy).

Crop boundary: for each K we find the contiguous window where the support foot is
planted on box K and the *swing* foot has not yet lifted toward box K+1, and cut at
the last such frame. Velocities are kept as-is through the crop; with --hold_s > 0
a frozen, zero-velocity tail of the final pose is appended so the robot has a
settled "standing on step K" reference (useful to include standing frames in the
data; set 0 for a pure mid-stride cut).

Defaults are auto-detected and printed; override any boundary with --end_frame_K.
A PNG of pelvis-z / support-foot-z with the crop lines is written next to the
outputs for a quick sanity check before committing to a 2000-episode collection.

Usage:
    python scripts/crop_motion_steps.py \
        --motion artifacts/walk_up_karen_stairs:v0/motion.npz \
        --out_dir artifacts/cropped_motions \
        --steps 1 2 --hold_s 0.6
"""
import argparse
import os

import numpy as np

# Box top heights (LOCAL z) from make_n_step_staircase.py constants. The staircase
# is placed with a pure yaw rotation at world z=0, so world box-top z == local z.
RISE = 0.17558225
STEP1_OFFSET = 0.02
BOX_TOP = {k: RISE * k + (STEP1_OFFSET if k == 1 else 0.0) for k in (1, 2, 3)}

# Per-frame arrays in the motion npz (axis 0 = time). Everything else (e.g. fps)
# is copied through unchanged.
TIME_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
# Velocity arrays zeroed during the frozen hold tail.
VEL_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def detect_feet(body_pos_w):
    """Return (left_idx, right_idx) body indices for the two foot links.

    Feet are the two bodies whose low (2nd-percentile) z is smallest -- i.e. the
    ones that actually rest on the ground. We then label them left/right by their
    lateral offset from the pelvis (body 0) in the motion's own frame.
    """
    z = body_pos_w[:, :, 2]
    low = np.percentile(z, 2, axis=0)
    feet = np.argsort(low)[:2]
    # Lateral (xy) offset from pelvis, averaged over the flat approach; sign splits L/R.
    pelvis_xy = body_pos_w[:, 0, :2]
    off = (body_pos_w[:, feet, :2] - pelvis_xy[:, None, :]).mean(axis=0)  # (2,2)
    # Use the larger-variance lateral axis to order them deterministically.
    lateral = off[:, np.argmax(off.var(axis=0))]
    a, b = feet[np.argsort(lateral)]
    return int(a), int(b)


def support_and_swing(body_pos_w, foot_idx):
    """Return (support_z, swing_z): per-frame min/max z over the two foot links."""
    fz = body_pos_w[:, list(foot_idx), 2]
    return fz.min(axis=1), fz.max(axis=1)


def auto_end_frame(support_z, swing_z, ground_z, k, tol_commit=0.06, swing_margin=0.12):
    """Frame of the best double-support stance on box k.

    The reference never pauses on box 1 or 2 -- the robot flows through, and by the
    time the support foot is settled the swing foot is already reaching for the next
    box. Freezing such a mid-lift frame gives a floating-foot target. Instead we want
    the brief double-support instant: the support foot is on box k AND the swing foot
    has come back down nearest to it (both feet planted together), which happens just
    after the lead foot lands and before the trailing foot lifts toward box k+1.

    We take the first contiguous window where the support foot is committed to box k
    and the swing foot has not yet lifted past it, then return the frame in that
    window where the swing foot is LOWEST (closest to a balanced two-foot stance).
    Returns -1 if no such window exists.
    """
    commit_thr = ground_z + BOX_TOP[k] - tol_commit
    reach_thr = ground_z + BOX_TOP[k] + swing_margin
    on_step = (support_z >= commit_thr) & (swing_z < reach_thr)
    if not on_step.any():
        return -1
    first = int(np.argmax(on_step))
    end = first
    while end + 1 < len(on_step) and on_step[end + 1]:
        end += 1
    # within [first, end], the double-support frame = swing foot lowest.
    window = slice(first, end + 1)
    return first + int(np.argmin(swing_z[window]))


def crop_one(data, start_frame, end_frame, hold_frames):
    """Return a dict of cropped arrays: frames [start_frame, end_frame] plus an
    optional frozen, zero-velocity hold tail of length hold_frames.

    start_frame drops the flat-ground approach so frame 0 of the output is already
    near the stairs -- so even the 10%-forced-to-frame-0 reset in the command term's
    adaptive sampling spawns at the climb, not on flat ground."""
    out = {}
    for key in data.files:
        if key not in TIME_KEYS:
            out[key] = data[key]
            continue
        arr = np.asarray(data[key])[start_frame : end_frame + 1].copy()
        if hold_frames > 0:
            last = arr[-1:].repeat(hold_frames, axis=0)
            if key in VEL_KEYS:
                last[:] = 0.0
            arr = np.concatenate([arr, last], axis=0)
        out[key] = arr
    return out


def main():
    ap = argparse.ArgumentParser(description="Crop the 3-step staircase motion into K-step motions.")
    ap.add_argument("--motion", required=True, help="Source motion.npz (the full 3-step climb).")
    ap.add_argument("--out_dir", required=True, help="Directory to write walk_up_<K>step.npz into.")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2],
                    help="Which K-step motions to emit (1, 2, and/or 3).")
    ap.add_argument("--start_frame", type=int, default=120,
                    help="Drop this many leading flat-ground approach frames from every output "
                         "motion (frame 0 of the output then sits just before the climb). "
                         "Matches the min_sample_idx~120 convention used for the 3-step climb.")
    ap.add_argument("--hold_s", type=float, default=0.6,
                    help="Seconds of frozen, zero-velocity 'standing on step K' tail to append (0 = pure cut).")
    for k in (1, 2, 3):
        ap.add_argument(f"--end_frame_{k}", type=int, default=None,
                        help=f"Override the auto-detected crop end frame for the {k}-step motion.")
    ap.add_argument("--no_plot", action="store_true", help="Skip the diagnostic PNG.")
    args = ap.parse_args()

    data = np.load(args.motion)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    bpos = data["body_pos_w"]
    T = bpos.shape[0]
    hold_frames = int(round(args.hold_s * fps))

    foot_idx = detect_feet(bpos)
    support_z, swing_z = support_and_swing(bpos, foot_idx)
    ground_z = float(np.median(support_z[: max(10, int(0.5 * fps))]))

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] source: {args.motion}  ({T} frames @ {fps:g} fps = {T/fps:.2f}s)")
    print(f"[INFO] foot body indices (L,R): {foot_idx}   ground support z ~{ground_z:.3f}")
    print(f"[INFO] box-top heights (world z): " + ", ".join(f"box{k}={ground_z+BOX_TOP[k]:.3f}" for k in (1, 2, 3)))

    overrides = {1: args.end_frame_1, 2: args.end_frame_2, 3: args.end_frame_3}
    end_frames = {}
    for k in args.steps:
        if k == 3 and overrides[3] is None:
            end = T - 1  # full motion already stands on box 3
        elif overrides[k] is not None:
            end = overrides[k]
        else:
            end = auto_end_frame(support_z, swing_z, ground_z, k)
            if end < 0:
                raise SystemExit(
                    f"[ERROR] Could not auto-detect a crop frame for {k}-step "
                    f"(support never settles on box{k}). Pass --end_frame_{k} explicitly "
                    f"after inspecting the diagnostic plot."
                )
        if end <= args.start_frame:
            raise SystemExit(
                f"[ERROR] {k}-step crop end ({end}) <= --start_frame ({args.start_frame}); "
                f"the trim would remove the whole climb. Lower --start_frame or override --end_frame_{k}."
            )
        end_frames[k] = end
        k_hold = hold_frames if k != 3 else 0
        out = crop_one(data, args.start_frame, end, k_hold)
        out_path = os.path.join(args.out_dir, f"walk_up_{k}step.npz")
        np.savez(out_path, **out)
        n_out = out["body_pos_w"].shape[0]
        print(f"[OK] {k}-step: frames [{args.start_frame}, {end}] ({(end-args.start_frame)/fps:.2f}s) "
              f"+ {k_hold} hold -> {n_out} frames  ({out_path})")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            t = np.arange(T) / fps
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(t, bpos[:, 0, 2], label="pelvis z", color="k", lw=1)
            ax.plot(t, support_z, label="support foot z (min)", color="tab:blue", lw=1)
            ax.plot(t, swing_z, label="swing foot z (max)", color="tab:cyan", lw=0.8, alpha=0.7)
            for k in (1, 2, 3):
                ax.axhline(ground_z + BOX_TOP[k], color="gray", ls=":", lw=0.8)
                ax.text(0.02, ground_z + BOX_TOP[k] + 0.005, f"box{k}", color="gray", fontsize=8)
            ax.axvspan(0, args.start_frame / fps, color="gray", alpha=0.15)
            ax.axvline(args.start_frame / fps, color="dimgray", lw=1.2, ls="--",
                       label=f"start trim @f{args.start_frame}")
            colors = {1: "tab:red", 2: "tab:green", 3: "tab:purple"}
            for k, end in end_frames.items():
                ax.axvline(end / fps, color=colors.get(k, "tab:orange"), lw=1.5, label=f"{k}-step cut @f{end}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("world z (m)")
            ax.legend(loc="upper left", fontsize=8, ncol=2)
            ax.set_title("Staircase motion crop boundaries")
            png = os.path.join(args.out_dir, "crop_boundaries.png")
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            print(f"[OK] diagnostic plot -> {png}")
        except Exception as e:  # plotting is best-effort
            print(f"[WARN] could not write diagnostic plot: {e}")


if __name__ == "__main__":
    main()
