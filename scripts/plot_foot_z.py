"""Plot robot vs reference foot z position from a play_bones_clip CSV.

Usage:
    python scripts/plot_foot_z.py path/to/bones_clip.csv
    python scripts/plot_foot_z.py path/to/bones_clip.csv --out foot_z.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Per-step CSV from play_bones_clip.py")
    parser.add_argument("--out", default=None, help="Save figure to file instead of showing")
    parser.add_argument("--fps", type=float, default=50.0, help="Simulation FPS for x-axis in seconds")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    t = df["step"].values / args.fps

    feet = [
        ("left_ankle_roll_link",  "Left foot",  "tab:blue"),
        ("right_ankle_roll_link", "Right foot", "tab:orange"),
    ]

    fig, axes = plt.subplots(len(feet), 1, figsize=(12, 4 * len(feet)), sharex=True)
    if len(feet) == 1:
        axes = [axes]

    for ax, (name, label, color) in zip(axes, feet):
        ref_col   = f"{name}_ref_z"
        robot_col = f"{name}_robot_z"
        err_col   = f"{name}_z_error"

        ax.plot(t, df[ref_col].values,   color=color, lw=1.5, label="reference", linestyle="--")
        ax.plot(t, df[robot_col].values, color=color, lw=1.5, label="robot",     alpha=0.8)
        ax.fill_between(t, df[ref_col].values, df[robot_col].values, alpha=0.15, color=color)

        # Mean error annotation
        mean_err = df[err_col].mean()
        ax.set_title(f"{label}  (mean |error| = {mean_err:.4f} m)", fontsize=11)
        ax.set_ylabel("z position (m)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time (s)")

    clip_name = os.path.splitext(os.path.basename(args.csv))[0]
    fig.suptitle(f"Foot z: robot vs reference\n{clip_name}", fontsize=12)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
