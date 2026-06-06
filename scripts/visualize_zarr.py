#!/usr/bin/env python3
"""Inspect and visualize a Staircase-G1-Collect-v0 dataset zarr.

The zarr is produced by ``scripts/rsl_rl/collect_dataset.py`` (see
``scripts/multi_collect.py``). It has this layout::

    <run>.zarr/
      meta/
        episode_ends, num_episodes, collect_steps, decimation,
        min_delay, max_delay, noise_level, hip_noise, knee_noise,
        ankle_noise, wandb_run, timestamp
      data/
        body_pos        (T, 90)   30 bodies x 3 (world frame)
        body_rot        (T, 120)  30 x 4 quaternion
        body_lin_vel    (T, 90)
        body_ang_vel    (T, 90)
        joint_pos       (T, 29)
        joint_vel       (T, 29)
        root_pos        (T, 3)
        root_rot        (T, 4)
        act             (T, 29)
        rgb_embed       (T, 768)  SigLIP2 base pooled (head RGB camera)
        rgb_embed_flipped (T, 768)
        depth_embed     (T, 1024) DeFM ViT-L/14 class token (head depth)
        depth_embed_flipped (T, 1024)

Usage examples::

    # Summary + overview + plots for 5 episodes spread across the dataset
    python scripts/visualize_zarr.py STAIRCASE_DATA_COLLECTED/<run>.zarr

    # Specific episodes + a 3D body-points animation gif for each
    python scripts/visualize_zarr.py STAIRCASE_DATA_COLLECTED/<run>.zarr \\
        --episodes 17 42

    # Pick the most recent .zarr in a folder automatically; skip the gifs for a quick look
    python scripts/visualize_zarr.py STAIRCASE_DATA_COLLECTED --no-animate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.animation import FuncAnimation, PillowWriter

NUM_BODIES = 30
NUM_JOINTS = 29

STAIRCASE_POSITION = np.array([3.2, -0.2, 0.0])
STAIRCASE_YAW_DEG = 94.0
STAIRCASE_HALF_EXTENTS_XY = np.array([1.5, 0.5])


def resolve_zarr_path(path: Path) -> Path:
    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zgroup").exists()):
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.zarr"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No .zarr stores in {path}")
        chosen = candidates[-1]
        print(f"[info] No explicit zarr; using newest in dir: {chosen.name}")
        return chosen
    raise FileNotFoundError(path)


def _read_full(arr) -> np.ndarray:
    return np.asarray(arr[...])


def episode_bounds(root):
    ends = _read_full(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    return starts, ends.astype(np.int64)


def print_summary(root, zarr_path: Path) -> None:
    print(f"\n=== {zarr_path} ===")
    print("\n[meta]")
    for k in sorted(root["meta"].keys()):
        v = _read_full(root["meta"][k])
        if v.ndim == 0:
            print(f"  {k:20s} = {v}")
        elif v.size <= 8:
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype} value={v.tolist()}")
        else:
            print(f"  {k:20s} shape={tuple(v.shape)} dtype={v.dtype} "
                  f"min={v.min()} max={v.max()} mean={v.mean():.2f}")

    print("\n[data]")
    for k in sorted(root["data"].keys()):
        arr = root["data"][k]
        print(f"  {k:25s} shape={tuple(arr.shape)} dtype={arr.dtype} chunks={arr.chunks}")

    starts, ends = episode_bounds(root)
    if len(ends) == 0:
        print("\n[episodes] none")
        return
    lens = ends - starts
    print(
        f"\n[episodes] n={len(lens)} total_steps={int(ends[-1])} "
        f"len min={int(lens.min())} max={int(lens.max())} "
        f"mean={lens.mean():.1f} std={lens.std():.1f}"
    )


def load_episode(root, ep_idx: int):
    starts, ends = episode_bounds(root)
    if not 0 <= ep_idx < len(starts):
        raise IndexError(f"episode {ep_idx} out of range [0, {len(starts)})")
    s, e = int(starts[ep_idx]), int(ends[ep_idx])
    ep = {k: np.asarray(root["data"][k][s:e]) for k in root["data"].keys()}
    ep["body_pos"] = ep["body_pos"].reshape(-1, NUM_BODIES, 3)
    ep["body_rot"] = ep["body_rot"].reshape(-1, NUM_BODIES, 4)
    ep["body_lin_vel"] = ep["body_lin_vel"].reshape(-1, NUM_BODIES, 3)
    ep["body_ang_vel"] = ep["body_ang_vel"].reshape(-1, NUM_BODIES, 3)
    return ep, (s, e)


def stair_footprint_xy() -> np.ndarray:
    yaw = np.deg2rad(STAIRCASE_YAW_DEG)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    hx, hy = STAIRCASE_HALF_EXTENTS_XY
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return corners @ R.T + STAIRCASE_POSITION[:2]


def _draw_footprint(ax, alpha=0.18, label="staircase (approx)"):
    foot = stair_footprint_xy()
    closed = np.vstack([foot, foot[:1]])
    ax.fill(closed[:, 0], closed[:, 1], color="orange", alpha=alpha, label=label)


def plot_overview(root, out_path: Path, max_eps: int = 200) -> None:
    starts, ends = episode_bounds(root)
    n_eps = len(starts)
    if n_eps == 0:
        print("[skip] no episodes to plot")
        return
    lens = ends - starts
    rng = np.random.default_rng(0)
    sample = rng.choice(n_eps, size=min(n_eps, max_eps), replace=False)
    root_pos_all = root["data/root_pos"]

    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(lens, bins=min(40, max(5, n_eps // 4)), color="steelblue")
    ax1.set_xlabel("episode length (steps)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Episode lengths (n={n_eps})")

    ax2 = fig.add_subplot(2, 3, 2)
    for ep in sample:
        rp = np.asarray(root_pos_all[starts[ep]:ends[ep]])
        ax2.plot(rp[:, 0], rp[:, 1], lw=0.5, alpha=0.4)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title(f"Root XY ({len(sample)} sampled eps)")
    _draw_footprint(ax2)
    ax2.scatter(*STAIRCASE_POSITION[:2], c="orange", s=30, marker="x")
    ax2.legend(loc="best", fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    for ep in sample[:50]:
        rp = np.asarray(root_pos_all[starts[ep]:ends[ep], 2])
        ax3.plot(np.linspace(0, 1, len(rp)), rp, lw=0.5, alpha=0.4)
    ax3.set_xlabel("t / T")
    ax3.set_ylabel("pelvis z (m)")
    ax3.set_title("Root z vs normalized time")

    ep0_s, ep0_e = int(starts[0]), int(ends[0])
    jp0 = np.asarray(root["data/joint_pos"][ep0_s:ep0_e])
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(jp0.T, aspect="auto", cmap="viridis", origin="lower")
    ax4.set_xlabel("step")
    ax4.set_ylabel("joint idx")
    ax4.set_title(f"joint_pos heatmap (ep 0, T={jp0.shape[0]})")
    fig.colorbar(im4, ax=ax4, fraction=0.04)

    act0 = np.asarray(root["data/act"][ep0_s:ep0_e])
    ax5 = fig.add_subplot(2, 3, 5)
    lim = float(np.percentile(np.abs(act0), 99)) if act0.size else 1.0
    lim = max(lim, 1e-3)
    im5 = ax5.imshow(act0.T, aspect="auto", cmap="coolwarm", origin="lower", vmin=-lim, vmax=lim)
    ax5.set_xlabel("step")
    ax5.set_ylabel("joint idx")
    ax5.set_title(f"act heatmap (ep 0)  |  |a|_99 = {lim:.3f}")
    fig.colorbar(im5, ax=ax5, fraction=0.04)

    ax6 = fig.add_subplot(2, 3, 6)
    if "rgb_embed" in root["data"]:
        rgb_n = np.linalg.norm(np.asarray(root["data/rgb_embed"][ep0_s:ep0_e]), axis=-1)
        ax6.plot(rgb_n, label="||rgb_embed||", color="C0")
    if "depth_embed" in root["data"]:
        d_n = np.linalg.norm(np.asarray(root["data/depth_embed"][ep0_s:ep0_e]), axis=-1)
        ax6.plot(d_n, label="||depth_embed||", color="C3")
    ax6.set_xlabel("step")
    ax6.set_ylabel("L2 norm")
    ax6.set_title("Vision embedding norms (ep 0)")
    ax6.legend(fontsize=8)

    fig.suptitle(out_path.parent.name + "  —  overview", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_episode(root, ep_idx: int, out_path: Path) -> None:
    ep, (s, e) = load_episode(root, ep_idx)
    T = ep["body_pos"].shape[0]

    fig = plt.figure(figsize=(15, 9))

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    for b in range(NUM_BODIES):
        ax3d.plot(
            ep["body_pos"][:, b, 0],
            ep["body_pos"][:, b, 1],
            ep["body_pos"][:, b, 2],
            lw=0.6, alpha=0.6,
        )
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title(f"{NUM_BODIES} body trajectories, ep {ep_idx} (T={T})")

    ax_xy = fig.add_subplot(2, 3, 2)
    root_xy = ep["body_pos"][:, 0, :2]
    ax_xy.plot(root_xy[:, 0], root_xy[:, 1], "k-", lw=1.2)
    ax_xy.scatter(root_xy[0, 0], root_xy[0, 1], c="g", s=40, zorder=3, label="start")
    ax_xy.scatter(root_xy[-1, 0], root_xy[-1, 1], c="r", s=40, zorder=3, label="end")
    _draw_footprint(ax_xy, alpha=0.25)
    ax_xy.set_aspect("equal"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_title("Root XY this episode")
    ax_xy.legend(fontsize=8)

    ax_z = fig.add_subplot(2, 3, 3)
    ax_z.plot(ep["body_pos"][:, 0, 2], "k-", label="pelvis z")
    foot_idx_left = min(3, NUM_BODIES - 1)
    foot_idx_right = min(9, NUM_BODIES - 1)
    ax_z.plot(ep["body_pos"][:, foot_idx_left, 2], "C0", alpha=0.6, label=f"body[{foot_idx_left}] z")
    ax_z.plot(ep["body_pos"][:, foot_idx_right, 2], "C3", alpha=0.6, label=f"body[{foot_idx_right}] z")
    ax_z.set_xlabel("step"); ax_z.set_ylabel("z (m)")
    ax_z.set_title("Heights")
    ax_z.legend(fontsize=8)

    ax_jp = fig.add_subplot(2, 3, 4)
    ax_jp.plot(ep["joint_pos"], lw=0.6)
    ax_jp.set_xlabel("step"); ax_jp.set_ylabel("joint pos (rel)")
    ax_jp.set_title(f"joint_pos ({ep['joint_pos'].shape[1]} dofs)")

    ax_jv = fig.add_subplot(2, 3, 5)
    ax_jv.plot(ep["joint_vel"], lw=0.6)
    ax_jv.set_xlabel("step"); ax_jv.set_ylabel("joint vel")
    ax_jv.set_title("joint_vel")

    ax_a = fig.add_subplot(2, 3, 6)
    ax_a.plot(ep["act"], lw=0.6)
    ax_a.set_xlabel("step"); ax_a.set_ylabel("action (policy output, pre-noise)")
    ax_a.set_title("act")

    fig.suptitle(f"episode {ep_idx}  steps {s}..{e}  ({T} frames)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def animate_episode(
    root,
    ep_idx: int,
    out_path: Path,
    fps: int = 25,
    decimation: int = 1,
) -> None:
    ep, _ = load_episode(root, ep_idx)
    pos = ep["body_pos"][::max(1, decimation)]
    T = pos.shape[0]

    pad = 0.3
    mins = pos.reshape(-1, 3).min(0) - pad
    maxs = pos.reshape(-1, 3).max(0) + pad
    foot = stair_footprint_xy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], pos[0, :, 2], s=22, c="C0", depthshade=True)

    closed = np.vstack([foot, foot[:1]])
    ax.plot(closed[:, 0], closed[:, 1], np.zeros(len(closed)), "orange", lw=2, alpha=0.7)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(min(0.0, mins[2]), maxs[2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    title = ax.set_title("")

    def update(t):
        p = pos[t]
        scat._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        title.set_text(f"ep {ep_idx}  frame {t}/{T - 1}")
        return [scat, title]

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("zarr_path", type=Path, help="Path to a .zarr (or a folder containing .zarr's)")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices to plot/animate (default: 5 evenly spread across the dataset)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (default: <zarr>_viz next to the store)")
    parser.add_argument("--no-animate", action="store_true",
                        help="Skip the per-episode 3D body-points animation gifs (saved by default)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--decimation", type=int, default=1,
                        help="Keep every Nth frame when animating")
    parser.add_argument("--max-eps-overview", type=int, default=200,
                        help="Cap number of episodes drawn in the overview XY plot")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--no-overview", action="store_true")
    parser.add_argument("--no-episode", action="store_true")
    args = parser.parse_args()

    zpath = resolve_zarr_path(args.zarr_path)
    root = zarr.open(str(zpath), mode="r")
    out = args.out or zpath.parent / (zpath.stem + "_viz")
    out.mkdir(parents=True, exist_ok=True)

    starts, _ = episode_bounds(root)
    n_eps = len(starts)
    if args.episodes is not None:
        episodes = args.episodes
    else:
        k = min(5, n_eps)
        episodes = sorted(set(np.linspace(0, n_eps - 1, k).round().astype(int).tolist()))
        print(f"[info] No --episodes given; visualizing {len(episodes)} spread across dataset: {episodes}")

    if not args.no_summary:
        print_summary(root, zpath)
    if not args.no_overview:
        plot_overview(root, out / "overview.png", max_eps=args.max_eps_overview)
    if not args.no_episode:
        for ep_idx in episodes:
            plot_episode(root, ep_idx, out / f"episode_{ep_idx:04d}.png")
    if not args.no_animate:
        for ep_idx in episodes:
            animate_episode(
                root,
                ep_idx,
                out / f"episode_{ep_idx:04d}.gif",
                fps=args.fps,
                decimation=args.decimation,
            )


if __name__ == "__main__":
    main()
