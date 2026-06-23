#!/usr/bin/env python
"""Encode every clip in a Zarr motion store into VAE latents and dump per-clip
summaries (mean + std of per-window z over the clip) for Phase 5 K-means.

Output: an .npz file with keys
    clip_names: (N_clips,) array of strings
    z_mean:     (N_clips, latent_dim) float32
    z_std:      (N_clips, latent_dim) float32

Usage:
    python scripts/dump_motion_latents.py \
        --zarr_path /move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
        --vae_ckpt logs/motion_vae/v1/motion_vae.pt \
        --output_path logs/motion_vae/v1/clip_latents.npz \
        --batch_size 256

This script reads the VAE's checkpoint metadata to get layout, window,
latent_dim, and normalization stats — so it works with any VAE trained by
`scripts/train_motion_vae.py` without re-specifying those settings.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import zarr as _zarr

# Load motion_vae.py by file path so we DON'T trigger
# `whole_body_tracking/__init__.py` (which imports .tasks → Isaac Lab →
# `omni.log`, only available under Isaac Sim's AppLauncher).
import importlib.util as _importlib_util  # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOTION_VAE_PATH = os.path.abspath(os.path.join(
    _HERE, "..", "source", "whole_body_tracking",
    "whole_body_tracking", "utils", "motion_vae.py",
))
_spec = _importlib_util.spec_from_file_location("motion_vae_mod", _MOTION_VAE_PATH)
_motion_vae = _importlib_util.module_from_spec(_spec)
sys.modules["motion_vae_mod"] = _motion_vae
_spec.loader.exec_module(_motion_vae)
FeatureLayout = _motion_vae.FeatureLayout
MotionVAE = _motion_vae.MotionVAE
ZarrMotionWindowDataset = _motion_vae.ZarrMotionWindowDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--per_clip_window_stride", type=int, default=8,
                   help="When encoding a clip, take a window every N frames "
                        "and average. Smaller = more samples, larger = faster.")
    args = p.parse_args()

    print(f"[dump] loading VAE checkpoint: {args.vae_ckpt}", flush=True)
    ckpt = torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)
    layout = FeatureLayout(
        body_names=list(ckpt["layout"]["body_names"]),
        joint_names=list(ckpt["layout"]["joint_names"]),
    )
    window = int(ckpt["window"])
    latent_dim = int(ckpt["latent_dim"])
    hidden = tuple(ckpt.get("hidden_dims", (512, 256)))
    norm_mean = np.asarray(ckpt["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(ckpt["norm_std"], dtype=np.float32)

    device = torch.device(args.device)
    model = MotionVAE(window, layout.per_frame_dim, latent_dim, hidden_dims=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build a lightweight reader over the zarr (no train/val split; iterate all clips).
    store = _zarr.open(args.zarr_path, mode="r")
    clip_start = store["clip_start_idx"][:]
    clip_end = store["clip_end_idx"][:]
    if "clip_names" in store:
        clip_names = [str(n) for n in store["clip_names"][:]]
    else:
        clip_names = [f"clip_{i}" for i in range(len(clip_start))]

    # Reuse the dataset's _fetch_window logic (with normalization) per clip.
    helper = ZarrMotionWindowDataset.__new__(ZarrMotionWindowDataset)
    helper.zarr_path = args.zarr_path
    helper.store = store
    helper.layout = layout
    helper.window = window
    helper.mean = norm_mean
    helper.std = norm_std
    # body_idx resolution: zarr stores body_names as an ARRAY (not an attr).
    # Mirrors the fix in ZarrMotionWindowDataset.
    zarr_body_names = None
    if "body_names" in store:
        zarr_body_names = [str(n) for n in store["body_names"][:]]
    elif "body_names" in store.attrs:
        zarr_body_names = list(store.attrs["body_names"])
    if zarr_body_names is not None:
        missing = [n for n in layout.body_names if n not in zarr_body_names]
        if missing:
            raise RuntimeError(
                f"layout.body_names not present in zarr body_names: {missing}. "
                f"Available ({len(zarr_body_names)}): {zarr_body_names}"
            )
        helper.body_idx = np.array(
            [zarr_body_names.index(n) for n in layout.body_names], dtype=np.int64
        )
        print(f"[dump]   subsetting bodies via body_names array: "
              f"{len(zarr_body_names)} -> {len(layout.body_names)}", flush=True)
    else:
        helper.body_idx = None

    n_clips = len(clip_start)
    z_mean = np.zeros((n_clips, latent_dim), dtype=np.float32)
    z_std = np.zeros((n_clips, latent_dim), dtype=np.float32)

    stride = max(1, int(args.per_clip_window_stride))

    print(f"[dump] encoding {n_clips} clips, window={window} stride={stride}", flush=True)

    for ci in range(n_clips):
        clip_len = int(clip_end[ci] - clip_start[ci])
        if clip_len < window:
            # too short — skip; latents stay at 0 (downstream K-means will see).
            continue
        # Window starts: 0, stride, 2*stride, ...
        starts = list(range(0, clip_len - window + 1, stride))
        # Build a batch of all windows for this clip.
        windows = np.zeros((len(starts), window, layout.per_frame_dim), dtype=np.float32)
        # Read the whole clip's tensors once, slice locally.
        s = int(clip_start[ci]); e = int(clip_end[ci])
        body_pos  = store["body_pos_w"][s:e]
        body_quat = store["body_quat_w"][s:e]
        body_lin  = store["body_lin_vel_w"][s:e]
        body_ang  = store["body_ang_vel_w"][s:e]
        joint_pos = store["joint_pos"][s:e]
        joint_vel = store["joint_vel"][s:e]
        if helper.body_idx is not None:
            body_pos  = body_pos[:, helper.body_idx]
            body_quat = body_quat[:, helper.body_idx]
            body_lin  = body_lin[:, helper.body_idx]
            body_ang  = body_ang[:, helper.body_idx]
        for j, st in enumerate(starts):
            local_pos  = body_pos[st:st+window]
            local_quat = body_quat[st:st+window]
            local_lin  = body_lin[st:st+window]
            local_ang  = body_ang[st:st+window]
            local_jp   = joint_pos[st:st+window]
            local_jv   = joint_vel[st:st+window]
            pelvis = local_pos[:, 0:1, :]
            local_pos = local_pos - pelvis
            per_body = np.concatenate(
                [local_pos.reshape(window, -1), local_quat.reshape(window, -1),
                 local_lin.reshape(window, -1), local_ang.reshape(window, -1)], axis=-1,
            )
            windows[j] = np.concatenate([per_body, local_jp, local_jv], axis=-1)
        # normalize
        windows = (windows - norm_mean) / norm_std
        # encode in mini-batches
        zs = []
        with torch.no_grad():
            for k in range(0, len(starts), args.batch_size):
                xb = torch.from_numpy(windows[k:k+args.batch_size]).to(device, non_blocking=True)
                mu, _logvar = model.encode(xb)
                zs.append(mu.cpu().numpy())
        if zs:
            z_all = np.concatenate(zs, axis=0)  # (n_windows, latent_dim)
            z_mean[ci] = z_all.mean(axis=0)
            z_std[ci] = z_all.std(axis=0)

        if (ci + 1) % 100 == 0:
            print(f"[dump]   {ci+1}/{n_clips}", flush=True)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez(
        args.output_path,
        clip_names=np.array(clip_names, dtype=object),
        z_mean=z_mean,
        z_std=z_std,
    )
    print(f"[dump] saved {n_clips} per-clip latents to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
