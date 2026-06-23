#!/usr/bin/env python
"""Train a VMP-style motion VAE on a Zarr motion store.

No Isaac Lab dependency — runs on any box with PyTorch + zarr.

Usage:
    python scripts/train_motion_vae.py \
        --zarr_path /move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr \
        --window 32 --latent_dim 16 \
        --batch_size 256 --epochs 10 --lr 3e-4 --kl_target 1e-3 \
        --num_workers 4 \
        --output_dir logs/motion_vae/v1 \
        --symmetric_augment

Outputs (in --output_dir):
    motion_vae.pt              : checkpoint with model + normalizer + meta
    train_log.csv              : per-epoch loss / recon / kl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Load motion_vae.py by file path so we DON'T trigger
# `whole_body_tracking/__init__.py` (which imports .tasks → Isaac Lab →
# `omni.log`, only available when launched via Isaac Sim's AppLauncher).
# The VAE pipeline is pure PyTorch + zarr and has no Isaac dependency.
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
ZarrMotionWindowDataset = _motion_vae.ZarrMotionWindowDataset
MotionVAE = _motion_vae.MotionVAE
vae_loss = _motion_vae.vae_loss
build_window_reflect_op = _motion_vae.build_window_reflect_op
sym_augment_batch = _motion_vae.sym_augment_batch


# Default G1 layout — matches generalist task's body / joint name lists.
# Body subset = the 14 tracked bodies used by the motion command.
G1_BODY_SUBSET = [
    "pelvis",
    "left_hip_roll_link",  "left_knee_link",  "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",  "left_elbow_link",  "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
G1_FULL_JOINTS = [
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",  "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",  "left_wrist_roll_joint",  "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--kl_target", type=float, default=1e-3,
                   help="Target β (KL weight). Ramped from 0 over --kl_warmup_epochs.")
    p.add_argument("--kl_warmup_epochs", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--symmetric_augment", action="store_true",
                   help="Apply 50% y-mirror augmentation per window during training.")
    p.add_argument("--sym_aug_prob", type=float, default=0.5)
    p.add_argument("--include_keywords", type=str, default=None,
                   help="Optional comma-separated clip-name keywords (case-insensitive).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_train_batches", type=int, default=0,
                   help="If >0, train at most this many batches per epoch (smoke test).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    layout = FeatureLayout(body_names=G1_BODY_SUBSET, joint_names=G1_FULL_JOINTS)
    print(f"[VAE] feature layout: {layout.num_bodies} bodies × ({layout.per_body_dim} per body)"
          f" + 2 × {layout.num_joints} joints = {layout.per_frame_dim}/frame")
    print(f"[VAE] window={args.window}, flat dim = {args.window * layout.per_frame_dim}")

    include_kws = [k.strip() for k in args.include_keywords.split(",")] if args.include_keywords else None
    train_ds = ZarrMotionWindowDataset(
        args.zarr_path, layout, window=args.window,
        clip_split="train", val_frac=args.val_frac, split_seed=args.split_seed,
        include_keywords=include_kws,
    )
    val_ds = ZarrMotionWindowDataset(
        args.zarr_path, layout, window=args.window,
        clip_split="val", val_frac=args.val_frac, split_seed=args.split_seed,
        include_keywords=include_kws,
    )

    # Share normalization stats: val should be normalized with train's mean/std.
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False,
    )

    device = torch.device(args.device)
    hidden = tuple(int(h) for h in args.hidden_dims.split(","))
    model = MotionVAE(args.window, layout.per_frame_dim, args.latent_dim, hidden_dims=hidden).to(device)
    print(f"[VAE] model params: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Sym-aug op (CPU; moved to device per batch).
    sym_op = build_window_reflect_op(layout).to(device) if args.symmetric_augment else None

    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_f = open(log_path, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "train_loss", "train_recon", "train_kl",
                    "val_loss", "val_recon", "val_kl", "beta"])
    log_f.flush()

    for epoch in range(args.epochs):
        beta = args.kl_target * min(1.0, epoch / max(1, args.kl_warmup_epochs))

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_sum = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x = batch.to(device, non_blocking=True)
            if sym_op is not None:
                x = sym_augment_batch(x, sym_op, prob=args.sym_aug_prob)
            x_recon, mu, logvar = model(x)
            losses = vae_loss(x, x_recon, mu, logvar, beta=beta)
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            n = x.shape[0]
            tr_sum["loss"]  += float(losses["loss"].item()) * n
            tr_sum["recon"] += float(losses["recon"].item()) * n
            tr_sum["kl"]    += float(losses["kl"].item()) * n
            tr_sum["n"]     += n
            if bi % 50 == 0:
                print(f"[epoch {epoch} batch {bi}] loss={float(losses['loss']):.5f} "
                      f"recon={float(losses['recon']):.5f} kl={float(losses['kl']):.4f} beta={beta:.1e}",
                      flush=True)
        tr_loss = tr_sum["loss"] / max(tr_sum["n"], 1)
        tr_recon = tr_sum["recon"] / max(tr_sum["n"], 1)
        tr_kl = tr_sum["kl"] / max(tr_sum["n"], 1)

        # ── val ────────────────────────────────────────────────────────────
        model.eval()
        va_sum = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device, non_blocking=True)
                x_recon, mu, logvar = model(x)
                losses = vae_loss(x, x_recon, mu, logvar, beta=beta)
                n = x.shape[0]
                va_sum["loss"]  += float(losses["loss"].item()) * n
                va_sum["recon"] += float(losses["recon"].item()) * n
                va_sum["kl"]    += float(losses["kl"].item()) * n
                va_sum["n"]     += n
        va_loss = va_sum["loss"] / max(va_sum["n"], 1)
        va_recon = va_sum["recon"] / max(va_sum["n"], 1)
        va_kl = va_sum["kl"] / max(va_sum["n"], 1)

        log_w.writerow([epoch, tr_loss, tr_recon, tr_kl, va_loss, va_recon, va_kl, beta])
        log_f.flush()
        print(f"[VAE] epoch {epoch}  train loss={tr_loss:.5f} recon={tr_recon:.5f} kl={tr_kl:.4f}"
              f"   val loss={va_loss:.5f} recon={va_recon:.5f} kl={va_kl:.4f}  beta={beta:.1e}",
              flush=True)

        # ── save checkpoint each epoch (overwrites) ────────────────────────
        ckpt = {
            "model_state_dict": model.state_dict(),
            "norm_mean": train_ds.mean,
            "norm_std": train_ds.std,
            "layout": {"body_names": layout.body_names, "joint_names": layout.joint_names},
            "window": args.window,
            "latent_dim": args.latent_dim,
            "hidden_dims": list(hidden),
            "per_frame_dim": layout.per_frame_dim,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "motion_vae.pt"))

    log_f.close()
    print(f"[VAE] done. checkpoint: {os.path.join(args.output_dir, 'motion_vae.pt')}")


if __name__ == "__main__":
    main()
