"""VMP-style reconstruction VAE on motion windows from a Zarr store.

Architecture
------------
Encoder:  MLP [W*F → 512 → 256] → (μ, log σ) heads (each `latent_dim`)
Decoder:  MLP [latent_dim → 256 → 512 → W*F]
Loss:     per-channel MSE reconstruction + β·KL with β-ramp 0→target over
          `kl_warmup_epochs` (then held constant).

`W` = window length in frames (default 32, ~1 s @ 33 Hz). `F` = per-frame
feature dim, built from the 14 tracked bodies' (pos + quat + lin_vel +
ang_vel) plus joint_pos + joint_vel. With the default G1 layout that's
14 * (3+4+3+3) + 29 + 29 = 240. Default total flat input dim = 32 * 240 = 7680.

This module is pure PyTorch + numpy + zarr — NO Isaac Lab dependency. Train
with `scripts/train_motion_vae.py` (separate process from RL).

Symmetric augmentation
----------------------
Train-time augmentation in the dataset: per-window 50% chance to reflect the
features through the SymmetricAugment per-term ops applied to the
flattened per-frame layout. Doubles effective data, lands L/R variants in
the same latent neighborhood (cleaner Phase-5 K-means clusters).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ────── per-frame feature layout ────────────────────────────────────────────


@dataclass
class FeatureLayout:
    """Defines how the (B, num_bodies, ...) and (B, num_joints) zarr arrays
    pack into a flat per-frame feature vector. Slicing is contiguous so the
    sym-aug op tables (Phase 0) work on the same layout."""
    body_names: list[str]   # tracked bodies in order
    joint_names: list[str]  # full robot joints in order

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def per_body_dim(self) -> int:
        return 3 + 4 + 3 + 3  # pos + quat + lin_vel + ang_vel

    @property
    def per_frame_dim(self) -> int:
        return self.num_bodies * self.per_body_dim + 2 * self.num_joints

    def slices(self) -> dict[str, tuple[int, int]]:
        """Return (start, end) flat indices for each per-frame feature block."""
        offsets = {}
        i = 0
        offsets["body_pos"]      = (i, i + 3 * self.num_bodies); i = offsets["body_pos"][1]
        offsets["body_quat"]     = (i, i + 4 * self.num_bodies); i = offsets["body_quat"][1]
        offsets["body_lin_vel"]  = (i, i + 3 * self.num_bodies); i = offsets["body_lin_vel"][1]
        offsets["body_ang_vel"]  = (i, i + 3 * self.num_bodies); i = offsets["body_ang_vel"][1]
        offsets["joint_pos"]     = (i, i + self.num_joints);     i = offsets["joint_pos"][1]
        offsets["joint_vel"]     = (i, i + self.num_joints);     i = offsets["joint_vel"][1]
        return offsets


# ────── Zarr-backed windowed dataset ────────────────────────────────────────


class ZarrMotionWindowDataset(Dataset):
    """Random (clip, start_frame) → window of W frames packed as per-frame
    features. Filters clips by length (skip clips shorter than W). Tracks
    body subset by name (matches the env's `body_names` from the multi-clip
    motion command cfg) and the full robot joint set.

    On-the-fly normalization: per-feature mean/std computed once at __init__
    over a random subsample of windows.
    """

    def __init__(
        self,
        zarr_path: str,
        layout: FeatureLayout,
        window: int = 32,
        clip_split: str = "train",  # "train" or "val"
        val_frac: float = 0.1,
        split_seed: int = 0,
        norm_sample: int = 4096,
        include_keywords: list[str] | None = None,
    ):
        import zarr as _zarr

        assert os.path.isdir(zarr_path), f"zarr not found: {zarr_path}"
        self.zarr_path = zarr_path
        self.layout = layout
        self.window = int(window)
        self.store = _zarr.open(zarr_path, mode="r")

        # Resolve clip subset (optionally restrict by keyword to e.g. "walk").
        all_clip_start = self.store["clip_start_idx"][:]
        all_clip_end = self.store["clip_end_idx"][:]
        total_clips = len(all_clip_start)
        if include_keywords:
            if "clip_names" not in self.store:
                raise RuntimeError(f"include_keywords given but {zarr_path} has no clip_names")
            names = [str(n) for n in self.store["clip_names"][:]]
            kws = [k.lower() for k in include_keywords]
            valid = [i for i in range(total_clips) if any(k in names[i].lower() for k in kws)]
        else:
            valid = list(range(total_clips))

        # Train/val split by CLIP id (so val clips never appear in train).
        rng = np.random.default_rng(split_seed)
        valid_arr = np.array(valid, dtype=np.int64)
        rng.shuffle(valid_arr)
        n_val = max(1, int(len(valid_arr) * val_frac))
        if clip_split == "train":
            keep = valid_arr[n_val:]
        elif clip_split == "val":
            keep = valid_arr[:n_val]
        else:
            raise ValueError(f"clip_split={clip_split!r}")

        # Per-clip frame ranges; drop clips shorter than `window`.
        clip_start = all_clip_start[keep]
        clip_end = all_clip_end[keep]
        clip_len = clip_end - clip_start
        long_enough = clip_len >= self.window
        self.clip_start = clip_start[long_enough]
        self.clip_end = clip_end[long_enough]
        self.clip_lens = self.clip_end - self.clip_start
        self.num_clips = int(self.clip_start.shape[0])

        # Total valid window starts = sum (clip_len - window + 1).
        per_clip_starts = np.maximum(0, self.clip_lens - self.window + 1)
        self.cum_starts = np.cumsum(per_clip_starts)
        self.num_windows = int(self.cum_starts[-1]) if self.num_clips > 0 else 0

        # Body name → index inside `_body_pos_w[frame, body, ...]` array.
        # The BONES zarr stores body_names as a zarr ARRAY (not an attr), and
        # the full zarr generally has more bodies than the tracked subset
        # (e.g. 30 G1 links vs the 14 in layout.body_names). Look up indices
        # by name; only fall back to identity if no body_names array exists
        # AND the existing body axis already matches layout.num_bodies.
        zarr_body_names = None
        if "body_names" in self.store:
            zarr_body_names = [str(n) for n in self.store["body_names"][:]]
        elif "body_names" in self.store.attrs:
            zarr_body_names = list(self.store.attrs["body_names"])
        if zarr_body_names is not None:
            missing = [n for n in layout.body_names if n not in zarr_body_names]
            if missing:
                raise RuntimeError(
                    f"layout.body_names not present in zarr body_names: {missing}. "
                    f"Available ({len(zarr_body_names)}): {zarr_body_names}"
                )
            self.body_idx = np.array(
                [zarr_body_names.index(n) for n in layout.body_names], dtype=np.int64
            )
            print(f"[VAE Dataset]   subsetting bodies via body_names array: "
                  f"{len(zarr_body_names)} -> {len(layout.body_names)}")
        else:
            # No body_names => assume identity ordering (legacy zarrs).
            self.body_idx = None

        print(f"[VAE Dataset] zarr={zarr_path}  split={clip_split}  clips={self.num_clips} "
              f"windows={self.num_windows}  window={self.window}  per_frame_dim={layout.per_frame_dim}")

        # Compute per-feature mean/std on a subsample (held in float32 on CPU).
        if self.num_windows == 0:
            raise RuntimeError("dataset has 0 windows — check window length vs clip lengths.")
        sample_n = min(norm_sample, self.num_windows)
        sample_idx = np.random.default_rng(split_seed).choice(self.num_windows, sample_n, replace=False)
        accum = np.zeros((sample_n, self.window, layout.per_frame_dim), dtype=np.float32)
        for i, gi in enumerate(sample_idx):
            accum[i] = self._fetch_window(int(gi))
        self.mean = accum.mean(axis=(0, 1))   # (per_frame_dim,)
        self.std = accum.std(axis=(0, 1)) + 1e-3
        print(f"[VAE Dataset]   mean range [{float(self.mean.min()):+.3f}, {float(self.mean.max()):+.3f}]"
              f"   std range [{float(self.std.min()):.3f}, {float(self.std.max()):.3f}]")

    # ── internal: convert a global window-index to (clip_i, start_frame) ────

    def _resolve(self, gi: int) -> tuple[int, int]:
        ci = int(np.searchsorted(self.cum_starts, gi, side="right"))
        prev = int(self.cum_starts[ci - 1]) if ci > 0 else 0
        local = gi - prev
        start = int(self.clip_start[ci]) + local
        return ci, start

    def _fetch_window(self, gi: int) -> np.ndarray:
        """Return (W, per_frame_dim) float32 array (unnormalized)."""
        _, start = self._resolve(gi)
        end = start + self.window
        # Read each block from the zarr only over [start:end], minimizing IO.
        body_pos = self.store["body_pos_w"][start:end]      # (W, num_zarr_bodies, 3)
        body_quat = self.store["body_quat_w"][start:end]    # (W, num_zarr_bodies, 4)
        body_lin = self.store["body_lin_vel_w"][start:end]
        body_ang = self.store["body_ang_vel_w"][start:end]
        joint_pos = self.store["joint_pos"][start:end]      # (W, num_joints)
        joint_vel = self.store["joint_vel"][start:end]
        if self.body_idx is not None:
            body_pos = body_pos[:, self.body_idx]
            body_quat = body_quat[:, self.body_idx]
            body_lin = body_lin[:, self.body_idx]
            body_ang = body_ang[:, self.body_idx]
        # Pelvis-frame normalization: subtract pelvis pos so body_pos is local.
        # (Pelvis is layout.body_names[0] by convention; safer to just zero out
        # global root translation.)
        pelvis_pos = body_pos[:, 0:1, :]
        body_pos = body_pos - pelvis_pos
        # Pack per-frame:
        W = body_pos.shape[0]
        per_body = np.concatenate(
            [body_pos.reshape(W, -1), body_quat.reshape(W, -1),
             body_lin.reshape(W, -1), body_ang.reshape(W, -1)], axis=-1
        )
        out = np.concatenate([per_body, joint_pos, joint_vel], axis=-1)
        return out.astype(np.float32)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw = self._fetch_window(int(idx))
        norm = (raw - self.mean) / self.std
        return torch.from_numpy(norm)  # (W, per_frame_dim) float32


# ────── VAE model ───────────────────────────────────────────────────────────


class MotionVAE(nn.Module):
    """Feedforward β-VAE on flattened motion windows.

    Input: (B, W, F) → flattens to (B, W*F).
    Output: reconstruction (B, W, F), plus (μ, log σ) of shape (B, latent_dim).
    """

    def __init__(self, window: int, per_frame_dim: int, latent_dim: int = 16,
                 hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.window = window
        self.per_frame_dim = per_frame_dim
        self.flat_dim = window * per_frame_dim
        self.latent_dim = latent_dim

        # Encoder MLP
        layers: list[nn.Module] = []
        in_dim = self.flat_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

        # Decoder MLP (reverse hidden_dims)
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, self.flat_dim))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.flatten(1))  # (B, hidden)
        return self.mu_head(h), self.logvar_head(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, self.window, self.per_frame_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> dict:
    """Return per-channel MSE recon + β·KL averaged over batch."""
    recon = F.mse_loss(x_recon, x, reduction="mean")
    # KL[N(mu, σ^2) || N(0, 1)] = 0.5 * Σ(σ^2 + μ^2 - 1 - log σ^2)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
    total = recon + beta * kl
    return {"loss": total, "recon": recon.detach(), "kl": kl.detach()}


# ────── symmetric-augmentation helper for VAE windows ──────────────────────


def build_window_reflect_op(layout: FeatureLayout) -> torch.Tensor:
    """Construct a (per_frame_dim × per_frame_dim) reflect op that matches
    the per-frame feature layout above. Applied as `x[..., :] @ op` per
    frame; broadcasts across the window dim.

    Reuses the primitives from `tasks.generalist.mdp.symmetric_augment`.
    """
    # Load symmetric_augment.py by file path — importing it via
    # `whole_body_tracking.tasks...` triggers the package __init__ → Isaac Lab
    # → `omni.log` (only available under Isaac Sim's AppLauncher). The
    # primitives below are pure numpy/torch and don't need the package init.
    import importlib.util as _importlib_util
    import os as _os
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _SYM_PATH = _os.path.abspath(_os.path.join(
        _HERE, "..", "tasks", "generalist", "mdp", "symmetric_augment.py",
    ))
    _sym_spec = _importlib_util.spec_from_file_location("symmetric_augment_mod", _SYM_PATH)
    _sym = _importlib_util.module_from_spec(_sym_spec)
    _sym_spec.loader.exec_module(_sym)
    build_Q = _sym.build_Q
    build_per_body_Rd = _sym.build_per_body_Rd
    build_per_body_Rd_pseudo = _sym.build_per_body_Rd_pseudo
    build_per_body = _sym.build_per_body

    # body_pos: per-body Rd (3 each)
    per_body_pos = build_per_body_Rd(layout.body_names)
    # body_quat: per-body Rot4 — TML's choice for quaternions is to encode rotation
    # via the underlying rotation matrix. For raw quats, the y-mirror maps
    # (w, x, y, z) → (w, -x, y, -z). We build a per-body 4x4 block that does this,
    # then stack via build_per_body.
    quat_op_one = torch.diag(torch.tensor([1.0, -1.0, 1.0, -1.0]))  # (w, x, y, z) y-mirror
    per_body_quat = build_per_body(layout.body_names, quat_op_one)
    # body_lin_vel: per-body Rd
    per_body_lin = build_per_body_Rd(layout.body_names)
    # body_ang_vel: per-body Rd_pseudo
    per_body_ang = build_per_body_Rd_pseudo(layout.body_names)
    # joint_pos, joint_vel: Q
    Q = build_Q(layout.joint_names)

    return torch.block_diag(
        per_body_pos, per_body_quat, per_body_lin, per_body_ang, Q, Q,
    )


def sym_augment_batch(x: torch.Tensor, op: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """Apply the per-frame reflection op to a fraction of windows in a batch.

    Args:
        x: (B, W, F) feature batch (normalized).
        op: (F, F) reflection op on per-frame features.
        prob: per-window probability of being reflected.
    """
    B = x.shape[0]
    mask = (torch.rand(B, device=x.device) < prob).view(B, 1, 1).expand_as(x)
    # Apply op via einsum: x_refl[b,w,f'] = sum_f x[b,w,f] * op[f,f']
    x_refl = torch.einsum("bwf,fg->bwg", x, op)
    return torch.where(mask, x_refl, x)
