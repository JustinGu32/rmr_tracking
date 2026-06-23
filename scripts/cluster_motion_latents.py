#!/usr/bin/env python
"""K-means cluster per-clip motion latents (output of dump_motion_latents.py)
into K data-driven categories.

Output: a JSON file with the centroids and per-clip cluster assignments,
consumed by the `latent_kmeans` categorizer registered in
`tasks/generalist/mdp/categorizers.py`.

Usage:
    python scripts/cluster_motion_latents.py \
        --latents_npz logs/motion_vae/v1/clip_latents.npz \
        --k 8 \
        --output_json logs/motion_vae/v1/clip_clusters_k8.json

Output JSON shape:
    {
      "k": 8,
      "feature_dim": 32,           # = 2 * latent_dim (mean + std concat)
      "centroids": [[...], ...],   # (k, feature_dim)
      "cluster_id_by_name": {
        "<clip_name>": <int 0..k-1>,
        ...
      },
      "cluster_sizes": [int, ...]  # length k
    }

This file is what `--latent_centroids_path` in train_bones.py points at.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ centroid initialization."""
    n, d = X.shape
    centroids = np.empty((k, d), dtype=X.dtype)
    # First centroid: random point.
    centroids[0] = X[rng.integers(n)]
    # Remaining: pick proportionally to squared distance from nearest centroid.
    closest = np.full(n, np.inf)
    for i in range(1, k):
        new_dist = ((X - centroids[i - 1]) ** 2).sum(axis=-1)
        closest = np.minimum(closest, new_dist)
        probs = closest / closest.sum()
        centroids[i] = X[rng.choice(n, p=probs)]
    return centroids


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, n_restarts: int = 5,
           seed: int = 0, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """Vanilla Lloyd's algorithm with k-means++ init and best-of-N restarts.

    Returns:
        centroids: (k, d)
        labels:    (n,) int64
        inertia:   float (sum of squared distances to assigned centroid)
    """
    best_centroids = None
    best_labels = None
    best_inertia = np.inf
    rng = np.random.default_rng(seed)
    n, d = X.shape

    for r in range(n_restarts):
        centroids = kmeans_plus_plus_init(X, k, rng)
        prev_labels = None
        for it in range(n_iter):
            # Assign: argmin over centroids.
            dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)  # (n, k)
            labels = dists.argmin(axis=-1)
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels
            # Update centroids: mean of assigned points (re-seed empty clusters).
            for ki in range(k):
                mask = labels == ki
                if mask.any():
                    centroids[ki] = X[mask].mean(axis=0)
                else:
                    centroids[ki] = X[rng.integers(n)]
        inertia = float(((X - centroids[labels]) ** 2).sum())
        if verbose:
            sizes = [int((labels == ki).sum()) for ki in range(k)]
            print(f"  restart {r}: inertia={inertia:.1f}  iters={it+1}  sizes={sizes}", flush=True)
        if inertia < best_inertia:
            best_centroids = centroids
            best_labels = labels
            best_inertia = inertia

    return best_centroids, best_labels, best_inertia


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_npz", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n_restarts", type=int, default=5)
    p.add_argument("--n_iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature_mode", choices=["mean", "mean_std"], default="mean_std",
                   help="Per-clip vector: 'mean' = z_mean (latent_dim); "
                        "'mean_std' = concat(z_mean, z_std) (2*latent_dim).")
    args = p.parse_args()

    print(f"[cluster] loading {args.latents_npz}", flush=True)
    data = np.load(args.latents_npz, allow_pickle=True)
    clip_names = list(data["clip_names"])
    z_mean = data["z_mean"].astype(np.float32)
    z_std = data["z_std"].astype(np.float32)
    print(f"[cluster] {len(clip_names)} clips  z_mean {z_mean.shape}  z_std {z_std.shape}", flush=True)

    # Per-clip feature vector.
    if args.feature_mode == "mean":
        feats = z_mean
    else:
        feats = np.concatenate([z_mean, z_std], axis=-1)
    # Drop clips with all-zero latents (too short to encode in Phase 4).
    valid_mask = (np.abs(z_mean).sum(axis=-1) > 1e-6) | (np.abs(z_std).sum(axis=-1) > 1e-6)
    valid_feats = feats[valid_mask]
    valid_names = [clip_names[i] for i in range(len(clip_names)) if valid_mask[i]]
    print(f"[cluster] {len(valid_names)} clips with non-zero latents (dropped "
          f"{int((~valid_mask).sum())})", flush=True)
    if len(valid_feats) < args.k:
        raise RuntimeError(f"k={args.k} > number of valid clips ({len(valid_feats)})")

    print(f"[cluster] running k-means: k={args.k}, restarts={args.n_restarts}", flush=True)
    centroids, labels, inertia = kmeans(
        valid_feats, k=args.k, n_iter=args.n_iter, n_restarts=args.n_restarts, seed=args.seed
    )
    sizes = [int((labels == ki).sum()) for ki in range(args.k)]
    print(f"[cluster] FINAL inertia={inertia:.1f}  sizes={sizes}", flush=True)

    # Build map clip_name → cluster_id. Clips that were dropped (zero latents)
    # get an explicit -1 sentinel so the runtime categorizer can flag them.
    cluster_id_by_name: dict[str, int] = {n: -1 for n in clip_names}
    for n, c in zip(valid_names, labels.tolist()):
        cluster_id_by_name[n] = int(c)

    out = {
        "k": int(args.k),
        "feature_dim": int(valid_feats.shape[-1]),
        "feature_mode": args.feature_mode,
        "centroids": centroids.tolist(),
        "cluster_id_by_name": cluster_id_by_name,
        "cluster_sizes": sizes,
        "inertia": inertia,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f)
    print(f"[cluster] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
