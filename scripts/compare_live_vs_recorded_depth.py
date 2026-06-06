"""
Compare LIVE depth embeddings (produced by sim2sim --debug_vision) against the
RECORDED depth embeddings in the training zarr.

Why: teacher-forcing on the recorded depth_embed reproduces training actions at
95% correlation, but closed-loop sim2sim (which re-encodes depth LIVE every step)
collapses. The depth preprocessing code is byte-identical between collection and
eval, but the actual embedding VALUES were never compared. The live depth embed
is the single substantive input that differs between the passing teacher-forcing
test and the failing closed-loop rollout. Flat-ground walking policies tolerated
any latent vision gap (walking barely needs depth); stair climbing does not.

If the live embeddings sit OUTSIDE the recorded distribution (very different
mean/std/range, or large nearest-neighbor distance / low cosine similarity to the
recorded set), the policy is being fed off-distribution vision -> concrete bug.
If they sit INSIDE it, vision is fine and the fall is closed-loop covariate shift.

Usage:
  python scripts/compare_live_vs_recorded_depth.py \
      --live videos/vision_stair_climbing/vision_debug/live_embeds.npz \
      --zarr /move/u/chrzhang/rmr_tracking/STAIRCASE_DATA_COLLECTED/merged_dataset.zarr
"""
import argparse
import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="live_embeds.npz from sim2sim --debug_vision")
    ap.add_argument("--zarr", default="/move/u/chrzhang/rmr_tracking/STAIRCASE_DATA_COLLECTED/merged_dataset.zarr")
    ap.add_argument("--n_ref", type=int, default=20000, help="Recorded embeddings to sample for the reference distribution.")
    args = ap.parse_args()

    live = np.load(args.live)["depth_embed"].astype(np.float32)   # (T, 1024)
    z = zarr.open(args.zarr, "r")
    rec_all = z["data/depth_embed"]
    n = rec_all.shape[0]
    idx = np.sort(np.random.RandomState(0).choice(n, size=min(args.n_ref, n), replace=False))
    rec = rec_all[idx].astype(np.float32)                          # (N, 1024)

    print(f"[INFO] live: {live.shape}  recorded(sampled): {rec.shape}")

    # 1. Global scalar stats
    def stats(name, x):
        print(f"  {name:9s} mean={x.mean():+.4f} std={x.std():.4f} "
              f"min={x.min():+.3f} max={x.max():+.3f} L2/row(mean)={np.linalg.norm(x,axis=1).mean():.2f}")
    print("\n=== global stats ===")
    stats("live", live)
    stats("recorded", rec)

    # 2. Per-dimension mean/std agreement (are the two distributions even aligned?)
    rec_mu, rec_sd = rec.mean(0), rec.std(0) + 1e-6
    live_mu = live.mean(0)
    z_off = (live_mu - rec_mu) / rec_sd      # how many recorded-stds the live mean is off, per dim
    print("\n=== per-dim mean offset (live mean vs recorded mean, in recorded stds) ===")
    print(f"  |offset| mean={np.abs(z_off).mean():.2f}  median={np.median(np.abs(z_off)):.2f}  "
          f"max={np.abs(z_off).max():.2f}  (#dims>3sigma = {(np.abs(z_off)>3).sum()}/1024)")

    # 3. Nearest-recorded distance + cosine for each live frame (is live IN the manifold?)
    #    Compare against recorded mean too as a sanity baseline.
    def nn_dist(a, B):
        # min L2 distance from each row of a to any row of B (chunked)
        d = np.empty(a.shape[0])
        for i in range(a.shape[0]):
            d[i] = np.sqrt(((B - a[i]) ** 2).sum(1)).min()
        return d
    def cos(a, b):
        return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    live_nn = nn_dist(live, rec)
    # baseline: typical nearest-neighbor distance WITHIN the recorded set
    rec_sample = rec[np.random.RandomState(1).choice(rec.shape[0], size=min(300, rec.shape[0]), replace=False)]
    rec_nn = nn_dist(rec_sample, rec)  # includes self (=0); use as loose lower ref
    rec_nn = rec_nn[rec_nn > 1e-6]
    print("\n=== nearest-recorded-embedding distance ===")
    print(f"  live frames : nn-dist mean={live_nn.mean():.2f} median={np.median(live_nn):.2f} max={live_nn.max():.2f}")
    print(f"  recorded-vs-recorded baseline: nn-dist mean={rec_nn.mean():.2f} median={np.median(rec_nn):.2f}")
    rec_centroid = rec.mean(0)
    print(f"  cosine(live_mean, recorded_mean) = {cos(live.mean(0), rec_centroid):+.3f}")

    print("\nInterpretation:")
    print("  live nn-dist >> recorded baseline, or |per-dim offset| large / low cosine")
    print("    -> live depth embeddings are OFF-distribution: a real eval-time vision bug.")
    print("  live nn-dist ~ recorded baseline and stats aligned")
    print("    -> vision is fine; the fall is closed-loop covariate shift (thin data coverage).")


if __name__ == "__main__":
    main()
