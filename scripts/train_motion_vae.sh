#!/bin/bash
#SBATCH --job-name=motion_vae
#SBATCH --partition=move --account=move
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/motion_vae_%j.out
#SBATCH --error=logs/slurm/motion_vae_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export PYTHONUNBUFFERED=1

# ============================================================
# Phase 4: train a VMP-style motion VAE on the cleaner locomotion zarr.
# Symmetric augmentation is ON so the encoder lands L/R variants in the
# same latent neighborhood (cleaner Phase-5 clusters).
# ============================================================

ZARR=/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr
OUT=logs/motion_vae/v1

# python scripts/train_motion_vae.py \
#     --zarr_path ${ZARR} \
#     --output_dir ${OUT} \
#     --window 32 --latent_dim 64 \
#     --hidden_dims 512,256 \
#     --batch_size 256 --epochs 10 --lr 3e-4 \
#     --kl_target 1e-3 --kl_warmup_epochs 5 \
#     --num_workers 4 \
#     --symmetric_augment

# ============================================================
# After training finishes, encode every clip:
# ============================================================
# python scripts/dump_motion_latents.py \
#     --zarr_path ${ZARR} \
#     --vae_ckpt ${OUT}/motion_vae.pt \
#     --output_path ${OUT}/clip_latents.npz \
#     --batch_size 256 \
#     --per_clip_window_stride 8

# ============================================================
# Then cluster into K categories:
# ============================================================
for K in 8 10 12; do
python scripts/cluster_motion_latents.py \
    --latents_npz logs/motion_vae/v1/clip_latents.npz \
    --output_json logs/motion_vae/v1/clip_clusters_k${K}.json \
    --k ${K} --n_restarts 5

# Patch the -1 sentinels (clips too short to encode) -> cluster 0 so the
# latent_kmeans categorizer never raises at RL-train time. NOTE: must be a
# heredoc, NOT `python -c "<indented>"` — leading whitespace on the first code
# line throws IndentationError (and the loop would silently skip the patch).
# The JSON path is passed as argv so it still expands ${K} (heredoc body is
# single-quoted, so no shell expansion happens inside it).
python - "logs/motion_vae/v1/clip_clusters_k${K}.json" <<'PY'
import json, sys
P = sys.argv[1]
d = json.load(open(P))
m = d['cluster_id_by_name']
neg = [name for name, c in m.items() if c < 0]
for name in neg:
    m[name] = 0
d['cluster_sizes'][0] += len(neg)
json.dump(d, open(P, 'w'))
print(f"  {P}: patched {len(neg)} sentinels -> cluster 0")
PY
done

echo "[VAE pipeline] DONE."
echo "  - VAE checkpoint:  ${OUT}/motion_vae.pt"
echo "  - per-clip latents: ${OUT}/clip_latents.npz"
echo "  - K=8 clusters:    ${OUT}/clip_clusters_k8.json"
echo ""
echo "Use the clusters JSON in RL training via:"
echo "  python scripts/rsl_rl/train_bones.py ... \\"
echo "      --task=Generalist-Flat-G1-v0 \\"
echo "      --categorizer_mode latent_kmeans \\"
echo "      --latent_centroids_path ${OUT}/clip_clusters_k8.json"
