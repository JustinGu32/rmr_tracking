#!/usr/bin/env python3
"""
Convert between BONES motion zarr (ZarrMotionLoader / rmr_tracking) and TML
ReplayBuffer zarr (diffusion_policy OfflineDataset).

BONES layout (arrays at store root):
  fps, clip_start_idx, clip_end_idx,
  joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  optional: content_props_desc

TML layout:
  meta/episode_ends  (int64, cumulative exclusive ends; length = n_episodes)
  data/{joint_pos, joint_vel, body_pos, body_rot, body_lin_vel, body_ang_vel,
        root_pos, root_rot, act}

Assumptions (verify on your data):
  - body order matches between corpora (Isaac G1 link order).
  - Quaternions are the same layout (typically w,x,y,z in Isaac Lab tensors).
  - root_* are taken from body index ``root_body_index`` (default 0 = pelvis).
  - ``act`` is often absent from BONES; use --act zeros (default) or supply
    a matching .npy / .npz (see --act-npy).

Examples:
  python convert_motion_zarr_format.py bones2tml -i /path/to/bones.zarr -o /path/to/tml.zarr
  python convert_motion_zarr_format.py tml2bones -i /path/to/tml.zarr -o /path/to/bones.zarr --fps 50
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


REQUIRED_BONES_KEYS = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]
CLIP_KEYS = ("clip_start_idx", "clip_end_idx")


def _open_zarr(path: str):
    import zarr

    return zarr.open(os.path.expanduser(path), mode="r")


def _require_keys(group, keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in group]
    if missing:
        raise SystemExit(f"{label}: missing keys {missing}. Found: {list(group.keys())}")


def _episode_ends_from_clips(
    clip_start: np.ndarray,
    clip_end: np.ndarray,
    T: int,
    *,
    repack: bool,
    keys: list[str],
    src,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    """
    Returns (episode_ends, repacked_arrays_or_none).
    If repacked_arrays is not None, all listed keys are fully repacked (T may change).
    """
    clip_start = np.asarray(clip_start, dtype=np.int64).ravel()
    clip_end = np.asarray(clip_end, dtype=np.int64).ravel()
    if clip_start.shape != clip_end.shape or clip_start.size == 0:
        raise SystemExit("clip_start_idx and clip_end_idx must be same nonzero length.")

    for i, (s, e) in enumerate(zip(clip_start, clip_end)):
        if s < 0 or e < s or e > T:
            raise SystemExit(f"Invalid clip {i}: start={s}, end={e}, T={T}")

    contiguous = clip_start[0] == 0 and bool(np.all(clip_start[1:] == clip_end[:-1])) and int(clip_end[-1]) == T

    if contiguous and not repack:
        return clip_end.astype(np.int64), None

    if contiguous and repack:
        return clip_end.astype(np.int64), None

    print(
        "[convert] Clips are not a perfect partition of [0, T) without gaps; "
        "repacking concatenated clip segments into a dense timeline."
    )
    repacked: dict[str, np.ndarray] = {}
    for k in keys:
        arr = np.asarray(src[k][:T])
        if arr.shape[0] != T:
            raise SystemExit(f"Key {k} has T={arr.shape[0]}, expected {T}")
        parts = [arr[s:e] for s, e in zip(clip_start, clip_end)]
        repacked[k] = np.concatenate(parts, axis=0)
    lens = [int(e - s) for s, e in zip(clip_start, clip_end)]
    episode_ends = np.cumsum(np.array(lens, dtype=np.int64))
    return episode_ends, repacked


def _load_act(T: int, joint_dim: int, args) -> np.ndarray:
    if args.act == "zeros":
        dim = args.act_dim if args.act_dim is not None else joint_dim
        print(f"[convert] Filling `act` with zeros, shape ({T}, {dim}).")
        return np.zeros((T, dim), dtype=np.float32)
    if args.act == "npy":
        if not args.act_npy:
            raise SystemExit("--act npy requires --act-npy PATH")
        act = np.load(args.act_npy)
        if act.shape[0] != T:
            raise SystemExit(f"act file has T={act.shape[0]}, expected {T}")
        return np.asarray(act, dtype=np.float32)
    raise SystemExit(f"Unknown --act {args.act}")


def bones_to_tml(src_path: str, dst_path: str, args) -> None:
    import zarr

    src = _open_zarr(src_path)
    _require_keys(src, REQUIRED_BONES_KEYS + list(CLIP_KEYS), "BONES zarr")
    if "fps" not in src:
        print("[convert] Warning: no `fps` in BONES zarr (TML ReplayBuffer does not require it).")

    T = int(src["joint_pos"].shape[0])
    joint_dim = int(src["joint_pos"].shape[1])

    clip_start = np.asarray(src["clip_start_idx"][:], dtype=np.int64).ravel()
    clip_end = np.asarray(src["clip_end_idx"][:], dtype=np.int64).ravel()
    T_used = int(clip_end[-1]) if clip_end.size else T
    if T_used < T:
        if args.truncate_to_clips:
            print(f"[convert] Truncating timeline from T={T} to clip_end[-1]={T_used}.")
            T = T_used
        else:
            raise SystemExit(
                f"joint_pos has T={T} but clip_end[-1]={T_used}. "
                "Use --truncate-to-clips or fix the dataset."
            )
    elif T_used > T:
        raise SystemExit(f"clip_end[-1]={T_used} exceeds joint_pos T={T}.")
    keys_to_maybe_repack = list(REQUIRED_BONES_KEYS)
    episode_ends, repacked = _episode_ends_from_clips(
        clip_start, clip_end, T, repack=args.repack, keys=keys_to_maybe_repack, src=src
    )

    def get_full(key: str) -> np.ndarray:
        if repacked is not None:
            return repacked[key]
        arr = np.asarray(src[key][:T], dtype=np.float32)
        return arr

    joint_pos = get_full("joint_pos")
    joint_vel = get_full("joint_vel")
    body_pos_w = get_full("body_pos_w")
    body_quat_w = get_full("body_quat_w")
    body_lin_vel_w = get_full("body_lin_vel_w")
    body_ang_vel_w = get_full("body_ang_vel_w")

    T2 = joint_pos.shape[0]
    if int(episode_ends[-1]) != T2:
        raise SystemExit(f"Internal error: episode_ends[-1]={episode_ends[-1]} != T2={T2}")

    rb = args.root_body_index
    if rb < 0 or rb >= body_pos_w.shape[1]:
        raise SystemExit(f"root_body_index={rb} out of range for n_bodies={body_pos_w.shape[1]}")

    root_pos = body_pos_w[:, rb, :].copy()
    root_rot = body_quat_w[:, rb, :].copy()

    act = _load_act(T2, joint_dim, args)

    if os.path.exists(dst_path):
        raise SystemExit(f"Output path exists: {dst_path} (refuse to overwrite).")

    store = zarr.DirectoryStore(os.path.expanduser(dst_path))
    root = zarr.group(store=store, overwrite=True)
    meta = root.create_group("meta")
    data = root.create_group("data")

    meta.array("episode_ends", data=episode_ends, dtype=np.int64, compressor=None)
    # Optional: helps round-trip / documentation (ReplayBuffer ignores unknown meta keys)
    if "fps" in src:
        fps_arr = np.asarray(src["fps"][:]).reshape(-1)
        meta.array("fps", data=fps_arr.astype(np.int32), dtype=np.int32, compressor=None)

    chunk_t = min(4096, T2) if T2 > 0 else 1
    chunks_2d = (chunk_t, joint_pos.shape[1])
    chunks_3d = (chunk_t, body_pos_w.shape[1], 3)
    chunks_3dq = (chunk_t, body_quat_w.shape[1], 4)
    chunks_act = (chunk_t, act.shape[1])

    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.NOSHUFFLE)

    data.array("joint_pos", data=joint_pos, chunks=chunks_2d, compressor=compressor)
    data.array("joint_vel", data=joint_vel, chunks=chunks_2d, compressor=compressor)
    data.array("body_pos", data=body_pos_w, chunks=chunks_3d, compressor=compressor)
    data.array("body_rot", data=body_quat_w, chunks=chunks_3dq, compressor=compressor)
    data.array("body_lin_vel", data=body_lin_vel_w, chunks=chunks_3d, compressor=compressor)
    data.array("body_ang_vel", data=body_ang_vel_w, chunks=chunks_3d, compressor=compressor)
    data.array("root_pos", data=root_pos, chunks=(chunk_t, 3), compressor=compressor)
    data.array("root_rot", data=root_rot, chunks=(chunk_t, 4), compressor=compressor)
    data.array("act", data=act, chunks=chunks_act, compressor=compressor)

    print(f"[convert] Wrote TML ReplayBuffer zarr: {dst_path}")
    print(f"  T={T2}, n_episodes={len(episode_ends)}, root_body_index={rb}")


def tml_to_bones(src_path: str, dst_path: str, args) -> None:
    import zarr

    src = _open_zarr(src_path)
    if "meta" not in src or "data" not in src:
        raise SystemExit("TML zarr must have `meta/` and `data/` groups.")
    meta, data = src["meta"], src["data"]
    _require_keys(meta, ["episode_ends"], "TML meta")
    needed = ["joint_pos", "joint_vel", "body_pos", "body_rot", "body_lin_vel", "body_ang_vel"]
    _require_keys(data, needed, "TML data")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64).ravel()
    if episode_ends.size == 0:
        raise SystemExit("Empty episode_ends.")
    T = int(episode_ends[-1])
    starts = np.concatenate([[0], episode_ends[:-1]])

    if os.path.exists(dst_path):
        raise SystemExit(f"Output path exists: {dst_path} (refuse to overwrite).")

    store = zarr.DirectoryStore(os.path.expanduser(dst_path))
    root = zarr.group(store=store, overwrite=True)

    compressor = zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.NOSHUFFLE)
    chunk_t = min(4096, T) if T > 0 else 1

    def copy_tml_to_bones(tml_name: str, bones_name: str, chunks):
        arr = np.asarray(data[tml_name][:], dtype=np.float32)
        if arr.shape[0] != T:
            raise SystemExit(f"{tml_name} T={arr.shape[0]} != meta episode_ends[-1]={T}")
        root.array(bones_name, data=arr, chunks=chunks, compressor=compressor)

    jp = np.asarray(data["joint_pos"][:])
    jb = jp.shape[1]
    nb = data["body_pos"].shape[1]

    copy_tml_to_bones("joint_pos", "joint_pos", (chunk_t, jb))
    copy_tml_to_bones("joint_vel", "joint_vel", (chunk_t, jb))
    copy_tml_to_bones("body_pos", "body_pos_w", (chunk_t, nb, 3))
    copy_tml_to_bones("body_rot", "body_quat_w", (chunk_t, nb, 4))
    copy_tml_to_bones("body_lin_vel", "body_lin_vel_w", (chunk_t, nb, 3))
    copy_tml_to_bones("body_ang_vel", "body_ang_vel_w", (chunk_t, nb, 3))

    root.array("clip_start_idx", data=starts.astype(np.int64), dtype=np.int64, compressor=None)
    root.array("clip_end_idx", data=episode_ends.astype(np.int64), dtype=np.int64, compressor=None)
    fps = int(args.fps)
    root.array("fps", data=np.array([fps], dtype=np.int32), dtype=np.int32, compressor=None)

    print(f"[convert] Wrote BONES zarr: {dst_path}")
    print(f"  T={T}, n_clips={len(episode_ends)}, fps={fps} (from --fps)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bones2tml", help="BONES flat zarr -> TML ReplayBuffer zarr")
    pb.add_argument("-i", "--input", required=True, help="Input BONES zarr path")
    pb.add_argument("-o", "--output", required=True, help="Output directory (new TML zarr)")
    pb.add_argument("--root-body-index", type=int, default=0, help="Body index for root_pos/root_rot (default 0)")
    pb.add_argument(
        "--repack",
        action="store_true",
        help="Concatenate clip segments if clips do not partition [0,T) contiguously",
    )
    pb.add_argument("--act", choices=("zeros", "npy"), default="zeros", help="How to fill required `act` array")
    pb.add_argument("--act-dim", type=int, default=None, help="With --act zeros, action dimension (default: joint dim)")
    pb.add_argument("--act-npy", type=str, default=None, help="Path to .npy for --act npy, shape (T, D_a)")
    pb.add_argument(
        "--truncate-to-clips",
        action="store_true",
        help="If clip_end[-1] < len(joint_pos), slice arrays to the used prefix",
    )

    pt = sub.add_parser("tml2bones", help="TML ReplayBuffer zarr -> BONES flat zarr")
    pt.add_argument("-i", "--input", required=True, help="Input TML zarr path")
    pt.add_argument("-o", "--output", required=True, help="Output directory (new BONES zarr)")
    pt.add_argument("--fps", type=int, default=50, help="fps written to output (TML meta may lack it)")

    args = p.parse_args()
    if args.cmd == "bones2tml":
        bones_to_tml(args.input, args.output, args)
    else:
        tml_to_bones(args.input, args.output, args)


if __name__ == "__main__":
    main()
