"""Drop mirrored clips (clip_names ending in `_M`) from a Zarr motion store.

Every `_M` clip in the source datasets has a non-mirrored sibling, so dropping
them halves clip count and per-frame storage without leaving orphans. Mirroring
will instead be reintroduced at training time via on-the-fly observation flip
(see discussion in repo notes).

Examples:
    python scripts/drop_mirror_clips.py \\
        --src /move/data/bones/g1/zarr/locomotion_33hz.zarr \\
        --dst /move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr

    python scripts/drop_mirror_clips.py \\
        --src /move/data/bones/g1/zarr/motions_33hz.zarr \\
        --dst /move/u/justingu/rmr_tracking/motions/motions_33hz.zarr \\
        --num_workers 16

    # Just report what would be dropped, don't write anything:
    python scripts/drop_mirror_clips.py --src ... --dst ... --dry_run
"""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr


# Per-frame arrays. Everything else is per-clip metadata or constants.
ARRAY_NAMES = ("joint_pos", "joint_vel", "body_pos_w",
               "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


_WORKER_STATE: dict = {}


def _init_worker():
    _WORKER_STATE["src"] = zarr.open(_WORKER_STATE["src_path"], mode="r")
    _WORKER_STATE["dst"] = zarr.open(_WORKER_STATE["dst_path"], mode="r+")


def _process_chunk(c: int) -> int:
    src = _WORKER_STATE["src"]
    dst = _WORKER_STATE["dst"]
    time_chunk = _WORKER_STATE["time_chunk"]
    total_frames = _WORKER_STATE["total_frames"]
    shapes_tail = _WORKER_STATE["shapes_tail"]
    jobs = _WORKER_STATE["chunk_jobs"][c]

    c_start = c * time_chunk
    c_end = min(c_start + time_chunk, total_frames)
    L = c_end - c_start
    for name, tail in shapes_tail.items():
        buf = np.empty((L,) + tail, dtype=np.float32)
        for bws, bwe, srs, sre in jobs:
            buf[bws:bwe] = src[name][srs:sre]
        dst[name][c_start:c_end] = buf
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source zarr path.")
    parser.add_argument("--dst", required=True, help="Destination zarr path.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Worker processes for the chunk-wise copy.")
    parser.add_argument("--time_chunk", type=int, default=8192,
                        help="Frames per chunk along the time axis in the output.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report keep/drop counts and exit without writing.")
    args = parser.parse_args()

    src = zarr.open(args.src, mode="r")
    clip_names = np.asarray(src["clip_names"][:])
    clip_start = src["clip_start_idx"][:]
    clip_end = src["clip_end_idx"][:]
    content_props = src["content_props"][:]
    content_props_desc = src["content_props_desc"][:]
    fps = src["fps"][:]
    body_names = src["body_names"][:]

    is_mirror = np.array([n.endswith("_M") for n in clip_names])
    keep_idx = np.where(~is_mirror)[0]
    drop_n = int(is_mirror.sum())
    keep_n = int((~is_mirror).sum())

    src_frames = int(clip_end[-1]) if len(clip_end) else 0
    kept_frames = int(np.sum(clip_end[keep_idx] - clip_start[keep_idx]))
    print(f"[src] {args.src}")
    print(f"  clips: {len(clip_names)}  (mirror: {drop_n}, keep: {keep_n})")
    print(f"  frames: {src_frames}  (keep: {kept_frames}, drop: {src_frames - kept_frames})")
    print(f"  estimated output frame fraction: {kept_frames / max(1, src_frames):.3f}")

    if drop_n == 0:
        print("[warn] No mirrored clips found — nothing to drop.")
    if args.dry_run:
        print("[dry_run] No output written.")
        return

    # New per-clip indices: dense pack the kept clips back-to-back.
    lengths = (clip_end[keep_idx] - clip_start[keep_idx]).astype(np.int64)
    total_frames = int(lengths.sum())
    new_start = np.empty(len(keep_idx), dtype=np.int64)
    new_end = np.empty(len(keep_idx), dtype=np.int64)
    cur = 0
    for k, L in enumerate(lengths.tolist()):
        new_start[k] = cur
        new_end[k] = cur + L
        cur += L
    assert cur == total_frames

    n_bodies = src["body_pos_w"].shape[1]
    n_joints = src["joint_pos"].shape[1]

    print(f"\n[out] {len(keep_idx)} clips, {total_frames} frames "
          f"({total_frames / int(fps[0]):.1f} s @ {int(fps[0])} Hz), "
          f"{n_bodies} bodies, {n_joints} joints")

    dst_path = Path(args.dst)
    if dst_path.exists():
        raise FileExistsError(
            f"Destination already exists: {dst_path}. "
            f"Refusing to overwrite — remove it manually if you intend to replace it."
        )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(str(dst_path), mode="w")

    time_chunk = min(args.time_chunk, total_frames) if total_frames else 1
    dst.zeros("joint_pos", shape=(total_frames, n_joints), dtype=np.float32,
              chunks=(time_chunk, n_joints))
    dst.zeros("joint_vel", shape=(total_frames, n_joints), dtype=np.float32,
              chunks=(time_chunk, n_joints))
    dst.zeros("body_pos_w", shape=(total_frames, n_bodies, 3), dtype=np.float32,
              chunks=(time_chunk, n_bodies, 3))
    dst.zeros("body_quat_w", shape=(total_frames, n_bodies, 4), dtype=np.float32,
              chunks=(time_chunk, n_bodies, 4))
    dst.zeros("body_lin_vel_w", shape=(total_frames, n_bodies, 3), dtype=np.float32,
              chunks=(time_chunk, n_bodies, 3))
    dst.zeros("body_ang_vel_w", shape=(total_frames, n_bodies, 3), dtype=np.float32,
              chunks=(time_chunk, n_bodies, 3))

    shapes_tail = {
        "joint_pos": (n_joints,),
        "joint_vel": (n_joints,),
        "body_pos_w": (n_bodies, 3),
        "body_quat_w": (n_bodies, 4),
        "body_lin_vel_w": (n_bodies, 3),
        "body_ang_vel_w": (n_bodies, 3),
    }

    # Build per-output-chunk job lists so each output chunk is written exactly
    # once by exactly one worker (mirrors make_locomotion_subset.py).
    n_out_chunks = (total_frames + time_chunk - 1) // time_chunk if total_frames else 0
    chunk_jobs: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_out_chunks)]
    for k, clip_i in enumerate(keep_idx.tolist()):
        s = int(clip_start[clip_i])
        ns, ne = int(new_start[k]), int(new_end[k])
        first_c = ns // time_chunk
        last_c = (ne - 1) // time_chunk
        for c in range(first_c, last_c + 1):
            c_start = c * time_chunk
            c_end = min(c_start + time_chunk, total_frames)
            os_ = max(ns, c_start)
            oe = min(ne, c_end)
            chunk_jobs[c].append((os_ - c_start, oe - c_start,
                                  s + (os_ - ns), s + (oe - ns)))

    _WORKER_STATE.update({
        "src_path": args.src,
        "dst_path": str(dst_path),
        "time_chunk": time_chunk,
        "total_frames": total_frames,
        "shapes_tail": shapes_tail,
        "chunk_jobs": chunk_jobs,
    })

    print(f"\n[copy] {n_out_chunks} output chunks via {args.num_workers} workers")
    if n_out_chunks > 0:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx,
                                 initializer=_init_worker) as ex:
            futures = [ex.submit(_process_chunk, c) for c in range(n_out_chunks)]
            completed = 0
            for fut in as_completed(futures):
                fut.result()
                completed += 1
                if completed % 50 == 0 or completed == n_out_chunks:
                    print(f"  wrote {completed}/{n_out_chunks} chunks")

    # Per-clip metadata + invariant arrays.
    sel_names = np.array([clip_names[i] for i in keep_idx], dtype=object)
    sel_props = np.array([content_props[i] for i in keep_idx], dtype=np.int32)
    sel_props_desc = np.array([content_props_desc[i] for i in keep_idx], dtype=object)

    dst.array("clip_names", sel_names, dtype=object, object_codec=zarr.codecs.VLenUTF8())
    dst.array("clip_start_idx", new_start.astype(np.int64))
    dst.array("clip_end_idx", new_end.astype(np.int64))
    dst.array("content_props", sel_props)
    dst.array("content_props_desc", sel_props_desc,
              dtype=object, object_codec=zarr.codecs.VLenUTF8())
    dst.array("fps", fps.astype(np.int32))
    dst.array("body_names", np.asarray(body_names, dtype=object),
              dtype=object, object_codec=zarr.codecs.VLenUTF8())

    print(f"\n[done] wrote {dst_path}")


if __name__ == "__main__":
    main()
