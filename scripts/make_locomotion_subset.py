"""Build a diversified subset of a locomotion Zarr motion store.

Groups clips by stem (clip_name with trailing `_<digits>` and `__A...` suffix stripped)
and round-robins across stems so the picks span many distinct motions, not just
variants of the same one.

Example:
    python scripts/make_locomotion_subset.py \
        --src /move/data/bones/g1/zarr/locomotion_33hz.zarr \
        --dst /move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr \
        --categories walk,jog,jump --per_category 30000 --seed 42
"""
import argparse
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr


# Per-frame arrays we copy from src to dst (everything else is metadata).
ARRAY_NAMES = ("joint_pos", "joint_vel", "body_pos_w",
               "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


# Worker globals — populated in main() before the pool forks, plus zarr
# handles re-opened per child in _init_worker. Children inherit everything
# else copy-on-write so we don't pay to pickle chunk_jobs to each worker.
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


def stem_of(name: str) -> str:
    base = name.split("__")[0]
    return re.sub(r"_\d+$", "", base)


def select_diverse(clip_names: np.ndarray, keyword: str, exclude: list[str], n: int, rng: np.random.RandomState) -> list[int]:
    kw = keyword.lower()
    by_stem: dict[str, list[int]] = {}
    for i, name in enumerate(clip_names):
        low = name.lower()
        if kw not in low:
            continue
        if any(e in low for e in exclude):
            continue
        by_stem.setdefault(stem_of(name), []).append(i)

    stems = sorted(by_stem.keys())
    rng.shuffle(stems)
    for s in stems:
        rng.shuffle(by_stem[s])

    selected: list[int] = []
    depth = 0
    while len(selected) < n:
        progressed = False
        for s in stems:
            if depth < len(by_stem[s]):
                selected.append(by_stem[s][depth])
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
        depth += 1

    print(f"[{keyword}] {len(by_stem)} distinct stems; selected {len(selected)} clips "
          f"(round-robin depth {depth}).")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--categories", default="walk,jog",
                        help="Comma-separated substrings (case-insensitive) to partition clips.")
    parser.add_argument("--per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Worker processes for the chunk-wise copy. Each output chunk "
                             "is written exactly once by exactly one worker.")
    args = parser.parse_args()

    src = zarr.open(args.src, mode="r")
    clip_names = src["clip_names"][:]
    clip_start = src["clip_start_idx"][:]
    clip_end = src["clip_end_idx"][:]
    content_props = src["content_props"][:]
    content_props_desc = src["content_props_desc"][:]
    fps = src["fps"][:]
    body_names = src["body_names"][:]

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    rng = np.random.RandomState(args.seed)

    all_selected: list[int] = []
    for c in categories:
        # Each category excludes the OTHER categories so clips land in exactly one bucket.
        others = [o for o in categories if o != c]
        idxs = select_diverse(clip_names, c, others, args.per_category, rng)
        all_selected.extend(idxs)

    # Compute total frames and new start/end indices
    lengths = [int(clip_end[i] - clip_start[i]) for i in all_selected]
    total_frames = int(sum(lengths))
    new_start = np.zeros(len(all_selected), dtype=np.int64)
    new_end = np.zeros(len(all_selected), dtype=np.int64)
    cur = 0
    for k, L in enumerate(lengths):
        new_start[k] = cur
        new_end[k] = cur + L
        cur += L
    assert cur == total_frames

    n_bodies = src["body_pos_w"].shape[1]
    n_joints = src["joint_pos"].shape[1]

    print(f"\n[subset] {len(all_selected)} clips, {total_frames} frames "
          f"({total_frames / int(fps[0]):.1f} s @ {int(fps[0])} Hz), "
          f"{n_bodies} bodies, {n_joints} joints")

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(str(dst_path), mode="w")

    # Preallocate flat per-frame arrays with reasonable chunking along time axis
    time_chunk = min(8192, total_frames) if total_frames else 1
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

    # Group every selected clip's contribution by its destination chunk so each
    # output chunk is filled by a single worker writing it exactly once. Without
    # this, the per-clip loop read-modify-writes each chunk ~chunk_size/clip_len
    # times AND can't be safely parallelized (adjacent clips share boundary chunks).
    n_out_chunks = (total_frames + time_chunk - 1) // time_chunk if total_frames else 0
    chunk_jobs: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n_out_chunks)]
    for k, clip_i in enumerate(all_selected):
        s = int(clip_start[clip_i])
        ns, ne = int(new_start[k]), int(new_end[k])
        first_c = ns // time_chunk
        last_c = (ne - 1) // time_chunk
        for c in range(first_c, last_c + 1):
            c_start = c * time_chunk
            c_end = min(c_start + time_chunk, total_frames)
            os = max(ns, c_start)
            oe = min(ne, c_end)
            chunk_jobs[c].append((os - c_start, oe - c_start,
                                  s + (os - ns), s + (oe - ns)))

    # Stash state for fork-based workers (read-only after this; copy-on-write).
    _WORKER_STATE.update({
        "src_path": args.src,
        "dst_path": str(dst_path),
        "time_chunk": time_chunk,
        "total_frames": total_frames,
        "shapes_tail": shapes_tail,
        "chunk_jobs": chunk_jobs,
    })

    # # Original serial clip-by-clip copy (kept for reference; superseded by the
    # # parallel chunk-wise pool below). Each iteration triggered a chunk RMW for
    # # every selected clip × every per-frame array, so it scaled poorly.
    # for k, clip_i in enumerate(all_selected):
    #     s, e = int(clip_start[clip_i]), int(clip_end[clip_i])
    #     ns, ne = int(new_start[k]), int(new_end[k])
    #     out_joint_pos[ns:ne] = src["joint_pos"][s:e]
    #     out_joint_vel[ns:ne] = src["joint_vel"][s:e]
    #     out_body_pos[ns:ne] = src["body_pos_w"][s:e]
    #     out_body_quat[ns:ne] = src["body_quat_w"][s:e]
    #     out_body_lin[ns:ne] = src["body_lin_vel_w"][s:e]
    #     out_body_ang[ns:ne] = src["body_ang_vel_w"][s:e]
    #     if (k + 1) % 25 == 0 or k == len(all_selected) - 1:
    #         print(f"  copied {k + 1}/{len(all_selected)} clips ({ne} frames)")

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
                if completed % 25 == 0 or completed == n_out_chunks:
                    print(f"  wrote {completed}/{n_out_chunks} chunks")

    # Metadata arrays
    sel_names = np.array([clip_names[i] for i in all_selected], dtype=object)
    sel_props = np.array([content_props[i] for i in all_selected], dtype=np.int32)
    sel_props_desc = np.array([content_props_desc[i] for i in all_selected], dtype=object)

    dst.array("clip_names", sel_names, dtype=object, object_codec=zarr.codecs.VLenUTF8())
    dst.array("clip_start_idx", new_start.astype(np.int64))
    dst.array("clip_end_idx", new_end.astype(np.int64))
    dst.array("content_props", sel_props)
    dst.array("content_props_desc", sel_props_desc, dtype=object, object_codec=zarr.codecs.VLenUTF8())
    dst.array("fps", fps.astype(np.int32))
    dst.array("body_names", body_names, dtype=object, object_codec=zarr.codecs.VLenUTF8())

    print(f"\n[done] wrote {dst_path}")
    print("\nSample selected clip names:")
    for i in range(0, len(sel_names), max(1, len(sel_names) // 20)):
        print(f"  [{i}] {sel_names[i]}")


if __name__ == "__main__":
    main()
