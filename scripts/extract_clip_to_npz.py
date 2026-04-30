"""Extract a single clip from a multi-clip zarr store into a standalone .npz.

The output .npz matches the single-motion format consumed by
`whole_body_tracking.tasks.tracking.mdp.commands.MotionLoader` (same format
produced by `scripts/csv_to_npz.py`):

    fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w
    [+ wrist_grasp_label if present in the zarr]

Usage:
    # By name (substring match; first hit wins, like play_bones_clip.py)
    python scripts/extract_clip_to_npz.py \
        --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
        --clip_name=Jump_002__A017_M

    # By clip_id (index into the filtered set — same ids shown in eval jsonl)
    python scripts/extract_clip_to_npz.py \
        --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
        --clip_id=86

    # Output is always written to ./motions/{clip_id}_{clip_name}.npz

    # Include "object manipulation" clips in the id space (disabled by default
    # to match eval/play scripts). Only affects clip_id indexing and name search.
    python scripts/extract_clip_to_npz.py ... --include_objects
"""

import argparse
import os

import numpy as np
import zarr


MOTION_ARRAY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def resolve_clip_id(
    store: zarr.Group,
    clip_id: int | None,
    clip_name: str | None,
    exclude_objects: bool,
) -> tuple[int, int, str, str]:
    """Return (filtered_clip_id, raw_clip_id, clip_name, clip_desc).

    `raw_clip_id` is the index into the unfiltered zarr arrays (what we need
    for slicing). `filtered_clip_id` matches play_bones_clip.py and the eval
    jsonl — the index AFTER filtering object-manipulation clips (when
    exclude_objects=True).
    """
    all_clip_start = store["clip_start_idx"][:]
    total_raw = len(all_clip_start)

    all_names = [""] * total_raw
    if "clip_names" in store:
        raw = store["clip_names"][:]
        for i in range(min(len(raw), total_raw)):
            all_names[i] = str(raw[i])

    all_descs = [""] * total_raw
    if "content_props_desc" in store:
        raw = store["content_props_desc"][:]
        for i in range(min(len(raw), total_raw)):
            all_descs[i] = str(raw[i])

    exclude_props = [
        "object manipulation", "wall", "chair", "obstacle",
        "edge", "safety pad", "railing", "box",
    ] if exclude_objects else None
    if exclude_props and "content_props_desc" in store:
        valid_indices = [
            i for i in range(total_raw)
            if not any(ep.lower() in all_descs[i].strip().lower() for ep in exclude_props)
        ]
    else:
        valid_indices = list(range(total_raw))

    filtered_names = [all_names[i] for i in valid_indices]

    if clip_id is not None:
        assert 0 <= clip_id < len(valid_indices), (
            f"clip_id {clip_id} out of range [0, {len(valid_indices)})"
        )
        raw_id = valid_indices[clip_id]
        print(f"[CLIP] clip_id={clip_id} (filtered) → raw_id={raw_id} \"{filtered_names[clip_id]}\"")
        return clip_id, raw_id, filtered_names[clip_id], all_descs[raw_id]

    if clip_name is not None:
        matches = [(i, n) for i, n in enumerate(filtered_names) if clip_name.lower() in n.lower()]
        if not matches:
            raise ValueError(
                f"No clips matching \"{clip_name}\" in {len(filtered_names)} filtered clips."
            )
        if len(matches) > 1:
            print(f"[CLIP] Multiple matches for \"{clip_name}\":")
            for i, n in matches[:20]:
                print(f"  clip_id={i}: {n}")
            if len(matches) > 20:
                print(f"  ... and {len(matches) - 20} more")
            print(f"[CLIP] Using first match: clip_id={matches[0][0]}")
        filt_id, name = matches[0]
        raw_id = valid_indices[filt_id]
        return filt_id, raw_id, name, all_descs[raw_id]

    raise ValueError("Must specify either --clip_id or --clip_name.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zarr_path", type=str, required=True,
                        help="Path to multi-clip zarr store.")
    parser.add_argument("--clip_id", type=int, default=None,
                        help="Clip index in the filtered dataset (matches eval jsonl).")
    parser.add_argument("--clip_name", type=str, default=None,
                        help="Clip name (substring match).")
    parser.add_argument("--include_objects", action="store_true", default=False,
                        help="Disable scene-prop filtering (default: filter out wall/chair/obstacle/etc.).")
    args = parser.parse_args()

    if (args.clip_id is None) == (args.clip_name is None):
        parser.error("Specify exactly one of --clip_id or --clip_name.")

    assert os.path.isdir(args.zarr_path), f"Not a zarr dir: {args.zarr_path}"
    store = zarr.open_group(args.zarr_path, mode="r")

    exclude_objects = not args.include_objects
    filt_id, raw_id, clip_name, clip_desc = resolve_clip_id(
        store, args.clip_id, args.clip_name, exclude_objects
    )

    start = int(store["clip_start_idx"][raw_id])
    end = int(store["clip_end_idx"][raw_id])  # exclusive
    n_frames = end - start
    fps = int(store["fps"][0])
    print(f"[CLIP] \"{clip_name}\" desc={clip_desc!r}  frames=[{start}, {end}) ({n_frames} @ {fps} fps)")

    out: dict[str, np.ndarray] = {"fps": np.array([fps], dtype=np.int64)}
    for key in MOTION_ARRAY_KEYS:
        out[key] = np.asarray(store[key][start:end], dtype=np.float32)

    if "wrist_grasp_label" in store:
        out["wrist_grasp_label"] = np.asarray(store["wrist_grasp_label"][start:end], dtype=bool)

    output_path = os.path.join("motions", f"{filt_id}_{clip_name}.npz")
    os.makedirs("motions", exist_ok=True)
    np.savez(output_path, **out)

    print(f"[CLIP] Wrote {output_path}")
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")


if __name__ == "__main__":
    main()
