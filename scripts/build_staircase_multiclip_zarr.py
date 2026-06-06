"""Build a staircase multiclip Zarr store from motion npz clips and staircase metadata.

Example:
    python scripts/build_staircase_multiclip_zarr.py \
        --npz /path/walk_up.npz \
        --npz /path/walk_down.npz \
        --metadata /path/walk_up/staircase_metadata.json \
        --metadata /path/walk_down/staircase_metadata.json \
        --staircase-pos 3.2 -0.2 0.0 \
        --staircase-pos 3.2 1.95 0.0 \
        --staircase-quat 0.68199836 0.0 0.0 0.73135370 \
        --staircase-quat 0.68199836 0.0 0.0 0.73135370 \
        --output-zarr data/staircase_multiclip.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


ARRAY_KEYS = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def decode_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_asset_paths(metadata_path: Path) -> tuple[str, str, str]:
    asset_dir = metadata_path.parent
    asset_path = asset_dir / "multi_boxes.urdf"
    raycast_asset_path = asset_dir / "multi_boxes_combined.urdf"
    usd_dir = Path("/tmp/isaaclab_staircase_usd") / asset_dir.name
    if not asset_path.exists():
        raise FileNotFoundError(f"Expected staircase asset next to metadata: {asset_path}")
    return str(asset_path.resolve()), str(raycast_asset_path.resolve()), str(usd_dir.resolve())


def parse_triplets(values: list[list[float]] | None, expected_len: int, name: str, width: int) -> list[list[float]] | None:
    if values is None:
        return None
    if len(values) != expected_len:
        raise ValueError(f"{name} count ({len(values)}) must match clip count ({expected_len}).")
    for value in values:
        if len(value) != width:
            raise ValueError(f"Each {name} entry must have width {width}, got {value}")
    return [list(map(float, value)) for value in values]


def create_string_dataset(root, name: str, values: list[str]):
    root.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, "VLenUTF8") else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Build staircase multiclip zarr from motion npz clips.")
    parser.add_argument("--npz", dest="npz_paths", action="append", required=True, help="Input motion npz path. Repeat once per clip.")
    parser.add_argument(
        "--metadata",
        dest="metadata_paths",
        action="append",
        required=True,
        help="Input staircase_metadata.json path. Repeat once per clip.",
    )
    parser.add_argument(
        "--staircase-pos",
        dest="staircase_positions",
        action="append",
        nargs=3,
        type=float,
        help="Per-clip staircase local position (x y z). Repeat once per clip unless the metadata JSON already contains staircase_pos.",
    )
    parser.add_argument(
        "--staircase-quat",
        dest="staircase_quats",
        action="append",
        nargs=4,
        type=float,
        help="Per-clip staircase local quaternion (w x y z). Repeat once per clip unless the metadata JSON already contains staircase_quat.",
    )
    parser.add_argument("--output-zarr", type=Path, required=True, help="Output Zarr directory.")
    args = parser.parse_args()

    npz_paths = [Path(path).resolve() for path in args.npz_paths]
    metadata_paths = [Path(path).resolve() for path in args.metadata_paths]
    if len(npz_paths) != len(metadata_paths):
        raise ValueError("--npz and --metadata must be provided the same number of times.")

    cli_positions = parse_triplets(args.staircase_positions, len(npz_paths), "staircase-pos", 3)
    cli_quats = parse_triplets(args.staircase_quats, len(npz_paths), "staircase-quat", 4)

    root = zarr.open(str(args.output_zarr), mode="w")
    motions = []
    clip_names: list[str] = []
    clip_start_idx: list[int] = []
    clip_end_idx: list[int] = []
    staircase_ids: list[int] = []
    staircase_pos: list[list[float]] = []
    staircase_quat: list[list[float]] = []
    staircase_asset_path: list[str] = []
    staircase_raycast_asset_path: list[str] = []
    staircase_usd_dir: list[str] = []
    staircase_asset_key_to_id: dict[tuple[str, str], int] = {}
    asset_manifest: dict[int, dict] = {}

    fps = None
    cursor = 0
    for index, (npz_path, metadata_path) in enumerate(zip(npz_paths, metadata_paths)):
        motion = np.load(npz_path)
        if fps is None:
            fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
        elif fps != int(np.asarray(motion["fps"]).reshape(-1)[0]):
            raise ValueError(f"All clips must share fps. Got {fps} and {motion['fps']} from {npz_path}")

        for key in ARRAY_KEYS:
            if key not in motion:
                raise KeyError(f"Missing {key} in {npz_path}")

        metadata = decode_json(metadata_path)
        pos_value = metadata.get("staircase_pos", cli_positions[index] if cli_positions is not None else None)
        quat_value = metadata.get("staircase_quat", cli_quats[index] if cli_quats is not None else None)
        if pos_value is None or quat_value is None:
            raise ValueError(
                f"Missing staircase pose for clip {npz_path.name}. "
                "Provide --staircase-pos/--staircase-quat or add staircase_pos/staircase_quat to the metadata JSON."
            )

        asset_path, raycast_asset_path, usd_dir = infer_asset_paths(metadata_path)
        asset_key = (asset_path, raycast_asset_path)
        if asset_key not in staircase_asset_key_to_id:
            staircase_asset_key_to_id[asset_key] = len(staircase_asset_key_to_id)
        staircase_id = staircase_asset_key_to_id[asset_key]
        asset_manifest[staircase_id] = {
            "staircase_id": staircase_id,
            "asset_path": asset_path,
            "raycast_asset_path": raycast_asset_path,
            "usd_dir": usd_dir,
            "metadata_path": str(metadata_path),
        }

        clip_len = int(motion["joint_pos"].shape[0])
        motions.append({key: np.asarray(motion[key], dtype=np.float32) for key in ARRAY_KEYS})
        clip_names.append(npz_path.stem)
        clip_start_idx.append(cursor)
        cursor += clip_len
        clip_end_idx.append(cursor)
        staircase_ids.append(staircase_id)
        staircase_pos.append(list(map(float, pos_value)))
        staircase_quat.append(list(map(float, quat_value)))
        staircase_asset_path.append(asset_path)
        staircase_raycast_asset_path.append(raycast_asset_path)
        staircase_usd_dir.append(usd_dir)

    if fps is None:
        raise ValueError("No clips provided.")

    root.create_dataset("fps", data=np.array([fps], dtype=np.int32))
    for key in ARRAY_KEYS:
        root.create_dataset(key, data=np.concatenate([motion[key] for motion in motions], axis=0), dtype=np.float32)
    root.create_dataset("clip_start_idx", data=np.asarray(clip_start_idx, dtype=np.int64))
    root.create_dataset("clip_end_idx", data=np.asarray(clip_end_idx, dtype=np.int64))
    create_string_dataset(root, "clip_names", clip_names)
    root.create_dataset("staircase_id", data=np.asarray(staircase_ids, dtype=np.int32))
    root.create_dataset("staircase_pos", data=np.asarray(staircase_pos, dtype=np.float32))
    root.create_dataset("staircase_quat", data=np.asarray(staircase_quat, dtype=np.float32))
    create_string_dataset(root, "staircase_asset_path", staircase_asset_path)
    create_string_dataset(root, "staircase_raycast_asset_path", staircase_raycast_asset_path)
    create_string_dataset(root, "staircase_usd_dir", staircase_usd_dir)

    manifest_path = args.output_zarr.with_suffix(".staircase_assets.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({"variants": [asset_manifest[key] for key in sorted(asset_manifest)]}, f, indent=2)

    print(
        f"[build_staircase_multiclip_zarr] wrote {args.output_zarr} with "
        f"{len(clip_names)} clips, {cursor} frames, {len(asset_manifest)} staircase variants"
    )
    print(f"[build_staircase_multiclip_zarr] asset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
