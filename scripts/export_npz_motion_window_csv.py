"""Export a pinned G1 NPZ frame window to the RMR CSV converter contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
REQUIRED_KEYS = (
    "fps",
    "dof_names",
    "body_names",
    "dof_positions",
    "body_positions",
    "body_rotations",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def export_window(
    source_path: Path,
    output_path: Path,
    *,
    source_sha256: str,
    start_frame: int,
    end_frame_inclusive: int,
) -> dict[str, object]:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if not source_path.is_file():
        raise ValueError(f"source NPZ does not exist: {source_path}")
    actual_sha256 = sha256_file(source_path)
    if actual_sha256 != source_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {source_sha256}, got {actual_sha256}"
        )

    with np.load(source_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_KEYS).difference(archive.files))
        if missing:
            raise ValueError(f"source NPZ missing arrays: {missing}")
        fps = np.asarray(archive["fps"]).reshape(-1)
        dof_names = tuple(map(str, archive["dof_names"]))
        body_names = tuple(map(str, archive["body_names"]))
        dof_positions = np.asarray(archive["dof_positions"])
        body_positions = np.asarray(archive["body_positions"])
        body_rotations = np.asarray(archive["body_rotations"])

    if fps.size != 1 or float(fps[0]) != 30.0:
        raise ValueError("source fps must equal 30")
    if dof_names != G1_JOINT_NAMES:
        raise ValueError("source G1 joint order does not match the RMR CSV contract")
    if not body_names or body_names[0] != "pelvis":
        raise ValueError("source body index zero must be pelvis")
    frames = dof_positions.shape[0]
    if dof_positions.shape != (frames, 29):
        raise ValueError("dof_positions must have shape (T, 29)")
    if body_positions.shape != (frames, len(body_names), 3):
        raise ValueError("body_positions has an incompatible shape")
    if body_rotations.shape != (frames, len(body_names), 4):
        raise ValueError("body_rotations has an incompatible shape")
    if not 0 <= start_frame <= end_frame_inclusive < frames:
        raise ValueError("requested frame window is outside the source")

    selection = slice(start_frame, end_frame_inclusive + 1)
    table = np.concatenate(
        (
            body_positions[selection, 0],
            body_rotations[selection, 0][:, [1, 2, 3, 0]],
            dof_positions[selection],
        ),
        axis=1,
    )
    if table.shape != (end_frame_inclusive - start_frame + 1, 36):
        raise ValueError("assembled CSV has an incompatible shape")
    if not np.isfinite(table).all():
        raise ValueError("assembled CSV contains non-finite values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".tmp")
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        np.savetxt(temporary_output, table, fmt="%.8f", delimiter=",")
        temporary_output.replace(output_path)
        manifest: dict[str, object] = {
            "format": "rmr-g1-motion-csv-v1",
            "input_path": str(source_path),
            "input_sha256": actual_sha256,
            "input_fps": 30,
            "input_frame_range_zero_indexed_inclusive": [
                start_frame,
                end_frame_inclusive,
            ],
            "joint_names": list(G1_JOINT_NAMES),
            "output_path": str(output_path),
            "output_sha256": sha256_file(output_path),
            "output_frames": int(table.shape[0]),
            "output_columns": int(table.shape[1]),
            "root_quaternion_order": "xyzw",
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_manifest.replace(manifest_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame-inclusive", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_window(
        args.source_path,
        args.output_path,
        source_sha256=args.source_sha256,
        start_frame=args.start_frame,
        end_frame_inclusive=args.end_frame_inclusive,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
