from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.export_npz_motion_window_csv import G1_JOINT_NAMES, export_window


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(path) -> None:
    frames = 5
    positions = np.zeros((frames, 2, 3), dtype=np.float32)
    positions[:, 0] = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    rotations = np.zeros((frames, 2, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    rotations[:, 0] = np.asarray([0.7, 0.1, 0.2, 0.3], dtype=np.float32)
    dof = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29)
    np.savez(
        path,
        fps=np.asarray([30.0], dtype=np.float32),
        dof_names=np.asarray(G1_JOINT_NAMES),
        body_names=np.asarray(["pelvis", "torso_link"]),
        dof_positions=dof,
        dof_velocities=np.zeros_like(dof),
        body_positions=positions,
        body_rotations=rotations,
        body_linear_velocities=np.zeros_like(positions),
        body_angular_velocities=np.zeros_like(positions),
    )


def test_export_window_uses_zero_indexed_inclusive_bounds_and_xyzw(tmp_path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "window.csv"
    _source_fixture(source)

    manifest = export_window(
        source,
        output,
        source_sha256=_sha256(source),
        start_frame=1,
        end_frame_inclusive=3,
    )

    table = np.loadtxt(output, delimiter=",")
    with np.load(source, allow_pickle=False) as archive:
        np.testing.assert_allclose(table[:, :3], archive["body_positions"][1:4, 0])
        np.testing.assert_allclose(
            table[:, 3:7], archive["body_rotations"][1:4, 0][:, [1, 2, 3, 0]]
        )
        np.testing.assert_allclose(table[:, 7:], archive["dof_positions"][1:4])
    assert manifest["input_frame_range_zero_indexed_inclusive"] == [1, 3]
    assert manifest["output_frames"] == 3
    assert manifest["output_sha256"] == _sha256(output)
    assert json.loads(output.with_suffix(".csv.manifest.json").read_text()) == manifest


def test_export_window_refuses_wrong_hash_and_overwrite(tmp_path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "window.csv"
    _source_fixture(source)

    with pytest.raises(ValueError, match="SHA-256"):
        export_window(
            source,
            output,
            source_sha256="0" * 64,
            start_frame=0,
            end_frame_inclusive=2,
        )

    output.write_text("occupied")
    with pytest.raises(FileExistsError):
        export_window(
            source,
            output,
            source_sha256=_sha256(source),
            start_frame=0,
            end_frame_inclusive=2,
        )
