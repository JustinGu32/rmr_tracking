from __future__ import annotations

import numpy as np
import pytest

from scripts.build_source_consistent_reference import (
    assemble_corrected_reference,
    compare_exact_prefix,
)


def _arrays(frames: int) -> dict[str, np.ndarray]:
    return {
        "fps": np.asarray([50], dtype=np.int64),
        "joint_pos": np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
        "joint_vel": np.arange(frames * 2, dtype=np.float32).reshape(frames, 2),
        "body_pos_w": np.arange(frames * 6, dtype=np.float32).reshape(frames, 2, 3),
        "body_quat_w": np.arange(frames * 8, dtype=np.float32).reshape(frames, 2, 4),
        "body_lin_vel_w": np.arange(frames * 6, dtype=np.float32).reshape(frames, 2, 3),
        "body_ang_vel_w": np.arange(frames * 6, dtype=np.float32).reshape(frames, 2, 3),
    }


def test_compare_exact_prefix_accepts_all_load_bearing_arrays(tmp_path) -> None:
    candidate = tmp_path / "candidate.npz"
    baseline = tmp_path / "baseline.npz"
    candidate_arrays = _arrays(5)
    baseline_arrays = {
        name: value[:3].copy() if value.ndim > 1 else value.copy()
        for name, value in candidate_arrays.items()
    }
    np.savez(candidate, **candidate_arrays)
    np.savez(baseline, **baseline_arrays)

    result = compare_exact_prefix(candidate, baseline, prefix_frames=3)

    assert result["all_arrays_equal"] is True
    assert result["arrays_equal"] == {
        "fps": True,
        "joint_pos": True,
        "joint_vel": True,
        "body_pos_w": True,
        "body_quat_w": True,
        "body_lin_vel_w": True,
        "body_ang_vel_w": True,
    }


def test_compare_exact_prefix_rejects_one_body_value_change(tmp_path) -> None:
    candidate = tmp_path / "candidate.npz"
    baseline = tmp_path / "baseline.npz"
    candidate_arrays = _arrays(5)
    baseline_arrays = {
        name: value[:3].copy() if value.ndim > 1 else value.copy()
        for name, value in candidate_arrays.items()
    }
    candidate_arrays["body_pos_w"][1, 1, 2] += 0.25
    np.savez(candidate, **candidate_arrays)
    np.savez(baseline, **baseline_arrays)

    with pytest.raises(ValueError, match="body_pos_w"):
        compare_exact_prefix(candidate, baseline, prefix_frames=3)


def test_assemble_replaces_only_nonroot_suffix_body_fields(tmp_path) -> None:
    source_path = tmp_path / "source.npz"
    fk_path = tmp_path / "fk.npz"
    output_path = tmp_path / "corrected.npz"
    source = _arrays(5)
    source.update(
        joint_names=np.asarray(["joint_a", "joint_b"]),
        root_body_name=np.asarray("pelvis"),
        root_body_index=np.asarray(0, dtype=np.int32),
    )
    fk = _arrays(5)
    for name in (
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        fk[name] += 1000.0
    np.savez(source_path, **source)
    np.savez(fk_path, **fk)

    result = assemble_corrected_reference(
        source_path,
        fk_path,
        output_path,
        prefix_frames=3,
        body_names=("pelvis", "torso_link"),
    )

    with np.load(output_path, allow_pickle=False) as output:
        for name in ("joint_pos", "joint_vel"):
            np.testing.assert_array_equal(output[name], source[name])
        for name in (
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            np.testing.assert_array_equal(output[name][:3], source[name][:3])
            np.testing.assert_array_equal(output[name][3:, 0], source[name][3:, 0])
            np.testing.assert_array_equal(output[name][3:, 1:], fk[name][3:, 1:])
        assert tuple(map(str, output["body_names"])) == ("pelvis", "torso_link")
    assert result["load_bearing_arrays_preserved"] is True
    assert result["exact_prefix_preserved"] is True
