#!/usr/bin/env python3
"""Verify that a holosoma-exported ONNX produces identical outputs to a
standard-exported + adapt_onnx-adapted ONNX.

Usage:
    python scripts/verify_holosoma_export.py \
        --holosoma path/to/policy_holosoma.onnx \
        --adapted  path/to/policy_adapted.onnx \
        [--num-steps 5] [--atol 1e-5] [--rtol 1e-4]

Both models must accept inputs named "obs" (shape [1, obs_dim]) and
"time_step" (shape [1, 1]), and produce outputs named "actions",
"joint_pos", "joint_vel", "ref_pos_xyz" (or equivalent), "ref_quat_xyzw".

The script feeds identical random observations to both models and compares
their outputs element-by-element.

Note: The holosoma model expects obs in holosoma's alphabetically-sorted
layout.  The adapted model may also expect this layout (if --reorder-obs
was used), or may expect the training declaration layout.  This script
assumes BOTH models accept the same obs layout (holosoma alphabetical),
which is the case when:
  - The holosoma model was exported with holosoma_exporter.py
  - The adapted model was produced by adapt_onnx.py with --reorder-obs

If the adapted model was produced WITHOUT --reorder-obs, you need to
permute obs differently for each model.  This script does not handle that
case (use --reorder-obs when adapting to make this comparison valid).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime


def load_metadata(model_path: str) -> dict:
    """Load ONNX metadata as a dict."""
    model = onnx.load(model_path)
    meta = {}
    for prop in model.metadata_props:
        try:
            meta[prop.key] = json.loads(prop.value)
        except (json.JSONDecodeError, ValueError):
            meta[prop.key] = prop.value
    return meta


def get_output_names(session: onnxruntime.InferenceSession) -> list[str]:
    return [o.name for o in session.get_outputs()]


def get_input_shapes(session: onnxruntime.InferenceSession) -> dict[str, list[int]]:
    return {i.name: i.shape for i in session.get_inputs()}


def run_model(
    session: onnxruntime.InferenceSession,
    obs: np.ndarray,
    time_step: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run an ONNX session and return outputs keyed by name."""
    output_names = get_output_names(session)
    results = session.run(output_names, {"obs": obs, "time_step": time_step})
    return dict(zip(output_names, results))


def compare_outputs(
    holosoma_out: dict[str, np.ndarray],
    adapted_out: dict[str, np.ndarray],
    atol: float,
    rtol: float,
    step: int,
) -> bool:
    """Compare outputs from two models. Returns True if all match."""
    all_ok = True

    # Map common output names (adapted model may have slightly different names)
    output_pairs = [
        ("actions", "actions"),
        ("joint_pos", "joint_pos"),
        ("joint_vel", "joint_vel"),
        ("ref_pos_xyz", "ref_pos_xyz"),
        ("ref_quat_xyzw", "ref_quat_xyzw"),
    ]

    for holo_name, adapt_name in output_pairs:
        if holo_name not in holosoma_out:
            print(f"  [SKIP] '{holo_name}' not in holosoma model outputs")
            continue
        if adapt_name not in adapted_out:
            print(f"  [SKIP] '{adapt_name}' not in adapted model outputs")
            continue

        h = holosoma_out[holo_name]
        a = adapted_out[adapt_name]

        if h.shape != a.shape:
            print(f"  [FAIL] {holo_name}: shape mismatch: holosoma={h.shape}, adapted={a.shape}")
            all_ok = False
            continue

        if np.allclose(h, a, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(h - a))
            print(f"  [PASS] {holo_name}: max_diff={max_diff:.2e} (shape {h.shape})")
        else:
            max_diff = np.max(np.abs(h - a))
            mean_diff = np.mean(np.abs(h - a))
            print(
                f"  [FAIL] {holo_name}: max_diff={max_diff:.2e}, "
                f"mean_diff={mean_diff:.2e} (shape {h.shape})"
            )
            # Print first few differing elements
            diff_mask = ~np.isclose(h, a, atol=atol, rtol=rtol)
            diff_indices = np.argwhere(diff_mask)
            for idx in diff_indices[:5]:
                idx_tuple = tuple(idx)
                print(
                    f"    idx={idx_tuple}: holosoma={h[idx_tuple]:.8f}, "
                    f"adapted={a[idx_tuple]:.8f}, diff={abs(h[idx_tuple]-a[idx_tuple]):.2e}"
                )
            if len(diff_indices) > 5:
                print(f"    ... and {len(diff_indices) - 5} more differing elements")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify holosoma-exported ONNX matches adapted ONNX"
    )
    parser.add_argument(
        "--holosoma", required=True, help="Path to holosoma-exported ONNX"
    )
    parser.add_argument(
        "--adapted", required=True, help="Path to standard-exported + adapted ONNX"
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="Number of time steps to test"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="Absolute tolerance for comparison"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-4, help="Relative tolerance for comparison"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading holosoma model: {args.holosoma}")
    holo_session = onnxruntime.InferenceSession(args.holosoma)
    holo_shapes = get_input_shapes(holo_session)
    print(f"  Inputs: {holo_shapes}")
    print(f"  Outputs: {get_output_names(holo_session)}")

    print(f"\nLoading adapted model: {args.adapted}")
    adapt_session = onnxruntime.InferenceSession(args.adapted)
    adapt_shapes = get_input_shapes(adapt_session)
    print(f"  Inputs: {adapt_shapes}")
    print(f"  Outputs: {get_output_names(adapt_session)}")

    # Verify input shapes match
    holo_obs_dim = holo_shapes["obs"][1]
    adapt_obs_dim = adapt_shapes["obs"][1]
    if holo_obs_dim != adapt_obs_dim:
        print(
            f"\nERROR: Obs dimensions differ: holosoma={holo_obs_dim}, adapted={adapt_obs_dim}"
        )
        sys.exit(1)

    print(f"\nObs dimension: {holo_obs_dim}")
    print(f"Testing {args.num_steps} time steps with atol={args.atol}, rtol={args.rtol}")

    # Load metadata for info
    holo_meta = load_metadata(args.holosoma)
    adapt_meta = load_metadata(args.adapted)
    print(f"\nHolosoma dof_names (first 5): {holo_meta.get('dof_names', [])[:5]}")
    print(f"Adapted dof_names (first 5): {adapt_meta.get('dof_names', [])[:5]}")
    print(f"Holosoma obs_names: {holo_meta.get('observation_names', [])}")
    print(f"Adapted obs_names: {adapt_meta.get('observation_names', [])}")

    all_passed = True
    for step in range(args.num_steps):
        print(f"\n--- Step {step} ---")
        obs = np.random.randn(1, holo_obs_dim).astype(np.float32) * 0.1
        time_step = np.array([[step]], dtype=np.float32)

        holo_out = run_model(holo_session, obs, time_step)
        adapt_out = run_model(adapt_session, obs, time_step)

        step_ok = compare_outputs(holo_out, adapt_out, args.atol, args.rtol, step)
        if not step_ok:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("The holosoma-exported ONNX produces identical outputs to the adapted ONNX.")
    else:
        print("RESULT: SOME CHECKS FAILED")
        print("The models produce different outputs. See details above.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
