"""Build a source-consistent long G1 reference through the native Isaac FK path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.export_npz_motion_window_csv import export_window

PROTOCOL = "g1-source-consistent-reference-build-v1"
RAW_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
BODY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return resolved


def _load_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(RAW_KEYS).difference(archive.files))
        if missing:
            raise ValueError(f"reference missing arrays: {missing}")
        return {name: np.asarray(archive[name]) for name in RAW_KEYS}


def compare_exact_prefix(
    candidate_path: Path, baseline_path: Path, *, prefix_frames: int
) -> dict[str, object]:
    """Require the candidate's raw prefix to be byte-value exact with baseline."""
    candidate = _load_raw(candidate_path)
    baseline = _load_raw(baseline_path)
    if prefix_frames <= 0:
        raise ValueError("prefix_frames must be positive")
    arrays_equal: dict[str, bool] = {}
    maximum_errors: dict[str, float] = {}
    for name in RAW_KEYS:
        left = candidate[name]
        right = baseline[name]
        if name == "fps":
            selected = left
        else:
            if right.shape[0] != prefix_frames or left.shape[0] < prefix_frames:
                raise ValueError(f"{name} does not have the requested prefix length")
            selected = left[:prefix_frames]
        if selected.shape != right.shape:
            raise ValueError(
                f"{name} prefix shape {selected.shape} != baseline {right.shape}"
            )
        equal = bool(np.array_equal(selected, right))
        arrays_equal[name] = equal
        maximum_errors[name] = float(
            np.max(np.abs(selected.astype(np.float64) - right.astype(np.float64)))
        )
        if not equal:
            raise ValueError(
                f"candidate prefix differs from baseline {name} "
                f"(max abs {maximum_errors[name]})"
            )
    return {
        "prefix_frames": prefix_frames,
        "arrays_equal": arrays_equal,
        "maximum_absolute_errors": maximum_errors,
        "all_arrays_equal": all(arrays_equal.values()),
    }


def assemble_corrected_reference(
    source_path: Path,
    fk_path: Path,
    output_path: Path,
    *,
    prefix_frames: int,
    body_names: tuple[str, ...],
) -> dict[str, object]:
    """Preserve every source field except non-root body state in the suffix."""
    if output_path.exists():
        raise FileExistsError(output_path)
    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: np.array(archive[name], copy=True) for name in archive.files}
    fk = _load_raw(fk_path)
    missing = sorted(set(RAW_KEYS).difference(source))
    if missing:
        raise ValueError(f"load-bearing source missing arrays: {missing}")
    frames = int(np.asarray(source["joint_pos"]).shape[0])
    if not 0 < prefix_frames < frames or len(body_names) <= 1:
        raise ValueError("invalid prefix or body count for corrected reference")
    for name in RAW_KEYS:
        if np.asarray(source[name]).shape != np.asarray(fk[name]).shape:
            raise ValueError(f"source and Isaac FK shapes differ for {name}")

    output = {name: np.array(value, copy=True) for name, value in source.items()}
    for name in BODY_KEYS:
        output[name][prefix_frames:, 1:] = fk[name][prefix_frames:, 1:]
    output["body_names"] = np.asarray(body_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **output)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise

    with np.load(output_path, allow_pickle=False) as archive:
        exact_prefix = all(
            np.array_equal(archive[name][:prefix_frames], source[name][:prefix_frames])
            for name in RAW_KEYS
            if name != "fps"
        )
        load_bearing = all(
            np.array_equal(archive[name], source[name])
            for name in ("fps", "joint_pos", "joint_vel")
        ) and all(
            np.array_equal(archive[name][:, 0], source[name][:, 0])
            for name in BODY_KEYS
        )
    if not exact_prefix or not load_bearing:
        raise ValueError("corrected reference changed protected source state")
    return {
        "frames": frames,
        "prefix_frames": prefix_frames,
        "exact_prefix_preserved": exact_prefix,
        "load_bearing_arrays_preserved": load_bearing,
        "replaced_fields": [f"{name}[{prefix_frames}:,1:]" for name in BODY_KEYS],
    }


def replay_state_errors(
    source_path: Path, fk_path: Path
) -> dict[str, float]:
    """Measure Isaac write/read drift before retaining only its body FK."""
    source = _load_raw(source_path)
    fk = _load_raw(fk_path)
    if any(source[name].shape != fk[name].shape for name in RAW_KEYS):
        raise ValueError("source and Isaac FK arrays have incompatible shapes")
    errors = {
        "joint_pos_max_abs": float(
            np.max(np.abs(source["joint_pos"] - fk["joint_pos"]))
        ),
        "joint_vel_max_abs": float(
            np.max(np.abs(source["joint_vel"] - fk["joint_vel"]))
        ),
    }
    for name in BODY_KEYS:
        errors[f"root_{name}_max_abs"] = float(
            np.max(np.abs(source[name][:, 0] - fk[name][:, 0]))
        )
    return errors


def _run_and_capture(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout: Path,
    stderr: Path,
) -> None:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"subprocess returned {result.returncode}; inspect {stdout} and {stderr}"
        )


def _git_identity(root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()

    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{u}")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    if head != expected_commit or upstream != expected_commit or tracked_status:
        raise ValueError(
            f"repository identity mismatch at {root}: head={head}, "
            f"upstream={upstream}, tracked_status={tracked_status!r}"
        )
    return {
        "root": str(root),
        "head": head,
        "upstream": upstream,
        "tracked_worktree_clean": True,
    }


def build_reference(args: argparse.Namespace) -> dict[str, object]:
    rmr_root = Path(__file__).resolve().parents[1]
    source = verify_file(args.source_path, args.source_sha256, "source motion")
    baseline_csv = verify_file(
        args.baseline_csv_path, args.baseline_csv_sha256, "baseline CSV"
    )
    baseline_prefix = verify_file(
        args.baseline_prefix_path,
        args.baseline_prefix_sha256,
        "baseline prefix reference",
    )
    load_bearing_reference = verify_file(
        args.load_bearing_reference_path,
        args.load_bearing_reference_sha256,
        "load-bearing long reference",
    )
    controller = verify_file(
        args.controller_path, args.controller_sha256, "controller"
    )
    converter = verify_file(
        args.converter_path, args.converter_sha256, "Isaac converter"
    )
    exporter = verify_file(
        Path(__file__).with_name("export_npz_motion_window_csv.py"),
        args.exporter_sha256,
        "CSV exporter",
    )
    repository_identity = {"rmr": _git_identity(rmr_root, args.rmr_code_commit)}

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "lafan_walk_win137_300.csv"
    raw_path = output_dir / "lafan_walk_win137_300_exact_state_isaac_fk.npz"
    named_path = output_dir / "lafan_walk_win137_300_source_consistent_named.npz"
    source_metadata_path = output_dir / "source_metadata.json"
    reference_manifest_path = output_dir / "reference_manifest.json"

    csv_manifest = export_window(
        source,
        csv_path,
        source_sha256=args.source_sha256,
        start_frame=args.start_frame,
        end_frame_inclusive=args.end_frame_inclusive,
    )
    baseline_lines = baseline_csv.read_bytes().splitlines(keepends=True)
    candidate_lines = csv_path.read_bytes().splitlines(keepends=True)
    csv_prefix_exact = (
        b"".join(candidate_lines[: len(baseline_lines)]) == baseline_csv.read_bytes()
    )
    if not csv_prefix_exact:
        raise ValueError(
            "extended source CSV does not preserve the exact baseline CSV prefix"
        )

    converter_argv = [
        str(Path(args.isaac_python).resolve()),
        str(converter),
        "--reference-input-path",
        str(load_bearing_reference),
        "--input-sha256",
        args.load_bearing_reference_sha256,
        "--output-fps",
        "50",
        "--output-path",
        str(raw_path),
        "--headless",
        "--device",
        args.device,
    ]
    _run_and_capture(
        converter_argv,
        cwd=rmr_root,
        environment=os.environ,
        stdout=output_dir / "isaac_converter_stdout.log",
        stderr=output_dir / "isaac_converter_stderr.log",
    )
    raw = _load_raw(raw_path)
    if raw["joint_pos"].shape != (args.expected_output_frames, 29):
        raise ValueError(
            f"expected {args.expected_output_frames} converted frames, got "
            f"{raw['joint_pos'].shape}"
        )
    state_errors = replay_state_errors(load_bearing_reference, raw_path)
    if max(state_errors.values()) > args.maximum_replay_state_error:
        raise ValueError(
            "Isaac state write/read drift exceeds the registered bound: "
            f"{state_errors}"
        )

    joint_names_path = Path(str(raw_path) + ".joint_names.npy")
    body_names_path = Path(str(raw_path) + ".body_names.npy")
    joint_names = tuple(map(str, np.load(joint_names_path, allow_pickle=False)))
    body_names = tuple(map(str, np.load(body_names_path, allow_pickle=False)))
    with np.load(controller, allow_pickle=False) as archive:
        controller_joint_names = tuple(map(str, archive["joint_names"]))
    if joint_names != controller_joint_names:
        raise ValueError("Isaac joint order differs from the pinned controller order")
    if len(body_names) != 30 or len(set(body_names)) != 30 or body_names[0] != "pelvis":
        raise ValueError("Isaac body order is not the expected 30-body pelvis-rooted G1")

    repair = assemble_corrected_reference(
        load_bearing_reference,
        raw_path,
        named_path,
        prefix_frames=args.prefix_frames,
        body_names=body_names,
    )
    exact_prefix = compare_exact_prefix(
        named_path, baseline_prefix, prefix_frames=args.prefix_frames
    )

    source_metadata = {
        "protocol": PROTOCOL,
        "source_motion": {"path": str(source), "sha256": args.source_sha256},
        "source_window_zero_indexed_inclusive": [
            args.start_frame,
            args.end_frame_inclusive,
        ],
        "baseline_csv": {
            "path": str(baseline_csv),
            "sha256": args.baseline_csv_sha256,
            "exact_prefix": csv_prefix_exact,
        },
        "baseline_reference": {
            "path": str(baseline_prefix),
            "sha256": args.baseline_prefix_sha256,
            "prefix_frames": args.prefix_frames,
            "exact_raw_array_prefix": True,
        },
        "load_bearing_long_reference": {
            "path": str(load_bearing_reference),
            "sha256": args.load_bearing_reference_sha256,
            "preserved_fields": [
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w[:,0]",
                "body_quat_w[:,0]",
                "body_lin_vel_w[:,0]",
                "body_ang_vel_w[:,0]",
            ],
        },
        "exporter": {"path": str(exporter), "sha256": args.exporter_sha256},
        "source_csv_manifest": csv_manifest,
        "isaac_converter": {
            "path": str(converter),
            "sha256": args.converter_sha256,
            "python": str(Path(args.isaac_python).resolve()),
            "device": args.device,
            "mode": "exact-named-reference-state-no-resampling",
        },
        "repository_identity": repository_identity,
        "joint_names": list(joint_names),
        "body_names": list(body_names),
    }
    source_metadata_path.write_text(
        json.dumps(source_metadata, indent=2, sort_keys=True) + "\n"
    )
    reference_manifest = {
        "format": "rmr_named_reference_v1_source_consistent_suffix_repair",
        "output_path": str(named_path),
        "output_sha256": sha256_file(named_path),
        "repair": repair,
        "exact_short_prefix": exact_prefix,
        "isaac_replay_state_errors": state_errors,
        "source_metadata_path": str(source_metadata_path),
        "source_metadata_sha256": sha256_file(source_metadata_path),
    }
    reference_manifest_path.write_text(
        json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n"
    )

    completion: dict[str, object] = {
        "protocol": PROTOCOL,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "output_frames": args.expected_output_frames,
        "input_source_frames": args.end_frame_inclusive - args.start_frame + 1,
        "csv_prefix_exact": csv_prefix_exact,
        "final_prefix_comparison": exact_prefix,
        "load_bearing_repair": repair,
        "isaac_replay_state_errors": state_errors,
        "repository_identity": repository_identity,
        "outputs": {},
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "no_dynamics_steps": True,
        "policy_evaluated": False,
        "optimizer_updates": 0,
    }
    output_paths = (
        csv_path,
        csv_path.with_suffix(".csv.manifest.json"),
        raw_path,
        joint_names_path,
        body_names_path,
        source_metadata_path,
        named_path,
        reference_manifest_path,
        output_dir / "isaac_converter_stdout.log",
        output_dir / "isaac_converter_stderr.log",
    )
    completion["outputs"] = {
        path.name: {"path": str(path), "sha256": sha256_file(path)}
        for path in output_paths
    }
    completion_path = output_dir / "completion.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--baseline-csv-path", type=Path, required=True)
    parser.add_argument("--baseline-csv-sha256", required=True)
    parser.add_argument("--baseline-prefix-path", type=Path, required=True)
    parser.add_argument("--baseline-prefix-sha256", required=True)
    parser.add_argument("--load-bearing-reference-path", type=Path, required=True)
    parser.add_argument("--load-bearing-reference-sha256", required=True)
    parser.add_argument("--controller-path", type=Path, required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame-inclusive", type=int, required=True)
    parser.add_argument("--prefix-frames", type=int, required=True)
    parser.add_argument("--expected-output-frames", type=int, required=True)
    parser.add_argument("--rmr-code-commit", required=True)
    parser.add_argument("--converter-path", type=Path, required=True)
    parser.add_argument("--converter-sha256", required=True)
    parser.add_argument("--exporter-sha256", required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-replay-state-error", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    completion = build_reference(build_parser().parse_args())
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
