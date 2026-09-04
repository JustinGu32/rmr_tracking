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
    prepare_root = Path(args.prepare_code_root).resolve()
    source = verify_file(args.source_path, args.source_sha256, "source motion")
    baseline_csv = verify_file(
        args.baseline_csv_path, args.baseline_csv_sha256, "baseline CSV"
    )
    baseline_prefix = verify_file(
        args.baseline_prefix_path,
        args.baseline_prefix_sha256,
        "baseline prefix reference",
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
    prepare_tool = verify_file(
        args.prepare_tool_path, args.prepare_tool_sha256, "reference preparer"
    )
    repository_identity = {
        "rmr": _git_identity(rmr_root, args.rmr_code_commit),
        "prepare": _git_identity(prepare_root, args.prepare_code_commit),
    }

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "lafan_walk_win137_300.csv"
    raw_path = output_dir / "lafan_walk_win137_300_isaac_raw.npz"
    named_path = output_dir / "lafan_walk_win137_300_source_consistent_named.npz"
    source_metadata_path = output_dir / "source_metadata.json"

    csv_manifest = export_window(
        source,
        csv_path,
        source_sha256=args.source_sha256,
        start_frame=args.start_frame,
        end_frame_inclusive=args.end_frame_inclusive,
    )
    baseline_lines = baseline_csv.read_bytes().splitlines(keepends=True)
    candidate_lines = csv_path.read_bytes().splitlines(keepends=True)
    csv_prefix_exact = b"".join(candidate_lines[: len(baseline_lines)]) == baseline_csv.read_bytes()
    if not csv_prefix_exact:
        raise ValueError("extended source CSV does not preserve the exact baseline CSV prefix")

    converter_argv = [
        str(Path(args.isaac_python).resolve()),
        str(converter),
        "--input-file",
        str(csv_path),
        "--input-sha256",
        str(csv_manifest["output_sha256"]),
        "--input-fps",
        "30",
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
    exact_prefix = compare_exact_prefix(
        raw_path, baseline_prefix, prefix_frames=args.prefix_frames
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
        "exporter": {"path": str(exporter), "sha256": args.exporter_sha256},
        "isaac_converter": {
            "path": str(converter),
            "sha256": args.converter_sha256,
            "python": str(Path(args.isaac_python).resolve()),
            "device": args.device,
        },
        "reference_preparer": {
            "path": str(prepare_tool),
            "sha256": args.prepare_tool_sha256,
            "python": str(Path(args.prepare_python).resolve()),
        },
        "repository_identity": repository_identity,
        "joint_names": list(joint_names),
        "body_names": list(body_names),
    }
    source_metadata_path.write_text(
        json.dumps(source_metadata, indent=2, sort_keys=True) + "\n"
    )

    prepare_argv = [
        str(Path(args.prepare_python).resolve()),
        "-m",
        "tools.prepare_g1_rmr_reference",
        "--input-path",
        str(raw_path),
        "--output-path",
        str(named_path),
        "--controller-path",
        str(controller),
        "--source-metadata-json",
        str(source_metadata_path),
    ]
    prepare_environment = dict(os.environ)
    prepare_environment["PYTHONPATH"] = "."
    _run_and_capture(
        prepare_argv,
        cwd=prepare_root,
        environment=prepare_environment,
        stdout=output_dir / "prepare_stdout.log",
        stderr=output_dir / "prepare_stderr.log",
    )

    completion: dict[str, object] = {
        "protocol": PROTOCOL,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "output_frames": args.expected_output_frames,
        "input_source_frames": args.end_frame_inclusive - args.start_frame + 1,
        "csv_prefix_exact": csv_prefix_exact,
        "raw_prefix_comparison": exact_prefix,
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
        named_path.with_suffix(".npz.manifest.json"),
        output_dir / "isaac_converter_stdout.log",
        output_dir / "isaac_converter_stderr.log",
        output_dir / "prepare_stdout.log",
        output_dir / "prepare_stderr.log",
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
    parser.add_argument("--prepare-code-root", type=Path, required=True)
    parser.add_argument("--prepare-code-commit", required=True)
    parser.add_argument("--prepare-tool-path", type=Path, required=True)
    parser.add_argument("--prepare-tool-sha256", required=True)
    parser.add_argument("--prepare-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    completion = build_reference(build_parser().parse_args())
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
