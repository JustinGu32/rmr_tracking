"""Run a fixed 2x2 actor/normalizer crossover through the strict source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contract import (
    LONG_LONG,
    LONG_SHORT,
    SHORT_LONG,
    SHORT_SHORT,
    classify_crossover,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict source-task 2x2 actor/normalizer crossover."
    )
    parser.add_argument("--short-checkpoint", required=True)
    parser.add_argument("--long-checkpoint", required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--evaluator",
        default=str(
            REPO_ROOT / "experiments" / "exact_long_source_eval" / "evaluate.py"
        ),
    )
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--ppo-output", default="delta-all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm-timeout-seconds", type=int, default=300)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.arm_timeout_seconds < 1:
        parser.error("--arm-timeout-seconds must be positive")
    if args.task != "Tracking-Flat-G1-v0":
        parser.error("the crossover requires the original Tracking-Flat-G1-v0 task")
    if args.ppo_output != "delta-all":
        parser.error("the crossover requires the delta-all action contract")
    return args


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _artifact_record(path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(output_dir)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _arm_command(
    args: argparse.Namespace,
    *,
    actor_checkpoint: Path,
    normalizer_checkpoint: Path,
    arm_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.evaluator),
        "--task",
        args.task,
        "--motion-file",
        str(args.motion_file),
        "--checkpoint-path",
        str(actor_checkpoint),
        "--normalizer-checkpoint-path",
        str(normalizer_checkpoint),
        "--output-dir",
        str(arm_dir / "source_eval"),
        "--episodes",
        str(args.episodes),
        "--eval-seed",
        str(args.eval_seed),
        "--outcome-label-set",
        "short-control",
        "--ppo-output",
        args.ppo_output,
        "--device",
        args.device,
    ]
    if args.render:
        command.append("--render")
    if args.headless:
        command.append("--headless")
    return command


def main() -> int:
    args = _parse_args()
    args.short_checkpoint = Path(args.short_checkpoint).expanduser().resolve()
    args.long_checkpoint = Path(args.long_checkpoint).expanduser().resolve()
    args.motion_file = Path(args.motion_file).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.evaluator = Path(args.evaluator).expanduser().resolve()

    for name in ("short_checkpoint", "long_checkpoint", "motion_file", "evaluator"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if args.short_checkpoint == args.long_checkpoint:
        raise ValueError("short and long checkpoints must be distinct")
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed result: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    arm_specs = [
        (SHORT_SHORT, args.short_checkpoint, args.short_checkpoint),
        (LONG_LONG, args.long_checkpoint, args.long_checkpoint),
        (SHORT_LONG, args.short_checkpoint, args.long_checkpoint),
        (LONG_SHORT, args.long_checkpoint, args.short_checkpoint),
    ]
    arm_records: list[dict[str, Any]] = []
    arm_outcomes: dict[str, str] = {}
    initial_qpos_hashes: list[str] = []
    initial_qvel_hashes: list[str] = []
    initial_observation_hashes: list[str] = []

    for arm_name, actor_checkpoint, normalizer_checkpoint in arm_specs:
        arm_dir = args.output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        command = _arm_command(
            args,
            actor_checkpoint=actor_checkpoint,
            normalizer_checkpoint=normalizer_checkpoint,
            arm_dir=arm_dir,
        )
        stdout_path = arm_dir / "launcher_stdout.log"
        stderr_path = arm_dir / "launcher_stderr.log"
        arm_started = _utc_now()
        monotonic_start = time.monotonic()
        timed_out = False
        with stdout_path.open("wb") as stdout_handle, stderr_path.open(
            "wb"
        ) as stderr_handle:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=args.arm_timeout_seconds,
                    check=False,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                return_code = 124
        duration_seconds = time.monotonic() - monotonic_start

        arm_result_path = arm_dir / "source_eval" / "result.json"
        arm_result: dict[str, Any] | None = None
        outcome = "invalid-execution"
        contract_errors: list[str] = []
        if return_code != 0:
            contract_errors.append(f"child return code {return_code}")
        if timed_out:
            contract_errors.append("child timed out")
        if arm_result_path.is_file():
            try:
                arm_result = json.loads(arm_result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                contract_errors.append(f"invalid arm result: {error}")
        else:
            contract_errors.append("arm result.json missing")

        if arm_result is not None:
            outcome = str(arm_result.get("outcome", "invalid-execution"))
            actor_record = arm_result.get("inputs", {}).get("checkpoint", {})
            normalizer_record = arm_result.get("inputs", {}).get(
                "normalizer_checkpoint", {}
            )
            if actor_record.get("sha256") != _sha256_file(actor_checkpoint):
                contract_errors.append("actor checkpoint identity mismatch")
            if normalizer_record.get("sha256") != _sha256_file(
                normalizer_checkpoint
            ):
                contract_errors.append("normalizer checkpoint identity mismatch")
            expected_override = actor_checkpoint != normalizer_checkpoint
            observed_override = arm_result.get("evaluation_contract", {}).get(
                "normalizer_override_applied"
            )
            if observed_override is not expected_override:
                contract_errors.append("normalizer override flag mismatch")
            episodes = arm_result.get("episodes", [])
            if len(episodes) != args.episodes:
                contract_errors.append("episode count mismatch")
            for episode in episodes:
                initial_qpos_hashes.append(str(episode.get("initial_qpos_sha256")))
                initial_qvel_hashes.append(str(episode.get("initial_qvel_sha256")))
                initial_observation_hashes.append(
                    str(episode.get("initial_policy_observation_sha256"))
                )
            if not bool(arm_result.get("all_numeric_finite")):
                contract_errors.append("nonfinite arm telemetry")
        if contract_errors:
            outcome = "invalid-execution"
        arm_outcomes[arm_name] = outcome
        arm_records.append(
            {
                "arm": arm_name,
                "actor_checkpoint": {
                    "path": str(actor_checkpoint),
                    "sha256": _sha256_file(actor_checkpoint),
                },
                "normalizer_checkpoint": {
                    "path": str(normalizer_checkpoint),
                    "sha256": _sha256_file(normalizer_checkpoint),
                },
                "command": command,
                "started_at": arm_started,
                "finished_at": _utc_now(),
                "duration_seconds": duration_seconds,
                "return_code": return_code,
                "timed_out": timed_out,
                "outcome": outcome,
                "contract_errors": contract_errors,
                "result_path": str(arm_result_path),
                "result_sha256": (
                    _sha256_file(arm_result_path) if arm_result_path.is_file() else None
                ),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
        )
        print(
            f"[CROSSOVER] arm={arm_name} outcome={outcome} "
            f"return_code={return_code} duration={duration_seconds:.3f}s",
            flush=True,
        )

    classification = classify_crossover(arm_outcomes)
    identical_initial_qpos = len(set(initial_qpos_hashes)) == 1
    identical_initial_qvel = len(set(initial_qvel_hashes)) == 1
    identical_initial_observation = len(set(initial_observation_hashes)) == 1
    if not (
        identical_initial_qpos
        and identical_initial_qvel
        and identical_initial_observation
    ):
        classification = {
            **classification,
            "outcome": "invalid-execution",
            "initial_identity_error": True,
        }

    arms_path = args.output_dir / "arms.json"
    _write_json(arms_path, {"arms": arm_records})
    artifact_paths = sorted(
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path != result_path and not path.name.endswith(".tmp")
    )
    result = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "outcome": classification["outcome"],
        "classification": classification,
        "inputs": {
            "short_checkpoint": {
                "path": str(args.short_checkpoint),
                "sha256": _sha256_file(args.short_checkpoint),
            },
            "long_checkpoint": {
                "path": str(args.long_checkpoint),
                "sha256": _sha256_file(args.long_checkpoint),
            },
            "motion": {
                "path": str(args.motion_file),
                "sha256": _sha256_file(args.motion_file),
            },
            "evaluator": {
                "path": str(args.evaluator),
                "sha256": _sha256_file(args.evaluator),
            },
            "code": {
                "repository": str(REPO_ROOT),
                "git_commit": _git_output("rev-parse", "HEAD"),
                "git_status_short": _git_output("status", "--short"),
                "run_py_sha256": _sha256_file(Path(__file__).resolve()),
                "contract_py_sha256": _sha256_file(
                    Path(__file__).resolve().parent / "contract.py"
                ),
            },
        },
        "evaluation_contract": {
            "task": args.task,
            "episodes_per_arm": args.episodes,
            "eval_seed_reapplied_per_episode": args.eval_seed,
            "ppo_output": args.ppo_output,
            "device": args.device,
            "render": args.render,
            "headless": args.headless,
            "arm_timeout_seconds": args.arm_timeout_seconds,
            "fixed_arm_order": [name for name, _, _ in arm_specs],
            "identical_initial_qpos_across_all_episodes_and_arms": (
                identical_initial_qpos
            ),
            "identical_initial_qvel_across_all_episodes_and_arms": (
                identical_initial_qvel
            ),
            "identical_initial_raw_observation_across_all_episodes_and_arms": (
                identical_initial_observation
            ),
        },
        "arms": arm_records,
        "artifacts": {
            str(path.relative_to(args.output_dir)): _artifact_record(
                path, args.output_dir
            )
            for path in artifact_paths
        },
        "claim_boundary": (
            "This fixed source-task crossover attributes short-prefix competence to "
            "actor weights, empirical actor-observation normalization, or their "
            "interaction. It performs no optimization and does not identify which "
            "training update or upstream objective produced either component."
        ),
    }
    _write_json(result_path, result)
    print(f"[CROSSOVER] outcome={result['outcome']}", flush=True)
    print(f"[CROSSOVER] result={result_path}", flush=True)
    return 1 if result["outcome"] == "invalid-execution" else 0


if __name__ == "__main__":
    raise SystemExit(main())
