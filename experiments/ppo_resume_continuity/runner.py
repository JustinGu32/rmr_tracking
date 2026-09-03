"""Run E010's exact full-update probe with the E011 scheduler repair."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from experiments.ppo_first_update_probe.runner import (
    PROBE_DIRECTORY_ENV,
    FirstUpdateProbeMotionOnPolicyRunner,
    _write_json,
)

from .design import synchronize_resume_scheduler


class ResumeContinuityMotionOnPolicyRunner(FirstUpdateProbeMotionOnPolicyRunner):
    """Synchronize one missing scalar, then preserve E010's native update probe."""

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> Any:
        output_value = os.environ.get(PROBE_DIRECTORY_ENV)
        if not output_value:
            raise RuntimeError(f"{PROBE_DIRECTORY_ENV} is required")
        output_dir = Path(output_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        intervention = synchronize_resume_scheduler(self.alg)
        _write_json(output_dir / "scheduler_intervention.json", intervention)
        print(
            "[RESUME-SCHEDULER-CONTINUITY] "
            f"scheduler={intervention['scheduler_learning_rate_after']:.9g} "
            f"adam_entries={intervention['optimizer_state_entries_after']}",
            flush=True,
        )

        result = super().learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )

        result_path = output_dir / "probe_result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        trace = payload.get("optimizer_trace", [])
        first_rate = float(trace[0]["learning_rate"]) if trace else None
        expected_first_rate = float(
            intervention["expected_first_applied_learning_rate"]
        )
        if first_rate is None or first_rate != expected_first_rate:
            raise RuntimeError(
                f"first adaptive-KL rate drift: {first_rate} != {expected_first_rate}"
            )
        payload["resume_scheduler_intervention"] = intervention
        payload["continuity_runner_completed"] = True
        _write_json(result_path, payload)
        print(
            f"[RESUME-SCHEDULER-CONTINUITY] complete result={result_path}",
            flush=True,
        )
        return result
