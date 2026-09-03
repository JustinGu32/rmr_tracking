"""Execute the pinned trainer with only the factorial runner import changed."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STABLE_TRAINER = REPO_ROOT / "scripts" / "rsl_rl" / "train.py"
LOCAL_WHOLE_BODY_SOURCE = REPO_ROOT / "source" / "whole_body_tracking"
EXPECTED_STABLE_TRAINER_SHA256 = (
    "3f15fff5008a17c7ff29fc029dd096c046c82b5f2c6bc2000c92122b6f73e082"
)
ORIGINAL_IMPORT = (
    "from whole_body_tracking.utils.my_on_policy_runner import "
    "MotionOnPolicyRunner as OnPolicyRunner"
)
FACTORIAL_IMPORT = (
    "from experiments.ppo_resume_state_discriminator.runner import "
    "ResumeStateDiscriminatorMotionOnPolicyRunner as OnPolicyRunner"
)


def patch_stable_trainer_source(source: str) -> str:
    """Replace exactly the stable runner selection and reject source drift."""
    count = source.count(ORIGINAL_IMPORT)
    if count != 1:
        raise RuntimeError(
            f"expected stable runner import exactly once, observed {count}"
        )
    return source.replace(ORIGINAL_IMPORT, FACTORIAL_IMPORT)


def main() -> None:
    trainer_bytes = STABLE_TRAINER.read_bytes()
    observed_sha = hashlib.sha256(trainer_bytes).hexdigest()
    if observed_sha != EXPECTED_STABLE_TRAINER_SHA256:
        raise RuntimeError(
            f"stable trainer hash drift: expected {EXPECTED_STABLE_TRAINER_SHA256}, observed {observed_sha}"
        )
    source = patch_stable_trainer_source(trainer_bytes.decode("utf-8"))
    for path in (
        str(REPO_ROOT),
        str(LOCAL_WHOLE_BODY_SOURCE),
        str(STABLE_TRAINER.parent),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    whole_body_spec = importlib.util.find_spec("whole_body_tracking")
    expected_origin = (
        LOCAL_WHOLE_BODY_SOURCE / "whole_body_tracking" / "__init__.py"
    ).resolve()
    observed_origin = (
        Path(whole_body_spec.origin).resolve()
        if whole_body_spec is not None and whole_body_spec.origin is not None
        else None
    )
    if observed_origin != expected_origin:
        raise RuntimeError(
            f"whole_body_tracking import drift: expected {expected_origin}, observed {observed_origin}"
        )
    print(f"[RESUME-STATE-CODE] whole_body_tracking={observed_origin}", flush=True)
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(STABLE_TRAINER),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, str(STABLE_TRAINER), "exec"), globals_dict)  # noqa: S102


if __name__ == "__main__":
    main()
