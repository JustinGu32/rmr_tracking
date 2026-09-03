import hashlib
from pathlib import Path

import pytest
from train import EXPECTED_STABLE_TRAINER_SHA256, patch_stable_trainer_source

REPO_ROOT = Path(__file__).resolve().parents[2]
STABLE_TRAINER = REPO_ROOT / "scripts" / "rsl_rl" / "train.py"


def test_pinned_trainer_selects_only_continuity_runner():
    source = STABLE_TRAINER.read_text(encoding="utf-8")
    assert hashlib.sha256(STABLE_TRAINER.read_bytes()).hexdigest() == (
        EXPECTED_STABLE_TRAINER_SHA256
    )
    patched = patch_stable_trainer_source(source)
    assert patched != source
    assert patched.count("ResumeContinuityMotionOnPolicyRunner as OnPolicyRunner") == 1


def test_patch_refuses_missing_or_ambiguous_seam():
    with pytest.raises(RuntimeError, match="exactly once"):
        patch_stable_trainer_source("print('missing')")
