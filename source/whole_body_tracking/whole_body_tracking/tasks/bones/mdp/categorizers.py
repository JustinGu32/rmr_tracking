"""Clip-name → motion-category mapping for hierarchical (category × reward-head) PopArt.

The hierarchical PopArt critic keeps separate running ``(mu, sigma)`` per
``(motion_category, reward_head)`` pair. This module decides, for a given clip
name, which motion category it belongs to.

Design notes specific to the BONES ``locomotion_*.zarr`` stores:

- Categories are derived from substrings of the clip name (there is no explicit
  category field in the zarr; ``content_props``/``content_props_desc`` describe
  *environment objects* such as ``wall``/``chair``, not the motion type).
- Unlike the original ``priority_substrings`` categorizer (which *raised* on any
  unmatched clip), every categorizer here is **total**: a clip that matches no
  keyword lands in an explicit ``other`` bucket. This was measured against
  ``/move/data/bones/g1/zarr/locomotion_50hz.zarr`` (39,466 clips); the fine
  keyword table below already yields 0 unmatched clips, but the ``other`` bucket
  guards against drift if a different zarr is used.

The fine keyword table is the single source of truth. Coarser presets
(``balanced``, ``mid``) are defined purely as a fine→coarse remap, so every
preset inherits the fine table's 100% coverage and consistent assignment.

Measured fine-category distribution on ``locomotion_50hz.zarr``::

    walk 34.0%  jog 20.3%  idle 14.1%  sit 12.6%  crouch 6.3%  turn 4.5%
    kneel 3.3%  jump 2.4%  run 1.2%   crawl 0.9%  misc 0.3%   stand_up 0.06%

Add a categorizer by extending ``PRESET_REMAP``/``PRESET_NAMES`` (or the fine
table) and referencing it by name in the command cfg field ``categorizer``.
"""

from __future__ import annotations

import torch

# ── Fine keyword table (single source of truth) ──────────────────────────────
# Ordered list of (substring keyword, fine category). First match wins, so put
# more-specific keywords before more-general ones (e.g. "stand_up" before any
# "stand", "on_the_edge"/"wall" idle variants before "walk").
FINE_RULES: list[tuple[str, str]] = [
    ("stand_up", "stand_up"),
    ("sit", "sit"),                 # sit, sitting, sit_on_heels, sit_cross_legged
    ("kneel", "kneel"),
    ("crouch", "crouch"),
    ("on_all_fours", "crawl"),
    ("idle", "idle"),
    ("wall", "idle"),               # wall_leaning_idle...
    ("on_the_edge", "idle"),
    ("crossed", "idle"),
    ("stance", "idle"),
    ("walk", "walk"),
    ("jog", "jog"),
    ("run", "run"),
    ("turn", "turn"),
    ("change", "turn"),             # change_direction
    ("jump", "jump"),
    ("vault", "jump"),
    ("step", "step"),
    ("mohak", "walk"),              # mohak_forward/backward — locomotion
    ("h_b_w", "walk"),
    ("panic", "misc"),
    ("bump", "misc"),
    ("touch", "misc"),
    ("avoid", "misc"),
    ("safety", "misc"),
]

# Fixed fine-category order. Indices are stable regardless of which categories
# actually appear in a given zarr (empty categories are harmless — PopArt skips
# categories with no samples in a rollout). ``other`` is the total-coverage
# fallback and must be last.
FINE_NAMES: list[str] = [
    "walk", "jog", "run", "turn", "jump",
    "idle", "sit", "crouch", "kneel", "crawl",
    "step", "stand_up", "misc", "other",
]

# ── Coarse presets: fine category → coarse category ──────────────────────────
PRESET_NAMES: dict[str, list[str]] = {
    "fine": FINE_NAMES,
    "balanced": ["locomotion", "turn", "jump", "static", "step", "misc", "other"],
    "mid": ["walk", "jog", "run", "turn", "jump", "idle", "sit", "crouch_kneel", "other"],
}

PRESET_REMAP: dict[str, dict[str, str]] = {
    "fine": {name: name for name in FINE_NAMES},
    "balanced": {
        "walk": "locomotion", "jog": "locomotion", "run": "locomotion",
        "turn": "turn",
        "jump": "jump",
        "idle": "static", "sit": "static", "crouch": "static",
        "kneel": "static", "crawl": "static", "stand_up": "static",
        "step": "step",
        "misc": "misc",
        "other": "other",
    },
    "mid": {
        "walk": "walk", "jog": "jog", "run": "run", "turn": "turn", "jump": "jump",
        "idle": "idle", "sit": "sit",
        "crouch": "crouch_kneel", "kneel": "crouch_kneel", "crawl": "crouch_kneel",
        "step": "other", "misc": "other", "stand_up": "other",
    },
}

DEFAULT_CATEGORIZER = "balanced"


def validate_categorizer(name: str) -> None:
    if name not in PRESET_NAMES:
        raise ValueError(
            f"Unknown categorizer {name!r}. Expected one of {sorted(PRESET_NAMES)}."
        )


def category_names(categorizer: str = DEFAULT_CATEGORIZER) -> list[str]:
    """Ordered category labels for a categorizer; index == category id."""
    validate_categorizer(categorizer)
    return list(PRESET_NAMES[categorizer])


def num_categories(categorizer: str = DEFAULT_CATEGORIZER) -> int:
    return len(category_names(categorizer))


def _fine_label(clip_name: str) -> str:
    lowered = clip_name.lower()
    for keyword, fine in FINE_RULES:
        if keyword in lowered:
            return fine
    return "other"


def categorize_clip_name(clip_name: str, categorizer: str = DEFAULT_CATEGORIZER) -> int:
    """Return the category index for a single clip name. Never raises."""
    validate_categorizer(categorizer)
    coarse = PRESET_REMAP[categorizer][_fine_label(clip_name)]
    return PRESET_NAMES[categorizer].index(coarse)


def categorize_clip_names(
    clip_names: list[str],
    categorizer: str = DEFAULT_CATEGORIZER,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """Map a list of clip names to a per-clip category-index tensor.

    Returns ``(clip_category[num_clips] long tensor, category_names list)``.
    """
    validate_categorizer(categorizer)
    names = category_names(categorizer)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    indices = [name_to_idx[PRESET_REMAP[categorizer][_fine_label(c)]] for c in clip_names]
    return torch.tensor(indices, dtype=torch.long, device=device), names
