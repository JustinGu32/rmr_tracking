"""Named clip-name → category-index functions used by `MultiClipMotionCommandPopArt`.

Use a `@register("name")` decorator to add a new categorizer; reference it by
name in the cfg field `categorizer`. Functions take a single clip name (str)
and return an int category index. Raise `ValueError` for clip names the
function does not handle — the command will then either raise or default,
depending on the cfg's `unmatched` field.
"""

from __future__ import annotations

from typing import Callable

CATEGORIZERS: dict[str, Callable[[str], int]] = {}

# Maps categorizer name → human-readable category labels, indexed by category int.
# Used for wandb log keys (`popart/sigma/walk` instead of `popart/sigma/cat_0`).
# Categorizers with no entry here fall back to `[f"cat_{i}" for i in range(K)]`.
CATEGORY_NAMES: dict[str, list[str]] = {
    "walk_vs_jog": ["walk", "jog"],
    "walk_vs_standup": ["walk", "stand_up"],
    "priority_substrings": ["stand_up", "walk", "jog", "run", "turn", "jump"],
}


def names_for(categorizer: str, num_categories: int) -> list[str]:
    """Resolve human-readable names for a categorizer, with fallback."""
    names = CATEGORY_NAMES.get(categorizer)
    if names is not None and len(names) >= num_categories:
        return list(names[:num_categories])
    return [f"cat_{i}" for i in range(num_categories)]


def make_priority_categorizer(names: list[str]) -> Callable[[str], int]:
    """Build a priority-substring categorizer from a list of category names.

    Returns a function that, given a clip name, returns the index of the
    first category in `names` whose lowercased keyword appears as a substring
    of the lowercased clip name. Raises ValueError on no match.

    Order matters: the first match wins. Put more-specific names first so
    e.g. `["stand_up", "walk"]` correctly buckets `stand_up_to_walk` as
    stand_up rather than walk.
    """
    if not names:
        raise ValueError("categories list cannot be empty")
    lowered = [n.lower() for n in names]

    def _fn(clip_name: str) -> int:
        n = clip_name.lower()
        for i, kw in enumerate(lowered):
            if kw in n:
                return i
        raise ValueError(f"unmatched clip: {clip_name!r}")

    return _fn


def register(name: str):
    def deco(fn: Callable[[str], int]) -> Callable[[str], int]:
        if name in CATEGORIZERS:
            raise ValueError(f"Categorizer {name!r} already registered.")
        CATEGORIZERS[name] = fn
        return fn

    return deco


@register("walk_vs_standup")
def _walk_vs_standup(clip_name: str) -> int:
    """Two categories: 0 = walk, 1 = stand_up. Substring match."""
    n = clip_name.lower()
    # Most-specific match first: stand_up before walk to avoid clips like
    # "stand_up_to_walk" landing in walk if order were reversed.
    if "stand_up" in n:
        return 1
    if "walk" in n:
        return 0
    raise ValueError(f"unmatched clip: {clip_name!r}")


@register("walk_vs_jog")
def _walk_vs_jog(clip_name: str) -> int:
    """Two categories: 0 = walk, 1 = jog. Substring match.

    `jog` is matched before `walk` so combo names like `walk_to_jog` land in
    jog. Adjust the order if you want the opposite convention.
    """
    n = clip_name.lower()
    if "jog" in n:
        return 1
    if "walk" in n:
        return 0
    raise ValueError(f"unmatched clip: {clip_name!r}")


@register("priority_substrings")
def _priority_substrings(clip_name: str) -> int:
    """Catch-most categorizer for the full BONES dataset.

    Edit the `rules` list to extend. The first matching keyword wins, so put
    more-specific rules first (e.g. "stand_up" before "stand").
    """
    n = clip_name.lower()
    rules: list[tuple[str, int]] = [
        ("stand_up", 0),
        ("walk", 1),
        ("jog", 2),
        ("run", 3),
        ("turn", 4),
        ("jump", 5),
    ]
    for kw, idx in rules:
        if kw in n:
            return idx
    raise ValueError(f"unmatched clip: {clip_name!r}")
