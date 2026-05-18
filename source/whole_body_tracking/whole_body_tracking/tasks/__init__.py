"""Package containing task implementations for various robotic environments."""

try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError as exc:
    if exc.name not in {"omni", "isaacsim"}:
        raise
else:
    _BLACKLIST_PKGS = ["utils"]
    import_packages(__name__, _BLACKLIST_PKGS)
