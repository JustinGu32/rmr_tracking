"""Python module serving as a project/extension template."""

# Register Gym environments when Isaac Sim dependencies are available.
try:
    from .tasks import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name not in {"omni", "isaacsim"}:
        raise
