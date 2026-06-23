"""MDP terms for the generalist motion-tracking task.

Mirrors popart/mdp but drops jumps.py (the jump-specific reward shaping) and
the category_idx observation function. The bad_anchor_pos_z_flight termination
that lived in jumps.py is inlined into terminations.py so it remains available
behind the --jump_tighten_anchor_z flag.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
