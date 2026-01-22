import gymnasium as gym

from . import agents
from .chair_env_cfg import G1ChairEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Chair-Step-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1ChairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ChairPPORunnerCfg",
    },
)