import gymnasium as gym

from . import agents
from .staircase_env_cfg import G1StaircaseEnvCfg, G1StaircasePlayEnvCfg
from ...staircase_env import StaircaseEnv

##
# Register Gym environments.
##

gym.register(
    id="Staircase-G1-v0",
    entry_point="whole_body_tracking.tasks.staircase.staircase_env:StaircaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StaircaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StaircasePPORunnerCfg",
    },
)

gym.register(
    id="Staircase-G1-v1",
    entry_point="whole_body_tracking.tasks.staircase.staircase_env:StaircaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StaircaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StaircasePPORunnerCfg",
    },
)

gym.register(
    id="Staircase-G1-Play-v0",
    entry_point="whole_body_tracking.tasks.staircase.staircase_env:StaircaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StaircasePlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StaircasePPORunnerCfg",
    },
)

gym.register(
    id="Staircase-G1-Baseline-v0",
    entry_point="whole_body_tracking.tasks.staircase.staircase_env:StaircaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1StaircasePlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StaircasePPORunnerCfg",
    },
)