import gymnasium as gym

from . import agents
from .chair_env_cfg import G1ChairEnvCfg
from ...chair_step_env import ChairStepEnv

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

gym.register(
    id="Chair-Step-G1-v1",
    entry_point="whole_body_tracking.tasks.chair_step.chair_step_env:ChairStepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1ChairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ChairPPORunnerCfg",
    },
)