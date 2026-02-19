import gymnasium as gym

from . import agents, chair_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Chair-Step-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chair_env_cfg.G1ChairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ChairPPORunnerCfg",
    },
)

# gym.register(
#     id="Chair-Step-G1-v1",
#     entry_point="whole_body_tracking.tasks.chair_step.chair_step_env:ChairStepEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": chair_env_cfg.G1ChairEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ChairPPORunnerCfg",
#     },
# )

gym.register(
    id="Chair-Step-G1-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chair_env_cfg.G1ChairCollectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ChairPPORunnerCfg",
    },
)
