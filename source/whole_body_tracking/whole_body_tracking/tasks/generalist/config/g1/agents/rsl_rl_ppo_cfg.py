from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1FlatPPORunnerCfgGeneralist(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner config for the generalist task.

    Uses the vanilla MotionOnPolicyRunner (no PopArt). Hyperparameters match
    the popart task's runner so A/B comparisons on the same dataset stay
    apples-to-apples.
    """

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_generalist"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
        # diffusion_collect (when present) stays in storage but isn't routed
        # to the actor or critic torsos.
    }
