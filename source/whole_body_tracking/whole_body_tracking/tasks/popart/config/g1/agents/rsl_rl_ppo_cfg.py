from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1FlatPPORunnerCfgPopArt(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner config for the PopArt task (Stage B).

    `class_name` is set to `PopArtMotionOnPolicyRunner` so the train script
    dispatches to our custom runner. The runner pops the popart-specific
    fields (`num_categories`, `popart_momentum`, `category_obs_group`) off
    the policy cfg dict before constructing `ActorCriticPopArt`.
    """

    class_name = "PopArtMotionOnPolicyRunner"

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_popart"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        # Clipped value loss is disabled in PopArt mode (see decision §1 in
        # markdowns/popart_implementation.md). Clipping in normalized space
        # would require snapshotting (μ_k, σ_k) at rollout time.
        use_clipped_value_loss=False,
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
        # `category` and `diffusion_collect` stay in obs/storage. `category` is
        # consumed by ActorCriticPopArt directly via obs[category_obs_group];
        # neither group is routed to the actor or critic torsos.
    }

    def __post_init__(self):
        # configclass may attach a parent __post_init__; call it if present.
        sup = super()
        if hasattr(sup, "__post_init__"):
            sup.__post_init__()
        # Attach PopArt-specific fields onto the policy cfg dict. These are
        # popped by PopArtMotionOnPolicyRunner._construct_algorithm before
        # the rest of policy_cfg is forwarded to ActorCriticPopArt.
        # `num_categories` is read from the command term at runtime (the env
        # is the source of truth — driven by `categories` on the env cfg). We
        # set a safe fallback here in case the command term doesn't expose it.
        self.policy.num_categories = 2
        self.policy.popart_momentum = 0.1
        self.policy.category_obs_group = "category"
