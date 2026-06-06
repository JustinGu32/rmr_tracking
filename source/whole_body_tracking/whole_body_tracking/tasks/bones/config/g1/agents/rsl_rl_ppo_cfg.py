from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BonesPopArtPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    use_popart_multihead: bool = False
    popart_head_mode: str = "per_term"
    popart_groups: dict[str, list[str]] | None = None
    popart_group_preset: str = "actual_individual"
    popart_grouped_actor_weight_mode: str = "uniform"
    popart_momentum: float = 0.1
    popart_epsilon: float = 1.0e-5
    popart_normalize_actor_weights: bool = False
    popart_actor_advantage_scaling: str = "whitened"
    # Hierarchical (motion-category x reward-head) PopArt. When True, the runner
    # builds a critic with C x H normalized outputs and reads obs['category'].
    popart_hierarchical: bool = False
    popart_num_categories: int | None = None
    popart_min_samples: int = 2
    popart_category_obs_group: str = "category"


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        critic_hidden_dims=[2048, 2048, 1024, 1024, 512, 512],
        activation="elu",
    )
    algorithm = BonesPopArtPpoAlgorithmCfg(
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


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
