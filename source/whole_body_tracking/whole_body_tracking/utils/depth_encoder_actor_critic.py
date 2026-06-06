from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization


# ── shared init helper ────────────────────────────────────────────────────────

def _init_actor_critic_common(
    policy,  # the ActorCritic subclass instance
    obs,
    obs_groups,
    num_actions: int,
    num_actor_obs_raw: int,
    num_critic_obs: int,
    encoded_actor_dim: int,
    depth_encoder: nn.Module,
    depth_start_idx: int,
    depth_dim: int,
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
    activation: str,
    actor_obs_normalization: bool,
    critic_obs_normalization: bool,
    init_noise_std: float,
    noise_std_type: str,
) -> None:
    """Populate the fields that DepthEncoderActorCritic and DepthCNNActorCritic share."""
    nn.Module.__init__(policy)
    Normal.set_default_validate_args(False)

    policy.obs_groups = obs_groups

    actor_mlp = MLP(encoded_actor_dim, num_actions, actor_hidden_dims, activation)
    policy.actor = _DepthEncodedActor(
        depth_encoder=depth_encoder,
        actor_mlp=actor_mlp,
        depth_start_idx=depth_start_idx,
        depth_dim=depth_dim,
        num_raw_obs=num_actor_obs_raw,
    )
    policy.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

    policy.actor_obs_normalization = actor_obs_normalization
    policy.actor_obs_normalizer = (
        EmpiricalNormalization(num_actor_obs_raw) if actor_obs_normalization else nn.Identity()
    )
    policy.critic_obs_normalization = critic_obs_normalization
    policy.critic_obs_normalizer = (
        EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
    )

    policy.noise_std_type = noise_std_type
    if noise_std_type == "scalar":
        policy.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif noise_std_type == "log":
        policy.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
        raise ValueError(f"Unknown noise_std_type: {noise_std_type!r}. Expected 'scalar' or 'log'.")

    policy.distribution = None


class _DepthEncodedActor(nn.Module):
    """Depth encoder + actor MLP fused into one module.

    Accepts the raw flat policy obs vector (depth embedded at depth_start_idx),
    splits out the depth slice, encodes it, and runs the actor MLP on the
    concatenated (non-depth | depth_latent) vector.

    The __getitem__ / in_features shim makes this compatible with the
    _OnnxPolicyExporter / _OnnxMotionPolicyExporter which do:
        obs = torch.zeros(1, self.actor[0].in_features)
        actions = self.actor(self.normalizer(obs))
    """

    def __init__(
        self,
        depth_encoder: nn.Module,
        actor_mlp: nn.Module,
        depth_start_idx: int,
        depth_dim: int,
        num_raw_obs: int,
    ):
        super().__init__()
        self.depth_encoder = depth_encoder
        self.actor_mlp = actor_mlp
        self.depth_start_idx = depth_start_idx
        self.depth_dim = depth_dim
        # ONNX exporter reads self.actor[0].in_features to size the dummy input
        self.in_features = num_raw_obs

    def __getitem__(self, idx: int) -> "_DepthEncodedActor":
        if idx == 0:
            return self
        raise IndexError(idx)

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        start = self.depth_start_idx
        end = start + self.depth_dim
        before = raw_obs[:, :start]
        depth = raw_obs[:, start:end]
        after = raw_obs[:, end:]
        depth_latent = self.depth_encoder(depth)
        encoded = torch.cat([before, after, depth_latent], dim=-1)
        return self.actor_mlp(encoded)


class DepthEncoderActorCritic(ActorCritic):
    """ActorCritic with a small MLP encoder for the flattened depth observation.

    The flat policy obs vector is split at runtime into:
      - non-depth: everything before and after the depth slice
      - depth: obs[:, depth_start_idx : depth_start_idx + depth_dim]

    The depth slice is compressed to depth_latent_dim via a small MLP, then
    concatenated with the non-depth obs before the actor MLP.  The critic
    receives the privileged obs (no depth term) unchanged.

    The combined depth-encoder + actor-MLP is exposed as self.actor so that
    the existing ONNX exporters work without modification.

    Switch on/off via --depth_encoder in train_stairs.py; no env-cfg changes
    needed (depth stays flat at 768 dims in the obs vector).
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        depth_start_idx: int,
        depth_dim: int = 768,
        depth_latent_dim: int = 64,
        depth_encoder_hidden_dims: list[int] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if depth_encoder_hidden_dims is None:
            depth_encoder_hidden_dims = [256, 128]

        if kwargs:
            print("DepthEncoderActorCritic: ignoring unexpected kwargs: " + str(list(kwargs.keys())))

        num_actor_obs_raw = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        encoded_actor_dim = num_actor_obs_raw - depth_dim + depth_latent_dim

        depth_encoder = MLP(
            input_dim=depth_dim,
            output_dim=depth_latent_dim,
            hidden_dims=depth_encoder_hidden_dims,
            activation=activation,
            last_activation=activation,
        )

        _init_actor_critic_common(
            self, obs, obs_groups, num_actions,
            num_actor_obs_raw, num_critic_obs, encoded_actor_dim,
            depth_encoder, depth_start_idx, depth_dim,
            actor_hidden_dims, critic_hidden_dims, activation,
            actor_obs_normalization, critic_obs_normalization,
            init_noise_std, noise_std_type,
        )

        print(
            f"[DepthEncoderActorCritic] "
            f"depth_start_idx={depth_start_idx}  depth_dim={depth_dim}  depth_latent_dim={depth_latent_dim}"
        )
        print(
            f"[DepthEncoderActorCritic] "
            f"raw_actor_obs={num_actor_obs_raw}  encoded_actor_obs={encoded_actor_dim}  critic_obs={num_critic_obs}"
        )
        print(f"  depth_encoder : {depth_encoder}")
        print(f"  actor_mlp     : {self.actor.actor_mlp}")
        print(f"  critic        : {self.critic}")


# ── CNN depth encoder ─────────────────────────────────────────────────────────

class _CNNDepthEncoder(nn.Module):
    """3-layer stride-2 CNN that accepts a flat depth vector and returns a latent.

    Reshapes (N, depth_dim) → (N, C, H, W) internally, applies:
        Conv2d(C→32, k=3, s=2, p=1) + SiLU
        Conv2d(32→64, k=3, s=2, p=1) + SiLU
        Conv2d(64→128, k=3, s=2, p=1) + SiLU
        Flatten
        Linear(flatten_dim → latent_dim) + SiLU

    For 1×24×32 input the spatial dims after three stride-2 convolutions are
    3×4, giving flatten_dim = 128 × 3 × 4 = 1536.
    """

    def __init__(
        self,
        depth_channels: int,
        depth_height: int,
        depth_width: int,
        latent_dim: int,
    ):
        super().__init__()
        self.depth_channels = depth_channels
        self.depth_height = depth_height
        self.depth_width = depth_width

        self.conv = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        # compute flatten dim via a dummy forward pass — handles any H×W cleanly
        with torch.no_grad():
            dummy = torch.zeros(1, depth_channels, depth_height, depth_width)
            flatten_dim = self.conv(dummy).flatten(1).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, latent_dim),
            nn.SiLU(),
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        depth_img = depth_flat.view(-1, self.depth_channels, self.depth_height, self.depth_width)
        features = self.conv(depth_img).flatten(1)
        return self.fc(features)


class DepthCNNActorCritic(ActorCritic):
    """ActorCritic with a 3-layer stride-2 CNN encoder for the flattened depth observation.

    Identical to DepthEncoderActorCritic except the depth encoder is a CNN
    instead of an MLP.  The flat 768-dim depth slice is reshaped to (N,1,24,32)
    inside _CNNDepthEncoder before the convolutions are applied.

    Switch on/off via --depth_cnn in train_stairs.py; no env-cfg changes needed.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        depth_start_idx: int,
        depth_dim: int = 768,
        depth_height: int = 24,
        depth_width: int = 32,
        depth_channels: int = 1,
        depth_latent_dim: int = 64,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print("DepthCNNActorCritic: ignoring unexpected kwargs: " + str(list(kwargs.keys())))

        num_actor_obs_raw = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        encoded_actor_dim = num_actor_obs_raw - depth_dim + depth_latent_dim

        depth_encoder = _CNNDepthEncoder(
            depth_channels=depth_channels,
            depth_height=depth_height,
            depth_width=depth_width,
            latent_dim=depth_latent_dim,
        )

        _init_actor_critic_common(
            self, obs, obs_groups, num_actions,
            num_actor_obs_raw, num_critic_obs, encoded_actor_dim,
            depth_encoder, depth_start_idx, depth_dim,
            actor_hidden_dims, critic_hidden_dims, activation,
            actor_obs_normalization, critic_obs_normalization,
            init_noise_std, noise_std_type,
        )

        print(
            f"[DepthCNNActorCritic] "
            f"depth_start_idx={depth_start_idx}  input=({depth_channels},{depth_height},{depth_width})  "
            f"depth_latent_dim={depth_latent_dim}"
        )
        print(
            f"[DepthCNNActorCritic] "
            f"raw_actor_obs={num_actor_obs_raw}  encoded_actor_obs={encoded_actor_dim}  critic_obs={num_critic_obs}"
        )
        print(f"  depth_cnn : {depth_encoder}")
        print(f"  actor_mlp : {self.actor.actor_mlp}")
        print(f"  critic    : {self.critic}")
