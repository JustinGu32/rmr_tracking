"""Jump-specific MDP terms: foot-clearance reward shaping and terminations.

These terms target the failure mode where a tracking policy "bobs" in place
instead of leaving the ground during a reference jump, because grounded
tracking is a lower-risk local optimum. Everything here is opt-in (wired behind
WBT_JUMP_* env vars in popart_env_cfg.py) and OFF by default.

Design notes
------------
* "Flight" is detected physically, not by clip name: a frame is in flight when
  BOTH reference feet are above that clip's stance baseline by `flight_margin`.
  This self-gates — walk/stand always keep >=1 foot down, so the flight-keyed
  terms read ~0 there regardless of how the clip was categorized. The per-clip
  stance baseline + flight tables are precomputed on the command
  (`clip_foot_baseline`, see MultiClipMotionCommandPopArt._build_flight_tables).
* Contact is read from the `contact_forces` sensor restricted to the ankle
  bodies via the term's `sensor_cfg`.
* Terms that are NOT self-gating (below-z, foot-z) default to the "jump"
  category via `category_names` so they don't tax walk/stand.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.popart.mdp.commands import ANKLE_NAMES, MotionCommand
from whole_body_tracking.tasks.popart.mdp.rewards import _get_body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ANKLE_NAMES (the two G1 foot bodies, both must be tracked) is defined in
# commands.py and re-exported here for the foot-clearance terms.


# ── shared helpers ────────────────────────────────────────────────────────


def _env_origin_z(command: MotionCommand) -> torch.Tensor:
    """Per-env ground height (z of the env origin). Shape (N, 1)."""
    return command._env.scene.env_origins[:, 2:3]


def _ref_foot_height(command: MotionCommand, foot_idx) -> torch.Tensor:
    """Reference foot height above ground for each foot. Shape (N, n_feet)."""
    return command.body_pos_w[:, foot_idx, 2] - _env_origin_z(command)


def ref_flight_mask(command: MotionCommand, foot_idx) -> torch.Tensor:
    """True where the reference has both feet airborne (a real flight phase).

    Uses the per-clip stance baseline so it is robust to terrain offset and to
    differing nominal foot heights across clips. Shape (N,)."""
    heights = _ref_foot_height(command, foot_idx)  # (N, n_feet)
    baseline = command.clip_foot_baseline[command.clip_ids].unsqueeze(-1)  # (N, 1)
    return (heights > baseline + command.flight_margin).all(dim=-1)


def feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float) -> torch.Tensor:
    """Per-foot ground contact from the contact sensor. Shape (N, n_feet)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, n_feet, 3)
    return forces.norm(dim=-1) > force_threshold


def _category_mask(command: MotionCommand, category_names) -> torch.Tensor | None:
    """Per-env bool selecting envs whose category is in `category_names`.

    Returns None (no gating) when `category_names` is falsy, or when the command
    has no category info (non-popart). If specific categories are requested but
    NONE exist in this run, returns an all-False mask (disables the term) rather
    than applying it everywhere — the safe choice for non-self-gating penalties."""
    if not category_names:
        return None
    names = getattr(command, "category_names", None)
    if not names:
        return None
    ids = [names.index(n) for n in category_names if n in names]
    if not ids:
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    return torch.isin(command.category_idx, torch.tensor(ids, device=command.device))


def _apply_category_mask(value: torch.Tensor, command: MotionCommand, category_names) -> torch.Tensor:
    mask = _category_mask(command, category_names)
    return value if mask is None else value * mask.float()


# ── reward terms ──────────────────────────────────────────────────────────


def airborne_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
    category_names: list[str] | None = None,
) -> torch.Tensor:
    """(R1) Penalize ground contact while the reference is in flight.

    On flat ground, zero contact is only achievable by genuinely being
    airborne, so this cannot be hacked by tucking the legs. Self-gating via
    `ref_flight_mask` (≈0 for walk/stand). Returns the mean over feet of the
    contact indicator, negated, masked to flight frames. Range [-1, 0]."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    flight = ref_flight_mask(command, foot_idx).float()
    contact = feet_in_contact(env, sensor_cfg, force_threshold).float().mean(dim=-1)
    return _apply_category_mask(-contact * flight, command, category_names)


def airborne_flight_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
    category_names: list[str] | None = None,
) -> torch.Tensor:
    """(R2) Reward both feet being off the ground during reference flight.

    Dense positive signal toward takeoff that pairs with R1. Range [0, 1]."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    flight = ref_flight_mask(command, foot_idx).float()
    airborne = (~feet_in_contact(env, sensor_cfg, force_threshold)).all(dim=-1).float()
    return _apply_category_mask(airborne * flight, command, category_names)


def below_reference_anchor_z_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    category_names: list[str] | None = None,
) -> torch.Tensor:
    """(R3) Asymmetric penalty for the pelvis being BELOW the reference height.

    Penalizes `relu(ref_z - robot_z)` only — being too low (the bobbing
    failure) hurts, being above does not. NOT self-gating, so default to the
    'jump' category at the call site to avoid taxing walk/stand. Range (-inf, 0]
    but bounded in practice by the z-termination."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    deficit = torch.relu(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    return _apply_category_mask(-deficit, command, category_names)


def foot_below_threshold_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    gap: float = 0.10,
    category_names: list[str] | None = None,
) -> torch.Tensor:
    """(R4) Penalize feet lagging far below where the reference foot should be,
    while the reference foot is airborne. Per-foot, averaged. NOT self-gating —
    default to 'jump' at the call site. Range [-1, 0]."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    ref_world_z = command.body_pos_w[:, foot_idx, 2]
    rob_world_z = command.robot_body_pos_w[:, foot_idx, 2]
    ref_height = ref_world_z - _env_origin_z(command)
    baseline = command.clip_foot_baseline[command.clip_ids].unsqueeze(-1)
    ref_up = ref_height > baseline + command.flight_margin
    lagging = (ref_world_z - rob_world_z) > gap  # env-origin cancels in the diff
    penalty = -(ref_up & lagging).float().mean(dim=-1)
    return _apply_category_mask(penalty, command, category_names)


def contact_phase_match_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
    height_eps: float = 0.05,
    speed_eps: float = 0.5,
    category_names: list[str] | None = None,
) -> torch.Tensor:
    """(R5) Reward matching the reference foot contact pattern (GLOBAL).

    Reference stance per foot = foot near its stance baseline AND slow. Penalize
    the mean per-foot mismatch between actual contact and reference stance. This
    generalizes beyond jumps to also discourage double-stepping and foot-drag,
    so it is left ungated by default. Range [-1, 0]."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    ref_height = command.body_pos_w[:, foot_idx, 2] - _env_origin_z(command)
    baseline = command.clip_foot_baseline[command.clip_ids].unsqueeze(-1)
    ref_speed = command.body_lin_vel_w[:, foot_idx].norm(dim=-1)
    ref_stance = (ref_height < baseline + height_eps) & (ref_speed < speed_eps)
    contact = feet_in_contact(env, sensor_cfg, force_threshold)
    mismatch = (contact != ref_stance).float().mean(dim=-1)
    return _apply_category_mask(-mismatch, command, category_names)


# ── termination terms ─────────────────────────────────────────────────────


def grounded_during_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """(T1) Terminate on any ground contact while the reference is in flight.

    Sharp signal that makes the bobbing local optimum fatal. Pair with reward
    shaping / RSI — alone it can be hard to bootstrap (dies at takeoff)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    flight = ref_flight_mask(command, foot_idx)
    contact = feet_in_contact(env, sensor_cfg, force_threshold).any(dim=-1)
    return flight & contact


def grounded_during_flight_grace(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    grace_s: float = 0.2,
) -> torch.Tensor:
    """(T2) Gentler T1: terminate only on SUSTAINED contact during flight.

    Uses the contact sensor's `current_contact_time` (the scene enables
    `track_air_time`), so honest near-miss takeoff attempts are not killed."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    flight = ref_flight_mask(command, foot_idx)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sustained = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > grace_s).any(dim=-1)
    return flight & sustained


def bad_anchor_pos_z_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.12,
) -> torch.Tensor:
    """(T3) Terminate when the pelvis is far BELOW the reference during flight.

    A phase-conditional, one-sided tightening of `bad_anchor_pos_z_only`: fires
    only when the reference is airborne and the robot is >`threshold` below it.
    Converts bobbing into a termination so the EXISTING failure-based adaptive
    sampler upsamples these clips, without affecting walk/stand."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_idx = _get_body_indexes(command, ANKLE_NAMES)
    flight = ref_flight_mask(command, foot_idx)
    deficit = command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]
    return (deficit > threshold) & flight
