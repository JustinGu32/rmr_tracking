"""Symmetric (left/right mirror) data augmentation for the generalist task.

Adds an env wrapper that applies y-axis mirror to a fraction of envs at
training time. The policy effectively sees a doubled dataset without storing
mirrored motions on disk — useful when training on
`/move/u/justingu/rmr_tracking/motions/locomotion_33hz.zarr` (the cleaner,
un-mirrored zarr).

Math primer
-----------
A y-axis mirror M = diag(1, -1, 1) acts on 3-vectors. Linear vectors
(positions, linear velocities, accelerations) reflect with `Rd`. Pseudo-
vectors (angular velocities, magnetic moments) flip the components NOT
flipped by M, i.e. `Rd_pseudo = diag(-1, 1, -1)`.

A rotation matrix R transforms to M @ R @ M under the world's mirror. The
first column (the x-axis in body frame) is mapped through Rd; the second
column (the y-axis in body frame) is mapped through Rd_pseudo (because the
body y-axis itself flips sign under the mirror). So a 6-D "rot6d" obs
(first 2 cols of R) reflects via block_diag(Rd, Rd_pseudo) = `Rot6d`.

Joint vectors mirror by:
  1. swapping every paired joint i <-> mirror(i) (left↔right)
  2. flipping the sign of joints whose axis is along the mirror direction
     (heuristic: "roll" or "yaw" in the joint name)
The matrix `Q[i, j] = sign[i] if j == perm[i] else 0` does both.

Per-body 3-vectors (like body_pos in pelvis frame, stacked across bodies):
  swap left↔right bodies AND apply Rd to each body's 3 components.
  → 3N×3N matrix `per_body_Rd`. Analogously for pseudo-vectors and rot6d.

All these matrices are INVOLUTIONS: M @ M = I. So the same op is used to
reflect into and out of the mirror frame.

Reference
---------
Ported from Stanford-TML/TML-BeyondMimic
(diffusion_policy/utils/symm_utils.py): `get_joint_permutation_matrix`,
`get_joint_reflection_matrix`, `get_body_permutation_matrix`,
`get_reflect_op`, `get_reflect_reps`.
"""

from __future__ import annotations

import numpy as np
import torch


# ─── reflection-op primitives (port from TML) ────────────────────────────────


def get_joint_permutation_matrix(joint_names: list[str]) -> torch.Tensor:
    """Return perm (long, len N) where perm[i] = index of i's left↔right mirror.
    Joints with no mirror map to themselves."""
    perm = torch.arange(len(joint_names), dtype=torch.long)
    for i, name in enumerate(joint_names):
        if name.startswith("left"):
            mirror_name = "right" + name[len("left"):]
            if mirror_name in joint_names:
                j = joint_names.index(mirror_name)
                perm[i] = j
                perm[j] = i
    return perm


def get_joint_reflection_signs(joint_names: list[str]) -> torch.Tensor:
    """Return ±1 vector (float, len N): -1 for joints whose value flips sign
    under y-mirror. Heuristic from TML: roll and yaw joints flip; pitch does
    not."""
    signs = torch.ones(len(joint_names), dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if "roll" in name or "yaw" in name:
            signs[i] = -1.0
    return signs


def get_body_permutation_matrix(body_names: list[str]) -> torch.Tensor:
    """Same as joint perm but for body names. Midline bodies map to themselves."""
    perm = torch.arange(len(body_names), dtype=torch.long)
    for i, name in enumerate(body_names):
        if name.startswith("left"):
            mirror_name = "right" + name[len("left"):]
            if mirror_name in body_names:
                j = body_names.index(mirror_name)
                perm[i] = j
                perm[j] = i
    return perm


def build_Q(joint_names: list[str]) -> torch.Tensor:
    """N×N joint mirror op. Q[i, perm[i]] = sign[i]."""
    n = len(joint_names)
    perm = get_joint_permutation_matrix(joint_names)
    signs = get_joint_reflection_signs(joint_names)
    Q = torch.zeros(n, n, dtype=torch.float32)
    Q[torch.arange(n), perm] = signs
    return Q


def build_Rd() -> torch.Tensor:
    """3×3 y-mirror for world linear vectors."""
    return torch.diag(torch.tensor([1.0, -1.0, 1.0]))


def build_Rd_pseudo() -> torch.Tensor:
    """3×3 y-mirror for world pseudo-vectors (angular velocities)."""
    return torch.diag(torch.tensor([-1.0, 1.0, -1.0]))


def build_Rot6d() -> torch.Tensor:
    """6×6 op for the rot6d orientation obs.

    IMPORTANT — layout: the obs is built as
    `matrix_from_quat(...)[..., :2].reshape(B, -1)` (see observations.py
    `motion_anchor_ori_b` / `robot_body_ori_b`). `[..., :2]` keeps the first two
    COLUMNS of R (shape (...,3,2)), but `.reshape(-1)` flattens ROW-MAJOR, so the
    6-vector is interleaved by row:
        [R00, R01, R10, R11, R20, R21]
    NOT the column-contiguous [R00,R10,R20, R01,R11,R21].

    Under a y-mirror R -> M R M with M = diag(1,-1,1), element (i,j) scales by
    m_i * m_j where m = (1,-1,1). In the row-major (i,j) order above that is
    diag(1,-1,-1,1,1,-1).
    """
    m = [1.0, -1.0, 1.0]
    signs = [m[i] * m[j] for i in range(3) for j in range(2)]  # row-major (i, j<2)
    return torch.diag(torch.tensor(signs))
    # WRONG (assumed column-contiguous [R00,R10,R20,R01,R11,R21]): it mis-signs
    # entries R10/R11 yet still squares to I, so the involution self-test never
    # caught it. Kept for reference.
    # return torch.block_diag(build_Rd(), build_Rd_pseudo())


def build_per_body(body_names: list[str], per_body_op: torch.Tensor) -> torch.Tensor:
    """(N*K)×(N*K) op that, when applied as `x @ op` to a tensor flattened as
    (..., N, K), permutes bodies along the first body dim and applies
    `per_body_op` (K×K) to each body's K components.

    Layout convention: flat features are interleaved per body — for body i,
    the K features come consecutively at indices [i*K, (i+1)*K).
    """
    n = len(body_names)
    K = per_body_op.shape[0]
    assert per_body_op.shape == (K, K)
    perm = get_body_permutation_matrix(body_names)
    op = torch.zeros(n, n, K, K, dtype=torch.float32)
    for i, j in enumerate(perm.tolist()):
        op[i, j] = per_body_op
    # Convert (i, j, r, c) -> matrix with row (i*K + r), col (j*K + c).
    return op.permute(0, 2, 1, 3).reshape(n * K, n * K)


def build_per_body_Rd(body_names: list[str]) -> torch.Tensor:
    return build_per_body(body_names, build_Rd())


def build_per_body_Rd_pseudo(body_names: list[str]) -> torch.Tensor:
    return build_per_body(body_names, build_Rd_pseudo())


def build_per_body_Rot6d(body_names: list[str]) -> torch.Tensor:
    return build_per_body(body_names, build_Rot6d())


# ─── per-term op kinds + group-op builder ────────────────────────────────────


class ReflectOpBuilder:
    """Caches base reflection ops; constructs per-term ops by name."""

    def __init__(self, joint_names: list[str], body_subset_names: list[str]):
        self.joint_names = list(joint_names)
        self.body_subset_names = list(body_subset_names)
        self.Q = build_Q(joint_names)
        self.Rd = build_Rd()
        self.Rd_pseudo = build_Rd_pseudo()
        self.Rot6d = build_Rot6d()
        self.per_body_Rd = build_per_body_Rd(body_subset_names)
        self.per_body_Rd_pseudo = build_per_body_Rd_pseudo(body_subset_names)
        self.per_body_Rot6d = build_per_body_Rot6d(body_subset_names)

    def build_term_op(self, kind: str) -> torch.Tensor:
        """Return the base op for a kind. Group-level builder may replicate
        this op N times if the actual term dim is a multiple (history-wrap)."""
        if kind == "Q":
            return self.Q
        if kind == "Q_concat_2":
            # `command` term = joint_pos (29) + joint_vel (29), concatenated.
            return torch.block_diag(self.Q, self.Q)
        if kind == "Rd":
            return self.Rd
        if kind == "Rd_pseudo":
            return self.Rd_pseudo
        if kind == "Rot6d":
            return self.Rot6d
        if kind == "per_body_Rd":
            return self.per_body_Rd
        if kind == "per_body_Rd_pseudo":
            return self.per_body_Rd_pseudo
        if kind == "per_body_Rot6d":
            return self.per_body_Rot6d
        raise ValueError(f"Unknown reflect-op kind: {kind!r}")


# ─── default op tables for the generalist task ───────────────────────────────
#
# These MUST match the ObsGroup definitions in
# `tasks/generalist/generalist_env_cfg.py`. If you reorder, add, or remove
# obs terms in a group, you must also update the corresponding table OR the
# wrapper will fail with a dim-mismatch RuntimeError at startup.
#
# For the `expert` group (added in Phase 2), additional table entries get
# appended via `register_op_table_extension(...)` from the env cfg.

DEFAULT_OP_TABLES: dict[str, list[tuple[str, str]]] = {
    "policy": [
        ("command", "Q_concat_2"),
        ("motion_anchor_ori_b", "Rot6d"),
        ("base_ang_vel", "Rd_pseudo"),
        ("joint_pos", "Q"),
        ("joint_vel", "Q"),
        ("actions", "Q"),
    ],
    "critic": [
        ("command", "Q_concat_2"),
        ("motion_anchor_pos_b", "Rd"),
        ("motion_anchor_ori_b", "Rot6d"),
        ("body_pos", "per_body_Rd"),
        ("body_ori", "per_body_Rot6d"),
        ("base_lin_vel", "Rd"),
        ("base_ang_vel", "Rd_pseudo"),
        ("joint_pos", "Q"),
        ("joint_vel", "Q"),
        ("actions", "Q"),
    ],
    "expert": [
        # PrivilegedCfg prefix
        ("command", "Q_concat_2"),
        ("motion_anchor_pos_b", "Rd"),
        ("motion_anchor_ori_b", "Rot6d"),
        ("body_pos", "per_body_Rd"),
        ("body_ori", "per_body_Rot6d"),
        ("base_lin_vel", "Rd"),
        ("base_ang_vel", "Rd_pseudo"),
        ("joint_pos", "Q"),
        ("joint_vel", "Q"),
        ("actions", "Q"),
        # ExpertCfg extras (Phase 2). Order matches the attribute order in
        # `ExpertCfg` in tasks/generalist/generalist_env_cfg.py.
        ("contact_force_mag",  "per_foot_scalar"),
        ("contact_air_time",   "per_foot_scalar"),
        ("body_pos_err",       "per_body_Rd"),
        ("body_ori_err",       "per_body_Rot6d"),
        ("body_lin_vel_err",   "per_body_Rd"),
        ("body_ang_vel_err",   "per_body_Rd_pseudo"),
    ],
}

# For the per_foot_scalar kind: per-foot 1-D scalar (e.g. force magnitude or
# air time). Under mirror, the two feet swap. So op is a 2x2 permutation
# matrix (left foot ↔ right foot), no sign flip (magnitudes are positive).
def build_per_foot_scalar_op() -> torch.Tensor:
    op = torch.zeros(2, 2, dtype=torch.float32)
    op[0, 1] = 1.0
    op[1, 0] = 1.0
    return op


DEFAULT_ACTION_OP_KIND = "Q"


# ─── env wrapper ─────────────────────────────────────────────────────────────


try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as _RslRlBase
except Exception:  # pragma: no cover — only triggered during ast-level smoke tests
    _RslRlBase = object


class SymmetricAugmentWrapper(_RslRlBase):
    """Per-env left/right mirror augmentation for a vec env.

    For envs whose `reflected_mask[i]` is True at the current step:
        * obs returned to the consumer is reflected: `obs[i] @= obs_op`
        * action received from the consumer is unreflected before stepping
          physics: `action[i] @= action_op`  (involution: op @ op = I)

    The env's reward and physics see real-frame quantities only; from the
    policy's perspective, ~half of envs are running the mirrored task. The
    optimal policy satisfies the symmetry π(mirror(o)) = mirror(π(o)) and
    gets effectively 2× the data per gradient step.

    Mask re-roll happens at episode reset (per env).

    Subclasses RslRlVecEnvWrapper so the runner's property accesses on
    `episode_length_buf`, `unwrapped`, `cfg`, `observation_space`,
    `action_space`, etc. work unchanged. We bypass the base __init__ because
    we're re-wrapping an already-wrapped env (don't want to re-reset).

    Args:
        wrapped_env: an already-built RslRlVecEnvWrapper.
        joint_names: full ordered list of robot joint names (e.g. all 29 G1
            joints from `robot.joint_names`). Used to build Q.
        body_subset_names: tracked-body list (e.g. the 14 entries in
            `commands.motion.body_names`). Used to build per-body ops.
        foot_subset_names: optional pair of foot body names in left, right
            order. Only needed when the env exposes per-foot obs terms.
        sym_aug_prob: probability per env to be in reflected mode after reset.
            Default 0.5 (balanced).
        op_tables: per-group {term_name -> op_kind} table. Defaults to
            DEFAULT_OP_TABLES. Pass an override dict to add extra groups.
        action_op_kind: kind for the action reflect op. Defaults to "Q".
        groups_to_reflect: which obs-group keys in the TensorDict should be
            reflected. Defaults to ("policy", "critic"). Add "expert" when
            running the privileged expert.
        verify: when True (default), prints op shapes and verifies op @ op = I.

    Limitations:
        - The op tables MUST exactly match the obs-cfg term order. Mismatched
          order or unknown term name raises a clear assertion at startup.
        - Per-term ops are replicated when the obs dim is a multiple of the
          base op dim (history-wrapped terms). Mixed history lengths within
          one term are NOT supported.
    """

    def __init__(
        self,
        wrapped_env,
        joint_names: list[str],
        body_subset_names: list[str],
        foot_subset_names: list[str] | None = None,
        sym_aug_prob: float = 0.5,
        op_tables: dict[str, list[tuple[str, str]]] | None = None,
        action_op_kind: str = DEFAULT_ACTION_OP_KIND,
        groups_to_reflect: tuple[str, ...] = ("policy", "critic"),
        verify: bool = True,
    ):
        # Bypass RslRlVecEnvWrapper.__init__ (it expects a raw gym env and
        # would re-reset). Copy the bookkeeping attrs from the wrapped env
        # so inherited properties / methods keep working.
        self.env = wrapped_env.env  # the underlying gym env
        self.clip_actions = getattr(wrapped_env, "clip_actions", None)
        self.num_envs = wrapped_env.num_envs
        self.device = wrapped_env.device
        self.num_actions = wrapped_env.num_actions
        self.max_episode_length = wrapped_env.max_episode_length

        self.sym_aug_prob = float(sym_aug_prob)
        self.groups_to_reflect = tuple(groups_to_reflect)

        self.builder = ReflectOpBuilder(joint_names, body_subset_names)
        # Per-foot op only built if the term is actually in any table.
        self._per_foot_scalar_op = build_per_foot_scalar_op()
        self._foot_subset_names = foot_subset_names

        tables = dict(DEFAULT_OP_TABLES) if op_tables is None else dict(op_tables)
        # Build per-group flat ops by introspecting the env's obs manager.
        obs_manager = self.unwrapped.observation_manager
        self.obs_ops: dict[str, torch.Tensor] = {}
        for group in self.groups_to_reflect:
            if group not in tables:
                raise ValueError(
                    f"SymmetricAugmentWrapper: no op table for group {group!r}. "
                    f"Have tables for: {sorted(tables.keys())}."
                )
            table = tables[group]
            op = self._build_group_op(obs_manager, group, table)
            # to(device) here so per-step reflection is a single matmul on GPU.
            self.obs_ops[group] = op.to(self.device)

        self.action_op = self.builder.build_term_op(action_op_kind).to(self.device)

        # Per-env reflected flag, re-rolled at episode reset.
        self.reflected_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reroll_mask(torch.arange(self.num_envs, device=self.device))

        if verify:
            self._verify_ops()
            print(f"[SymAug] groups: {list(self.obs_ops.keys())}  "
                  f"sym_aug_prob={self.sym_aug_prob:.2f}  num_envs={self.num_envs}  "
                  f"reflected={int(self.reflected_mask.sum())}/{self.num_envs} initially")

    # ── op construction ──────────────────────────────────────────────────────

    def _build_term_op(self, kind: str) -> torch.Tensor:
        if kind == "per_foot_scalar":
            return self._per_foot_scalar_op
        return self.builder.build_term_op(kind)

    def _build_group_op(self, obs_manager, group_name: str, table: list[tuple[str, str]]) -> torch.Tensor:
        # Get this group's per-term ordered names and per-term dims.
        # Isaac Lab's ObservationManager keeps `_group_obs_term_names` and
        # `_group_obs_term_dim` (a list of int OR tuple per term).
        group_term_names = list(obs_manager._group_obs_term_names[group_name])
        group_term_dims = list(obs_manager._group_obs_term_dim[group_name])

        table_term_names = [t[0] for t in table]
        if table_term_names != group_term_names:
            raise RuntimeError(
                f"SymmetricAugmentWrapper: op table for group {group_name!r} "
                f"doesn't match the env's actual obs-term order.\n"
                f"  Table: {table_term_names}\n"
                f"  Env:   {group_term_names}\n"
                f"Fix the table in symmetric_augment.DEFAULT_OP_TABLES."
            )

        sub_ops = []
        for (term_name, op_kind), term_dim in zip(table, group_term_dims):
            base_op = self._build_term_op(op_kind)
            base_op_dim = base_op.shape[0]
            actual_dim = int(np.prod(term_dim if isinstance(term_dim, tuple) else (term_dim,)))
            if actual_dim == base_op_dim:
                sub_ops.append(base_op)
            elif actual_dim % base_op_dim == 0:
                # History-wrapped term: replicate the op `n` times via block_diag.
                n_repeat = actual_dim // base_op_dim
                sub_ops.append(torch.block_diag(*([base_op] * n_repeat)))
            else:
                raise RuntimeError(
                    f"SymmetricAugmentWrapper: term {term_name!r} (group {group_name!r}) "
                    f"has dim {actual_dim} (raw {term_dim}) but op_kind {op_kind!r} "
                    f"produces dim {base_op_dim} — not a multiple. Check the table."
                )
        return torch.block_diag(*sub_ops)

    def _verify_ops(self):
        # Check involution: op @ op == I (within float32 tolerance).
        for group, op in self.obs_ops.items():
            sq = op @ op
            eye = torch.eye(op.shape[0], device=op.device)
            err = (sq - eye).abs().max().item()
            assert err < 1e-5, f"obs op for {group!r} is not an involution (err={err:.2e})"
            print(f"[SymAug]   group {group!r}: op shape {tuple(op.shape)}, involution err {err:.2e}")
        sq = self.action_op @ self.action_op
        eye = torch.eye(self.action_op.shape[0], device=self.action_op.device)
        err = (sq - eye).abs().max().item()
        assert err < 1e-5, f"action op is not an involution (err={err:.2e})"
        print(f"[SymAug]   action op: shape {tuple(self.action_op.shape)}, involution err {err:.2e}")

    # ── mask management ──────────────────────────────────────────────────────

    def _reroll_mask(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.reflected_mask[env_ids] = torch.rand(len(env_ids), device=self.device) < self.sym_aug_prob

    # ── obs / action reflection ──────────────────────────────────────────────

    def _reflect_obs_inplace(self, obs):
        """Apply per-group reflect op to masked envs. `obs` is a TensorDict-like
        (must support __contains__, __getitem__, and __setitem__ with bool mask).
        """
        if not self.reflected_mask.any():
            return
        for group, op in self.obs_ops.items():
            if group not in obs:
                continue
            x = obs[group]
            # x: [num_envs, flat_dim]
            x[self.reflected_mask] = x[self.reflected_mask] @ op
            # No need to write back — in-place index assignment mutates the tensor.

    def _reflect_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.reflected_mask.any():
            return action
        # Clone so we don't mutate the policy's output buffer (some runners reuse it).
        action = action.clone()
        action[self.reflected_mask] = action[self.reflected_mask] @ self.action_op
        return action

    # ── env API overrides (call inherited base to do the actual env work,
    #    then apply reflection on top) ─────────────────────────────────────────

    def reset(self):
        obs, extras = super().reset()  # RslRlVecEnvWrapper.reset → TensorDict
        # All envs were reset → re-roll all.
        self._reroll_mask(torch.arange(self.num_envs, device=self.device))
        self._reflect_obs_inplace(obs)
        return obs, extras

    def step(self, actions: torch.Tensor):
        actions = self._reflect_action(actions)
        obs, rew, dones, extras = super().step(actions)
        # Re-roll mask for envs that just terminated (so the new episode gets a fresh roll).
        terminated = torch.where(dones)[0]
        if terminated.numel() > 0:
            self._reroll_mask(terminated)
        self._reflect_obs_inplace(obs)
        return obs, rew, dones, extras

    def get_observations(self):
        obs = super().get_observations()
        self._reflect_obs_inplace(obs)
        return obs
