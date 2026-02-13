"""Holosoma-compatible ONNX exporter for WBT policies.

Exports a training ONNX model that is directly loadable by holosoma inference
without any post-processing (no adapt_onnx.py, no runtime permutation).

The exported ONNX will have:
- Obs input in holosoma's alphabetically-sorted layout
- Actions, joint_pos, joint_vel in Holosoma/MuJoCo joint order
- ref_pos_xyz and ref_quat_xyzw for a single reference body (default: torso_link)
- Metadata in holosoma JSON format with holosoma field names
- Embedded URDF text for Pinocchio FK

Usage from play.py:
    python scripts/rsl_rl/play.py --task <task> ... --export-holosoma

Usage from MotionOnPolicyRunner:
    Set environment variable EXPORT_HOLOSOMA=1 before training.
"""

from __future__ import annotations

import copy
import json
import os

import onnx
import torch

from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp import MotionCommand

# =============================================================================
# Holosoma / MuJoCo G1 29-DOF joint order
# Matches holosoma_inference/utils/joint_orders.py MUJOCO_G1_JOINT_NAMES
# =============================================================================

HOLOSOMA_G1_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# =============================================================================
# Observation term name mapping: IsaacLab/rmr_tracking -> holosoma inference
# =============================================================================

ISAACLAB_TO_HOLOSOMA_OBS_NAME: dict[str, str] = {
    "command": "motion_command",
    "motion_anchor_ori_b": "motion_ref_ori_b",
    "motion_anchor_pos_b": "motion_ref_pos_b",
    "joint_pos": "dof_pos",
    "joint_vel": "dof_vel",
    "base_ang_vel": "base_ang_vel",
    "base_lin_vel": "base_lin_vel",
    "actions": "actions",
}


# Observation terms whose elements are joint-indexed and need within-block
# permutation from Holosoma joint order to IsaacLab joint order.
# Format: holosoma_term_name -> list of (sub_offset, sub_length) ranges.
# For "motion_command" (58 = joint_pos_29 + joint_vel_29), each half is
# independently joint-indexed.
JOINT_INDEXED_OBS_TERMS: dict[str, list[tuple[int, int]]] = {
    "dof_pos": [(0, 29)],
    "dof_vel": [(0, 29)],
    "actions": [(0, 29)],
    "motion_command": [(0, 29), (29, 29)],
}


def _compute_obs_permutation(
    isaaclab_obs_names: list[str],
    obs_dims: dict[str, int],
    joint_perm_isaac_to_holo: list[int],
) -> torch.LongTensor:
    """Compute a flat permutation from holosoma's alphabetical obs layout to
    the training (declaration) obs layout, including within-block joint
    element reordering for joint-indexed terms.

    The input obs vector has elements in holosoma's alphabetically-sorted
    term order, with joint-indexed terms (dof_pos, dof_vel, actions,
    motion_command) containing elements in **Holosoma joint order**.

    The output must be in the training's declaration term order, with
    joint-indexed terms containing elements in **IsaacLab joint order**
    (matching the normalizer's mean/std layout).

    Parameters
    ----------
    isaaclab_obs_names : list[str]
        Observation term names in IsaacLab declaration order (from
        ``env.observation_manager.active_terms["policy"]``).
    obs_dims : dict[str, int]
        Observation dimension per term, keyed by **holosoma** name.
    joint_perm_isaac_to_holo : list[int]
        Joint permutation where ``holosoma[i] = isaac[perm[i]]``.  Used to
        derive the inverse (Holosoma -> IsaacLab) for within-block reordering.

    Returns
    -------
    torch.LongTensor
        Permutation indices of length ``sum(obs_dims)``.  When applied as
        ``obs[:, perm]``, converts a holosoma-alphabetical-order vector
        (with Holosoma joint element order) into the training
        declaration-order vector (with IsaacLab joint element order).
    """
    # Inverse joint permutation: isaac[j] = holosoma[inv_perm[j]]
    num_joints = len(joint_perm_isaac_to_holo)
    inv_joint_perm = [0] * num_joints
    for holo_idx, isaac_idx in enumerate(joint_perm_isaac_to_holo):
        inv_joint_perm[isaac_idx] = holo_idx

    # Map IsaacLab names to holosoma names, preserving declaration order
    holosoma_names_decl = [ISAACLAB_TO_HOLOSOMA_OBS_NAME.get(n, n) for n in isaaclab_obs_names]

    # Holosoma sorts alphabetically
    holosoma_names_sorted = sorted(holosoma_names_decl)

    # Build flat offset maps
    def _flat_offsets(names: list[str]) -> dict[str, int]:
        offsets: dict[str, int] = {}
        cursor = 0
        for n in names:
            offsets[n] = cursor
            cursor += obs_dims[n]
        return offsets

    sorted_offsets = _flat_offsets(holosoma_names_sorted)
    decl_offsets = _flat_offsets(holosoma_names_decl)
    total_dim = sum(obs_dims[n] for n in holosoma_names_sorted)

    # Build permutation: for each position in the declaration-order flat vector
    # (output), which position in the sorted flat vector (input) to gather from.
    perm = [0] * total_dim
    for hname in holosoma_names_sorted:
        dim = obs_dims[hname]
        src_start = sorted_offsets[hname]  # position in sorted (input) vector
        dst_start = decl_offsets[hname]    # position in declaration (output) vector

        if hname in JOINT_INDEXED_OBS_TERMS:
            # Joint-indexed term: also permute within-block from Holosoma to
            # IsaacLab joint order.  Start with identity mapping for any
            # non-joint sub-ranges (shouldn't happen but safe).
            local_map = list(range(dim))
            for sub_offset, sub_length in JOINT_INDEXED_OBS_TERMS[hname]:
                for k in range(sub_length):
                    # Output position dst_start + sub_offset + k needs
                    # IsaacLab joint k.  In the input (Holosoma order),
                    # that's at position inv_joint_perm[k].
                    local_map[sub_offset + k] = sub_offset + inv_joint_perm[k]
            for j in range(dim):
                perm[dst_start + j] = src_start + local_map[j]
        else:
            # Non-joint-indexed term: straight block copy
            for j in range(dim):
                perm[dst_start + j] = src_start + j

    return torch.tensor(perm, dtype=torch.long)


def _compute_joint_permutation(
    isaaclab_joint_names: list[str],
    holosoma_joint_names: tuple[str, ...] = HOLOSOMA_G1_JOINT_ORDER,
) -> torch.LongTensor:
    """Compute permutation from IsaacLab joint order to Holosoma joint order.

    Parameters
    ----------
    isaaclab_joint_names : list[str]
        Joint names in IsaacLab order.
    holosoma_joint_names : tuple[str, ...]
        Joint names in Holosoma/MuJoCo order.

    Returns
    -------
    torch.LongTensor
        Indices such that ``tensor[:, perm]`` reorders from IsaacLab to Holosoma.
    """
    isaac_index = {name: i for i, name in enumerate(isaaclab_joint_names)}
    perm = [isaac_index[name] for name in holosoma_joint_names]
    return torch.tensor(perm, dtype=torch.long)


class _HolosomaOnnxExporter(torch.nn.Module):
    """Export a WBT motion policy as a holosoma-native ONNX.

    The exported model:
    - Accepts ``obs`` in holosoma's alphabetically-sorted layout
    - Outputs ``actions``, ``joint_pos``, ``joint_vel`` in Holosoma/MuJoCo joint order
    - Outputs ``ref_pos_xyz`` and ``ref_quat_xyzw`` for a single reference body
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        actor_critic,
        normalizer=None,
        ref_body_name: str = "torso_link",
        verbose: bool = False,
    ):
        super().__init__()
        self.verbose = verbose

        # --- Deep-copy actor + normalizer ---
        if hasattr(actor_critic, "actor"):
            self.actor = copy.deepcopy(actor_critic.actor)
        elif hasattr(actor_critic, "student"):
            self.actor = copy.deepcopy(actor_critic.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        # --- Motion data ---
        cmd: MotionCommand = env.command_manager.get_term("motion")
        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

        # --- Find reference body index ---
        body_names: list[str] = cmd.cfg.body_names
        if ref_body_name not in body_names:
            raise ValueError(
                f"Reference body '{ref_body_name}' not in body_names: {body_names}"
            )
        self.ref_body_idx = body_names.index(ref_body_name)

        # --- Joint permutation (IsaacLab -> Holosoma) ---
        isaaclab_joint_names = list(env.scene["robot"].data.joint_names)
        joint_perm = _compute_joint_permutation(isaaclab_joint_names)
        self.register_buffer("joint_perm", joint_perm)

        # --- Observation permutation ---
        # Get obs term names in IsaacLab declaration order
        isaaclab_obs_names = env.observation_manager.active_terms["policy"]

        # Build obs dims dict (holosoma names -> dim) from the env
        obs_dims: dict[str, int] = {}
        for isaac_name in isaaclab_obs_names:
            holo_name = ISAACLAB_TO_HOLOSOMA_OBS_NAME.get(isaac_name, isaac_name)
            # Get dim from the observation manager's term shapes
            term_idx = isaaclab_obs_names.index(isaac_name)
            term_dim = env.observation_manager.group_obs_term_dim["policy"][term_idx]
            # term_dim is a tuple, e.g. (29,) -- take the product for flat dim
            if isinstance(term_dim, tuple):
                dim = 1
                for d in term_dim:
                    dim *= d
            else:
                dim = term_dim
            obs_dims[holo_name] = dim

        # The obs permutation handles BOTH:
        # 1. Block-level reordering: alphabetical -> declaration order
        # 2. Within-block joint element reordering: Holosoma -> IsaacLab
        #    for joint-indexed terms (dof_pos, dof_vel, actions, motion_command)
        self.register_buffer(
            "obs_perm",
            _compute_obs_permutation(
                isaaclab_obs_names, obs_dims, joint_perm.tolist()
            ),
        )

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        # obs arrives in holosoma alphabetical order [1, obs_dim]
        # Permute to training declaration order for normalizer + actor
        obs_reordered = obs[:, self.obs_perm]

        # Run normalizer + actor -> actions in IsaacLab joint order
        actions_isaac = self.actor(self.normalizer(obs_reordered))

        # Permute actions to Holosoma joint order
        actions = actions_isaac[:, self.joint_perm]

        # Get motion data for this timestep
        t = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)

        # Permute joint_pos/joint_vel to Holosoma order
        joint_pos = self.joint_pos[t][:, self.joint_perm]
        joint_vel = self.joint_vel[t][:, self.joint_perm]

        # Extract single reference body
        ref_pos_xyz = self.body_pos_w[t][:, self.ref_body_idx]    # [1, 3]
        ref_quat_xyzw = self.body_quat_w[t][:, self.ref_body_idx]  # [1, 4]

        return actions, joint_pos, joint_vel, ref_pos_xyz, ref_quat_xyzw

    def export(self, path: str, filename: str):
        self.to("cpu")
        self.eval()
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "ref_pos_xyz",
                "ref_quat_xyzw",
            ],
            dynamic_axes={},
        )


def export_holosoma_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    ref_body_name: str = "torso_link",
    filename: str = "policy_holosoma.onnx",
    verbose: bool = False,
) -> str:
    """Export a WBT motion policy as a holosoma-native ONNX.

    This is the top-level entry point, parallel to ``export_motion_policy_as_onnx``.

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The Isaac Lab environment (unwrapped).
    actor_critic : object
        The policy module (must have ``.actor`` or ``.student``).
    path : str
        Directory to save the ONNX file.
    normalizer : object or None
        The empirical normalizer. If None, Identity is used.
    ref_body_name : str
        Body to extract for ref_pos_xyz / ref_quat_xyzw (default: "torso_link").
    filename : str
        Output filename (default: "policy_holosoma.onnx").
    verbose : bool
        Print ONNX export details.

    Returns
    -------
    str
        Full path to the exported ONNX file.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    exporter = _HolosomaOnnxExporter(
        env, actor_critic, normalizer, ref_body_name, verbose
    )
    exporter.export(path, filename)

    onnx_path = os.path.join(path, filename)
    print(f"[holosoma_exporter] Exported holosoma ONNX to: {onnx_path}")
    return onnx_path


def attach_holosoma_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename: str = "policy_holosoma.onnx",
    ref_body_name: str = "torso_link",
    urdf_path: str | None = None,
) -> None:
    """Attach holosoma-format metadata to an exported ONNX.

    Writes metadata with holosoma conventions:
    - JSON-serialized values (not CSV)
    - Holosoma field names: ``dof_names``, ``kp``, ``kd`` (not joint_names, joint_stiffness, joint_damping)
    - All joint-indexed values in Holosoma/MuJoCo order
    - Embedded URDF text for Pinocchio FK

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The Isaac Lab environment (unwrapped).
    run_path : str
        Wandb run path or identifier string.
    path : str
        Directory containing the ONNX file.
    filename : str
        ONNX filename.
    ref_body_name : str
        Body used for ref_pos_xyz / ref_quat_xyzw.
    urdf_path : str or None
        Optional override for the URDF file path. If None, resolved from
        ``env.scene["robot"].cfg.spawn.asset_path``.
    """
    onnx_path = os.path.join(path, filename)

    # --- Compute joint permutation indices ---
    isaaclab_joint_names = list(env.scene["robot"].data.joint_names)
    joint_perm = _compute_joint_permutation(isaaclab_joint_names).tolist()

    # --- Permute joint-indexed values to Holosoma order ---
    def _permute(values: list) -> list:
        return [values[i] for i in joint_perm]

    holosoma_dof_names = list(HOLOSOMA_G1_JOINT_ORDER)

    stiffness_isaac = env.scene["robot"].data.joint_stiffness[0].cpu().tolist()
    damping_isaac = env.scene["robot"].data.joint_damping[0].cpu().tolist()
    default_pos_isaac = env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist()
    action_scale_isaac = env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist()

    kp = _permute(stiffness_isaac)
    kd = _permute(damping_isaac)
    default_joint_pos = _permute(default_pos_isaac)
    action_scale = _permute(action_scale_isaac)

    # --- Observation names in holosoma alphabetical order ---
    isaaclab_obs_names = env.observation_manager.active_terms["policy"]
    holosoma_obs_names = sorted(
        ISAACLAB_TO_HOLOSOMA_OBS_NAME.get(n, n) for n in isaaclab_obs_names
    )

    # --- Observation history lengths ---
    observation_history_lengths: list[int] = []
    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [
            env.observation_manager.cfg.policy.history_length
        ] * len(isaaclab_obs_names)
    else:
        for name in isaaclab_obs_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    # --- URDF embedding ---
    robot_urdf_text = ""
    robot_urdf_relpath = ""
    if urdf_path is None:
        # Resolve from robot config
        spawn_cfg = env.scene["robot"].cfg.spawn
        if hasattr(spawn_cfg, "asset_path"):
            urdf_path = spawn_cfg.asset_path

    if urdf_path and os.path.isfile(urdf_path):
        with open(urdf_path, "r") as f:
            robot_urdf_text = f.read()
        robot_urdf_relpath = urdf_path
        print(f"[holosoma_exporter] Embedded URDF from: {urdf_path}")
    else:
        print(
            f"[holosoma_exporter] WARNING: Could not find URDF at '{urdf_path}'. "
            "Metadata will not contain robot_urdf. You may need to copy it from "
            "a reference ONNX or specify --urdf-path."
        )

    # --- Build metadata dict ---
    metadata = {
        "run_path": run_path,
        "dof_names": holosoma_dof_names,
        "kp": kp,
        "kd": kd,
        "default_joint_pos": default_joint_pos,
        "action_scale": action_scale,
        "observation_names": holosoma_obs_names,
        "observation_history_lengths": observation_history_lengths,
        "anchor_body_name": ref_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
        "command_names": env.command_manager.active_terms,
    }

    if robot_urdf_text:
        metadata["robot_urdf"] = robot_urdf_text
    if robot_urdf_relpath:
        metadata["robot_urdf_path"] = robot_urdf_relpath

    # --- Write metadata as JSON ---
    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = json.dumps(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
    print(f"[holosoma_exporter] Attached holosoma metadata to: {onnx_path}")
