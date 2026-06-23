"""DAgger fine-tuning for a popart-task generalist using a specialist expert.

Loads two RSL-RL checkpoints (student = baseline generalist; expert = specialist
trained on the failed clips) into the same process, then iterates:
    1. Roll out student with mild noise on failed-clip env.
    2. Label visited states with expert's deterministic action (act_inference).
    3. BC-train student on the aggregated (obs, expert_action) buffer.

Reuses the popart task and env unchanged. The only structural deviation from PPO
training is that we DON'T call runner.alg.update() — DAgger uses BC (MSE), not PPO.

Usage:
    python scripts/rsl_rl/dagger.py \\
        --task=Popart-Flat-G1-v0 \\
        --student_wandb=robot-mcrobotface/balanced_sampling/<baseline_id> \\
        --expert_wandb=robot-mcrobotface/balanced_sampling/<specialist_id> \\
        --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \\
        --include_clip_names_file eval_results/<baseline>/failed_clip_ids.json \\
        --categories stand_up,walk,jump --decimation 6 --activation swish \\
        --num_envs 256 --headless \\
        --logger wandb --log_project_name balanced_sampling \\
        --run_name dagger_student_v1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="DAgger fine-tuning loop.")
parser.add_argument("--num_envs", type=int, default=256, help="Envs for student rollout collection.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--zarr_path", type=str, default=None)
parser.add_argument("--decimation", type=int, default=None)
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"])
parser.add_argument("--categories", type=str, default=None)
parser.add_argument("--popart", type=str, default="off", choices=["on", "off"],
                    help="Must match training-time choice for BOTH student and expert.")
parser.add_argument("--include_clip_names_file", type=str, default=None,
                    help="JSON list of clip names. DAgger rolls out only on these.")
parser.add_argument("--sampling_mode", type=str, default="frame_uniform",
                    choices=["frame_uniform", "balanced", "clip_adaptive",
                             "cat_uniform_clip_adaptive", "cat_adaptive_clip_uniform",
                             "cat_adaptive_clip_adaptive",
                             "cat_blend_clip_uniform"],
                    help="Sampling mode during DAgger rollouts. Default frame_uniform "
                         "is safest when the include_clip_names filter leaves some cats empty.")

# Policy-source flags (wandb or local).
parser.add_argument("--student_wandb", type=str, default=None,
                    help="Wandb run path for the baseline (will become the student).")
parser.add_argument("--student_local", type=str, default=None,
                    help="Local .pt checkpoint path for the student (overrides --student_wandb).")
parser.add_argument("--expert_wandb", type=str, default=None,
                    help="Wandb run path for the specialist (expert).")
parser.add_argument("--expert_local", type=str, default=None,
                    help="Local .pt checkpoint path for the expert (overrides --expert_wandb).")

# DAgger hyperparams.
parser.add_argument("--n_iters", type=int, default=10)
parser.add_argument("--rollout_steps", type=int, default=200,
                    help="Env steps per rollout phase (per iter).")
parser.add_argument("--bc_epochs", type=int, default=3, help="Passes over the buffer per iter.")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--buffer_cap", type=int, default=1_000_000, help="FIFO buffer capacity (pairs).")
parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N iters.")
# Two-pool DAgger (anti-forgetting):
#   When enabled, the env loads the FULL zarr (no include filter). num_envs is
#   split into two pools: the first `failed_pool_frac` fraction always sample
#   from failed clips (labels from specialist `--expert_wandb`), and the rest
#   always sample from easy clips (labels from a frozen copy of the baseline,
#   i.e. the student's own init weights). This prevents catastrophic
#   forgetting on easy clips while still distilling the specialist's wins on
#   failed clips. In this mode, `--include_clip_names_file` is reinterpreted
#   as "which clips count as 'failed'" instead of "restrict the env to these".
parser.add_argument("--two_pool", action="store_true",
                    help="Enable two-pool DAgger (anti-forgetting). See module docstring.")
parser.add_argument("--failed_pool_frac", type=float, default=0.3,
                    help="(two-pool only) Fraction of envs assigned to the failed-clip pool. "
                         "Default 0.3 → 30%% specialist labels, 70%% baseline labels.")
# Privileged-expert DAgger (Phase 2): expert reads from a different obs group
# (with strictly more privileged signals) than the student. Both policies see
# the same env-step obs TensorDict, but each ActorCritic filters via its own
# obs_groups mapping. Requires the env cfg to expose the expert obs group.
parser.add_argument("--expert_obs_group", type=str, default="policy",
                    help="Which obs group the EXPERT consumes ('policy' = same as "
                         "student / backward-compatible, 'expert' = privileged "
                         "Phase-2 group). When 'expert' the expert ActorCritic is "
                         "built with a separate obs_groups mapping; the student "
                         "still reads from 'policy'. Only the action dim has to "
                         "match between the two — it does, since both control the "
                         "same robot.")
parser.add_argument("--cat_uniform_prob", type=float, default=None,
                    help="Probability of UNIFORM sampling at the CATEGORY stage "
                         "(used by cat-aware modes). 0 = pure adaptive, 1 = uniform. "
                         "Default in env cfg: 0.5.")
parser.add_argument("--clip_uniform_prob", type=float, default=None,
                    help="Probability of UNIFORM sampling at the CLIP stage "
                         "(used by *_clip_adaptive modes). 0 = pure adaptive, 1 = uniform. "
                         "Default in env cfg: 0.5.")
parser.add_argument("--symmetric_augment", action="store_true", default=False,
                    help="Apply symmetric (y-mirror) augmentation during DAgger rollouts. "
                         "Wraps the vec env with the same SymmetricAugmentWrapper used in "
                         "train_bones.py. Both expert and student see reflected obs and "
                         "produce reflected actions consistently; physics is unchanged.")
parser.add_argument("--sym_aug_prob", type=float, default=0.5,
                    help="Probability per env to be in reflected mode after reset.")
# # `--logger`, `--log_project_name`, `--run_name` are all added by
# # cli_args.add_rsl_rl_args below. Defining them here triggers argparse
# # conflicts. Pass them on the CLI (dagger.sh block 3 already does).
# parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
# parser.add_argument("--log_project_name", type=str, default="balanced_sampling")
# parser.add_argument("--run_name", type=str, default="dagger_student")
parser.add_argument("--seed", type=int, default=0)

# append RSL-RL cli arguments (for cli_args.parse_rsl_rl_cfg)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.popart = (args_cli.popart == "on")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import json
import os
import torch
import torch.nn.functional as F
import types

from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner


# ─── Helpers ────────────────────────────────────────────────────────────────

def _download_wandb_checkpoint(wandb_path: str, tmp_subdir: str) -> str:
    """Download the latest model_*.pt from a wandb run. Returns local path."""
    import wandb
    api = wandb.Api(timeout=60)
    run_path = wandb_path
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [f.name for f in wandb_run.files() if "model" in f.name]
    if "model" in wandb_path:
        fname = wandb_path.split("/")[-1]
    else:
        fname = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    dl_dir = f"./logs/rsl_rl/temp/{tmp_subdir}"
    wandb_run.file(str(fname)).download(dl_dir, replace=True)
    path = os.path.join(dl_dir, fname)
    print(f"[INFO] {tmp_subdir} checkpoint: {run_path}/{fname} -> {path}")
    return path


def _resolve_checkpoint(wandb_path: str | None, local_path: str | None, label: str) -> str:
    """Pick local override > wandb path."""
    if local_path is not None:
        assert os.path.isfile(local_path), f"--{label}_local file not found: {local_path}"
        return local_path
    if wandb_path is not None:
        return _download_wandb_checkpoint(wandb_path, tmp_subdir=label)
    raise ValueError(f"Provide --{label}_wandb or --{label}_local.")


class GPUReplayBuffer:
    """Fixed-capacity FIFO buffer for (obs, expert_action) pairs on GPU."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), device=device, dtype=torch.float32)
        self.acts = torch.zeros((capacity, action_dim), device=device, dtype=torch.float32)
        self.size = 0
        self.ptr = 0
        self.device = device

    def add_batch(self, obs: torch.Tensor, acts: torch.Tensor):
        n = obs.shape[0]
        if n == 0:
            return
        # Write at ptr with wraparound.
        if self.ptr + n <= self.capacity:
            self.obs[self.ptr:self.ptr + n] = obs
            self.acts[self.ptr:self.ptr + n] = acts
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:first]
            self.acts[self.ptr:] = acts[:first]
            self.obs[:n - first] = obs[first:]
            self.acts[:n - first] = acts[first:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.acts[idx]


def _save_checkpoint(
    path: str,
    student_policy: torch.nn.Module,
    iter_num: int,
    optimizer: torch.optim.Optimizer | None = None,
):
    """Save in the rsl_rl `OnPolicyRunner.load` format.

    rsl_rl's load reads `infos` unconditionally and `optimizer_state_dict`
    when load_optimizer=True (its default). Include both so the resulting
    .pt file is a drop-in replacement for any PPO-trained checkpoint —
    eval_specialist_pool.py and play_bones_clip.py can load it without
    special-casing DAgger output.
    """
    payload = {
        "model_state_dict": student_policy.state_dict(),
        "iter": iter_num,
        "infos": {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    # Mirror to wandb so eval_specialist_pool.py / play_bones_clip.py can pull
    # this checkpoint via --wandb_path like any other run. No-op if wandb not
    # active (e.g. --logger tensorboard).
    try:
        import wandb
        if wandb.run is not None:
            wandb.save(path, base_path=os.path.dirname(path), policy="now")
    except Exception as e:
        print(f"[DAGGER] (non-fatal) wandb.save failed: {e}")
    print(f"[DAGGER] saved checkpoint -> {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # Resolve agent_cfg flags exactly like train_bones (so the ActorCritic
    # built by the runner has matching architecture to BOTH checkpoints).
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation

    if args_cli.popart:
        agent_cfg.class_name = "PopArtMotionOnPolicyRunner"
    else:
        agent_cfg.class_name = "MotionOnPolicyRunner"
        for k in ("num_categories", "popart_momentum", "category_obs_group"):
            if hasattr(agent_cfg.policy, k):
                delattr(agent_cfg.policy, k)

    # ── Env cfg ──
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    assert args_cli.zarr_path is not None and os.path.isdir(args_cli.zarr_path), \
        f"--zarr_path required and must exist: {args_cli.zarr_path}"
    env_cfg.commands.motion.zarr_path = args_cli.zarr_path

    if args_cli.categories is not None and hasattr(env_cfg.commands.motion, "categories"):
        cats = [s.strip() for s in args_cli.categories.split(",") if s.strip()]
        env_cfg.commands.motion.categories = cats
        print(f"[INFO] PopArt categories: {cats}")

    # In single-pool mode, --include_clip_names_file restricts the env to those
    # clips. In two-pool mode, the env loads ALL clips (no restriction); the
    # file's contents are later used to PARTITION clips into failed/easy pools.
    if args_cli.include_clip_names_file is not None and not args_cli.two_pool \
            and hasattr(env_cfg.commands.motion, "include_clip_names"):
        with open(args_cli.include_clip_names_file, "r") as f:
            names = json.load(f)
        env_cfg.commands.motion.include_clip_names = names
        print(f"[INFO] Restricting DAgger rollouts to {len(names)} clips from {args_cli.include_clip_names_file}")
    elif args_cli.two_pool:
        if args_cli.include_clip_names_file is None:
            raise ValueError("--two_pool requires --include_clip_names_file (the failed clip list).")
        print(f"[INFO] Two-pool mode: env loads FULL zarr; "
              f"{args_cli.include_clip_names_file} defines the failed-clip pool.")

    if hasattr(env_cfg.commands.motion, "sampling_mode"):
        env_cfg.commands.motion.sampling_mode = args_cli.sampling_mode
        print(f"[INFO] sampling_mode = {args_cli.sampling_mode}")

    # ── Build env ──
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # Optional symmetric augmentation (cleaner-zarr training). Mirrors the
    # train_bones.py wiring (same wrapper, same op tables).
    if args_cli.symmetric_augment:
        from whole_body_tracking.tasks.generalist.mdp.symmetric_augment import SymmetricAugmentWrapper
        raw_env = env.unwrapped
        robot = raw_env.scene["robot"]
        joint_names = list(robot.joint_names)
        body_subset_names = list(env_cfg.commands.motion.body_names)
        groups = ["policy", "critic"]
        for g in ("expert",):
            if g in raw_env.observation_manager._group_obs_term_names:
                groups.append(g)
        env = SymmetricAugmentWrapper(
            env,
            joint_names=joint_names,
            body_subset_names=body_subset_names,
            sym_aug_prob=args_cli.sym_aug_prob,
            groups_to_reflect=tuple(groups),
        )

    device = torch.device(agent_cfg.device)

    # ── Optionally route the student's adaptive-sampling uniform-mix probs ──
    if args_cli.cat_uniform_prob is not None and hasattr(env_cfg.commands.motion, "cat_uniform_prob"):
        env_cfg.commands.motion.cat_uniform_prob = float(args_cli.cat_uniform_prob)
    if args_cli.clip_uniform_prob is not None and hasattr(env_cfg.commands.motion, "clip_uniform_prob"):
        env_cfg.commands.motion.clip_uniform_prob = float(args_cli.clip_uniform_prob)

    # ── Build runner; load expert -> snapshot -> load student ──
    if agent_cfg.class_name == "PopArtMotionOnPolicyRunner":
        from whole_body_tracking.utils.popart_runner import PopArtMotionOnPolicyRunner
        _RunnerCls = PopArtMotionOnPolicyRunner
    else:
        _RunnerCls = MotionOnPolicyRunner

    student_path = _resolve_checkpoint(args_cli.student_wandb, args_cli.student_local, "student")
    expert_path = _resolve_checkpoint(args_cli.expert_wandb, args_cli.expert_local, "expert")

    # The STUDENT runner. Its ActorCritic reads from the obs groups in
    # agent_cfg.obs_groups (defaults: policy="policy", critic="critic").
    runner = _RunnerCls(env, agent_cfg.to_dict(), log_dir=None, device=device.type)

    # ── Expert build ──
    # If --expert_obs_group=="policy" (default), the expert and student have
    # identical architectures — reuse the single runner (legacy code path).
    # Otherwise (e.g. "expert"), build a SECOND runner with obs_groups routed
    # to that group, load the expert into it, snapshot, then discard the
    # second runner. The two runners share the env (no contention; we don't
    # call .learn() on the expert one).
    if args_cli.expert_obs_group == "policy":
        # legacy: load expert into the same runner, snapshot
        runner.load(expert_path)
        expert_policy = copy.deepcopy(runner.alg.policy)
        for p in expert_policy.parameters():
            p.requires_grad_(False)
        expert_policy.eval()
        if hasattr(expert_policy, "actor_obs_normalizer"):
            expert_policy.actor_obs_normalizer.eval()
        print("[DAGGER] expert snapshot taken (frozen) — shared obs_groups.")
    else:
        # privileged path: build a separate runner with the expert's obs_groups.
        # Verify the requested obs group is actually present in the env.
        if args_cli.expert_obs_group not in env.unwrapped.observation_manager._group_obs_term_names:
            raise RuntimeError(
                f"--expert_obs_group={args_cli.expert_obs_group!r} but the env doesn't "
                f"expose that group. Have: "
                f"{list(env.unwrapped.observation_manager._group_obs_term_names.keys())}"
            )
        expert_agent_cfg = copy.deepcopy(agent_cfg)
        expert_agent_cfg.obs_groups = {
            "policy": [args_cli.expert_obs_group],
            "critic": [args_cli.expert_obs_group],
        }
        expert_runner = _RunnerCls(env, expert_agent_cfg.to_dict(), log_dir=None, device=device.type)
        expert_runner.load(expert_path)
        expert_policy = copy.deepcopy(expert_runner.alg.policy)
        for p in expert_policy.parameters():
            p.requires_grad_(False)
        expert_policy.eval()
        if hasattr(expert_policy, "actor_obs_normalizer"):
            expert_policy.actor_obs_normalizer.eval()
        # Don't keep the expert runner around — it would re-write env state
        # during any subsequent .learn() call.
        del expert_runner
        print(f"[DAGGER] expert snapshot taken (frozen) — privileged obs_group="
              f"{args_cli.expert_obs_group!r}.")

    runner.load(student_path)
    student_policy = runner.alg.policy
    # Freeze normalizer (its stats matched the loaded weights — don't drift).
    if hasattr(student_policy, "actor_obs_normalizer"):
        student_policy.actor_obs_normalizer.eval()
    print("[DAGGER] student loaded into runner (trainable).")

    # ── Two-pool: baseline-frozen expert + per-pool sampler ──
    baseline_frozen_policy = None
    failed_env_mask = None
    easy_env_mask = None
    if args_cli.two_pool:
        # Snapshot baseline (= student at init) as a frozen "easy-pool expert".
        baseline_frozen_policy = copy.deepcopy(student_policy)
        for p in baseline_frozen_policy.parameters():
            p.requires_grad_(False)
        baseline_frozen_policy.eval()
        if hasattr(baseline_frozen_policy, "actor_obs_normalizer"):
            baseline_frozen_policy.actor_obs_normalizer.eval()
        print("[DAGGER two-pool] baseline frozen for easy-pool labels.")

        # Resolve failed clip names → indices into the loaded zarr.
        motion_cmd_init = env.unwrapped.command_manager.get_term("motion")
        all_clip_names = [str(n) for n in motion_cmd_init.motion.clip_names]
        with open(args_cli.include_clip_names_file, "r") as f:
            failed_name_set = set(json.load(f))
        failed_indices = [i for i, n in enumerate(all_clip_names) if n in failed_name_set]
        easy_indices = [i for i in range(len(all_clip_names)) if i not in set(failed_indices)]
        if not failed_indices:
            raise RuntimeError(
                f"--two_pool: 0 of {len(failed_name_set)} failed clip names matched "
                f"the {len(all_clip_names)} clips loaded from the zarr. "
                "Check --categories / --zarr_path."
            )
        if not easy_indices:
            raise RuntimeError("--two_pool: 0 easy clips — every loaded clip is in the failed set.")
        print(f"[DAGGER two-pool] failed clips matched: {len(failed_indices)}/{len(failed_name_set)}")
        print(f"[DAGGER two-pool] easy clips:           {len(easy_indices)}")

        failed_indices_t = torch.tensor(failed_indices, dtype=torch.long, device=device)
        easy_indices_t = torch.tensor(easy_indices, dtype=torch.long, device=device)

        # Partition envs. First `failed_pool_size` envs always pull from failed
        # clips; the rest always pull from easy clips.
        failed_pool_size = int(args_cli.num_envs * args_cli.failed_pool_frac)
        easy_pool_size = args_cli.num_envs - failed_pool_size
        if failed_pool_size <= 0 or easy_pool_size <= 0:
            raise ValueError(
                f"--two_pool: bad split — failed_pool_size={failed_pool_size}, easy={easy_pool_size}. "
                "Pick --failed_pool_frac strictly between 0 and 1."
            )
        failed_env_mask = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        failed_env_mask[:failed_pool_size] = True
        easy_env_mask = ~failed_env_mask
        print(f"[DAGGER two-pool] env split: failed=[0,{failed_pool_size}), "
              f"easy=[{failed_pool_size},{args_cli.num_envs})  ({args_cli.failed_pool_frac:.2f} / "
              f"{1 - args_cli.failed_pool_frac:.2f})")

        # Install a custom _resample_command that draws per-pool.
        def _two_pool_resample(self, env_ids):
            if len(env_ids) == 0:
                return
            env_ids_t = torch.as_tensor(env_ids, device=self.device) \
                if not isinstance(env_ids, torch.Tensor) else env_ids
            n = len(env_ids_t)
            is_failed = self._two_pool_failed_mask[env_ids_t]  # [n] bool

            new_clip_ids = torch.empty(n, dtype=torch.long, device=self.device)
            n_failed = int(is_failed.sum().item())
            n_easy = n - n_failed
            if n_failed > 0:
                sampled = self._two_pool_failed_indices[
                    torch.randint(0, len(self._two_pool_failed_indices),
                                  (n_failed,), device=self.device)
                ]
                new_clip_ids[is_failed] = sampled
            if n_easy > 0:
                sampled = self._two_pool_easy_indices[
                    torch.randint(0, len(self._two_pool_easy_indices),
                                  (n_easy,), device=self.device)
                ]
                new_clip_ids[~is_failed] = sampled

            clip_starts = self.motion.clip_start_idx[new_clip_ids]
            clip_ends = self.motion.clip_end_idx[new_clip_ids]
            clip_lens = (clip_ends - clip_starts).clamp_min(1)

            self.clip_ids[env_ids_t] = new_clip_ids
            self.clip_start[env_ids_t] = clip_starts
            self.clip_end[env_ids_t] = clip_ends
            rand_unit = torch.rand(n, device=self.device)
            offsets = (rand_unit * clip_lens.float()).long()
            self.time_steps[env_ids_t] = clip_starts + offsets

            if hasattr(self, "clip_to_category") and hasattr(self, "category_idx"):
                self.category_idx[env_ids_t] = self.clip_to_category[new_clip_ids]
            self._cache_current_frames()

        motion_cmd_init._two_pool_failed_mask = failed_env_mask
        motion_cmd_init._two_pool_failed_indices = failed_indices_t
        motion_cmd_init._two_pool_easy_indices = easy_indices_t
        motion_cmd_init._resample_command = types.MethodType(_two_pool_resample, motion_cmd_init)
        print("[DAGGER two-pool] installed per-pool _resample_command on motion_cmd.")

    # ── Wandb init (independent of runner's logger) ──
    if args_cli.logger == "wandb":
        import wandb
        wandb.init(
            project=args_cli.log_project_name,
            name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}",
            config={
                "n_iters": args_cli.n_iters,
                "rollout_steps": args_cli.rollout_steps,
                "bc_epochs": args_cli.bc_epochs,
                "lr": args_cli.lr,
                "batch_size": args_cli.batch_size,
                "buffer_cap": args_cli.buffer_cap,
                "num_envs": args_cli.num_envs,
                "sampling_mode": args_cli.sampling_mode,
                "student_wandb": args_cli.student_wandb,
                "expert_wandb": args_cli.expert_wandb,
                "include_clip_names_file": args_cli.include_clip_names_file,
                "categories": args_cli.categories,
                "activation": args_cli.activation,
                "decimation": args_cli.decimation,
                "popart": args_cli.popart,
            },
        )

    # ── Buffer ──
    # # OLD: tried to pre-extract a "policy" tensor from obs_init via
    # # `isinstance(obs, dict)`. That missed `tensordict.TensorDict` (NOT a
    # # dict subclass), and even when extraction did work the policy expected
    # # a TensorDict input — `ActorCritic.act()` internally calls
    # # `get_actor_obs(td)` which indexes the TD by group names. Result: a
    # # TensorDict flowed all the way to the buffer and crashed `setitem`.
    # obs_init = env.get_observations()
    # if isinstance(obs_init, tuple):
    #     obs_init = obs_init[0]
    # if isinstance(obs_init, dict):
    #     obs_init = obs_init.get("policy", next(iter(obs_init.values())))
    # obs_init = obs_init.to(device)
    # obs_dim = obs_init.shape[-1]

    # NEW: keep the TensorDict for the policies; route through
    # `student_policy.get_actor_obs` to derive the flat actor obs tensor
    # used for the buffer + BC update.
    obs_init = env.get_observations()
    if isinstance(obs_init, tuple):
        obs_init = obs_init[0]
    obs_init = obs_init.to(device)  # TensorDict.to is supported
    flat_obs_init = student_policy.get_actor_obs(obs_init)  # [num_envs, obs_dim]
    obs_dim = flat_obs_init.shape[-1]
    action_dim = env.action_space.shape[-1]
    print(f"[DAGGER] obs_dim={obs_dim}  action_dim={action_dim}  num_envs={args_cli.num_envs}")
    buffer = GPUReplayBuffer(args_cli.buffer_cap, obs_dim, action_dim, device)

    # ── Optimizer ──
    optimizer = torch.optim.Adam(student_policy.parameters(), lr=args_cli.lr)

    # ── Log dir for checkpoints ──
    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}_dagger",
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[DAGGER] log_dir = {log_dir}")

    # # OLD helper that tried to flatten obs to a tensor — wrong direction.
    # # ActorCritic.act/act_inference expect the TensorDict so they can run
    # # `get_actor_obs(td)` themselves (multi-group concat). We keep `obs` as
    # # a TensorDict throughout the rollout and flatten only when storing
    # # into the buffer.
    # def _obs_to_actor_tensor(obs_out):
    #     if isinstance(obs_out, dict):
    #         return obs_out.get("policy", next(iter(obs_out.values()))).to(device)
    #     if isinstance(obs_out, tuple):
    #         return _obs_to_actor_tensor(obs_out[0])
    #     return obs_out.to(device)

    def _move_obs_to_device(obs_out):
        """Strip outer tuple from env.step / env.reset return; move to device.
        `obs_out` may be a Tensor, TensorDict, or (td, info) tuple."""
        if isinstance(obs_out, tuple):
            obs_out = obs_out[0]
        return obs_out.to(device)

    obs = obs_init  # TensorDict on device
    global_step = 0

    # ── DAgger loop ──
    for it in range(args_cli.n_iters):
        # 1. Roll out student (stochastic) + label with expert (deterministic).
        student_policy.eval()
        rollout_obs_list = []
        rollout_acts_list = []
        n_terminations = 0
        n_steps = 0
        with torch.no_grad():
            for step in range(args_cli.rollout_steps):
                # Student action (sampled, for exploration). `obs` is a
                # TensorDict; the policy unwraps it via get_actor_obs().
                student_action = student_policy.act(obs)

                # Expert label on same obs (deterministic mean). In two-pool
                # mode: route per env — failed pool → specialist, easy pool →
                # frozen baseline. Out of place into `expert_action` of shape
                # [num_envs, action_dim].
                if args_cli.two_pool:
                    expert_action = torch.empty_like(student_action)
                    # TensorDict supports boolean-mask indexing along batch dim.
                    failed_obs_td = obs[failed_env_mask]
                    easy_obs_td = obs[easy_env_mask]
                    if failed_obs_td.batch_size[0] > 0:
                        expert_action[failed_env_mask] = expert_policy.act_inference(failed_obs_td)
                    if easy_obs_td.batch_size[0] > 0:
                        expert_action[easy_env_mask] = baseline_frozen_policy.act_inference(easy_obs_td)
                else:
                    expert_action = expert_policy.act_inference(obs)

                # # OLD: stored the raw TensorDict in the buffer, which then
                # # crashed `self.obs[ptr:end] = obs` since obs wasn't a Tensor.
                # rollout_obs_list.append(obs.detach())

                # Flatten once via get_actor_obs (same op the policy uses
                # internally) so the buffer holds a [num_envs, obs_dim] tensor.
                flat_obs = student_policy.get_actor_obs(obs).detach()
                rollout_obs_list.append(flat_obs)
                rollout_acts_list.append(expert_action.detach())

                step_out = env.step(student_action.to(env.device))
                # rsl_rl wrapper: (obs, rewards, dones, extras)
                next_obs, _rew, dones, _info = step_out
                n_terminations += int(dones.sum().item())
                n_steps += int(dones.numel())
                # Keep next_obs as TensorDict (don't pre-flatten).
                obs = _move_obs_to_device(next_obs)

        all_obs = torch.cat(rollout_obs_list, dim=0)
        all_acts = torch.cat(rollout_acts_list, dim=0)
        buffer.add_batch(all_obs, all_acts)
        term_rate = n_terminations / max(n_steps, 1)
        print(f"[DAGGER iter {it}] rollout: {all_obs.shape[0]} pairs added, "
              f"buffer={buffer.size}/{buffer.capacity}, term_rate={term_rate:.4f}")

        # 2. BC update.
        student_policy.train()
        if hasattr(student_policy, "actor_obs_normalizer"):
            student_policy.actor_obs_normalizer.eval()  # keep normalizer frozen
        if hasattr(student_policy, "critic_obs_normalizer"):
            student_policy.critic_obs_normalizer.eval()
        total_batches = max(buffer.size // args_cli.batch_size, 1)
        last_loss = float("nan")
        loss_sum = 0.0
        loss_count = 0
        for epoch in range(args_cli.bc_epochs):
            for _ in range(total_batches):
                obs_b, expert_acts_b = buffer.sample(args_cli.batch_size)
                # # OLD: act_inference internally calls get_actor_obs, which
                # # indexes a TensorDict by group names — would crash on the
                # # already-flat buffer tensor. Skip get_actor_obs and run
                # # normalizer + actor manually.
                # pred = student_policy.act_inference(obs_b)
                obs_b_norm = student_policy.actor_obs_normalizer(obs_b)
                pred = student_policy.actor(obs_b_norm)
                loss = F.mse_loss(pred, expert_acts_b)
                optimizer.zero_grad()
                loss.backward()
                # Clip grads for stability when buffer is small.
                torch.nn.utils.clip_grad_norm_(student_policy.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = float(loss.item())
                loss_sum += last_loss
                loss_count += 1
        mean_loss = loss_sum / max(loss_count, 1)
        print(f"[DAGGER iter {it}] BC: mean_loss={mean_loss:.6f} (over {loss_count} batches)")

        # 3. Log + checkpoint.
        if args_cli.logger == "wandb":
            import wandb
            wandb.log({
                "iter": it,
                "bc/mean_loss": mean_loss,
                "bc/last_loss": last_loss,
                "rollout/term_rate": term_rate,
                "rollout/n_terminations": n_terminations,
                "rollout/n_done_signals": n_steps,
                "buffer/size": buffer.size,
            }, step=global_step)
            global_step += 1

        if (it + 1) % args_cli.save_every == 0 or it == args_cli.n_iters - 1:
            ckpt_path = os.path.join(log_dir, f"model_{it}.pt")
            _save_checkpoint(ckpt_path, student_policy, it, optimizer=optimizer)

    print(f"[DAGGER] done. Final checkpoint in {log_dir}")

    # ── Final ONNX export for hardware deployment ──
    # Wraps actor + normalizer + motion lookup tables into a single graph the
    # G1 inference stack can load. Same artifact as `utils/exporter.py`
    # produces for PPO runs — `play_bones_clip.py` / hardware paths consume it.
    # NOTE: the exporter reads `cmd.motion.joint_pos` etc., which for the
    # multi-clip popart command is the FULL concatenated frame array across
    # every loaded clip. `time_step_total` becomes that global length, so the
    # ONNX expects an external `time_step` input that's a *global* frame
    # index — deployment must map (clip_name, frame) -> global index using
    # clip_start_idx / clip_end_idx baked into the .npz/zarr.
    try:
        from whole_body_tracking.utils.exporter import (
            export_motion_policy_as_onnx,
            attach_onnx_metadata,
        )
        onnx_filename = f"policy_iter{args_cli.n_iters - 1}.onnx"
        # Policy gets moved to CPU during export; do it last so nothing after
        # this point depends on the GPU copy.
        export_motion_policy_as_onnx(
            env=env.unwrapped,
            actor_critic=student_policy,
            path=log_dir,
            normalizer=getattr(student_policy, "actor_obs_normalizer", None),
            filename=onnx_filename,
            verbose=False,
        )
        run_path_str = args_cli.student_wandb or args_cli.student_local or "dagger"
        attach_onnx_metadata(env.unwrapped, run_path_str, log_dir, filename=onnx_filename)
        onnx_path = os.path.join(log_dir, onnx_filename)
        print(f"[DAGGER] exported ONNX -> {onnx_path}")
        if args_cli.logger == "wandb":
            try:
                import wandb
                if wandb.run is not None:
                    wandb.save(onnx_path, base_path=log_dir, policy="now")
                    print(f"[DAGGER] uploaded ONNX to wandb run")
            except Exception as e:
                print(f"[DAGGER] (non-fatal) wandb.save(onnx) failed: {e}")
    except Exception as e:
        print(f"[DAGGER] (non-fatal) ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

    if args_cli.logger == "wandb":
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
