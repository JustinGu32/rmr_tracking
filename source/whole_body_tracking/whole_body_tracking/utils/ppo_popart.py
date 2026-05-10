"""PPOPopArt — PPO with multi-head value normalization.

Overrides exactly two methods of RSL-RL's `PPO`:

- `compute_returns`: standard scalar GAE in unnormalized space (via super),
  then masked PopArt stats update on per-category returns. The ART
  correction inside `update_masked` keeps the unnormalized predictions
  continuous across the stats shift, so rollout-recorded values remain
  consistent.

- `update`: copy of the vanilla PPO loop with three localized swaps:
    1. Critic forward via `evaluate_columns` → gather active column.
    2. Critic loss in normalized space, masked to one column per sample.
    3. Per-category advantage whitening over the minibatch.

`use_clipped_value_loss` is forced to False here (clipping in normalized
space would require snapshotting (μ_k, σ_k) at rollout time and threading
them through the buffer — the PopArt paper does not use clipping). The
agent cfg also sets the field to False; this guard is belt-and-suspenders.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO


class PPOPopArt(PPO):
    """Drop-in PPO with PopArt-style normalization for a multi-head critic."""

    def compute_returns(self, obs):
        # Scalar GAE / returns in unnormalized space, just like vanilla PPO.
        super().compute_returns(obs)

        # Masked PopArt stats update on the full rollout's returns.
        cat_obs_group = self.policy.category_obs_group
        # storage.observations is a TensorDict over groups; per-step shape [T, N, dim].
        cat_buf = self.storage.observations[cat_obs_group]               # [T, N, 1]
        ret_buf = self.storage.returns                                   # [T, N, 1]
        val_buf = self.storage.values                                    # [T, N, 1]
        rew_buf = self.storage.rewards                                   # [T, N, 1]

        cat_flat = cat_buf.reshape(-1).long()                            # [T*N]
        ret_flat = ret_buf.reshape(-1)                                   # [T*N]
        val_flat = val_buf.reshape(-1)                                   # [T*N]
        rew_flat = rew_buf.reshape(-1)                                   # [T*N]

        K = self.policy.num_categories
        self.policy.value_normalizer.update_masked(ret_flat, cat_flat, K=K)

        # ── Per-category diagnostics for wandb logging ────────────────────
        # Computed AFTER update_masked so (μ_k, σ_k, debias) reflect the
        # state used by the upcoming PPO update step.
        # Diagnostic keys use human-readable category names ("walk", "jog", …)
        # when available, falling back to "cat_0", "cat_1" otherwise.
        names = self.policy.category_names

        self.popart_diagnostics = {}
        with torch.no_grad():
            mean_k, std_k = self.policy.value_normalizer.mean_std()      # [K], [K]
            self.popart_diagnostics["debias"] = float(
                self.policy.value_normalizer.debias.item()
            )
            total_n = max(1, cat_flat.numel())
            for k in range(K):
                mask = cat_flat == k
                n_k = int(mask.sum().item())
                label = names[k]
                self.popart_diagnostics[f"mean/{label}"] = float(mean_k[k].item())
                self.popart_diagnostics[f"sigma/{label}"] = float(std_k[k].item())
                self.popart_diagnostics[f"fraction/{label}"] = n_k / total_n
                if n_k > 0:
                    ret_k = ret_flat[mask]
                    val_k = val_flat[mask]
                    rew_k = rew_flat[mask]
                    self.popart_diagnostics[f"return_mean/{label}"] = float(ret_k.mean().item())
                    self.popart_diagnostics[f"value_mean/{label}"] = float(val_k.mean().item())
                    # Mean per-step reward (undiscounted) for this category — the
                    # one extra signal over standard PopArt diagnostics. Lets you
                    # see "walk reward improving while jog reward stalls" without
                    # the full Episode_Reward/* breakdown chart explosion.
                    self.popart_diagnostics[f"reward_per_step/{label}"] = float(rew_k.mean().item())
                    if n_k > 1:
                        self.popart_diagnostics[f"return_std/{label}"] = float(ret_k.std().item())
                        # Explained variance: 1 - Var(returns - values) / Var(returns).
                        # Closer to 1 = critic predicts returns well for this category.
                        var_returns = ret_k.var().item()
                        if var_returns > 1e-8:
                            var_residual = (ret_k - val_k).var().item()
                            self.popart_diagnostics[f"explained_variance/{label}"] = float(
                                1.0 - var_residual / var_returns
                            )

    def update(self):  # noqa: C901 — mirrors PPO.update's structure intentionally
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        # PopArt path does not use RND or symmetry — guard so we don't silently mix.
        if self.rnd is not None:
            raise NotImplementedError("PPOPopArt does not currently support RND.")
        if self.symmetry is not None:
            raise NotImplementedError("PPOPopArt does not currently support symmetry.")

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        cat_obs_group = self.policy.category_obs_group
        K = self.policy.num_categories

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            original_batch_size = obs_batch.batch_size[0]

            # ── Per-category advantage whitening (replaces vanilla advantage norm) ──
            cat_batch = obs_batch[cat_obs_group]
            if cat_batch.dim() == 2:
                cat_batch = cat_batch.squeeze(-1)
            cat_batch = cat_batch.long()                                 # [B]

            adv = advantages_batch.clone()                               # [B, 1]
            with torch.no_grad():
                for k in range(K):
                    m = cat_batch == k
                    if m.sum().item() >= 2:
                        slab = adv[m]
                        adv[m] = (slab - slab.mean()) / (slab.std() + 1e-8)
                    else:
                        # Single-sample category: re-center to zero, leave scale.
                        adv[m] = adv[m] - adv[m].mean()
            advantages_batch = adv

            # ── Critic forward: full K-vector, then gather active column. ─────────
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_K_batch = self.policy.evaluate_columns(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )                                                            # [B, K] (normalized)

            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # ── Adaptive KL learning-rate schedule (unchanged from vanilla PPO) ──
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # ── Surrogate loss (unchanged structure, with whitened advantages). ──
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ── Critic loss in normalized space, gathered to active category. ────
            value_norm_active = value_K_batch.gather(
                -1, cat_batch.unsqueeze(-1)
            )                                                            # [B, 1]

            with torch.no_grad():
                pop_mean, pop_std = self.policy.value_normalizer.mean_std()  # [K], [K]
                mu_per_sample = pop_mean[cat_batch].unsqueeze(-1)           # [B, 1]
                sigma_per_sample = pop_std[cat_batch].unsqueeze(-1)         # [B, 1]
                target_norm = (returns_batch - mu_per_sample) / sigma_per_sample  # [B, 1]

            # Forced unclipped value loss (decision §1 in the implementation doc).
            value_loss = (value_norm_active - target_norm).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
