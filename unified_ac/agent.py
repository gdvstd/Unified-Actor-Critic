"""The unified GPI loop: Collect -> Evaluate -> Retrieve -> Improve.

One agent class; every named algorithm is a UnifiedConfig passed to it.
The behavior policy's exploration noise (delta) lives in act() — it shapes
the buffer and appears in no objective.
"""

from __future__ import annotations

import torch
from torch import Tensor

from unified_ac.buffers import ReplayBatch, RolloutBuffer
from unified_ac.config import UnifiedConfig
from unified_ac.distributions import TanhDirac
from unified_ac.losses import actor_loss, critic_loss
from unified_ac.networks import Actor, CriticEnsemble
from unified_ac.targets import PolyakTargets, bootstrap_B


class UnifiedActorCritic:
    def __init__(
        self,
        cfg: UnifiedConfig,
        obs_dim: int,
        act_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> None:
        self.cfg = cfg
        self.actor = Actor(obs_dim, act_dim, cfg, hidden)
        self.critics = CriticEnsemble(obs_dim, act_dim, cfg, hidden)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)
        # replay decouples targets by Polyak copies; rollout by freezing y
        self.targets = (
            PolyakTargets(self.actor, self.critics, cfg)
            if cfg.data == "replay"
            else None
        )
        self._updates = 0

    # ---- Collect: the behavior policy beta ----

    @torch.no_grad()
    def act(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        dist = self.actor.dist(obs)
        if deterministic:
            return dist.mode
        action = dist.rsample()
        if self.cfg.alpha == 0.0 and self.cfg.explore_noise > 0.0:
            noise = torch.randn_like(action) * self.cfg.explore_noise
            action = (action + noise).clamp(-1.0, 1.0)
        return action

    @torch.no_grad()
    def act_with_log_prob(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Collection for rollout data: the anchor's log-prob rides along."""
        dist = self.actor.dist(obs)
        if isinstance(dist, TanhDirac):
            return dist.rsample(), torch.zeros(obs.shape[0])
        action, log_prob, _ = dist.sample_with_log_prob()
        return action, log_prob

    # ---- Evaluate + Retrieve + Improve: replay regime ----

    def update_replay(self, batch: ReplayBatch) -> dict[str, float]:
        cfg = self.cfg
        if cfg.data != "replay":
            raise ValueError("config data regime is 'rollout'; use update_rollout()")
        assert self.targets is not None

        with torch.no_grad():
            b = bootstrap_B(batch.next_obs, cfg, self.targets.actor, self.targets.critics)
            y = batch.reward + cfg.gamma * (1.0 - batch.terminated.float()) * b

        loss, metrics = critic_loss(self.critics, batch.obs, batch.act, y, cfg)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self._updates += 1

        if self._updates % cfg.policy_delay == 0:
            a_loss, a_metrics = actor_loss(self.actor, cfg, batch.obs, critics=self.critics)
            self.actor_opt.zero_grad()
            a_loss.backward()
            self.actor_opt.step()
            self.targets.update(self.actor, self.critics)
            metrics |= a_metrics
        return metrics

    # ---- Evaluate + Retrieve + Improve: rollout regime ----

    def update_rollout(
        self,
        rollout: RolloutBuffer,
        epochs: int = 1,
        minibatch_size: int | None = None,
        normalize_adv: bool = False,
    ) -> dict[str, float]:
        cfg = self.cfg
        if cfg.data != "rollout":
            raise ValueError("config data regime is 'replay'; use update_replay()")
        if cfg.sig == "q":
            raise NotImplementedError(
                "queryable signature on rollout data is an unoccupied interior "
                "cell — deferred to Phase 5"
            )

        rollout.compute_targets(self.critics, cfg)  # frozen for the epoch loop
        metrics: dict[str, float] = {}
        for _ in range(epochs):
            for mb in rollout.minibatches(minibatch_size):
                c_loss, c_metrics = critic_loss(self.critics, mb.obs, None, mb.y, cfg)
                self.critic_opt.zero_grad()
                c_loss.backward()
                self.critic_opt.step()

                psi = mb.adv
                if normalize_adv:
                    # per-minibatch standardization (CleanRL PPO's norm_adv);
                    # a training detail, not an objective dial
                    psi = (psi - psi.mean()) / (psi.std() + 1e-8)
                a_loss, a_metrics = actor_loss(
                    self.actor, cfg, mb.obs,
                    act=mb.act, psi=psi, anchor_log_prob=mb.anchor_log_prob,
                )
                self.actor_opt.zero_grad()
                a_loss.backward()
                self.actor_opt.step()
                metrics = c_metrics | a_metrics
        self._updates += 1
        return metrics
