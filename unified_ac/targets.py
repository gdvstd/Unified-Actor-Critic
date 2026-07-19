"""The unified critic target: lambda-return, bootstrap evaluation B, and the
Polyak half of target decoupling.

Convention: rollout tensors are time-major (T, ...); bootstrap[t] is B(s_{t+1})
evaluated at the real next observation (final_observation for truncated steps).
The rollout end is a truncation unless terminated (D3): the (1 - done) mask
uses `terminated` only, and truncated boundaries always append B.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from unified_ac.config import UnifiedConfig
from unified_ac.networks import Actor, CriticEnsemble


def lambda_return(
    rewards: Tensor,
    bootstrap: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float,
    lam: float,
) -> Tensor:
    """y_t^(lambda) by backward recursion, truncation-aware.

    Within an episode: y_t = r_t + gamma * ((1-lam) * B_t + lam * y_{t+1}).
    Termination zeroes the tail; truncation replaces it with B_t entirely.
    """
    if not (rewards.shape == bootstrap.shape == terminated.shape == truncated.shape):
        raise ValueError("rewards, bootstrap, terminated, truncated must share shape")
    horizon = rewards.shape[0]
    cont = 1.0 - terminated.float()
    cut = truncated.float()
    y = torch.zeros_like(rewards)
    y[horizon - 1] = rewards[horizon - 1] + gamma * cont[horizon - 1] * bootstrap[horizon - 1]
    for t in reversed(range(horizon - 1)):
        blended = cut[t] * bootstrap[t] + (1.0 - cut[t]) * (
            (1.0 - lam) * bootstrap[t] + lam * y[t + 1]
        )
        y[t] = rewards[t] + gamma * cont[t] * blended
    return y


def bootstrap_B(
    next_obs: Tensor,
    cfg: UnifiedConfig,
    actor_target: Actor,
    critic_target: CriticEnsemble,
) -> Tensor:
    """The bootstrap evaluation B(s): the one place every hedge lives.

    sig=v: bare V(s) — the precomputed signature receives no actor queries,
    so it carries no defenses. sig=q: min over target twins at the target
    policy's action, smoothed by clip(rho * zeta, +-c), minus the soft term
    at the *unperturbed* action when entropy is active.
    """
    if cfg.sig == "v":
        return critic_target(next_obs).min(dim=0).values

    dist = actor_target.dist(next_obs)
    log_prob = None
    if cfg.alpha > 0.0 and cfg.eta > 0.0:
        action, log_prob, _ = dist.sample_with_log_prob()
    else:
        action = dist.rsample()

    if cfg.rho > 0.0:
        noise = (torch.randn_like(action) * cfg.rho).clamp(-cfg.clip_c, cfg.clip_c)
        query_action = (action + noise).clamp(-1.0, 1.0)
    else:
        query_action = action

    value = critic_target(next_obs, query_action).min(dim=0).values
    if log_prob is not None:
        value = value - cfg.eta * log_prob
    return value


class PolyakTargets:
    """Replay-side target decoupling: slow copies of actor and critics.

    tau = 1 degenerates to exact tracking (SAC's actor); the rollout side
    realizes the same decoupling by freezing y for the epoch loop instead.
    """

    def __init__(self, actor: Actor, critics: CriticEnsemble, cfg: UnifiedConfig) -> None:
        self.tau_actor = cfg.tau_actor
        self.tau_critic = cfg.tau_critic
        self.actor = _frozen_copy(actor)
        self.critics = _frozen_copy(critics)

    def update(self, actor: Actor, critics: CriticEnsemble) -> None:
        _polyak(self.actor, actor, self.tau_actor)
        _polyak(self.critics, critics, self.tau_critic)


def _frozen_copy(module: nn.Module) -> nn.Module:
    target = copy.deepcopy(module)
    for param in target.parameters():
        param.requires_grad_(False)
    return target


def _polyak(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.mul_(1.0 - tau).add_(tau * p)
