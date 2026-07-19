"""The two unified objectives: the critic's regression and the actor's J.

Actor sign convention: the paper maximizes J; optimizers minimize, so
actor_loss returns -J. The score and direct paths differentiate the same J —
their agreement in expectation is asserted by the estimator-agreement test.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from unified_ac.config import UnifiedConfig
from unified_ac.networks import Actor, CriticEnsemble


def critic_loss(
    critics: CriticEnsemble,
    obs: Tensor,
    act: Tensor | None,
    y: Tensor,
    cfg: UnifiedConfig,
) -> tuple[Tensor, dict[str, float]]:
    """L(phi) = sum_i MSE(C_i, sg[y]); the action participates iff sig = q."""
    target = y.detach()
    preds = critics(obs, act if cfg.sig == "q" else None)
    loss = sum(F.mse_loss(preds[i], target) for i in range(preds.shape[0]))
    metrics = {
        "critic_loss": float(loss.detach()),
        "q_mean": float(preds.mean().detach()),
        "target_mean": float(target.mean()),
    }
    return loss, metrics


def actor_loss(
    actor: Actor,
    cfg: UnifiedConfig,
    obs: Tensor,
    critics: CriticEnsemble | None = None,
    act: Tensor | None = None,
    psi: Tensor | None = None,
    anchor_log_prob: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """-J(theta). Contract by grad type:

    direct — needs `critics` (queryable Psi); samples its own actions and
    differentiates through the critic.
    score — needs `act` (logged actions) and `psi` (Psi evaluated there);
    `anchor_log_prob` additionally when anchor=old.
    """
    if cfg.grad == "direct":
        if critics is None:
            raise ValueError("the direct path queries the critic: pass `critics`")
        return _direct(actor, cfg, obs, critics, act, anchor_log_prob)
    if act is None or psi is None:
        raise ValueError("the score path needs logged actions and psi")
    if cfg.anchor == "old" and anchor_log_prob is None:
        raise ValueError("anchor=old needs the anchor's log-probs")
    return _score(actor, cfg, obs, act, psi, anchor_log_prob)


def _entropy_coef(cfg: UnifiedConfig) -> float:
    return (cfg.eta + cfg.folk_entropy_bonus) if cfg.alpha > 0.0 else 0.0


def _direct(actor, cfg, obs, critics, act, anchor_log_prob):
    dist = actor.dist(obs)
    ent_coef = _entropy_coef(cfg)
    log_prob = None
    if ent_coef > 0.0:
        # SAC estimator: one sample serves both Psi and the entropy term
        action, log_prob, _ = dist.sample_with_log_prob()
    else:
        action = dist.rsample()

    j = critics(obs, action).min(dim=0).values
    if log_prob is not None:
        j = j - ent_coef * log_prob

    metrics = {"psi_mean": float(j.mean().detach())}
    j = j.mean()
    if log_prob is not None:
        metrics["entropy"] = float(-log_prob.mean().detach())
    if cfg.anchor == "old" and cfg.kl_coef > 0.0:
        if act is None or anchor_log_prob is None:
            raise ValueError("the KL penalty needs logged actions and anchor log-probs")
        kl = (anchor_log_prob - dist.log_prob(act)).mean()
        j = j - cfg.kl_coef * kl
        metrics["kl"] = float(kl.detach())
    metrics["actor_loss"] = float(-j.detach())
    return -j, metrics


def _score(actor, cfg, obs, act, psi, anchor_log_prob):
    dist = actor.dist(obs)
    log_prob = dist.log_prob(act)
    psi = psi.detach()

    if cfg.anchor == "old":
        ratio = (log_prob - anchor_log_prob).exp()
    else:
        # value 1, gradient = grad log pi: the C4 identities, implemented
        ratio = (log_prob - log_prob.detach()).exp()

    metrics: dict[str, float] = {"ratio_mean": float(ratio.mean().detach())}
    if cfg.ratio_clip is not None and cfg.anchor == "old":
        clipped = ratio.clamp(1.0 - cfg.ratio_clip, 1.0 + cfg.ratio_clip)
        surrogate = torch.min(ratio * psi, clipped * psi)
        metrics["clip_frac"] = float((ratio * psi > clipped * psi).float().mean())
    else:
        surrogate = ratio * psi

    j = surrogate.mean()
    ent_coef = _entropy_coef(cfg)
    if ent_coef > 0.0:
        entropy = dist.entropy_estimate().mean()
        j = j + ent_coef * entropy
        metrics["entropy"] = float(entropy.detach())
    if cfg.anchor == "old" and cfg.kl_coef > 0.0:
        kl = (anchor_log_prob - log_prob).mean()
        j = j - cfg.kl_coef * kl
        metrics["kl"] = float(kl.detach())
    metrics["actor_loss"] = float(-j.detach())
    return -j, metrics
