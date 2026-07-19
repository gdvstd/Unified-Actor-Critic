"""The recovery table as code: each named algorithm is a UnifiedConfig, not a class.

Rollout presets set tau_critic = tau_actor = 1.0: under rollout data the
target is decoupled by freezing y^(lambda) for the epoch loop, not by Polyak
tracking, so the Polyak coefficients are unused (D4).

Constructing ddpg() emits the B1 warning by design — DDPG is the unhedged
point of the deadly-triad region, which is the paper's thesis, not a bug.
"""

from __future__ import annotations

from unified_ac.config import UnifiedConfig


def a2c() -> UnifiedConfig:
    return UnifiedConfig(
        alpha=1.0, sig="v", lam=1.0, num_critics=1,
        anchor="current", grad="score", data="rollout",
        sigma_mode="global",
        tau_critic=1.0, tau_actor=1.0,
    )


def ppo() -> UnifiedConfig:
    return UnifiedConfig(
        alpha=1.0, sig="v", lam=0.95, num_critics=1,
        anchor="old", grad="score", data="rollout",
        sigma_mode="global", ratio_clip=0.2,
        tau_critic=1.0, tau_actor=1.0,
    )


def ddpg() -> UnifiedConfig:
    return UnifiedConfig(
        alpha=0.0, sig="q", lam=0.0, num_critics=1,
        anchor="current", grad="direct", data="replay",
        explore_noise=0.1,
    )


def td3() -> UnifiedConfig:
    return UnifiedConfig(
        alpha=0.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay",
        rho=0.2, clip_c=0.5, policy_delay=2,
        explore_noise=0.1,
    )


def sac() -> UnifiedConfig:
    return UnifiedConfig(
        alpha=1.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay",
        eta=0.2,
        tau_actor=1.0,  # SAC decouples only the critic: theta_bar = theta
    )
