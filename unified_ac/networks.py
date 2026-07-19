"""Network shells: one actor, one critic ensemble, both shaped by the config.

The actor owns mu (and a sigma head only when the policy is stochastic); the
critic ensemble holds M networks whose input signature follows cfg.sig. All
algorithm-specific behavior lives in the objectives, not here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from unified_ac.config import UnifiedConfig
from unified_ac.distributions import TanhDirac, TanhGaussian, make_policy_dist

_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: UnifiedConfig,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.alpha = cfg.alpha
        self.sigma_min = cfg.sigma_min
        self.sigma_mode = cfg.sigma_mode
        self.trunk = _mlp(obs_dim, hidden[-1], hidden[:-1])
        self.mu_head = nn.Linear(hidden[-1], act_dim)
        if self.alpha > 0.0 and self.sigma_mode == "state":
            self.log_std_head = nn.Linear(hidden[-1], act_dim)
        elif self.alpha > 0.0:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: Tensor) -> TanhGaussian | TanhDirac:
        features = torch.relu(self.trunk(obs))
        mu = self.mu_head(features)
        if self.alpha == 0.0:
            sigma = torch.ones_like(mu)
        elif self.sigma_mode == "state":
            log_std = self.log_std_head(features).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
            sigma = log_std.exp()
        else:
            sigma = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp().expand_as(mu)
        return make_policy_dist(mu, sigma, self.alpha, self.sigma_min)


class CriticEnsemble(nn.Module):
    """M critics; input signature (s, a) vs (s) follows cfg.sig."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: UnifiedConfig,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.sig = cfg.sig
        in_dim = obs_dim + act_dim if cfg.sig == "q" else obs_dim
        self.members = nn.ModuleList(
            _mlp(in_dim, 1, hidden) for _ in range(cfg.num_critics)
        )

    def forward(self, obs: Tensor, act: Tensor | None = None) -> Tensor:
        if self.sig == "q":
            if act is None:
                raise ValueError("C6: the Q signature requires an action argument")
            x = torch.cat([obs, act], dim=-1)
        else:
            x = obs
        return torch.stack([member(x).squeeze(-1) for member in self.members])
