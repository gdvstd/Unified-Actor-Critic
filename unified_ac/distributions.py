"""Tanh-squashed policy distributions and the deterministic boundary.

The boundary alpha = sigma_min = 0 is singular for every density object,
so TanhDirac raises on log_prob/entropy/kl (C1/C2/C3) while keeping rsample
differentiable — the direct gradient is the one estimator that crosses the
boundary smoothly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

_LOG_2 = math.log(2.0)
_ATANH_EPS = 1e-6


class DensityUndefinedError(RuntimeError):
    """A density object was requested on the deterministic boundary."""


def _tanh_log_det(pre_tanh: Tensor) -> Tensor:
    """log(1 - tanh(u)^2), numerically stable for large |u|."""
    return 2.0 * (_LOG_2 - pre_tanh - F.softplus(-2.0 * pre_tanh))


class TanhGaussian:
    """tanh(N(mu, std^2)) with change-of-variables log-probs, per-dim diagonal."""

    def __init__(self, mu: Tensor, std: Tensor) -> None:
        self.mu = mu
        self.std = std
        self._normal = Normal(mu, std)

    def rsample(self) -> Tensor:
        return torch.tanh(self._normal.rsample())

    def sample_with_log_prob(self) -> tuple[Tensor, Tensor, Tensor]:
        """Preferred path: log-prob from the pre-tanh sample, no atanh roundtrip."""
        pre_tanh = self._normal.rsample()
        return torch.tanh(pre_tanh), self._log_prob_pre_tanh(pre_tanh), pre_tanh

    def log_prob(self, action: Tensor) -> Tensor:
        clamped = action.clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
        return self._log_prob_pre_tanh(torch.atanh(clamped))

    def _log_prob_pre_tanh(self, pre_tanh: Tensor) -> Tensor:
        return (self._normal.log_prob(pre_tanh) - _tanh_log_det(pre_tanh)).sum(-1)

    def entropy_estimate(self, num_samples: int = 1) -> Tensor:
        """Sample-based -E[log pi]; the tanh-squashed entropy has no closed form."""
        pre_tanh = self._normal.rsample((num_samples,))
        return -self._log_prob_pre_tanh(pre_tanh).mean(0)

    def kl(self, other: "TanhGaussian") -> Tensor:
        """Closed-form diagonal-Gaussian KL in pre-tanh space (D9).

        tanh is a bijection, so the KL is invariant under the squash.
        """
        var_ratio = (self.std / other.std) ** 2
        mean_term = ((self.mu - other.mu) / other.std) ** 2
        return 0.5 * (var_ratio + mean_term - 1.0 - var_ratio.log()).sum(-1)

    @property
    def mode(self) -> Tensor:
        return torch.tanh(self.mu)


class TanhDirac:
    """The deterministic boundary: a point mass at tanh(mu)."""

    def __init__(self, mu: Tensor) -> None:
        self.mu = mu

    def rsample(self) -> Tensor:
        return torch.tanh(self.mu)

    sample = rsample

    def log_prob(self, action: Tensor) -> Tensor:
        raise DensityUndefinedError(
            "C1: a Dirac has no density; the score estimator is undefined here"
        )

    def entropy_estimate(self, num_samples: int = 1) -> Tensor:
        raise DensityUndefinedError(
            "C2: entropy diverges on the deterministic boundary"
        )

    def kl(self, other: object) -> Tensor:
        raise DensityUndefinedError(
            "C3: the KL integrand needs a density; none exists on the boundary"
        )

    @property
    def mode(self) -> Tensor:
        return torch.tanh(self.mu)


def make_policy_dist(
    mu: Tensor, sigma: Tensor, alpha: float, sigma_min: float
) -> TanhGaussian | TanhDirac:
    """Sigma_total = alpha^2 diag(sigma^2) + sigma_min^2 I; alpha = sigma_min = 0
    collapses to the point mass (the paper's deterministic boundary)."""
    if alpha == 0.0:
        if sigma_min > 0.0:
            raise ValueError(
                "C5: sigma_min > 0 requires alpha > 0; at alpha = 0 the floor "
                "would promote collection noise into the policy"
            )
        return TanhDirac(mu)
    std = torch.sqrt((alpha * sigma) ** 2 + sigma_min**2)
    return TanhGaussian(mu, std)
