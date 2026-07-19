"""Phase 0: TanhGaussian numerics and the deterministic boundary.

The boundary alpha = sigma_min = 0 is singular for every density object
(C1/C2/C3); the distribution layer enforces that by raising, while rsample
stays smooth so the direct gradient can cross the boundary.
"""

import math

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from unified_ac.distributions import (
    DensityUndefinedError,
    TanhDirac,
    TanhGaussian,
    make_policy_dist,
)


def _analytic_log_prob(mu, std, pre_tanh):
    """Normal log-prob minus the tanh change-of-variables correction."""
    base = Normal(mu, std).log_prob(pre_tanh)
    correction = 2.0 * (
        math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
    )
    return (base - correction).sum(-1)


class TestTanhGaussian:
    def setup_method(self):
        torch.manual_seed(0)
        self.mu = torch.tensor([[0.3, -0.7], [1.2, 0.0]])
        self.std = torch.tensor([[0.5, 1.0], [0.2, 0.8]])

    def test_sample_and_log_prob_matches_analytic(self):
        dist = TanhGaussian(self.mu, self.std)
        action, log_prob, pre_tanh = dist.sample_with_log_prob()
        assert torch.allclose(action, torch.tanh(pre_tanh))
        expected = _analytic_log_prob(self.mu, self.std, pre_tanh)
        assert torch.allclose(log_prob, expected, atol=1e-6)

    def test_log_prob_from_action_roundtrips(self):
        dist = TanhGaussian(self.mu, self.std)
        action, log_prob, _ = dist.sample_with_log_prob()
        assert torch.allclose(dist.log_prob(action), log_prob, atol=1e-4)

    def test_rsample_carries_gradients(self):
        mu = self.mu.clone().requires_grad_(True)
        log_std = torch.zeros_like(self.mu).requires_grad_(True)
        action = TanhGaussian(mu, log_std.exp()).rsample()
        action.sum().backward()
        assert mu.grad is not None and mu.grad.abs().sum() > 0
        assert log_std.grad is not None and log_std.grad.abs().sum() > 0

    def test_actions_are_bounded(self):
        dist = TanhGaussian(self.mu, self.std * 10.0)
        action = dist.rsample()
        assert action.abs().max() < 1.0

    def test_kl_matches_torch_closed_form(self):
        p = TanhGaussian(self.mu, self.std)
        q = TanhGaussian(self.mu + 0.5, self.std * 2.0)
        expected = kl_divergence(
            Normal(self.mu, self.std), Normal(self.mu + 0.5, self.std * 2.0)
        ).sum(-1)
        assert torch.allclose(p.kl(q), expected, atol=1e-6)

    def test_entropy_estimate_increases_with_scale(self):
        torch.manual_seed(1)
        narrow = TanhGaussian(self.mu, self.std * 0.01)
        wide = TanhGaussian(self.mu, self.std)
        n = 512
        assert (
            wide.entropy_estimate(n).mean() > narrow.entropy_estimate(n).mean()
        )

    def test_mode_is_tanh_mu(self):
        dist = TanhGaussian(self.mu, self.std)
        assert torch.allclose(dist.mode, torch.tanh(self.mu))


class TestTanhDirac:
    def setup_method(self):
        self.mu = torch.tensor([[0.3, -0.7]])

    def test_rsample_is_tanh_mu_and_differentiable(self):
        mu = self.mu.clone().requires_grad_(True)
        action = TanhDirac(mu).rsample()
        assert torch.allclose(action, torch.tanh(mu))
        action.sum().backward()
        assert mu.grad is not None and mu.grad.abs().sum() > 0

    def test_density_objects_raise(self):
        dist = TanhDirac(self.mu)
        with pytest.raises(DensityUndefinedError, match="C1"):
            dist.log_prob(torch.tanh(self.mu))
        with pytest.raises(DensityUndefinedError, match="C2"):
            dist.entropy_estimate()
        with pytest.raises(DensityUndefinedError, match="C3"):
            dist.kl(dist)

    def test_mode_is_tanh_mu(self):
        assert torch.allclose(TanhDirac(self.mu).mode, torch.tanh(self.mu))


class TestFactory:
    """make_policy_dist implements Sigma = alpha^2 diag(sigma^2) + sigma_min^2 I."""

    def setup_method(self):
        self.mu = torch.zeros(2, 3)
        self.sigma = torch.ones(2, 3)

    def test_boundary_returns_dirac(self):
        dist = make_policy_dist(self.mu, self.sigma, alpha=0.0, sigma_min=0.0)
        assert isinstance(dist, TanhDirac)

    def test_interior_returns_gaussian_with_combined_std(self):
        dist = make_policy_dist(self.mu, self.sigma, alpha=2.0, sigma_min=0.5)
        assert isinstance(dist, TanhGaussian)
        expected = math.sqrt(2.0**2 * 1.0 + 0.5**2)
        assert torch.allclose(dist.std, torch.full_like(self.sigma, expected))

    def test_floor_without_learned_scale_is_rejected(self):
        # config-level C5; the factory refuses defensively too
        with pytest.raises(ValueError, match="C5"):
            make_policy_dist(self.mu, self.sigma, alpha=0.0, sigma_min=0.1)

    def test_floor_bounds_std_below(self):
        tiny = torch.full_like(self.sigma, 1e-8)
        dist = make_policy_dist(self.mu, tiny, alpha=1.0, sigma_min=0.3)
        assert (dist.std >= 0.3 - 1e-6).all()
