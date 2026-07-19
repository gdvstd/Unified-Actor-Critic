"""Phase 2: the unified actor objective J, dispatched on grad type.

The scientifically load-bearing tests: the analytic DPG check at the
deterministic boundary, and the estimator-agreement test — score and direct
gradients of the same J on the same Psi agree in expectation, which is the
empirical heart of the unification claim.
"""

import pytest
import torch

from unified_ac.config import UnifiedConfig
from unified_ac.distributions import TanhDirac
from unified_ac.losses import actor_loss
from unified_ac.networks import Actor, CriticEnsemble


def cfg_direct(**overrides):
    kwargs = dict(
        alpha=0.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay", explore_noise=0.1,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


def cfg_score(**overrides):
    kwargs = dict(
        alpha=1.0, sig="v", lam=1.0, num_critics=1,
        anchor="current", grad="score", data="rollout", sigma_mode="global",
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


class AnalyticPsi:
    """A stub queryable critic: Psi(s, a) = -||a - target||^2, ensemble dim 1."""

    def __init__(self, target):
        self.target = target

    def __call__(self, obs, act):
        return (-((act - self.target) ** 2).sum(-1)).unsqueeze(0)

    def parameters(self):
        return iter(())


class TestAnalyticDPG:
    def test_autograd_matches_hand_chain_rule_at_the_boundary(self):
        """Linear actor mu = obs @ theta.T, Dirac policy, quadratic Psi:
        dJ/dtheta must equal grad_theta tanh(mu) * grad_a Psi exactly."""
        torch.manual_seed(0)
        theta = torch.randn(2, 3, requires_grad=True)
        obs = torch.randn(5, 3)
        target = torch.tensor([0.1, -0.2])

        mu = obs @ theta.T
        action = TanhDirac(mu).rsample()
        j = (-((action - target) ** 2).sum(-1)).mean()
        j.backward()

        with torch.no_grad():
            d_psi_d_a = -2.0 * (action - target)          # (5, 2)
            d_a_d_mu = 1.0 - action**2                    # tanh'
            hand = (d_psi_d_a * d_a_d_mu).T @ obs / 5.0   # (2, 3)
        assert torch.allclose(theta.grad, hand, atol=1e-6)

    def test_direct_loss_is_finite_and_density_free_for_dirac(self):
        cfg = cfg_direct()
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        loss, metrics = actor_loss(actor, cfg, torch.randn(8, 4), critics=critics)
        loss.backward()
        assert torch.isfinite(loss)
        for p in actor.parameters():
            assert torch.isfinite(p.grad).all()
        assert "entropy" not in metrics and "kl" not in metrics


class TestEstimatorAgreement:
    def test_score_and_direct_gradients_agree_in_expectation(self):
        """Same actor, same analytic Psi, anchor=current: the two estimators
        differentiate the same J and must produce the same gradient."""
        torch.manual_seed(0)
        n = 100_000
        obs = torch.randn(1, 3).repeat(n, 1)
        psi_fn = AnalyticPsi(target=torch.tensor([0.3, -0.1]))

        c_direct = cfg_direct(alpha=1.0, explore_noise=0.0, sigma_mode="global",
                              num_critics=2)
        actor = Actor(obs_dim=3, act_dim=2, cfg=c_direct, hidden=(16, 16))

        loss_d, _ = actor_loss(actor, c_direct, obs, critics=psi_fn)
        grads_direct = torch.autograd.grad(loss_d, list(actor.parameters()))

        c_score = cfg_score(sig="q", lam=0.0, data="replay", num_critics=2)
        with torch.no_grad():
            act = actor.dist(obs).rsample()
            psi = psi_fn(obs, act).squeeze(0)
        loss_s, _ = actor_loss(actor, c_score, obs, act=act, psi=psi)
        grads_score = torch.autograd.grad(loss_s, list(actor.parameters()))

        flat_d = torch.cat([g.flatten() for g in grads_direct])
        flat_s = torch.cat([g.flatten() for g in grads_score])
        rel_err = (flat_d - flat_s).norm() / (flat_d.norm() + 1e-12)
        assert rel_err < 0.1, f"estimators disagree: rel_err={rel_err:.4f}"


class TestScorePath:
    def setup_method(self):
        torch.manual_seed(1)
        self.cfg = cfg_score()
        self.actor = Actor(obs_dim=4, act_dim=2, cfg=self.cfg, hidden=(32, 32))
        self.obs = torch.randn(16, 4)
        with torch.no_grad():
            self.act = self.actor.dist(self.obs).rsample()
        self.psi = torch.randn(16)

    def test_current_anchor_gradient_equals_plain_score_loss(self):
        """C4 as a gradient statement: with anchor=current the ratio is
        identically 1 and J's gradient is the plain score-function gradient."""
        loss, _ = actor_loss(self.actor, self.cfg, self.obs, act=self.act, psi=self.psi)
        grads = torch.autograd.grad(loss, list(self.actor.parameters()))

        log_prob = self.actor.dist(self.obs).log_prob(self.act)
        plain = -(log_prob * self.psi).mean()
        expected = torch.autograd.grad(plain, list(self.actor.parameters()))
        for g, e in zip(grads, expected):
            assert torch.allclose(g, e, atol=1e-6)

    def test_clip_zeroes_gradient_on_the_clipped_branch(self):
        """ratio > 1+eps with positive advantage: the min selects the clipped
        constant branch and the sample contributes zero gradient."""
        cfg = cfg_score(anchor="old", ratio_clip=0.2, lam=0.95)
        with torch.no_grad():
            log_prob_now = self.actor.dist(self.obs).log_prob(self.act)
        anchor_lp = log_prob_now - 1.0  # ratio = e ~ 2.72 > 1.2 everywhere

        loss_pos, _ = actor_loss(
            self.actor, cfg, self.obs,
            act=self.act, psi=torch.ones(16), anchor_log_prob=anchor_lp,
        )
        grads = torch.autograd.grad(loss_pos, list(self.actor.parameters()))
        assert all(g.abs().sum() == 0 for g in grads)

        loss_neg, _ = actor_loss(
            self.actor, cfg, self.obs,
            act=self.act, psi=-torch.ones(16), anchor_log_prob=anchor_lp,
        )
        grads = torch.autograd.grad(loss_neg, list(self.actor.parameters()))
        assert any(g.abs().sum() > 0 for g in grads)

    def test_kl_penalty_realization(self):
        cfg = cfg_score(anchor="old", kl_coef=0.1, lam=0.95)
        with torch.no_grad():
            anchor_lp = self.actor.dist(self.obs).log_prob(self.act) + 0.3
        loss, metrics = actor_loss(
            self.actor, cfg, self.obs,
            act=self.act, psi=self.psi, anchor_log_prob=anchor_lp,
        )
        assert torch.isfinite(loss)
        assert "kl" in metrics

    def test_folk_entropy_bonus_enters_j_only(self):
        with pytest.warns(Warning, match="C9"):
            cfg = cfg_score(folk_entropy_bonus=0.5)
        torch.manual_seed(3)
        loss_folk, metrics = actor_loss(
            self.actor, cfg, self.obs, act=self.act, psi=self.psi
        )
        loss_plain, _ = actor_loss(
            self.actor, self.cfg, self.obs, act=self.act, psi=self.psi
        )
        assert "entropy" in metrics
        assert not torch.allclose(loss_folk, loss_plain)


class TestContractGuards:
    def test_score_requires_logged_actions_and_psi(self):
        cfg = cfg_score()
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        with pytest.raises(ValueError, match="score"):
            actor_loss(actor, cfg, torch.randn(4, 4))

    def test_old_anchor_requires_anchor_log_probs(self):
        cfg = cfg_score(anchor="old", ratio_clip=0.2, lam=0.95)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        with pytest.raises(ValueError, match="anchor"):
            actor_loss(
                actor, cfg, torch.randn(4, 4),
                act=torch.rand(4, 2) * 2 - 1, psi=torch.randn(4),
            )

    def test_direct_requires_queryable_critic(self):
        cfg = cfg_direct()
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        with pytest.raises(ValueError, match="direct"):
            actor_loss(actor, cfg, torch.randn(4, 4))


class TestDirectOldAnchorInterior:
    """direct + anchor=old is a legal interior point no named algorithm
    occupies — valid means definitionally coherent, not known to be useful."""

    def setup_method(self):
        self.cfg = cfg_direct(
            alpha=1.0, anchor="old", kl_coef=0.05, explore_noise=0.0
        )
        self.actor = Actor(obs_dim=4, act_dim=2, cfg=self.cfg, hidden=(32, 32))
        self.critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=self.cfg, hidden=(32, 32))
        self.obs = torch.randn(8, 4)

    def test_kl_penalty_computes_on_logged_actions(self):
        with torch.no_grad():
            act = self.actor.dist(self.obs).rsample()
            anchor_lp = self.actor.dist(self.obs).log_prob(act)
        loss, metrics = actor_loss(
            self.actor, self.cfg, self.obs,
            critics=self.critics, act=act, anchor_log_prob=anchor_lp,
        )
        assert torch.isfinite(loss)
        assert "kl" in metrics

    def test_kl_penalty_without_logged_data_raises(self):
        with pytest.raises(ValueError, match="KL"):
            actor_loss(self.actor, self.cfg, self.obs, critics=self.critics)


class TestSoftDirectPath:
    def test_sac_form_single_sample_entropy(self):
        """eta > 0, direct: J = E[minQ(s, a) - eta * log pi(a|s)] with the
        same sampled action in both terms (the SAC estimator)."""
        cfg = cfg_direct(alpha=1.0, eta=0.5, explore_noise=0.0)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg, hidden=(32, 32))
        obs = torch.randn(8, 4)

        torch.manual_seed(11)
        loss, metrics = actor_loss(actor, cfg, obs, critics=critics)

        torch.manual_seed(11)
        a, logp, _ = actor.dist(obs).sample_with_log_prob()
        expected = -(critics(obs, a).min(dim=0).values - 0.5 * logp).mean()
        assert torch.allclose(loss, expected, atol=1e-6)
        assert "entropy" in metrics
