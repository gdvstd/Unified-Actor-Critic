"""Phase 1: the lambda-return, the bootstrap evaluation B, Polyak decoupling,
and the unified critic loss — hand-computed fixtures throughout.

Convention: tensors are time-major (T, ...). bootstrap[t] is B(s_{t+1}),
already evaluated at the *real* next observation (final_observation for
truncated steps). The rollout end is a truncation unless terminated (D3).
"""

import copy

import pytest
import torch

from unified_ac.config import TypeBWarning, UnifiedConfig
from unified_ac.losses import critic_loss
from unified_ac.networks import Actor, CriticEnsemble
from unified_ac.targets import PolyakTargets, bootstrap_B, lambda_return


GAMMA = 0.9


def _no_dones(t=3):
    return torch.zeros(t, dtype=torch.bool), torch.zeros(t, dtype=torch.bool)


class TestLambdaReturn:
    def setup_method(self):
        self.r = torch.tensor([1.0, 2.0, 3.0])
        self.b = torch.tensor([10.0, 20.0, 30.0])

    def test_lam0_is_one_step_backup(self):
        term, trunc = _no_dones()
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=0.0)
        assert torch.allclose(y, torch.tensor([10.0, 20.0, 30.0]))

    def test_lam1_terminated_episode_is_pure_monte_carlo(self):
        # termination at the last step: no bootstrap anywhere in the episode
        term = torch.tensor([False, False, True])
        trunc = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=1.0)
        assert torch.allclose(y, torch.tensor([5.23, 4.7, 3.0]))

    def test_lam1_rollout_end_is_a_truncation(self):
        # same episode cut by the rollout boundary: the tail bootstraps
        term, trunc = _no_dones()
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=1.0)
        y2 = 3.0 + GAMMA * 30.0
        y1 = 2.0 + GAMMA * y2
        y0 = 1.0 + GAMMA * y1
        assert torch.allclose(y, torch.tensor([y0, y1, y2]))

    def test_interior_lambda_hand_computed(self):
        term, trunc = _no_dones()
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=0.5)
        assert torch.allclose(y, torch.tensor([16.525, 24.5, 30.0]))

    def test_mid_rollout_termination_blocks_bootstrap(self):
        term = torch.tensor([False, True, False])
        trunc = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=1.0)
        # y1 = r1 exactly; y0 chains onto it; y2 opens a fresh episode
        assert torch.allclose(y, torch.tensor([1.0 + GAMMA * 2.0, 2.0, 30.0]))

    def test_mid_rollout_truncation_bootstraps(self):
        term = torch.zeros(3, dtype=torch.bool)
        trunc = torch.tensor([False, True, False])
        y = lambda_return(self.r, self.b, term, trunc, GAMMA, lam=1.0)
        y1 = 2.0 + GAMMA * 20.0  # truncated: bootstrap at final_observation
        assert torch.allclose(y, torch.tensor([1.0 + GAMMA * y1, y1, 30.0]))

    def test_shape_mismatch_raises(self):
        term, trunc = _no_dones()
        with pytest.raises(ValueError, match="share shape"):
            lambda_return(self.r, self.b[:2], term, trunc, GAMMA, lam=0.5)

    def test_batched_shape(self):
        r = self.r.unsqueeze(-1).repeat(1, 4)
        b = self.b.unsqueeze(-1).repeat(1, 4)
        term = torch.zeros(3, 4, dtype=torch.bool)
        trunc = torch.zeros(3, 4, dtype=torch.bool)
        y = lambda_return(r, b, term, trunc, GAMMA, lam=0.5)
        assert y.shape == (3, 4)
        assert torch.allclose(y[:, 0], torch.tensor([16.525, 24.5, 30.0]))


def q_cfg(**overrides):
    kwargs = dict(
        alpha=0.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay", explore_noise=0.1,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


class TestBootstrapB:
    def setup_method(self):
        torch.manual_seed(0)
        self.obs = torch.randn(8, 4)

    def _nets(self, cfg, act_dim=2):
        actor = Actor(obs_dim=4, act_dim=act_dim, cfg=cfg)
        critics = CriticEnsemble(obs_dim=4, act_dim=act_dim, cfg=cfg)
        return actor, critics

    def test_sig_v_is_bare_state_value(self):
        cfg = UnifiedConfig(
            alpha=1.0, sig="v", lam=1.0, num_critics=1,
            anchor="current", grad="score", data="rollout", sigma_mode="global",
        )
        actor, critics = self._nets(cfg)
        b = bootstrap_B(self.obs, cfg, actor, critics)
        assert torch.allclose(b, critics(self.obs).squeeze(0))

    def test_sig_q_deterministic_is_min_over_target_twins(self):
        cfg = q_cfg()
        actor, critics = self._nets(cfg)
        b = bootstrap_B(self.obs, cfg, actor, critics)
        a = actor.dist(self.obs).mode
        expected = critics(self.obs, a).min(dim=0).values
        assert torch.allclose(b, expected)

    def test_smoothing_perturbation_is_clipped(self):
        # rho >> c: the clip must bound the target action's deviation
        cfg = q_cfg(rho=10.0, clip_c=0.05)
        actor, critics = self._nets(cfg)
        captured = {}

        original = CriticEnsemble.forward

        def spy(self_, obs, act=None):
            captured["act"] = act
            return original(self_, obs, act)

        CriticEnsemble.forward = spy
        try:
            bootstrap_B(self.obs, cfg, actor, critics)
        finally:
            CriticEnsemble.forward = original
        clean = actor.dist(self.obs).mode
        deviation = (captured["act"] - clean).abs()
        assert (deviation <= 0.05 + 1e-6).all()
        assert captured["act"].abs().max() <= 1.0

    def test_soft_term_present_iff_entropy_active(self):
        cfg_soft = q_cfg(alpha=1.0, eta=0.5, explore_noise=0.0)
        actor, critics = self._nets(cfg_soft)

        torch.manual_seed(42)
        b_soft = bootstrap_B(self.obs, cfg_soft, actor, critics)

        # manual mirror with the same sample path
        torch.manual_seed(42)
        a, logp, _ = actor.dist(self.obs).sample_with_log_prob()
        expected = critics(self.obs, a).min(dim=0).values - 0.5 * logp
        assert torch.allclose(b_soft, expected, atol=1e-6)

    def test_soft_term_uses_unperturbed_action_for_log_prob(self):
        # smoothing noise belongs to the critic, not the policy being measured;
        # rho > 0 with alpha > 0 is deliberately the B3 configuration
        with pytest.warns(TypeBWarning, match="B3"):
            cfg = q_cfg(alpha=1.0, eta=0.5, rho=0.3, clip_c=0.5, explore_noise=0.0)
        actor, critics = self._nets(cfg)

        torch.manual_seed(7)
        b = bootstrap_B(self.obs, cfg, actor, critics)

        torch.manual_seed(7)
        a, logp, _ = actor.dist(self.obs).sample_with_log_prob()
        noise = (torch.randn_like(a) * cfg.rho).clamp(-cfg.clip_c, cfg.clip_c)
        a_perturbed = (a + noise).clamp(-1.0, 1.0)
        expected = critics(self.obs, a_perturbed).min(dim=0).values - 0.5 * logp
        assert torch.allclose(b, expected, atol=1e-6)


class TestPolyakTargets:
    def test_update_is_exact_convex_combination(self):
        cfg = q_cfg(tau_critic=0.3, tau_actor=0.7)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg)
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        targets = PolyakTargets(actor, critics, cfg)

        # nudge online params, then check the exact Polyak formula
        with torch.no_grad():
            for p in list(actor.parameters()) + list(critics.parameters()):
                p.add_(1.0)
        before = [p.clone() for p in targets.critics.parameters()]
        online = [p.clone() for p in critics.parameters()]
        targets.update(actor, critics)
        for tp, old, new in zip(targets.critics.parameters(), before, online):
            assert torch.allclose(tp, 0.3 * new + 0.7 * old, atol=1e-7)

    def test_targets_start_as_copies_and_require_no_grad(self):
        cfg = q_cfg()
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg)
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        targets = PolyakTargets(actor, critics, cfg)
        for tp, p in zip(targets.critics.parameters(), critics.parameters()):
            assert torch.equal(tp, p)
            assert not tp.requires_grad

    def test_tau_one_tracks_online_exactly(self):
        # SAC's actor: theta_bar = theta at every update
        cfg = q_cfg(alpha=1.0, eta=0.2, tau_actor=1.0, explore_noise=0.0)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg)
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        targets = PolyakTargets(actor, critics, cfg)
        with torch.no_grad():
            for p in actor.parameters():
                p.add_(1.0)
        targets.update(actor, critics)
        for tp, p in zip(targets.actor.parameters(), actor.parameters()):
            assert torch.equal(tp, p)


class TestCriticLoss:
    def test_matches_manual_mse_sum_over_ensemble(self):
        cfg = q_cfg()
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        obs, act = torch.randn(16, 4), torch.rand(16, 2) * 2 - 1
        y = torch.randn(16)
        loss, metrics = critic_loss(critics, obs, act, y, cfg)
        preds = critics(obs, act)
        expected = sum(
            torch.nn.functional.mse_loss(preds[i], y) for i in range(2)
        )
        assert torch.allclose(loss, expected)
        assert "critic_loss" in metrics

    def test_target_is_stop_gradient(self):
        cfg = q_cfg()
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        obs, act = torch.randn(8, 4), torch.rand(8, 2) * 2 - 1
        y = torch.randn(8, requires_grad=True)
        loss, _ = critic_loss(critics, obs, act, y, cfg)
        loss.backward()
        assert y.grad is None or y.grad.abs().sum() == 0

    def test_sig_v_loss_ignores_actions(self):
        cfg = UnifiedConfig(
            alpha=1.0, sig="v", lam=1.0, num_critics=1,
            anchor="current", grad="score", data="rollout", sigma_mode="global",
        )
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg)
        obs = torch.randn(8, 4)
        y = torch.randn(8)
        loss, _ = critic_loss(critics, obs, None, y, cfg)
        expected = torch.nn.functional.mse_loss(critics(obs)[0], y)
        assert torch.allclose(loss, expected)


class TestHedgesAbsentAtLambdaOne:
    def test_completed_episode_target_contains_no_critic_estimate(self):
        """lam=1 + terminated: y is the observed return; every hedge and the
        target parameters themselves drop out of the loss graph."""
        r = torch.tensor([1.0, 2.0, 3.0])
        b = torch.full((3,), 999.0, requires_grad=True)  # poisoned bootstrap
        term = torch.tensor([False, False, True])
        trunc = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(r, b, term, trunc, GAMMA, lam=1.0)
        assert torch.allclose(y, torch.tensor([5.23, 4.7, 3.0]))
        y.sum().backward()
        assert b.grad.abs().sum() == 0
