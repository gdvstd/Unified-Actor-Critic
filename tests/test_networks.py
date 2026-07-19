"""Phase 1: network shells — shapes, sigma modes, and the deterministic actor."""

import torch

from unified_ac.config import UnifiedConfig
from unified_ac.distributions import TanhDirac, TanhGaussian
from unified_ac.networks import Actor, CriticEnsemble


def cfg_q(**overrides):
    kwargs = dict(
        alpha=0.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay", explore_noise=0.1,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


def cfg_v(**overrides):
    kwargs = dict(
        alpha=1.0, sig="v", lam=0.95, num_critics=1,
        anchor="old", grad="score", data="rollout",
        sigma_mode="global", ratio_clip=0.2,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


class TestCriticEnsemble:
    def test_q_signature_shapes(self):
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg_q())
        out = critics(torch.randn(8, 4), torch.rand(8, 2))
        assert out.shape == (2, 8)

    def test_v_signature_shapes_and_no_action_input(self):
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg_v())
        out = critics(torch.randn(8, 4))
        assert out.shape == (1, 8)

    def test_q_signature_without_action_raises(self):
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg_q())
        with torch.no_grad():
            try:
                critics(torch.randn(8, 4))
                raise AssertionError("expected ValueError")
            except ValueError as err:
                assert "C6" in str(err)

    def test_ensemble_members_are_independent(self):
        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg_q())
        out = critics(torch.randn(8, 4), torch.rand(8, 2))
        assert not torch.allclose(out[0], out[1])


class TestActor:
    def test_deterministic_actor_yields_dirac_and_has_no_sigma_params(self):
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg_q())
        dist = actor.dist(torch.randn(8, 4))
        assert isinstance(dist, TanhDirac)
        assert not any("sigma" in n or "log_std" in n for n, _ in actor.named_parameters())

    def test_global_sigma_mode_has_state_independent_log_std(self):
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg_v())
        dist = actor.dist(torch.randn(8, 4))
        assert isinstance(dist, TanhGaussian)
        # same sigma for every state
        assert torch.allclose(dist.std[0], dist.std[3])

    def test_state_sigma_mode_is_state_dependent(self):
        cfg = cfg_q(alpha=1.0, eta=0.2, sigma_mode="state", explore_noise=0.0)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg)
        obs = torch.randn(8, 4) * 3.0
        dist = actor.dist(obs)
        assert isinstance(dist, TanhGaussian)
        assert not torch.allclose(dist.std[0], dist.std[3])

    def test_sigma_min_floors_the_std(self):
        cfg = cfg_v(sigma_min=0.4)
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg)
        dist = actor.dist(torch.randn(8, 4))
        assert (dist.std >= 0.4 - 1e-6).all()

    def test_action_shape_and_bounds(self):
        actor = Actor(obs_dim=4, act_dim=2, cfg=cfg_v())
        action = actor.dist(torch.randn(8, 4)).rsample()
        assert action.shape == (8, 2)
        assert action.abs().max() < 1.0
