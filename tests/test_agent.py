"""Phase 3: the unified GPI loop — delay semantics, behavior policy, and the
smoke matrix: every preset plus two interior points runs one full update
cycle without error or NaN.
"""

import warnings

import pytest
import torch

from unified_ac import presets
from unified_ac.agent import UnifiedActorCritic
from unified_ac.buffers import ReplayBuffer, RolloutBuffer
from unified_ac.config import UnifiedConfig

OBS_DIM, ACT_DIM = 3, 2
HIDDEN = (32, 32)


def _make_agent(cfg):
    return UnifiedActorCritic(cfg, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN)


def _filled_replay(n=32):
    torch.manual_seed(0)
    buf = ReplayBuffer(capacity=64, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    for i in range(n):
        buf.add(
            obs=torch.randn(OBS_DIM),
            act=torch.rand(ACT_DIM) * 2 - 1,
            reward=float(torch.randn(())),
            next_obs=torch.randn(OBS_DIM),
            terminated=(i % 10 == 9),
        )
    return buf


def _filled_rollout(agent, t=16):
    torch.manual_seed(0)
    buf = RolloutBuffer(capacity=t, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    for i in range(t):
        obs = torch.randn(OBS_DIM)
        action, log_prob = agent.act_with_log_prob(obs.unsqueeze(0))
        buf.add(
            obs=obs,
            act=action.squeeze(0),
            reward=float(torch.randn(())),
            next_obs=torch.randn(OBS_DIM),
            terminated=False,
            truncated=(i == t - 1),
            log_prob=float(log_prob.squeeze(0)),
        )
    return buf


def _params(module):
    return [p.clone() for p in module.parameters()]


def _changed(before, module):
    return any(
        not torch.allclose(b, p) for b, p in zip(before, module.parameters())
    )


class TestPolicyDelay:
    def test_actor_and_targets_update_every_d_critic_steps(self):
        cfg = presets.td3()  # policy_delay=2
        agent = _make_agent(cfg)
        buf = _filled_replay()

        actor_before = _params(agent.actor)
        critic_before = _params(agent.critics)
        target_before = _params(agent.targets.critics)
        agent.update_replay(buf.sample(16))
        assert _changed(critic_before, agent.critics)
        assert not _changed(actor_before, agent.actor)
        assert not _changed(target_before, agent.targets.critics)

        agent.update_replay(buf.sample(16))
        assert _changed(actor_before, agent.actor)
        assert _changed(target_before, agent.targets.critics)


class TestBehaviorPolicy:
    def test_deterministic_boundary_adds_collection_noise(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = presets.ddpg()
        agent = _make_agent(cfg)
        obs = torch.randn(4, OBS_DIM)
        torch.manual_seed(0)
        explore = agent.act(obs)
        greedy = agent.act(obs, deterministic=True)
        assert not torch.allclose(explore, greedy)
        assert explore.abs().max() <= 1.0
        # delta lives in the loop, not the policy: mode is noise-free
        assert torch.allclose(greedy, agent.actor.dist(obs).mode)

    def test_stochastic_policy_samples_its_own_noise(self):
        agent = _make_agent(presets.sac())
        obs = torch.randn(4, OBS_DIM)
        a1, a2 = agent.act(obs), agent.act(obs)
        assert not torch.allclose(a1, a2)


def _interior_td3_soft():
    """TD3 + entropy: stochastic smoothed target — the B3 interior point."""
    return UnifiedConfig(
        alpha=1.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay",
        rho=0.2, clip_c=0.5, eta=0.1, policy_delay=2,
    )


def _interior_sac_smoothed():
    """SAC + target smoothing: the other B3 interior point."""
    return UnifiedConfig(
        alpha=1.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay",
        rho=0.1, clip_c=0.3, eta=0.2, tau_actor=1.0,
    )


class TestSmokeMatrixReplay:
    @pytest.mark.parametrize(
        "factory",
        [presets.ddpg, presets.td3, presets.sac,
         _interior_td3_soft, _interior_sac_smoothed],
    )
    def test_full_update_cycle_runs_finite(self, factory):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = factory()
        agent = _make_agent(cfg)
        buf = _filled_replay()
        actor_before = _params(agent.actor)
        for _ in range(2 * cfg.policy_delay):
            metrics = agent.update_replay(buf.sample(16))
            assert all(
                torch.isfinite(torch.tensor(v)) for v in metrics.values()
            ), metrics
        assert _changed(actor_before, agent.actor)


class TestSmokeMatrixRollout:
    @pytest.mark.parametrize("factory", [presets.a2c, presets.ppo])
    def test_full_update_cycle_runs_finite(self, factory):
        cfg = factory()
        agent = _make_agent(cfg)
        buf = _filled_rollout(agent)
        actor_before = _params(agent.actor)
        critic_before = _params(agent.critics)
        metrics = agent.update_rollout(buf, epochs=2, minibatch_size=8)
        assert all(
            torch.isfinite(torch.tensor(v)) for v in metrics.values()
        ), metrics
        assert _changed(actor_before, agent.actor)
        assert _changed(critic_before, agent.critics)

    def test_frozen_targets_are_stable_across_epochs(self):
        """D4: y is computed once from pre-update parameters; the epoch loop
        must not move it even though the critic updates in between."""
        cfg = presets.ppo()
        agent = _make_agent(cfg)
        buf = _filled_rollout(agent)
        buf.compute_targets(agent.critics, cfg)
        y_before = buf.y.clone()
        agent.update_rollout(buf, epochs=3, minibatch_size=8)
        assert torch.equal(buf.y, y_before)


class TestCollectionEdges:
    def test_dirac_collection_yields_zero_log_prob(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agent = _make_agent(presets.ddpg())
        obs = torch.randn(4, OBS_DIM)
        action, log_prob = agent.act_with_log_prob(obs)
        assert torch.allclose(action, agent.actor.dist(obs).mode)
        assert torch.equal(log_prob, torch.zeros(4))

    def test_queryable_signature_on_rollout_is_deferred(self):
        # legal interior cell (C-valid), implementation deferred to Phase 5
        cfg = UnifiedConfig(
            alpha=1.0, sig="q", lam=0.5, num_critics=2,
            anchor="current", grad="direct", data="rollout",
        )
        agent = _make_agent(cfg)
        ppo_agent = _make_agent(presets.ppo())
        with pytest.raises(NotImplementedError, match="interior"):
            agent.update_rollout(_filled_rollout(ppo_agent))


class TestUpdateDispatch:
    def test_update_replay_on_rollout_config_raises(self):
        agent = _make_agent(presets.ppo())
        with pytest.raises(ValueError, match="data"):
            agent.update_replay(_filled_replay().sample(8))

    def test_update_rollout_on_replay_config_raises(self):
        agent = _make_agent(presets.sac())
        ppo_agent = _make_agent(presets.ppo())
        with pytest.raises(ValueError, match="data"):
            agent.update_rollout(_filled_rollout(ppo_agent))
