"""Phase 4: exact reduction — identical networks, identical fixed batch,
seed-matched sampling: our unified objectives against CleanRL's transcribed
update math (tests/cleanrl_reference.py, D10).

Tolerances: TD3 and PPO paths are algebraically identical (atol ~1e-6);
SAC paths differ only by CleanRL's +1e-6 epsilon inside the tanh correction
vs our exact softplus form (atol 1e-4 on values, rtol 1e-3 on grads).

The two deliberate deviations are asserted *as* deviations:
- TD3 actor: ours equals the min-substituted reference, not the Q1 form (D2).
- PPO value loss: ours equals 2x CleanRL's 0.5*MSE (pure scale factor).
"""

import torch

import tests.cleanrl_reference as ref
from unified_ac import presets
from unified_ac.config import UnifiedConfig
from unified_ac.losses import actor_loss, critic_loss
from unified_ac.networks import Actor, CriticEnsemble
from unified_ac.signal import advantage_residual
from unified_ac.targets import PolyakTargets, bootstrap_B, lambda_return

OBS_DIM, ACT_DIM, BATCH = 3, 2, 16
HIDDEN = (64, 64)


def _grads(loss, params):
    return torch.autograd.grad(loss, list(params))


def _assert_grads_close(g1, g2, rtol=1e-5, atol=1e-6):
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), (
            f"grad mismatch: max abs diff {(a - b).abs().max():.2e}"
        )


def _replay_fixture(cfg, seed=0):
    torch.manual_seed(seed)
    actor = Actor(OBS_DIM, ACT_DIM, cfg, hidden=HIDDEN)
    critics = CriticEnsemble(OBS_DIM, ACT_DIM, cfg, hidden=HIDDEN)
    targets = PolyakTargets(actor, critics, cfg)
    # decorrelate targets from online nets so min/target terms are non-trivial
    with torch.no_grad():
        for p in targets.critics.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    batch = dict(
        obs=torch.randn(BATCH, OBS_DIM),
        act=torch.rand(BATCH, ACT_DIM) * 2 - 1,
        reward=torch.randn(BATCH),
        next_obs=torch.randn(BATCH, OBS_DIM),
        dones=torch.zeros(BATCH),  # no truncation events in fixed batches (D3)
    )
    qf = [
        (lambda o, a, m=m: m(torch.cat([o, a], dim=-1)).squeeze(-1))
        for m in critics.members
    ]
    qf_t = [
        (lambda o, a, m=m: m(torch.cat([o, a], dim=-1)).squeeze(-1))
        for m in targets.critics.members
    ]
    return actor, critics, targets, batch, qf, qf_t


class TestSACReduction:
    def setup_method(self):
        self.cfg = presets.sac()
        (self.actor, self.critics, self.targets,
         self.batch, self.qf, self.qf_t) = _replay_fixture(self.cfg)

    def test_critic_target_and_loss_match(self):
        b = self.batch
        torch.manual_seed(7)
        with torch.no_grad():
            boot = bootstrap_B(b["next_obs"], self.cfg, self.targets.actor, self.targets.critics)
            y_ours = b["reward"] + self.cfg.gamma * (1 - b["dones"]) * boot
        loss_ours, _ = critic_loss(self.critics, b["obs"], b["act"], y_ours, self.cfg)

        torch.manual_seed(7)
        with torch.no_grad():
            dist = self.targets.actor.dist(b["next_obs"])
            y_ref = ref.sac_critic_target(
                dist.mu, dist.std, self.qf_t[0], self.qf_t[1],
                b["next_obs"], b["reward"], b["dones"],
                self.cfg.gamma, self.cfg.eta,
            )
        loss_ref = ref.sac_critic_loss(self.qf[0], self.qf[1], b["obs"], b["act"], y_ref)

        assert torch.allclose(y_ours, y_ref, atol=1e-4)
        assert torch.allclose(loss_ours, loss_ref, atol=1e-4)
        _assert_grads_close(
            _grads(loss_ours, self.critics.parameters()),
            _grads(loss_ref, self.critics.parameters()),
            rtol=1e-3, atol=1e-5,
        )

    def test_actor_loss_matches(self):
        b = self.batch
        torch.manual_seed(9)
        loss_ours, _ = actor_loss(self.actor, self.cfg, b["obs"], critics=self.critics)

        torch.manual_seed(9)
        dist = self.actor.dist(b["obs"])
        loss_ref = ref.sac_actor_loss(
            dist.mu, dist.std, self.qf[0], self.qf[1], b["obs"], self.cfg.eta
        )

        assert torch.allclose(loss_ours, loss_ref, atol=1e-4)
        torch.manual_seed(9)
        loss_ours, _ = actor_loss(self.actor, self.cfg, b["obs"], critics=self.critics)
        g_ours = _grads(loss_ours, self.actor.parameters())
        torch.manual_seed(9)
        dist = self.actor.dist(b["obs"])
        loss_ref = ref.sac_actor_loss(
            dist.mu, dist.std, self.qf[0], self.qf[1], b["obs"], self.cfg.eta
        )
        g_ref = _grads(loss_ref, self.actor.parameters())
        _assert_grads_close(g_ours, g_ref, rtol=1e-3, atol=1e-5)


class TestTD3Reduction:
    def setup_method(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cfg = presets.td3()
        (self.actor, self.critics, self.targets,
         self.batch, self.qf, self.qf_t) = _replay_fixture(self.cfg)

    def test_critic_target_and_loss_match_exactly(self):
        b = self.batch
        torch.manual_seed(3)
        with torch.no_grad():
            boot = bootstrap_B(b["next_obs"], self.cfg, self.targets.actor, self.targets.critics)
            y_ours = b["reward"] + self.cfg.gamma * (1 - b["dones"]) * boot
        loss_ours, _ = critic_loss(self.critics, b["obs"], b["act"], y_ours, self.cfg)

        torch.manual_seed(3)
        with torch.no_grad():
            target_mu = self.targets.actor.dist(b["next_obs"]).mode
            y_ref = ref.td3_critic_target(
                target_mu, self.qf_t[0], self.qf_t[1],
                b["next_obs"], b["act"], b["reward"], b["dones"],
                self.cfg.gamma, self.cfg.rho, self.cfg.clip_c,
            )
        loss_ref = ref.sac_critic_loss(self.qf[0], self.qf[1], b["obs"], b["act"], y_ref)

        assert torch.allclose(y_ours, y_ref, atol=1e-6)
        assert torch.allclose(loss_ours, loss_ref, atol=1e-6)

    def test_actor_loss_matches_min_reference_not_qf1(self):
        b = self.batch
        tanh_mu = self.actor.dist(b["obs"]).mode
        # center the twin gap so the min provably mixes both critics
        # (fresh init can leave one net uniformly below the other); the 1e-3
        # offset avoids an exact tie, where stacked-min and elementwise-min
        # route subgradients differently
        with torch.no_grad():
            gap = self.qf[1](b["obs"], tanh_mu) - self.qf[0](b["obs"], tanh_mu)
            self.critics.members[1][-1].bias -= gap.median() + 1e-3
            q = self.critics(b["obs"], tanh_mu)
            picks_q1 = (q.min(dim=0).values == q[0])
            assert picks_q1.any() and not picks_q1.all()

        loss_ours, _ = actor_loss(self.actor, self.cfg, b["obs"], critics=self.critics)

        loss_min = ref.td3_actor_loss_min(self.qf[0], self.qf[1], b["obs"], tanh_mu)
        loss_qf1 = ref.td3_actor_loss_qf1(self.qf[0], b["obs"], tanh_mu)

        assert torch.allclose(loss_ours, loss_min, atol=1e-6)
        _assert_grads_close(
            _grads(loss_ours, self.actor.parameters()),
            _grads(loss_min, self.actor.parameters()),
        )
        # the documented D2 deviation from CleanRL's literal form
        assert not torch.allclose(loss_ours, loss_qf1, atol=1e-6)


class TestPPOReduction:
    def setup_method(self):
        self.cfg = presets.ppo()
        torch.manual_seed(1)
        self.actor = Actor(OBS_DIM, ACT_DIM, self.cfg, hidden=HIDDEN)
        self.critics = CriticEnsemble(OBS_DIM, ACT_DIM, self.cfg, hidden=HIDDEN)

    def test_gae_and_returns_match_exactly(self):
        t = 16
        rewards = torch.randn(t)
        values = torch.randn(t)
        next_value = torch.randn(())
        no_dones = torch.zeros(t, dtype=torch.bool)

        bootstrap = torch.cat([values[1:], next_value.unsqueeze(0)])
        y = lambda_return(rewards, bootstrap, no_dones, no_dones,
                          self.cfg.gamma, self.cfg.lam)
        adv_ours = advantage_residual(y, values)

        adv_ref, returns_ref = ref.ppo_gae(rewards, values, next_value,
                                           self.cfg.gamma, self.cfg.lam)
        assert torch.allclose(adv_ours, adv_ref, atol=1e-6)
        assert torch.allclose(y, returns_ref, atol=1e-6)

    def test_pg_loss_matches_exactly_with_active_clipping(self):
        obs = torch.randn(BATCH, OBS_DIM)
        with torch.no_grad():
            dist = self.actor.dist(obs)
            act = dist.rsample()
            # spread the ratios so both clip branches activate
            old_lp = dist.log_prob(act) + torch.randn(BATCH) * 0.5
        adv = torch.randn(BATCH)

        loss_ours, metrics = actor_loss(
            self.actor, self.cfg, obs, act=act, psi=adv, anchor_log_prob=old_lp
        )
        newlogprob = self.actor.dist(obs).log_prob(act)
        loss_ref = ref.ppo_pg_loss(newlogprob, old_lp, adv, self.cfg.ratio_clip)

        assert metrics["clip_frac"] > 0, "fixture failed to activate clipping"
        assert torch.allclose(loss_ours, loss_ref, atol=1e-6)
        _assert_grads_close(
            _grads(loss_ours, self.actor.parameters()),
            _grads(loss_ref, self.actor.parameters()),
        )

    def test_value_loss_is_cleanrl_times_two(self):
        obs = torch.randn(BATCH, OBS_DIM)
        returns = torch.randn(BATCH)
        loss_ours, _ = critic_loss(self.critics, obs, None, returns, self.cfg)
        newvalue = self.critics(obs)[0]
        loss_ref = ref.ppo_v_loss_unclipped(newvalue, returns)
        assert torch.allclose(loss_ours, 2.0 * loss_ref, atol=1e-6)
