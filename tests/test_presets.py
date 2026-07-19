"""Phase 0: the recovery table as executable assertions.

Every preset must construct without raising. Four of five are warning-free;
DDPG alone sits in the deadly-triad region (replay + direct + no hedges), so
constructing it emits exactly the B1 warning — the paper's thesis, asserted.
"""

import warnings

import pytest

from unified_ac import presets
from unified_ac.config import TypeBWarning, UnifiedConfig


ALL_PRESETS = [presets.a2c, presets.ppo, presets.ddpg, presets.td3, presets.sac]


@pytest.mark.parametrize("factory", ALL_PRESETS)
def test_every_preset_constructs(factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(factory(), UnifiedConfig)


@pytest.mark.parametrize(
    "factory", [presets.a2c, presets.ppo, presets.td3, presets.sac]
)
def test_hedged_presets_are_warning_free(factory):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory()


def test_ddpg_warns_b1_and_nothing_else(factory=presets.ddpg):
    with pytest.warns(TypeBWarning, match="B1") as record:
        factory()
    assert len(record) == 1


class TestRecoveryTable:
    """Row-by-row: the paper's table, §6 of PLAN.md."""

    def test_a2c_row(self):
        cfg = presets.a2c()
        assert (cfg.data, cfg.sig, cfg.lam, cfg.num_critics) == ("rollout", "v", 1.0, 1)
        assert (cfg.rho, cfg.eta) == (0.0, 0.0)
        assert cfg.alpha > 0
        assert (cfg.anchor, cfg.grad) == ("current", "score")
        assert cfg.kl_coef == 0.0 and cfg.ratio_clip is None

    def test_ppo_row(self):
        cfg = presets.ppo()
        assert (cfg.data, cfg.sig, cfg.num_critics) == ("rollout", "v", 1)
        assert 0.0 < cfg.lam <= 1.0
        assert cfg.alpha > 0
        assert (cfg.anchor, cfg.grad) == ("old", "score")
        assert cfg.ratio_clip == 0.2 and cfg.kl_coef == 0.0

    def test_ddpg_row(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = presets.ddpg()
        assert (cfg.data, cfg.sig, cfg.lam, cfg.num_critics) == ("replay", "q", 0.0, 1)
        assert (cfg.rho, cfg.eta, cfg.alpha) == (0.0, 0.0, 0.0)
        assert (cfg.anchor, cfg.grad) == ("current", "direct")
        assert cfg.explore_noise > 0
        assert cfg.policy_delay == 1

    def test_td3_row(self):
        cfg = presets.td3()
        assert (cfg.data, cfg.sig, cfg.lam, cfg.num_critics) == ("replay", "q", 0.0, 2)
        assert cfg.rho == 0.2 and cfg.clip_c == 0.5
        assert (cfg.eta, cfg.alpha) == (0.0, 0.0)
        assert (cfg.anchor, cfg.grad) == ("current", "direct")
        assert cfg.policy_delay == 2
        assert cfg.explore_noise > 0

    def test_sac_row(self):
        cfg = presets.sac()
        assert (cfg.data, cfg.sig, cfg.lam, cfg.num_critics) == ("replay", "q", 0.0, 2)
        assert cfg.rho == 0.0 and cfg.eta > 0
        assert cfg.alpha > 0
        assert (cfg.anchor, cfg.grad) == ("current", "direct")
        # SAC decouples only the critic: the actor's Polyak coefficient is 1
        assert cfg.tau_actor == 1.0
        assert cfg.tau_critic < 1.0

    def test_coherent_hard_return_forms(self):
        # the table records eta=0 on rollout rows (C9-coherent), and the
        # folk actor-only bonus stays off by default
        for factory in (presets.a2c, presets.ppo):
            cfg = factory()
            assert cfg.eta == 0.0
            assert cfg.folk_entropy_bonus == 0.0
