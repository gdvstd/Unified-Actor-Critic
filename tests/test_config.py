"""Phase 0: the constraint set (C1-C9 Type A, B1-B4 Type B) as executable spec.

Every Type A violation must raise InvalidConfigError carrying the paper's
constraint ID. C4 marks configs inert (warn, not raise). Type B only warns.
"""

import dataclasses
import warnings

import pytest

from unified_ac.config import (
    FolkEntropyWarning,
    InertConfigWarning,
    InvalidConfigError,
    TypeBWarning,
    UnifiedConfig,
)


def sac_like(**overrides) -> dict:
    """A fully valid baseline config (SAC row of the recovery table)."""
    kwargs = dict(
        alpha=1.0,
        sig="q",
        lam=0.0,
        num_critics=2,
        anchor="current",
        grad="direct",
        data="replay",
        eta=0.2,
        tau_actor=1.0,
    )
    kwargs.update(overrides)
    return kwargs


def ppo_like(**overrides) -> dict:
    """A fully valid on-policy baseline (PPO row)."""
    kwargs = dict(
        alpha=1.0,
        sig="v",
        lam=0.95,
        num_critics=1,
        anchor="old",
        grad="score",
        data="rollout",
        ratio_clip=0.2,
        sigma_mode="global",
    )
    kwargs.update(overrides)
    return kwargs


def make(**overrides) -> UnifiedConfig:
    return UnifiedConfig(**sac_like(**overrides))


class TestValidBaselines:
    def test_sac_like_constructs_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            make()

    def test_ppo_like_constructs_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            UnifiedConfig(**ppo_like())

    def test_config_is_frozen(self):
        cfg = make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.alpha = 0.5


class TestTypeAConstraints:
    """Each violation raises with the paper's constraint ID in the message."""

    def test_c1_score_requires_stochastic_policy(self):
        # score estimator contains grad log pi; a Dirac has no density
        with pytest.raises(InvalidConfigError, match="C1"):
            UnifiedConfig(
                **ppo_like(alpha=0.0, anchor="current", ratio_clip=None)
            )

    def test_c2_entropy_requires_stochastic_policy(self):
        with pytest.raises(InvalidConfigError, match="C2"):
            make(alpha=0.0, eta=0.5, explore_noise=0.1)

    def test_c2_covers_folk_entropy_bonus(self):
        # folk bonus is entropy in J, so it falls under C2 as well
        with pytest.raises(InvalidConfigError, match="C2"):
            make(alpha=0.0, eta=0.0, folk_entropy_bonus=0.01, explore_noise=0.1)

    def test_c3_old_anchor_requires_stochastic_policy(self):
        # the IS ratio and KL integrand are built from densities
        with pytest.raises(InvalidConfigError, match="C3"):
            make(alpha=0.0, eta=0.0, anchor="old", explore_noise=0.1)

    def test_c5_sigma_floor_requires_stochastic_policy(self):
        with pytest.raises(InvalidConfigError, match="C5"):
            make(alpha=0.0, eta=0.0, sigma_min=0.1, explore_noise=0.1)

    def test_c6_direct_gradient_requires_q_signature(self):
        # a precomputed advantage has no action argument to differentiate
        with pytest.raises(InvalidConfigError, match="C6"):
            UnifiedConfig(
                **ppo_like(grad="direct", anchor="current", ratio_clip=None)
            )

    def test_c7_interior_lambda_requires_rollout_data(self):
        # a shuffled buffer of transitions cannot assemble the lambda-return
        with pytest.raises(InvalidConfigError, match="C7"):
            make(lam=0.5)

    def test_c8_target_smoothing_requires_q_signature(self):
        # the perturbation lives in the target's action slot; V has none
        with pytest.raises(InvalidConfigError, match="C8"):
            UnifiedConfig(**ppo_like(rho=0.2, clip_c=0.5))

    def test_trust_region_realizations_are_mutually_exclusive(self):
        with pytest.raises(InvalidConfigError, match="trust-region"):
            UnifiedConfig(**ppo_like(kl_coef=0.1, ratio_clip=0.2))


class TestC4Inert:
    """anchor=current makes the ratio 1 and the KL 0: inert, not invalid."""

    def test_kl_coef_inert_under_current_anchor(self):
        with pytest.warns(InertConfigWarning, match="C4"):
            make(kl_coef=0.1)

    def test_ratio_clip_inert_under_current_anchor(self):
        with pytest.warns(InertConfigWarning, match="C4"):
            make(ratio_clip=0.2)


class TestC9Folk:
    def test_folk_entropy_bonus_warns_c9_violation(self):
        with pytest.warns(FolkEntropyWarning, match="C9"):
            make(folk_entropy_bonus=0.01)


class TestTypeBWarnings:
    """Predictions, not rules: warn and construct."""

    def test_b1_deadly_triad_region(self):
        # replay + direct gradient + no hedges (M=1, rho=0, eta=0)
        with pytest.warns(TypeBWarning, match="B1"):
            cfg = make(
                alpha=0.0, eta=0.0, num_critics=1, explore_noise=0.1,
                tau_actor=0.005,
            )
        assert cfg.num_critics == 1

    def test_b2_under_exploration(self):
        # deterministic boundary with no collection noise
        with pytest.warns(TypeBWarning, match="B2"):
            make(alpha=0.0, eta=0.0, num_critics=2, explore_noise=0.0,
                 tau_actor=0.005)

    def test_b3_smoothing_redundant_under_stochastic_policy(self):
        with pytest.warns(TypeBWarning, match="B3"):
            make(rho=0.2, clip_c=0.5)

    def test_b4_query_defenses_without_queries(self):
        # twin critics under the precomputed signature defend against
        # queries that never happen
        with pytest.warns(TypeBWarning, match="B4"):
            UnifiedConfig(**ppo_like(num_critics=2))

    def test_b4_soft_target_without_queries(self):
        with pytest.warns(TypeBWarning, match="B4"):
            UnifiedConfig(**ppo_like(eta=0.1))


class TestDomains:
    @pytest.mark.parametrize(
        "overrides",
        [
            dict(alpha=-0.1),
            dict(lam=1.5),
            dict(lam=-0.1),
            dict(num_critics=3),
            dict(num_critics=0),
            dict(rho=-0.1),
            dict(eta=-0.1),
            dict(gamma=0.0),
            dict(gamma=1.1),
            dict(tau_critic=0.0),
            dict(tau_actor=1.5),
            dict(policy_delay=0),
            dict(explore_noise=-0.1),
            dict(sigma_min=-0.1),
            dict(ratio_clip=0.0),
        ],
    )
    def test_out_of_domain_raises(self, overrides):
        with pytest.raises(InvalidConfigError, match="domain"):
            make(**overrides)
