"""The toggle space and its constraint set.

Type A constraints (C1-C9) are definitional: violating one produces an
expression with an undefined term, so construction raises. C4 marks
configurations inert rather than invalid (warn). Type B constraints (B1-B4)
are empirical predictions: logged as warnings, never enforced — enforcing B1
would assume the paper's conclusion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal


class InvalidConfigError(ValueError):
    """Type A violation: the configuration contains an undefined term."""

    def __init__(self, constraint_id: str, message: str) -> None:
        self.constraint_id = constraint_id
        super().__init__(f"{constraint_id}: {message}")


class InertConfigWarning(UserWarning):
    """C4: a hyperparameter is present but provably without effect."""


class FolkEntropyWarning(UserWarning):
    """C9 violation escape hatch: actor-only entropy over a hard-return critic."""


class TypeBWarning(UserWarning):
    """Type B prediction: empirically risky or redundant, never enforced."""


@dataclass(frozen=True)
class UnifiedConfig:
    # ---- policy ----
    alpha: float                                   # learned stochasticity scale; 0 = deterministic boundary
    sig: Literal["q", "v"]                         # critic signature: queryable Q vs precomputed advantage
    lam: float                                     # bootstrap depth of the lambda-return
    num_critics: int                               # M
    anchor: Literal["current", "old"]              # where samples and the trust region are pinned
    grad: Literal["score", "direct"]               # how J is differentiated
    data: Literal["rollout", "replay"]
    # ---- optional dials ----
    sigma_min: float = 0.0                         # policy variance floor
    sigma_mode: Literal["state", "global"] = "state"
    rho: float = 0.0                               # target policy smoothing scale
    clip_c: float = 0.0                            # smoothing clip
    eta: float = 0.0                               # entropy temperature, shared by B and J (C9)
    kl_coef: float = 0.0                           # lambda_KL
    ratio_clip: float | None = None                # PPO's realization of the same trust region
    folk_entropy_bonus: float = 0.0                # D6: deliberate C9 violation, logged
    tau_critic: float = 0.005
    tau_actor: float = 0.005
    policy_delay: int = 1                          # d
    explore_noise: float = 0.0                     # delta: behavior-policy noise, in no objective
    gamma: float = 0.99

    def __post_init__(self) -> None:
        validate(self)


def validate(cfg: UnifiedConfig) -> None:
    """Domain checks, then Type A (raise), then C4/C9/Type B (warn)."""
    _check_domains(cfg)
    _check_type_a(cfg)
    _warn_inert_and_type_b(cfg)


def _check_domains(cfg: UnifiedConfig) -> None:
    bounds = [
        (cfg.alpha >= 0.0, "alpha >= 0"),
        (cfg.sigma_min >= 0.0, "sigma_min >= 0"),
        (0.0 <= cfg.lam <= 1.0, "lam in [0, 1]"),
        (cfg.num_critics in (1, 2), "num_critics in {1, 2}"),
        (cfg.rho >= 0.0, "rho >= 0"),
        (cfg.clip_c >= 0.0, "clip_c >= 0"),
        (cfg.eta >= 0.0, "eta >= 0"),
        (cfg.kl_coef >= 0.0, "kl_coef >= 0"),
        (cfg.ratio_clip is None or cfg.ratio_clip > 0.0, "ratio_clip None or > 0"),
        (cfg.folk_entropy_bonus >= 0.0, "folk_entropy_bonus >= 0"),
        (0.0 < cfg.tau_critic <= 1.0, "tau_critic in (0, 1]"),
        (0.0 < cfg.tau_actor <= 1.0, "tau_actor in (0, 1]"),
        (cfg.policy_delay >= 1, "policy_delay >= 1"),
        (cfg.explore_noise >= 0.0, "explore_noise >= 0"),
        (0.0 < cfg.gamma <= 1.0, "gamma in (0, 1]"),
    ]
    for ok, requirement in bounds:
        if not ok:
            raise InvalidConfigError("domain", f"requires {requirement}")


def _check_type_a(cfg: UnifiedConfig) -> None:
    deterministic = cfg.alpha == 0.0
    if cfg.grad == "score" and deterministic:
        raise InvalidConfigError(
            "C1", "the score estimator contains grad log pi; a Dirac has no density"
        )
    if (cfg.eta > 0.0 or cfg.folk_entropy_bonus > 0.0) and deterministic:
        raise InvalidConfigError(
            "C2", "entropy terms diverge on the deterministic boundary"
        )
    if cfg.anchor == "old" and deterministic:
        raise InvalidConfigError(
            "C3", "the IS ratio and KL integrand are built from densities"
        )
    if cfg.sigma_min > 0.0 and deterministic:
        raise InvalidConfigError(
            "C5", "the floor bounds the learned variance; alpha = 0 leaves none"
        )
    if cfg.grad == "direct" and cfg.sig != "q":
        raise InvalidConfigError(
            "C6", "the direct estimator needs grad_a Psi; a precomputed "
            "advantage has no action argument"
        )
    if cfg.lam > 0.0 and cfg.data != "rollout":
        raise InvalidConfigError(
            "C7", "the lambda-return sums residuals along consecutive timesteps; "
            "a shuffled replay buffer cannot assemble it"
        )
    if cfg.rho > 0.0 and cfg.sig != "q":
        raise InvalidConfigError(
            "C8", "smoothing perturbs the target's action slot; V has none"
        )
    if cfg.anchor == "old" and cfg.kl_coef > 0.0 and cfg.ratio_clip is not None:
        raise InvalidConfigError(
            "trust-region", "kl_coef and ratio_clip realize the same bound; "
            "set only one"
        )


def _warn_inert_and_type_b(cfg: UnifiedConfig) -> None:
    if cfg.anchor == "current" and (cfg.kl_coef > 0.0 or cfg.ratio_clip is not None):
        warnings.warn(
            "C4: anchor=current makes the ratio 1 and the KL 0; "
            "kl_coef/ratio_clip are inert",
            InertConfigWarning,
            stacklevel=4,
        )
    if cfg.folk_entropy_bonus > 0.0:
        warnings.warn(
            "C9: actor-only entropy bonus over a hard-return critic — the loop "
            "is no longer policy iteration on a single objective (folk form)",
            FolkEntropyWarning,
            stacklevel=4,
        )
    hedged = cfg.num_critics >= 2 or cfg.rho > 0.0 or cfg.eta > 0.0
    if cfg.data == "replay" and cfg.grad == "direct" and not hedged:
        warnings.warn(
            "B1: replay + direct gradient with no hedges — deadly-triad region",
            TypeBWarning,
            stacklevel=4,
        )
    if cfg.alpha == 0.0 and cfg.explore_noise == 0.0:
        warnings.warn(
            "B2: deterministic policy with no collection noise — under-exploration",
            TypeBWarning,
            stacklevel=4,
        )
    if cfg.rho > 0.0 and cfg.alpha > 0.0:
        warnings.warn(
            "B3: target smoothing under a stochastic policy is redundant "
            "up to the clip",
            TypeBWarning,
            stacklevel=4,
        )
    if cfg.sig == "v" and (cfg.num_critics >= 2 or cfg.eta > 0.0):
        warnings.warn(
            "B4: query-defenses under the precomputed signature — the actor "
            "never queries the critic at new actions",
            TypeBWarning,
            stacklevel=4,
        )
