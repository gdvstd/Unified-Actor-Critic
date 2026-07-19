"""Psi retrieval: what the actor gets from the critic.

Queryable signature: the fitted function itself, min over the M *online*
critics (D2: the unified form is the min; TD3's Q1-only actor is a
historical accident we do not reproduce).

Precomputed signature: the fitting residual y - V, which *is* GAE — the
delta-sum identity is asserted in tests, nothing is recomputed.
"""

from __future__ import annotations

from torch import Tensor

from unified_ac.networks import CriticEnsemble


def q_min(critics: CriticEnsemble, obs: Tensor, act: Tensor) -> Tensor:
    """Psi = min_i Q_phi_i(s, a) with online parameters."""
    return critics(obs, act).min(dim=0).values


def advantage_residual(y: Tensor, values: Tensor) -> Tensor:
    """Psi = A_hat = y^(lambda) - V_phi(s): the critic's regression target
    minus its own prediction. One subtraction; GAE by identity."""
    return y - values
