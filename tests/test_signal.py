"""Phase 2: Psi retrieval — GAE derived as the regression residual.

The paper's identity: A_t = y_t^(lambda) - V(s_t) = sum_k (gamma*lam)^k delta_{t+k}.
Nothing new is computed for the actor; the critic's regression target and the
actor's learning signal are the same object measured from two baselines.
"""

import torch

from unified_ac.signal import advantage_residual, q_min
from unified_ac.targets import lambda_return

GAMMA, LAM = 0.9, 0.95


def _gae_reference(rewards, values, next_values, terminated, gamma, lam):
    """Independent implementation: the delta-sum form, forward definition."""
    horizon = rewards.shape[0]
    cont = 1.0 - terminated.float()
    deltas = rewards + gamma * cont * next_values - values
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(horizon)):
        running = deltas[t] + gamma * lam * cont[t] * running
        adv[t] = running
    return adv


class TestGAEResidualIdentity:
    def setup_method(self):
        self.r = torch.tensor([1.0, 2.0, 3.0])
        self.v = torch.tensor([2.0, 4.0, 6.0])
        self.v_next = torch.tensor([4.0, 6.0, 8.0])  # V(s_{t+1}), incl. V(s_T)

    def test_identity_on_truncated_rollout(self):
        term = torch.zeros(3, dtype=torch.bool)
        trunc = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(self.r, self.v_next, term, trunc, GAMMA, LAM)
        residual = advantage_residual(y, self.v)
        expected = _gae_reference(self.r, self.v, self.v_next, term, GAMMA, LAM)
        assert torch.allclose(residual, expected, atol=1e-6)

    def test_identity_with_mid_rollout_termination(self):
        term = torch.tensor([False, True, False])
        trunc = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(self.r, self.v_next, term, trunc, GAMMA, LAM)
        residual = advantage_residual(y, self.v)
        expected = _gae_reference(self.r, self.v, self.v_next, term, GAMMA, LAM)
        assert torch.allclose(residual, expected, atol=1e-6)


class TestQMin:
    def test_min_over_online_ensemble(self):
        from tests.test_networks import cfg_q
        from unified_ac.networks import CriticEnsemble

        critics = CriticEnsemble(obs_dim=4, act_dim=2, cfg=cfg_q())
        obs, act = torch.randn(8, 4), torch.rand(8, 2) * 2 - 1
        psi = q_min(critics, obs, act)
        assert torch.allclose(psi, critics(obs, act).min(dim=0).values)
