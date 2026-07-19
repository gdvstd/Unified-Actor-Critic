"""Phase 3: replay (shuffled transitions) vs rollout (trajectory-ordered).

The C7 boundary lives in the buffer layer: only the trajectory-ordered
RolloutBuffer can serve lambda > 0, and its frozen targets are computed once
per rollout (D4's rollout-side decoupling).
"""

import pytest
import torch

from unified_ac.buffers import ReplayBuffer, RolloutBuffer
from unified_ac.config import UnifiedConfig
from unified_ac.targets import lambda_return


def v_cfg(**overrides):
    kwargs = dict(
        alpha=1.0, sig="v", lam=0.95, num_critics=1,
        anchor="old", grad="score", data="rollout",
        sigma_mode="global", ratio_clip=0.2,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


class StubVCritic:
    """Duck-typed CriticEnsemble: V(s) = sum(s), ensemble dim 1."""

    def __call__(self, obs, act=None):
        return obs.sum(-1).unsqueeze(0)


class TestReplayBuffer:
    def _add_n(self, buf, n, offset=0.0):
        for i in range(n):
            buf.add(
                obs=torch.full((3,), offset + float(i)),
                act=torch.zeros(2),
                reward=float(i),
                next_obs=torch.full((3,), offset + float(i) + 0.5),
                terminated=False,
            )

    def test_fifo_overwrite(self):
        buf = ReplayBuffer(capacity=5, obs_dim=3, act_dim=2)
        self._add_n(buf, 7)
        assert len(buf) == 5
        # oldest two entries (0, 1) evicted: rewards present are 2..6
        assert set(buf._reward[: len(buf)].tolist()) == {2.0, 3.0, 4.0, 5.0, 6.0}

    def test_sample_shapes_and_dtypes(self):
        buf = ReplayBuffer(capacity=64, obs_dim=3, act_dim=2)
        self._add_n(buf, 20)
        batch = buf.sample(8)
        assert batch.obs.shape == (8, 3) and batch.obs.dtype == torch.float32
        assert batch.act.shape == (8, 2)
        assert batch.reward.shape == (8,)
        assert batch.terminated.dtype == torch.bool

    def test_sample_before_fill_raises(self):
        buf = ReplayBuffer(capacity=64, obs_dim=3, act_dim=2)
        self._add_n(buf, 4)
        with pytest.raises(ValueError, match="fewer"):
            buf.sample(8)


class TestRolloutBuffer:
    def _fill(self, buf, t=3):
        for i in range(t):
            buf.add(
                obs=torch.full((3,), float(i)),
                act=torch.zeros(2),
                reward=float(i + 1),
                next_obs=torch.full((3,), float(i) + 0.5),
                terminated=False,
                truncated=False,
                log_prob=0.1 * i,
            )

    def test_overfill_raises(self):
        buf = RolloutBuffer(capacity=2, obs_dim=3, act_dim=2)
        self._fill(buf, 2)
        assert buf.full
        with pytest.raises(ValueError, match="full"):
            self._fill(buf, 1)

    def test_compute_targets_wires_values_and_bootstrap_correctly(self):
        """V from obs, B from next_obs, y via lambda_return, adv = y - V."""
        cfg = v_cfg()
        buf = RolloutBuffer(capacity=3, obs_dim=3, act_dim=2)
        self._fill(buf)
        buf.compute_targets(StubVCritic(), cfg)

        values = torch.tensor([0.0, 3.0, 6.0])        # sum of obs
        bootstrap = torch.tensor([1.5, 4.5, 7.5])     # sum of next_obs
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.zeros(3, dtype=torch.bool)
        y = lambda_return(rewards, bootstrap, dones, dones, cfg.gamma, cfg.lam)
        assert torch.allclose(buf.y, y)
        assert torch.allclose(buf.adv, y - values)

    def test_targets_are_frozen_tensors(self):
        cfg = v_cfg()
        buf = RolloutBuffer(capacity=3, obs_dim=3, act_dim=2)
        self._fill(buf)
        buf.compute_targets(StubVCritic(), cfg)
        assert not buf.y.requires_grad and not buf.adv.requires_grad

    def test_minibatches_partition_every_index_once(self):
        cfg = v_cfg()
        buf = RolloutBuffer(capacity=8, obs_dim=3, act_dim=2)
        self._fill(buf, 8)
        buf.compute_targets(StubVCritic(), cfg)
        seen = []
        for mb in buf.minibatches(batch_size=3):
            seen.extend(mb.obs[:, 0].tolist())
        assert sorted(seen) == [float(i) for i in range(8)]

    def test_minibatches_before_targets_raises(self):
        buf = RolloutBuffer(capacity=3, obs_dim=3, act_dim=2)
        self._fill(buf)
        with pytest.raises(ValueError, match="compute_targets"):
            next(iter(buf.minibatches(batch_size=2)))

    def test_clear_resets_for_reuse(self):
        cfg = v_cfg()
        buf = RolloutBuffer(capacity=3, obs_dim=3, act_dim=2)
        self._fill(buf)
        buf.compute_targets(StubVCritic(), cfg)
        buf.clear()
        assert not buf.full and len(buf) == 0
        self._fill(buf, 3)
        assert buf.full
