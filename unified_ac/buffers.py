"""Data regimes. The C7 boundary lives here: the shuffled ReplayBuffer stores
independent transitions and cannot assemble a lambda-return; the
trajectory-ordered RolloutBuffer can, and it also realizes the rollout side
of target decoupling (D4) by freezing y and the advantage once per rollout.

Replay stores `terminated` only (D3): truncated transitions keep
bootstrapping, and next_obs is always the real successor observation.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

import torch
from torch import Tensor

from unified_ac.config import UnifiedConfig
from unified_ac.targets import lambda_return


class ReplayBatch(NamedTuple):
    obs: Tensor
    act: Tensor
    reward: Tensor
    next_obs: Tensor
    terminated: Tensor


class RolloutBatch(NamedTuple):
    obs: Tensor
    act: Tensor
    y: Tensor
    adv: Tensor
    anchor_log_prob: Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self._obs = torch.zeros(capacity, obs_dim)
        self._act = torch.zeros(capacity, act_dim)
        self._reward = torch.zeros(capacity)
        self._next_obs = torch.zeros(capacity, obs_dim)
        self._terminated = torch.zeros(capacity, dtype=torch.bool)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: Tensor,
        act: Tensor,
        reward: float,
        next_obs: Tensor,
        terminated: bool,
    ) -> None:
        i = self._ptr
        self._obs[i] = obs
        self._act[i] = act
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._terminated[i] = terminated
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size > self._size:
            raise ValueError(
                f"buffer holds fewer transitions ({self._size}) than requested "
                f"({batch_size})"
            )
        idx = torch.randint(self._size, (batch_size,))
        return ReplayBatch(
            obs=self._obs[idx],
            act=self._act[idx],
            reward=self._reward[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
        )


class RolloutBuffer:
    """One trajectory-ordered rollout plus its frozen targets."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = capacity
        self._obs = torch.zeros(capacity, obs_dim)
        self._act = torch.zeros(capacity, act_dim)
        self._reward = torch.zeros(capacity)
        self._next_obs = torch.zeros(capacity, obs_dim)
        self._terminated = torch.zeros(capacity, dtype=torch.bool)
        self._truncated = torch.zeros(capacity, dtype=torch.bool)
        self._log_prob = torch.zeros(capacity)
        self._ptr = 0
        self.y: Tensor | None = None
        self.adv: Tensor | None = None
        self.values: Tensor | None = None

    def __len__(self) -> int:
        return self._ptr

    @property
    def full(self) -> bool:
        return self._ptr == self.capacity

    def add(
        self,
        obs: Tensor,
        act: Tensor,
        reward: float,
        next_obs: Tensor,
        terminated: bool,
        truncated: bool,
        log_prob: float,
    ) -> None:
        if self.full:
            raise ValueError("rollout buffer is full; call clear() after the update")
        i = self._ptr
        self._obs[i] = obs
        self._act[i] = act
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        self._terminated[i] = terminated
        self._truncated[i] = truncated
        self._log_prob[i] = log_prob
        self._ptr = i + 1

    def compute_targets(self, critics, cfg: UnifiedConfig) -> None:
        """Freeze y and the advantage from pre-update parameters (D4).

        The advantage is the regression residual y - V: GAE by identity,
        nothing recomputed for the actor.
        """
        t = self._ptr
        with torch.no_grad():
            values = critics(self._obs[:t]).min(dim=0).values
            bootstrap = critics(self._next_obs[:t]).min(dim=0).values
            y = lambda_return(
                self._reward[:t], bootstrap,
                self._terminated[:t], self._truncated[:t],
                cfg.gamma, cfg.lam,
            )
        self.values = values
        self.y = y
        self.adv = y - values

    def minibatches(self, batch_size: int | None = None) -> Iterator[RolloutBatch]:
        if self.y is None or self.adv is None:
            raise ValueError("call compute_targets() before minibatches()")
        t = self._ptr
        batch_size = batch_size or t
        perm = torch.randperm(t)
        for start in range(0, t, batch_size):
            idx = perm[start : start + batch_size]
            yield RolloutBatch(
                obs=self._obs[idx],
                act=self._act[idx],
                y=self.y[idx],
                adv=self.adv[idx],
                anchor_log_prob=self._log_prob[idx],
            )

    def clear(self) -> None:
        self._ptr = 0
        self.y = self.adv = self.values = None
