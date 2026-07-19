"""Phase 3: end-to-end smoke on Pendulum-v1 — both regimes complete a real
collect/update/evaluate cycle. Learning quality is Phase 4's concern.
"""

import math
import warnings

import pytest

from unified_ac import presets
from unified_ac.train import train


class TestEndToEnd:
    def test_replay_regime_on_pendulum(self):
        agent, ret = train(
            "Pendulum-v1", presets.sac(), total_steps=300,
            hidden=(32, 32), learning_starts=100, batch_size=32,
        )
        assert math.isfinite(ret)
        assert agent._updates == 200

    def test_rollout_regime_on_pendulum(self):
        agent, ret = train(
            "Pendulum-v1", presets.ppo(), total_steps=256,
            hidden=(32, 32), rollout_length=128, epochs=2, minibatch_size=32,
        )
        assert math.isfinite(ret)
        assert agent._updates == 2

    def test_deterministic_boundary_end_to_end(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = presets.td3()
        agent, ret = train(
            "Pendulum-v1", cfg, total_steps=300,
            hidden=(32, 32), learning_starts=100, batch_size=32,
        )
        assert math.isfinite(ret)
