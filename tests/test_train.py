"""Phase 3: end-to-end smoke on Pendulum-v1 — both regimes complete a real
collect/update/evaluate cycle. Learning quality is the benchmark's concern.
"""

import math
import warnings

from unified_ac import presets
from unified_ac.train import train


class TestEndToEnd:
    def test_replay_regime_on_pendulum(self):
        result = train(
            "Pendulum-v1", presets.sac(), total_steps=300,
            hidden=(32, 32), learning_starts=100, batch_size=32,
        )
        assert math.isfinite(result.final_return)
        assert result.agent._updates == 200

    def test_rollout_regime_on_pendulum(self):
        result = train(
            "Pendulum-v1", presets.ppo(), total_steps=256,
            hidden=(32, 32), rollout_length=128, epochs=2, minibatch_size=32,
        )
        assert math.isfinite(result.final_return)
        assert result.agent._updates == 2

    def test_deterministic_boundary_end_to_end(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = presets.td3()
        result = train(
            "Pendulum-v1", cfg, total_steps=300,
            hidden=(32, 32), learning_starts=100, batch_size=32,
        )
        assert math.isfinite(result.final_return)


class TestEvalLogging:
    def test_history_records_periodic_and_final_evals(self):
        result = train(
            "Pendulum-v1", presets.sac(), total_steps=300,
            hidden=(32, 32), learning_starts=100, batch_size=32,
            eval_every=100, eval_episodes=1,
        )
        steps = [s for s, _ in result.history]
        assert steps == [100, 200, 300, 300]  # 3 periodic + final
        assert all(math.isfinite(r) for _, r in result.history)

    def test_history_has_only_final_without_eval_every(self):
        result = train(
            "Pendulum-v1", presets.ppo(), total_steps=128,
            hidden=(32, 32), rollout_length=128, epochs=1, minibatch_size=32,
        )
        assert len(result.history) == 1

    def test_rollout_regime_records_history(self):
        result = train(
            "Pendulum-v1", presets.ppo(), total_steps=256,
            hidden=(32, 32), rollout_length=128, epochs=1, minibatch_size=32,
            eval_every=128, eval_episodes=1,
        )
        steps = [s for s, _ in result.history]
        assert steps == [128, 256, 256]
