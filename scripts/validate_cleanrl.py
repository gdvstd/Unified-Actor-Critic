"""Curve-level validation: run presets multi-seed on Pendulum-v1 and report
final-return statistics (§8 of PLAN.md). Not CI — scripted and slow.

Usage:
    uv run python scripts/validate_cleanrl.py --preset sac --seeds 3 --steps 15000
    uv run python scripts/validate_cleanrl.py --preset ppo --seeds 3 --steps 40000

Reference points (well-known Pendulum-v1 behavior, ~15-30k steps for
off-policy, ~50-100k for on-policy): random policy ~= -1200 mean return;
a learned policy reaches better than -400, good runs -150..-250.
"""

from __future__ import annotations

import argparse
import statistics
import time

from unified_ac import presets
from unified_ac.train import evaluate, make_env, train

PRESETS = {
    "a2c": presets.a2c,
    "ppo": presets.ppo,
    "ddpg": presets.ddpg,
    "td3": presets.td3,
    "sac": presets.sac,
}

REPLAY_KWARGS = dict(learning_starts=1000, batch_size=256)
ROLLOUT_KWARGS = dict(rollout_length=2048, epochs=10, minibatch_size=64)


def run(preset: str, seeds: int, steps: int, env_id: str) -> None:
    factory = PRESETS[preset]
    returns: list[float] = []
    for seed in range(seeds):
        cfg = factory()
        kwargs = REPLAY_KWARGS if cfg.data == "replay" else ROLLOUT_KWARGS
        start = time.time()
        result = train(env_id, cfg, total_steps=steps, seed=seed, **kwargs)
        # a steadier final estimate than the single eval inside train()
        ret = evaluate(make_env(env_id, seed + 1000), result.agent, episodes=10)
        returns.append(ret)
        print(f"  seed {seed}: return {ret:8.1f}  ({time.time() - start:.0f}s)")
    mean = statistics.mean(returns)
    spread = statistics.stdev(returns) if len(returns) > 1 else 0.0
    print(f"{preset} @ {steps} steps: mean {mean:.1f} +- {spread:.1f}  {returns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--env", default="Pendulum-v1")
    args = parser.parse_args()
    run(args.preset, args.seeds, args.steps, args.env)
