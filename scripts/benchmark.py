"""Phase 5 benchmark harness: named presets + interior points on Gymnasium envs.

Each interior experiment validates a specific claim of the paper (PLAN.md §7
Phase 5): B3 redundancy (sac_smoothed, td3_soft), the DDPG hedge decomposition
/ B1 demonstration (ddpg -> ddpg_twin / ddpg_smooth -> td3), the alpha
interpolation across the deterministic boundary, and the PPO lambda sweep.

Pendulum-v1 is the smoke tier (screening only, no discriminative power);
MuJoCo v5 envs are the evidence tier.

Usage:
    uv run python scripts/benchmark.py --group named --env Pendulum-v1 --steps 10000 --seeds 1
    uv run python scripts/benchmark.py --group all --env Hopper-v5 --steps 300000 --seeds 3
    uv run python scripts/benchmark.py --experiments sac,td3_soft --env HalfCheetah-v5 --steps 100000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from pathlib import Path
from typing import Callable

from unified_ac import presets
from unified_ac.config import UnifiedConfig
from unified_ac.train import train


def _sac_frame(**overrides) -> UnifiedConfig:
    kwargs = dict(
        alpha=1.0, sig="q", lam=0.0, num_critics=2,
        anchor="current", grad="direct", data="replay",
        eta=0.2, tau_actor=1.0,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


def _ppo_frame(**overrides) -> UnifiedConfig:
    kwargs = dict(
        alpha=1.0, sig="v", lam=0.95, num_critics=1,
        anchor="old", grad="score", data="rollout",
        sigma_mode="global", ratio_clip=0.2,
        tau_critic=1.0, tau_actor=1.0,
    )
    kwargs.update(overrides)
    return UnifiedConfig(**kwargs)


EXPERIMENTS: dict[str, Callable[[], UnifiedConfig]] = {
    # ---- named rows ----
    "a2c": presets.a2c,
    "ppo": presets.ppo,
    "ddpg": presets.ddpg,
    "td3": presets.td3,
    "sac": presets.sac,
    # ---- B3: smoothing under a stochastic policy should be redundant ----
    "sac_smoothed": lambda: _sac_frame(rho=0.1, clip_c=0.3),
    "td3_soft": lambda: _sac_frame(  # stochastic TD3: between TD3 and SAC
        eta=0.1, rho=0.2, clip_c=0.5, tau_actor=0.005, policy_delay=2
    ),
    # ---- hedge decomposition: DDPG -> TD3, one hedge at a time (B1 demo) ----
    "ddpg_twin": lambda: _sac_frame(
        alpha=0.0, eta=0.0, tau_actor=0.005, explore_noise=0.1
    ),
    "ddpg_smooth": lambda: _sac_frame(
        alpha=0.0, eta=0.0, num_critics=1, rho=0.2, clip_c=0.5,
        tau_actor=0.005, explore_noise=0.1,
    ),
    # ---- alpha interpolation: the deterministic boundary as a limit ----
    "sac_alpha_03": lambda: _sac_frame(alpha=0.3),
    "sac_alpha_01": lambda: _sac_frame(alpha=0.1),
    "sac_alpha_0": lambda: _sac_frame(  # boundary: eta must vanish (C2)
        alpha=0.0, eta=0.0, tau_actor=0.005, explore_noise=0.1
    ),
    # ---- PPO lambda sweep: the bootstrap-depth dial on-policy ----
    "ppo_lam_0": lambda: _ppo_frame(lam=0.0),
    "ppo_lam_05": lambda: _ppo_frame(lam=0.5),
    "ppo_lam_1": lambda: _ppo_frame(lam=1.0),
}

GROUPS = {
    "named": ["a2c", "ppo", "ddpg", "td3", "sac"],
    "interior": [
        "sac_smoothed", "td3_soft", "ddpg_twin", "ddpg_smooth",
        "sac_alpha_03", "sac_alpha_01", "sac_alpha_0",
        "ppo_lam_0", "ppo_lam_05", "ppo_lam_1",
    ],
    "all": list(EXPERIMENTS),
}

REPLAY_KWARGS = dict(learning_starts=1000, batch_size=256)
ROLLOUT_KWARGS = dict(rollout_length=2048, epochs=10, minibatch_size=64)
# anchor=current cannot correct for stale samples, so A2C is one full-batch
# pass per rollout (its historical form: "PPO with 1 epoch and no clip")
EXTRA_KWARGS: dict[str, dict] = {
    "a2c": dict(epochs=1, minibatch_size=None),
    # CleanRL PPO's norm_adv (training detail, not an objective dial)
    "ppo": dict(normalize_adv=True),
    "ppo_lam_0": dict(normalize_adv=True),
    "ppo_lam_05": dict(normalize_adv=True),
    "ppo_lam_1": dict(normalize_adv=True),
}


def run_one(name: str, env_id: str, steps: int, seed: int,
            eval_every: int, out_dir: Path) -> float:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = EXPERIMENTS[name]()
    constraint_notes = [str(w.message) for w in caught]

    kwargs = dict(REPLAY_KWARGS if cfg.data == "replay" else ROLLOUT_KWARGS)
    kwargs.update(EXTRA_KWARGS.get(name, {}))
    start = time.time()
    result = train(
        env_id, cfg, total_steps=steps, seed=seed,
        eval_every=eval_every, **kwargs,
    )
    elapsed = time.time() - start

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": name,
        "env": env_id,
        "seed": seed,
        "steps": steps,
        "final_return": result.final_return,
        "history": result.history,
        "constraint_notes": constraint_notes,
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / f"{name}_seed{seed}.json").write_text(json.dumps(record, indent=2))
    print(f"  {name} seed {seed}: {result.final_return:9.1f}  ({elapsed:.0f}s)"
          + (f"  [{'; '.join(n.split(':')[0] for n in constraint_notes)}]"
             if constraint_notes else ""))
    return result.final_return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", help="comma-separated experiment names")
    parser.add_argument("--group", choices=sorted(GROUPS), help="named group")
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if args.experiments:
        names = [n.strip() for n in args.experiments.split(",")]
        unknown = [n for n in names if n not in EXPERIMENTS]
        if unknown:
            raise SystemExit(f"unknown experiments: {unknown}")
    elif args.group:
        names = GROUPS[args.group]
    else:
        raise SystemExit("pass --experiments or --group")

    eval_every = args.eval_every or max(1, args.steps // 10)
    out_dir = Path(args.out) / args.env

    summary: dict[str, dict] = {}
    for name in names:
        print(f"[{name}] {args.env} x {args.steps} steps x {args.seeds} seeds")
        finals = [
            run_one(name, args.env, args.steps, seed, eval_every, out_dir)
            for seed in range(args.seeds)
        ]
        summary[name] = {
            "mean": statistics.mean(finals),
            "std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
            "finals": finals,
        }

    print(f"\n=== {args.env} @ {args.steps} steps, {args.seeds} seeds ===")
    width = max(len(n) for n in summary)
    for name, s in sorted(summary.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{name:<{width}}  {s['mean']:9.1f} +- {s['std']:.1f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
