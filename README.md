# Unified-Actor-Critic

A2C, PPO, DDPG, TD3, and SAC as **one algorithm with settings** — the reference
implementation for *"Unifying RL Actors under the Performance Difference Lemma"*.

There are no algorithm classes here. There is one critic objective, one actor
objective, and a frozen config whose toggles select a point in the space the paper
derives. Every named algorithm is a row of a table; the space between the rows is
searchable.

## Recover SAC in 5 lines

```python
from unified_ac.config import UnifiedConfig
from unified_ac.train import train

cfg = UnifiedConfig(alpha=1.0, sig="q", lam=0.0, num_critics=2, eta=0.2,
                    anchor="current", grad="direct", data="replay", tau_actor=1.0)
agent, final_return = train("Pendulum-v1", cfg, total_steps=10_000)
```

That config *is* SAC (also available as `presets.sac()`). Flip `alpha=0, eta=0`,
add `rho=0.2, clip_c=0.5, policy_delay=2, explore_noise=0.1` and it is TD3.
Drop the second critic and the smoothing too and it is DDPG — and the constraint
engine will warn you (`B1`) that you have just built the unhedged point of the
deadly-triad region, which is the paper's thesis in a log line.

## The recovery table

| | data | sig | λ | M | ρ / c | η | α | anchor | grad | trust region | τ_c / τ_a | d | δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2C** | rollout | v | 1 | 1 | 0 | 0 | >0 | current | score | — | frozen-epoch | 1 | — |
| **PPO** | rollout | v | 0.95 | 1 | 0 | 0 | >0 | old | score | ε_clip=0.2 | frozen-epoch | 1 | — |
| **DDPG** | replay | q | 0 | 1 | 0 | 0 | 0 | current | direct | — | 0.005 / 0.005 | 1 | 0.1 |
| **TD3** | replay | q | 0 | 2 | 0.2 / 0.5 | 0 | 0 | current | direct | — | 0.005 / 0.005 | 2 | 0.1 |
| **SAC** | replay | q | 0 | 2 | 0 | 0.2 | >0 | current | direct | — | 0.005 / **1.0** | 1 | — |

Dials: `α` learned stochasticity scale (0 = deterministic boundary), `sig` critic
signature (queryable Q vs precomputed advantage), `λ` bootstrap depth of the
λ-return, `M` twin-critic count, `ρ/c` target policy smoothing, `η` entropy
temperature (shared by critic target and actor objective — C9), `anchor` where the
trust region is pinned, `grad` score vs direct estimator, `τ` Polyak coefficients,
`d` policy delay, `δ` behavior-policy exploration noise (lives in the loop, in no
objective).

## The constraint engine

Not every point in the toggle space is an algorithm. `UnifiedConfig` enforces the
paper's constraint set at construction:

- **Type A (raise)** — definitional: `C1` score needs a density, `C6` direct needs
  the Q signature, `C7` λ>0 needs trajectory-ordered data, `C8` smoothing needs an
  action slot, … Violations produce `InvalidConfigError` with the constraint ID.
- **C4 (inert)** — `anchor=current` makes the ratio 1 and the KL 0; setting them
  anyway warns rather than raises.
- **Type B (warn, never enforce)** — empirical predictions: `B1` deadly-triad
  region, `B2` under-exploration, `B3` redundant smoothing, `B4` query-defenses
  without queries. Enforcing B1 would assume the paper's conclusion, so it is only
  logged — and deliberately runnable.

## Layout

```
unified_ac/
├── config.py         # the toggle space + C1-C9 / B1-B4 as code
├── presets.py        # the recovery table: a2c(), ppo(), ddpg(), td3(), sac()
├── distributions.py  # TanhGaussian + TanhDirac (the deterministic boundary)
├── networks.py       # Actor, CriticEnsemble — shaped by the config, no subclasses
├── targets.py        # λ-return (truncation-aware), bootstrap evaluation B, Polyak
├── signal.py         # Ψ retrieval: min over critics, or GAE as the residual y - V
├── losses.py         # the two boxed objectives; actor J dispatched score/direct
├── buffers.py        # replay (shuffled) vs rollout (trajectory-ordered, frozen y)
├── agent.py          # the GPI loop: Collect → Evaluate → Retrieve → Improve
└── train.py          # environment loop for both regimes
```

## Validation

`tests/test_cleanrl_equivalence.py` proves **exact reduction**: with identical
networks and a fixed batch, our unified objectives reproduce CleanRL's update math
(transcribed with line-numbered provenance in `tests/cleanrl_reference.py`) —
TD3/PPO paths to 1e-6 including gradients, SAC to 1e-4 (their tanh-correction
epsilon vs our exact softplus). Two deviations are deliberate and asserted as such:
the TD3 actor consumes `min(Q1, Q2)` rather than Q1 (the unified form; the paper
argues Q1 is a historical accident), and value-loss scale differs by the constant 2.

```bash
uv sync
uv run pytest                                   # 144 tests, 100% coverage
uv run python scripts/validate_cleanrl.py --preset sac --seeds 3 --steps 15000
```

Single-seed sanity: SAC reaches ≈ -117 on Pendulum-v1 after 10k steps (random
≈ -1200) in about a minute on CPU.

## Status

Phases 0–4 of [docs/PLAN.md](docs/PLAN.md) complete. Next (Phase 5): interior-point
experiments between the named rows, the B1 deadly-triad demonstration, and offline
RL as the pessimism extreme (paper Appendix A).
