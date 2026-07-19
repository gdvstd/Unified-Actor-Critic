# Implementation Plan: Unified Actor-Critic Training Module

> Source of truth: the paper draft *"Unifying RL Actors under the Performance Difference
> Lemma"* (latest revision, 2026-07). This plan supersedes the previous PLAN.md, which
> used the pre-revision notation (α = entropy temp, β = smoothing, CQL as core preset).
> Companion: [RELATED_WORK.md](./RELATED_WORK.md).

## 1. Goal

One `UnifiedActorCritic` module in which A2C, PPO, DDPG, TD3, and SAC are **rows of a
config table**, not classes. Two boxed objectives from the paper are the only losses in
the codebase:

- **Critic**: `L(φ) = Σ_{i≤M} MSE(C_φᵢ, sg[y^(λ)])`, with the λ-return target and the
  bootstrap evaluation `B(s)` switching on the signature.
- **Actor**: `J(θ) = E[(π_θ/π_anchor)·Ψ] + 𝟙{α>0}·(η·H(π_θ) − λ_KL·KL(π_anchor ‖ π_θ))`,
  with the PPO clip as the alternative realization of the trust-region column.

Success criteria:
1. **Exact reduction** — with ported weights and a fixed batch, each preset's critic
   loss, actor loss, and gradients match CleanRL's implementation within float tolerance.
2. **Behavioral reproduction** — learning curves per preset statistically match CleanRL
   on Pendulum-v1 (mandatory) and one MuJoCo task (optional).
3. **Searchable interior** — any C1–C9-legal toggle combination runs; Type B
   configurations run with logged warnings.

## 2. Notation → code mapping

| Paper | Code | Owner | Domain |
|---|---|---|---|
| α (stochasticity scale) | `alpha` | policy | ≥ 0; 0 = deterministic boundary |
| σ_min (variance floor) | `sigma_min` | policy | ≥ 0; requires `alpha > 0` (C5) |
| sig ∈ {Q, V} | `sig` | critic | `"q"` / `"v"` |
| λ (bootstrap depth) | `lam` | critic | [0, 1] |
| M (critic count) | `num_critics` | critic | {1, 2} |
| ρ, c (target smoothing) | `rho`, `clip_c` | critic | ≥ 0; ρ>0 ⇒ sig=q (C8) |
| η (entropy temperature) | `eta` | **shared** critic+actor (C9) | ≥ 0 |
| anchor ∈ {current, old} | `anchor` | actor | `"current"` / `"old"` |
| grad ∈ {score, direct} | `grad` | actor | direct ⇒ sig=q (C6) |
| λ_KL / ε_clip | `kl_coef` / `ratio_clip` | actor | mutually exclusive realizations |
| data ∈ {rollout, replay} | `data` | loop | λ>0 ⇒ rollout (C7) |
| τ (Polyak, critic & actor) | `tau_critic`, `tau_actor` | loop | SAC: `tau_actor=1` |
| d (delay ratio) | `policy_delay` | loop | ≥ 1 |
| δ (collection noise, behavior policy) | `explore_noise` | loop | in no objective; B2 if 0 with α=0 |

## 3. Design decisions (points the paper leaves open)

Each becomes a documented flag or a fixed convention; none alters the boxed objectives.

| # | Decision | Resolution |
|---|---|---|
| D1 | **Action bounds.** The paper's diagonal Gaussian is unbounded; envs are `[-1,1]^d`. | Tanh-squashed Gaussian as the concrete policy class, with log-prob correction (CleanRL numerics). Deterministic boundary = `tanh(μ_θ)`. Entropy under tanh has no closed form → use `−E[log π]` sample estimate everywhere `H` appears. |
| D2 | **Ψ reduction for the actor.** Paper: `Ψ = min_{i≤M} Q_φᵢ(s,·)`. Real TD3's actor uses Q₁ only; SAC uses min. | **No toggle — Ψ = min always.** The unified form is the min; TD3's Q₁-only actor is a historical accident (Fujimoto et al.'s stated rationale is computational convenience, and the difference is empirically benign). Consequence: our TD3 actor loss deliberately deviates from CleanRL's; equivalence testing is scoped accordingly (§8). The paper should state this explicitly. |
| D3 | **Finite-horizon truncation.** Rollouts end mid-episode, so `y^(λ)` needs a bootstrapped tail `B(s_T)` even at λ=1. | **Truncation-aware everywhere, no flag** — the SB3-standard treatment: the `(1−done)` mask uses `terminated` only; truncated boundaries (TimeLimit, rollout cut, autoreset — using `final_observation`) always append `B(s_T)`. λ=1 means "no bootstrap *within terminated episodes*". One GAE-style backward recursion implements all λ ∈ [0,1]. Note: CleanRL's off-policy scripts do the same; CleanRL PPO conflates truncation with termination mid-rollout — we do not reproduce that. |
| D4 | **Target decoupling has two mechanisms** (Polyak nets under replay; frozen `y^(λ)` per epoch loop under rollouts). | One `TargetProvider` interface, two implementations: `PolyakTargets(τ_critic, τ_actor)` and `FrozenEpochTargets` (computes `y^(λ)`, `Â`, and anchor log-probs once from pre-update params). This is the paper's "one idea: a target that does not move during the fit". |
| D5 | **σ parametrization.** Paper: state-dependent σ_θ(s) scaled by global α. PPO practice: state-independent `log_std` parameter. | `sigma_mode ∈ {"state", "global"}`. SAC preset = `"state"`, PPO/A2C presets = `"global"`. Both are `Σ = α²·diag(σ²) + σ_min²·I` instances. |
| D6 | **C9 folk violation** (actor-only entropy bonus with hard-return critic, shipped A2C/PPO). | `folk_entropy_bonus: float = 0.0`. When > 0, adds `η_folk·H` to J only, emits a logged C9-violation warning. Default off; presets record the coherent η=0 forms per the paper's table. |
| D7 | **η auto-tuning** (SAC v2 target-entropy). | v2 backlog. v1 uses fixed η. |
| D8 | **Value-loss clipping** (CleanRL PPO detail, not in the unified objective). | `value_clip: float | None = None`; enabled only inside the CleanRL comparison harness, never in presets. |
| D9 | **KL estimator** for `kl_coef > 0`. | Closed-form diagonal-Gaussian KL in pre-tanh space (tanh is a bijection; KL is invariant). Documented as such. |

## 4. Config + constraint engine

```python
@dataclass(frozen=True)
class UnifiedConfig:
    # policy
    alpha: float; sigma_min: float; sigma_mode: str
    # critic
    sig: Literal["q", "v"]; lam: float; num_critics: int
    rho: float; clip_c: float; eta: float
    # actor
    anchor: Literal["current", "old"]; grad: Literal["score", "direct"]
    kl_coef: float; ratio_clip: float | None; folk_entropy_bonus: float
    # loop
    data: Literal["rollout", "replay"]
    tau_critic: float; tau_actor: float; policy_delay: int; explore_noise: float
    # generic
    gamma: float = 0.99
```

`validate(cfg)` implements the paper's constraint table **verbatim, with its IDs**:

- **Type A — raise `InvalidConfigError` with the constraint ID**:
  C1 `grad=score ⇒ alpha>0` · C2 `eta>0 ⇒ alpha>0` · C3 `anchor=old ⇒ alpha>0` ·
  C5 `sigma_min>0 ⇒ alpha>0` · C6 `grad=direct ⇒ sig=q` · C7 `lam>0 ⇒ data=rollout` ·
  C8 `rho>0 ⇒ sig=q` · C9 enforced structurally (one `eta` field feeds both objectives;
  only `folk_entropy_bonus` can break coherence, and it logs).
- **Type A — inert, not invalid** (C4): `anchor=current` with `kl_coef>0` or
  `ratio_clip` set → log "inert hyperparameter", short-circuit ratio ≡ 1, KL ≡ 0.
- **Type B — `warnings.warn` + structured log, never raise**:
  B1 `data=replay ∧ grad=direct ∧ num_critics=1 ∧ rho=0 ∧ eta=0` → deadly-triad region ·
  B2 `alpha=0 ∧ explore_noise=0` → under-exploration ·
  B3 `rho>0 ∧ alpha>0` → smoothing redundant ·
  B4 query-defenses with `sig=v` → redundant hedges.

Tests assert every constraint fires on a minimal violating config, and that A2C, PPO,
TD3, SAC construct warning-free. **DDPG is the exception by design**: it is exactly the
unhedged point of the triad region, so constructing it emits B1 — the paper's thesis,
asserted as a test. B1 stays deliberately runnable; it is the thesis experiment.

## 5. Module layout

```
unified_ac/
├── config.py          # UnifiedConfig + validate() (§4) — the constraint set as code
├── presets.py         # a2c(), ppo(), ddpg(), td3(), sac() → UnifiedConfig (§6 table)
├── distributions.py   # TanhGaussian(μ, α·σ, σ_min): rsample/log_prob/entropy_est/kl
│                      # boundary α=σ_min=0 → TanhDirac (log_prob/entropy raise C1/C2)
├── networks.py        # Actor trunk → μ head (+ σ head per sigma_mode);
│                      # CriticEnsemble(M, sig): Q(s,a)×M or V(s)
├── targets.py         # lambda_return(): D3 backward recursion over a rollout;
│                      # bootstrap_B(): the paper's case split — sig=q: min-target-Q with
│                      #   clip(ρζ,±c) smoothing and 𝟙{α>0}·η·log π̄ soft term; sig=v: V_φ̄(s)
│                      # TargetProvider: PolyakTargets | FrozenEpochTargets (D4)
├── signal.py          # Ψ retrieval: sig=q → min over the M online critics (D2);
│                      #   sig=v → Â = y^(λ) − V_φ(s)  (GAE as residual — one subtraction,
│                      #   nothing recomputed; identity asserted in tests)
├── losses.py          # critic_loss(): Σᵢ MSE against sg[y^(λ)]
│                      # actor_loss(): unified J with ratio/clip/KL/entropy assembly,
│                      #   dispatched on grad ∈ {score, direct}
├── buffers.py         # ReplayBuffer (uniform, transitions); RolloutBuffer (trajectory-
│                      #   ordered, stores anchor log-probs; the C7 boundary lives here:
│                      #   only RolloutBuffer can serve lam>0)
├── agent.py           # the unified GPI loop, verbatim from the paper:
│                      #   1 Collect (β = π_θ or tanh(μ)+δ·noise) → 2 Evaluate →
│                      #   3 Retrieve Ψ → 4 Improve every d steps, refresh θ̄
└── train.py           # env loop, eval, seeding, structured logging (incl. B-warnings)
tests/                 # §7
scripts/validate_cleanrl.py   # §8
```

Pure update functions `(params, batch, cfg) → (loss, metrics)`; files ≤ ~300 lines.

## 6. Presets and hyperparameters

The paper's recovery table, plus loop-level parameters and CleanRL-aligned defaults:

| | data | sig | λ | M | ρ / c | η | α | anchor | grad | trust region | τ_c / τ_a | d | δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2C** | rollout | v | 1 | 1 | 0 | 0 | >0 | current | score | — | frozen-epoch | 1 | — |
| **PPO** | rollout | v | 0.95 | 1 | 0 | 0 | >0 | old | score | ε_clip=0.2 | frozen-epoch | 1 | — |
| **DDPG** | replay | q | 0 | 1 | 0 | 0 | 0 | current | direct | — | 0.005 / 0.005 | 1 | 0.1 |
| **TD3** | replay | q | 0 | 2 | 0.2 / 0.5 | 0 | 0 | current | direct | — | 0.005 / 0.005 | 2 | 0.1 |
| **SAC** | replay | q | 0 | 2 | 0 | 0.2 | >0 | current | direct | — | 0.005 / **1.0** | 1 | — |

Non-table defaults: lr 3e-4 (A2C 7e-4; SAC q-lr 1e-3 per CleanRL), batch 256 (replay) /
rollout 2048 with 10 epochs × minibatch 64 (PPO) / n=5-step rollouts (A2C), buffer 1e6,
learning-starts 25k (replay). SAC η auto-tune deferred (D7). TRPO is documented as
PPO's row with a hard constraint — not implemented in v1.

## 7. TDD phases

Write tests first per phase (RED → GREEN → refactor); ≥80% coverage; tiny nets on CPU.

**Phase 0 — config + distributions.** Every preset validates silently; every C1–C8
violation raises with the right ID; C4 logs inert; every B1–B4 combo warns without
raising. TanhGaussian log-prob vs analytic (incl. squash correction); `rsample` carries
grad; boundary α=σ_min=0 raises on density calls; D9 KL matches closed form.

**Phase 1 — targets + critic loss.** `lambda_return` vs hand-computed values on a
3-step trajectory at λ ∈ {0, 0.5, 1}, incl. truncation-tail bootstrap (D3). `bootstrap_B`
sig=q: smoothing noise clipped at ±c, applied only in the target action slot; soft term
present iff α>0 ∧ η>0, evaluated at the *unperturbed* ã. sig=v: bare `V_φ̄`. Polyak
exactness; frozen-epoch targets bit-stable across epochs. λ=1: weight on B vanishes
within completed episodes (hedges provably absent from the loss graph).

**Phase 2 — Ψ + actor loss.** GAE-residual identity `Â = y^(λ) − V_φ(s)` equals the
δ-sum form. Analytic DPG check (quadratic critic, linear actor: autograd = hand chain
rule) at the α=0 boundary. **Estimator agreement**: score and direct estimators agree in
expectation on a fixed smooth Ψ (the unification's empirical heart). anchor=current ⇒
bit-exact with ratio/KL omitted. PPO clip gradient-zero semantics on the clipped branch.
`folk_entropy_bonus` warns and only touches J.

**Phase 3 — buffers + GPI loop.** RolloutBuffer refuses transitions-only serving for
λ>0 (C7 as a runtime guard too); replay FIFO/shapes; `policy_delay` actually delays
actor + θ̄ refresh; **smoke matrix**: one full `update()` for all five presets plus two
interior points (e.g. TD3+η>0, SAC+ρ>0 — B3 warning expected) without error/NaN.

**Phase 4 — CleanRL equivalence (core deliverable).** Loss-level (CI): port weights +
fixed batch into CleanRL's SAC/TD3/PPO update code; assert losses and grads match to
float tolerance. Fixed batches contain **no truncation events**, so the done-handling
difference (D3) never enters. Two documented deviations are asserted *as* deviations:
TD3 actor loss matches CleanRL's only after substituting min for Q₁ in the reference
(one-line change in the harness, per D2); everything else matches verbatim.
Curve-level (scripted, §8).

**Phase 5 — the payoff: benchmark validation of the interior.** Two tiers with
distinct roles. **Smoke tier (Pendulum-v1):** the full experiment matrix at short
horizons, screening for NaN/divergence/config bugs only — Pendulum has no
discriminative power (everything solves it, hedge effects don't manifest), so its
numbers are never evidence. **Evidence tier (Gymnasium MuJoCo v5: Hopper,
HalfCheetah, Walker2d):** where the paper's claims are actually tested — 100-300k
steps × 3 seeds for relative comparisons, 1M × 5 seeds for headline configs later.
The matrix, each row validating a specific claim:
- named presets (baseline rows of the recovery table);
- **TD3+η** (stochastic TD3) and **SAC+ρ** — B3's testable prediction: if smoothing
  under a stochastic policy is redundant, SAC+ρ must not outperform SAC;
- **DDPG hedge decomposition** (DDPG → +M=2 only → +ρ only → TD3) — which hedge
  carries the deadly-triad protection; doubles as the B1 demonstration;
- **α interpolation** (α ∈ {0, 0.1, 0.3, 1} on the SAC frame) — the deterministic
  boundary as a continuous limit, read off a performance curve;
- **PPO λ sweep** (λ ∈ {0, 0.5, 0.95, 1}) — the bootstrap-depth dial on-policy.
Infrastructure: periodic-eval logging in train() (learning curves, not just final
returns), `scripts/benchmark.py` with the experiment registry and JSON results.

**Phase 6 / backlog.** Offline RL as the pessimism extreme (paper Appendix A):
train_offline() (Collect step removed), logged-buffer datasets first, then Minari;
CQL(H) toggle as an additional critic hedge; the B4 substitution experiment
(pessimism vs query restriction as substitutes). Also: η auto-tuning (D7),
vectorized collection, IQL (outside the actor toggles — needs a third consumption
mode).

## 8. Validation harness

**Loss level (D10, implemented as `tests/cleanrl_reference.py`):** instead of importing
the CleanRL package (which drags SB3/tyro dependencies and uses a different sigma
parametrization, so weight-file porting cannot be apples-to-apples), the reference is a
faithful transcription of CleanRL's update math with line-numbered provenance, fetched
from master. Shared between the two sides: network forwards and distribution
parameters. Independent: target assembly, tanh corrections, loss assembly — i.e.
everything the unification claims to reproduce. SAC paths agree to ~1e-4 (their +1e-6
tanh-correction epsilon vs our exact softplus); TD3/PPO paths agree to 1e-6 including
gradients. The D2 (TD3 min-actor) and value-loss-scale (ours = 2 × their 0.5·MSE)
deviations are asserted *as* deviations.

**Curve level:** `scripts/validate_cleanrl.py` runs presets multi-seed on Pendulum-v1
and reports final-return statistics against well-known reference behavior (random
~-1200; learned better than -400). Order: SAC → TD3 → PPO → DDPG → A2C. HalfCheetah-v4
optional when MuJoCo is available.

## 9. Risks

- **Silent math drift** across shared code paths (e.g. soft term leaking into a TD3
  target) — mitigated by Phase 1 loss-graph assertions + Phase 4 exact reduction in CI.
- **Tanh-Gaussian numerics** are the classic SAC divergence source — copy CleanRL's
  clamping, test against it directly (D1).
- **Documented deviations from CleanRL** (D2 min-actor for TD3, D3 truncation-aware
  PPO) mean curve comparison is statistical, not bit-exact — both deviations are
  empirically benign or favorable, but if TD3/PPO curves drift beyond CI overlap, these
  are the first two suspects to ablate.
- **λ-return truncation handling** (D3) is where on-policy implementations silently
  diverge — hand-computed fixtures cover terminated vs truncated boundaries.

## 10. Milestones

1. Phase 0–1: constraint engine + critic path fully unit-tested.
2. Phase 2: estimator-agreement test green — first substantive milestone.
3. Phase 3: five-preset smoke matrix green.
4. Phase 4: SAC exact reduction vs CleanRL, then TD3, PPO.
5. Pendulum curves; then Phase 5 interior experiments (B1 demonstration).

Stack: Python 3.11+, PyTorch, Gymnasium, `uv`; CleanRL pinned for tests only.
