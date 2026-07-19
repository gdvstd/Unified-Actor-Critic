# Related Work: Existing Implementations & How Our Unification Differs

Survey of how existing RL libraries implement DDPG, TD3, SAC, CQL, A2C, PPO, and how
the unified actor-critic module differs. Companion to [PLAN.md](./PLAN.md).

## 1. Existing library designs

### CleanRL / Spinning Up — duplication by design
- One self-contained file per algorithm (`ddpg_continuous_action.py`,
  `td3_continuous_action.py`, `sac_continuous_action.py`, `ppo_continuous_action.py`);
  Spinning Up mirrors this per-directory.
- Networks, update loop, Polyak averaging re-implemented in every file (buffer borrowed
  from SB3). Philosophy: legibility and reproducibility over reuse.
- Consequence: TD3 and SAC files share ~80% of their logic but the relationship is only
  visible by diffing. Variant research (TD3+entropy, SAC+smoothing) = copy-paste-edit
  with no guarantee only the intended term changed.

### Stable-Baselines3 — inheritance encodes history, not math
- Hierarchy: `BaseAlgorithm → OnPolicyAlgorithm / OffPolicyAlgorithm → {PPO, A2C} /
  {DDPG, TD3, SAC}`. Base classes share *infrastructure* (collection, buffers,
  schedules); the math (Bellman targets, actor loss) is re-implemented in each
  algorithm's `train()`.
- `class DDPG(TD3)`: DDPG is literally TD3 with the tricks disabled via constructor
  defaults — the one axis of the unified space SB3 expresses, and only because
  DDPG ⊂ TD3 historically.
- SAC is a sibling of TD3 in the type system despite differing by exactly our toggles
  (stochastic target policy + entropy vs deterministic + smoothing). The min-twin-critic
  Bellman structure is written twice. A2C and PPO are siblings despite A2C being PPO
  with one epoch and no clipping ("A2C is a special case of PPO", Huang et al. 2022).
- Off-diagonal points unreachable: no entropy knob in TD3's target, no smoothing knob in
  SAC, on-/off-policy wall baked into the class tree.

### Tianshou — deeper inheritance, same limitation
- `TD3Policy(DDPGPolicy)`, `SACPolicy(DDPGPolicy)` with per-subclass hook overrides
  (e.g., `_target_q()`); 2.x separates `Algorithm` from `Policy`.
- Most honest existing attempt at "these algorithms are variations of each other", but
  the variation mechanism is method overriding: the expressible set = the set of
  subclasses someone wrote. No dial between SAC and TD3, only nodes in a tree.

### TorchRL — modular plumbing, monolithic objectives
- `DDPGLoss`, `TD3Loss`, `SACLoss`, `CQLLoss`, `ClipPPOLoss`, `KLPENPPOLoss` as
  swappable `LossModule`s over TensorDict; pluggable value estimators
  (TD(0)/TD(λ)/GAE via `make_value_estimator`); shared `SoftUpdate`/`HardUpdate`.
- What's modularized is the *interface around* the loss; each loss class hardcodes its
  algorithm's objective internally. A library of discrete algorithm-objects with common
  plumbing — not a parameterized objective space.
- d3rlpy: same role for offline RL (CQL/IQL as separate classes + shared factories).
- coax (JAX): most compositional — assemble `td_learning` + policy objectives
  (`DeterministicPG`, `SoftPG`, `ClippedSurrogate`) freehand — but composes rather than
  unifies; no single objective.

### Academic precedents (cite and differentiate in the draft)
- **Expected Policy Gradients** (Ciosek & Whiteson 2018): DPG as a limiting case of
  stochastic PG → our deterministic/stochastic toggle.
- **Interpolated Policy Gradient / Q-Prop** (Gu et al. 2017): interpolate
  score-function/on-policy and critic-based/off-policy gradients → our grad-type and
  anchor toggles.
- **Regularized MDPs** (Geist et al. 2019), **Leverage the Average** (Vieillard et al.
  2020): unify entropy-/KL-regularized value iteration → our α and λ terms.
- **Mirror Learning** (Kuba et al. 2022): PPO/TRPO as one family.
- None produced a single *implemented* objective with exact-reduction tests spanning all
  six algorithms plus the offline/conservative hedge; each unifies one axis theoretically.

## 2. How our design differs

1. **We modularize the objective; existing libraries modularize around it.** Industry
   pattern: shared buffers/loggers/updaters + per-algorithm loss code. Ours: one
   `actor_loss.py` + one `targets.py` where the equation terms (IS ratio, entropy, KL
   anchor, smoothing noise, min-twin, CQL penalty) are toggles and coefficients.
   "Which algorithm?" is a point in a documented config space, not a class name.
2. **Configuration space instead of class tree.** Inheritance expresses supersets along
   historical lineage only (DDPG⊂TD3 works; TD3↔SAC doesn't → duplication). A flat
   `AgentConfig` expresses the full product space including off-diagonal points no
   library reaches without new code: TD3+entropy, SAC+target-smoothing, KL-anchored SAC,
   CQL as a toggle on any off-policy config. Interpolation/ablation between named
   algorithms is a config diff, not a fork.
3. **The on-/off-policy wall comes down at the objective level.** Every surveyed library
   hard-splits these at the top of its architecture. Here the split survives only where
   it is real (buffer type, Ψ estimator); the actor objective is shared, because the PDL
   analysis says the difference is anchor + critic-signal choice.
4. **Correctness anchored by exact-reduction tests, not convention.** No library proves
   its DDPG is TD3-minus-tricks (SB3 asserts by subclassing, CleanRL by convention).
   Phase-4 tests (identical weights + batch → loss match vs CleanRL per preset) turn
   "X is a special case of the unified objective" into a CI assertion — and are the
   mitigation for the classic criticism of unified code paths that CleanRL exists to
   avoid.
5. **Theory-first decomposition.** Existing module boundaries follow engineering seams
   (collector/buffer/loss/updater). Ours follow the paper's two bottlenecks — the
   reliability-hedged critic and the PDL-derived actor — so the code structure *is* the
   argument. Weaker as a production library (TorchRL wins on throughput/breadth),
   stronger as a research and pedagogical instrument.

**Novelty caveat:** closest spirit-match is coax's compositional design; closest theory
matches are EPG/IPG. The defensible claim is the *combination*: single closed-form
objective, 2×2 actor matrix + critic-hedge taxonomy, offline included, exact-reduction
verification.
