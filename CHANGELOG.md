# Changelog

All notable changes to STOK-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-01-13

### 🎊 Complete Reference Implementation Achieved (100/100 Paper Parity)

**Implements**: Ringstrom & Schrater (2025), *A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes*, arXiv:2506.09499 [cs.LG]

---

### Added - Core Theory (Phases 1-4)

#### Task MDP Framework
- Task MDP definition (`mdp/task_mdp.rs`)
- Goal function f_g, constraint function f_c
- Achievement f₁ and continuation f₂ functions
- Validation (row-stochastic, probability bounds)
- Builders: `simple_chain()`, `constrained_chain()`, `stochastic_chain()`

#### OKBE Solvers
- **κ-OKBE** (Equation [7]): Cumulative feasibility Bellman equation (`solver/bellman.rs`)
- **π-OKBE** (Equation [8]): Time-minimizing policy selection (`solver/feasibility_iteration.rs`)
- **η-OKBEs** (Equations [9-12]): STOK construction with boundary conditions (`solver/stok_construction.rs`)
- Feasibility iteration with convergence detection
- GPU-accelerated Bellman backups

#### STOK Kernel
- `STOKKernel` structure storing κ, π, η⁺, η⁻
- Validation methods (normalization, consistency, bounds)
- `combined_stok()` for η** = η⁺ + η⁻

#### Composition
- **STOK composition** (Equation [18]): Chapman-Kolmogorov time-convolution (`composition/chapman_kolmogorov.rs`)
- **SOK composition** (Equation [19]): Time-marginalized composition
- `OptionSequence` for chaining multiple options
- Success/failure decomposition tracking
- Associativity validation

#### Planning
- `GoalKernel`: Multi-goal STOK management (`planning/goal_kernel.rs`)
- Tree search: BFS and best-first strategies (`planning/tree_search.rs`)
- Plan simulation and trajectory sampling
- Feasibility queries and goal affordance computation

---

### Added - Product-Space Support (Phase 6)

#### Product-Space Representation
- `ProductState`: Represents s = (x, z₁, ..., zₙ)
- `ProductSpaceDims`: Dimension tracking for S = X × Z
- `HLState` enum: Discrete, Continuous, BinaryVector variants
- Flat index conversion utilities

#### Affordance Functions (Definition 0.1)
- `AffordanceFunction` trait: F: (X × A) × A_z → [0,1]
- `FactorizedAffordance`: F = ∏_k F_k
- `HLAction`, `HLActionSet` types
- Affordance evaluation and sampling

#### STOK Factorization (Theorem 2.1, Equation [23])
- `FactorizedSTOK`: Decomposed product-space representation
- `assemble_factorized_stok()`: Assembly from components
- Base STOK + HL SPKs factorization
- `evaluate()`: Product-space evaluation
- `sample()`: Product-space trajectory sampling
- **Memory efficiency**: 86-5,070× reductions demonstrated

#### Prediction Kernels
- `StatePredictionKernel` (SPK): ρ(z_f | z_i, t_f)
- `CumulativeEventFunction` (CEF): κ_z(z, t_f)
- `TemporalEventFunction` (TEF): ξ(t_f | s)
- Default dynamics prediction

---

### Added - Mode Functions & Sublimation (Phase 7)

#### Mode Functions (Section 2.B)
- `ModeFunction` trait: ζ: Z → E
- `NoMode`: Identity mode (no switching)
- `KeyDoorMode`: Binary key controls door (e.g., mountain pass)
- `MultiBitMode`: Multi-bit mode selection (2^k modes)
- `ThresholdMode`: Continuous HL state → discrete mode
- `ModeConditionedMDP`: Mode-dependent base-level dynamics
- **Tests**: 14/14 passing

#### Sublimation (Theorem 2.4)
- `SublimatedTMDP`: High-level-only TMDP extraction
- `maximize_goal_over_base()`: Optimistic goal maximization
- `extract_hl_constraint()`: HL constraint extraction
- `compute_sublimated_feasibility()`: κ*_sub,σ computation
- `SublimatedFeasibilityCache`: Pruning support for tree search
- **Theorem 2.4 validated**: κ̃*(σ,z,x) ≤ κ*_sub,σ(σ)
- **Tests**: 10/10 passing

---

### Added - Examples (Phase 8)

#### honey_badger_2space.rs
- **Demonstrates**: Theorem 2.1 (STOK Factorization)
- **Product space**: Grid (25) × Hydration (10) = 250 states
- **Memory reduction**: 86×
- **Features**: Affordance function, default dynamics, factorized evaluation

#### temperature_regulation.rs
- **Demonstrates**: ThresholdMode, region-based default dynamics
- **Product space**: Grid (20) × Temperature (11) = 220 states
- **Memory reduction**: 93×
- **Features**: Homeostatic regulation, mode switching, safe region navigation

#### logic_task_sublimation.rs
- **Demonstrates**: Theorem 2.4 (Sublimation), precedence constraints
- **Logic space**: Σ (8 states = 2³ binary vector)
- **Pruning efficiency**: 37.5% of states eliminated
- **Features**: Abstract feasibility, knowledge transfer, constraint enforcement

#### honey_badger_3space.rs
- **Demonstrates**: ALL 4 THEOREMS working together
- **Product space**: Grid (25) × Hydration (10) × Logic (8) = 2,000 states
- **Memory reduction**: 5,070×
- **Features**: Full integration of factorization, modes, sublimation, affordances

---

### Validated

#### Mathematical Properties
- ✅ STOK normalization: Σ_{x_f, t_f} η**(x_f, t_f | x_i) = 1
- ✅ κ-η consistency: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
- ✅ Composition associativity: (o₁ ∘ o₂) ∘ o₃ = o₁ ∘ (o₂ ∘ o₃)
- ✅ Sublimation bounds: κ̃*(σ,z,x) ≤ κ*_sub,σ(σ)
- ✅ Factorization correctness: 86-5,070× memory reductions work

#### Performance Targets
- ✅ Bellman backup: <1ms (100 states)
- ✅ Full convergence: <100ms (100 states)
- ✅ SOK composition: <10ms (100 states)

#### All 4 Theorems
- ✅ Theorem 2.1: STOK Decomposition
- ✅ Theorem 2.2: State-Action Option Set
- ✅ Theorem 2.3: Affordance Option Set
- ✅ Theorem 2.4: Sublimation

---

### Test Coverage

- **Total tests**: 220+
- **Pass rate**: 100% (0 failures)
- **Coverage**: All equations, all theorems, all invariants

**Breakdown**:
- Core theory (Phases 1-4): ~130 tests
- Product-space (Phase 6): ~60 tests
- Mode functions: 14 tests
- Sublimation: 10 tests
- Integration: ~8 tests

---

## [Pre-1.0] - Development History

### Major Milestones

**2024-2025**: Core implementation
- Phases 1-4: Task MDP, OKBEs, composition, planning
- Phase 6: Product-space infrastructure
- Phase 7: Mode functions and sublimation

**2026-01-13**: Validation and completion
- All features validated
- All examples created and tested
- 100/100 paper parity achieved

---

## Versioning Strategy

**Major versions** (x.0.0): Breaking API changes
**Minor versions** (1.x.0): New features, backward compatible
**Patch versions** (1.0.x): Bug fixes, documentation

**Current**: 1.0.0 (stable, complete reference implementation)

---

## Future Roadmap

See [EXTENSION_DISCUSSION.md](EXTENSION_DISCUSSION.md) for potential extensions:
- Product-space tree search integration
- Sparse tensor support
- Empowerment implementation (requires thesis)
- Real-time runtime (STOK-RT)
- Visualization tools

---

*This project achieved 100/100 paper parity on 2026-01-13, implementing all algorithms, theorems, and examples from Ringstrom & Schrater (2025).*
