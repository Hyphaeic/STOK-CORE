# Changelog

All notable changes to STOK-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note on parity status**: The 1.0.0 entry below describes the state at the time of release (a self-assessment claiming "100/100" paper parity). Subsequent traceability work under `docs/dev/PAPER_*.md` identified gaps and behaviors that did not match the local PDF. Those gaps are tracked as PP-XXX tickets on `docs/dev/PAPER_PDF_TASKBOARD.md` — that board is the authoritative source for actual paper parity going forward. Entries below 1.0.0 record the work after the audit.

---

## [Unreleased]

### Added — M7 option-set builders + final parity report
- **`build_state_option_set`** (PP-701): Theorem 2.2 reusable option-set builder. Returns one `STOKKernel` per BL state — `f_g(x', a) = 1` iff `x' = goal_state`.
- **`build_affordance_option_set`** (PP-701): Theorem 2.3 reusable option-set builder. Returns one `STOKKernel` per affordance HL action — `f_g(x, a) = 1` iff `F(α | x, a) > 0`.
- **`docs/assessments/LOCAL_PDF_PARITY_REPORT.md`** (PP-703): per-item paper-parity verdicts for every numbered Definition / Equation / Theorem / Corollary / Algorithm in the PDF. Tracks the B.1-B.14 resolution table from `PAPER_ASSUMPTIONS.md` and the open follow-ups.
- **`docs/assessments/VERIFICATION_RECORD.md`** (PP-704): exact test counts (282 passing, 0 failing across 8 test binaries + 5 doctests) and example build verification.
- **Examples (PP-702)**: `cargo check --examples` is clean. Existing examples use the reduced low-dim API; the new paper-faithful APIs are exercised by integration tests rather than examples.
- **Scope freeze (PP-705)**: `docs/dev/PAPER_PDF_TASKBOARD.md` "Recent Progress" section explicitly freezes the paper-parity scope; post-paper research is delegated to `docs/EXTENSION_DISCUSSION.md` (with a header disclaimer redirecting to the parity report for current status).

### Added — M6 Algorithm 2 paper-faithful tree search
- **`algorithm_2_search`** (PP-601/602/604): paper-faithful Algorithm 2 over composite states. Takes a `FactorizedGoalKernel` + base `TaskMDP` + initial `ProductState` + `Algorithm2Config`. Expansion follows Algorithm 2 lines 13-19: STOK jump (mode of `FactorizedSTOK::evaluate`), HL boundary update via `boundary_hl_distribution`, BL one-step via `base_mdp.transition` + option policy, time +1 advance, multiplicative κ̃ accumulation. Lines 32-34 (κ̃-max then time-min) implemented as the leaf-set filter.
- **`Algorithm2Node`, `Algorithm2Config`, `Algorithm2Stats`, `Algorithm2Plan`, `Algorithm2Result`** (PP-601): composite-state search machinery. `Algorithm2Node` carries a `ProductState` rather than a flat BL state.
- **`Algorithm2Config.sublimated_feasibility`** (PP-603): when supplied, candidate children whose composite HL state is abstractly infeasible (per Theorem 2.4 contrapositive) are pruned and `Algorithm2Stats.nodes_pruned_by_sublimation` is incremented. The reduced `tree_search` still has its `sublimated_feasibility` field unused — that remains in the reduced/legacy API.
- **Strategy parity** (PP-604): BFS, DFS, and best-first share the same expansion logic. Tests verify they agree on best κ̃ within tolerance.
- **Integration tests** (PP-605): `tests/algorithm_2_tests.rs` (4 tests): feasibility-maximization, search-strategy agreement, sublimation pruning fires on infeasible HL states, time-minimizing tiebreak among κ̃-max plans. Plus 3 inline `algorithm_2_tests` in `tree_search.rs`.

### Added — M5 Goal Kernel & Plan Kernel (paper-faithful planning APIs)
- **`FactorizedGoalKernel`** (PP-501): paper-faithful Goal Kernel `G` from Eq [24]. Carries per-goal `FactorizedSTOK`s, the shared `FactorizedAffordance` `F(α | x, a)`, the HL kernels `P_z_k(z' | z, α_k)` for boundary HL transitions, and the product-space dimensions. Distinct from the existing reduced `GoalKernel` (which remains for low-dim callers).
- **`FactorizedGoalKernel::boundary_hl_distribution`** (PP-502): implements the Eq [24] HL marginalization `Σ_{z_f} P_z(z' | z_f, α_f) · ρ_z(z_f | z, t_f)` per HL space. Returns a `Vec<Tensor<B, 1>>` (one distribution per HL space). `boundary_affordance_prob` returns the affordance factor `F(α | x_f, a_f)` for the boundary BL action.
- **`PlanKernel`, `PlanStep`, `PlanTrace`** (PP-503): the m-fold composition `G_m` of the Goal Kernel under a meta-policy `μ = (o_g1, ..., o_gm)`. Deterministic simulation (`simulate_deterministic`) takes the mode of the Factorized STOK termination at each step plus an argmax HL boundary update; stochastic sampling (`sample`) uses the FactorizedSTOK's sample method. Both return a `PlanTrace` with per-step `(state, time, cumulative_feasibility)`.
- **API split** (PP-504): `src/planning/mod.rs` doc explicitly distinguishes the reduced low-dimensional API (`GoalKernel`, `tree_search`, `best_first_search`) from the paper-faithful one (`FactorizedGoalKernel`, `PlanKernel`). PAPER_API_AUDIT Section 9 updated to reflect the new entry points.
- **Integration tests** (PP-505): `tests/goal_kernel_tests.rs` (6 tests) — assembly from BL STOKs + HL SPKs + product-space dynamics, feasibility routing, boundary HL marginalization sums to 1, deterministic and stochastic plan simulation, empty-sequence identity.

### Added — M3 factorization (general Theorem 2.1 / Eq [23])
- **`assemble_factorized_stok_with_hl_events`** (PP-301/302/303): new entry point that accepts per-HL-space `TaskMDP`s, solves each via `feasibility_iteration`, derives per-space CEFs, the base CEF, and the base SPK. The resulting `FactorizedSTOK` carries optional `base_cef` and `base_spk` fields.
- **`FactorizedSTOK::product_space_tef`**: on-the-fly product-space TEF computation per Theorem 6.1 (`ξ_s(t_f|s) = ∏_k κ̄_k(s_k, t_f-1) - ∏_k κ̄_k(s_k, t_f)`).
- **`FactorizedSTOK::evaluate`** now dispatches between the general Eq [23] path (`ξ · ρ_π · ∏ ρ_z`) and the Cor 6.1 path (`η_πx · ∏ ρ_z`) based on whether CEFs are populated.
- **`FactorizedSTOK::sample`** likewise dispatches: general path samples `t_f` from the product TEF then `x_f` from the base SPK; Cor 6.1 path samples from the BL STOK termination distribution as before.
- **`FactorizedSTOK::feasibility_approx_is_exact`** (PP-304): reports whether `feasibility_approx` is mathematically exact (true under Cor 6.1, false under the general path where it returns the upper bound `κ_base`).
- **Tests**: `tests/factorization_tests.rs` cross-checks Cor 6.1 against direct product-DP within 1e-3 (`chk_co6_1_factorized_matches_direct_dp_no_hl_events`); validates the general-path mass budget and sampling completion.

### Added — M0..M2/M4 + PP-106 partial fix
- **PP-106 partial**: `extract_pi_time_minimizing` now uses a lexicographic key (f_2 primary, time-proxy secondary) so κ-optimal actions with `f_2 = 0` (immediate termination per Eq [8]) win over slower alternatives. Full fix for the ν-uniform corner remains open (needs a structural-progress heuristic).

### Fixed
- **η⁻ over-counting from infeasible initial states** (`src/solver/stok_construction.rs`, PP-105). The η⁻ recursion (Eq [10]) propagated mass from initial states with `κ = 0` even though the boundary (Eq [12]) had already absorbed all mass via the `𝟙̄_κ` branch, producing `Σ η** > 1` along trajectories through multiple infeasible states. Per Appendix 5's absorbing-Markov-chain construction, the row of `B_π` only sums to 1 at infeasible states when `f_2` is treated as 0 there. Fix: gate `f_2` for both η⁺ and η⁻ recursions by feasibility — `f̃_2 = 𝟙_κ · f_2`. Verified by `tests/stok_tests.rs::invariants_constrained_chain`.

### Changed (breaking)
- **`compose_stoks` now preserves η⁺/η⁻ decomposition by default** (PP-201). `ComposedSTOK::eta_plus` and `ComposedSTOK::eta_minus` are now `Tensor<B, 3>` instead of `Option<Tensor<B, 3>>`. The lossy "all mass into η⁺" fallback in `ComposedSTOK::to_stok_kernel` has been removed; the conversion is now lossless. `compose_stoks_with_decomposition` is retained as a `#[deprecated]` alias and will be removed in 0.3.0.
- **`SublimatedTMDP::from_product` signature** now takes an additional `hl_constraint: Option<Tensor<B, 2>>` argument (PP-402). Pass `None` to use `placeholder_hl_constraint_all_free` (renamed from the misleading `extract_hl_constraint`, which is retained as a `#[deprecated]` alias).
- **`maximize_goal_over_base` signature** now takes `n_hl_states: usize` explicitly (PP-401), removing the binary-only hardcoding. Asserts that the affordance dims match the base MDP.

### Added — paper-faithful invariant tests
- **`tests/stok_tests.rs`** (PP-105): 6 cross-module invariant tests over Eq [15], [16], [29] on deterministic and stochastic kernels, plus absorbing-policy enforcement positive and negative cases.
- **`tests/composition_tests.rs`** (PP-203): 9 theorem-style tests for Eq [18-19] — normalization, decomposition identity, κ-η consistency, associativity, sequence↔pairwise equivalence, terminal event semantics, SOK row-stochasticity.
- **`tests/sublimation_tests.rs`** (PP-403/404): 4 tests for Theorem 2.4 on non-binary HL spaces, including a direct-DP cross-check of the bound on a 9-state product witness.
- Inline analytic-case tests for κ-OKBE (PP-102), π-OKBE time-min on κ-tie cases (PP-103), η boundary/recursion (PP-104), Definition 1.1 partition of unity (PP-101).

### Added — agent-facing documentation
- `docs/dev/PAPER_TRACEABILITY_MATRIX.md` — every numbered item in the PDF mapped to code path, test path, status, notes (PP-001).
- `docs/dev/PAPER_ASSUMPTIONS.md` — paper-backed vs repo-local assumptions, with PP-XXX owners (PP-002).
- `docs/dev/PAPER_ACCEPTANCE_CHECKLIST.md` — `CHK-*` pass conditions per theorem/equation/algorithm (PP-003).
- `docs/dev/PAPER_API_AUDIT.md` — public-symbol classification (paper-faithful / reduced / approximate / utility) and required labeling actions (PP-004).

### Known follow-ups (not yet ticketed in the original board)
- **`extract_pi_time_minimizing` κ-tied uniform-ν edge case** (`src/solver/feasibility_iteration.rs`): when all reachable states have `κ = 1` under a self-looping initial policy, ν grows unbounded uniformly across states, so the time-proxy `f_2 · E[ν]` cannot break the κ-tie at non-self-loop alternatives that have `f_2 = 0`. Discovered while writing PP-404 product-DP cross-check; worked around with single-action witnesses. A correct refinement would minimize expected time directly via η⁺ rather than via the ν proxy.

### Milestone status
- M0 (`PP-001..004`) ✅ — paper traceability docs landed.
- M1 (`PP-101..105`) ✅ — OKBE / STOK basics paper-faithful with theorem-style tests. Plus PP-106 partial fix.
- M2 (`PP-201..204`) ✅ — composition preserves η⁺/η⁻ decomposition end-to-end.
- M3 (`PP-301..306`) ✅ — general Theorem 2.1 / Eq [23] path implemented; Cor 6.1 cross-checked against direct product DP.
- M4 (`PP-401..404`) ✅ — sublimation generic across HL cardinalities; Theorem 2.4 bound verified by direct DP.
- M5 (`PP-501..505`) ✅ — `FactorizedGoalKernel` (Eq [24]) + `PlanKernel` (G_m) + reduced/paper-faithful API split.
- M6 (`PP-601..605`) ✅ — `algorithm_2_search` with composite-state expansion, sublimation pruning, BFS/DFS/best-first parity, κ̃-max time-min leaf filter.
- M7 (`PP-701..705`) ✅ — Theorem 2.2/2.3 option-set builders; final parity report; verification record; scope freeze. **All paper-parity milestones complete.**

---

## [1.0.0] - 2026-01-13

### Complete Reference Implementation Achieved (100/100 Paper Parity)

**Implements**: Ringstrom & Schrater (2025), *A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes*, arXiv:2506.09499 [cs.LG]

> The "100/100" claim below represents a self-assessment at release time. The traceability work under [Unreleased] identified specific gaps where the implementation was reduced or approximate relative to the paper; see `docs/dev/PAPER_PDF_TASKBOARD.md` for current parity status.

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
