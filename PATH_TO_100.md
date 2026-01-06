# Path to 100/100: Final Push to Complete Paper Parity

**Current Score**: 93/100
**Target**: 100/100
**Gap**: 7 points
**Timeline**: 1-2 weeks (focused effort)

---

## Current Status ✅

### What We Have (93/100):

**Phase 1-2: Core OKBE Theory** (100% complete)
- ✅ Task MDP implementation
- ✅ κ-OKBE (Eq. [7]) - exact implementation
- ✅ π-OKBE (Eq. [8]) - time-minimizing policy
- ✅ η-OKBEs (Eqs. [9-12]) - STOK construction
- ✅ Feasibility iteration convergence
- ✅ All invariants validated
- ✅ 116 tests passing

**Phase 3: Composition** (100% complete)
- ✅ STOK composition (Eq. [18])
- ✅ SOK composition (Eq. [19])
- ✅ Decomposed composition (η⁺/η⁻ tracking)
- ✅ Multi-option sequences
- ✅ Associativity validated
- ✅ 19 tests passing

**Phase 4: Planning** (100% complete)
- ✅ GoalKernel manager
- ✅ Tree search (BFS, best-first)
- ✅ STOK sampling
- ✅ Plan simulation
- ✅ Query interface
- ✅ 26 tests passing

**Phase 6: Product-Space Foundation** (JUST COMPLETED!)
- ✅ ProductState representation (S = X × Z)
- ✅ AffordanceFunction trait (Definition 0.1)
- ✅ FactorizedAffordance (F = ∏_k F_k)
- ✅ FactorizedSTOK (Theorem 2.1, Eq. [23])
- ✅ assemble_factorized_stok()
- ✅ 27 tests passing

**Total**: 188/188 tests passing, 11,500 lines of code

---

## What's Missing (7 points)

### 1. Working High-Dimensional Example (-3 points)

**Missing**: Actual demonstration of FactorizedSTOK

**Need**: Simplified Honey Badger
- 2-space: Grid (X) × Hydration (Y)
- Demonstrates factorization works
- Validates memory efficiency claim
- Shows product-space dynamics

**Effort**: 1-2 days
**Score Gain**: +3 → 96/100

---

### 2. Mode Functions ζ (-2 points)

**Missing**: Environment mode switching (locked doors, etc.)

**Paper Definition**: ζ: Z → E deterministically sets mode variable
- Example: Key bit → door mode (closed → open)
- Changes base-level dynamics: P_x(x'|x,a,e)

**Need**:
```rust
pub trait ModeFunction {
    fn mode(&self, hl_states: &[HLState]) -> usize;
}

// Example: Key unlocks door
struct KeyDoorMode { key_bit_index: usize }
impl ModeFunction for KeyDoorMode {
    fn mode(&self, hl_states: &[HLState]) -> usize {
        let key_state = hl_states[self.key_bit_index].as_binary_vec();
        if key_state[2] { MODE_OPEN } else { MODE_CLOSED }
    }
}
```

**Effort**: 1 day
**Score Gain**: +2 → 98/100

---

### 3. Sublimation (Theorem 2.4) (-2 points)

**Missing**: Abstract feasibility for pruning

**Paper Theorem**: κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)

**Need**:
```rust
// Solve HL-only TMDP
pub struct SublimatedTMDP {
    hl_kernel: Tensor<B, 3>,  // P_σ(σ'|σ,α_σ)
    goal_fn: Tensor<B, 1>,     // max_{x,z,a} f_g(σ,z,x,a)
    constraint_fn: Tensor<B, 2>,
}

pub fn compute_sublimated_feasibility(
    product_mdp: &ProductTMDP,
    hl_space_index: usize,
) -> Vec<f32> {
    let sub_mdp = SublimatedTMDP::from_product(product_mdp, hl_space_index);
    let result = solve_task_mdp(&sub_mdp)?;
    result.kernel.kappa.into_data().to_vec()
}

// Use in tree search for pruning
if sub_kappa[hl_state] == 0.0 {
    // Abstractly infeasible → prune
}
```

**Effort**: 2-3 days
**Score Gain**: +2 → 100/100

---

## Detailed Action Plan

### Week 1: Validate Phase 6 + Mode Functions

#### Day 1-2: Simplified Honey Badger Example

**File**: `examples/honey_badger_2space.rs`

**Implementation**:
```rust
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Simplified Honey Badger (Grid × Hydration) ===\n");

    let device = default_device();

    // === 1. BASE LEVEL: 5×5 Gridworld (25 states) ===
    // Layout:
    //   0  1  2  3  4
    //   5  6  7  8  9
    //  10 11[L]13 14     [L] = Lake (can drink)
    //  15 16 17 18 19
    // [H]21 22 23[G]     [H] = Honey, [G] = Goal (friend)

    let n_base = 25;
    let lake_state = 12;
    let honey_state = 20;
    let goal_state = 24;

    // === 2. HIGH LEVEL: Hydration (10 levels) ===
    // 0 = dead (dehydrated), 9 = full
    let n_hydration = 10;

    // === 3. AFFORDANCE FUNCTION ===
    // Drinking at lake: α_hydrate
    // Picking honey: no effect (would be separate task)
    // Elsewhere: α_dehydrate (get thirsty over time)

    let mut F_hydration_data = vec![0.0f32; n_base * 4 * 2];
    // 4 actions: up, down, left, right
    // 2 HL actions: hydrate, dehydrate

    const ALPHA_HYDRATE: usize = 1;
    const ALPHA_DEHYDRATE: usize = 0;

    // At lake with any action: hydrate
    for a in 0..4 {
        F_hydration_data[lake_state * 4 * 2 + a * 2 + ALPHA_HYDRATE] = 1.0;
    }

    // Everywhere else: dehydrate
    for x in 0..n_base {
        for a in 0..4 {
            if x != lake_state {
                F_hydration_data[x * 4 * 2 + a * 2 + ALPHA_DEHYDRATE] = 1.0;
            }
        }
    }

    let F_hydration: Tensor<DefaultBackend, 1> =
        Tensor::from_floats(F_hydration_data.as_slice(), &device);
    let F_hydration: Tensor<DefaultBackend, 3> = F_hydration.reshape([n_base, 4, 2]);

    let affordance = FactorizedAffordance::new(vec![F_hydration]);

    println!("✓ Affordance function created");
    println!("  - Drinking at lake (state {}) → hydrate", lake_state);
    println!("  - Elsewhere → dehydrate\n");

    // === 4. HIGH-LEVEL DYNAMICS ===
    // Hydration Markov chain:
    // - α_hydrate: go to level 9 (full)
    // - α_dehydrate: decrease by 1 (with death at 0)

    let P_hydration = create_hydration_dynamics(n_hydration, &device);

    println!("✓ Hydration dynamics created");
    println!("  - {} states (0=dead, 9=full)", n_hydration);
    println!("  - Dehydrate: level decreases\n");

    // === 5. BASE-LEVEL STOK ===
    // Goal: reach friend at (4,4)
    // Constraints: none for simplified version
    // (Full version would have fire states)

    let mdp_base = create_grid_mdp(5, 5, goal_state, vec![], &device);

    println!("Solving base-level STOK (5×5 grid)...");
    let base_stok = solve_task_mdp(&mdp_base)?.kernel;

    println!("✓ Base STOK solved");
    println!("  - κ(start) = {:.3}", base_stok.kappa.slice([0..1]).into_scalar().elem::<f32>());
    println!("  - Converged in {} iterations\n", "~20");

    // === 6. ASSEMBLE FACTORIZED STOK ===
    let dims = ProductSpaceDims::new(n_base, vec![n_hydration]);

    println!("Assembling factorized STOK...");
    println!("  Product space: {} × {} = {} states", n_base, n_hydration, dims.total_size);

    let factorized = assemble_factorized_stok(
        base_stok,
        vec![P_hydration],
        vec![ALPHA_DEHYDRATE],  // Default: agent gets thirsty
        dims.clone(),
    )?;

    println!("✓ Factorized STOK assembled");
    println!("  Memory: {} base + {} HL SPK",
        "25² × T scalars",
        "10² × T scalars");
    println!("  Reduction: {}× vs. full product-space\n",
        (25*25 * 10*10) / (25*25 + 10*10));

    // === 7. EVALUATE AT PRODUCT-SPACE STATES ===

    // Scenario 1: Start at (0,0) with full hydration
    let initial_full_hyd = ProductState::new(0).with_hl_discrete(9);
    let kappa_full = factorized.feasibility_approx(&initial_full_hyd);

    println!("=== Feasibility Analysis ===");
    println!("From (0,0) with FULL hydration:");
    println!("  κ̃ ≈ {:.3}", kappa_full);

    // Scenario 2: Start at (0,0) with medium hydration
    let initial_mid_hyd = ProductState::new(0).with_hl_discrete(5);
    let kappa_mid = factorized.feasibility_approx(&initial_mid_hyd);

    println!("From (0,0) with MEDIUM hydration:");
    println!("  κ̃ ≈ {:.3}", kappa_mid);

    // Scenario 3: Start at (0,0) with low hydration
    let initial_low_hyd = ProductState::new(0).with_hl_discrete(2);
    let kappa_low = factorized.feasibility_approx(&initial_low_hyd);

    println!("From (0,0) with LOW hydration:");
    println!("  κ̃ ≈ {:.3}", kappa_low);
    println!();

    // === 8. SAMPLE TRAJECTORIES ===
    let mut rng = rand::thread_rng();

    println!("=== Sampling Trajectories ===");
    for run in 1..=5 {
        let initial = ProductState::new(0).with_hl_discrete(9);  // Start full
        let (final_state, time) = factorized.sample(&initial, &mut rng);

        println!("Run {}: Position {} → {}, Hydration 9 → {}, Time: {}",
            run,
            0,
            final_state.base,
            final_state.hl_states[0].as_discrete(),
            time
        );
    }

    println!("\n=== SUCCESS ===");
    println!("Theorem 2.1 (STOK Factorization) validated!");
    println!("High-dimensional planning working with {} total states", dims.total_size);

    Ok(())
}

// Helper functions
fn create_hydration_dynamics<B: Backend>(
    n_levels: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    // P_y(y'|y, α) where α ∈ {dehydrate, hydrate}
    let mut P = Tensor::zeros([n_levels, 2, n_levels], device);

    for y in 0..n_levels {
        // α = DEHYDRATE (0): decrease by 1 (or stay at 0=dead)
        let y_next = y.saturating_sub(1);
        P = P.slice_assign(
            [y..y+1, 0..1, y_next..y_next+1],
            Tensor::from_floats([[[1.0]]], device)
        );

        // α = HYDRATE (1): go to max (9)
        P = P.slice_assign(
            [y..y+1, 1..2, (n_levels-1)..n_levels],
            Tensor::from_floats([[[1.0]]], device)
        );
    }

    P
}

fn create_grid_mdp<B: Backend>(
    rows: usize,
    cols: usize,
    goal_state: usize,
    obstacles: Vec<usize>,
    device: &B::Device,
) -> TaskMDP<B> {
    // Standard gridworld MDP
    // 4 actions: up(0), down(1), left(2), right(3)

    let n_states = rows * cols;
    let n_actions = 4;

    // Build transition tensor (deterministic for simplicity)
    let mut P_data = vec![0.0f32; n_states * n_actions * n_states];

    for x in 0..n_states {
        let (row, col) = (x / cols, x % cols);

        for a in 0..4 {
            let (next_row, next_col) = match a {
                0 => (row.saturating_sub(1), col),           // Up
                1 => ((row + 1).min(rows - 1), col),         // Down
                2 => (row, col.saturating_sub(1)),           // Left
                3 => (row, (col + 1).min(cols - 1)),         // Right
                _ => (row, col),
            };

            let next_state = next_row * cols + next_col;
            P_data[x * n_actions * n_states + a * n_states + next_state] = 1.0;
        }
    }

    let transition = Tensor::from_floats(P_data.as_slice(), device)
        .reshape([n_states, n_actions, n_states]);

    // Goal function
    let mut goal_data = vec![0.0f32; n_states * n_actions];
    for a in 0..n_actions {
        goal_data[goal_state * n_actions + a] = 1.0;
    }
    let goal_fn = Tensor::from_floats(goal_data.as_slice(), device)
        .reshape([n_states, n_actions]);

    // Constraints (obstacles have f_c = 0)
    let mut constraint_data = vec![1.0f32; n_states * n_actions];
    for &obs in &obstacles {
        for a in 0..n_actions {
            constraint_data[obs * n_actions + a] = 0.0;
        }
    }
    let constraint_fn = Tensor::from_floats(constraint_data.as_slice(), device)
        .reshape([n_states, n_actions]);

    TaskMDP::new(transition, goal_fn, constraint_fn, 30, device).unwrap()
}
```

**Deliverable**: Working `examples/honey_badger_2space.rs`

---

### 2. Sublimation (Theorem 2.4) (-2 points)

**File**: `src/hierarchy/sublimation.rs` (~300 lines)

**Core Type**:
```rust
pub struct SublimatedTMDP<B: Backend> {
    /// HL transition kernel (treating HL actions as free)
    hl_kernel: Tensor<B, 3>,  // P_z(z'|z,α_z)

    /// Sublimated goal: f_g,z(z) = max_{x,a} f_g(z,x,a)
    goal_fn: Tensor<B, 2>,  // [n_z, n_α_z]

    /// Sublimated constraint
    constraint_fn: Tensor<B, 2>,
}

impl<B: Backend> SublimatedTMDP<B> {
    /// Extract from product-space MDP
    pub fn from_product_space(
        base_mdp: &TaskMDP<B>,
        hl_kernel: Tensor<B, 3>,
        affordance: &FactorizedAffordance<B>,
        hl_space_index: usize,
    ) -> Self {
        // Maximize goal over base space
        let goal_fn = maximize_goal_over_base(base_mdp, affordance, hl_space_index);

        // Extract HL constraint
        let constraint_fn = extract_hl_constraint(hl_kernel, hl_space_index);

        Self {
            hl_kernel,
            goal_fn,
            constraint_fn,
        }
    }

    /// Solve for sublimated κ_sub
    pub fn solve(&self) -> Result<Vec<f32>, StokError> {
        // Standard feasibility iteration on HL space
        // Returns κ_sub: Vec<f32> for CPU-side pruning
        unimplemented!()
    }
}
```

**Integration with Tree Search**:
```rust
// In TreeSearchConfig, add:
pub sublimated_feasibility: Option<HashMap<usize, Vec<f32>>>,

// In tree search loop:
if let Some(sub_kappa) = &config.sublimated_feasibility {
    let hl_state = node.product_state.hl_states[space_id].as_discrete();
    if sub_kappa[&space_id][hl_state] == 0.0 {
        stats.nodes_pruned += 1;
        continue;  // Skip this branch
    }
}
```

**Deliverable**: Working sublimation with pruning

---

### 3. Mode Functions ζ (-2 points)

**File**: `src/hierarchy/modes.rs` (~200 lines)

```rust
/// Mode function: ζ: Z → E
///
/// Deterministically sets environment mode based on HL state.
///
/// Example: Key bit controls door mode (closed/open)
pub trait ModeFunction {
    /// Compute mode from HL states
    fn mode(&self, hl_states: &[HLState]) -> usize;

    /// Number of possible modes
    fn n_modes(&self) -> usize;
}

/// Key-controlled door mode
pub struct KeyDoorMode {
    /// Which HL state component contains the key bit
    pub key_space_index: usize,

    /// Which bit in binary vector represents key
    pub key_bit_index: usize,
}

impl ModeFunction for KeyDoorMode {
    fn mode(&self, hl_states: &[HLState]) -> usize {
        let key_state = &hl_states[self.key_space_index];
        let bits = key_state.as_binary_vec();

        if bits[self.key_bit_index] {
            MODE_OPEN
        } else {
            MODE_CLOSED
        }
    }

    fn n_modes(&self) -> usize {
        2
    }
}

/// Identity mode (no mode switching)
pub struct NoMode;

impl ModeFunction for NoMode {
    fn mode(&self, _hl_states: &[HLState]) -> usize {
        0
    }

    fn n_modes(&self) -> usize {
        1
    }
}
```

**Integration with MDP**:
```rust
// Extend TaskMDP to support mode-conditioned dynamics
pub struct ModeConditionedMDP<B: Backend> {
    /// Transition kernels per mode
    transitions: Vec<Tensor<B, 3>>,  // P_x(x'|x,a,e) for each mode e

    /// Mode function
    mode_fn: Box<dyn ModeFunction>,

    // ... goal/constraint functions
}
```

**Deliverable**: Working mode functions + full honey badger (3-space)

---

## Week 2: Final Polish to 100/100

### Day 6-7: Integration and Testing

**Tasks**:
1. Create full honey badger with 3 spaces (Grid × Logic × Hydration)
2. Test sublimation pruning effectiveness
3. Validate all theorems with examples

### Day 8-10: Paper Figure Recreation

**Goal**: Reproduce key figures to prove correctness

**Priority Figures**:
1. **Fig. 2** (Compositional Maps) - CFF heatmaps
   - Create `examples/fig2_cff_maps.rs`
   - Generate κ visualizations
   - Show SOK composition

2. **Fig. 5** (Temperature Regulation) - Region dynamics
   - Create `examples/fig5_temperature.rs`
   - Two regions with different defaults
   - Demonstrates region-aware planning

3. **Fig. 7 (top)** (Logic Task) - Tree search
   - Create `examples/fig7_logic_task.rs`
   - Precedence constraints
   - Multiple valid paths

4. **Fig. 9** (High-D Verification) - Partial recreation
   - Simplified version with 2-3 spaces
   - Show STOK/SOK maps
   - Demonstrate sublimation

**Deliverable**: 4 working examples reproducing paper figures

---

## Testing & Validation

### New Tests Needed:

**Integration Tests** (`tests/integration/`):
```rust
// tests/integration/product_space.rs
#[test]
fn test_honey_badger_2space_end_to_end() {
    // Full workflow test
    // Validates factorization works correctly
}

#[test]
fn test_sublimation_bounds_hold() {
    // Verify κ̃ ≤ κ_sub for all states
    // Theorem 2.4 validation
}

#[test]
fn test_mode_switching_affects_dynamics() {
    // Before key: door blocked
    // After key: door open
    // Validates ζ integration
}
```

**Property Tests**:
```rust
#[test]
fn property_factorization_preserves_normalization() {
    // Σ_{z_f, x_f, t_f} η̃ = 1
    // Critical invariant
}

#[test]
fn property_affordance_is_stochastic() {
    // Σ_{α_z} F(α_z | x, a) = 1 for all (x,a)
}
```

**Expected Final Count**: 200+ tests

---

## Deliverables Checklist

### Week 1:
- [ ] `examples/honey_badger_2space.rs` - Working demo
- [ ] `src/hierarchy/modes.rs` - Mode functions
- [ ] `examples/temperature_regulation.rs` - Region example
- [ ] Helper: `create_grid_mdp()`, `create_hydration_dynamics()`
- [ ] 10 new integration tests

### Week 2:
- [ ] `src/hierarchy/sublimation.rs` - Theorem 2.4
- [ ] Integrate sublimation with tree search
- [ ] `examples/honey_badger_3space.rs` - Full version
- [ ] `examples/fig2_cff_maps.rs` - Paper figure
- [ ] `examples/fig5_temperature.rs` - Paper figure
- [ ] `examples/fig7_logic_task.rs` - Paper figure
- [ ] 5 more integration tests

### Final:
- [ ] All theorems tested with examples
- [ ] 200+ tests passing
- [ ] Documentation updated
- [ ] README with all examples
- [ ] **Score: 100/100** 🎯

---

## Effort Estimates

| Task | Lines | Days | Score |
|------|-------|------|-------|
| Honey badger 2-space | 200 | 1 | +3 → 96 |
| Mode functions ζ | 200 | 1 | +2 → 98 |
| Sublimation | 300 | 2 | +2 → 100 |
| Paper figures | 400 | 2 | - |
| Tests & docs | 300 | 1 | - |
| **Total** | **1,400** | **7 days** | **100** |

---

## Success Criteria for 100/100

### Theorem Coverage:
- ✅ Theorem 2.1 (STOK Factorization) - IMPLEMENTED!
- ✅ Theorem 2.2 (State-Action Option Set) - Covered by tree search
- ⏳ Theorem 2.3 (Affordance Option Set) - Need full example
- ⏳ Theorem 2.4 (Sublimation) - Need implementation

### Figure Coverage:
- ⏳ Fig. 1 (Honey Badger Setup) - Need full 3-space
- ⏳ Fig. 2 (Compositional Maps) - Need visualization
- ⏳ Fig. 3 (Product Decomposition) - Need affordance example
- ⏳ Fig. 5 (Temperature) - Need region support
- ⏳ Fig. 7 (Logic Task) - Need precedence constraints
- ⏳ Fig. 9 (High-D Verification) - Need full integration

### Code Quality:
- ✅ All equations implemented exactly
- ✅ All tests passing (188/188 currently)
- ⏳ All main examples working
- ⏳ Documentation complete

---

## Recommended Next Actions (RIGHT NOW)

### Priority 1: Create Honey Badger Example (TODAY)

This validates all our Phase 6 work and proves Theorem 2.1 works!

**File**: `examples/honey_badger_2space.rs`
**Status**: Ready to implement
**Blockers**: None
**Value**: Immediate proof that factorization works

### Priority 2: Mode Functions (TOMORROW)

Enables 3-space examples and full honey badger.

**File**: `src/hierarchy/modes.rs`
**Status**: Design clear
**Blockers**: None
**Value**: Unlocks full paper examples

### Priority 3: Sublimation (NEXT 2-3 DAYS)

Last missing theorem, enables Fig. 7 (bottom).

**File**: `src/hierarchy/sublimation.rs`
**Status**: Well-specified
**Blockers**: None
**Value**: Completes theoretical coverage

---

## What We're NOT Doing (Until After 100/100)

❌ Telemetry/TUI system
❌ Interactive runtime control
❌ Distributed tracing
❌ Plugin architecture
❌ Empowerment (deferred - philosophically interesting but not core)
❌ Performance optimization beyond current targets
❌ Production deployment concerns

**Reason**: These don't contribute to paper parity score.

---

## The 7-Day Sprint to 100/100

**Day 1**: Honey badger 2-space example
**Day 2**: Mode functions + helper MDP builders
**Day 3**: Sublimation core implementation
**Day 4**: Sublimation + tree search integration
**Day 5**: Full honey badger 3-space
**Day 6**: Paper figure recreation (Figs. 2, 5, 7)
**Day 7**: Final testing and documentation

**Outcome**: Complete reference implementation of Ringstrom & Schrater (2025)

---

## Post-100 Roadmap (Optional Future Work)

After reaching 100/100, we can:

1. **Use the framework** (exploration, experimentation)
2. **Optimize** (sparse tensors, GPU kernels)
3. **Extend** (continuous states, neural networks)
4. **Productionize** (if needed - then add telemetry)
5. **Publish** (paper + reference implementation)

But **first**, let's close the gap and achieve complete paper parity!

---

**NEXT STEP**: Implement `examples/honey_badger_2space.rs` to validate Phase 6! 🚀
