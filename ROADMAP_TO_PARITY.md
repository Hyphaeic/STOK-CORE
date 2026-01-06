# Roadmap to Full Paper Parity: STOK-Core Implementation

**Goal**: Achieve 100% implementation coverage of Ringstrom & Schrater (2025)

**Current Status**: 83/100 (B+)
**Target Status**: 100/100 (A+)

**Timeline**: 3-6 months (part-time) or 6-12 weeks (full-time)

---

## Phase Breakdown

| Phase | Focus | Effort | Score Gain | New Score |
|-------|-------|--------|------------|-----------|
| **Current** | Core single-space | - | - | **83/100** |
| **Phase 5** | Validation & Examples | 2 weeks | +5 | **88/100** |
| **Phase 6** | Product-Space Foundation | 3 weeks | +8 | **96/100** |
| **Phase 7** | High-Dimensional Integration | 2 weeks | +4 | **100/100** |

---

## Phase 5: Validation, Visualization, and Examples (2 weeks)

**Goal**: Prove current implementation is correct and create compelling demos

**Score Gain**: +5 points (88/100)
- +3 for comprehensive testing
- +2 for visualization and examples

### 5.1 Integration Test Suite (Week 1)

**Deliverable**: `tests/integration/` with 20+ new tests

#### 5.1.1 Known Solution Tests (`known_solutions.rs`)

**Purpose**: Validate against hand-calculated examples

```rust
#[test]
fn test_3state_deterministic_composition() {
    // States: 0 -> 1 -> 2
    // Option 1: 0->1 (κ₁(0)=1, E[t]=1)
    // Option 2: 1->2 (κ₂(1)=1, E[t]=1)
    // Composed: 0->2 (κ_μ(0)=1, E[t]=2)

    // Hand-calculated expectations:
    // η_μ(2, t=2 | 0) = 1.0
    // All other (x_f, t_f) should be 0

    let (stok1, stok2) = create_deterministic_chain_options();
    let composed = compose_stoks(&stok1, &stok2)?;

    // Verify time=2, state=2 has all probability mass
    let eta_2_2_0: f32 = composed.eta
        .slice([0..1, 2..3, 2..3])
        .into_scalar()
        .elem();
    assert!((eta_2_2_0 - 1.0).abs() < 1e-6);

    // Verify κ
    assert!((composed.kappa[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_stochastic_composition_known_distribution() {
    // 2-state MDP with known transition probabilities
    // P(1|0,a) = 0.7, P(0|0,a) = 0.3
    // Compose two identical options
    // Hand-calculate expected distribution

    // Expected: κ_μ(0) = 0.7 * 0.7 = 0.49
    let stok = create_stochastic_option(0.7);
    let composed = compose_stoks(&stok, &stok)?;

    assert!((composed.kappa[0] - 0.49).abs() < 1e-4);
}

#[test]
fn test_constraint_composition_failure_propagation() {
    // Option 1: Always fails (κ₁ = 0)
    // Option 2: Always succeeds (κ₂ = 1)
    // Composed: Always fails (κ_μ = 0)

    let stok1 = create_always_fails_option();
    let stok2 = create_always_succeeds_option();
    let composed = compose_stoks(&stok1, &stok2)?;

    // All mass should be in η⁻
    let total_failure: f32 = composed.eta_minus
        .sum_dim(2).squeeze::<2>()
        .sum_dim(1).squeeze::<1>()
        [0].elem();
    assert!((total_failure - 1.0).abs() < 1e-6);
}
```

**Expected Outcome**: 10 new integration tests with analytical solutions

#### 5.1.2 Property-Based Tests (`properties.rs`)

**Purpose**: Validate mathematical properties hold universally

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn property_kappa_monotonic_during_FI(
        n_states in 5usize..20,
        max_time in 5usize..20
    ) {
        // Create random valid MDP
        let mdp = random_valid_mdp(n_states, max_time);

        // Run FI with history
        let config = FeasibilityIterationConfig {
            record_history: true,
            ..Default::default()
        };
        let result = feasibility_iteration(&mdp, config)?;

        // Check monotonicity: κⁱ⁺¹ ≥ κⁱ element-wise
        for (kappa_old, kappa_new) in result.kappa_history.windows(2) {
            for (old, new) in kappa_old.iter().zip(kappa_new) {
                assert!(new >= old - 1e-6);
            }
        }
    }

    #[test]
    fn property_composition_associative(
        n_states in 3usize..10,
        max_time in 3usize..8
    ) {
        let stok = random_normalized_stok(n_states, max_time);

        // (o ∘ o) ∘ o
        let left = {
            let temp = compose_stoks(&stok, &stok)?;
            compose_stoks(&temp.to_stok_kernel()?, &stok)?
        };

        // o ∘ (o ∘ o)
        let right = {
            let temp = compose_stoks(&stok, &stok)?;
            compose_stoks(&stok, &temp.to_stok_kernel()?)?
        };

        // Should be equal within tolerance
        let diff: f32 = (left.eta - right.eta)
            .abs()
            .max()
            .into_scalar()
            .elem();
        assert!(diff < 1e-3);  // Looser tolerance for numerical accumulation
    }

    #[test]
    fn property_sok_composition_is_matmul(
        n_states in 3usize..10
    ) {
        let chi1 = random_stochastic_matrix(n_states);
        let chi2 = random_stochastic_matrix(n_states);

        let sok1 = StateOptionKernel::from_chi(chi1.clone());
        let sok2 = StateOptionKernel::from_chi(chi2.clone());

        let composed = compose_soks(&sok1, &sok2)?;
        let expected = chi1.matmul(chi2);

        let diff: f32 = (composed.chi - expected)
            .abs()
            .max()
            .into_scalar()
            .elem();
        assert!(diff < 1e-5);
    }
}
```

**Expected Outcome**: 5 property-based tests with random inputs

#### 5.1.3 GridWorld Integration Tests (`gridworld_planning.rs`)

```rust
#[test]
fn test_gridworld_10x10_multi_goal_planning() {
    // Create 10×10 grid (100 states)
    // Add 3 goals at different locations
    // Add 5 obstacle states (constraints)

    let mdp1 = TaskMDP::grid_with_goal(10, 10, (2, 3), obstacles);
    let mdp2 = TaskMDP::grid_with_goal(10, 10, (7, 8), obstacles);
    let mdp3 = TaskMDP::grid_with_goal(10, 10, (9, 9), obstacles);

    // Solve
    let stok1 = solve_task_mdp(&mdp1)?.kernel;
    let stok2 = solve_task_mdp(&mdp2)?.kernel;
    let stok3 = solve_task_mdp(&mdp3)?.kernel;

    // Build goal kernel
    let mut kernel = GoalKernel::new(100, device);
    kernel.add_goal(GoalId(0), stok1, "waypoint1", Some(23))?;
    kernel.add_goal(GoalId(1), stok2, "waypoint2", Some(78))?;
    kernel.add_goal(GoalId(2), stok3, "goal", Some(99))?;

    // Search for plan from (0,0) to (9,9)
    let plan = tree_search(&kernel, 0, TreeSearchConfig {
        max_depth: 5,
        target_goal: Some(GoalId(2)),
        ..Default::default()
    }).best_plan.expect("Plan should exist");

    // Validate plan
    assert!(plan.feasibility > 0.0);
    assert!(plan.options.contains(&GoalId(2)));

    // Simulate
    let sim = simulate_plan(&plan, &kernel, 1000, &mut rng);
    println!("Success rate: {:.1}%", sim.success_rate * 100.0);
}

#[test]
fn test_gridworld_stochastic_like_fig2() {
    // Recreate Fig. 2 from paper:
    // - 5×5 grid
    // - 80% intended direction, 6.67% each adjacent
    // - Two goal sets (3 states each)
    // - Constraint states

    let noisy_mdp_goal1 = TaskMDP::noisy_grid(5, 5, 0.8, goal_set_1, constraints);
    let stok1 = solve_task_mdp(&noisy_mdp_goal1)?;

    let noisy_mdp_goal2 = TaskMDP::noisy_grid(5, 5, 0.8, goal_set_2, constraints);
    let stok2 = solve_task_mdp(&noisy_mdp_goal2)?;

    // Compose
    let composed = compose_stoks(&stok1, &stok2)?;

    // Visualize (if visualization tools exist)
    visualize_kappa_map(&stok1, "goal1_kappa.png");
    visualize_kappa_map(&stok2, "goal2_kappa.png");
    visualize_sok_composition(&stok1, &stok2, &composed, "composition.png");
}
```

**Expected Outcome**: 5 gridworld integration tests

---

### 5.2 Visualization Tools (Week 1-2)

**Deliverable**: `src/viz/` module + `examples/visualizations/`

#### 5.2.1 Core Visualization Functions (`src/viz/mod.rs`)

```rust
/// Render κ as heatmap (like Fig. 2 CFF maps)
pub fn visualize_kappa_map(
    stok: &STOKKernel,
    grid_shape: (usize, usize),
    output_path: &str,
) -> Result<(), VizError> {
    // Extract κ to CPU
    let kappa: Vec<f32> = stok.kappa.into_data().to_vec()?;

    // Reshape to 2D grid
    let grid = reshape_to_grid(&kappa, grid_shape);

    // Render as heatmap (yellow=1.0, blue=0.0)
    // Use `plotters` or `image` crate
    save_heatmap(&grid, output_path)
}

/// Render STOK distribution as bar plot (like Fig. 2 bottom row)
pub fn visualize_stok_distribution(
    stok: &STOKKernel,
    initial_state: usize,
    output_path: &str,
) -> Result<(), VizError> {
    // Extract η(·, · | initial_state)
    let eta = stok.combined_stok()
        .slice([initial_state..initial_state+1, :, :]);

    // Create bar plot with:
    // - X-axis: (x_f, t_f) pairs
    // - Y-axis: probability
    // - Colors: green=η⁺, red=η⁻

    save_barplot(&eta, output_path)
}

/// Render SOK as matrix (like Fig. 2 middle row)
pub fn visualize_sok_matrix(
    sok: &StateOptionKernel,
    output_path: &str,
) -> Result<(), VizError> {
    // χ(x_f | x_i) as 2D matrix
    // Green=success, red=failure

    let chi_plus = sok.chi_plus.as_ref();
    let chi_minus = sok.chi_minus.as_ref();

    save_sok_matrix(chi_plus, chi_minus, output_path)
}

/// Render tree search expansion
pub fn visualize_tree_search(
    search_result: &SearchResult,
    output_path: &str,
) -> Result<(), VizError> {
    // Graph showing:
    // - Nodes (states)
    // - Edges (options)
    // - Colors by cumulative feasibility
    // - Best path highlighted

    save_tree_diagram(&search_result, output_path)
}
```

**Dependencies**: Add to Cargo.toml:
```toml
[dependencies]
plotters = "0.3"  # For plots and heatmaps
image = "0.24"    # For image manipulation
```

**Expected Outcome**: Reusable visualization library

#### 5.2.2 Example Visualizations (`examples/visualizations/`)

```rust
// examples/visualizations/fig2_recreation.rs
fn main() -> Result<(), Box<dyn Error>> {
    // Recreate simplified Fig. 2
    let device = default_device();

    // 5×5 noisy grid
    let grid_mdp_1 = create_noisy_grid_mdp(5, 5, 0.8, goal_set_1, constraints);
    let stok1 = solve_task_mdp(&grid_mdp_1)?;

    let grid_mdp_2 = create_noisy_grid_mdp(5, 5, 0.8, goal_set_2, constraints);
    let stok2 = solve_task_mdp(&grid_mdp_2)?;

    // Visualize individual κ maps
    visualize_kappa_map(&stok1, (5, 5), "output/fig2_kappa_goal1.png")?;
    visualize_kappa_map(&stok2, (5, 5), "output/fig2_kappa_goal2.png")?;

    // Compose and visualize
    let sok1 = StateOptionKernel::from_stok(&stok1);
    let sok2 = StateOptionKernel::from_stok(&stok2);
    let composed_sok = compose_soks(&sok1, &sok2)?;

    visualize_sok_matrix(&composed_sok, "output/fig2_composed_sok.png")?;

    // STOK bar plots
    visualize_stok_distribution(&stok1, 0, "output/fig2_stok1_dist.png")?;
    visualize_stok_distribution(&composed_sok.to_stok()?, 0, "output/fig2_composed_dist.png")?;

    println!("Generated Fig. 2 visualizations in output/");
    Ok(())
}
```

**Expected Outcome**: Reproducible paper-style figures

---

### 5.3 Example Gallery (`examples/`) (Week 2)

#### 5.3.1 Tutorial Examples

**File**: `examples/tutorial/01_hello_stok.rs`
```rust
//! Simplest possible STOK example
fn main() {
    // 3-state chain: 0 -> 1 -> 2 (goal)
    let device = default_device();
    let mdp = TaskMDP::simple_chain(3, 5, &device);

    // Solve for STOK
    let result = solve_task_mdp(&mdp).unwrap();
    println!("Feasibility from state 0: {}", result.kernel.kappa[0]);
    println!("Converged in {} iterations", result.convergence.iteration);

    // Expected: κ* = [1, 1, 1] (all states can reach goal)
}
```

**File**: `examples/tutorial/02_composition.rs`
```rust
//! Composing two options
fn main() {
    // Create two sub-tasks
    let device = default_device();

    // Task 1: Navigate to waypoint (state 5)
    let mdp1 = create_chain_with_goal(10, 5);
    let stok1 = solve_task_mdp(&mdp1)?.kernel;

    // Task 2: Navigate from waypoint to goal (state 9)
    let mdp2 = create_chain_with_goal(10, 9);
    let stok2 = solve_task_mdp(&mdp2)?.kernel;

    // Compose
    let composed = compose_stoks(&stok1, &stok2)?;

    println!("Option 1: κ(0) = {}", stok1.kappa[0]);
    println!("Option 2: κ(5) = {}", stok2.kappa[5]);
    println!("Composed: κ_μ(0) = {}", composed.kappa[0]);
}
```

**File**: `examples/tutorial/03_planning.rs`
```rust
//! Multi-goal planning with tree search
fn main() {
    // Build goal kernel with 3 goals
    let mut kernel = GoalKernel::new(20, device);
    // ... add goals ...

    // Search
    let result = tree_search(&kernel, 0, TreeSearchConfig {
        max_depth: 5,
        target_goal: Some(GoalId(2)),
        ..Default::default()
    });

    if let Some(plan) = result.best_plan {
        println!("Found plan: {}", plan);
        println!("Feasibility: {:.3}", plan.feasibility);

        // Simulate
        let sim = simulate_plan(&plan, &kernel, 1000, &mut rng);
        println!("Simulated success rate: {:.1}%", sim.success_rate * 100.0);
    }
}
```

**Expected Outcome**: 5 tutorial examples (increasing complexity)

#### 5.3.2 Application Examples

**File**: `examples/applications/waypoint_delivery.rs`
```rust
//! Package delivery with ordered waypoints
//! Demonstrates: composition with precedence constraints

fn main() {
    // Scenario: Robot must visit 4 locations in order
    // Locations: [Depot(0), CustomerA(10), CustomerB(25), CustomerC(40), Depot(0)]

    // Solve individual segments
    let stok_0_to_A = solve_navigation(0, 10)?;
    let stok_A_to_B = solve_navigation(10, 25)?;
    let stok_B_to_C = solve_navigation(25, 40)?;
    let stok_C_to_depot = solve_navigation(40, 0)?;

    // Compose full route
    let route = OptionSequence::new()
        .then(stok_0_to_A, Some("pickup_A"))
        .then(stok_A_to_B, Some("deliver_A_pickup_B"))
        .then(stok_B_to_C, Some("deliver_B_pickup_C"))
        .then(stok_C_to_depot, Some("deliver_C_return"));

    let full_route = compose_sequence(&route)?;

    println!("Full route feasibility: {:.3}", full_route.kappa[0]);
    println!("Expected time: {:.1} steps", expected_time(&full_route, 0));
}
```

**File**: `examples/applications/constrained_navigation.rs`
```rust
//! Navigation with hazard avoidance
//! Demonstrates: constraint handling, infeasible region identification

fn main() {
    // 10×10 grid with lava fields
    let obstacles = vec![(3,3), (3,4), (3,5), (4,3), (4,4), (4,5)];  // 2×3 lava field

    let mdp = TaskMDP::grid_with_obstacles(10, 10, goal=(9,9), obstacles);

    // Solve
    let result = solve_task_mdp(&mdp)?;

    // Visualize which states are feasible
    visualize_kappa_map(&result.kernel, (10, 10), "feasibility_map.png")?;

    // Expected: states near lava have lower κ
    // States on opposite side of lava from goal may have κ = 0
}
```

**Expected Outcome**: 3-5 application examples

---

### 5.4 Performance Benchmarking (Week 2)

**File**: `benches/paper_claims.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_paper_claim_single_backup(c: &mut Criterion) {
    // Paper claim: <1ms for 100 states
    let mut group = c.benchmark_group("paper_claim_bellman");

    group.bench_function("100_states", |b| {
        let device = default_device();
        let mdp = TaskMDP::simple_chain(100, 50, &device);
        let kappa = Tensor::zeros([100], &device);

        b.iter(|| {
            bellman_backup_kappa(&kappa, &mdp)
        });
    });

    group.finish();
}

fn bench_paper_claim_sok_composition(c: &mut Criterion) {
    // Paper claim: <10ms for S=100
    let mut group = c.benchmark_group("paper_claim_sok_comp");

    for n_states in [50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("states", n_states),
            n_states,
            |b, &n| {
                let device = default_device();
                let mdp = TaskMDP::simple_chain(n, 20, &device);
                let stok = solve_task_mdp(&mdp).unwrap().kernel;
                let sok = StateOptionKernel::from_stok(&stok);

                b.iter(|| {
                    compose_soks(&sok, &sok).unwrap()
                });
            }
        );
    }

    group.finish();
}

fn bench_paper_claim_stok_composition(c: &mut Criterion) {
    // Paper claim: <100ms for S=50, T=20
    let mut group = c.benchmark_group("paper_claim_stok_comp");

    group.bench_function("S50_T20", |b| {
        let device = default_device();
        let mdp = TaskMDP::simple_chain(50, 20, &device);
        let stok = solve_task_mdp(&mdp).unwrap().kernel;

        b.iter(|| {
            compose_stoks(&stok, &stok).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    paper_claims,
    bench_paper_claim_single_backup,
    bench_paper_claim_sok_composition,
    bench_paper_claim_stok_composition
);
criterion_main!(paper_claims);
```

**Expected Outcome**: Performance validation report

---

### 5.5 Documentation (`docs/`) (Week 2)

**Files to Create**:

1. **`docs/THEORY_TO_CODE.md`** - Equation mapping
   ```markdown
   | Paper Equation | Code Location | Notes |
   |----------------|---------------|-------|
   | [7] κ-OKBE | solver/bellman.rs:45 | Batched GPU impl |
   | [18] Composition | composition/chapman_kolmogorov.rs:200 | Time-convolution |
   ```

2. **`docs/USER_GUIDE.md`** - Complete tutorial
3. **`docs/PERFORMANCE.md`** - Optimization guide
4. **`README.md`** - Update with Phase 3-4 features

---

## Phase 6: Product-Space Foundation (3 weeks)

**Goal**: Enable high-dimensional STOK factorization

**Score Gain**: +8 points (96/100)
- +5 for product-space support
- +3 for affordance function

### 6.1 Product-Space State Representation (Week 1)

**Deliverable**: `src/hierarchy/product_space.rs`

#### 6.1.1 Core Types

```rust
/// High-level state component
#[derive(Clone, Debug)]
pub enum HLState {
    /// Discrete state (e.g., binary logic)
    Discrete(usize),
    /// Continuous discretized state (e.g., hydration level)
    Continuous(f32),
    /// Binary vector (e.g., task completion flags)
    BinaryVector(Vec<bool>),
}

/// Product-space state: s = (x, z₁, z₂, ..., zₙ)
#[derive(Clone, Debug)]
pub struct ProductState {
    /// Base-level state (X)
    pub base: usize,

    /// High-level state components (Z₁, ..., Zₙ)
    pub hl_states: Vec<HLState>,
}

impl ProductState {
    pub fn new(base: usize) -> Self {
        Self {
            base,
            hl_states: vec![],
        }
    }

    pub fn with_hl_discrete(mut self, state: usize) -> Self {
        self.hl_states.push(HLState::Discrete(state));
        self
    }

    pub fn with_hl_binary_vector(mut self, bits: Vec<bool>) -> Self {
        self.hl_states.push(HLState::BinaryVector(bits));
        self
    }

    /// Convert to flat index (for tensor operations)
    pub fn to_flat_index(&self, dims: &ProductSpaceDims) -> usize {
        // Compute index in flattened product space
        let mut index = self.base;
        let mut stride = dims.base_size;

        for (hl_state, hl_size) in self.hl_states.iter().zip(&dims.hl_sizes) {
            let hl_idx = match hl_state {
                HLState::Discrete(i) => *i,
                HLState::BinaryVector(bits) => binary_to_index(bits),
                HLState::Continuous(v) => discretize(*v, *hl_size),
            };
            index += hl_idx * stride;
            stride *= hl_size;
        }

        index
    }

    /// Convert from flat index
    pub fn from_flat_index(index: usize, dims: &ProductSpaceDims) -> Self {
        // Inverse of to_flat_index
        // Extract base and each HL component
        unimplemented!("Conversion from flat index")
    }
}

/// Product-space dimensions
#[derive(Clone, Debug)]
pub struct ProductSpaceDims {
    /// Base-level state space size
    pub base_size: usize,

    /// High-level state space sizes
    pub hl_sizes: Vec<usize>,

    /// Total product-space size: |X| × |Z₁| × ... × |Zₙ|
    pub total_size: usize,
}

impl ProductSpaceDims {
    pub fn new(base_size: usize, hl_sizes: Vec<usize>) -> Self {
        let total_size = base_size * hl_sizes.iter().product::<usize>();
        Self {
            base_size,
            hl_sizes,
            total_size,
        }
    }

    pub fn n_hl_spaces(&self) -> usize {
        self.hl_sizes.len()
    }
}
```

**Tests**:
- Flat index conversion round-trip
- Product-space size calculation
- HLState serialization

---

### 6.2 Affordance Function (Week 1-2)

**Deliverable**: `src/hierarchy/affordance.rs`

#### 6.2.1 Core Trait

```rust
/// Affordance function: F(α_z | x, a) ∈ [0,1]
///
/// Links base-level state-actions to high-level actions.
///
/// Paper Definition 0.1:
/// F(α_z | x, a) = ∏_k F_k(α_{z_k} | x, a)
pub trait AffordanceFunction<B: Backend> {
    /// Compute P(α | x, a)
    fn probability(&self, hl_action: &HLAction, base_state: usize, base_action: usize) -> f32;

    /// Get support: which HL actions have non-zero probability?
    fn support(&self, base_state: usize, base_action: usize) -> Vec<HLAction>;

    /// Sample HL action from distribution
    fn sample(&self, base_state: usize, base_action: usize, rng: &mut impl Rng) -> HLAction;
}

/// High-level action
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HLAction {
    /// Component action for space k
    Component {
        space_id: usize,
        action_id: usize,
    },
    /// Vector of component actions (α_z₁, α_z₂, ...)
    Vector(Vec<usize>),
}
```

#### 6.2.2 Concrete Implementation

```rust
/// Factorized affordance: F = ∏_k F_k
pub struct FactorizedAffordance<B: Backend> {
    /// Component affordances F_k(α_{z_k} | x, a)
    /// Each has shape [n_base_states, n_base_actions, n_hl_actions_k]
    components: Vec<Tensor<B, 3>>,

    base_dims: (usize, usize),  // (n_states, n_actions)
    hl_action_sizes: Vec<usize>,
}

impl<B: Backend> FactorizedAffordance<B> {
    /// Create from component tensors
    pub fn new(components: Vec<Tensor<B, 3>>) -> Self {
        // Validate shapes match
        // Extract dimensions
        unimplemented!()
    }

    /// Evaluate F(α | x, a) = ∏_k F_k(α_k | x, a)
    pub fn evaluate(&self, hl_action: &HLAction, x: usize, a: usize) -> f32 {
        match hl_action {
            HLAction::Vector(alphas) => {
                let mut prob = 1.0f32;
                for (k, &alpha_k) in alphas.iter().enumerate() {
                    let component_prob: f32 = self.components[k]
                        .clone()
                        .slice([x..x+1, a..a+1, alpha_k..alpha_k+1])
                        .into_scalar()
                        .elem();
                    prob *= component_prob;
                }
                prob
            }
            _ => unimplemented!(),
        }
    }
}

impl<B: Backend> AffordanceFunction<B> for FactorizedAffordance<B> {
    fn probability(&self, hl_action: &HLAction, base_state: usize, base_action: usize) -> f32 {
        self.evaluate(hl_action, base_state, base_action)
    }

    fn support(&self, base_state: usize, base_action: usize) -> Vec<HLAction> {
        // Return all HL actions with prob > 0
        // Iterate through Cartesian product of component action spaces
        unimplemented!("Support computation")
    }

    fn sample(&self, x: usize, a: usize, rng: &mut impl Rng) -> HLAction {
        // Sample from factorized distribution
        let mut sampled = vec![];
        for component in &self.components {
            let dist: Vec<f32> = component
                .clone()
                .slice([x..x+1, a..a+1, :])
                .reshape([component.dims()[2]])
                .into_data()
                .to_vec()
                .unwrap();
            sampled.push(sample_categorical(&dist, rng));
        }
        HLAction::Vector(sampled)
    }
}
```

**Example Usage**:
```rust
// Honey badger: drinking at lake causes hydration
let mut F_hydration = Tensor::zeros([n_states, n_actions, 2], &device);  // 2 HL actions: hydrate/dehydrate

// At lake location with drink action: P(α_hyd | x_lake, a_drink) = 1.0
F_hydration[lake_state][drink_action][ALPHA_HYDRATE] = 1.0;

// Elsewhere: P(α_dehyd | x, a) = 1.0 (agent gets thirstier)
for x in 0..n_states {
    for a in 0..n_actions {
        if x != lake_state || a != drink_action {
            F_hydration[x][a][ALPHA_DEHYDRATE] = 1.0;
        }
    }
}

let affordance = FactorizedAffordance::new(vec![F_hydration]);
```

**Tests**:
- Factorization: F = ∏_k F_k
- Sampling from affordance
- Support computation

---

### 6.3 Composition Function λ (Week 2)

**Deliverable**: `src/hierarchy/composition_fn.rs`

#### 6.3.1 Implementation

```rust
/// Composition function: builds product-space kernel
///
/// Paper Definition 2.1:
/// P_s(s'|s,a) = λ(P_z, F^z_x, ζ, P_x)
///             = Σ_{α_z,e} P_z(z'|z,α_z) F^z_x(α_z|x,a) P_x(x'|x,a,e) ζ(e|z)
pub fn compose_kernels<B: Backend>(
    P_x: &Tensor<B, 3>,           // Base kernel [n_x, n_a, n_x]
    P_z: &Vec<Tensor<B, 3>>,      // HL kernels [n_z_k, n_α_k, n_z_k]
    F: &FactorizedAffordance<B>,  // Affordance function
    zeta: Option<&ModeFunction>,  // Optional mode function
) -> Tensor<B, 3> {  // Product-space kernel [n_s, n_a, n_s]

    let n_x = P_x.dims()[0];
    let n_a = P_x.dims()[1];

    // Compute product-space size
    let n_s = n_x * P_z.iter().map(|p| p.dims()[0]).product::<usize>();

    // Initialize product-space kernel
    let mut P_s = Tensor::zeros([n_s, n_a, n_s], &P_x.device());

    // For each (s, a, s') tuple:
    // P_s(s'|s,a) = Σ_{α_z} ∏_k P_{z_k}(z'_k|z_k,α_{z_k}) × F(α_z|x,a) × P_x(x'|x,a,e)

    for s_i in 0..n_s {
        let state_i = ProductState::from_flat_index(s_i, &product_dims);

        for a in 0..n_a {
            // Get HL action distribution from affordance
            let hl_actions = F.support(state_i.base, a);

            for hl_action in hl_actions {
                let F_prob = F.probability(&hl_action, state_i.base, a);

                // Compute mode (if mode function exists)
                let e = match zeta {
                    Some(zf) => zf.mode(&state_i.hl_states),
                    None => 0,
                };

                // Iterate over next states
                for s_j in 0..n_s {
                    let state_j = ProductState::from_flat_index(s_j, &product_dims);

                    // Compute transition probability
                    let mut prob = F_prob;

                    // Base-level transition
                    prob *= P_x_at_mode(P_x, state_i.base, a, state_j.base, e);

                    // HL transitions
                    prob *= hl_transition_prob(P_z, &state_i.hl_states, &hl_action, &state_j.hl_states);

                    // Accumulate
                    P_s[s_i][a][s_j] += prob;
                }
            }
        }
    }

    P_s
}
```

**Note**: This is computationally expensive (O(|S|² × |A|)). For large product spaces, need sparse implementation.

---

### 6.4 STOK Factorization Assembly (Week 2-3)

**Deliverable**: `src/hierarchy/factorization.rs`

#### 6.4.1 Factorized STOK Type

```rust
/// Factorized STOK representation
///
/// Implements Theorem 2.1, Equation [23]:
/// η̃(z_f, x_f, t_f | z, x) = ξ(t_f | z, x) · ρ_π(x_f | x, t_f) · ∏_k ρ_k(z_k,f | z_k, t_f)
pub struct FactorizedSTOK<B: Backend> {
    /// Base-level STOK: η_π(x_f, t_f | x)
    pub base_stok: STOKKernel<B>,

    /// High-level SPKs: ρ_k(z_k,f | z_k, t_f) for each space Z_k
    pub hl_spks: Vec<StatePredictionKernel<B>>,

    /// Temporal event function: ξ(t_f | z, x)
    pub tef: Option<TemporalEventFunction<B>>,

    /// Cumulative event functions (one per space)
    pub cefs: Vec<CumulativeEventFunction<B>>,

    /// Metadata
    pub product_dims: ProductSpaceDims,
}

impl<B: Backend> FactorizedSTOK<B> {
    /// Evaluate factorized STOK at a product-space state-time
    ///
    /// Returns η̃(z_f, x_f, t_f | z_i, x_i)
    pub fn evaluate(
        &self,
        initial: &ProductState,
        final_state: &ProductState,
        time: usize,
    ) -> f32 {
        // Base-level component
        let eta_base: f32 = self.base_stok.combined_stok()
            .slice([initial.base..initial.base+1, final_state.base..final_state.base+1, time..time+1])
            .into_scalar()
            .elem();

        // High-level SPK components
        let mut hl_prob = 1.0f32;
        for (k, spk) in self.hl_spks.iter().enumerate() {
            let z_i = initial.hl_states[k].as_discrete();
            let z_f = final_state.hl_states[k].as_discrete();

            let rho_k: f32 = spk.predict(z_i, time)[z_f].elem();
            hl_prob *= rho_k;
        }

        // TEF component (if available)
        let tef_prob = self.tef.as_ref()
            .map(|t| t.event_at_time(initial.flat_index(), time))
            .unwrap_or(1.0);

        eta_base * hl_prob * tef_prob
    }

    /// Sample from factorized STOK
    pub fn sample(
        &self,
        initial: &ProductState,
        rng: &mut impl Rng,
    ) -> (ProductState, usize) {
        // 1. Sample base-level (x_f, t_f) from base STOK
        let (x_f, t_f) = STOKSampler::sample_termination(&self.base_stok, initial.base, rng);

        // 2. For each HL space, predict z_f using SPK
        let mut hl_final = vec![];
        for (k, (spk, z_i)) in self.hl_spks.iter().zip(&initial.hl_states).enumerate() {
            let z_f = spk.sample(z_i.as_discrete(), t_f, rng);
            hl_final.push(HLState::Discrete(z_f));
        }

        let final_state = ProductState {
            base: x_f,
            hl_states: hl_final,
        };

        (final_state, t_f)
    }

    /// Get feasibility at product-space state
    pub fn feasibility(&self, state: &ProductState) -> f32 {
        // Base feasibility
        let kappa_base: f32 = self.base_stok.kappa[state.base].elem();

        // Modulated by HL event probability (if CEFs available)
        let mut hl_modulation = 1.0f32;
        // Compute κ̄_z (no HL events) across all spaces
        // This requires integrating over time...

        kappa_base * hl_modulation
    }
}
```

#### 6.4.2 Construction from Components

```rust
/// Build factorized STOK from components
///
/// # Arguments
///
/// * `base_stok` - STOK for base-level space X
/// * `hl_kernels` - Transition kernels for each HL space (with default actions)
/// * `affordance` - Affordance function linking base to HL
/// * `product_dims` - Product-space dimensions
///
/// # Returns
///
/// Factorized STOK ready for evaluation/sampling
pub fn build_factorized_stok<B: Backend>(
    base_stok: STOKKernel<B>,
    hl_kernels: Vec<Tensor<B, 3>>,  // P_z_k(z'|z,α) for each space
    default_actions: Vec<usize>,     // Default α^ℓ for each space
    max_time: usize,
) -> Result<FactorizedSTOK<B>, StokError> {
    // 1. Build SPKs from HL kernels with default actions
    let mut hl_spks = vec![];
    for (k, (P_z_k, α_default)) in hl_kernels.iter().zip(&default_actions).enumerate() {
        // Extract P_z_k(:, :, α_default) as default Markov chain
        let P_default = P_z_k.slice([:, :, *α_default]);
        let spk = StatePredictionKernel::new(P_default, max_time);
        hl_spks.push(spk);
    }

    // 2. Compute CEFs for each HL space
    // (This requires solving HL-only STOKs - see sublimation)
    let cefs = vec![];  // Placeholder

    // 3. Assemble
    Ok(FactorizedSTOK {
        base_stok,
        hl_spks,
        tef: None,  // Computed on-demand
        cefs,
        product_dims: unimplemented!(),
    })
}
```

**Tests**:
- Build factorized STOK from components
- Evaluate at product-space states
- Sample trajectories

---

### 6.5 Example: Simplified Honey Badger (Week 3)

**Deliverable**: `examples/honey_badger_simplified.rs`

**Simplifications**:
- Use 5×5 grid (not full map)
- Single HL space: hydration only (no logic yet)
- No mode functions (no locked doors)

```rust
fn main() {
    let device = default_device();

    // Base level: 5×5 grid (25 states)
    let n_base = 25;

    // High level: Hydration (5 levels: 0=dead, 4=full)
    let n_hydration = 5;

    // Create affordance: drinking at lake causes hydration
    let lake_location = 12;  // State (2, 2) in grid
    let mut F_hydration = Tensor::zeros([n_base, 4, 2], &device);  // 4 actions, 2 HL actions

    // At lake with drink action: α_hyd
    F_hydration[lake_location][DRINK_ACTION][ALPHA_HYDRATE] = 1.0;

    // Everywhere else: α_dehyd (get thirstier)
    for x in 0..n_base {
        for a in 0..4 {
            if x != lake_location || a != DRINK_ACTION {
                F_hydration[x][a][ALPHA_DEHYDRATE] = 1.0;
            }
        }
    }

    // HL dynamics: hydration Markov chain
    let P_hydration = create_hydration_dynamics(n_hydration);

    // Solve base-level STOK (goal = reach honey at (4,4))
    let goal_state = 24;
    let mdp_base = TaskMDP::grid_with_goal(5, 5, goal_state, vec![]);
    let base_stok = solve_task_mdp(&mdp_base)?.kernel;

    // Build factorized STOK
    let factorized = build_factorized_stok(
        base_stok,
        vec![P_hydration],
        vec![ALPHA_DEHYDRATE],  // Default action
        20,
    )?;

    // Evaluate feasibility from (x=0, hydration=4)
    let initial = ProductState::new(0).with_hl_discrete(4);
    let kappa = factorized.feasibility(&initial);

    println!("Feasibility from start (full hydration): {:.3}", kappa);

    // Sample trajectory
    let mut rng = rand::thread_rng();
    let (final_state, time) = factorized.sample(&initial, &mut rng);

    println!("Reached: base={}, hydration={}, time={}",
        final_state.base,
        final_state.hl_states[0].as_discrete(),
        time
    );
}
```

**Expected Outcome**: Working 2-space example demonstrating factorization

---

### 6.6 Deliverables for Phase 6

**Code** (~2,500 lines):
- `src/hierarchy/mod.rs` - Module organization
- `src/hierarchy/product_space.rs` - State representation (500 lines)
- `src/hierarchy/affordance.rs` - Affordance function (600 lines)
- `src/hierarchy/composition_fn.rs` - Composition λ (400 lines)
- `src/hierarchy/factorization.rs` - STOK factorization (500 lines)
- `src/hierarchy/regions.rs` - Region and default variables (300 lines)
- Tests (200 lines)

**Examples**:
- `examples/honey_badger_simplified.rs` - 2-space demo
- `examples/temperature_regulation.rs` - Region-based planning

**Tests**:
- 15 unit tests for hierarchy components
- 3 integration tests

**Documentation**:
- Update theory-to-code mapping
- High-dimensional planning guide

---

## Phase 7: High-Dimensional Integration and Advanced Features (2 weeks)

**Goal**: Complete paper parity with all figures reproducible

**Score Gain**: +4 points (100/100)
- +2 for sublimation
- +1 for full paper examples working
- +1 for empowerment basics

### 7.1 Sublimation (Theorem 2.4) (Week 1)

**Deliverable**: `src/hierarchy/sublimation.rs`

#### 7.1.1 Implementation

```rust
/// Sublimated Task MDP: abstracted to HL space only
///
/// Paper: M_sub,σ = ⟨Σ, A_σ, P_σ, f_g,σ, f_c,σ⟩
pub struct SublimatedTMDP<B: Backend> {
    /// HL transition kernel (treating HL actions as free variables)
    hl_kernel: Tensor<B, 3>,  // [n_z, n_α_z, n_z]

    /// Sublimated goal function: f_g,σ(σ) = max_{z,x,a} f_g(σ, z, x, a)
    goal_fn: Tensor<B, 1>,  // [n_z]

    /// Sublimated constraint: f_c,σ(σ, α_σ)
    constraint_fn: Tensor<B, 2>,  // [n_z, n_α_z]
}

impl<B: Backend> SublimatedTMDP<B> {
    /// Create from product-space TMDP
    pub fn from_product_tmdp(
        product_mdp: &ProductTMDP<B>,
        hl_space_index: usize,
    ) -> Self {
        // Extract HL kernel for space k
        // Maximize goal over all other dimensions
        unimplemented!("Sublimation extraction")
    }

    /// Solve for sublimated feasibility
    pub fn solve(&self) -> Result<STOKKernel<B>, StokError> {
        // Standard feasibility iteration on HL space
        let config = FeasibilityIterationConfig::default();
        feasibility_iteration(self, config)
    }
}

/// Compute sublimated feasibility for pruning
///
/// Paper Theorem 2.4: κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)
pub fn compute_sublimated_feasibility<B: Backend>(
    product_mdp: &ProductTMDP<B>,
    hl_space_index: usize,
) -> Vec<f32> {
    let sub_mdp = SublimatedTMDP::from_product_tmdp(product_mdp, hl_space_index);
    let result = sub_mdp.solve()?;

    // Return κ_sub as CPU vector for fast pruning checks
    result.kernel.kappa.into_data().to_vec().unwrap()
}
```

#### 7.1.2 Integration with Tree Search

```rust
// In tree_search.rs, add:

pub struct TreeSearchConfig {
    // ... existing fields ...

    /// Sublimated feasibility for pruning
    pub sublimated_kappa: Option<HashMap<usize, Vec<f32>>>,  // space_id -> κ_sub
}

// In tree search loop:
fn should_prune_by_sublimation(
    node: &SearchNode,
    config: &TreeSearchConfig,
) -> bool {
    if let Some(ref sub_kappas) = config.sublimated_kappa {
        for (space_id, kappa_sub) in sub_kappas {
            let hl_state = node.product_state.hl_states[*space_id].as_discrete();
            if kappa_sub[hl_state] == 0.0 {
                return true;  // Abstractly infeasible → prune
            }
        }
    }
    false
}
```

**Tests**:
- Sublimated feasibility bounds full feasibility
- Pruning reduces search nodes
- Fig. 7 (bottom) recreation

---

### 7.2 Feature Functions (Week 1)

**Deliverable**: `src/hierarchy/features.rs`

**Purpose**: Enable modularity and remapping (Fig. 6-7)

```rust
/// Feature set: maps states to semantic features
pub struct FeatureSet {
    features: HashSet<String>,  // {"lake", "tree", "honey", ...}
}

/// State-to-feature mapping: H_ψ(ψ | x, a)
pub trait StateFeatureMap<B: Backend> {
    fn features_at(&self, state: usize, action: usize) -> FeatureSet;
}

/// Feature-to-action mapping: H_α(α | ψ)
pub trait FeatureActionMap {
    fn hl_action_for(&self, features: &FeatureSet) -> Vec<(usize, f32)>;  // (α, prob)
}

/// Composed affordance: F = H_α ∘ H_ψ
pub struct FeatureBasedAffordance<B: Backend> {
    state_to_features: Box<dyn StateFeatureMap<B>>,
    features_to_action: Box<dyn FeatureActionMap>,
}

impl<B: Backend> AffordanceFunction<B> for FeatureBasedAffordance<B> {
    fn probability(&self, hl_action: &HLAction, x: usize, a: usize) -> f32 {
        let features = self.state_to_features.features_at(x, a);
        let actions = self.features_to_action.hl_action_for(&features);

        actions.iter()
            .find(|(α, _)| α == hl_action.id())
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    }
}
```

**Example**: Remapping (like Fig. 7)
```rust
// Original: Lake at (2,3) → α_hyd
let H_psi_original: HashMap<(usize,usize), FeatureSet> = ...;
H_psi_original[(2,3)] = FeatureSet::from(["lake", "water"]);

// Remapped: Lake at different location (4,1) → α_hyd
let H_psi_remapped: HashMap<(usize,usize), FeatureSet> = ...;
H_psi_remapped[(4,1)] = FeatureSet::from(["lake", "water"]);

// Same H_α: "water" → α_hyd
// Different F = H_α ∘ H_psi due to different H_psi

// Sublimated κ_sub can be reused!
```

---

### 7.3 Full Paper Examples (Week 2)

#### 7.3.1 Honey Badger (Fig. 1, 3, 9)

**File**: `examples/paper_examples/honey_badger.rs`

**Full Implementation**:
```rust
fn main() -> Result<(), Box<dyn Error>> {
    // Base level: 10×10 gridworld
    let n_x = 100;

    // High level:
    // - Σ: Binary vector (3 bits: honey, flowers, key)
    // - Y: Hydration (10 levels)
    let n_sigma = 8;   // 2³ binary states
    let n_hydration = 10;

    // Total product space: 100 × 8 × 10 = 8,000 states

    // 1. Create affordances
    let F_logic = create_item_pickup_affordance();     // Picking items sets bits
    let F_hydration = create_hydration_affordance();   // Drinking hydrates
    let F_combined = combine_affordances(vec![F_logic, F_hydration]);

    // 2. Create HL dynamics
    let P_sigma = create_logic_kernel();      // Static (identity with bit-flip actions)
    let P_hydration = create_hydration_kernel();  // Dynamic (agent gets thirsty)

    // 3. Mode function: key unlocks door
    let zeta = KeyUnlockMode::new(door_location, key_bit_index);

    // 4. Define goals
    // Sub-goals: get honey, get flowers, get key
    let F_goals = F_combined.to_goal_functions();  // Returns {f_g1, f_g2, f_g3, f_g4}

    // 5. Define constraints
    // - Fire states: f_c(x) = 0 at fire locations
    // - Dehydration: f_c(hydration=0) = 0 (death)
    // - Precedence: key ≺ (honey, flowers) enforced in f_c,σ

    // 6. Solve ensemble of base-level STOKs
    let mut goal_kernels = vec![];
    for (goal_id, f_g) in F_goals.iter().enumerate() {
        // Create TMDP for this goal
        let mdp = create_goal_specific_mdp(f_g, &constraints);
        let stok = solve_task_mdp(&mdp)?.kernel;
        goal_kernels.push((GoalId(goal_id as u32), stok));
    }

    // 7. Build factorized STOKs
    let mut factorized_goals = vec![];
    for (goal_id, base_stok) in goal_kernels {
        let fact_stok = build_factorized_stok(
            base_stok,
            vec![P_sigma.clone(), P_hydration.clone()],
            vec![ALPHA_NO_CHANGE, ALPHA_DEHYDRATE],  // Defaults
            30,
        )?;
        factorized_goals.push((goal_id, fact_stok));
    }

    // 8. Tree search over product space
    let initial = ProductState::new(0)  // Start at depot
        .with_hl_binary_vector(vec![false, false, false])  // No items
        .with_hl_discrete(9);  // Full hydration

    let plan = tree_search_factorized(
        &factorized_goals,
        &initial,
        GoalId(4),  // Deliver to friend
        TreeSearchConfig { max_depth: 8, ..Default::default() },
    )?;

    println!("Honey Badger Plan: {}", plan);
    println!("Success probability: {:.1}%", plan.feasibility * 100.0);

    // 9. Visualize (like Fig. 9)
    visualize_product_space_plan(&plan, &factorized_goals, "fig9_recreation.png")?;
}
```

**Expected Outcome**: Reproduce Fig. 1, 3, 9 behavior

#### 7.3.2 Temperature Regulation (Fig. 5)

**File**: `examples/paper_examples/temperature_regulation.rs`

```rust
fn main() {
    // Two regions: Hot (warms agent) and Cold (cools agent)
    // Agent must alternate to survive

    // Define regions
    let R_hot = region_from_states(vec![0, 1, 2, 10, 11, 12]);  // Left side
    let R_cold = region_from_states(vec![7, 8, 9, 17, 18, 19]); // Right side

    // Default variables
    let alpha_warm = 1;  // Temperature increases
    let alpha_cool = 0;  // Temperature decreases

    // HL dynamics: temperature Markov chain
    // State 0 = frozen (dead), State 10 = overheated (dead)
    let P_temp = create_temperature_dynamics(11);

    // Constraints: death states
    let mut constraints = Tensor::ones([n_base, n_actions], &device);
    // At temp=0 or temp=10: f_c = 0
    // (This is handled by HL constraint)

    // Solve with region-aware factorization
    let result = solve_multi_region_problem(
        base_mdp,
        P_temp,
        vec![R_hot, R_cold],
        vec![alpha_warm, alpha_cool],
    )?;

    println!("Found plan navigating between hot/cold regions");
    visualize_temperature_trajectory(&result, "fig5_recreation.png")?;
}
```

---

### 7.4 Empowerment Basics (Week 2)

**Deliverable**: `src/empowerment/mod.rs`

**Goal**: Implement intrinsic motivation (Section 3.C)

```rust
/// Channel capacity computation for empowerment
///
/// Paper: E_n(G | s) = max_{p(o^n)} I(O^n ; ST_n | s)
pub fn compute_empowerment<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    state: &ProductState,
    horizon: usize,
) -> f32 {
    // 1. Enumerate option sequences of length n
    let sequences = enumerate_sequences(goal_kernel.goal_ids(), horizon);

    // 2. For each sequence, compute outcome distribution
    let mut outcome_dists = vec![];
    for seq in &sequences {
        let composed = compose_goal_sequence(goal_kernel, seq)?;
        let dist = composed.termination_distribution(state);
        outcome_dists.push(dist);
    }

    // 3. Optimize input distribution p(o^n) to maximize MI
    // This is a convex optimization: I(O; ST) is concave in p(o)
    let optimal_p = optimize_input_distribution(&outcome_dists)?;

    // 4. Compute mutual information
    let empowerment = mutual_information(&optimal_p, &outcome_dists);

    empowerment
}

/// Empowerment gain from state change
///
/// Paper: V(σ=1 | σ=0; G, s, n) = E_n(G | s_σ₁) - E_n(G | s_σ₀)
pub fn empowerment_gain<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    state_before: &ProductState,
    state_after: &ProductState,
    horizon: usize,
) -> f32 {
    let E_before = compute_empowerment(goal_kernel, state_before, horizon);
    let E_after = compute_empowerment(goal_kernel, state_after, horizon);

    E_after - E_before
}
```

**Example**: Fig. 10 recreation
```rust
// Empowerment with/without key
let state_no_key = ProductState::new(door_location)
    .with_hl_binary_vector(vec![false, false, false]);  // No key

let state_with_key = ProductState::new(door_location)
    .with_hl_binary_vector(vec![false, false, true]);  // Has key

let E_gain = empowerment_gain(&kernel, &state_no_key, &state_with_key, 3);
println!("Value of key: {:.3} (empowerment gain)", E_gain);
```

**Note**: This requires solving a convex optimization. Use external solver or implement Blahut-Arimoto algorithm.

---

### 7.5 Remaining Paper Figures (Week 2)

#### Fig. 6: Option Remapping
```rust
// examples/paper_examples/fig6_remapping.rs
// Shows same options work with different feature functions
```

#### Fig. 7: Sublimation and Task Transfer
```rust
// examples/paper_examples/fig7_sublimation.rs
// Shows sublimated feasibility pruning tree search
// Shows task remapping
```

#### Fig. 8: Multi-Level Planning
```rust
// examples/paper_examples/fig8_taxi_problem.rs
// Wealth accumulation + mode switching + task completion
```

---

### 7.6 Deliverables for Phase 7

**Code** (~1,500 lines):
- `src/hierarchy/sublimation.rs` (400 lines)
- `src/hierarchy/regions.rs` (300 lines)
- `src/empowerment/mod.rs` (500 lines)
- `src/empowerment/channel_capacity.rs` (300 lines)

**Examples** (~1,000 lines):
- All paper figures reproducible
- honey_badger.rs (full version)
- temperature_regulation.rs
- taxi_problem.rs
- sublimation_demo.rs
- empowerment_demo.rs

**Tests**:
- 10 sublimation tests
- 5 empowerment tests
- End-to-end paper figure validation

---

## Dependency Graph

```
Phase 5 (Validation)
└── Can start immediately ✅

Phase 6 (Product-Space)
├── Requires: None
└── Enables: Phase 7

Phase 7 (Integration)
├── Requires: Phase 6 complete
└── Enables: Full paper parity
```

---

## Detailed Implementation Priorities

### Priority 1 (Must-Have for Parity): Product-Space Support

**Impact**: Unlocks 70% of paper examples

**Components**:
1. ProductState representation
2. ProductSpaceDims tracking
3. Affordance function F
4. Composition function λ
5. FactorizedSTOK assembly

**Complexity**: ⭐⭐⭐⭐ High
- New abstractions required
- Significant architectural changes
- Complex tensor operations

**Estimated Effort**: 3 weeks

---

### Priority 2 (High Value): Sublimation

**Impact**: Tree search efficiency + Fig. 7

**Components**:
1. SublimatedTMDP extraction
2. Sublimated feasibility computation
3. Pruning integration

**Complexity**: ⭐⭐⭐ Medium
- Builds on existing FI
- Straightforward extraction

**Estimated Effort**: 1 week

---

### Priority 3 (Nice-to-Have): Empowerment

**Impact**: Intrinsic motivation theory

**Components**:
1. Channel capacity computation
2. Mutual information maximization
3. Empowerment gain

**Complexity**: ⭐⭐⭐⭐ High
- Requires optimization solver
- Complex information theory

**Estimated Effort**: 2 weeks (basic), 4 weeks (full)

---

### Priority 4 (Polish): Visualization

**Impact**: Demos and paper figure recreation

**Components**:
1. Heatmap rendering
2. Bar plot generation
3. Tree diagram visualization

**Complexity**: ⭐⭐ Low-Medium
- Standard visualization libraries
- Independent of core theory

**Estimated Effort**: 1 week

---

## Milestones and Success Criteria

### Milestone 1: Validated Core (End of Phase 5)

**Criteria**:
- ✅ 180+ tests passing (current 161 + 20 new)
- ✅ Property-based tests validate all claims
- ✅ Performance benchmarks meet paper targets
- ✅ Fig. 2 (simplified) reproducible
- ✅ Tutorial examples working

**Deliverable**: "Core STOK implementation validated and documented"

---

### Milestone 2: High-Dimensional MVP (End of Phase 6)

**Criteria**:
- ✅ Product-space states working
- ✅ Affordance function functional
- ✅ 2-space example (simplified honey badger) working
- ✅ Factorized STOK evaluation correct
- ✅ 190+ tests passing

**Deliverable**: "High-dimensional STOK factorization operational"

---

### Milestone 3: Full Paper Parity (End of Phase 7)

**Criteria**:
- ✅ All main paper figures reproducible (Figs. 1-10)
- ✅ Sublimation integrated
- ✅ Empowerment basics working
- ✅ 200+ tests passing
- ✅ Score: 100/100

**Deliverable**: "Complete implementation of Ringstrom & Schrater (2025)"

---

## Risk Assessment and Mitigation

### Risk 1: Product-Space Complexity Explosion

**Problem**: |S| = |X| × |Z₁| × ... × |Zₙ| grows exponentially

**Paper Solution**: Factorization (Theorem 2.1) avoids this

**Our Mitigation**:
- Implement factorized representation first
- Never materialize full product-space kernel
- Use sparse tensors where possible
- Lazy evaluation of η̃

**Contingency**: Limit to 2-3 HL spaces for demos

---

### Risk 2: Numerical Stability in Long Compositions

**Problem**: Probability mass may drift after many compositions

**Paper Claim**: Normalization preserved

**Our Mitigation**:
- Test composition chains of length 5-10
- Measure cumulative error
- Add optional renormalization
- Property-based tests

**Contingency**: Document maximum safe composition depth

---

### Risk 3: Empowerment Optimization Complexity

**Problem**: max_{p(o^n)} I(O; ST) is non-trivial

**Paper**: Uses Blahut-Arimoto algorithm (implicit)

**Our Mitigation**:
- Start with small horizon (n=2-3)
- Use existing optimization library (e.g., `argmin` crate)
- Accept approximate solutions

**Contingency**: Defer to future work, document theoretical approach

---

### Risk 4: Visualization Performance

**Problem**: Rendering large tensors may be slow

**Mitigation**:
- Downsample for visualization
- Use efficient plotting libraries
- Generate static images (not interactive)

**Contingency**: Provide raw data export for external visualization

---

## Resource Requirements

### Development Environment

**Hardware**:
- GPU: NVIDIA RTX 3060+ or AMD equivalent (for WGPU)
- RAM: 16GB+ (for large product spaces)
- Storage: 5GB for build artifacts

**Software**:
- Rust 1.75+ with WGPU backend
- Python 3.9+ (for visualization scripts, optional)
- LaTeX (for paper-quality figure generation, optional)

---

### External Dependencies (New)

**Phase 5**:
```toml
[dependencies]
plotters = "0.3"              # Visualization
image = "0.24"                # Image manipulation

[dev-dependencies]
proptest = "1.4"              # Property-based testing
```

**Phase 6**:
```toml
[dependencies]
ndarray = "0.15"              # Sparse tensor operations (optional)
sprs = "0.11"                 # Sparse matrix library (optional)
```

**Phase 7**:
```toml
[dependencies]
argmin = "0.9"                # Optimization for empowerment
argmin-math = "0.3"
```

---

## Testing Strategy for Each Phase

### Phase 5 Testing

**Unit Tests**: 20 new tests
- 10 known solutions
- 5 property-based
- 5 gridworld scenarios

**Integration Tests**: 5 new tests
- Multi-goal planning
- Long composition chains
- Stochastic planning

**Benchmarks**: 3 new benches
- Compare vs. paper claims
- Measure composition overhead
- Profile GPU utilization

**Total Tests**: 180+ (current 161 + 20)

---

### Phase 6 Testing

**Unit Tests**: 15 new tests
- Product-space construction
- Flat index conversion
- Affordance evaluation
- SPK composition

**Integration Tests**: 3 new tests
- 2-space simplified honey badger
- Factorized STOK evaluation
- Product-space planning

**Total Tests**: 195+ (180 + 15)

---

### Phase 7 Testing

**Unit Tests**: 10 new tests
- Sublimation bounds
- Region handling
- Empowerment computation

**Integration Tests**: 5 new tests
- All paper figures (Figs. 1-10)
- Feature remapping
- Multi-level planning

**Total Tests**: 210+ (195 + 15)

---

## Success Metrics

### Code Metrics

| Metric | Current | Phase 5 | Phase 6 | Phase 7 |
|--------|---------|---------|---------|---------|
| Lines of code | 10,040 | 10,500 | 13,000 | 14,500 |
| Test count | 161 | 180 | 195 | 210 |
| Test coverage | ~85% | ~90% | ~92% | ~95% |
| Examples | 0 | 8 | 10 | 15 |

### Theoretical Coverage

| Category | Current | Phase 5 | Phase 6 | Phase 7 |
|----------|---------|---------|---------|---------|
| Core equations | 100% | 100% | 100% | 100% |
| Theorems | 0/4 | 0/4 | 2/4 | 4/4 |
| Figures reproducible | 0/10 | 1/10 | 3/10 | 10/10 |
| Overall score | 83/100 | 88/100 | 96/100 | 100/100 |

---

## Timeline and Effort Estimates

### Conservative Estimate (Part-Time: 10-15 hrs/week)

| Phase | Duration | Cumulative | Difficulty |
|-------|----------|------------|------------|
| Phase 5 | 2 weeks | 2 weeks | ⭐⭐ Easy |
| Phase 6 | 4 weeks | 6 weeks | ⭐⭐⭐⭐ Hard |
| Phase 7 | 3 weeks | 9 weeks | ⭐⭐⭐ Medium |

**Total**: ~2-3 months part-time

### Aggressive Estimate (Full-Time: 40 hrs/week)

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 5 | 1 week | 1 week |
| Phase 6 | 2 weeks | 3 weeks |
| Phase 7 | 1 week | 4 weeks |

**Total**: ~1 month full-time

---

## Recommended Approach

### Week-by-Week Plan (Part-Time Schedule)

**Weeks 1-2: Phase 5 - Validation**
- Week 1: Integration tests + property tests
- Week 2: Examples + benchmarks + visualization

**Weeks 3-6: Phase 6 - Product-Space**
- Week 3: ProductState + ProductSpaceDims + tests
- Week 4: AffordanceFunction trait + FactorizedAffordance
- Week 5: Composition function λ + tests
- Week 6: FactorizedSTOK assembly + simplified honey badger

**Weeks 7-9: Phase 7 - Integration**
- Week 7: Sublimation + pruning integration
- Week 8: Full paper examples (Figs. 1, 3, 5, 7-9)
- Week 9: Empowerment basics + final polish

---

## Alternative: Incremental Approach

If full product-space is too ambitious, consider:

### Option A: Focus on 2-Space Problems

- Implement only X × Z₁ (not full X × Z₁ × ... × Zₙ)
- Simpler but still demonstrates factorization
- Covers honey badger (X × Hydration) and temp regulation

**Effort**: 2 weeks instead of 4
**Coverage**: ~90/100 instead of 100/100

### Option B: Theoretical Examples Only

- Implement product-space for hand-crafted small examples
- Skip full generality
- Focus on proving correctness

**Effort**: 1 week
**Coverage**: ~88/100

---

## Key Decision Points

### Decision 1: Sparse vs. Dense Tensors?

**Question**: How to handle large product spaces?

**Options**:
1. **Dense** (current approach)
   - Simple implementation
   - Limited to ~10,000 states
   - Works for demos

2. **Sparse**
   - Scales to millions of states
   - Complex implementation
   - Required for real applications

**Recommendation**: Start dense, add sparse later if needed

---

### Decision 2: Full Generality vs. Examples?

**Question**: Implement general product-space or just examples?

**Options**:
1. **Full generality**: ProductState with arbitrary components
   - More engineering effort
   - Research-ready framework

2. **Example-specific**: Hard-code for honey badger, etc.
   - Faster to implement
   - Validates theory
   - Not reusable

**Recommendation**: **Full generality** - it's only slightly more work and much more valuable

---

### Decision 3: Empowerment Priority?

**Question**: Is empowerment critical for parity?

**Analysis**:
- Empowerment is Section 3.C (7 pages of 35)
- ~20% of paper
- Philosophically important but algorithmically independent
- Requires external optimization solver

**Options**:
1. **Implement fully**: Achieves 100/100
2. **Basic only**: Achieves 98/100 (good enough)
3. **Skip**: Achieves 96/100

**Recommendation**: **Basic implementation** in Phase 7, full version as future work

---

## Success Scenarios

### Best Case (Full-Time, 4 weeks)

- All phases complete
- All figures reproducible
- 210+ tests passing
- Score: 100/100
- **Ready for paper submission** demonstrating implementation

### Realistic Case (Part-Time, 9 weeks)

- Phases 5-7 complete
- Main figures reproducible
- 200+ tests passing
- Score: 98/100
- **Production-ready for research use**

### Minimum Viable (2-3 weeks)

- Phase 5 complete
- Phase 6 partially complete (2-space only)
- Core examples working
- Score: 92/100
- **Suitable for demonstrations and teaching**

---

## Deliverables Summary

### By End of Roadmap

**Code**:
- 14,500+ lines of implementation
- 210+ tests
- 15 examples
- Full visualization suite

**Documentation**:
- Complete user guide
- Theory-to-code mapping
- Performance tuning guide
- All paper figures annotated

**Validation**:
- Property-based testing
- Known solution tests
- Performance benchmarks vs. paper
- All main theorems validated

**Research Artifacts**:
- Reproducible paper figures
- Working high-dimensional examples
- Empowerment computation
- Ready for publication/extension

---

## Conclusion

**Current State**: Excellent foundation (83/100)

**Path to 100/100**: Clear and achievable
- Phase 5: 2 weeks (low risk)
- Phase 6: 3-4 weeks (high value, medium risk)
- Phase 7: 2 weeks (polish, low risk)

**Total Effort**: 7-8 weeks part-time or 4 weeks full-time

**Key Insight**: We have ~85% of the *code complexity* done, but only ~30% of the *paper's scope*. The remaining work is mostly **connecting existing pieces** rather than building new algorithms.

**Recommendation**:
1. **Start with Phase 5** (validation) - low risk, high confidence boost
2. **Commit to Phase 6** (product-space) - unlocks most value
3. **Evaluate Phase 7** need based on goals

The framework is **already publication-worthy for single-space STOK theory**. Full parity makes it a **complete reference implementation** of the paper.
