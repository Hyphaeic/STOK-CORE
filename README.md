# The STOK-KERNEL Framework: From Task Definition to Composite Planning

This repository implements the "State-Time Option Kernel" pipeline defined by Ringstrom & Schrater (2025). Instead of documenting APIs, this README walks the lifecycle of a data object—from raw dynamics to composite plans—using the Honey Badger story from the paper as the running example.

## 1. System Architecture: The Data Object Hierarchy

### Level 1 · The World (`TransitionTensor`)
A world is a tensor `P[x, a, x’]` that encodes the physics \(P_x\). Builders such as `TaskMDP::simple_chain` or the grid constructors in `examples/honey_badger_2space.rs` create this tensor with the correct row-stochastic structure (see `TaskMDP::validate_stochastic`). At this level you only model uncontrolled motion: the Honey Badger grid defines moves `UP/DOWN/LEFT/RIGHT`, wrap-around walls, and the lake tile where hydration actions become possible later.

### Level 2 · The Task (`TaskMDP`)
`TaskMDP` wraps the world with two functions: the goal selector \(f_g\) and the constraint selector \(f_c\). The struct ( `src/mdp/task_mdp.rs`) stores `transition`, `goal_fn`, `constraint_fn`, and automatically derives the achievement/continuation surfaces \(f_1 = f_g · f_c\) and \(f_2 = (1-f_g) · f_c\). For Honey Badger:
- Goal: reaching the friend state when hydration is nonzero.
- Constraint: hydration must stay above 0, so `f_c` zeroes out actions that would dehydrate a dry agent.
The result is a **Task object** that supplies every term needed by the κ-OKBE fixed point.

### Level 3 · The Option (`STOKKernel` aka OKBE solution)
Solving a task with `solve_task_mdp` runs feasibility iteration over \(f_1, f_2, P\) and outputs an `STOKKernel`. This object contains:
- `policy` (the π** argmax of Equation (8))
- `kappa` (the cumulative feasibility values)
- `eta_plus` / `eta_minus` (success/failure termination densities over \(x_f, t_f\))
In practice this gives the Honey Badger a predictive map: for every starting tile, you know the entire time-resolved distribution of “reach friend hydrated” vs. “die of thirst”. No scalar rewards are used; everything comes from \(f_1\) and \(f_2\).

### Level 4 · The Plan (`GoalKernel` and composed STOKs)
Plans emerge when options are chained. The `GoalKernel` ( `src/planning/goal_kernel.rs`) stores many STOKs keyed by `GoalId`, caches their time-marginal SOKs, and acts as a planner that can answer “which goal option is feasible next?” Sequential composition uses the Chapman-Kolmogorov routines (`src/composition/chapman_kolmogorov.rs`) to convolve η-tensors, so a plan such as `Option_Water → Option_Honey → Option_Friend` is implemented via η-composition rather than ad-hoc scripting. The result is a **GoalKernel** object—a palette of composable plans that stay probabilistically normalized.

## 2. The “Honey Badger” Workflow (Tutorial)

### Step 1 · Define the Affordance Function (F)
Affordances link base actions to high-level effects. In `examples/honey_badger_2space.rs`, `FactorizedAffordance` builds a tensor `[x, a, α_z]` where:
- For any action at the lake tile, `F(α_hydrate | lake, a) = 1.0` (drink → hydration up).
- Elsewhere, `F(α_dehydrate | x, a) = 1.0` (every move costs water).
This implements Definition 0.1: high-level action vectors are sampled from `F` so that “drinking at the lake” deterministically maps to the high-level state “hydration full”. When you instantiate a `ProductState`, you can now evolve the hydration component in sync with the grid motion.

### Step 2 · Sublimation (Feasibility gating)
Before spending GPU time on the full κ-OKBE, run a sublimated check on the logic space only:
```rust
use stok_core::hierarchy::sublimation::SublimatedTMDP;
let sublimated = SublimatedTMDP::from_product(&base_mdp, hl_kernel, &affordance, hydration_space_id)?;
let kappa_sigma = sublimated.solve()?;
if kappa_sigma[sigma_badger_state] == 0.0 {
    println!("Logic task infeasible — prune this branch before pathfinding.");
}
```
This is Theorem 2.4 in action: `kappa_sigma` upper-bounds the full κ̃ on the product space. In the Honey Badger story, you check “Can the logical task ‘arrive with honey before dehydrating’ ever succeed?” before launching heavy planning.

### Step 3 · Solver Execution (Predictive OKBE map)
Once the sublimated task passes, you construct the base Task MDP, run `solve_task_mdp`, and assemble the factorized STOK:
```rust
let base_stok = solve_task_mdp(&grid_task)?;            // κ*, π**, η⁺, η⁻
let factorized = assemble_factorized_stok(
    base_stok,
    vec![hydration_dynamics],
    vec![ALPHA_DEHYDRATE],
    ProductSpaceDims::new(n_grid, vec![n_hydration]),
)?;
```
`factorized.evaluate(initial, final, t)` now returns \(η̃(z_f, x_f, t_f | z, x)\): a predictive map detailing *where* and *when* the Honey Badger succeeds or fails. Inspecting `factorized.base().eta_plus` gives the full temporal profile of successful arrivals; `factorized.hl_spk` captures how hydration drifts during execution.

## 3. Verification & Interpretability

Reading a STOK answers operational questions:
- **Probability of dying of dehydration before goal?** Sum the `eta_minus` slab over `x_f` that correspond to the “hydration = 0” slice and over all times. In code:
  ```rust
  let eta_minus = badger_stok.eta_minus.clone();
  let dehydration_prob = eta_minus
      .slice([start_state..start_state+1, dehydration_state..dehydration_state+1, 0..T])
      .sum().into_scalar().elem();
  ```
- **When is risk highest?** Use `prediction::CumulativeEventFunction::from_stok` and `TemporalEventFunction::from_cef` to get ξ(t | s). Plotting ξ for the hydration-failure branch shows the exact timestep where dehydration dominates.
- **Policy inspection.** Because κ, π, η⁺, and η⁻ sit in one object, calling `stok.validate()` enforces normalization (Ση** = 1) and κ-η consistency so that every interpretability result is probability-sound.

Following this pipeline keeps the implementation faithful to the Option Kernel theory: compose worlds → tasks → options → plans, with mathematical guarantees surviving each transition from Honey Badger narratives to GPU kernels.
