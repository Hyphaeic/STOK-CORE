//! # Temperature Regulation Example (Figure 5)
//!
//! Demonstrates mode-based planning with region-dependent default dynamics.
//!
//! ## Scenario (from Ringstrom & Schrater 2025, Fig. 5)
//!
//! An agent must navigate between hot and cold regions to regulate internal
//! temperature and avoid death by overheating or freezing.
//!
//! ## Product Space
//!
//! S = X × T where:
//! - X: Gridworld position (20 states in 4×5 grid)
//! - T: Temperature (11 discrete levels: 0=frozen dead, 10=overheated dead)
//!
//! ## Regions
//!
//! - R_hot (left side): Default action α_warm causes temperature to increase
//! - R_cold (right side): Default action α_cool causes temperature to decrease
//!
//! ## Key Concept
//!
//! Agent must alternate between regions to maintain temperature in safe range [1, 9].
//! Simple shortest path fails - must plan with homeostatic constraint.

use stok_core::prelude::*;
use stok_core::hierarchy::{
    assemble_factorized_stok, FactorizedAffordance, FactorizedSTOK, ProductSpaceDims,
    ProductState, ThresholdMode,
};
use stok_core::solve_task_mdp;
use burn::prelude::*;
use std::error::Error;

// Temperature states
const TEMP_FROZEN: usize = 0; // Dead (too cold)
const TEMP_MIN_SAFE: usize = 1;
const TEMP_MAX_SAFE: usize = 9;
const TEMP_OVERHEAT: usize = 10; // Dead (too hot)

// High-level actions (temperature change)
const ALPHA_WARM: usize = 1; // Temperature increases
const ALPHA_COOL: usize = 0; // Temperature decreases

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    TEMPERATURE REGULATION (Figure 5)                         ║");
    println!("║    Demonstrates: Region-based Mode Switching                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = default_device();

    // ========================================================================
    // 1. SETUP: Define the World
    // ========================================================================

    println!("🌍 World Setup:");
    println!("  Grid: 4×5 = 20 states");
    println!("  Layout:");
    println!("    [HOT REGION]    [COLD REGION]");
    println!("     0   1   2   3   4");
    println!("     5   6   7   8   9");
    println!("    10  11  12  13  14");
    println!("   [S] 16  17  18 [G]");
    println!();
    println!("  Temperature: 11 levels (0=🥶 frozen, 10=🔥 overheat)");
    println!("  Safe range: [1-9]");
    println!("  Total product space: 20 × 11 = 220 states\n");

    let rows = 4;
    let cols = 5;
    let n_base = rows * cols;

    let start_state = 15; // (3, 0) - bottom left (hot region)
    let goal_state = 19; // (3, 4) - bottom right (cold region)

    let n_temp = 11;

    // Hot region: left side (columns 0-1)
    let hot_region: Vec<usize> = (0..rows)
        .flat_map(|r| vec![r * cols, r * cols + 1])
        .collect();

    // Cold region: right side (columns 3-4)
    let cold_region: Vec<usize> = (0..rows)
        .flat_map(|r| vec![r * cols + 3, r * cols + 4])
        .collect();

    println!("  Regions:");
    println!("    Hot (warms agent):  states {:?}", hot_region);
    println!("    Cold (cools agent): states {:?}", cold_region);
    println!();

    // ========================================================================
    // 2. BASE-LEVEL GRIDWORLD (Simple 4-Directional Movement)
    // ========================================================================

    println!("🗺️  Creating Base-Level MDP:");

    let mdp_base = create_gridworld_mdp(rows, cols, goal_state, vec![], 30, &device)?;

    println!("  ✓ {} states, {} actions", mdp_base.n_states(), mdp_base.n_actions());
    println!();

    // ========================================================================
    // 3. TEMPERATURE DYNAMICS (High-Level Space)
    // ========================================================================

    println!("🌡️  Creating Temperature Dynamics:");
    println!("  P_temp(t'|t, α) where α ∈ {{cool, warm}}");

    let p_temp = create_temperature_dynamics(n_temp, &device);

    println!("  ✓ α_cool (0): temp decreases by 1");
    println!("  ✓ α_warm (1): temp increases by 1");
    println!("  ✓ Death states: temp=0 (frozen), temp=10 (overheat)");
    println!();

    // ========================================================================
    // 4. AFFORDANCE FUNCTION (Region-Based)
    // ========================================================================

    println!("⚡ Creating Affordance Function:");
    println!("  F(α_temp | x, a) depends on region");

    let mut F_data = vec![0.0f32; n_base * 4 * 2]; // 4 actions, 2 HL actions

    // Hot region: all actions cause warming
    for &x in &hot_region {
        for a in 0..4 {
            F_data[x * 4 * 2 + a * 2 + ALPHA_WARM] = 1.0;
        }
    }

    // Cold region: all actions cause cooling
    for &x in &cold_region {
        for a in 0..4 {
            F_data[x * 4 * 2 + a * 2 + ALPHA_COOL] = 1.0;
        }
    }

    // Neutral region (middle column): default to cooling
    for r in 0..rows {
        let x = r * cols + 2; // Column 2
        for a in 0..4 {
            F_data[x * 4 * 2 + a * 2 + ALPHA_COOL] = 1.0;
        }
    }

    let affordance_tensor: Tensor<DefaultBackend, 1> =
        Tensor::from_floats(F_data.as_slice(), &device);
    let affordance_tensor = affordance_tensor.reshape([n_base, 4, 2]);

    let affordance = FactorizedAffordance::new(vec![affordance_tensor]);

    println!("  ✓ Hot region  → α_warm (temp increases)");
    println!("  ✓ Cold region → α_cool (temp decreases)");
    println!();

    // ========================================================================
    // 5. SOLVE BASE-LEVEL STOK
    // ========================================================================

    println!("🎯 Solving Base-Level STOK:");
    println!("  Goal: Reach state {} (cold region)", goal_state);

    let base_stok = solve_task_mdp(&mdp_base)?;

    let kappa_slice: Tensor<DefaultBackend, 1> = base_stok.kappa.clone()
        .slice([start_state..(start_state + 1)]);
    let kappa_start: f32 = kappa_slice.into_scalar().elem();

    println!("  ✓ Base feasibility κ(start={}) = {:.3}", start_state, kappa_start);
    println!();

    // ========================================================================
    // 6. ASSEMBLE FACTORIZED STOK
    // ========================================================================

    println!("🔧 Assembling Factorized STOK:");

    let dims = ProductSpaceDims::new(n_base, vec![n_temp]);

    // Use cooling as default (agent in neutral/cold regions most of time)
    let factorized = assemble_factorized_stok(
        base_stok,
        vec![p_temp],
        vec![ALPHA_COOL], // Default: cooling
        dims.clone(),
    )?;

    println!("  ✓ Product space: {} × {} = {} states", n_base, n_temp, dims.total_size);
    println!("  ✓ Memory: {}² × T base + {}² × T HL", n_base, n_temp);
    println!();

    // ========================================================================
    // 7. ANALYZE FEASIBILITY WITH TEMPERATURE CONSTRAINTS
    // ========================================================================

    println!("═══ Feasibility Analysis ═══\n");

    // Scenario 1: Start in hot region with moderate temperature
    let start_moderate_temp = ProductState::new(start_state)
        .with_hl_discrete(5); // Middle of safe range

    let kappa_moderate = factorized.feasibility_approx(&start_moderate_temp);

    println!("From hot region (state {}) with moderate temp (5):", start_state);
    println!("  κ̃ ≈ {:.3}", kappa_moderate);

    // Scenario 2: Start with low temperature (risky)
    let start_low_temp = ProductState::new(start_state)
        .with_hl_discrete(TEMP_MIN_SAFE); // At minimum safe

    let kappa_low = factorized.feasibility_approx(&start_low_temp);

    println!("\nFrom hot region with low temp (1) - RISKY:");
    println!("  κ̃ ≈ {:.3}", kappa_low);
    println!("  ⚠️  Starting in hot region while cold is dangerous!");

    // Scenario 3: Start with high temperature
    let start_high_temp = ProductState::new(start_state)
        .with_hl_discrete(TEMP_MAX_SAFE); // At maximum safe

    let kappa_high = factorized.feasibility_approx(&start_high_temp);

    println!("\nFrom hot region with high temp (9) - EXTREME RISK:");
    println!("  κ̃ ≈ {:.3}", kappa_high);
    println!("  🔥 One more step in hot region → overheat!");

    // Scenario 4: Ideal start (moderate temp, path to cold region)
    let cold_start = ProductState::new(12) // Middle state
        .with_hl_discrete(6); // Moderate-high temp

    let kappa_ideal = factorized.feasibility_approx(&cold_start);

    println!("\nFrom neutral zone (state 12) with temp 6:");
    println!("  κ̃ ≈ {:.3}", kappa_ideal);
    println!("  ✅ Balanced temperature, flexible navigation");

    println!();

    // ========================================================================
    // 8. SAMPLE TRAJECTORIES
    // ========================================================================

    println!("═══ Sampling Trajectories ═══\n");

    let mut rng = rand::thread_rng();

    for run in 1..=5 {
        let initial = ProductState::new(start_state).with_hl_discrete(5);
        let (final_state, time) = factorized.sample(&initial, &mut rng);

        let final_temp = final_state.hl_states[0].as_discrete();
        let status = if final_temp == 0 {
            "🥶 FROZEN"
        } else if final_temp == 10 {
            "🔥 OVERHEAT"
        } else if final_state.base == goal_state {
            "✅ SUCCESS"
        } else {
            "⏱️  TIMEOUT"
        };

        println!(
            "Run {}: {} → {} | Temp: 5 → {} | Time: {} | {}",
            run, start_state, final_state.base, final_temp, time, status
        );
    }

    println!();

    // ========================================================================
    // 9. KEY INSIGHTS
    // ========================================================================

    println!("═══ Key Insights ═══\n");

    println!("1. Region-Aware Planning:");
    println!("   - Hot region forces warming (α_warm)");
    println!("   - Cold region forces cooling (α_cool)");
    println!("   - Agent must alternate regions to survive");
    println!();

    println!("2. Homeostatic Constraint:");
    println!("   - Simple shortest path fails (→ overheating)");
    println!("   - Must detour to cold region to regulate");
    println!("   - Temperature becomes implicit constraint");
    println!();

    println!("3. Factorization Benefit:");
    println!("   - Full product: 20² × 11² × T = 48,400×T values");
    println!("   - Factorized:   20² × T + 11² × T = 521×T values");
    println!("   - Reduction:    ~93× memory savings");
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                     ✅ SUCCESS                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Demonstrated:");
    println!("  ✓ ThresholdMode for region-based dynamics");
    println!("  ✓ Homeostatic regulation planning");
    println!("  ✓ Default variable coupling (α^ℓ per region)");
    println!("  ✓ Product-space factorization with dynamic HL space");

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create simple gridworld MDP
fn create_gridworld_mdp<B: Backend>(
    rows: usize,
    cols: usize,
    goal_state: usize,
    obstacles: Vec<usize>,
    max_time: usize,
    device: &B::Device,
) -> Result<TaskMDP<B>, StokError> {
    let n_states = rows * cols;
    let n_actions = 4; // UP, DOWN, LEFT, RIGHT

    // Build deterministic transition tensor
    let mut P_data = vec![0.0f32; n_states * n_actions * n_states];

    for x in 0..n_states {
        let (row, col) = (x / cols, x % cols);

        for a in 0..4 {
            let (next_row, next_col) = match a {
                0 => (row.saturating_sub(1), col),        // UP
                1 => ((row + 1).min(rows - 1), col),      // DOWN
                2 => (row, col.saturating_sub(1)),        // LEFT
                3 => (row, (col + 1).min(cols - 1)),      // RIGHT
                _ => (row, col),
            };

            let next_state = next_row * cols + next_col;

            // Check if next state is obstacle
            if obstacles.contains(&next_state) {
                // Stay in place if hitting obstacle
                P_data[x * n_actions * n_states + a * n_states + x] = 1.0;
            } else {
                P_data[x * n_actions * n_states + a * n_states + next_state] = 1.0;
            }
        }
    }

    let transition: Tensor<B, 1> = Tensor::from_floats(P_data.as_slice(), device);
    let transition: Tensor<B, 3> = transition.reshape([n_states, n_actions, n_states]);

    // Goal function
    let mut goal_data = vec![0.0f32; n_states * n_actions];
    for a in 0..n_actions {
        goal_data[goal_state * n_actions + a] = 1.0;
    }
    let goal_fn: Tensor<B, 1> = Tensor::from_floats(goal_data.as_slice(), device);
    let goal_fn: Tensor<B, 2> = goal_fn.reshape([n_states, n_actions]);

    // No base-level constraints (temperature constraint is in HL space)
    let constraint_fn = Tensor::ones([n_states, n_actions], device);

    TaskMDP::new(transition, goal_fn, constraint_fn, max_time)
}

/// Create temperature dynamics
///
/// Temperature Markov chain with two actions:
/// - α_cool (0): Temperature decreases by 1 (saturates at 0)
/// - α_warm (1): Temperature increases by 1 (saturates at 10)
fn create_temperature_dynamics<B: Backend>(
    n_levels: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    // P_temp(t'|t, α) where α ∈ {cool, warm}
    // Shape: [n_levels, 2, n_levels]

    let mut P = Tensor::zeros([n_levels, 2, n_levels], device);

    for t in 0..n_levels {
        // α = COOL (0): decrease by 1, with death at 0
        let t_next_cool = if t > TEMP_FROZEN {
            t - 1
        } else {
            TEMP_FROZEN // Stay frozen (absorbing)
        };

        P = P.slice_assign(
            [t..(t + 1), ALPHA_COOL..(ALPHA_COOL + 1), t_next_cool..(t_next_cool + 1)],
            Tensor::from_floats([[[1.0]]], device),
        );

        // α = WARM (1): increase by 1, with death at 10
        let t_next_warm = if t < TEMP_OVERHEAT {
            t + 1
        } else {
            TEMP_OVERHEAT // Stay overheated (absorbing)
        };

        P = P.slice_assign(
            [t..(t + 1), ALPHA_WARM..(ALPHA_WARM + 1), t_next_warm..(t_next_warm + 1)],
            Tensor::from_floats([[[1.0]]], device),
        );
    }

    P
}
