//! PP-505: Integration tests for FactorizedGoalKernel and PlanKernel
//! assembly from BL STOKs, HL SPKs, and product-space dynamics.
//!
//! These tests build a small CTMDP from independent components — a base
//! TaskMDP, an HL kernel, an affordance — then assemble a
//! FactorizedGoalKernel and PlanKernel and verify end-to-end correctness:
//!   - per-goal FactorizedSTOKs are consistent with their inputs,
//!   - boundary HL distribution sums to 1 (paper Eq [24] HL marginalization),
//!   - plan kernel simulation produces a valid trace with non-decreasing time
//!     and bounded cumulative feasibility,
//!   - empty meta-policy is a no-op identity over the initial state.

use burn::prelude::*;
use stok_core::backend::{default_device, DefaultBackend};
use stok_core::hierarchy::{
    assemble_factorized_stok, FactorizedAffordance, ProductSpaceDims, ProductState,
};
use stok_core::mdp::TaskMDP;
use stok_core::planning::{FactorizedGoalKernel, GoalInfo, PlanKernel};
use stok_core::solver::solve_task_mdp;
use stok_core::stok::STOKKernel;
use stok_core::types::GoalId;

// ============================================================================
// Witness: 3 base × 2 HL with two distinct goals
// ============================================================================

fn build_two_goal_witness() -> (FactorizedGoalKernel<DefaultBackend>, ProductState) {
    let device = default_device();

    // Base: 3-state simple chain (3 actions: left/right/stay).
    let base_mdp_for_g0: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 8, &device);
    let base_mdp_for_g1: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 8, &device);

    let base_stok_g0: STOKKernel<DefaultBackend> = solve_task_mdp(&base_mdp_for_g0).unwrap();
    let base_stok_g1: STOKKernel<DefaultBackend> = solve_task_mdp(&base_mdp_for_g1).unwrap();

    // HL kernel: 2 HL states, 2 HL actions. α=0 stays, α=1 advances.
    let mut hl_data = vec![0.0f32; 2 * 2 * 2];
    hl_data[0 * 2 * 2 + 0 * 2 + 0] = 1.0; // (0, α=0) → 0
    hl_data[0 * 2 * 2 + 1 * 2 + 1] = 1.0; // (0, α=1) → 1
    hl_data[1 * 2 * 2 + 0 * 2 + 1] = 1.0; // (1, α=0) → 1
    hl_data[1 * 2 * 2 + 1 * 2 + 1] = 1.0; // (1, α=1) → 1
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_data.as_slice(), &device).reshape([2, 2, 2]);

    // Affordance: shape must match base MDP. For simple_chain: 3 base states × 3 actions × 2 HL actions.
    // Default: (x, a) → α=0 (no-op). Override: (x=2, a=*) → α=1 (advance — when at the BL goal).
    let mut f_data = vec![0.0f32; 3 * 3 * 2];
    for x in 0..3 {
        for a in 0..3 {
            f_data[x * 3 * 2 + a * 2 + 0] = 1.0;
        }
    }
    for a in 0..3 {
        f_data[2 * 3 * 2 + a * 2 + 0] = 0.0;
        f_data[2 * 3 * 2 + a * 2 + 1] = 1.0;
    }
    let f_tensor: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 2]);
    let affordance = FactorizedAffordance::new(vec![f_tensor]);

    let dims = ProductSpaceDims::new(3, vec![2]);

    // Build two FactorizedSTOKs (Cor 6.1 path — both options are no-HL-event).
    let f_stok_g0 =
        assemble_factorized_stok(base_stok_g0, vec![hl_kernel.clone()], vec![0], dims.clone())
            .unwrap();
    let f_stok_g1 =
        assemble_factorized_stok(base_stok_g1, vec![hl_kernel.clone()], vec![0], dims.clone())
            .unwrap();

    // Assemble FactorizedGoalKernel.
    let mut gk = FactorizedGoalKernel::new(affordance, vec![hl_kernel], dims);
    gk.add_option(
        GoalId(0),
        f_stok_g0,
        GoalInfo {
            id: GoalId(0),
            name: "g0".to_string(),
            target_state: Some(2),
            max_time: 8,
        },
    )
    .unwrap();
    gk.add_option(
        GoalId(1),
        f_stok_g1,
        GoalInfo {
            id: GoalId(1),
            name: "g1".to_string(),
            target_state: Some(2),
            max_time: 8,
        },
    )
    .unwrap();

    let initial = ProductState::new(0).with_hl_discrete(0);
    (gk, initial)
}

#[test]
fn goal_kernel_assembled_from_components_has_both_goals() {
    let (gk, _) = build_two_goal_witness();
    assert_eq!(gk.n_goals(), 2);
    assert!(gk.has_goal(GoalId(0)));
    assert!(gk.has_goal(GoalId(1)));
    assert_eq!(gk.dims().base_size, 3);
    assert_eq!(gk.dims().n_hl_spaces(), 1);
}

#[test]
fn goal_kernel_feasibility_routes_to_factorized_stok() {
    let (gk, initial) = build_two_goal_witness();
    let kappa_g0 = gk.query_feasibility(GoalId(0), &initial);
    let kappa_g1 = gk.query_feasibility(GoalId(1), &initial);
    // Both options should be feasible from (z=0, x=0) since simple_chain reaches goal.
    assert!(kappa_g0 > 0.0);
    assert!(kappa_g1 > 0.0);
    // Unknown goal returns 0.
    assert_eq!(gk.query_feasibility(GoalId(99), &initial), 0.0);
}

#[test]
fn goal_kernel_boundary_hl_distribution_sums_to_one() {
    let (gk, _) = build_two_goal_witness();
    // For each HL state and each boundary HL action, the marginalization
    // Σ_{z'} G_HL(z' | z, α, t_f) must produce a valid distribution.
    for z_init in 0..2 {
        for alpha in 0..2 {
            for t_f in 1..6 {
                let dists = gk
                    .boundary_hl_distribution(GoalId(0), &[z_init], &[alpha], t_f)
                    .unwrap();
                let probs: Vec<f32> = dists[0].clone().into_data().to_vec().unwrap();
                let sum: f32 = probs.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "boundary HL dist (z={}, α={}, t_f={}) must sum to 1, got {}",
                    z_init,
                    alpha,
                    t_f,
                    sum
                );
            }
        }
    }
}

#[test]
fn plan_kernel_two_step_simulation_advances_correctly() {
    let (gk, initial) = build_two_goal_witness();
    let plan_kernel = PlanKernel::new(&gk, vec![GoalId(0), GoalId(1)]);
    let trace = plan_kernel.simulate_deterministic(&initial);

    assert_eq!(trace.steps.len(), 2);
    // Time strictly increases per step (boundary +1 included).
    let t0 = trace.steps[0].time;
    let t1 = trace.steps[1].time;
    assert!(
        t1 > t0,
        "second step time {} must exceed first step time {}",
        t1,
        t0
    );
    assert!(trace.total_time == t1);
    // Cumulative feasibility is monotone non-increasing.
    assert!(
        trace.steps[1].cumulative_feasibility <= trace.steps[0].cumulative_feasibility + 1e-6,
        "cumulative feasibility should not increase across options"
    );
    // Final state base is in valid range.
    assert!(trace.final_state.base < 3);
}

#[test]
fn plan_kernel_stochastic_sampling_is_deterministic_under_seed() {
    use rand::SeedableRng;
    let (gk, initial) = build_two_goal_witness();
    let plan_kernel = PlanKernel::new(&gk, vec![GoalId(0)]);

    // Two runs with the same seed should produce the same trace.
    let mut rng_a = rand::rngs::StdRng::seed_from_u64(42);
    let mut rng_b = rand::rngs::StdRng::seed_from_u64(42);
    let trace_a = plan_kernel.sample(&initial, &mut rng_a);
    let trace_b = plan_kernel.sample(&initial, &mut rng_b);

    assert_eq!(trace_a.total_time, trace_b.total_time);
    assert_eq!(trace_a.final_state.base, trace_b.final_state.base);
    assert_eq!(
        trace_a.final_state.hl_states.len(),
        trace_b.final_state.hl_states.len()
    );
}

#[test]
fn plan_kernel_empty_sequence_is_identity() {
    let (gk, initial) = build_two_goal_witness();
    let plan_kernel: PlanKernel<DefaultBackend> = PlanKernel::new(&gk, vec![]);
    let trace = plan_kernel.simulate_deterministic(&initial);
    assert!(trace.steps.is_empty());
    assert_eq!(trace.total_time, 0);
    assert_eq!(trace.final_state.base, initial.base);
    assert_eq!(trace.final_cumulative_feasibility, 1.0);
}
