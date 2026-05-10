//! PP-105: Theorem-style invariant tests for STOK normalization, κ-η consistency,
//! and absorbing-policy enforcement.
//!
//! These tests cross module boundaries (TaskMDP → feasibility_iteration → STOKKernel)
//! and exercise the paper's invariants on a battery of deterministic and stochastic
//! witness problems. Each test names the equation it validates.
//!
//! References (Ringstrom & Schrater 2025):
//! - Eq [15]: κ(x) = Σ_{x_f, t_f} η+(x_f, t_f | x)
//! - Eq [16]: 1 - κ(x) = Σ_{x_f, t_f} η-(x_f, t_f | x)
//! - Eq [29]: Σ_{x_f, t_f} η**(x_f, t_f | x) = 1
//! - Appendix 5: STOK normalization requires absorbing/transient nonterminal dynamics
//!   under the policy chain.

use burn::prelude::*;
use stok_core::backend::{default_device, DefaultBackend};
use stok_core::mdp::TaskMDP;
use stok_core::solver::{feasibility_iteration, FeasibilityIterationConfig};
use stok_core::stok::STOKKernel;

// ============================================================================
// Helpers
// ============================================================================

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Assert all three identities on the given STOKKernel.
fn assert_stok_invariants(kernel: &STOKKernel<DefaultBackend>, tol: f32) {
    let s = kernel.n_states();

    let kappa: Vec<f32> = kernel.kappa.clone().into_data().to_vec().unwrap();

    // Σ over (x_f, t_f) of η+ → length-S vector.
    let kappa_from_eta_plus = kernel
        .eta_plus
        .clone()
        .sum_dim(2)
        .squeeze::<2>()
        .sum_dim(1)
        .squeeze::<1>();
    let kappa_eta_plus_data: Vec<f32> = kappa_from_eta_plus.into_data().to_vec().unwrap();

    // Σ over (x_f, t_f) of η-.
    let kappa_from_eta_minus = kernel
        .eta_minus
        .clone()
        .sum_dim(2)
        .squeeze::<2>()
        .sum_dim(1)
        .squeeze::<1>();
    let kappa_eta_minus_data: Vec<f32> = kappa_from_eta_minus.into_data().to_vec().unwrap();

    for i in 0..s {
        // Eq [15]: κ = Σ η+
        assert!(
            approx(kappa[i], kappa_eta_plus_data[i], tol),
            "Eq [15] κ = Σ η+ violated at state {}: κ={}, Σ η+ = {}",
            i,
            kappa[i],
            kappa_eta_plus_data[i]
        );

        // Eq [16]: 1 - κ = Σ η-
        assert!(
            approx(1.0 - kappa[i], kappa_eta_minus_data[i], tol),
            "Eq [16] 1-κ = Σ η- violated at state {}: 1-κ={}, Σ η- = {}",
            i,
            1.0 - kappa[i],
            kappa_eta_minus_data[i]
        );

        // Eq [29]: Σ η** = 1 (combined)
        let total = kappa_eta_plus_data[i] + kappa_eta_minus_data[i];
        assert!(
            approx(total, 1.0, tol),
            "Eq [29] Σ η** = 1 violated at state {}: total = {}",
            i,
            total
        );
    }
}

// ============================================================================
// PP-105 (a)/(b)/(c): Invariants across multiple witness problems
// ============================================================================

#[test]
fn invariants_deterministic_chain() {
    let device = default_device();
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(8, 20, &device);
    let result =
        feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
    assert_stok_invariants(&result.kernel, 1e-4);
}

#[test]
fn invariants_constrained_chain() {
    let device = default_device();
    // Fire at state 4 splits the chain — states 0..=4 infeasible, 5..=9 feasible.
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(10, 4, 20, &device);
    let result =
        feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
    assert_stok_invariants(&result.kernel, 1e-4);
}

#[test]
fn invariants_stochastic_chain_high_success() {
    let device = default_device();
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(6, 0.9, 25, &device);
    let result =
        feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
    assert_stok_invariants(&result.kernel, 1e-3);
}

#[test]
fn invariants_stochastic_chain_low_success() {
    let device = default_device();
    // 60% success per step — slower convergence + heavier dispersion in η. The
    // η+ tail is geometrically long, so we need a generous time horizon
    // (max_time = 80) on BOTH the MDP and the feasibility-iteration config to
    // capture ≥99.9% of the mass before truncation.
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(5, 0.6, 80, &device);
    let config = FeasibilityIterationConfig::default().with_max_time(80);
    let result = feasibility_iteration(&mdp, config).unwrap();
    assert_stok_invariants(&result.kernel, 1e-3);
}

// ============================================================================
// PP-105 (d): Absorbing-policy enforcement (Appendix 5 prerequisite)
// ============================================================================

/// Construct a 2-state TaskMDP whose only feasible policy creates a closed
/// recurrent class (a 2-cycle) where every state has f_2 = 1 — i.e. the
/// nonterminal block of the policy chain is recurrent, not transient. Per
/// Appendix 5, STOK normalization (Eq [29]) is undefined here, and the
/// implementation's `check_policy_absorption_release` must return Err.
fn make_non_absorbing_mdp(
    device: &<DefaultBackend as Backend>::Device,
) -> TaskMDP<DefaultBackend> {
    let n = 2;
    let a = 1;
    let mut p = vec![0.0f32; n * a * n];
    // 0 → 1, 1 → 0 (a 2-cycle).
    p[0 * a * n + 0 * n + 1] = 1.0;
    p[1 * a * n + 0 * n + 0] = 1.0;
    let trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(p.as_slice(), device).reshape([n, a, n]);

    // No goal anywhere, no constraint violations — f_1 = 0, f_2 = 1 everywhere.
    let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([n, a], device);
    let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n, a], device);

    TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).expect("non-absorbing test MDP")
}

#[test]
fn invariants_absorbing_policy_enforced_negative_case() {
    let device = default_device();
    let mdp = make_non_absorbing_mdp(&device);

    let result = feasibility_iteration(&mdp, FeasibilityIterationConfig::default());

    // The implementation MUST refuse to construct a STOK on this problem,
    // because Eq [29] (Σ η** = 1) cannot hold when the policy chain has no
    // path to termination.
    assert!(
        result.is_err(),
        "Non-absorbing policy chain must be rejected, but feasibility_iteration succeeded"
    );
}

#[test]
fn invariants_absorbing_policy_enforced_positive_case() {
    // The complement: a chain MDP whose policy IS absorbing (terminates at
    // the goal). feasibility_iteration must succeed and the invariants must
    // hold — this guards against an over-eager Err from the absorption check.
    let device = default_device();
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 15, &device);

    let result = feasibility_iteration(&mdp, FeasibilityIterationConfig::default())
        .expect("absorbing-policy positive case must succeed");
    assert_stok_invariants(&result.kernel, 1e-4);
}
