//! PP-203: Theorem-oriented tests for STOK composition (Eqs [18], [19]).
//!
//! These tests pin the invariants the paper's Chapman-Kolmogorov equations
//! must satisfy: normalization, decomposition, associativity, κ-η consistency,
//! and equivalence of `compose_sequence([a,b,c])` with iterated pairwise
//! `compose_stoks(compose_stoks(a, b), c)`.

use burn::prelude::*;
use stok_core::backend::{default_device, DefaultBackend};
use stok_core::composition::{
    compose_sequence, compose_soks, compose_stoks, OptionSequence, StateOptionKernel,
};
use stok_core::mdp::TaskMDP;
use stok_core::solver::{feasibility_iteration, FeasibilityIterationConfig};
use stok_core::stok::STOKKernel;

// ============================================================================
// Helpers
// ============================================================================

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Solve a simple-chain MDP and return its STOK.
fn deterministic_stok(n: usize, max_time: usize) -> STOKKernel<DefaultBackend> {
    let device = default_device();
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(n, max_time, &device);
    let config = FeasibilityIterationConfig::default().with_max_time(max_time);
    let result = feasibility_iteration(&mdp, config).unwrap();
    result.kernel
}

/// Solve a stochastic-chain MDP and return its STOK.
fn stochastic_stok(n: usize, p_success: f32, max_time: usize) -> STOKKernel<DefaultBackend> {
    let device = default_device();
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(n, p_success, max_time, &device);
    let config = FeasibilityIterationConfig::default().with_max_time(max_time);
    let result = feasibility_iteration(&mdp, config).unwrap();
    result.kernel
}

/// Sum a 3D tensor along axes 1 and 2 to give a [S] vector.
fn sum_to_state_vec(t: &Tensor<DefaultBackend, 3>) -> Vec<f32> {
    t.clone()
        .sum_dim(2)
        .squeeze::<2>()
        .sum_dim(1)
        .squeeze::<1>()
        .into_data()
        .to_vec()
        .unwrap()
}

// ============================================================================
// Eq [18] — Normalization preservation under composition
// ============================================================================

#[test]
fn eq18_pairwise_compose_preserves_normalization() {
    let stok = deterministic_stok(5, 10);
    let composed = compose_stoks(&stok, &stok).unwrap();

    let total = sum_to_state_vec(&composed.eta);
    for (i, &m) in total.iter().enumerate() {
        assert!(
            approx(m, 1.0, 1e-4),
            "Σ η_μ(.|x={}) = {}, expected 1.0",
            i,
            m
        );
    }
}

#[test]
fn eq18_pairwise_compose_decomposition_identity() {
    // η_μ = η+_μ + η-_μ (Eq [17] preserved through composition).
    let stok = deterministic_stok(5, 10);
    let composed = compose_stoks(&stok, &stok).unwrap();

    let sum = composed.eta_plus.clone() + composed.eta_minus.clone();
    let diff: f32 = (composed.eta.clone() - sum)
        .abs()
        .max()
        .into_scalar()
        .elem();
    assert!(
        diff < 1e-5,
        "η_μ ≠ η+_μ + η-_μ: max diff = {}",
        diff
    );
}

#[test]
fn eq18_pairwise_compose_kappa_consistency() {
    // κ_μ = Σ η+_μ (Eq [15] preserved through composition).
    let stok = deterministic_stok(5, 10);
    let composed = compose_stoks(&stok, &stok).unwrap();

    let kappa: Vec<f32> = composed.kappa.clone().into_data().to_vec().unwrap();
    let kappa_from_eta = sum_to_state_vec(&composed.eta_plus);

    for i in 0..kappa.len() {
        assert!(
            approx(kappa[i], kappa_from_eta[i], 1e-5),
            "κ_μ(x={}) = {}, Σ η+_μ = {}",
            i,
            kappa[i],
            kappa_from_eta[i]
        );
    }
}

#[test]
fn eq18_stochastic_compose_preserves_normalization() {
    // Same invariant, this time on a stochastic kernel — needs more time
    // horizon to capture the full convolution tail.
    let stok = stochastic_stok(4, 0.85, 25);
    let composed = compose_stoks(&stok, &stok).unwrap();

    let total = sum_to_state_vec(&composed.eta);
    for (i, &m) in total.iter().enumerate() {
        assert!(
            approx(m, 1.0, 5e-3),
            "stochastic Σ η_μ(.|x={}) = {}, expected ≈ 1.0",
            i,
            m
        );
    }
}

// ============================================================================
// Associativity: compose(a, compose(b, c)) ≈ compose(compose(a, b), c)
// ============================================================================

#[test]
fn eq18_compose_is_associative_deterministic() {
    let a = deterministic_stok(4, 6);
    let b = deterministic_stok(4, 6);
    let c = deterministic_stok(4, 6);

    // Left-associated: ((a · b) · c)
    let ab = compose_stoks(&a, &b).unwrap();
    let ab_kernel = ab.to_stok_kernel().unwrap();
    let left = compose_stoks(&ab_kernel, &c).unwrap();

    // Right-associated: (a · (b · c))
    let bc = compose_stoks(&b, &c).unwrap();
    let bc_kernel = bc.to_stok_kernel().unwrap();
    let right = compose_stoks(&a, &bc_kernel).unwrap();

    // Same time horizon (max_time conventions match).
    assert_eq!(left.max_time, right.max_time);

    // η tensors must agree element-wise within numerical tolerance.
    let diff: f32 = (left.eta.clone() - right.eta.clone())
        .abs()
        .max()
        .into_scalar()
        .elem();
    assert!(
        diff < 1e-4,
        "Associativity violated: max |left.η - right.η| = {}",
        diff
    );
}

// ============================================================================
// Sequence ↔ pairwise equivalence
// ============================================================================

#[test]
fn compose_sequence_matches_iterated_pairwise() {
    let a = deterministic_stok(4, 6);
    let b = deterministic_stok(4, 6);
    let c = deterministic_stok(4, 6);

    // Via compose_sequence builder.
    let seq = OptionSequence::new()
        .then(a.clone(), Some("a"))
        .then(b.clone(), Some("b"))
        .then(c.clone(), Some("c"));
    let from_sequence = compose_sequence(&seq).unwrap();

    // Manual iterated pairwise: compose(compose(a, b), c).
    let ab = compose_stoks(&a, &b).unwrap();
    let ab_kernel = ab.to_stok_kernel().unwrap();
    let manual = compose_stoks(&ab_kernel, &c).unwrap();

    assert_eq!(from_sequence.max_time, manual.max_time);

    let diff: f32 = (from_sequence.eta.clone() - manual.eta.clone())
        .abs()
        .max()
        .into_scalar()
        .elem();
    assert!(
        diff < 1e-5,
        "compose_sequence ≠ iterated pairwise: max diff = {}",
        diff
    );
}

// ============================================================================
// Eq [19] SOK composition (matrix multiplication)
// ============================================================================

#[test]
fn eq19_sok_composition_is_matrix_multiplication() {
    let device = default_device();

    let chi1: Tensor<DefaultBackend, 2> =
        Tensor::from_floats([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.0, 0.4, 0.6]], &device);
    let chi2: Tensor<DefaultBackend, 2> =
        Tensor::from_floats([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.4, 0.4, 0.2]], &device);

    let sok1 = StateOptionKernel::from_chi(chi1.clone());
    let sok2 = StateOptionKernel::from_chi(chi2.clone());
    let composed = compose_soks(&sok1, &sok2).unwrap();

    // χ_μ = χ_1 @ χ_2.
    let expected = chi1.matmul(chi2);
    let diff: f32 = (composed.chi - expected).abs().max().into_scalar().elem();
    assert!(diff < 1e-6, "Eq [19]: max |χ_μ - χ_1 χ_2| = {}", diff);
}

#[test]
fn eq19_sok_composition_preserves_row_normalization() {
    let device = default_device();

    let chi1: Tensor<DefaultBackend, 2> =
        Tensor::from_floats([[0.4, 0.6], [0.7, 0.3]], &device);
    let chi2: Tensor<DefaultBackend, 2> =
        Tensor::from_floats([[0.2, 0.8], [0.5, 0.5]], &device);

    let composed = compose_soks(
        &StateOptionKernel::from_chi(chi1),
        &StateOptionKernel::from_chi(chi2),
    )
    .unwrap();

    // Each row of χ_μ should still sum to 1.
    let row_sums: Vec<f32> = composed
        .chi
        .clone()
        .sum_dim(1)
        .squeeze::<1>()
        .into_data()
        .to_vec()
        .unwrap();
    for (i, &s) in row_sums.iter().enumerate() {
        assert!(approx(s, 1.0, 1e-6), "row {} sums to {}, expected 1.0", i, s);
    }
}

// ============================================================================
// Terminal event semantics: failure-then-anything stays failed
// ============================================================================

#[test]
fn terminal_event_o1_failure_propagates_to_sequence_failure() {
    // If o1 has nonzero η-, the sequence's η-_μ inherits at least that mass.
    // Concretely: η-_μ ≥ η-_1 (failure at any time in o1 ⟹ failure of sequence).
    let device = default_device();
    // Constrained chain with fire at state 2 — states 0, 1, 2 have all mass in η-.
    let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(5, 2, 10, &device);
    let stok1 = feasibility_iteration(&mdp, FeasibilityIterationConfig::default())
        .unwrap()
        .kernel;
    // o2 = an unconstrained chain (any STOK with same n_states).
    let stok2 = deterministic_stok(5, 10);

    let composed = compose_stoks(&stok1, &stok2).unwrap();

    // For initial states inside the infeasible region of o1 (κ_1 = 0):
    //   η+_μ should still be 0 (nothing can succeed).
    //   η-_μ should sum to 1 (everything fails).
    let kappa1: Vec<f32> = stok1.kappa.clone().into_data().to_vec().unwrap();
    let mu_eta_minus_total = sum_to_state_vec(&composed.eta_minus);
    let mu_eta_plus_total = sum_to_state_vec(&composed.eta_plus);

    for i in 0..5 {
        if kappa1[i] < 1e-5 {
            // Infeasible-under-o1 initial state: failure must dominate.
            assert!(
                approx(mu_eta_minus_total[i], 1.0, 1e-3),
                "infeasible state {}: Σ η-_μ = {}, expected ≈ 1.0",
                i,
                mu_eta_minus_total[i]
            );
            assert!(
                mu_eta_plus_total[i] < 1e-5,
                "infeasible state {}: Σ η+_μ = {}, expected ≈ 0.0",
                i,
                mu_eta_plus_total[i]
            );
        }
    }
}
