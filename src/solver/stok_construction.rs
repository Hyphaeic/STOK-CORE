//! STOK construction from converged κ* and π**.
//!
//! After feasibility iteration converges, this module builds the full
//! State-Time Option Kernel by computing η⁺ (success termination) and
//! η⁻ (failure termination) distributions.
//!
//! # Key Equations (Ringstrom & Schrater 2025)
//!
//! ## Boundary Conditions (t = t₀)
//! - Eq [11]: η⁺(x_j, t₀ | x_i) = f₁(x_i, π*(x_i)) · δ_{ij}
//! - Eq [12]: η⁻(x_j, t₀ | x_i) = [𝟙_κ(x_i)(1 - f_c(x_i, π*)) + 𝟙̄_κ(x_i)] · δ_{ij}
//!
//! ## Recursive Updates (t > t₀)  
//! - Eq [9]: η⁺(x_f, t_f | x) = f₂(x, π*) · E_{x'~P_π}[η⁺(x_f, t_f-1 | x')]
//! - Eq [10]: η⁻(x_f, t_f | x) = f₂(x, π*) · E_{x'~P_π}[η⁻(x_f, t_f-1 | x')]

use burn::prelude::*;
use burn::tensor::Int;

use crate::mdp::TaskMDP;
use crate::solver::bellman::{gather_by_policy, get_policy_transition};
use crate::stok::STOKKernel;
use crate::types::{STOKDimensions, StokError};
use crate::utils::diag_matrix;

// ============================================================================
// Main STOK Construction
// ============================================================================

/// Construct a complete STOK from converged κ* and π**.
///
/// This function builds the full termination probability distributions
/// η⁺ and η⁻ using the recursive equations from the paper.
///
/// # Arguments
/// * `kappa` - Converged cumulative feasibility, shape [S]
/// * `policy` - Optimal policy, shape [S]
/// * `mdp` - Task MDP
/// * `max_time` - Maximum time horizon for the STOK
///
/// # Returns
/// Complete STOKKernel with η⁺, η⁻, κ, π
///
/// # Invariants Guaranteed
/// - Σ_{x_f, t_f} η**(x_f, t_f | x) = 1.0 for all x (normalization)
/// - κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x) (consistency)
pub fn construct_stok<B: Backend>(
    kappa: &Tensor<B, 1>,
    policy: &Tensor<B, 1, Int>,
    mdp: &TaskMDP<B>,
    max_time: usize,
    kappa_threshold: f32,
) -> Result<STOKKernel<B>, StokError> {
    let device = mdp.device();
    let s = mdp.n_states();

    // Validate inputs
    if max_time == 0 {
        return Err(StokError::InvalidDimension {
            name: "max_time".into(),
            value: 0,
            reason: "max_time must be > 0 for STOK construction".into(),
        });
    }

    // ========================================================================
    // Step 1: Extract policy-conditioned quantities
    // ========================================================================

    // P_π(x'|x) = P(x'|x, π(x))
    let p_pi = get_policy_transition(&mdp.transition, policy);

    // f₁_π(x) = f₁(x, π(x)) - immediate success probability
    let f1_pi = gather_by_policy(&mdp.f1, policy);
    // f₂_π(x) = f₂(x, π(x)) - continuation probability

    let f2_pi = gather_by_policy(&mdp.f2, policy);

    // f_c_π(x) = f_c(x, π(x)) - constraint satisfaction probability
    let fc_pi = gather_by_policy(&mdp.constraint_fn, policy);

    // ========================================================================
    // Step 2: Initialize η tensors
    // ========================================================================

    let mut eta_plus: Tensor<B, 3> = Tensor::zeros([s, s, max_time], &device);
    let mut eta_minus: Tensor<B, 3> = Tensor::zeros([s, s, max_time], &device);

    // ========================================================================
    // Step 3: Compute boundary conditions (t = 0)
    // ========================================================================

    // η⁺(x_j, t₀ | x_i) = f₁(x_i, π*) · δ_{ij}
    // This is a diagonal matrix with f₁_π on the diagonal
    let eta_plus_t0 = compute_eta_plus_boundary(&f1_pi);

    // η⁻(x_j, t₀ | x_i) = [𝟙_κ(x_i)(1 - f_c) + 𝟙̄_κ(x_i)] · δ_{ij}
    let eta_minus_t0 = compute_eta_minus_boundary(&fc_pi, kappa, kappa_threshold);

    // Set t=0 slices
    eta_plus = set_time_slice(eta_plus, &eta_plus_t0, 0);
    eta_minus = set_time_slice(eta_minus, &eta_minus_t0, 0);

    // ========================================================================
    // Step 4: Forward propagation (t = 1..T-1)
    // ========================================================================

    for t in 1..max_time {
        // Get previous time slice
        let eta_plus_prev = get_time_slice(&eta_plus, t - 1);
        let eta_minus_prev = get_time_slice(&eta_minus, t - 1);

        // Compute new time slice using recursive update
        let eta_plus_t = stok_time_step(&eta_plus_prev, &f2_pi, &p_pi);
        let eta_minus_t = stok_time_step(&eta_minus_prev, &f2_pi, &p_pi);

        // Store in tensor
        eta_plus = set_time_slice(eta_plus, &eta_plus_t, t);
        eta_minus = set_time_slice(eta_minus, &eta_minus_t, t);
    }

    // ========================================================================
    // Step 5: Construct kernel
    // ========================================================================

    let dims = STOKDimensions {
        n_states: s,
        max_time,
    };

    Ok(STOKKernel::from_components(
        eta_plus,
        eta_minus,
        kappa.clone(),
        policy.clone(),
        dims,
    ))
}

// ============================================================================
// Boundary Condition Computation
// ============================================================================

/// Compute η⁺ boundary condition at t = 0.
///
/// Eq [11]: η⁺(x_j, t₀ | x_i) = f₁(x_i, π*(x_i)) · δ_{ij}
///
/// Returns diagonal matrix with f₁_π on diagonal.
fn compute_eta_plus_boundary<B: Backend>(f1_pi: &Tensor<B, 1>) -> Tensor<B, 2> {
    diag_matrix(f1_pi)
}

/// Compute η⁻ boundary condition at t = 0.
///
/// Eq [12]: η⁻(x_j, t₀ | x_i) = [𝟙_κ(x_i)(1 - f_c(x_i, π*)) + 𝟙̄_κ(x_i)] · δ_{ij}
///
/// Where:
/// - 𝟙_κ(x) = 1 if κ(x) > threshold, else 0 (feasibility indicator)
/// - 𝟙̄_κ(x) = 1 - 𝟙_κ(x) (infeasibility indicator)
///
/// Interpretation:
/// - For feasible states: immediate failure = constraint violation = (1 - f_c)
/// - For infeasible states: immediate failure = 1 (terminate immediately)
fn compute_eta_minus_boundary<B: Backend>(
    fc_pi: &Tensor<B, 1>,
    kappa: &Tensor<B, 1>,
    threshold: f32,
) -> Tensor<B, 2> {
    let device = kappa.device();
    let s = kappa.dims()[0];

    // Feasibility indicator: 𝟙_κ(x) = 1 if κ(x) > threshold
    let feasible_mask = kappa.clone().greater_elem(threshold);
    let feasible_indicator: Tensor<B, 1> = feasible_mask.clone().float();

    // Infeasibility indicator: 𝟙̄_κ(x) = 1 - 𝟙_κ(x)
    let ones: Tensor<B, 1> = Tensor::ones([s], &device);
    let infeasible_indicator = ones.clone() - feasible_indicator.clone();

    // For feasible states: (1 - f_c)
    // For infeasible states: 1
    let eta_minus_diag = feasible_indicator * (ones - fc_pi.clone()) + infeasible_indicator;

    diag_matrix(&eta_minus_diag)
}

// ============================================================================
// Recursive Update Step
// ============================================================================

/// Compute one time step of STOK propagation.
///
/// Implements Eq [9] and [10]:
/// η(x_f, t | x) = f₂(x, π*) · E_{x'~P_π}[η(x_f, t-1 | x')]
///
/// # Arguments
/// * `eta_prev` - Previous time slice η(·, t-1 | ·), shape [S, S]
/// * `f2_pi` - Continuation probability f₂(x, π(x)), shape [S]
/// * `p_pi` - Policy-conditioned transitions P_π(x'|x), shape [S, S]
///
/// # Returns
/// New time slice η(·, t | ·), shape [S, S]
fn stok_time_step<B: Backend>(
    eta_prev: &Tensor<B, 2>,
    f2_pi: &Tensor<B, 1>,
    p_pi: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    // E_{x' ~ P_π} [η(x_f, t-1 | x')]
    // = Σ_{x'} P_π(x'|x) · η(x_f, t-1 | x')
    // = P_π @ η_prev
    //
    // Note: P_π is [S, S] where P_π[x, x'] = P(x'|x, π(x))
    // eta_prev is [S, S] where eta_prev[x, x_f] = η(x_f, t-1 | x)
    //
    // We want: expected[x, x_f] = Σ_{x'} P_π[x, x'] · eta_prev[x', x_f]
    // This is: P_π @ eta_prev
    let expected_eta = p_pi.clone().matmul(eta_prev.clone()); // [S, S]

    // f₂(x) * E[η]
    // Expand f2_pi to [S, 1] for broadcasting
    let f2_expanded = f2_pi.clone().unsqueeze_dim(1); // [S, 1]

    expected_eta * f2_expanded // [S, S]
}

// ============================================================================
// Tensor Slice Utilities
// ============================================================================

/// Extract a 2D time slice from a 3D STOK tensor.
///
/// # Arguments
/// * `tensor` - 3D tensor of shape [S, S, T]
/// * `t` - Time index to extract
///
/// # Returns
/// 2D tensor of shape [S, S] representing η(·, t | ·)
fn get_time_slice<B: Backend>(tensor: &Tensor<B, 3>, t: usize) -> Tensor<B, 2> {
    let dims = tensor.dims();
    let s = dims[0];

    tensor
        .clone()
        .slice([0..s, 0..s, t..(t + 1)])
        .squeeze::<2>() // Output is 2D tensor [S, S]
}

/// Set a 2D time slice in a 3D STOK tensor.
///
/// # Arguments
/// * `tensor` - 3D tensor of shape [S, S, T]
/// * `slice` - 2D tensor of shape [S, S] to insert
/// * `t` - Time index at which to insert
///
/// # Returns
/// Updated 3D tensor with the slice inserted at time t
fn set_time_slice<B: Backend>(
    tensor: Tensor<B, 3>,
    slice: &Tensor<B, 2>,
    t: usize,
) -> Tensor<B, 3> {
    let dims = tensor.dims();
    let s = dims[0];

    // Expand slice to 3D: [S, S] -> [S, S, 1]
    let slice_3d = slice.clone().unsqueeze_dim(2);

    // Use slice_assign to update the tensor at time t
    tensor.slice_assign([0..s, 0..s, t..(t + 1)], slice_3d)
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Validate STOK normalization invariant.
///
/// For each initial state x, the total probability mass should sum to 1:
/// Σ_{x_f, t_f} η**(x_f, t_f | x) = 1.0
///
/// # Returns
/// - `Ok(())` if normalization holds within tolerance
/// - `Err` with details if violated
pub fn validate_stok_normalization<B: Backend>(
    eta_plus: &Tensor<B, 3>,
    eta_minus: &Tensor<B, 3>,
    tolerance: f32,
) -> Result<(), StokError> {
    let combined = eta_plus.clone() + eta_minus.clone();

    // Sum over x_f (dim 1) and t_f (dim 2)
    let row_sums = combined
        .sum_dim(2)
        .squeeze::<2>() // [S, S, T] -> [S, S]
        .sum_dim(1)
        .squeeze::<1>(); // [S, S] -> [S]

    let row_sums_data: Vec<f32> = row_sums.into_data().to_vec().unwrap();

    for (i, &sum) in row_sums_data.iter().enumerate() {
        if (sum - 1.0).abs() > tolerance {
            return Err(StokError::NotNormalized {
                sum,
                expected: 1.0,
                tolerance,
            });
        }
    }

    Ok(())
}

/// Validate κ-η consistency.
///
/// κ(x) should equal the sum of η⁺ over all final states and times:
/// κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
pub fn validate_kappa_eta_consistency<B: Backend>(
    kappa: &Tensor<B, 1>,
    eta_plus: &Tensor<B, 3>,
    tolerance: f32,
) -> Result<(), StokError> {
    // Sum η⁺ over x_f and t_f
    let kappa_from_eta = eta_plus
        .clone()
        .sum_dim(2)
        .squeeze::<2>() // [S, S, T] -> [S, S]
        .sum_dim(1)
        .squeeze::<1>(); // [S, S] -> [S]

    let kappa_data: Vec<f32> = kappa.clone().into_data().to_vec().unwrap();
    let kappa_eta_data: Vec<f32> = kappa_from_eta.into_data().to_vec().unwrap();

    for (i, (&k, &k_eta)) in kappa_data.iter().zip(kappa_eta_data.iter()).enumerate() {
        if (k - k_eta).abs() > tolerance {
            return Err(StokError::InvalidProbability {
                value: k_eta,
                context: format!("κ-η mismatch at state {}: κ={}, Ση⁺={}", i, k, k_eta),
            });
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::mdp::TaskMDP;
    use crate::solver::feasibility_iteration::{feasibility_iteration, FeasibilityIterationConfig};
    use crate::utils::approx_eq;

    /// Run feasibility iteration to convergence (κ-OKBE then π-OKBE).
    fn run_to_convergence<B: Backend>(mdp: &TaskMDP<B>) -> (Tensor<B, 1>, Tensor<B, 1, Int>) {
        // Use the main solver so tests match the paper-aligned implementation.
        let config = FeasibilityIterationConfig {
            compute_full_stok: false,
            ..Default::default()
        };

        let result = feasibility_iteration(mdp, config).unwrap();
        (result.kernel.kappa.clone(), result.kernel.policy.clone())
    }

    #[test]
    fn test_construct_stok_shapes() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);

        let (kappa, policy) = run_to_convergence(&mdp);
        let stok = construct_stok(&kappa, &policy, &mdp, 10, 1e-6).unwrap();

        assert_eq!(stok.eta_plus.dims(), [5, 5, 10]);
        assert_eq!(stok.eta_minus.dims(), [5, 5, 10]);
        assert_eq!(stok.kappa.dims(), [5]);
        assert_eq!(stok.policy.dims(), [5]);
    }

    #[test]
    fn test_stok_normalization() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 15, &device);

        let (kappa, policy) = run_to_convergence(&mdp);
        let stok = construct_stok(&kappa, &policy, &mdp, 15, 1e-6).unwrap();

        // Validate normalization
        let result = validate_stok_normalization(&stok.eta_plus, &stok.eta_minus, 1e-4);
        assert!(result.is_ok(), "Normalization failed: {:?}", result);
    }

    #[test]
    fn test_kappa_eta_consistency() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 15, &device);

        let (kappa, policy) = run_to_convergence(&mdp);
        let stok = construct_stok(&kappa, &policy, &mdp, 15, 1e-6).unwrap();

        // Validate κ-η consistency
        let result = validate_kappa_eta_consistency(&stok.kappa, &stok.eta_plus, 1e-4);
        assert!(result.is_ok(), "κ-η consistency failed: {:?}", result);
    }

    #[test]
    fn test_eta_plus_boundary() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 5, &device);

        let (kappa, policy) = run_to_convergence(&mdp);
        let stok = construct_stok(&kappa, &policy, &mdp, 5, 1e-6).unwrap();

        // At t=0, η⁺ should be diagonal
        let eta_plus_t0 = get_time_slice(&stok.eta_plus, 0);
        let eta_plus_data: Vec<f32> = eta_plus_t0.into_data().to_vec().unwrap();

        // Off-diagonal elements should be zero
        assert!(approx_eq(eta_plus_data[1], 0.0, 1e-6)); // [0,1]
        assert!(approx_eq(eta_plus_data[2], 0.0, 1e-6)); // [0,2]
        assert!(approx_eq(eta_plus_data[3], 0.0, 1e-6)); // [1,0]
        assert!(approx_eq(eta_plus_data[5], 0.0, 1e-6)); // [1,2]

        // Goal state (2) should have η⁺[2,2,0] = 1.0
        assert!(
            approx_eq(eta_plus_data[8], 1.0, 1e-5),
            "Goal state η⁺[2,2,0] should be 1.0, got {}",
            eta_plus_data[8]
        );
    }

    #[test]
    fn test_constrained_stok() {
        let device = default_device();
        // Chain with fire at state 1
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(3, 1, 5, &device);

        let (kappa, policy) = run_to_convergence(&mdp);
        let stok = construct_stok(&kappa, &policy, &mdp, 5, 1e-6).unwrap();

        // States 0 and 1 should have all mass in η⁻ (infeasible)
        // State 2 (goal) should have all mass in η⁺

        let kappa_data: Vec<f32> = stok.kappa.clone().into_data().to_vec().unwrap();

        // Check feasibility
        assert!(kappa_data[0] < 0.01, "State 0 should be infeasible");
        assert!(kappa_data[1] < 0.01, "State 1 (fire) should be infeasible");
        assert!(
            approx_eq(kappa_data[2], 1.0, 1e-5),
            "State 2 (goal) should be feasible"
        );
    }

    #[test]
    fn test_get_set_time_slice() {
        let device = default_device();

        let tensor: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 3, 5], &device);
        let slice: Tensor<DefaultBackend, 2> = Tensor::ones([3, 3], &device);

        let updated = set_time_slice(tensor, &slice, 2);
        let retrieved = get_time_slice(&updated, 2);

        let retrieved_data: Vec<f32> = retrieved.into_data().to_vec().unwrap();

        for &v in &retrieved_data {
            assert!(approx_eq(v, 1.0, 1e-6));
        }
    }
}
