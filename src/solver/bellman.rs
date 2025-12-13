//! Bellman operators for STOK computation.
//!
//! Implements the κ-OKBE (Option Kernel Bellman Equation) from
//! Ringstrom & Schrater (2025) Equation [7]:
//!
//! ```text
//! κ*(x) = max_a [ f₁(x,a) + f₂(x,a) · Σ_{x'} P(x'|x,a) · κ*(x') ]
//! ```
//!
//! This is the core optimization primitive that computes the optimal
//! cumulative feasibility function.

use burn::prelude::*;
use burn::tensor::Int;

use crate::mdp::TaskMDP;

// ============================================================================
// Core Bellman Operator
// ============================================================================

/// Perform a single Bellman backup for the κ-OKBE.
///
/// Implements Equation [7] from the paper:
/// ```text
/// κ*(x) = max_a [ f₁(x,a) + f₂(x,a) · E_{x'~P}[κ*(x')] ]
/// ```
///
/// # Arguments
/// * `kappa` - Current feasibility estimate, shape [S]
/// * `mdp` - Task MDP containing P, f₁, f₂
///
/// # Returns
/// Tuple of:
/// * `kappa_new` - Updated feasibility, shape [S]
/// * `policy` - Greedy policy (argmax actions), shape [S]
///
/// # Tensor Operations
/// 1. Compute expected future feasibility: E[κ(x')] for each (x,a)
/// 2. Compute Q-values: Q(x,a) = f₁ + f₂ * E[κ]
/// 3. Maximize over actions to get new κ and policy
///
/// # Performance
/// This function is designed for GPU execution with JIT fusion:
/// - Single matmul for expected value computation
/// - Fused elementwise ops for Q-values
/// - Parallel reduction for max/argmax
pub fn bellman_backup_kappa<B: Backend>(
    kappa: &Tensor<B, 1>,
    mdp: &TaskMDP<B>,
) -> (Tensor<B, 1>, Tensor<B, 1, Int>) {
    let s = mdp.n_states();
    let a = mdp.n_actions();
    
    // Step 1: Compute expected future feasibility E_{x'}[κ(x')]
    // Reshape P from [S, A, S] to [S*A, S] for batched matmul
    let p_flat = mdp.transition.clone().reshape([s * a, s]);
    
    // Reshape κ to column vector [S, 1]
    let kappa_col = kappa.clone().reshape([s, 1]);
    
    // Batched matmul: [S*A, S] × [S, 1] = [S*A, 1]
    // Then reshape to [S, A]
    let expected_kappa = p_flat.matmul(kappa_col).reshape([s, a]);
    
    // Step 2: Compute Q-values
    // Q(x, a) = f₁(x,a) + f₂(x,a) * E[κ(x')]
    // This should fuse with Step 1 in JIT compilation
    let q_values = mdp.f1.clone() + mdp.f2.clone() * expected_kappa;
    
    // Step 3: Maximize over actions
    // For each state, find max Q-value and argmax action
    let (kappa_new, policy) = q_values.max_dim_with_indices(1);
    
    // Squeeze from [S, 1] to [S]
    (kappa_new.squeeze(1), policy.squeeze(1))
}

/// Compute Q-values without performing the max reduction.
///
/// Useful for policy analysis and time-minimization tiebreaking.
///
/// # Returns
/// Q-values tensor of shape [S, A]
pub fn compute_q_values<B: Backend>(
    kappa: &Tensor<B, 1>,
    mdp: &TaskMDP<B>,
) -> Tensor<B, 2> {
    let s = mdp.n_states();
    let a = mdp.n_actions();
    
    let p_flat = mdp.transition.clone().reshape([s * a, s]);
    let kappa_col = kappa.clone().reshape([s, 1]);
    let expected_kappa = p_flat.matmul(kappa_col).reshape([s, a]);
    
    mdp.f1.clone() + mdp.f2.clone() * expected_kappa
}

// ============================================================================
// Policy Extraction Utilities
// ============================================================================

/// Extract policy-conditioned transition matrix P_π.
///
/// Given P(x'|x,a) and π(x), compute P_π(x'|x) = P(x'|x, π(x))
///
/// # Arguments
/// * `transition` - Full transition tensor, shape [S, A, S]
/// * `policy` - Policy tensor, shape [S] with action indices
///
/// # Returns
/// Policy-conditioned transitions, shape [S, S]
pub fn get_policy_transition<B: Backend>(
    transition: &Tensor<B, 3>,
    policy: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let s = transition.dims()[0];
    let s_next = transition.dims()[2];
    
    // Expand policy for gathering: [S] -> [S, 1, S_next]
    let policy_expanded = policy.clone()
        .reshape([s, 1])
        .expand([s, 1, s_next]);
    
    // Gather along action dimension (dim 1)
    // [S, A, S] gather with [S, 1, S] -> [S, 1, S] -> [S, S]
    transition.clone()
        .gather(1, policy_expanded)
        .squeeze(1)
}

/// Extract values at policy actions: v_π(x) = v(x, π(x))
///
/// # Arguments
/// * `values` - Values for each state-action pair, shape [S, A]
/// * `policy` - Policy tensor, shape [S] with action indices
///
/// # Returns
/// Policy-conditioned values, shape [S]
pub fn gather_by_policy<B: Backend>(
    values: &Tensor<B, 2>,
    policy: &Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let s = values.dims()[0];
    
    // Expand policy for gathering: [S] -> [S, 1]
    let policy_expanded = policy.clone().reshape([s, 1]);
    
    // Gather along action dimension and squeeze
    values.clone()
        .gather(1, policy_expanded)
        .squeeze(1)
}

// ============================================================================
// Optional: Time-Minimization Policy (π-OKBE)
// ============================================================================

/// Refine policy to minimize expected time among equally-feasible actions.
///
/// Implements Equation [8] from the paper (simplified version):
/// Among actions that achieve κ*(x), select the one with minimum expected time.
///
/// # Arguments
/// * `q_values` - Q-values from κ-OKBE, shape [S, A]
/// * `kappa_max` - Maximum κ per state, shape [S]
/// * `expected_time` - Expected time-to-goal per (s,a), shape [S, A]
/// * `tolerance` - Tolerance for "equally optimal"
///
/// # Returns
/// Refined policy with time-minimization tiebreaking, shape [S]
///
/// # Note
/// For MVP, this can be skipped - just use argmax from κ-OKBE.
#[allow(dead_code)]
pub fn bellman_backup_policy_tiebreak<B: Backend>(
    q_values: &Tensor<B, 2>,
    kappa_max: &Tensor<B, 1>,
    expected_time: &Tensor<B, 2>,
    tolerance: f32,
) -> Tensor<B, 1, Int> {
    let s = q_values.dims()[0];
    
    // Create mask for optimal actions: Q(x,a) ≥ κ*(x) - tolerance
    let kappa_expanded = kappa_max.clone().unsqueeze_dim(1); // [S, 1]
    let threshold = kappa_expanded.clone() - tolerance;
    let optimal_mask = q_values.clone().greater_equal(threshold);
    
    // For non-optimal actions, set time to large value
    let large_value: f32 = 1e10;
    let large_tensor: Tensor<B, 2> = Tensor::full([s, expected_time.dims()[1]], large_value, &expected_time.device());
    
    // Where optimal, use actual time; elsewhere use large value
    let masked_time = expected_time.clone()
        .mask_where(optimal_mask, large_tensor);
    
    // Select minimum-time action among optimal
    let (_, policy) = masked_time.min_dim_with_indices(1);
    policy.squeeze(1)
}

// ============================================================================
// Debug Utilities
// ============================================================================

/// Compute a single-state Bellman backup on CPU for debugging.
///
/// # Returns
/// Tuple of (new_kappa_value, optimal_action_index)
#[allow(dead_code)]
pub fn bellman_backup_single_state<B: Backend>(
    state: usize,
    kappa: &Tensor<B, 1>,
    mdp: &TaskMDP<B>,
) -> (f32, usize) {
    let n_actions = mdp.n_actions();
    let n_states = mdp.n_states();
    
    // Extract data to CPU
    let kappa_data: Vec<f32> = kappa.clone().into_data().to_vec().unwrap();
    let f1_data: Vec<f32> = mdp.f1.clone().into_data().to_vec().unwrap();
    let f2_data: Vec<f32> = mdp.f2.clone().into_data().to_vec().unwrap();
    let p_data: Vec<f32> = mdp.transition.clone().into_data().to_vec().unwrap();
    
    let mut best_q = f32::NEG_INFINITY;
    let mut best_action = 0usize;
    
    for a in 0..n_actions {
        // Compute expected future feasibility
        let mut expected_kappa = 0.0f32;
        for s_next in 0..n_states {
            // P[state, action, s_next]
            let p_idx = state * n_actions * n_states + a * n_states + s_next;
            expected_kappa += p_data[p_idx] * kappa_data[s_next];
        }
        
        // Compute Q-value
        let sa_idx = state * n_actions + a;
        let q = f1_data[sa_idx] + f2_data[sa_idx] * expected_kappa;
        
        if q > best_q {
            best_q = q;
            best_action = a;
        }
    }
    
    (best_q, best_action)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DefaultBackend, default_device};
    use crate::mdp::TaskMDP;
    use crate::utils::approx_eq;

    #[test]
    fn test_bellman_backup_shapes() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        let kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([5], &device);
        
        let (kappa_new, policy) = bellman_backup_kappa(&kappa, &mdp);
        
        assert_eq!(kappa_new.dims(), [5]);
        assert_eq!(policy.dims(), [5]);
    }

    #[test]
    fn test_bellman_backup_goal_state() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        // Zero initialization
        let kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([5], &device);
        
        // After one backup, goal state should have κ = 1.0
        let (kappa_new, _) = bellman_backup_kappa(&kappa, &mdp);
        let kappa_data: Vec<f32> = kappa_new.into_data().to_vec().unwrap();
        
        // Goal state is state 4 (last state in 5-state chain)
        assert!(approx_eq(kappa_data[4], 1.0, 1e-6), 
            "Goal state should have κ=1.0, got {}", kappa_data[4]);
    }

    #[test]
    fn test_bellman_backup_propagation() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        // Run multiple iterations to propagate feasibility
        let mut kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([5], &device);
        
        for _ in 0..5 {
            let (kappa_new, _) = bellman_backup_kappa(&kappa, &mdp);
            kappa = kappa_new;
        }
        
        let kappa_data: Vec<f32> = kappa.into_data().to_vec().unwrap();
        
        // All states should be feasible (κ = 1.0) after enough iterations
        for (i, &k) in kappa_data.iter().enumerate() {
            assert!(approx_eq(k, 1.0, 1e-5), 
                "State {} should have κ=1.0, got {}", i, k);
        }
    }

    #[test]
    fn test_bellman_backup_constrained() {
        let device = default_device();
        // Create chain with fire at state 2
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(5, 2, 10, &device);
        
        // Run to convergence
        let mut kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([5], &device);
        
        for _ in 0..10 {
            let (kappa_new, _) = bellman_backup_kappa(&kappa, &mdp);
            kappa = kappa_new;
        }
        
        let kappa_data: Vec<f32> = kappa.into_data().to_vec().unwrap();
        
        // States 0, 1 are blocked by fire at state 2
        // States 3, 4 should be feasible
        assert!(kappa_data[0] < 0.01, "State 0 should be infeasible, got {}", kappa_data[0]);
        assert!(kappa_data[1] < 0.01, "State 1 should be infeasible, got {}", kappa_data[1]);
        assert!(kappa_data[2] < 0.01, "State 2 (fire) should be infeasible, got {}", kappa_data[2]);
        assert!(approx_eq(kappa_data[3], 1.0, 1e-5), "State 3 should be feasible, got {}", kappa_data[3]);
        assert!(approx_eq(kappa_data[4], 1.0, 1e-5), "State 4 should be feasible, got {}", kappa_data[4]);
    }

    #[test]
    fn test_get_policy_transition() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 5, &device);
        
        // Create a simple policy: all states take action 0
        let policy: Tensor<DefaultBackend, 1, Int> = Tensor::zeros([3], &device);
        
        let p_pi = get_policy_transition(&mdp.transition, &policy);
        
        assert_eq!(p_pi.dims(), [3, 3]);
        
        // Should be row-stochastic
        let row_sums = p_pi.clone().sum_dim(1);
        let row_sums_data: Vec<f32> = row_sums.into_data().to_vec().unwrap();
        
        for (i, &sum) in row_sums_data.iter().enumerate() {
            assert!(approx_eq(sum, 1.0, 1e-5), 
                "Row {} should sum to 1.0, got {}", i, sum);
        }
    }

    #[test]
    fn test_gather_by_policy() {
        let device = default_device();
        
        // Create test values [S=3, A=2]
        let values: Tensor<DefaultBackend, 2> = 
            Tensor::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
        
        // Policy: state 0 -> action 1, state 1 -> action 0, state 2 -> action 1
        let policy: Tensor<DefaultBackend, 1, Int> = 
            Tensor::from_ints([1, 0, 1], &device);
        
        let gathered = gather_by_policy(&values, &policy);
        let gathered_data: Vec<f32> = gathered.into_data().to_vec().unwrap();
        
        // Expected: [values[0,1], values[1,0], values[2,1]] = [2.0, 3.0, 6.0]
        assert!(approx_eq(gathered_data[0], 2.0, 1e-6));
        assert!(approx_eq(gathered_data[1], 3.0, 1e-6));
        assert!(approx_eq(gathered_data[2], 6.0, 1e-6));
    }

    #[test]
    fn test_policy_validity() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        let kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([5], &device);
        
        let (_, policy) = bellman_backup_kappa(&kappa, &mdp);
        let policy_data: Vec<i64> = policy.into_data().to_vec().unwrap();
        
        let n_actions = mdp.n_actions() as i64;
        for (i, &a) in policy_data.iter().enumerate() {
            assert!(a >= 0 && a < n_actions,
                "State {} has invalid action {}, should be in [0, {})", i, a, n_actions);
        }
    }
}