//! Bellman operators for feasibility iteration.
//!
//! Implements the κ-OKBE (Option Kernel Bellman Equation) which computes
//! the optimal cumulative feasibility function.
//!
//! # Mathematical Background
//!
//! ## κ-OKBE (Equation [7])
//!
//! ```text
//! κ*(x) = max_a [ f₁(x,a) + f₂(x,a) · Σ_{x'} P(x'|x,a) · κ*(x') ]
//! ```
//!
//! Where:
//! - `f₁(x,a) = f_g(x,a) · f_c(x,a)` — immediate goal with constraint
//! - `f₂(x,a) = (1 - f_g(x,a)) · f_c(x,a)` — continuation weight
//! - `P(x'|x,a)` — transition probability
//!
//! ## Interpretation
//!
//! - **Term 1**: Probability of achieving goal NOW while satisfying constraint
//! - **Term 2**: Probability of NOT achieving goal NOW, satisfying constraint,
//!   and eventually achieving goal from successor state
//!
//! # Reference
//!
//! Ringstrom & Schrater (2025), Section 4, Appendix E

use burn::prelude::*;
use burn::tensor::Int;

use crate::mdp::TaskMDP;

// ============================================================================
// Core κ-OKBE Operator
// ============================================================================

/// Single Bellman backup step for κ-OKBE (Equation [7]).
///
/// Computes one iteration of the feasibility Bellman equation:
/// ```text
/// κ_new(x) = max_a [ f₁(x,a) + f₂(x,a) · E_{x'}[κ(x')] ]
/// π(x) = argmax_a [ ... ]
/// ```
///
/// # Arguments
/// * `kappa` - Current feasibility estimate, shape [S]
/// * `mdp` - Task MDP containing P, f₁, f₂
///
/// # Returns
/// Tuple of (new_kappa, policy) where:
/// - `new_kappa`: Updated feasibility, shape [S]
/// - `policy`: Optimal action indices, shape [S] (Int tensor)
///
/// # Tensor Operations
///
/// 1. Compute expected future feasibility: `E[κ(x')] = Σ_{x'} P(x'|x,a) κ(x')`
/// 2. Compute Q-values: `Q(x,a) = f₁(x,a) + f₂(x,a) · E[κ(x')]`
/// 3. Maximize over actions: `κ_new = max_a Q`, `π = argmax_a Q`
///
/// # Example
/// ```ignore
/// let (kappa_new, policy) = bellman_backup_kappa(&kappa, &mdp);
/// ```
pub fn bellman_backup_kappa<B: Backend>(
    kappa: &Tensor<B, 1>,
    mdp: &TaskMDP<B>,
) -> (Tensor<B, 1>, Tensor<B, 1, Int>) {
    let s = mdp.n_states();
    let a = mdp.n_actions();
    
    // Step 1: Compute expected future feasibility E_{x'}[κ(x')]
    // P has shape [S, A, S], we need E[κ] for each (state, action) pair
    //
    // Reshape P from [S, A, S] to [S*A, S] for batch matmul
    // Reshape κ from [S] to [S, 1] as column vector
    // Result: [S*A, 1] -> reshape to [S, A]
    
    // TODO: Implement Step 1
    // let P_flat = mdp.transition().clone().reshape([s * a, s]);
    // let kappa_col = kappa.clone().reshape([s, 1]);
    // let expected_kappa = P_flat.matmul(kappa_col).reshape([s, a]);
    
    // Step 2: Compute Q-values
    // Q(x, a) = f₁(x,a) + f₂(x,a) * E[κ(x')]
    
    // TODO: Implement Step 2
    // let Q = mdp.f1().clone() + mdp.f2().clone() * expected_kappa;
    
    // Step 3: Maximize over actions
    // κ_new(x) = max_a Q(x,a)
    // π(x) = argmax_a Q(x,a)
    
    // TODO: Implement Step 3
    // let (kappa_new, policy) = Q.max_dim_with_indices(1);
    // let kappa_new = kappa_new.squeeze(1);  // [S, 1] -> [S]
    // let policy = policy.squeeze(1);         // [S, 1] -> [S]
    
    // TODO: Return actual result
    // (kappa_new, policy)
    
    todo!("Implement bellman_backup_kappa - see steps above")
}

// ============================================================================
// Policy Extraction Helpers
// ============================================================================

/// Extract policy-conditioned transition matrix P_π.
///
/// Given transition tensor P[x, a, x'] and policy π[x], extracts
/// the transition matrix under the policy: P_π[x, x'] = P[x, π(x), x']
///
/// # Arguments
/// * `transition` - Full transition tensor, shape [S, A, S]
/// * `policy` - Policy action indices, shape [S]
///
/// # Returns
/// Policy-conditioned transition matrix, shape [S, S]
pub fn get_policy_transition<B: Backend>(
    transition: &Tensor<B, 3>,
    policy: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    // TODO: Implement
    // This requires gathering along the action dimension using the policy
    // 
    // For each state x:
    //   P_π[x, :] = P[x, π(x), :]
    //
    // Burn's gather operation or manual indexing needed
    
    todo!("Implement get_policy_transition")
}

/// Gather values from a 2D tensor using policy indices.
///
/// Given tensor f[x, a] and policy π[x], extracts f^π[x] = f[x, π(x)]
///
/// # Arguments
/// * `f` - Function tensor, shape [S, A]
/// * `policy` - Policy action indices, shape [S]
///
/// # Returns
/// Policy-conditioned values, shape [S]
pub fn gather_by_policy<B: Backend>(
    f: &Tensor<B, 2>,
    policy: &Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    // TODO: Implement
    // This gathers f[x, π(x)] for each state x
    //
    // Approach: Use Burn's gather operation along dimension 1
    
    todo!("Implement gather_by_policy")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    // TODO: Add tests
    // - test_bellman_backup_single_step
    // - test_bellman_backup_goal_state (κ should be 1.0)
    // - test_bellman_backup_unreachable (κ should stay 0)
    // - test_bellman_backup_constrained (f_c = 0 means κ = 0)
    // - test_policy_extraction
    // - test_gather_by_policy
}
