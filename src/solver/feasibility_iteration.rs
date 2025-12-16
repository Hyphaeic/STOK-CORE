//! Feasibility Iteration - The Core Optimization Loop.
//!
//! This module implements the main algorithm that computes the optimal
//! cumulative feasibility function κ* and policy π**, then constructs
//! the complete STOK.
//!
//! # Algorithm Overview
//!
//! ```text
//! 1. Initialize: κ ← 0, π ← 0  (CRITICAL: must be zero)
//! 2. Repeat until convergence:
//!    (κ_new, π_new) ← bellman_backup_kappa(κ, MDP)
//!    δ ← ||κ_new - κ||_∞
//!    κ ← κ_new, π ← π_new
//!    if δ < ε: break
//! 3. Construct STOK from (κ*, π**)
//! ```
//!
//! # Theoretical Guarantees
//!
//! Per Ringstrom & Schrater (2025) Appendix E:
//! - Convergence is guaranteed (contraction mapping)
//! - κ is monotonically non-decreasing
//! - Zero initialization is essential for correctness

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Int;

use crate::mdp::TaskMDP;
use crate::stok::STOKKernel;
use crate::types::StokError;
use crate::types::STOKDimensions;
use crate::solver::bellman::{
    bellman_backup_kappa,
    compute_q_values,
    get_policy_transition,
    gather_by_policy,
};
use crate::solver::convergence::{
    ConvergenceConfig, ConvergenceState, ConvergenceReason,
    check_kappa_convergence, should_check_convergence, validate_monotonicity,
};
use crate::solver::stok_construction::construct_stok;
use crate::utils::validate_probability_tensor;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for feasibility iteration.
#[derive(Debug, Clone)]
pub struct FeasibilityIterationConfig {
    /// Convergence detection settings.
    pub convergence: ConvergenceConfig,
    
    /// Maximum time horizon for STOK construction.
    pub max_time: usize,
    
    /// Whether to compute full STOK (η⁺, η⁻) after κ converges.
    /// If false, returns STOKKernel with only κ and π populated.
    pub compute_full_stok: bool,
    
    /// Whether to validate intermediate κ values during iteration.
    /// Useful for debugging but adds overhead.
    pub validate_intermediate: bool,
    
    /// Whether to check monotonicity invariant during iteration.
    /// Violations indicate bugs.
    pub check_monotonicity: bool,
    
    /// Whether to record κ history for analysis.
    pub record_history: bool,
}

impl Default for FeasibilityIterationConfig {
    fn default() -> Self {
        Self {
            convergence: ConvergenceConfig::default(),
            max_time: 20,
            compute_full_stok: true,
            validate_intermediate: false,
            check_monotonicity: false,
            record_history: false,
        }
    }
}

impl FeasibilityIterationConfig {
    /// Create config for quick testing (reduced iterations, higher tolerance).
    pub fn fast() -> Self {
        Self {
            convergence: ConvergenceConfig::fast(),
            max_time: 10,
            compute_full_stok: true,
            validate_intermediate: false,
            check_monotonicity: false,
            record_history: false,
        }
    }
    
    /// Create config for debugging (validation enabled, history recorded).
    pub fn debug() -> Self {
        Self {
            convergence: ConvergenceConfig::strict(),
            max_time: 30,
            compute_full_stok: true,
            validate_intermediate: true,
            check_monotonicity: true,
            record_history: true,
        }
    }
    
    /// Set max time horizon.
    pub fn with_max_time(mut self, max_time: usize) -> Self {
        self.max_time = max_time;
        self
    }
    
    /// Disable full STOK computation (only compute κ and π).
    pub fn without_stok(mut self) -> Self {
        self.compute_full_stok = false;
        self
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Timing information for feasibility iteration.
#[derive(Debug, Clone)]
pub struct IterationTiming {
    /// Total time including STOK construction.
    pub total_ms: f64,
    
    /// Time spent in Bellman backup iterations.
    pub iteration_ms: f64,
    
    /// Average time per iteration.
    pub per_iteration_ms: f64,
    
    /// Time spent constructing STOK (if computed).
    pub stok_construction_ms: f64,
}

/// Complete result from feasibility iteration.
#[derive(Debug)]
pub struct FeasibilityIterationResult<B: Backend> {
    /// The computed STOK kernel.
    pub kernel: STOKKernel<B>,
    
    /// Final convergence state.
    pub convergence: ConvergenceState,
    
    /// Optional: κ values at each iteration for analysis.
    pub kappa_history: Option<Vec<Vec<f32>>>,
    
    /// Timing breakdown.
    pub timing: IterationTiming,
}

impl<B: Backend> FeasibilityIterationResult<B> {
    /// Check if iteration converged successfully.
    pub fn converged(&self) -> bool {
        self.convergence.reason == ConvergenceReason::ToleranceReached
    }
    
    /// Get number of iterations performed.
    pub fn iterations(&self) -> usize {
        self.convergence.iteration
    }
    
    /// Get final convergence delta.
    pub fn final_delta(&self) -> f32 {
        self.convergence.delta
    }
}

// ============================================================================
// Main Algorithm
// ============================================================================

/// Run feasibility iteration to compute κ* and optionally construct STOK.
///
/// This is the main entry point for solving a Task MDP.
///
/// # Arguments
/// * `mdp` - Task MDP to solve
/// * `config` - Configuration options
///
/// # Returns
/// Complete result including STOK, convergence info, and timing
///
/// # Example
/// ```ignore
/// let mdp = TaskMDP::simple_chain(10, 20, &device);
/// let config = FeasibilityIterationConfig::default();
/// let result = feasibility_iteration(&mdp, config)?;
/// 
/// println!("Converged in {} iterations", result.iterations());
/// let kappa = result.kernel.kappa;
/// ```
pub fn feasibility_iteration<B: Backend>(
    mdp: &TaskMDP<B>,
    config: FeasibilityIterationConfig,
) -> Result<FeasibilityIterationResult<B>, StokError> {
    let start_time = Instant::now();
    let device = mdp.device();
    let s = mdp.n_states();
    
    // Initialize
    let mut kappa: Tensor<B, 1> = Tensor::zeros([s], &device);
    let mut policy: Tensor<B, 1, Int> = Tensor::zeros([s], &device);
    
    let mut convergence_state = if config.record_history {
        ConvergenceState::with_history()
    } else {
        ConvergenceState::new()
    };
    
    let mut kappa_history: Option<Vec<Vec<f32>>> = if config.record_history {
        Some(Vec::with_capacity(config.convergence.max_iterations))
    } else {
        None
    };
    
    let iteration_start = Instant::now();
    
    // Main iteration loop
    loop {
        let kappa_old = kappa.clone();
        
        // Standard Bellman backup (no tie-break during iteration)
        let (kappa_new, policy_new) = bellman_backup_kappa(&kappa, mdp);
        
        kappa = kappa_new;
        policy = policy_new;
        
        // Record history
        if let Some(ref mut history) = kappa_history {
            let kappa_vec: Vec<f32> = kappa.clone().into_data().to_vec().unwrap();
            history.push(kappa_vec);
        }
        
        // Check convergence
        if should_check_convergence(convergence_state.iteration, &config.convergence) {
            let delta = check_kappa_convergence(&kappa_old, &kappa);
            convergence_state.update(delta, &config.convergence);
            
            if convergence_state.should_terminate(&config.convergence) {
                break;
            }
        } else {
            convergence_state.increment();
        }
    }
    
    // ========================================================================
    // Stage B (π-OKBE): refine π among κ*-optimal actions to minimize time/suspension.
    // This avoids κ-tie policies that create non-absorbing chains (normalization failures).
    // ========================================================================
    let policy = extract_pi_time_minimizing(&kappa, mdp, 1e-6, 1e-6, 50, 2048);
    
    let iteration_time = iteration_start.elapsed();
    
    // Construct STOK
    let stok_start = Instant::now();
    
    let kernel = if config.compute_full_stok {
        construct_stok(&kappa, &policy, mdp, config.max_time)?
    } else {
        let dims = STOKDimensions::new(s, config.max_time);
        STOKKernel::from_kappa_policy(kappa, policy, dims, &device)
    };
    
    let stok_time = stok_start.elapsed();
    let total_time = start_time.elapsed();
    let iterations = convergence_state.iteration.max(1) as f64;
    
    Ok(FeasibilityIterationResult {
        kernel,
        convergence: convergence_state,
        kappa_history,
        timing: IterationTiming {
            total_ms: total_time.as_secs_f64() * 1000.0,
            iteration_ms: iteration_time.as_secs_f64() * 1000.0,
            per_iteration_ms: iteration_time.as_secs_f64() * 1000.0 / iterations,
            stok_construction_ms: stok_time.as_secs_f64() * 1000.0,
        },
    })
}

/// Convenience wrapper: solve MDP with default configuration.
///
/// This is the simplest API for basic usage.
///
/// # Example
/// ```ignore
/// let mdp = TaskMDP::simple_chain(10, 20, &device);
/// let stok = solve_task_mdp(&mdp)?;
/// ```
pub fn solve_task_mdp<B: Backend>(
    mdp: &TaskMDP<B>,
) -> Result<STOKKernel<B>, StokError> {
    let result = feasibility_iteration(mdp, FeasibilityIterationConfig::default())?;
    Ok(result.kernel)
}

/// Compute only κ* without STOK construction.
///
/// Faster than full solve when η distributions aren't needed.
pub fn compute_kappa<B: Backend>(
    mdp: &TaskMDP<B>,
) -> Result<(Tensor<B, 1>, Tensor<B, 1, Int>), StokError> {
    let config = FeasibilityIterationConfig::default().without_stok();
    let result = feasibility_iteration(mdp, config)?;
    Ok((result.kernel.kappa, result.kernel.policy))
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Validate κ values are in valid probability range [0, 1].
fn validate_kappa_bounds<B: Backend>(kappa: &Tensor<B, 1>) -> Result<(), StokError> {
    if !validate_probability_tensor(kappa, 1e-5) {
        let max_val: f32 = kappa.clone().max().into_scalar().elem();
        let min_val: f32 = kappa.clone().min().into_scalar().elem();
        return Err(StokError::InvalidProbability {
            value: if min_val < 0.0 { min_val } else { max_val },
            context: "kappa bounds violation".into(),
        });
    }
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================



// ============================================================================
// π-OKBE policy refinement (time / suspension minimization among κ*-optimal actions)
// ============================================================================

/// Refine π among κ*-optimal actions by minimizing expected continuation time.
///
/// This is the practical MVP version of the paper’s “minimize expected time” step
/// in feasibility iteration (κ-OKBE then π-OKBE).
///
/// We do a lightweight policy-iteration loop:
/// 1) Fix π, solve ν from: ν(x) = 1 + f₂(x,π(x)) · E[ν(x′)]
/// 2) Improve π(x) = argmin_{a ∈ A*_x} f₂(x,a) · E[ν(x′)]
fn extract_pi_time_minimizing<B: Backend>(
    kappa_star: &Tensor<B, 1>,
    mdp: &TaskMDP<B>,
    tie_tolerance: f32,
    nu_epsilon: f32,
    max_policy_iters: usize,
    max_nu_iters: usize,
) -> Tensor<B, 1, Int> {
    let device = mdp.device();
    let s = mdp.n_states();
    let a = mdp.n_actions();

    // Qκ(x,a) = f1 + f2 · E[κ*]
    let q_values = compute_q_values(kappa_star, mdp);

    // Start from a deterministic κ-greedy policy
    let (_, init_policy) = q_values.clone().max_dim_with_indices(1);
    let mut policy: Tensor<B, 1, Int> = init_policy.squeeze::<1>();

    for _ in 0..max_policy_iters {
        let nu = solve_nu_for_policy(mdp, &policy, nu_epsilon, max_nu_iters);

        // expected_nu(x,a) = E_{x'~P}[nu(x')]
        let p_flat = mdp.transition.clone().reshape([s * a, s]);
        let nu_col = nu.clone().reshape([s, 1]);
        let expected_nu = p_flat.matmul(nu_col).reshape([s, a]);

        // time_proxy(x,a) = f2(x,a) · E[nu(x')]
        let time_proxy = mdp.f2.clone() * expected_nu;

        let policy_new = argmin_time_among_kappa_optimal(
            &q_values,
            kappa_star,
            &time_proxy,
            tie_tolerance,
            &device,
        );

        // Stop if stable
        let old_vec: Vec<i32> = policy.clone().into_data().to_vec().unwrap();
        let new_vec: Vec<i32> = policy_new.clone().into_data().to_vec().unwrap();
        if old_vec == new_vec {
            return policy_new;
        }

        policy = policy_new;
    }

    policy
}

/// Solve ν for a fixed policy π:
/// ν(x) = 1 + f₂(x,π(x)) · E_{x'~P_π}[ν(x′)]
fn solve_nu_for_policy<B: Backend>(
    mdp: &TaskMDP<B>,
    policy: &Tensor<B, 1, Int>,
    epsilon: f32,
    max_iters: usize,
) -> Tensor<B, 1> {
    let device = mdp.device();
    let s = mdp.n_states();

    let p_pi = get_policy_transition(&mdp.transition, policy); // [S,S]
    let f2_pi = gather_by_policy(&mdp.f2, policy);            // [S]
    let ones: Tensor<B, 1> = Tensor::ones([s], &device);

    let mut nu: Tensor<B, 1> = Tensor::zeros([s], &device);

    for _ in 0..max_iters {
        let nu_col = nu.clone().reshape([s, 1]);
        let expected = p_pi.clone().matmul(nu_col).squeeze::<1>(); // [S]
        let nu_new = ones.clone() + f2_pi.clone() * expected;

        let diff = (nu_new.clone() - nu.clone()).abs();
        let delta: f32 = diff.max().into_scalar().elem();

        nu = nu_new;
        if delta < epsilon {
            break;
        }
    }

    nu
}

/// Select π(x) = argmin time_proxy(x,a) over κ*-optimal actions A*_x.
fn argmin_time_among_kappa_optimal<B: Backend>(
    q_values: &Tensor<B, 2>,
    kappa_star: &Tensor<B, 1>,
    time_proxy: &Tensor<B, 2>,
    tie_tolerance: f32,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let s = q_values.dims()[0];
    let a = q_values.dims()[1];

    // A*_x mask: Qκ(x,a) >= κ*(x) - eps
    let kappa_expanded = kappa_star.clone().unsqueeze_dim::<2>(1).expand([s, a]);
    let threshold = kappa_expanded - tie_tolerance;
    let optimal_mask = q_values.clone().greater_equal(threshold);
    let optimal_mask_f: Tensor<B, 2> = optimal_mask.float();

    // Non-optimal actions get huge time so they are never selected by min
    let large_value: f32 = 1e10;
    let large_tensor: Tensor<B, 2> = Tensor::full([s, a], large_value, device);
    let ones: Tensor<B, 2> = Tensor::ones([s, a], device);

    let masked_time = time_proxy.clone() * optimal_mask_f.clone()
        + large_tensor * (ones - optimal_mask_f);

    let (_, policy) = masked_time.min_dim_with_indices(1);
    policy.squeeze::<1>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DefaultBackend, default_device};
    use crate::mdp::TaskMDP;
    use crate::utils::approx_eq;

    #[test]
    fn test_feasibility_iteration_trivial() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 10, &device);
        
        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        // All states should be feasible
        let kappa_data: Vec<f32> = result.kernel.kappa.into_data().to_vec().unwrap();
        
        for (i, &k) in kappa_data.iter().enumerate() {
            assert!(approx_eq(k, 1.0, 1e-4), 
                "State {} should have κ=1.0, got {}", i, k);
        }
    }

    #[test]
    fn test_feasibility_iteration_converges() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
        
        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        assert!(result.converged(), 
            "Should converge, got reason: {}", result.convergence.reason);
        assert!(result.iterations() > 0, "Should have at least 1 iteration");
        assert!(result.iterations() <= 100, 
            "Should converge in reasonable iterations, got {}", result.iterations());
    }

    #[test]
    fn test_feasibility_iteration_constrained() {
        let device = default_device();
        // Chain with fire at state 5
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(10, 5, 20, &device);
        
        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        let kappa_data: Vec<f32> = result.kernel.kappa.into_data().to_vec().unwrap();
        
        // States 0-5 should be infeasible (blocked by fire)
        for i in 0..=5 {
            assert!(kappa_data[i] < 0.01, 
                "State {} should be infeasible, got κ={}", i, kappa_data[i]);
        }
        
        // States 6-9 should be feasible
        for i in 6..10 {
            assert!(approx_eq(kappa_data[i], 1.0, 1e-4),
                "State {} should be feasible, got κ={}", i, kappa_data[i]);
        }
    }

    #[test]
    fn test_feasibility_iteration_stochastic() {
        let device = default_device();
        // 80% success probability
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(5, 0.8, 15, &device);
        
        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        let kappa_data: Vec<f32> = result.kernel.kappa.into_data().to_vec().unwrap();
        
        // All states should be feasible with probability < 1.0 (stochastic)
        // But goal state should have κ = 1.0
        assert!(approx_eq(kappa_data[4], 1.0, 1e-4), 
            "Goal state should have κ=1.0, got {}", kappa_data[4]);
        
        // Non-goal states should have 0 < κ < 1
        for i in 0..4 {
            assert!(kappa_data[i] > 0.5, 
                "State {} should have κ > 0.5, got {}", i, kappa_data[i]);
            assert!(kappa_data[i] <= 1.0,
                "State {} should have κ ≤ 1.0, got {}", i, kappa_data[i]);
        }
    }

    #[test]
    fn test_solve_task_mdp() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        let stok = solve_task_mdp(&mdp).unwrap();
        
        assert_eq!(stok.eta_plus.dims(), [5, 5, 20]);
        assert_eq!(stok.kappa.dims(), [5]);
    }

    #[test]
    fn test_compute_kappa_only() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        let (kappa, policy) = compute_kappa(&mdp).unwrap();
        
        assert_eq!(kappa.dims(), [5]);
        assert_eq!(policy.dims(), [5]);
        
        // Verify κ values
        let kappa_data: Vec<f32> = kappa.into_data().to_vec().unwrap();
        for &k in &kappa_data {
            assert!(approx_eq(k, 1.0, 1e-4));
        }
    }

    #[test]
    fn test_timing_information() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        assert!(result.timing.total_ms > 0.0, "Total time should be positive");
        assert!(result.timing.iteration_ms >= 0.0, "Iteration time should be non-negative");
        assert!(result.timing.per_iteration_ms >= 0.0, "Per-iteration time should be non-negative");
    }

    #[test]
    fn test_history_recording() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 5, &device);
        
        let config = FeasibilityIterationConfig {
            record_history: true,
            ..Default::default()
        };
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        assert!(result.kappa_history.is_some(), "History should be recorded");
        let history = result.kappa_history.unwrap();
        assert!(!history.is_empty(), "History should have entries");
        
        // Each entry should have 3 values (one per state)
        for entry in &history {
            assert_eq!(entry.len(), 3);
        }
    }

    #[test]
    fn test_stok_normalization() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 15, &device);
        
        let result = feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
        
        // ========== DIAGNOSTIC: Check policy and termination ==========
        
        // 1. Print final policy
        let policy_data: Vec<i32> = result.kernel.policy.clone().into_data().to_vec().unwrap();
        eprintln!("Final policy: {:?}", policy_data);
        
        // 2. Print κ values
        let kappa_data: Vec<f32> = result.kernel.kappa.clone().into_data().to_vec().unwrap();
        eprintln!("Final κ: {:?}", kappa_data);
        
        // 3. Check if policy leads to goal (simple reachability check)
        // For simple_chain: action 0 = stay/left, action 1 = right (toward goal)
        let mut can_reach_goal = vec![false; 5];
        can_reach_goal[4] = true; // Goal state
        for _ in 0..5 {
            for s in 0..5 {
                let action = policy_data[s] as usize;
                // In simple_chain, action 1 moves right
                if action == 1 && s < 4 {
                    can_reach_goal[s] = can_reach_goal[s + 1];
                }
            }
        }
        eprintln!("Can reach goal under policy: {:?}", can_reach_goal);
        
        // 4. Check row-mass of η⁺ + η⁻
        let combined = result.kernel.combined_stok();
        let row_mass: Tensor<DefaultBackend, 1> = combined
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>();
        let row_mass_data: Vec<f32> = row_mass.into_data().to_vec().unwrap();
        eprintln!("Row mass (should be 1.0): {:?}", row_mass_data);
        
        // 5. Identify the issue
        for (i, &mass) in row_mass_data.iter().enumerate() {
            if (mass - 1.0).abs() > 1e-4 {
                let tail = 1.0 - mass;
                eprintln!(
                    "State {}: mass={:.6}, tail={:.6} ({})",
                    i, mass, tail,
                    if tail > 0.0 { "non-termination or truncation" } else { "overcounting" }
                );
            }
        }
        
        // ========== END DIAGNOSTIC ==========
        
        // Original assertion
        assert!(result.kernel.validate(1e-4).is_ok(), 
            "Validation failed: {:?}", result.kernel.validate(1e-4));
    }

    #[test]
    fn test_max_iterations_termination() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 5, &device);
        
        let config = FeasibilityIterationConfig {
            convergence: ConvergenceConfig::default()
                .with_max_iterations(2)
                .with_epsilon(1e-20), // Very tight, won't converge
            ..Default::default()
        };
        
        let result = feasibility_iteration(&mdp, config).unwrap();
        
        assert_eq!(result.convergence.reason, ConvergenceReason::MaxIterationsReached);
        assert_eq!(result.iterations(), 2);
    }
}