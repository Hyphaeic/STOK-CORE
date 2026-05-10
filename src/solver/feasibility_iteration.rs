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
use crate::solver::bellman::{
    bellman_backup_kappa, compute_q_values, gather_by_policy, get_policy_transition,
};
use crate::solver::convergence::{
    check_kappa_convergence, should_check_convergence, ConvergenceConfig,
    ConvergenceReason, ConvergenceState,
};
use crate::solver::stok_construction::construct_stok;
use crate::stok::STOKKernel;
use crate::types::STOKDimensions;
use crate::types::StokError;
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
    let eps = config.convergence.epsilon.max(1e-6);
    let policy = extract_pi_time_minimizing(&kappa, mdp, eps, eps, 50, 2048);

    // Fail-fast: STOK normalization requires absorbing/transient nonterminal dynamics.
    // Two-tier check: fast screen (always), SCC fallback (when risky).
    check_policy_absorption_release(mdp, &policy, eps)?;

    let iteration_time = iteration_start.elapsed();

    // Construct STOK
    let stok_start = Instant::now();

    let kernel = if config.compute_full_stok {
        construct_stok(&kappa, &policy, mdp, config.max_time, eps)?
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
pub fn solve_task_mdp<B: Backend>(mdp: &TaskMDP<B>) -> Result<STOKKernel<B>, StokError> {
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

fn check_policy_absorption<B: Backend>(
    mdp: &TaskMDP<B>,
    policy: &Tensor<B, 1, Int>,
    prob_epsilon: f32,
    termination_epsilon: f32,
) -> Result<(), StokError> {
    let s = mdp.n_states();

    // Policy-conditioned transition and continuation probability.
    let p_pi = get_policy_transition(&mdp.transition, policy); // [S,S]
    let f2_pi = gather_by_policy(&mdp.f2, policy); // [S]

    let p: Vec<f32> = p_pi.into_data().to_vec().unwrap();
    let f2: Vec<f32> = f2_pi.into_data().to_vec().unwrap();

    // Build adjacency on support of P_π (termination handled separately via f₂).
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); s];
    for i in 0..s {
        let row = &p[i * s..(i + 1) * s];
        for (j, &prob) in row.iter().enumerate() {
            if prob > prob_epsilon {
                adj[i].push(j);
            }
        }
    }

    // Tarjan SCC to find closed recurrent classes under P_π support.
    let mut index: i32 = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack: Vec<bool> = vec![false; s];
    let mut indices: Vec<i32> = vec![-1; s];
    let mut lowlink: Vec<i32> = vec![0; s];
    let mut sccs: Vec<Vec<usize>> = Vec::new();

    fn strongconnect(
        v: usize,
        index: &mut i32,
        stack: &mut Vec<usize>,
        on_stack: &mut Vec<bool>,
        indices: &mut Vec<i32>,
        lowlink: &mut Vec<i32>,
        adj: &Vec<Vec<usize>>,
        sccs: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = *index;
        lowlink[v] = *index;
        *index += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w] == -1 {
                strongconnect(w, index, stack, on_stack, indices, lowlink, adj, sccs);
                lowlink[v] = lowlink[v].min(lowlink[w]);
            } else if on_stack[w] {
                lowlink[v] = lowlink[v].min(indices[w]);
            }
        }

        if lowlink[v] == indices[v] {
            let mut comp: Vec<usize> = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                comp.push(w);
                if w == v {
                    break;
                }
            }
            sccs.push(comp);
        }
    }

    for v in 0..s {
        if indices[v] == -1 {
            strongconnect(
                v,
                &mut index,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlink,
                &adj,
                &mut sccs,
            );
        }
    }

    // Bad class for STOK normalization:
    // - closed under P_π support (no outgoing edges), AND
    // - effectively never terminates in the class (f₂ ≈ 1 everywhere).
    for comp in sccs {
        let mut in_comp = vec![false; s];
        for &v in &comp {
            in_comp[v] = true;
        }

        let mut has_outgoing = false;
        for &v in &comp {
            for &w in &adj[v] {
                if !in_comp[w] {
                    has_outgoing = true;
                    break;
                }
            }
            if has_outgoing {
                break;
            }
        }

        if !has_outgoing {
            let all_never_terminate = comp.iter().all(|&v| f2[v] >= 1.0 - termination_epsilon);

            if all_never_terminate {
                return Err(StokError::InvalidProbability {
                        value: 1.0,
                        context: format!(
                            "Non-absorbing policy-induced closed class in nonterminal dynamics (|C|={}): states {:?}. \
    This violates the absorbing/transience assumption required for STOK normalization.",
                            comp.len(),
                            comp
                        ),
                    });
            }
        }
    }

    Ok(())
}

/// Release-safe absorption/transience guard with two-tier checking.
///
/// Tier 1 (always run): Cheap screen based on min termination probability.
/// Tier 2 (risky only): Full SCC analysis to detect closed recurrent classes.
///
/// # Arguments
/// * `prob_epsilon` - Threshold for non-zero transition (default: 1e-8)
/// * `termination_delta` - Threshold for "effectively terminates" (default: 1e-6)
fn check_policy_absorption_release<B: Backend>(
    mdp: &TaskMDP<B>,
    policy: &Tensor<B, 1, Int>,
    epsilon: f32,
) -> Result<(), StokError> {
    let prob_eps = 1e-8_f32.max(epsilon * 0.01);
    let term_delta = 1e-6_f32.max(epsilon);

    // Tier 1: Fast screen
    let risk = absorption_screen_fast(mdp, policy, term_delta);

    match risk {
        AbsorptionRisk::Safe => Ok(()),
        AbsorptionRisk::Risky => {
            // Tier 2: Full SCC analysis
            check_policy_absorption(mdp, policy, prob_eps, term_delta)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AbsorptionRisk {
    Safe,  // All states have sufficient termination probability
    Risky, // Requires SCC analysis
}

/// Tier 1: Cheap absorption screen.
///
/// Computes min termination probability across all states.
/// If min(1 - f2_π) >= delta, the process is guaranteed absorbing.
fn absorption_screen_fast<B: Backend>(
    mdp: &TaskMDP<B>,
    policy: &Tensor<B, 1, Int>,
    termination_delta: f32,
) -> AbsorptionRisk {
    let f2_pi = gather_by_policy(&mdp.f2, policy);
    let device = f2_pi.device();
    let s = mdp.n_states();

    // Termination probability: 1 - f2_π(x)
    let ones: Tensor<B, 1> = Tensor::ones([s], &device);
    let term_prob = ones - f2_pi;

    // Min termination probability
    let min_term: f32 = term_prob.min().into_scalar().elem();

    if min_term >= termination_delta {
        AbsorptionRisk::Safe
    } else {
        AbsorptionRisk::Risky
    }
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
/// Implements Eq [8]: π**(x) = argmin_{a ∈ A*_x} f_2(x, a) · E[(t_f + 1) · η+].
/// We approximate `E[(t_f + 1) · η+]` with a ν-iteration that solves
/// `ν(x) = 1 + f_2(x, π(x)) · E[ν(x')]` (the expected time to absorption under
/// the current policy chain). When the chain is absorbing, ν matches the paper
/// quantity. When it isn't, ν grows unboundedly — see the limitation below.
///
/// **PP-106 partial fix:** the argmin tie-break now uses f_2 as the primary key
/// (`f_2 = 0` actions are immediate-termination per Eq [8] and dominate any
/// further proxy). The ν proxy is the secondary key. This handles cases where
/// some κ-optimal actions have `f_2 = 0` (immediate goal/constraint) and others
/// don't.
///
/// **Known limitation (full PP-106 still open):** when *every* κ-optimal action
/// at a state has the same f_2 AND the current policy creates a non-absorbing
/// chain (so ν is uniform-large), neither key distinguishes the actions. The
/// argmin then picks lowest index. A complete fix needs a global heuristic
/// (e.g. shortest-path distance to a state where some action has f_1 > 0 or
/// f_c < 1) that can break ties by structural progress rather than by ν or f_2
/// alone. Current callers that hit this corner are typically degenerate (either
/// the MDP truly has no path to termination, or every initial policy is
/// recurrent) and surface as `check_policy_absorption_release` errors.
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
            &mdp.f2,
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
    let f2_pi = gather_by_policy(&mdp.f2, policy); // [S]
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

/// Select π(x) over κ*-optimal actions A*_x using a lexicographic key:
///   (1) primary: f_2(x, a) — actions with f_2 = 0 immediately terminate per
///       Eq [8] (the product f_2 · E[(t_f+1)·η+] vanishes regardless of η+);
///   (2) secondary: time_proxy(x, a) = f_2(x, a) · E[ν(x')], the ν-iteration
///       proxy for E[(t_f+1)·η+] which is meaningful when the policy chain
///       is absorbing.
/// Combined into a single scalar `f_2 · BIG + time_proxy` so a single argmin
/// over the κ-optimal mask yields the lexicographic winner. Non-optimal
/// actions are masked to a huge sentinel value.
fn argmin_time_among_kappa_optimal<B: Backend>(
    q_values: &Tensor<B, 2>,
    kappa_star: &Tensor<B, 1>,
    time_proxy: &Tensor<B, 2>,
    f2: &Tensor<B, 2>,
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

    // Lexicographic combined key: f_2 dominates time_proxy. f_2 ∈ [0, 1], time_proxy
    // is bounded above by ν_max which is O(max_nu_iters); BIG ≫ that range.
    const BIG: f32 = 1.0e6;
    let combined_key = f2.clone() * BIG + time_proxy.clone();

    let large_value: f32 = 1.0e10;
    let large_tensor: Tensor<B, 2> = Tensor::full([s, a], large_value, device);
    let ones: Tensor<B, 2> = Tensor::ones([s, a], device);

    let masked_key =
        combined_key * optimal_mask_f.clone() + large_tensor * (ones - optimal_mask_f);

    let (_, policy) = masked_key.min_dim_with_indices(1);
    policy.squeeze::<1>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
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
            assert!(
                approx_eq(k, 1.0, 1e-4),
                "State {} should have κ=1.0, got {}",
                i,
                k
            );
        }
    }

    #[test]
    fn test_feasibility_iteration_converges() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);

        let config = FeasibilityIterationConfig::default();
        let result = feasibility_iteration(&mdp, config).unwrap();

        assert!(
            result.converged(),
            "Should converge, got reason: {}",
            result.convergence.reason
        );
        assert!(result.iterations() > 0, "Should have at least 1 iteration");
        assert!(
            result.iterations() <= 100,
            "Should converge in reasonable iterations, got {}",
            result.iterations()
        );
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
            assert!(
                kappa_data[i] < 0.01,
                "State {} should be infeasible, got κ={}",
                i,
                kappa_data[i]
            );
        }

        // States 6-9 should be feasible
        for i in 6..10 {
            assert!(
                approx_eq(kappa_data[i], 1.0, 1e-4),
                "State {} should be feasible, got κ={}",
                i,
                kappa_data[i]
            );
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
        assert!(
            approx_eq(kappa_data[4], 1.0, 1e-4),
            "Goal state should have κ=1.0, got {}",
            kappa_data[4]
        );

        // Non-goal states should have 0 < κ < 1
        for i in 0..4 {
            assert!(
                kappa_data[i] > 0.5,
                "State {} should have κ > 0.5, got {}",
                i,
                kappa_data[i]
            );
            assert!(
                kappa_data[i] <= 1.0,
                "State {} should have κ ≤ 1.0, got {}",
                i,
                kappa_data[i]
            );
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

        assert!(
            result.timing.total_ms > 0.0,
            "Total time should be positive"
        );
        assert!(
            result.timing.iteration_ms >= 0.0,
            "Iteration time should be non-negative"
        );
        assert!(
            result.timing.per_iteration_ms >= 0.0,
            "Per-iteration time should be non-negative"
        );
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
        let row_mass: Tensor<DefaultBackend, 1> =
            combined.sum_dim(2).squeeze::<2>().sum_dim(1).squeeze::<1>();
        let row_mass_data: Vec<f32> = row_mass.into_data().to_vec().unwrap();
        eprintln!("Row mass (should be 1.0): {:?}", row_mass_data);

        // 5. Identify the issue
        for (i, &mass) in row_mass_data.iter().enumerate() {
            if (mass - 1.0).abs() > 1e-4 {
                let tail = 1.0 - mass;
                eprintln!(
                    "State {}: mass={:.6}, tail={:.6} ({})",
                    i,
                    mass,
                    tail,
                    if tail > 0.0 {
                        "non-termination or truncation"
                    } else {
                        "overcounting"
                    }
                );
            }
        }

        // ========== END DIAGNOSTIC ==========

        // Original assertion
        assert!(
            result.kernel.validate(1e-4).is_ok(),
            "Validation failed: {:?}",
            result.kernel.validate(1e-4)
        );
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

        assert_eq!(
            result.convergence.reason,
            ConvergenceReason::MaxIterationsReached
        );
        assert_eq!(result.iterations(), 2);
    }

    // ========================================================================
    // PP-103 / CHK-ALG1 stage 2: π-OKBE time-minimization on κ-tie cases
    // ========================================================================

    /// Build a 4-state TaskMDP where state 0 has two actions that BOTH achieve
    /// κ = 1 but with different expected times to the goal. Action ordering
    /// is deliberately reversed (action 0 = slow path, action 1 = fast path)
    /// so a naive lowest-index or f_1-tiebreak rule would pick the WRONG
    /// action — only paper-faithful Eq [8] time-minimization picks action 1.
    ///
    /// State graph (deterministic, 2 actions each):
    ///   0 --a0--> 2 --*--> 1 --*--> 3 (goal)         path length 3
    ///   0 --a1--> 1 --*--> 3 (goal)                  path length 2
    fn make_kappa_tie_mdp(
        device: &<DefaultBackend as Backend>::Device,
    ) -> TaskMDP<DefaultBackend> {
        let n = 4;
        let a = 2;
        let mut p = vec![0.0f32; n * a * n];
        let idx = |s: usize, act: usize, sp: usize| s * a * n + act * n + sp;

        // State 0: action 0 → 2 (slow), action 1 → 1 (fast)
        p[idx(0, 0, 2)] = 1.0;
        p[idx(0, 1, 1)] = 1.0;
        // State 1: both actions → 3 (goal)
        p[idx(1, 0, 3)] = 1.0;
        p[idx(1, 1, 3)] = 1.0;
        // State 2: both actions → 1
        p[idx(2, 0, 1)] = 1.0;
        p[idx(2, 1, 1)] = 1.0;
        // State 3: absorbing under both actions
        p[idx(3, 0, 3)] = 1.0;
        p[idx(3, 1, 3)] = 1.0;

        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(p.as_slice(), device).reshape([n, a, n]);

        // Goal at state 3 only.
        let mut fg = vec![0.0f32; n * a];
        fg[3 * a + 0] = 1.0;
        fg[3 * a + 1] = 1.0;
        let goal: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(fg.as_slice(), device).reshape([n, a]);

        // No constraints.
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n, a], device);

        TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).expect("κ-tie MDP")
    }

    /// PP-103: at the κ-tie state 0, π** must select action 1 (the fast path).
    /// A f_1-max or lowest-index tiebreak would pick action 0 — that would be
    /// non-paper-faithful per Eq [8].
    #[test]
    fn test_pi_okbe_picks_time_minimizing_action_at_kappa_tie() {
        let device = default_device();
        let mdp = make_kappa_tie_mdp(&device);

        let result =
            feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();

        let kappa: Vec<f32> = result.kernel.kappa.clone().into_data().to_vec().unwrap();
        let policy: Vec<i32> = result.kernel.policy.clone().into_data().to_vec().unwrap();

        // Sanity: state 0 is a κ-tie (both actions reach the goal with κ=1).
        assert!(
            approx_eq(kappa[0], 1.0, 1e-5),
            "Setup invalid: state 0 should be feasible (κ=1), got {}",
            kappa[0]
        );

        // The actual π-OKBE acceptance: π(0) = 1, the time-minimizing action.
        assert_eq!(
            policy[0], 1,
            "π(0) should be 1 (fast path), got {} — Eq [8] time-min violated",
            policy[0]
        );
    }

    /// PP-106 partial fix: when one κ-optimal action has `f_2 = 0` (immediate
    /// termination per Eq [8]) and another has `f_2 > 0`, the f_2 primary key
    /// must select the immediate-termination action regardless of which has
    /// the lower lexical index.
    ///
    /// Witness: 3-state MDP. State 0 has two actions:
    ///   - action 0 (lower index): f_g = 0, f_c = 1 → f_2 = 1, transitions to
    ///     state 1 from which a multi-step path eventually reaches the goal.
    ///   - action 1: f_g = 1, f_c = 1 → f_2 = 0, immediate goal-success at
    ///     state 0 itself.
    /// Both actions give κ = 1 → κ-tied. Per Eq [8] we want π(0) = 1 (immediate
    /// success). A naive lowest-index tiebreak would pick action 0.
    #[test]
    fn test_pi_okbe_f2_tiebreak_prefers_immediate_termination() {
        let device = default_device();

        // 3 states, 2 actions.
        let n = 3;
        let a = 2;
        let mut p = vec![0.0f32; n * a * n];
        let idx = |s: usize, act: usize, sp: usize| s * a * n + act * n + sp;
        // State 0: action 0 → state 1, action 1 → state 0 (self-loop, but
        // immediately terminates due to f_g = 1 so f_2 = 0).
        p[idx(0, 0, 1)] = 1.0;
        p[idx(0, 1, 0)] = 1.0;
        // State 1: both actions → state 2 (goal).
        p[idx(1, 0, 2)] = 1.0;
        p[idx(1, 1, 2)] = 1.0;
        // State 2: absorbing under both actions.
        p[idx(2, 0, 2)] = 1.0;
        p[idx(2, 1, 2)] = 1.0;
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(p.as_slice(), &device).reshape([n, a, n]);

        // f_g: state 0 action 1 has goal (f_g = 1), state 2 (any action) has goal.
        let mut fg = vec![0.0f32; n * a];
        fg[0 * a + 1] = 1.0;
        fg[2 * a + 0] = 1.0;
        fg[2 * a + 1] = 1.0;
        let goal: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(fg.as_slice(), &device).reshape([n, a]);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n, a], &device);
        let mdp =
            TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).expect("witness MDP");

        let result =
            feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();

        let policy: Vec<i32> = result.kernel.policy.into_data().to_vec().unwrap();

        // π(0) must be 1 (immediate termination via f_2 = 0), not 0 (longer path).
        assert_eq!(
            policy[0], 1,
            "PP-106: f_2 primary key must pick the immediate-termination action; got {}",
            policy[0]
        );
    }

    /// PP-103: confirm that the deprecated `bellman_backup_kappa_with_tiebreak`
    /// would have picked the wrong action on this κ-tie problem (action 0 by
    /// lowest-index fallthrough). This documents WHY the deprecated function
    /// is unsafe and pins the active path away from it.
    ///
    /// On `make_kappa_tie_mdp`, both actions at state 0 have f_1 = 0 (it is
    /// not a goal state), so the f_1-max tiebreak degenerates to argmax over
    /// a uniform vector — implementation falls back to lowest index = 0. We
    /// exercise this directly so any future re-introduction of f_1-tiebreak
    /// in the active path will be visible.
    #[test]
    #[allow(deprecated)]
    fn test_deprecated_tiebreak_picks_wrong_action_at_kappa_tie() {
        use crate::solver::bellman::bellman_backup_kappa_with_tiebreak;

        let device = default_device();
        let mdp = make_kappa_tie_mdp(&device);

        // First converge κ via the standard backup so we have κ* to feed in.
        let mut kappa: Tensor<DefaultBackend, 1> =
            Tensor::zeros([mdp.n_states()], &device);
        for _ in 0..50 {
            let (kappa_new, _) = bellman_backup_kappa(&kappa, &mdp);
            kappa = kappa_new;
        }

        let (_, deprecated_policy) =
            bellman_backup_kappa_with_tiebreak(&kappa, &mdp, 1e-6);
        let dp_vec: Vec<i32> = deprecated_policy.into_data().to_vec().unwrap();

        // Document that the deprecated tiebreak picks the slow path here.
        // The active path under feasibility_iteration() must NOT match it.
        assert_eq!(
            dp_vec[0], 0,
            "If this changes, re-evaluate whether the deprecated function is \
             actually doing what its name claims; currently it lowest-indexes \
             on f_1-ties and that gives action 0 here."
        );
    }
}
