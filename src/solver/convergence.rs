//! Convergence detection utilities for feasibility iteration.
//!
//! Provides configuration and state tracking for the iterative
//! optimization loop that computes κ*.
//!
//! # Convergence Criterion
//! The feasibility iteration converges when:
//! ```text
//! δ = ||κ^{d+1} - κ^{d}||_∞ < ε
//! ```
//!
//! # Theoretical Guarantee
//! Per Ringstrom & Schrater (2025) Appendix E, feasibility iteration
//! is a contraction mapping, guaranteeing convergence.

use crate::utils::{
    linf_distance, DEFAULT_CHECK_INTERVAL, DEFAULT_CONVERGENCE_TOLERANCE, DEFAULT_MAX_ITERATIONS,
};
use burn::prelude::*;

// ============================================================================
// Convergence Configuration
// ============================================================================

/// Configuration for feasibility iteration convergence.
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Maximum allowed iterations before forced termination.
    pub max_iterations: usize,

    /// L∞ norm threshold for convergence detection.
    /// Iteration stops when ||κ_new - κ_old||_∞ < epsilon.
    pub epsilon: f32,

    /// How often to check convergence (every N iterations).
    /// Set to 1 for accurate detection, higher for performance.
    pub check_interval: usize,

    /// Whether to stop immediately upon convergence.
    /// If false, continues until max_iterations.
    pub early_stop: bool,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            epsilon: DEFAULT_CONVERGENCE_TOLERANCE,
            check_interval: DEFAULT_CHECK_INTERVAL,
            early_stop: true,
        }
    }
}

impl ConvergenceConfig {
    /// Create a new config with specified max iterations.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Create a new config with specified epsilon.
    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Create a new config with specified check interval.
    pub fn with_check_interval(mut self, n: usize) -> Self {
        self.check_interval = n.max(1); // At least 1
        self
    }

    /// Create a new config with early stopping disabled.
    pub fn without_early_stop(mut self) -> Self {
        self.early_stop = false;
        self
    }

    /// Create a strict config for testing (low tolerance, check every iteration).
    pub fn strict() -> Self {
        Self {
            max_iterations: 10000,
            epsilon: 1e-8,
            check_interval: 1,
            early_stop: true,
        }
    }

    /// Create a fast config for benchmarking (higher tolerance, batched checks).
    pub fn fast() -> Self {
        Self {
            max_iterations: 500,
            epsilon: 1e-4,
            check_interval: 5,
            early_stop: true,
        }
    }
}

// ============================================================================
// Convergence Reason
// ============================================================================

/// Reason why iteration terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Iteration is still running.
    NotConverged,

    /// L∞ norm fell below epsilon threshold.
    ToleranceReached,

    /// Maximum iterations reached without convergence.
    MaxIterationsReached,
}

impl std::fmt::Display for ConvergenceReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConverged => write!(f, "not converged"),
            Self::ToleranceReached => write!(f, "tolerance reached"),
            Self::MaxIterationsReached => write!(f, "max iterations reached"),
        }
    }
}

// ============================================================================
// Convergence State
// ============================================================================

/// Tracks the state of convergence during iteration.
#[derive(Debug, Clone)]
pub struct ConvergenceState {
    /// Current iteration count.
    pub iteration: usize,

    /// Current L∞ distance between consecutive κ estimates.
    pub delta: f32,

    /// Whether convergence has been detected.
    pub converged: bool,

    /// Reason for termination (if terminated).
    pub reason: ConvergenceReason,

    /// Optional: history of delta values for analysis.
    pub delta_history: Vec<f32>,

    /// Whether to record delta history.
    record_history: bool,
}

impl ConvergenceState {
    /// Create a new convergence state.
    pub fn new() -> Self {
        Self {
            iteration: 0,
            delta: f32::INFINITY,
            converged: false,
            reason: ConvergenceReason::NotConverged,
            delta_history: Vec::new(),
            record_history: false,
        }
    }

    /// Create a new convergence state that records history.
    pub fn with_history() -> Self {
        Self {
            record_history: true,
            delta_history: Vec::with_capacity(100),
            ..Self::new()
        }
    }

    /// Update state after a convergence check.
    pub fn update(&mut self, new_delta: f32, config: &ConvergenceConfig) {
        self.iteration += 1;
        self.delta = new_delta;

        if self.record_history {
            self.delta_history.push(new_delta);
        }

        // Check convergence conditions
        if new_delta < config.epsilon {
            self.converged = true;
            self.reason = ConvergenceReason::ToleranceReached;
        } else if self.iteration >= config.max_iterations {
            self.reason = ConvergenceReason::MaxIterationsReached;
        }
    }

    /// Increment iteration counter without checking convergence.
    /// Used when batching convergence checks.
    pub fn increment(&mut self) {
        self.iteration += 1;
    }

    /// Check if iteration should terminate.
    pub fn should_terminate(&self, config: &ConvergenceConfig) -> bool {
        // Always terminate if we hit max iterations
        if self.iteration >= config.max_iterations {
            return true;
        }

        // Terminate on convergence if early stopping is enabled
        if self.converged && config.early_stop {
            return true;
        }

        false
    }

    /// Check if the iteration converged successfully.
    pub fn is_converged(&self) -> bool {
        self.reason == ConvergenceReason::ToleranceReached
    }

    /// Get convergence rate (ratio of consecutive deltas).
    /// Returns None if not enough history.
    pub fn convergence_rate(&self) -> Option<f32> {
        if self.delta_history.len() < 2 {
            return None;
        }
        let n = self.delta_history.len();
        let prev = self.delta_history[n - 2];
        let curr = self.delta_history[n - 1];

        if prev.abs() < 1e-10 {
            None
        } else {
            Some(curr / prev)
        }
    }
}

impl Default for ConvergenceState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convergence Check Functions
// ============================================================================

/// Check if we should perform a convergence check this iteration.
///
/// For performance, convergence checks can be batched every N iterations.
#[inline]
pub fn should_check_convergence(iteration: usize, config: &ConvergenceConfig) -> bool {
    iteration % config.check_interval == 0 || iteration == 0
}

/// Compute convergence metric between old and new κ.
///
/// # Returns
/// L∞ distance: max_i |κ_new[i] - κ_old[i]|
pub fn check_kappa_convergence<B: Backend>(old: &Tensor<B, 1>, new: &Tensor<B, 1>) -> f32 {
    linf_distance(old, new)
}

/// Validate that κ values are monotonically non-decreasing.
///
/// Per the paper, κ should never decrease during feasibility iteration.
/// A decrease indicates a bug.
///
/// # Returns
/// - `Ok(())` if monotonicity holds
/// - `Err(max_decrease)` if any value decreased, with the magnitude
pub fn validate_monotonicity<B: Backend>(
    old: &Tensor<B, 1>,
    new: &Tensor<B, 1>,
    tolerance: f32,
) -> Result<(), f32> {
    let diff = new.clone() - old.clone();
    let min_diff: f32 = diff.min().into_scalar().elem();

    if min_diff < -tolerance {
        Err(-min_diff)
    } else {
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};

    #[test]
    fn test_convergence_config_default() {
        let config = ConvergenceConfig::default();
        assert_eq!(config.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_eq!(config.epsilon, DEFAULT_CONVERGENCE_TOLERANCE);
        assert_eq!(config.check_interval, 1);
        assert!(config.early_stop);
    }

    #[test]
    fn test_convergence_config_builder() {
        let config = ConvergenceConfig::default()
            .with_max_iterations(500)
            .with_epsilon(1e-4)
            .with_check_interval(5);

        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.epsilon, 1e-4);
        assert_eq!(config.check_interval, 5);
    }

    #[test]
    fn test_convergence_state_new() {
        let state = ConvergenceState::new();
        assert_eq!(state.iteration, 0);
        assert_eq!(state.delta, f32::INFINITY);
        assert!(!state.converged);
        assert_eq!(state.reason, ConvergenceReason::NotConverged);
    }

    #[test]
    fn test_convergence_state_update_converged() {
        let mut state = ConvergenceState::new();
        let config = ConvergenceConfig::default().with_epsilon(1e-4);

        // Update with delta below epsilon
        state.update(1e-5, &config);

        assert_eq!(state.iteration, 1);
        assert!(state.converged);
        assert_eq!(state.reason, ConvergenceReason::ToleranceReached);
    }

    #[test]
    fn test_convergence_state_update_not_converged() {
        let mut state = ConvergenceState::new();
        let config = ConvergenceConfig::default().with_epsilon(1e-4);

        // Update with delta above epsilon
        state.update(1e-3, &config);

        assert_eq!(state.iteration, 1);
        assert!(!state.converged);
        assert_eq!(state.reason, ConvergenceReason::NotConverged);
    }

    #[test]
    fn test_convergence_state_max_iterations() {
        let mut state = ConvergenceState::new();
        let config = ConvergenceConfig::default()
            .with_max_iterations(5)
            .with_epsilon(1e-10); // Very tight, won't converge

        for i in 0..5 {
            state.update(0.1, &config);
            if i < 4 {
                assert!(
                    !state.should_terminate(&config),
                    "Should not terminate at iteration {}",
                    i
                );
            }
        }

        assert!(state.should_terminate(&config));
        assert_eq!(state.reason, ConvergenceReason::MaxIterationsReached);
    }

    #[test]
    fn test_should_check_convergence() {
        let config = ConvergenceConfig::default().with_check_interval(5);

        assert!(should_check_convergence(0, &config)); // Always check first
        assert!(!should_check_convergence(1, &config));
        assert!(!should_check_convergence(4, &config));
        assert!(should_check_convergence(5, &config));
        assert!(should_check_convergence(10, &config));
    }

    #[test]
    fn test_check_kappa_convergence() {
        let device = default_device();

        let old: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.0, 0.5, 1.0], &device);
        let new: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.1, 0.5, 0.9], &device);

        let delta = check_kappa_convergence(&old, &new);

        // Max difference is 0.1 (at positions 0 and 2)
        assert!((delta - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_validate_monotonicity_ok() {
        let device = default_device();

        let old: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.0, 0.5, 0.8], &device);
        let new: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.1, 0.6, 0.9], &device);

        assert!(validate_monotonicity(&old, &new, 1e-6).is_ok());
    }

    #[test]
    fn test_validate_monotonicity_violation() {
        let device = default_device();

        let old: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.5, 0.5, 0.8], &device);
        let new: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.4, 0.6, 0.9], &device);

        let result = validate_monotonicity(&old, &new, 1e-6);
        assert!(result.is_err());

        let decrease = result.unwrap_err();
        assert!((decrease - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_convergence_state_history() {
        let mut state = ConvergenceState::with_history();
        let config = ConvergenceConfig::default();

        state.update(1.0, &config);
        state.update(0.5, &config);
        state.update(0.25, &config);

        assert_eq!(state.delta_history.len(), 3);

        let rate = state.convergence_rate().unwrap();
        assert!((rate - 0.5).abs() < 1e-6); // 0.25 / 0.5 = 0.5
    }
}
