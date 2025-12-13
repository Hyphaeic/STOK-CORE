//! Convergence detection for feasibility iteration.
//!
//! Monitors the L∞ distance between successive κ estimates and
//! determines when the iteration has converged to κ*.
//!
//! # Convergence Criterion
//!
//! ```text
//! δ^(d) = ‖κ^(d+1) - κ^(d)‖_∞ = max_x |κ^(d+1)(x) - κ^(d)(x)|
//! Converged iff δ^(d) < ε
//! ```
//!
//! # Theoretical Guarantee
//!
//! Per Appendix E of Ringstrom & Schrater (2025), the κ-OKBE Bellman
//! operator is a contraction mapping, guaranteeing:
//! - Convergence to unique fixed point κ*
//! - Monotonic increase of κ (κ^(d+1) ≥ κ^(d) element-wise)

use std::fmt;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for convergence detection.
#[derive(Clone, Debug)]
pub struct ConvergenceConfig {
    /// Convergence tolerance (L∞ norm threshold)
    /// Default: 1e-6
    pub epsilon: f32,
    
    /// Maximum iterations before forced termination
    /// Default: 1000
    pub max_iterations: usize,
    
    /// Check convergence every N iterations (for performance)
    /// Default: 1 (check every iteration)
    pub check_interval: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            max_iterations: 1000,
            check_interval: 1,
        }
    }
}

impl ConvergenceConfig {
    /// Create config with custom tolerance.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Create config with custom max iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }
    
    /// Create config with batched convergence checks.
    ///
    /// For large state spaces, checking every iteration can be
    /// expensive due to GPU-CPU synchronization. Batching helps.
    pub fn with_check_interval(mut self, interval: usize) -> Self {
        self.check_interval = interval.max(1);
        self
    }
}

// ============================================================================
// State Tracking
// ============================================================================

/// Reason for iteration termination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Iteration has not yet terminated
    NotConverged,
    /// δ < ε achieved
    ToleranceReached,
    /// max_iterations exceeded
    MaxIterationsReached,
}

impl fmt::Display for ConvergenceReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConverged => write!(f, "not converged"),
            Self::ToleranceReached => write!(f, "tolerance reached"),
            Self::MaxIterationsReached => write!(f, "max iterations reached"),
        }
    }
}

/// State of the convergence process.
#[derive(Clone, Debug)]
pub struct ConvergenceState {
    /// Current iteration count
    pub iteration: usize,
    
    /// Most recent δ = ‖κ_new - κ_old‖_∞
    pub delta: f32,
    
    /// Whether iteration has converged
    pub converged: bool,
    
    /// Reason for termination (if converged)
    pub reason: ConvergenceReason,
}

impl ConvergenceState {
    /// Create new convergence state at iteration 0.
    pub fn new() -> Self {
        Self {
            iteration: 0,
            delta: f32::INFINITY,
            converged: false,
            reason: ConvergenceReason::NotConverged,
        }
    }
    
    /// Update state after computing new delta.
    pub fn update(&mut self, delta: f32, config: &ConvergenceConfig) {
        self.iteration += 1;
        self.delta = delta;
        
        if delta < config.epsilon {
            self.converged = true;
            self.reason = ConvergenceReason::ToleranceReached;
        } else if self.iteration >= config.max_iterations {
            self.converged = true;
            self.reason = ConvergenceReason::MaxIterationsReached;
        }
    }
    
    /// Check if iteration should terminate.
    pub fn should_terminate(&self, _config: &ConvergenceConfig) -> bool {
        self.converged
    }
}

impl Default for ConvergenceState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convergence Checking Functions
// ============================================================================

/// Check if convergence should be evaluated this iteration.
///
/// For performance, convergence checks can be batched every N iterations.
#[inline]
pub fn should_check_convergence(iteration: usize, config: &ConvergenceConfig) -> bool {
    iteration % config.check_interval == 0
}

/// Compute convergence delta from two κ tensors.
///
/// Returns `‖kappa_new - kappa_old‖_∞`
pub fn compute_convergence_delta<B: burn::prelude::Backend>(
    kappa_old: &burn::prelude::Tensor<B, 1>,
    kappa_new: &burn::prelude::Tensor<B, 1>,
) -> f32 {
    // TODO: Implement using linf_distance from utils::numerics
    // crate::utils::linf_distance(kappa_old, kappa_new)
    
    todo!("Implement compute_convergence_delta")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_config_default() {
        let config = ConvergenceConfig::default();
        assert_eq!(config.epsilon, 1e-6);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.check_interval, 1);
    }

    #[test]
    fn test_convergence_config_builder() {
        let config = ConvergenceConfig::default()
            .with_epsilon(1e-4)
            .with_max_iterations(500)
            .with_check_interval(10);
        
        assert_eq!(config.epsilon, 1e-4);
        assert_eq!(config.max_iterations, 500);
        assert_eq!(config.check_interval, 10);
    }

    #[test]
    fn test_convergence_state_tolerance_reached() {
        let config = ConvergenceConfig::default().with_epsilon(1e-4);
        let mut state = ConvergenceState::new();
        
        state.update(1e-5, &config);  // Below tolerance
        
        assert!(state.converged);
        assert_eq!(state.reason, ConvergenceReason::ToleranceReached);
    }

    #[test]
    fn test_convergence_state_max_iterations() {
        let config = ConvergenceConfig::default().with_max_iterations(2);
        let mut state = ConvergenceState::new();
        
        state.update(1.0, &config);  // iter 1, not converged
        assert!(!state.converged);
        
        state.update(0.5, &config);  // iter 2, max reached
        assert!(state.converged);
        assert_eq!(state.reason, ConvergenceReason::MaxIterationsReached);
    }

    #[test]
    fn test_should_check_convergence() {
        let config = ConvergenceConfig::default().with_check_interval(5);
        
        assert!(should_check_convergence(0, &config));
        assert!(!should_check_convergence(1, &config));
        assert!(!should_check_convergence(4, &config));
        assert!(should_check_convergence(5, &config));
        assert!(should_check_convergence(10, &config));
    }
}
