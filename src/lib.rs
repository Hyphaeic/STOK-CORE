//! # STOK-Core: State-Time Option Kernels
//!
//! GPU-accelerated implementation of State-Time Option Kernels (STOKs)
//! for compositional, feasibility-aware hierarchical planning.
//!
//! ## Overview
//!
//! STOKs replace scalar value functions with full probability distributions:
//!
//! ```text
//! η(x_f, t_f | x_i) = P(terminate at state x_f, time t_f | started at x_i)
//! ```
//!
//! This enables:
//! - **Exact composition** via Chapman-Kolmogorov equations (Eq. [18])
//! - **Native constraint handling** through feasibility functions
//! - **Temporal precision** via time-aware predictions
//!
//! ## Mathematical Foundation
//!
//! ### Task MDP
//!
//! A Task MDP M = ⟨X, A, P, f_g, f_c⟩ extends standard MDPs with:
//! - **f_g(x,a)**: Goal satisfaction probability
//! - **f_c(x,a)**: Constraint satisfaction probability
//!
//! ### κ-OKBE (Equation [7])
//!
//! The cumulative feasibility function κ*(x) is computed via:
//!
//! ```text
//! κ*(x) = max_a [f₁(x,a) + f₂(x,a) Σ P(x'|x,a) κ*(x')]
//! ```
//!
//! Where f₁ = f_g × f_c and f₂ = (1-f_g) × f_c.
//!
//! ### STOK Decomposition (Equations [15-17])
//!
//! ```text
//! η** = η⁺ + η⁻
//! κ(x) = Σ_{x_f,t_f} η⁺(x_f, t_f | x)
//! 1 - κ(x) = Σ_{x_f,t_f} η⁻(x_f, t_f | x)
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use stok_core::prelude::*;
//!
//! // Create a simple chain MDP
//! let device = default_device();
//! let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
//!
//! // Validate the MDP
//! mdp.validate().expect("MDP should be valid");
//!
//! // Create empty STOK kernel (populated by feasibility iteration in Phase 2)
//! let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 20, &device);
//! ```
//!
//! ## Module Structure
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`types`] | Index newtypes, dimensions, error types |
//! | [`backend`] | GPU device management |
//! | [`mdp`] | Task MDP definition and builders |
//! | [`stok`] | STOK kernel structure |
//!
//! ## Implementation Phases
//!
//! This crate implements the STOK framework in four phases:
//!
//! 1. **Phase 1** (Current): Data structures and memory layout
//! 2. **Phase 2**: Feasibility iteration (MVP)
//! 3. **Phase 3**: Chapman-Kolmogorov composition
//! 4. **Phase 4**: Planning interface and tree search
//!
//! ## Performance Targets
//!
//! | Operation | Target | Scale |
//! |-----------|--------|-------|
//! | Bellman backup | < 1ms | 100 states |
//! | Full convergence | < 100ms | 100 states |
//! | STOK composition | < 10ms | 100 states |
//!
//! ## Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). "Compositionality and Bounds for
//! Optimal Value Functions in Reinforcement Learning." *arXiv preprint*.
//!
//! ## Equation Reference
//!
//! | Equation | Description |
//! |----------|-------------|
//! | [7] | κ-OKBE: Bellman equation for cumulative feasibility |
//! | [8] | π-OKBE: Time-minimizing policy selection |
//! | [9-10] | STOK recursive updates for t > t₀ |
//! | [11-12] | STOK boundary conditions at t = t₀ |
//! | [15-16] | κ from η⁺, (1-κ) from η⁻ |
//! | [17] | Combined STOK: η** = η⁺ + η⁻ |
//! | [18] | Chapman-Kolmogorov STOK composition |
//! | [19] | SOK composition (matrix multiplication) |

// Module declarations
pub mod types;
pub mod backend;
pub mod mdp;
pub mod stok;
pub mod solver;
pub mod utils;

/// Common imports for STOK usage.
///
/// This prelude re-exports the most commonly used types and functions
/// for convenient single-line imports.
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::prelude::*;
///
/// let device = default_device();
/// let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
/// ```
pub mod prelude {
    pub use crate::types::{
        StateIdx, ActionIdx, TimeIdx, GoalId,
        MDPDimensions, STOKDimensions, StokError,
        DEFAULT_CONVERGENCE_TOLERANCE,
        DEFAULT_PROBABILITY_TOLERANCE,
    };
    
    pub use crate::backend::{
        DefaultBackend, DefaultDevice,
        DeviceConfig, DeviceManager,
        default_device, cpu_device,
    };
    
    pub use crate::mdp::TaskMDP;
    pub use crate::stok::{STOKKernel, STOKData};
}

// Top-level re-exports
pub use prelude::*;

// ============================================================================
// Integration Tests (in lib.rs for doc visibility)
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::prelude::*;

    /// Test end-to-end MDP creation and validation
    #[test]
    fn test_mdp_workflow() {
        let device = default_device();
        
        // Create MDP
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
        
        // Validate
        assert!(mdp.validate().is_ok());
        
        // Check dimensions
        assert_eq!(mdp.n_states(), 10);
        assert_eq!(mdp.n_actions(), 3);
        assert_eq!(mdp.max_time(), 20);
    }

    /// Test STOK kernel creation
    #[test]
    fn test_kernel_workflow() {
        let device = default_device();
        
        // Create kernel
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 20, &device);
        
        // Check dimensions
        assert_eq!(kernel.n_states(), 10);
        assert_eq!(kernel.max_time(), 20);
        
        // Zero-initialized kernel should pass bounds check
        assert!(kernel.validate_probability_bounds().is_ok());
    }

    /// Test constrained MDP
    #[test]
    fn test_constrained_mdp() {
        let device = default_device();
        
        // Fire state at position 5
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(10, 5, 20, &device);
        
        assert!(mdp.validate().is_ok());
        
        // Check that fire state has f_c = 0
        let constraint_data: Vec<f32> = mdp.constraint_fn.clone()
            .into_data()
            .to_vec()
            .unwrap();
        
        for a in 0..3 {
            assert_eq!(constraint_data[5 * 3 + a], 0.0);
        }
    }

    /// Test stochastic MDP
    #[test]
    fn test_stochastic_mdp() {
        let device = default_device();
        
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(10, 0.8, 20, &device);
        
        // Should be properly normalized
        assert!(mdp.validate_stochastic(1e-5).is_ok());
        assert!(mdp.validate().is_ok());
    }

    /// Test derived functions computation
    #[test]
    fn test_derived_functions() {
        let device = default_device();
        
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);
        
        // At non-goal states: f_g = 0, f_c = 1
        // So f_1 = 0, f_2 = 1
        
        // At goal state (4): f_g = 1, f_c = 1
        // So f_1 = 1, f_2 = 0
        
        let f1_data: Vec<f32> = mdp.f1.clone().into_data().to_vec().unwrap();
        let f2_data: Vec<f32> = mdp.f2.clone().into_data().to_vec().unwrap();
        
        // Check goal state
        for a in 0..3 {
            assert_eq!(f1_data[4 * 3 + a], 1.0);
            assert_eq!(f2_data[4 * 3 + a], 0.0);
        }
        
        // Check non-goal state
        for a in 0..3 {
            assert_eq!(f1_data[0 * 3 + a], 0.0);
            assert_eq!(f2_data[0 * 3 + a], 1.0);
        }
    }
}
// Phase 2 re-exports
pub use solver::{
    feasibility_iteration,
    solve_task_mdp,
    FeasibilityIterationConfig,
    FeasibilityIterationResult,
    ConvergenceConfig,
    ConvergenceState,
};
