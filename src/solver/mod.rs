//! Feasibility iteration solver for STOK computation.
//!
//! This module implements the core optimization loop that computes
//! the optimal cumulative feasibility function κ* and policy π**,
//! then constructs the full State-Time Option Kernel (STOK).
//!
//! # Overview
//!
//! The solver implements Algorithm 1 from Ringstrom & Schrater (2025):
//!
//! 1. **Initialize**: κ⁰ ← 0, π⁰ ← 0 (zero initialization critical)
//! 2. **Iterate**: Apply κ-OKBE Bellman backup until convergence
//! 3. **Construct**: Build η⁺, η⁻ from converged κ*, π**
//!
//! # Key Equations
//!
//! - **Eq [7]**: κ-OKBE - `κ*(x) = max_a [f₁(x,a) + f₂(x,a) E[κ*(x')]]`
//! - **Eq [11-12]**: STOK boundary conditions at t=0
//! - **Eq [9-10]**: STOK time propagation for t>0
//!
//! # Example
//!
//! ```ignore
//! use stok_core::prelude::*;
//! use stok_core::solver::feasibility_iteration;
//!
//! let device = default_device();
//! let mdp = TaskMDP::simple_chain(10, 20, &device);
//! let config = FeasibilityIterationConfig::default();
//!
//! let result = feasibility_iteration(&mdp, config)?;
//! println!("Converged in {} iterations", result.convergence.iteration);
//! ```

pub mod bellman;
pub mod convergence;
pub mod feasibility_iteration;
pub mod stok_construction;

// Re-exports for public API
pub use bellman::bellman_backup_kappa;
pub use convergence::{ConvergenceConfig, ConvergenceState, ConvergenceReason};
pub use feasibility_iteration::{
    FeasibilityIterationConfig,
    FeasibilityIterationResult,
    IterationTiming,
    feasibility_iteration,
    solve_task_mdp,
};
pub use stok_construction::construct_stok;
