//! Solver module for STOK computation.
//!
//! This module provides the core optimization algorithms for computing
//! State-Time Option Kernels via feasibility iteration.
//!
//! # Overview
//!
//! The solver implements the feasibility iteration algorithm from
//! Ringstrom & Schrater (2025), which computes:
//! - κ* - optimal cumulative feasibility function
//! - π** - optimal policy
//! - η⁺, η⁻ - success/failure termination distributions
//!
//! # Quick Start
//!
//! ```ignore
//! use stok_core::{TaskMDP, solve_task_mdp};
//!
//! // Create an MDP
//! let mdp = TaskMDP::simple_chain(10, 20, &device);
//!
//! // Solve to get complete STOK
//! let stok = solve_task_mdp(&mdp)?;
//!
//! // Access results
//! let feasibility = stok.kappa;      // κ*(x) for each state
//! let policy = stok.policy;          // π**(x) for each state
//! let success_dist = stok.eta_plus;  // η⁺(x_f, t_f | x)
//! ```
//!
//! # Detailed Usage
//!
//! For more control over the optimization:
//!
//! ```ignore
//! use stok_core::solver::{feasibility_iteration, FeasibilityIterationConfig};
//!
//! let config = FeasibilityIterationConfig {
//!     convergence: ConvergenceConfig::default().with_epsilon(1e-8),
//!     max_time: 50,
//!     compute_full_stok: true,
//!     ..Default::default()
//! };
//!
//! let result = feasibility_iteration(&mdp, config)?;
//!
//! println!("Converged in {} iterations", result.iterations());
//! println!("Per-iteration time: {:.2}ms", result.timing.per_iteration_ms);
//! ```
//!
//! # Module Structure
//!
//! - [`bellman`] - Bellman backup operators (κ-OKBE)
//! - [`convergence`] - Convergence detection utilities
//! - [`feasibility_iteration`] - Main optimization loop
//! - [`stok_construction`] - η⁺/η⁻ computation

pub mod bellman;
pub mod convergence;
pub mod feasibility_iteration;
pub mod stok_construction;

// Re-export primary types
pub use bellman::{
    bellman_backup_kappa,
    compute_q_values,
    get_policy_transition,
    gather_by_policy,
};

pub use convergence::{
    ConvergenceConfig,
    ConvergenceState,
    ConvergenceReason,
    check_kappa_convergence,
    should_check_convergence,
    validate_monotonicity,
};

pub use feasibility_iteration::{
    FeasibilityIterationConfig,
    FeasibilityIterationResult,
    IterationTiming,
    feasibility_iteration,
    solve_task_mdp,
    compute_kappa,
};

pub use stok_construction::{
    construct_stok,
    validate_stok_normalization,
    validate_kappa_eta_consistency,
};