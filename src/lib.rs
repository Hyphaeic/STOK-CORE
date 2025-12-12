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
//! - **Exact composition** via Chapman-Kolmogorov equations
//! - **Native constraint handling** through feasibility functions
//! - **Temporal precision** via time-aware predictions
//!
//! ## Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Compositionality and Bounds for
//! Optimal Value Functions in Reinforcement Learning.
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
//! // Create empty STOK kernel (will be populated by feasibility iteration in Phase 2)
//! let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 20, &device);
//! ```

// Module declarations
pub mod types;
pub mod backend;
pub mod mdp;
pub mod stok;

// Re-exports for convenient access
pub mod prelude {
    //! Common imports for STOK usage.
    
    pub use crate::types::{
        StateIdx, ActionIdx, TimeIdx, GoalId,
        MDPDimensions, StokError,
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