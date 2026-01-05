//! State-Time Option Kernel (STOK) structures.
//!
//! STOKs are predictive probability kernels over termination events,
//! enabling exact composition via Chapman-Kolmogorov equations.
//!
//! # Overview
//!
//! Unlike scalar value functions, STOKs encode the full distribution
//! over termination events:
//!
//! ```text
//! η(x_f, t_f | x_i) = P(terminate at state x_f, time t_f | start at x_i)
//! ```
//!
//! # Decomposition (Equations [15-17])
//!
//! The combined STOK decomposes into success and failure components:
//!
//! ```text
//! η** = η⁺ + η⁻
//! ```
//!
//! Where:
//! - **η⁺**: Success termination (goal reached without constraint violation)
//! - **η⁻**: Failure termination (constraint violated before goal)
//!
//! # Key Invariants
//!
//! 1. **Normalization**: Σ_{x_f, t_f} η**(x_f, t_f | x) = 1 for all x
//! 2. **κ-η consistency**: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
//! 3. **Feasibility bound**: 1 - κ(x) = Σ_{x_f, t_f} η⁻(x_f, t_f | x)
//!
//! # Composition (Phase 3)
//!
//! STOKs compose via Chapman-Kolmogorov equations (Equation [18]):
//!
//! ```text
//! η_μ(x_μ, t_μ | x) = Σ_{x_m, t_m} η_2(x_μ, t_μ-t_m | x_m) η_1(x_m, t_m | x)
//! ```
//!
//! # Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Section G.2: "State-Time Option Kernel"

mod kernel;

pub use kernel::{STOKData, STOKKernel};
