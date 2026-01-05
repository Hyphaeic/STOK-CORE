//! # Compositional STOK Operations
//!
//! This module implements Chapman-Kolmogorov composition for State-Time Option Kernels,
//! enabling sequential option chaining as described in Ringstrom & Schrater (2025).
//!
//! ## Key Concepts
//!
//! - **STOK Composition**: Combines two STOKs to predict termination distribution
//!   for executing options sequentially. This involves a convolution in time
//!   and marginalization over intermediate states.
//!
//! - **SOK Composition**: Simpler time-marginalized version that reduces to
//!   matrix multiplication.
//!
//! ## Mathematical Background
//!
//! STOK composition implements Equation [18] from the paper:
//! ```text
//! η_μ(x_μ, t_μ | x) = Σ_{x_{f1}} Σ_{t_{f1}} η_{o2}(x_μ, t_μ - t_{f1} | x_{f1}) · η_{o1}(x_{f1}, t_{f1} | x)
//! ```
//!
//! SOK composition implements Equation [19]:
//! ```text
//! χ_μ(x_μ | x) = Σ_{x_{f1}} χ_{o2}(x_μ | x_{f1}) · χ_{o1}(x_{f1} | x)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use stok_core::composition::{compose_stoks, StateOptionKernel};
//!
//! // Compose two options
//! let composed = compose_stoks(&stok1, &stok2)?;
//!
//! // Or work with SOKs for time-invariant tasks
//! let sok1 = StateOptionKernel::from_stok(&stok1);
//! let sok2 = StateOptionKernel::from_stok(&stok2);
//! let sok_composed = compose_soks(&sok1, &sok2)?;
//! ```

mod chapman_kolmogorov;
mod sequence;
mod sok;
mod validation;

pub use chapman_kolmogorov::{
    compose_soks, compose_stoks, compose_stoks_with_decomposition, composed_time_horizon,
    ComposedSTOK,
};

pub use sok::StateOptionKernel;

pub use sequence::{compose_sequence, compose_sok_sequence, OptionSequence};

pub use validation::{
    validate_composed_stok, validate_composition_consistency, validate_sok_stok_consistency,
};
