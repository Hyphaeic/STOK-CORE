//! # Prediction Utilities
//!
//! This module provides utilities for high-dimensional STOK factorization
//! and state prediction as described in Theorem 2.1 of the paper.
//!
//! ## Key Components
//!
//! - **StatePredictionKernel (SPK)**: Predicts final states under default dynamics
//! - **CumulativeEventFunction (CEF)**: Tracks event probabilities over time
//! - **TemporalEventFunction (TEF)**: First-event timing distributions
//!
//! These components enable the STOK factorization (Equation [23]):
//! ```text
//! η̃(z_f, x_f, t_f | z, x) = ξ(t_f | z, x) · ρ_π(x_f | x, t_f) · ∏_k ρ_k(z_k,f | z_k, t_f)
//! ```

mod cef;
mod spk;

pub use cef::{CumulativeEventFunction, TemporalEventFunction};
pub use spk::StatePredictionKernel;
