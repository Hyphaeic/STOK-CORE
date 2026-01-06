//! # Hierarchical and Product-Space STOK Support
//!
//! This module implements high-dimensional STOK factorization for product spaces.
//!
//! ## Mathematical Foundation
//!
//! Product space: S = X × Z₁ × ... × Zₙ where:
//! - X: Base-level state space (e.g., spatial position)
//! - Zₖ: High-level state spaces (e.g., hydration, task logic)
//!
//! ## Key Components
//!
//! - **ProductState**: Represents states in S = X × Z
//! - **AffordanceFunction**: Links base actions to HL transformations (Definition 0.1)
//! - **FactorizedSTOK**: STOK factorization (Theorem 2.1, Equation [23])
//! - **Composition λ**: Builds product-space kernels (Definition 2.1)
//!
//! ## STOK Factorization (Equation [23])
//!
//! ```text
//! η̃(z_f, x_f, t_f | z, x) = ξ(t_f | z, x) · ρ_π(x_f | x, t_f) · ∏_k ρ_k(z_k,f | z_k, t_f)
//! ```
//!
//! Where:
//! - ξ: Temporal Event Function (when does first event occur?)
//! - ρ_π: Base-level State Prediction Kernel
//! - ρ_k: High-level State Prediction Kernels
//!
//! ## Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Section 2: "Compositional Task MDPs"

mod affordance;
mod factorization;
mod modes;
mod product_space;
mod sublimation;

pub use affordance::{
    AffordanceFunction, FactorizedAffordance, HLAction, HLActionSet,
};
pub use factorization::{
    FactorizedSTOK, assemble_factorized_stok,
};
pub use modes::{
    KeyDoorMode, ModeConditionedMDP, ModeFunction, MultiBitMode, NoMode, ThresholdMode,
    MODE_CLOSED, MODE_OPEN,
};
pub use product_space::{
    HLState, ProductSpaceDims, ProductState,
};
pub use sublimation::{
    SublimatedFeasibilityCache, SublimatedTMDP, compute_sublimated_feasibility,
    extract_hl_constraint, maximize_goal_over_base,
};
