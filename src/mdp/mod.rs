//! Task MDP definitions for STOK computation.
//!
//! This module provides the `TaskMDP` structure, which extends standard MDPs
//! with explicit goal and constraint functions for feasibility-aware planning.
//!
//! # Task MDP Definition
//!
//! A Task MDP M = ⟨X, A, P, f_g, f_c⟩ consists of:
//!
//! | Component | Type | Description |
//! |-----------|------|-------------|
//! | X | State space | Discrete states indexed 0..n_states |
//! | A | Action space | Actions indexed 0..n_actions |
//! | P(x'│x,a) | Tensor [S,A,S] | Transition dynamics (row-stochastic) |
//! | f_g(x,a) | Tensor [S,A] | Goal satisfaction probability ∈ [0,1] |
//! | f_c(x,a) | Tensor [S,A] | Constraint satisfaction probability ∈ [0,1] |
//!
//! # Derived Functions
//!
//! From f_g and f_c, we derive (used in κ-OKBE, Equation [7]):
//!
//! - **f_1 = f_g × f_c**: Achievement function (goal AND constraint satisfied)
//! - **f_2 = (1 - f_g) × f_c**: Continuation function (NOT goal AND constraint)
//!
//! # Builder Methods
//!
//! The module provides test MDP builders:
//!
//! - `TaskMDP::simple_chain()` - Deterministic linear chain
//! - `TaskMDP::constrained_chain()` - Chain with constraint-violating state
//! - `TaskMDP::stochastic_chain()` - Chain with stochastic transitions
//!
//! # Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). "Compositionality and Bounds for
//! Optimal Value Functions in Reinforcement Learning." Section 2.

mod task_mdp;

pub use task_mdp::TaskMDP;