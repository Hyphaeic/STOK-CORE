//! Task MDP definitions and builders.
//!
//! A Task MDP (TMDP) is an MDP augmented with:
//! - f_g: goal satisfaction function
//! - f_c: constraint satisfaction function
//!
//! This enables feasibility-aware planning per Ringstrom & Schrater (2025).

mod task_mdp;

pub use task_mdp::TaskMDP;