//! # Planning with Goal Kernels
//!
//! This module implements the planning interface for STOK-based control.
//! It provides utilities for managing multiple goal-conditioned STOKs,
//! querying feasibility, and searching for optimal option sequences.
//!
//! ## Key Components
//!
//! - **GoalKernel**: Manages a collection of STOKs for different goals
//! - **Tree Search**: BFS and best-first search over option sequences
//! - **Sampling**: Sample termination distributions from STOKs
//! - **Plan**: Representation and validation of option sequences
//!
//! ## Example
//!
//! ```rust,ignore
//! use stok_core::planning::{GoalKernel, tree_search, TreeSearchConfig};
//!
//! // Create goal kernel with multiple options
//! let mut kernel = GoalKernel::new(n_states, device);
//! kernel.add_goal(GoalId(0), stok1, "reach_waypoint", Some(10))?;
//! kernel.add_goal(GoalId(1), stok2, "reach_goal", Some(20))?;
//!
//! // Search for plan
//! let config = TreeSearchConfig {
//!     max_depth: 5,
//!     target_goal: Some(GoalId(1)),
//!     ..Default::default()
//! };
//!
//! let result = tree_search(&kernel, initial_state, config);
//! if let Some(plan) = result.best_plan {
//!     println!("Found plan with feasibility: {}", plan.feasibility);
//! }
//! ```

mod goal_kernel;
mod plan;
mod query;
mod sampling;
mod tree_search;

pub use goal_kernel::{GoalInfo, GoalKernel};
pub use plan::{Plan, PlanError, SimulatedTrajectory, SimulationResult, simulate_plan};
pub use query::PlanningQuery;
pub use sampling::{STOKSampler, TerminationOutcome, sample_categorical};
pub use tree_search::{
    best_first_search, tree_search, SearchNode, SearchResult, SearchStats, SearchStrategy,
    TreeSearchConfig,
};
