//! # Planning with Goal Kernels
//!
//! Two distinct API levels live here (PP-504):
//!
//! - **Reduced low-dimensional planning** (legacy / example-grade):
//!   - [`GoalKernel`] — flat `HashMap<GoalId, STOKKernel>` for problems where
//!     the planning state is just the BL state `x ∈ X`.
//!   - [`tree_search`] / [`best_first_search`] — search over option sequences
//!     using only flat BL state. Sublimation pruning is not yet wired in
//!     (TODO at `tree_search.rs:272`, owned by PP-603).
//!
//! - **Paper-faithful planning** (M5+):
//!   - [`FactorizedGoalKernel`] — Eq [24] Goal Kernel `G` carrying per-goal
//!     `FactorizedSTOK`s, the affordance, and HL kernels for boundary updates.
//!   - [`PlanKernel`] — m-fold composition `G_m` of the Goal Kernel under a
//!     meta-policy, with deterministic and stochastic simulation.
//!
//! Algorithm 2 (PP-601..605) will replace the reduced tree search with one
//! that operates on the paper-faithful composite state via the
//! `FactorizedGoalKernel` / `PlanKernel` pair.

mod goal_kernel;
mod option_set;
mod plan;
mod query;
mod sampling;
mod tree_search;

pub use goal_kernel::{FactorizedGoalEntry, FactorizedGoalKernel, GoalInfo, GoalKernel};
pub use option_set::{build_affordance_option_set, build_state_option_set};
pub use plan::{
    simulate_plan, Plan, PlanError, PlanKernel, PlanStep, PlanTrace, SimulatedTrajectory,
    SimulationResult,
};
pub use query::PlanningQuery;
pub use sampling::{STOKSampler, TerminationOutcome, sample_categorical};
pub use tree_search::{
    algorithm_2_search, best_first_search, tree_search, Algorithm2Config, Algorithm2Node,
    Algorithm2Plan, Algorithm2Result, Algorithm2Stats, SearchNode, SearchResult, SearchStats,
    SearchStrategy, TreeSearchConfig,
};
