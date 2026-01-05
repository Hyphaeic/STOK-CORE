//! # Planning Query Interface
//!
//! High-level API for querying the goal kernel and finding plans.

use super::goal_kernel::GoalKernel;
use super::plan::Plan;
use super::tree_search::{best_first_search, tree_search, SearchStrategy, TreeSearchConfig};
use crate::types::GoalId;
use burn::prelude::*;

/// High-level query interface for planning
///
/// Provides convenient methods for common planning queries.
///
/// # Example
///
/// ```rust,ignore
/// let query = PlanningQuery::new(&goal_kernel);
///
/// // Can we reach the goal?
/// if query.is_reachable(0, GoalId(10), 5) {
///     // Find best plan
///     if let Some(plan) = query.find_plan(0, GoalId(10), 5) {
///         println!("Plan: {}", plan);
///     }
/// }
/// ```
pub struct PlanningQuery<'a, B: Backend> {
    goal_kernel: &'a GoalKernel<B>,
}

impl<'a, B: Backend> PlanningQuery<'a, B> {
    /// Create new planning query interface
    pub fn new(goal_kernel: &'a GoalKernel<B>) -> Self {
        Self { goal_kernel }
    }

    /// Find best plan to reach target goal
    ///
    /// Uses breadth-first search by default.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `target_goal` - Goal to reach
    /// * `max_depth` - Maximum option sequence length
    ///
    /// # Returns
    ///
    /// Best plan found, or None if no plan exists
    pub fn find_plan(
        &self,
        initial_state: usize,
        target_goal: GoalId,
        max_depth: usize,
    ) -> Option<Plan> {
        let config = TreeSearchConfig {
            max_depth,
            target_goal: Some(target_goal),
            ..Default::default()
        };

        let result = tree_search(self.goal_kernel, initial_state, config);
        result.best_plan
    }

    /// Find plan using best-first search
    ///
    /// Prioritizes high-feasibility paths.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `target_goal` - Goal to reach
    /// * `max_depth` - Maximum option sequence length
    ///
    /// # Returns
    ///
    /// Best plan found, or None
    pub fn find_plan_best_first(
        &self,
        initial_state: usize,
        target_goal: GoalId,
        max_depth: usize,
    ) -> Option<Plan> {
        let config = TreeSearchConfig {
            max_depth,
            target_goal: Some(target_goal),
            search_strategy: SearchStrategy::BestFirst,
            ..Default::default()
        };

        let result = best_first_search(self.goal_kernel, initial_state, config);
        result.best_plan
    }

    /// Find all feasible plans up to depth
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `target_goal` - Goal to reach
    /// * `max_depth` - Maximum option sequence length
    ///
    /// # Returns
    ///
    /// Vector of all plans found
    pub fn find_all_plans(
        &self,
        initial_state: usize,
        target_goal: GoalId,
        max_depth: usize,
    ) -> Vec<Plan> {
        let config = TreeSearchConfig {
            max_depth,
            target_goal: Some(target_goal),
            track_all_paths: true,
            ..Default::default()
        };

        let result = tree_search(self.goal_kernel, initial_state, config);
        result.all_plans
    }

    /// Check if target is reachable from state
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `target_goal` - Goal to reach
    /// * `max_depth` - Maximum search depth
    ///
    /// # Returns
    ///
    /// true if a plan exists
    pub fn is_reachable(
        &self,
        initial_state: usize,
        target_goal: GoalId,
        max_depth: usize,
    ) -> bool {
        // Direct check first
        if self.goal_kernel.is_feasible(target_goal, initial_state) {
            return true;
        }

        // Search for indirect path
        self.find_plan(initial_state, target_goal, max_depth)
            .is_some()
    }

    /// Query best feasibility to reach target
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `target_goal` - Goal to reach
    /// * `max_depth` - Maximum search depth
    ///
    /// # Returns
    ///
    /// Best feasibility found, or 0.0 if no plan exists
    pub fn best_feasibility(
        &self,
        initial_state: usize,
        target_goal: GoalId,
        max_depth: usize,
    ) -> f32 {
        self.find_plan(initial_state, target_goal, max_depth)
            .map(|p| p.feasibility)
            .unwrap_or(0.0)
    }

    /// Get directly achievable goals from state
    ///
    /// Returns goals with κ_g(x) > 0
    ///
    /// # Arguments
    ///
    /// * `state` - Current state
    ///
    /// # Returns
    ///
    /// Vector of feasible goal IDs
    pub fn achievable_goals(&self, state: usize) -> Vec<GoalId> {
        self.goal_kernel.feasible_goals(state)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::stok::STOKKernel;
    use burn::prelude::Tensor;

    #[test]
    fn test_query_interface_creation() {
        let device = default_device();
        let kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device);
        let _query = PlanningQuery::new(&kernel);
    }

    #[test]
    fn test_is_reachable_direct() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 5, &device);
        stok.kappa = Tensor::from_floats([1.0, 1.0, 0.0, 0.0, 1.0], &device);

        kernel.add_goal(GoalId(0), stok, "goal", None).unwrap();

        let query = PlanningQuery::new(&kernel);

        // Direct reachability
        assert!(query.is_reachable(0, GoalId(0), 5));
        assert!(query.is_reachable(1, GoalId(0), 5));
        assert!(!query.is_reachable(2, GoalId(0), 1)); // Infeasible
    }

    #[test]
    fn test_achievable_goals() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(3, device.clone());

        let mut stok1: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok1.kappa = Tensor::from_floats([1.0, 0.0, 0.0], &device);
        kernel.add_goal(GoalId(0), stok1, "g0", None).unwrap();

        let mut stok2: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok2.kappa = Tensor::from_floats([1.0, 1.0, 0.0], &device);
        kernel.add_goal(GoalId(1), stok2, "g1", None).unwrap();

        let query = PlanningQuery::new(&kernel);

        // From state 0: both goals achievable
        let achievable = query.achievable_goals(0);
        assert_eq!(achievable.len(), 2);

        // From state 1: only goal 1 achievable
        let achievable1 = query.achievable_goals(1);
        assert_eq!(achievable1.len(), 1);
        assert!(achievable1.contains(&GoalId(1)));
    }
}
