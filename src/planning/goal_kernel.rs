//! # Goal Kernel Manager
//!
//! Manages a collection of goal-conditioned STOKs for multi-goal planning.
//!
//! Implements the Goal Kernel from Equation [24] of the paper.

use crate::composition::StateOptionKernel;
use crate::stok::STOKKernel;
use crate::types::{GoalId, StokError};
use burn::prelude::*;
use std::collections::HashMap;

/// Metadata about a goal
#[derive(Clone, Debug)]
pub struct GoalInfo {
    /// Unique goal identifier
    pub id: GoalId,

    /// Human-readable name
    pub name: String,

    /// Target state (if goal is to reach specific state)
    pub target_state: Option<usize>,

    /// Maximum time horizon for this goal's STOK
    pub max_time: usize,
}

/// Goal Kernel: manages collection of STOKs for planning
///
/// Implements the goal kernel G from Equation [24]:
/// ```text
/// G(z', x', t_f + t + 1 | (z, x)^ℓ, t, o_{ℓ,g}) = ...
/// ```
///
/// Provides:
/// - Storage and retrieval of goal-conditioned STOKs
/// - Feasibility queries
/// - Affordance computation (which goals are reachable)
///
/// # Example
///
/// ```rust,ignore
/// let mut kernel = GoalKernel::new(n_states, device);
/// kernel.add_goal(GoalId(0), stok1, "waypoint", Some(5))?;
/// kernel.add_goal(GoalId(1), stok2, "goal", Some(10))?;
///
/// let feasible = kernel.feasible_goals(current_state);
/// let kappa = kernel.query_feasibility(GoalId(0), current_state);
/// ```
pub struct GoalKernel<B: Backend> {
    /// STOKs indexed by goal ID
    stoks: HashMap<GoalId, STOKKernel<B>>,

    /// SOKs (precomputed from STOKs for fast queries)
    soks: HashMap<GoalId, StateOptionKernel<B>>,

    /// Goal metadata
    goal_info: HashMap<GoalId, GoalInfo>,

    /// Shared state space size
    n_states: usize,

    /// Device
    device: B::Device,
}

impl<B: Backend> GoalKernel<B> {
    /// Create empty goal kernel
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states in state space
    /// * `device` - Device for tensor operations
    pub fn new(n_states: usize, device: B::Device) -> Self {
        Self {
            stoks: HashMap::new(),
            soks: HashMap::new(),
            goal_info: HashMap::new(),
            n_states,
            device,
        }
    }

    /// Add a goal with its STOK
    ///
    /// # Arguments
    ///
    /// * `id` - Unique goal identifier
    /// * `stok` - STOK kernel for this goal
    /// * `name` - Human-readable name
    /// * `target_state` - Optional target state index
    ///
    /// # Returns
    ///
    /// Ok if added successfully
    ///
    /// # Errors
    ///
    /// Returns error if state spaces don't match
    pub fn add_goal(
        &mut self,
        id: GoalId,
        stok: STOKKernel<B>,
        name: impl Into<String>,
        target_state: Option<usize>,
    ) -> Result<(), StokError> {
        // Validate state space compatibility
        if stok.n_states() != self.n_states {
            return Err(StokError::DimensionMismatch {
                expected: vec![self.n_states],
                got: vec![stok.n_states()],
            });
        }

        // Precompute SOK for fast queries
        let sok = StateOptionKernel::from_stok(&stok);

        // Store metadata
        let info = GoalInfo {
            id,
            name: name.into(),
            target_state,
            max_time: stok.max_time(),
        };

        self.stoks.insert(id, stok);
        self.soks.insert(id, sok);
        self.goal_info.insert(id, info);

        Ok(())
    }

    /// Get STOK for a goal
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    ///
    /// # Returns
    ///
    /// Reference to STOK if it exists
    pub fn get_stok(&self, goal: GoalId) -> Option<&STOKKernel<B>> {
        self.stoks.get(&goal)
    }

    /// Get SOK for a goal
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    ///
    /// # Returns
    ///
    /// Reference to SOK if it exists
    pub fn get_sok(&self, goal: GoalId) -> Option<&StateOptionKernel<B>> {
        self.soks.get(&goal)
    }

    /// Get goal info
    pub fn get_info(&self, goal: GoalId) -> Option<&GoalInfo> {
        self.goal_info.get(&goal)
    }

    /// Check if goal exists in kernel
    pub fn has_goal(&self, goal: GoalId) -> bool {
        self.stoks.contains_key(&goal)
    }

    /// Get number of goals in kernel
    pub fn n_goals(&self) -> usize {
        self.stoks.len()
    }

    /// Get device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get all goal IDs
    pub fn goal_ids(&self) -> Vec<GoalId> {
        self.stoks.keys().copied().collect()
    }

    /// Query feasibility of reaching goal from state
    ///
    /// Returns κ_g(x)
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID to query
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Feasibility value, or 0.0 if goal doesn't exist
    pub fn query_feasibility(&self, goal: GoalId, state: usize) -> f32 {
        self.stoks
            .get(&goal)
            .map(|stok| {
                stok.kappa
                    .clone()
                    .slice([state..(state + 1)])
                    .into_scalar()
                    .elem()
            })
            .unwrap_or(0.0)
    }

    /// Batch query: feasibility for all goals from state
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// HashMap mapping goal IDs to feasibility values
    pub fn query_all_feasibilities(&self, state: usize) -> HashMap<GoalId, f32> {
        self.stoks
            .iter()
            .map(|(id, stok)| {
                let kappa: f32 = stok
                    .kappa
                    .clone()
                    .slice([state..(state + 1)])
                    .into_scalar()
                    .elem();
                (*id, kappa)
            })
            .collect()
    }

    /// Get feasible goals from a state
    ///
    /// Returns all goals with κ_g(x) > 0
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Vector of feasible goal IDs
    pub fn feasible_goals(&self, state: usize) -> Vec<GoalId> {
        let mut goal_ids: Vec<GoalId> = self.stoks.keys().copied().collect();
        goal_ids.sort();

        goal_ids
            .into_iter()
            .filter(|id| self.query_feasibility(*id, state) > 0.0)
            .collect()
    }

    /// Check if a specific goal is feasible from state
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// true if κ_g(x) > 0
    pub fn is_feasible(&self, goal: GoalId, state: usize) -> bool {
        self.query_feasibility(goal, state) > 0.0
    }

    /// Query expected termination time for goal from state
    ///
    /// Computes E[t_f | x] = Σ_{x_f, t_f} t_f · η(x_f, t_f | x)
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Expected time, or None if goal doesn't exist
    pub fn query_expected_time(&self, goal: GoalId, state: usize) -> Option<f32> {
        let stok = self.stoks.get(&goal)?;

        // E[t_f | x] = Σ_{x_f, t_f} t_f * η(x_f, t_f | x)
        let eta = stok.combined_stok(); // [S, S, T]
        let t_max = stok.max_time();

        let mut expected_time = 0.0f32;
        for t in 0..t_max {
            let eta_t: f32 = eta
                .clone()
                .slice([state..(state + 1), 0..self.n_states, t..(t + 1)])
                .sum()
                .into_scalar()
                .elem();
            expected_time += (t as f32) * eta_t;
        }

        Some(expected_time)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use burn::prelude::Tensor;

    #[test]
    fn test_goal_kernel_creation() {
        let device = default_device();
        let kernel: GoalKernel<DefaultBackend> = GoalKernel::new(10, device);

        assert_eq!(kernel.n_goals(), 0);
        assert_eq!(kernel.n_states, 10);
    }

    #[test]
    fn test_add_goal() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 10, &device);
        kernel
            .add_goal(GoalId(0), stok, "test_goal", Some(4))
            .unwrap();

        assert_eq!(kernel.n_goals(), 1);
        assert!(kernel.has_goal(GoalId(0)));
        assert!(!kernel.has_goal(GoalId(1)));
    }

    #[test]
    fn test_add_goal_dimension_mismatch() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        // Create STOK with different state space
        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(10, 10, &device);
        let result = kernel.add_goal(GoalId(0), stok, "mismatched", None);

        assert!(result.is_err());
        match result {
            Err(StokError::DimensionMismatch { .. }) => {} // Expected
            _ => panic!("Should return DimensionMismatch error"),
        }
    }

    #[test]
    fn test_query_feasibility() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(3, device.clone());

        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok.kappa = Tensor::from_floats([0.5, 0.8, 1.0], &device);

        kernel.add_goal(GoalId(0), stok, "goal", None).unwrap();

        assert!((kernel.query_feasibility(GoalId(0), 0) - 0.5).abs() < 1e-5);
        assert!((kernel.query_feasibility(GoalId(0), 1) - 0.8).abs() < 1e-5);
        assert!((kernel.query_feasibility(GoalId(0), 2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_feasible_goals() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(3, device.clone());

        // Goal 0: feasible from states 1, 2
        let mut stok0: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok0.kappa = Tensor::from_floats([0.0, 0.8, 1.0], &device);
        kernel.add_goal(GoalId(0), stok0, "goal0", None).unwrap();

        // Goal 1: feasible from all states
        let mut stok1: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok1.kappa = Tensor::from_floats([1.0, 1.0, 1.0], &device);
        kernel.add_goal(GoalId(1), stok1, "goal1", None).unwrap();

        // From state 0: only goal 1 is feasible
        let feasible_from_0 = kernel.feasible_goals(0);
        assert_eq!(feasible_from_0.len(), 1);
        assert!(feasible_from_0.contains(&GoalId(1)));

        // From state 1: both goals feasible
        let feasible_from_1 = kernel.feasible_goals(1);
        assert_eq!(feasible_from_1.len(), 2);
    }
}
