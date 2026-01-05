//! Task MDP definition for STOK computation.
//!
//! A Task MDP extends standard MDPs with explicit goal and constraint functions,
//! enabling feasibility-aware planning per Ringstrom & Schrater (2025).
//!
//! # Mathematical Definition
//!
//! A Task MDP is defined as M = ⟨X, A, P, f_g, f_c⟩ where:
//! - X: State space (|X| = n_states)
//! - A: Action space (|A| = n_actions)
//! - P(x'|x,a): Transition dynamics (row-stochastic)
//! - f_g(x,a) ∈ [0,1]: Goal satisfaction probability
//! - f_c(x,a) ∈ [0,1]: Constraint satisfaction probability
//!
//! # Derived Functions
//!
//! From f_g and f_c, we derive (used in κ-OKBE, Equation [7]):
//! - f_1 = f_g × f_c: Achievement function (goal AND constraint satisfied)
//! - f_2 = (1 - f_g) × f_c: Continuation function (NOT goal AND constraint)
//!
//! # Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Section 2: "Options Kernels and OKBEs"

use crate::types::{MDPDimensions, StokError, DEFAULT_PROBABILITY_TOLERANCE};
use burn::prelude::*;

/// Task MDP with goal and constraint functions.
///
/// Core data structure for STOK computation. Stores transition dynamics
/// and goal/constraint specifications, with derived functions computed
/// automatically.
///
/// # Tensor Conventions
///
/// - `transition[x, a, x']` = P(x' | x, a) - probability of reaching x' from x via action a
/// - `goal_fn[x, a]` = f_g(x, a) - probability goal is satisfied at state-action
/// - `constraint_fn[x, a]` = f_c(x, a) - probability constraint is NOT violated (1 = safe)
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::prelude::*;
///
/// let device = default_device();
/// let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
/// mdp.validate().expect("MDP should be valid");
/// ```
#[derive(Debug)]
pub struct TaskMDP<B: Backend> {
    /// Transition dynamics P(x'|x,a), shape [n_states, n_actions, n_states]
    ///
    /// Must be row-stochastic: Σ_{x'} P(x'|x,a) = 1 for all (x, a)
    pub transition: Tensor<B, 3>,

    /// Goal function f_g(x,a), shape [n_states, n_actions]
    ///
    /// Values in [0, 1]. Probability that goal is achieved at (x, a).
    pub goal_fn: Tensor<B, 2>,

    /// Constraint function f_c(x,a), shape [n_states, n_actions]
    ///
    /// Values in [0, 1]. Probability that constraint is NOT violated.
    /// f_c = 1 means safe, f_c = 0 means definite constraint violation.
    pub constraint_fn: Tensor<B, 2>,

    /// Achievement function f_1 = f_g × f_c, shape [n_states, n_actions]
    ///
    /// Probability of achieving goal AND satisfying constraint.
    /// Used in κ-OKBE (Equation [7]).
    pub f1: Tensor<B, 2>,

    /// Continuation function f_2 = (1 - f_g) × f_c, shape [n_states, n_actions]
    ///
    /// Probability of NOT achieving goal AND satisfying constraint.
    /// Used in κ-OKBE (Equation [7]).
    pub f2: Tensor<B, 2>,

    /// Cached dimension information
    pub dims: MDPDimensions,
}

impl<B: Backend> TaskMDP<B> {
    // ========================================================================
    // Constructor
    // ========================================================================

    /// Create a new TaskMDP with validation.
    ///
    /// Validates tensor shapes and computes derived functions f_1 and f_2.
    ///
    /// # Arguments
    ///
    /// * `transition` - Transition tensor P(x'|x,a), shape [S, A, S]
    /// * `goal_fn` - Goal function f_g(x,a), shape [S, A]
    /// * `constraint_fn` - Constraint function f_c(x,a), shape [S, A]
    /// * `max_time` - Maximum time horizon for STOK computation
    ///
    /// # Returns
    ///
    /// `Ok(TaskMDP)` if shapes are valid, `Err(StokError)` otherwise.
    ///
    /// # Errors
    ///
    /// - `StokError::DimensionMismatch` if tensor shapes are inconsistent
    pub fn new(
        transition: Tensor<B, 3>,
        goal_fn: Tensor<B, 2>,
        constraint_fn: Tensor<B, 2>,
        max_time: usize,
    ) -> Result<Self, StokError> {
        // Shape validation
        let t_dims = transition.dims();
        let g_dims = goal_fn.dims();
        let c_dims = constraint_fn.dims();

        // Transition should be [S, A, S]
        if t_dims[0] != t_dims[2] {
            return Err(StokError::DimensionMismatch {
                expected: vec![t_dims[0], t_dims[1], t_dims[0]],
                got: t_dims.to_vec(),
            });
        }

        let n_states = t_dims[0];
        let n_actions = t_dims[1];

        // Goal and constraint should be [S, A]
        if g_dims != [n_states, n_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_states, n_actions],
                got: g_dims.to_vec(),
            });
        }
        if c_dims != [n_states, n_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_states, n_actions],
                got: c_dims.to_vec(),
            });
        }

        // Compute derived functions (Equation [7] components)
        // f_1 = f_g × f_c (achievement: goal AND constraint satisfied)
        let f1 = goal_fn.clone() * constraint_fn.clone();

        // f_2 = (1 - f_g) × f_c (continuation: NOT goal AND constraint satisfied)
        let device = goal_fn.device();
        let ones: Tensor<B, 2> = Tensor::ones([n_states, n_actions], &device);
        let f2 = (ones - goal_fn.clone()) * constraint_fn.clone();

        let dims = MDPDimensions::new(n_states, n_actions, max_time);
        dims.validate()?;

        Ok(Self {
            transition,
            goal_fn,
            constraint_fn,
            f1,
            f2,
            dims,
        })
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Number of states in the MDP.
    pub fn n_states(&self) -> usize {
        self.dims.n_states
    }

    /// Number of actions available at each state.
    pub fn n_actions(&self) -> usize {
        self.dims.n_actions
    }

    /// Maximum time horizon for STOK computation.
    pub fn max_time(&self) -> usize {
        self.dims.max_time
    }

    /// Reference to dimension specification.
    pub fn dims(&self) -> &MDPDimensions {
        &self.dims
    }

    /// Get device from transition tensor.
    pub fn device(&self) -> B::Device {
        self.transition.device()
    }

    // ========================================================================
    // Validation Methods
    // ========================================================================

    /// Validate transition tensor is row-stochastic.
    ///
    /// Checks that Σ_{x'} P(x'|x,a) = 1 for all (x, a) within tolerance.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from 1.0
    ///
    /// # Returns
    ///
    /// `Ok(())` if row-stochastic, `Err(StokError::NotNormalized)` otherwise.
    pub fn validate_stochastic(&self, tolerance: f32) -> Result<(), StokError> {
        // Sum over last dimension (x') and reshape to [S, A]
        // Note: Using reshape instead of squeeze to handle n_states=1 edge case
        let n_states = self.dims.n_states;
        let n_actions = self.dims.n_actions;
        let row_sums: Tensor<B, 2> = self
            .transition
            .clone()
            .sum_dim(2)
            .reshape([n_states, n_actions]);
        let device = row_sums.device();
        let ones: Tensor<B, 2> = Tensor::ones(row_sums.dims(), &device);
        let diff = (row_sums - ones).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        if max_diff > tolerance {
            return Err(StokError::NotNormalized {
                sum: 1.0 + max_diff, // Approximate actual sum
                expected: 1.0,
                tolerance,
            });
        }
        Ok(())
    }

    /// Validate all probability values are in [0, 1].
    ///
    /// Checks goal_fn and constraint_fn for valid probability bounds.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all values valid, `Err(StokError::InvalidProbability)` otherwise.
    pub fn validate_probability_bounds(&self) -> Result<(), StokError> {
        // Check goal function
        let g_min: f32 = self.goal_fn.clone().min().into_scalar().elem();
        let g_max: f32 = self.goal_fn.clone().max().into_scalar().elem();
        if g_min < 0.0 || g_max > 1.0 {
            return Err(StokError::InvalidProbability {
                value: if g_min < 0.0 { g_min } else { g_max },
                context: "goal_fn".into(),
            });
        }

        // Check constraint function
        let c_min: f32 = self.constraint_fn.clone().min().into_scalar().elem();
        let c_max: f32 = self.constraint_fn.clone().max().into_scalar().elem();
        if c_min < 0.0 || c_max > 1.0 {
            return Err(StokError::InvalidProbability {
                value: if c_min < 0.0 { c_min } else { c_max },
                context: "constraint_fn".into(),
            });
        }

        Ok(())
    }

    /// Run all validation checks.
    ///
    /// Validates both stochastic property and probability bounds.
    ///
    /// # Returns
    ///
    /// `Ok(())` if MDP is valid, first `Err(StokError)` encountered otherwise.
    pub fn validate(&self) -> Result<(), StokError> {
        self.validate_stochastic(DEFAULT_PROBABILITY_TOLERANCE)?;
        self.validate_probability_bounds()?;
        Ok(())
    }

    // ========================================================================
    // Builder Methods for Test MDPs
    // ========================================================================

    /// Create a simple deterministic chain MDP.
    ///
    /// ```text
    /// States: 0 -- 1 -- 2 -- ... -- (n-1)
    ///         ↑                      ↑
    ///       start                  goal
    /// ```
    ///
    /// # Actions
    ///
    /// - 0: move left (or stay at boundary)
    /// - 1: move right (or stay at boundary)
    /// - 2: stay in place
    ///
    /// # Goal/Constraints
    ///
    /// - Goal: state (n-1) with any action (f_g = 1.0)
    /// - Constraints: none (all f_c = 1.0)
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states in the chain
    /// * `max_time` - Maximum time horizon
    /// * `device` - Backend device for tensor allocation
    pub fn simple_chain(n_states: usize, max_time: usize, device: &B::Device) -> Self {
        let n_actions = 3; // left, right, stay

        // Build transition tensor P[x, a, x']
        let mut trans_data = vec![0.0f32; n_states * n_actions * n_states];

        for x in 0..n_states {
            for a in 0..n_actions {
                let x_next = match a {
                    0 => x.saturating_sub(1),       // left
                    1 => (x + 1).min(n_states - 1), // right
                    _ => x,                         // stay
                };
                let idx = x * n_actions * n_states + a * n_states + x_next;
                trans_data[idx] = 1.0;
            }
        }

        let transition: Tensor<B, 3> = Tensor::<B, 1>::from_floats(trans_data.as_slice(), device)
            .reshape([n_states, n_actions, n_states]);

        // Goal: only at final state (f_g = 1.0 at state n-1)
        let mut goal_data = vec![0.0f32; n_states * n_actions];
        for a in 0..n_actions {
            let idx = (n_states - 1) * n_actions + a;
            goal_data[idx] = 1.0;
        }
        let goal_fn: Tensor<B, 2> = Tensor::<B, 1>::from_floats(goal_data.as_slice(), device)
            .reshape([n_states, n_actions]);

        // No constraints (all f_c = 1.0)
        let constraint_fn: Tensor<B, 2> = Tensor::ones([n_states, n_actions], device);

        Self::new(transition, goal_fn, constraint_fn, max_time)
            .expect("simple_chain should always produce valid MDP")
    }

    /// Create chain MDP with a constraint-violating "fire" state.
    ///
    /// Same as `simple_chain` but with f_c = 0 at the specified fire state.
    /// Used for testing constraint handling in feasibility iteration.
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states in the chain
    /// * `fire_state` - State index where constraint is violated (f_c = 0)
    /// * `max_time` - Maximum time horizon
    /// * `device` - Backend device
    ///
    /// # Panics
    ///
    /// Panics if `fire_state >= n_states`.
    pub fn constrained_chain(
        n_states: usize,
        fire_state: usize,
        max_time: usize,
        device: &B::Device,
    ) -> Self {
        assert!(fire_state < n_states, "fire_state must be < n_states");

        let n_actions = 3;

        // Start with simple chain
        let base = Self::simple_chain(n_states, max_time, device);

        // Set constraint to 0 at fire state
        let mut constraint_data = vec![1.0f32; n_states * n_actions];
        for a in 0..n_actions {
            constraint_data[fire_state * n_actions + a] = 0.0;
        }
        let constraint_fn: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(constraint_data.as_slice(), device)
                .reshape([n_states, n_actions]);

        Self::new(base.transition, base.goal_fn, constraint_fn, max_time)
            .expect("constrained_chain should produce valid MDP")
    }

    /// Create chain MDP with stochastic transitions.
    ///
    /// Similar to `simple_chain` but with probabilistic transitions:
    /// - `success_prob` chance of intended direction
    /// - `(1 - success_prob) / 2` chance of adjacent outcomes
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states
    /// * `success_prob` - Probability of moving in intended direction (0.0 to 1.0)
    /// * `max_time` - Maximum time horizon
    /// * `device` - Backend device
    pub fn stochastic_chain(
        n_states: usize,
        success_prob: f32,
        max_time: usize,
        device: &B::Device,
    ) -> Self {
        let n_actions = 3;
        let fail_prob = (1.0 - success_prob) / 2.0;

        let mut trans_data = vec![0.0f32; n_states * n_actions * n_states];

        for x in 0..n_states {
            for a in 0..n_actions {
                let intended = match a {
                    0 => x.saturating_sub(1),
                    1 => (x + 1).min(n_states - 1),
                    _ => x,
                };

                // Intended outcome
                let idx = x * n_actions * n_states + a * n_states + intended;
                trans_data[idx] += success_prob;

                // Possible deviations (for non-stay actions)
                if a != 2 {
                    let left = x.saturating_sub(1);
                    let right = (x + 1).min(n_states - 1);

                    let left_idx = x * n_actions * n_states + a * n_states + left;
                    let right_idx = x * n_actions * n_states + a * n_states + right;

                    trans_data[left_idx] += fail_prob;
                    trans_data[right_idx] += fail_prob;
                } else {
                    // Stay action: all probability on current state
                    trans_data[idx] = 1.0;
                }
            }
        }

        let transition: Tensor<B, 3> = Tensor::<B, 1>::from_floats(trans_data.as_slice(), device)
            .reshape([n_states, n_actions, n_states]);

        // Goal: only at final state
        let mut goal_data = vec![0.0f32; n_states * n_actions];
        for a in 0..n_actions {
            goal_data[(n_states - 1) * n_actions + a] = 1.0;
        }
        let goal_fn: Tensor<B, 2> = Tensor::<B, 1>::from_floats(goal_data.as_slice(), device)
            .reshape([n_states, n_actions]);

        let constraint_fn: Tensor<B, 2> = Tensor::ones([n_states, n_actions], device);

        Self::new(transition, goal_fn, constraint_fn, max_time)
            .expect("stochastic_chain should produce valid MDP")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};

    // ========================================================================
    // Positive Validation Tests (valid MDPs)
    // ========================================================================

    #[test]
    fn test_simple_chain_creation() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);

        assert_eq!(mdp.n_states(), 10);
        assert_eq!(mdp.n_actions(), 3);
        assert_eq!(mdp.transition.dims(), [10, 3, 10]);
    }

    #[test]
    fn test_simple_chain_validation() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(10, 20, &device);
        assert!(mdp.validate().is_ok());
    }

    #[test]
    fn test_constrained_chain() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::constrained_chain(10, 4, 20, &device);

        // Extract constraint values at fire state
        let constraint_data: Vec<f32> = mdp.constraint_fn.clone().into_data().to_vec().unwrap();

        // Fire state (4) should have f_c = 0 for all actions
        for a in 0..3 {
            assert_eq!(constraint_data[4 * 3 + a], 0.0);
        }
    }

    #[test]
    fn test_stochastic_chain_normalization() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::stochastic_chain(10, 0.8, 20, &device);
        assert!(mdp.validate_stochastic(1e-5).is_ok());
    }

    #[test]
    fn test_derived_functions() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 10, &device);

        let f1_data: Vec<f32> = mdp.f1.clone().into_data().to_vec().unwrap();
        let f2_data: Vec<f32> = mdp.f2.clone().into_data().to_vec().unwrap();

        // At goal state (4), f_1 = 1.0, f_2 = 0.0
        // At other states, f_1 = 0.0, f_2 = 1.0
        for a in 0..3 {
            // Goal state
            assert_eq!(f1_data[4 * 3 + a], 1.0);
            assert_eq!(f2_data[4 * 3 + a], 0.0);
            // Non-goal state
            assert_eq!(f1_data[0 * 3 + a], 0.0);
            assert_eq!(f2_data[0 * 3 + a], 1.0);
        }
    }

    // ========================================================================
    // Negative Validation Tests (invalid MDPs should fail)
    // ========================================================================

    #[test]
    fn test_invalid_mdp_non_stochastic_transitions() {
        let device = default_device();

        // Create transition tensor that doesn't sum to 1
        // All zeros means sum = 0, not 1
        let bad_trans: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 3], &device);
        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let result = TaskMDP::<DefaultBackend>::new(bad_trans, goal, constraint, 10);
        // Should succeed in construction but fail validation
        assert!(result.is_ok());
        let mdp = result.unwrap();

        // validate_stochastic should fail
        let validation = mdp.validate_stochastic(1e-5);
        assert!(validation.is_err());

        match validation.unwrap_err() {
            StokError::NotNormalized { expected, .. } => {
                assert_eq!(expected, 1.0);
            }
            _ => panic!("Expected NotNormalized error"),
        }
    }

    #[test]
    fn test_invalid_mdp_dimension_mismatch_transition() {
        let device = default_device();

        // Transition with inconsistent first and third dimensions
        // [3, 2, 4] instead of [3, 2, 3]
        let bad_trans: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 4], &device);
        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let result = TaskMDP::<DefaultBackend>::new(bad_trans, goal, constraint, 10);
        assert!(result.is_err());

        match result.unwrap_err() {
            StokError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, vec![3, 2, 3]);
                assert_eq!(got, vec![3, 2, 4]);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_invalid_mdp_dimension_mismatch_goal() {
        let device = default_device();

        // Goal with wrong shape [4, 2] instead of [3, 2]
        let trans: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 3], &device);
        let bad_goal: Tensor<DefaultBackend, 2> = Tensor::zeros([4, 2], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let result = TaskMDP::<DefaultBackend>::new(trans, bad_goal, constraint, 10);
        assert!(result.is_err());

        match result.unwrap_err() {
            StokError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, vec![3, 2]);
                assert_eq!(got, vec![4, 2]);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_invalid_mdp_dimension_mismatch_constraint() {
        let device = default_device();

        // Constraint with wrong shape [3, 4] instead of [3, 2]
        let trans: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 3], &device);
        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);
        let bad_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 4], &device);

        let result = TaskMDP::<DefaultBackend>::new(trans, goal, bad_constraint, 10);
        assert!(result.is_err());

        match result.unwrap_err() {
            StokError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, vec![3, 2]);
                assert_eq!(got, vec![3, 4]);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_invalid_mdp_goal_out_of_bounds_negative() {
        let device = default_device();

        // Create valid structure but with negative goal values
        let trans_data = vec![1.0f32; 3 * 2 * 3]; // Valid stochastic (uniform)
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(trans_data.as_slice(), &device)
                .reshape([3, 2, 3])
                / 3.0; // Normalize to sum to 1

        // Goal with negative value
        let mut goal_data = vec![0.0f32; 3 * 2];
        goal_data[0] = -0.5; // Invalid!
        let goal: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(goal_data.as_slice(), &device).reshape([3, 2]);

        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let result = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10);
        assert!(result.is_ok());

        let mdp = result.unwrap();
        let validation = mdp.validate_probability_bounds();
        assert!(validation.is_err());

        match validation.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value < 0.0);
                assert_eq!(context, "goal_fn");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_invalid_mdp_goal_out_of_bounds_over_one() {
        let device = default_device();

        let trans_data = vec![1.0f32; 3 * 2 * 3];
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(trans_data.as_slice(), &device)
                .reshape([3, 2, 3])
                / 3.0;

        // Goal with value > 1.0
        let mut goal_data = vec![0.5f32; 3 * 2];
        goal_data[0] = 1.5; // Invalid!
        let goal: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(goal_data.as_slice(), &device).reshape([3, 2]);

        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let mdp = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).unwrap();
        let validation = mdp.validate_probability_bounds();
        assert!(validation.is_err());

        match validation.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value > 1.0);
                assert_eq!(context, "goal_fn");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_invalid_mdp_constraint_out_of_bounds() {
        let device = default_device();

        let trans_data = vec![1.0f32; 3 * 2 * 3];
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(trans_data.as_slice(), &device)
                .reshape([3, 2, 3])
                / 3.0;

        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);

        // Constraint with negative value
        let mut constraint_data = vec![1.0f32; 3 * 2];
        constraint_data[0] = -0.1; // Invalid!
        let constraint: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(constraint_data.as_slice(), &device)
                .reshape([3, 2]);

        let mdp = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).unwrap();
        let validation = mdp.validate_probability_bounds();
        assert!(validation.is_err());

        match validation.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value < 0.0);
                assert_eq!(context, "constraint_fn");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_invalid_mdp_zero_max_time() {
        let device = default_device();

        let trans_data = vec![1.0f32; 3 * 2 * 3];
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(trans_data.as_slice(), &device)
                .reshape([3, 2, 3])
                / 3.0;

        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        // Zero max_time should fail
        let result = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 0);
        assert!(result.is_err());

        match result.unwrap_err() {
            StokError::InvalidDimension { name, value, .. } => {
                assert_eq!(name, "max_time");
                assert_eq!(value, 0);
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_validate_catches_all_errors() {
        let device = default_device();

        // Create MDP with multiple issues: non-stochastic + invalid goal
        let bad_trans: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 3], &device);

        let mut goal_data = vec![0.0f32; 3 * 2];
        goal_data[0] = -0.5;
        let bad_goal: Tensor<DefaultBackend, 2> =
            Tensor::<DefaultBackend, 1>::from_floats(goal_data.as_slice(), &device).reshape([3, 2]);

        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let mdp = TaskMDP::<DefaultBackend>::new(bad_trans, bad_goal, constraint, 10).unwrap();

        // Full validate should fail (catches first error - stochastic)
        let result = mdp.validate();
        assert!(result.is_err());
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_single_state_mdp() {
        let device = default_device();

        // Minimal valid MDP: 1 state, 1 action
        let trans: Tensor<DefaultBackend, 3> = Tensor::ones([1, 1, 1], &device);
        let goal: Tensor<DefaultBackend, 2> = Tensor::ones([1, 1], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([1, 1], &device);

        let mdp = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 5).unwrap();
        assert!(mdp.validate().is_ok());
        assert_eq!(mdp.n_states(), 1);
        assert_eq!(mdp.n_actions(), 1);
    }

    #[test]
    fn test_all_constraint_violated() {
        let device = default_device();

        // MDP where all states violate constraints (f_c = 0 everywhere)
        let trans_data = vec![1.0f32; 3 * 2 * 3];
        let trans: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(trans_data.as_slice(), &device)
                .reshape([3, 2, 3])
                / 3.0;

        let goal: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device);
        let constraint: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device); // All violated!

        let mdp = TaskMDP::<DefaultBackend>::new(trans, goal, constraint, 10).unwrap();
        assert!(mdp.validate().is_ok()); // Structure is valid

        // f_1 and f_2 should both be zero everywhere
        let f1_data: Vec<f32> = mdp.f1.into_data().to_vec().unwrap();
        let f2_data: Vec<f32> = mdp.f2.into_data().to_vec().unwrap();
        assert!(f1_data.iter().all(|&v| v == 0.0));
        assert!(f2_data.iter().all(|&v| v == 0.0));
    }
}
