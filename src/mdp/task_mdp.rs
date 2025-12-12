//! Task MDP definition for STOK computation.
//!
//! A Task MDP extends standard MDPs with explicit goal and constraint functions,
//! enabling feasibility-aware planning per Ringstrom & Schrater (2025).
//!
//! Key components:
//! - P(x'|x,a): Transition dynamics
//! - f_g(x,a): Goal satisfaction probability  
//! - f_c(x,a): Constraint satisfaction probability
//! - f_1 = f_g * f_c: Achievement function (goal AND constraint)
//! - f_2 = (1-f_g) * f_c: Continuation function (NOT goal AND constraint)

use burn::prelude::*;
use crate::types::{MDPDimensions, StokError, DEFAULT_PROBABILITY_TOLERANCE};

/// Task MDP with goal and constraint functions.
///
/// Tensor index conventions:
/// - `transition[x, a, x']` = P(x' | x, a)
/// - `goal_fn[x, a]` = f_g(x, a)
/// - `constraint_fn[x, a]` = f_c(x, a)
#[derive(Debug)]
pub struct TaskMDP<B: Backend> {
    /// Transition dynamics P(x'|x,a), shape [S, A, S]
    /// Row-stochastic: sums to 1 over last dimension
    pub transition: Tensor<B, 3>,

    /// Goal function f_g(x,a) ∈ [0,1], shape [S, A]
    /// Probability of achieving goal at (x, a)
    pub goal_fn: Tensor<B, 2>,

    /// Constraint function f_c(x,a) ∈ [0,1], shape [S, A]
    /// Probability of NOT violating constraint (1 = safe, 0 = violation)
    pub constraint_fn: Tensor<B, 2>,

    /// Achievement function f_1 = f_g * f_c, shape [S, A]
    /// Probability of achieving goal AND satisfying constraint
    pub f1: Tensor<B, 2>,

    /// Continuation function f_2 = (1 - f_g) * f_c, shape [S, A]
    /// Probability of NOT achieving goal AND satisfying constraint
    pub f2: Tensor<B, 2>,

    /// Cached dimensions
    pub dims: MDPDimensions,
}

impl<B: Backend> TaskMDP<B> {
    /// Create a new TaskMDP from components.
    ///
    /// Validates shapes and computes derived functions f_1, f_2.
    pub fn new(
        transition: Tensor<B, 3>,
        goal_fn: Tensor<B, 2>,
        constraint_fn: Tensor<B, 2>,
        max_time: usize,
    ) -> Result<Self, StokError> {
        // Extract and validate dimensions
        let t_dims = transition.dims();
        let g_dims = goal_fn.dims();
        let c_dims = constraint_fn.dims();

        // Transition should be [S, A, S]
        if t_dims.len() != 3 || t_dims[0] != t_dims[2] {
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

        // Compute derived functions
        // f_1 = f_g * f_c (achievement: goal AND constraint satisfied)
        let f1 = goal_fn.clone() * constraint_fn.clone();

        // f_2 = (1 - f_g) * f_c (continuation: NOT goal AND constraint satisfied)
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

    pub fn n_states(&self) -> usize {
        self.dims.n_states
    }

    pub fn n_actions(&self) -> usize {
        self.dims.n_actions
    }

    pub fn max_time(&self) -> usize {
        self.dims.max_time
    }

    pub fn dims(&self) -> &MDPDimensions {
        &self.dims
    }

    /// Get device from transition tensor
    pub fn device(&self) -> B::Device {
        self.transition.device()
    }

    // ========================================================================
    // Validation Methods
    // ========================================================================

    /// Validate transition tensor is row-stochastic (sums to 1 over x')
    pub fn validate_stochastic(&self, tolerance: f32) -> Result<(), StokError> {
        // Sum over last dimension (x') and squeeze to get [S, A]
        let row_sums: Tensor<B, 2> = self.transition.clone().sum_dim(2).squeeze::<2>();
        let device = row_sums.device();
        let ones: Tensor<B, 2> = Tensor::ones(row_sums.dims(), &device);
        let diff = (row_sums - ones).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        if max_diff > tolerance {
            return Err(StokError::NotNormalized {
                sum: max_diff,
                expected: 1.0,
                tolerance,
            });
        }
        Ok(())
    }

    /// Validate all probability values are in [0, 1]
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

    /// Run all validation checks
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
    /// States: 0 -- 1 -- 2 -- ... -- (n-1)
    /// Actions: 0=left, 1=right, 2=stay
    /// Goal: state (n-1)
    /// No constraints (all f_c = 1)
    pub fn simple_chain(n_states: usize, max_time: usize, device: &B::Device) -> Self {
        let n_actions = 3; // left, right, stay

        // Build transition tensor
        let mut trans_data = vec![0.0f32; n_states * n_actions * n_states];

        for x in 0..n_states {
            for a in 0..n_actions {
                let x_next = match a {
                    0 => x.saturating_sub(1),           // left
                    1 => (x + 1).min(n_states - 1),    // right
                    _ => x,                             // stay
                };
                let idx = x * n_actions * n_states + a * n_states + x_next;
                trans_data[idx] = 1.0;
            }
        }

        let transition: Tensor<B, 3> = Tensor::<B, 1>::from_floats(
            trans_data.as_slice(),
            device,
        ).reshape([n_states, n_actions, n_states]);

        // Goal: only at final state
        let mut goal_data = vec![0.0f32; n_states * n_actions];
        for a in 0..n_actions {
            let idx = (n_states - 1) * n_actions + a;
            goal_data[idx] = 1.0;
        }
        let goal_fn: Tensor<B, 2> = Tensor::<B, 1>::from_floats(goal_data.as_slice(), device)
            .reshape([n_states, n_actions]);

        // No constraints
        let constraint_fn: Tensor<B, 2> = Tensor::ones([n_states, n_actions], device);

        Self::new(transition, goal_fn, constraint_fn, max_time)
            .expect("simple_chain should always produce valid MDP")
    }

    /// Create chain MDP with a constraint-violating "fire" state.
    ///
    /// All actions at fire_state have f_c = 0.
    pub fn constrained_chain(
        n_states: usize,
        fire_state: usize,
        max_time: usize,
        device: &B::Device,
    ) -> Self {
        let n_actions = 3;

        // Start with simple chain
        let base = Self::simple_chain(n_states, max_time, device);

        // Set constraint to 0 at fire state
        let mut constraint_data = vec![1.0f32; n_states * n_actions];
        for a in 0..n_actions {
            let idx = fire_state * n_actions + a;
            constraint_data[idx] = 0.0;
        }
        let constraint_fn: Tensor<B, 2> = Tensor::<B, 1>::from_floats(constraint_data.as_slice(), device)
            .reshape([n_states, n_actions]);

        // Recompute f1, f2
        let f1 = base.goal_fn.clone() * constraint_fn.clone();
        let ones: Tensor<B, 2> = Tensor::ones([n_states, n_actions], device);
        let f2 = (ones - base.goal_fn.clone()) * constraint_fn.clone();

        Self {
            transition: base.transition,
            goal_fn: base.goal_fn,
            constraint_fn,
            f1,
            f2,
            dims: base.dims,
        }
    }

    /// Create chain MDP with stochastic transitions.
    ///
    /// `success_prob` chance of moving in intended direction,
    /// remaining probability split between staying and opposite direction.
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
                let (primary, secondary) = match a {
                    0 => (x.saturating_sub(1), (x + 1).min(n_states - 1)), // left
                    1 => ((x + 1).min(n_states - 1), x.saturating_sub(1)), // right  
                    _ => (x, x),                                            // stay
                };

                let base_idx = x * n_actions * n_states + a * n_states;

                if a == 2 {
                    // Stay action is deterministic
                    trans_data[base_idx + x] = 1.0;
                } else {
                    trans_data[base_idx + primary] += success_prob;
                    trans_data[base_idx + x] += fail_prob;
                    trans_data[base_idx + secondary] += fail_prob;
                }
            }
        }

        let transition: Tensor<B, 3> = Tensor::<B, 1>::from_floats(trans_data.as_slice(), device)
            .reshape([n_states, n_actions, n_states]);

        // Goal at final state
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
    use crate::backend::{DefaultBackend, default_device};

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
        let constraint_data: Vec<f32> = mdp.constraint_fn.clone()
            .into_data()
            .to_vec()
            .unwrap();
        
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
}