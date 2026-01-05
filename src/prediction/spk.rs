//! # State Prediction Kernel (SPK)
//!
//! Predicts final state distributions under default (uncontrolled) dynamics.
//!
//! Used in high-dimensional STOK factorization (Theorem 2.1, Equation [23]).

use burn::prelude::*;
use rand::Rng;

/// State Prediction Kernel for default dynamics
///
/// Predicts ρ(z_f | z_i, t) = P^t(z_i, z_f) where P is the default Markov chain.
///
/// Precomputes matrix powers for efficient queries.
///
/// # Use Case
///
/// High-dimensional planning: predict high-level state evolution
/// while agent executes base-level policy.
pub struct StatePredictionKernel<B: Backend> {
    /// Cached matrix powers: P^1, P^2, ..., P^max_time
    /// Each has shape [n_states, n_states]
    powers: Vec<Tensor<B, 2>>,

    /// Base transition matrix P (one-step)
    base_transition: Tensor<B, 2>,

    /// Number of states in this space
    n_states: usize,

    /// Maximum precomputed time horizon
    max_time: usize,

    /// Device
    device: B::Device,
}

impl<B: Backend> StatePredictionKernel<B> {
    /// Create SPK from transition matrix under default action
    ///
    /// # Arguments
    ///
    /// * `transition` - Base transition matrix P(z'|z), shape [S, S]
    /// * `max_time` - Maximum time horizon to precompute
    ///
    /// # Returns
    ///
    /// SPK with precomputed matrix powers
    pub fn new(transition: Tensor<B, 2>, max_time: usize) -> Self {
        let dims = transition.dims();
        assert_eq!(dims[0], dims[1], "Transition matrix must be square");
        let n_states = dims[0];
        let device = transition.device();

        // Precompute matrix powers: P, P^2, P^3, ...
        let mut powers = Vec::with_capacity(max_time);
        let mut current = transition.clone();

        for _ in 0..max_time {
            powers.push(current.clone());
            current = current.matmul(transition.clone());
        }

        Self {
            powers,
            base_transition: transition,
            n_states,
            max_time,
            device,
        }
    }

    /// Get prediction distribution after t steps
    ///
    /// Returns ρ(· | z_i, t) as distribution over final states
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `time` - Number of steps
    ///
    /// # Returns
    ///
    /// Tensor of shape [S] with final state distribution
    ///
    /// # Panics
    ///
    /// Panics if time is 0 or exceeds max_time
    pub fn predict(&self, initial_state: usize, time: usize) -> Tensor<B, 1> {
        assert!(
            time > 0 && time <= self.max_time,
            "Time {} out of precomputed range [1, {}]",
            time,
            self.max_time
        );

        // Get row from P^t
        let power_matrix = &self.powers[time - 1]; // 0-indexed
        power_matrix
            .clone()
            .slice([initial_state..(initial_state + 1), 0..self.n_states])
            .reshape([self.n_states])
    }

    /// Get full prediction matrix for time t
    ///
    /// Returns ρ(z_f | z_i, t) for all (z_i, z_f) pairs
    ///
    /// # Arguments
    ///
    /// * `time` - Number of steps
    ///
    /// # Returns
    ///
    /// Tensor of shape [S, S]
    pub fn predict_matrix(&self, time: usize) -> Tensor<B, 2> {
        assert!(time > 0 && time <= self.max_time);
        self.powers[time - 1].clone()
    }

    /// Sample final state
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    /// * `time` - Number of steps
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Sampled final state index
    pub fn sample(
        &self,
        initial_state: usize,
        time: usize,
        rng: &mut impl Rng,
    ) -> usize {
        let distribution = self.predict(initial_state, time);
        let probs: Vec<f32> = distribution.into_data().to_vec().unwrap();

        crate::planning::sample_categorical(&probs, rng)
    }

    /// Get device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get maximum time horizon
    pub fn max_time(&self) -> usize {
        self.max_time
    }

    /// Get number of states
    pub fn n_states(&self) -> usize {
        self.n_states
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
    fn test_spk_identity() {
        let device = default_device();

        // Identity transition (always stay in same state)
        let transition: Tensor<DefaultBackend, 2> = Tensor::eye(3, &device);
        let spk = StatePredictionKernel::new(transition, 5);

        // After any time, should stay in same state
        for t in 1..=5 {
            for state in 0..3 {
                let dist = spk.predict(state, t);
                let probs: Vec<f32> = dist.into_data().to_vec().unwrap();

                // Should be 1.0 at current state, 0.0 elsewhere
                assert!((probs[state] - 1.0).abs() < 1e-5);
                for (i, &p) in probs.iter().enumerate() {
                    if i != state {
                        assert!(p.abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn test_spk_deterministic_chain() {
        let device = default_device();

        // Deterministic chain: 0 -> 1 -> 2 -> 0 (cycle)
        let transition: Tensor<DefaultBackend, 2> = Tensor::from_floats(
            [
                [0.0, 1.0, 0.0], // From state 0, always go to state 1
                [0.0, 0.0, 1.0], // From state 1, always go to state 2
                [1.0, 0.0, 0.0], // From state 2, always go to state 0
            ],
            &device,
        );

        let spk = StatePredictionKernel::new(transition, 5);

        // After 1 step from state 0, should be at state 1
        let dist_1 = spk.predict(0, 1);
        let probs_1: Vec<f32> = dist_1.into_data().to_vec().unwrap();
        assert!((probs_1[1] - 1.0).abs() < 1e-5);

        // After 2 steps from state 0, should be at state 2
        let dist_2 = spk.predict(0, 2);
        let probs_2: Vec<f32> = dist_2.into_data().to_vec().unwrap();
        assert!((probs_2[2] - 1.0).abs() < 1e-5);

        // After 3 steps from state 0, should be back at state 0
        let dist_3 = spk.predict(0, 3);
        let probs_3: Vec<f32> = dist_3.into_data().to_vec().unwrap();
        assert!((probs_3[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_spk_matrix_power() {
        let device = default_device();

        let transition: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.5, 0.5], [0.3, 0.7]], &device);

        let spk = StatePredictionKernel::new(transition.clone(), 3);

        // P^2 should equal P @ P
        let p2_manual = transition.clone().matmul(transition.clone());
        let p2_computed = spk.predict_matrix(2);

        let diff: f32 = (p2_manual - p2_computed).abs().max().into_scalar().elem();
        assert!(diff < 1e-5);
    }
}
