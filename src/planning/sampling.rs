//! # STOK Sampling Utilities
//!
//! Utilities for sampling from STOKs and SOKs.
//!
//! Sampling is essential for:
//! - Plan simulation (Monte Carlo evaluation)
//! - Stochastic tree search
//! - Trajectory generation

use crate::composition::StateOptionKernel;
use crate::stok::STOKKernel;
use burn::prelude::*;
use rand::Rng;

/// Outcome of option execution
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationOutcome {
    /// Option succeeded (reached goal)
    Success,
    /// Option failed (violated constraint or became infeasible)
    Failure,
}

/// STOK sampling utilities
pub struct STOKSampler<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> STOKSampler<B> {
    /// Sample termination (final_state, final_time) from STOK
    ///
    /// Samples from the distribution η**(x_f, t_f | x_i)
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to sample from
    /// * `initial_state` - Starting state
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (final_state, final_time)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut rng = rand::thread_rng();
    /// let (final_state, duration) = STOKSampler::sample_termination(&stok, 0, &mut rng);
    /// ```
    pub fn sample_termination(
        stok: &STOKKernel<B>,
        initial_state: usize,
        rng: &mut impl Rng,
    ) -> (usize, usize) {
        let s = stok.n_states();
        let t_max = stok.max_time();

        // Get distribution over (x_f, t_f) from initial state
        let eta = stok.combined_stok(); // [S, S, T]
        let dist = eta
            .clone()
            .slice([initial_state..(initial_state + 1), 0..s, 0..t_max])
            .reshape([s * t_max]);

        // Convert to CPU for sampling
        let probs: Vec<f32> = dist.into_data().to_vec().unwrap();

        // Sample flat index
        let flat_idx = sample_categorical(&probs, rng);

        // Convert back to (x_f, t_f)
        let x_f = flat_idx / t_max;
        let t_f = flat_idx % t_max;

        (x_f, t_f)
    }

    /// Sample only final state (marginalize time)
    ///
    /// Equivalent to sampling from the SOK
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to sample from
    /// * `initial_state` - Starting state
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Final state index
    pub fn sample_final_state(
        stok: &STOKKernel<B>,
        initial_state: usize,
        rng: &mut impl Rng,
    ) -> usize {
        let sok = StateOptionKernel::from_stok(stok);
        Self::sample_from_sok(&sok, initial_state, rng)
    }

    /// Sample from SOK directly
    ///
    /// Samples from χ(x_f | x_i) distribution
    ///
    /// # Arguments
    ///
    /// * `sok` - State Option Kernel to sample from
    /// * `initial_state` - Starting state
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Final state index
    pub fn sample_from_sok(
        sok: &StateOptionKernel<B>,
        initial_state: usize,
        rng: &mut impl Rng,
    ) -> usize {
        let s = sok.n_states;
        let dist = sok
            .chi
            .clone()
            .slice([initial_state..(initial_state + 1), 0..s])
            .reshape([s]);

        let probs: Vec<f32> = dist.into_data().to_vec().unwrap();
        sample_categorical(&probs, rng)
    }

    /// Sample success vs failure outcome
    ///
    /// Samples from Bernoulli distribution with parameter κ(x)
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to sample from
    /// * `initial_state` - Starting state
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Success if goal reached, Failure if constraint violated
    pub fn sample_outcome(
        stok: &STOKKernel<B>,
        initial_state: usize,
        rng: &mut impl Rng,
    ) -> TerminationOutcome {
        let kappa: f32 = stok
            .kappa
            .clone()
            .slice([initial_state..(initial_state + 1)])
            .into_scalar()
            .elem();

        if rng.gen::<f32>() < kappa {
            TerminationOutcome::Success
        } else {
            TerminationOutcome::Failure
        }
    }
}

/// Sample from categorical distribution
///
/// # Arguments
///
/// * `probs` - Probability distribution (should sum to ~1.0)
/// * `rng` - Random number generator
///
/// # Returns
///
/// Index sampled from the distribution
///
/// # Panics
///
/// Panics if probs is empty
pub fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    assert!(!probs.is_empty(), "Cannot sample from empty distribution");

    let total: f32 = probs.iter().sum();
    if total == 0.0 {
        // All probabilities are zero - return random index
        return rng.gen_range(0..probs.len());
    }

    let mut r = rng.gen::<f32>() * total;

    for (idx, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return idx;
        }
    }

    // Fallback to last index (shouldn't happen with proper normalization)
    probs.len() - 1
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::types::STOKDimensions;
    use burn::prelude::Tensor;

    #[test]
    fn test_sample_categorical_uniform() {
        let mut rng = rand::thread_rng();
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        // Sample many times and check distribution
        let mut counts = vec![0; 4];
        for _ in 0..1000 {
            let idx = sample_categorical(&probs, &mut rng);
            counts[idx] += 1;
        }

        // Each should be ~250 (allow wide margin for randomness)
        for count in counts {
            assert!(count > 150 && count < 350, "Count {} outside expected range", count);
        }
    }

    #[test]
    fn test_sample_categorical_deterministic() {
        let mut rng = rand::thread_rng();
        let probs = vec![0.0, 1.0, 0.0];

        // Should always return index 1
        for _ in 0..100 {
            let idx = sample_categorical(&probs, &mut rng);
            assert_eq!(idx, 1);
        }
    }

    #[test]
    fn test_sample_outcome() {
        let device = default_device();
        let mut rng = rand::thread_rng();

        // Create STOK with known κ
        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(2, 5, &device);
        stok.kappa = Tensor::from_floats([0.8, 0.2], &device);

        // Sample from state 0 (high feasibility)
        let mut successes = 0;
        let n = 1000;
        for _ in 0..n {
            if STOKSampler::sample_outcome(&stok, 0, &mut rng) == TerminationOutcome::Success {
                successes += 1;
            }
        }

        // Should be ~800 successes (allow margin for randomness)
        let success_rate = successes as f32 / n as f32;
        assert!((success_rate - 0.8).abs() < 0.05, "Success rate: {}", success_rate);
    }

    #[test]
    fn test_sample_final_state_identity() {
        let device = default_device();
        let mut rng = rand::thread_rng();

        // Create identity STOK (always stay in same state)
        let n_states = 3;
        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(n_states, 1, &device);

        // Set η⁺ to identity at t=0
        let mut eta_plus: Tensor<DefaultBackend, 3> =
            Tensor::zeros([n_states, n_states, 1], &device);
        eta_plus = eta_plus.slice_assign(
            [0..n_states, 0..n_states, 0..1],
            Tensor::eye(n_states, &device).reshape([n_states, n_states, 1]),
        );
        stok.eta_plus = eta_plus;

        // Sample from each state - should stay in same state
        for initial in 0..n_states {
            let final_state = STOKSampler::sample_final_state(&stok, initial, &mut rng);
            assert_eq!(final_state, initial, "Identity STOK should preserve state");
        }
    }
}
