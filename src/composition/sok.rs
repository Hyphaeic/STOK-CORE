//! # State Option Kernel (SOK) Utilities
//!
//! SOKs are time-marginalized STOKs that represent the final state distribution
//! without temporal information. They compose via simple matrix multiplication.
//!
//! ## Mathematical Definition
//!
//! A State Option Kernel χ(x_f | x_i) is computed from a STOK by marginalizing time:
//!
//! ```text
//! χ(x_f | x_i) = Σ_{t_f} η**(x_f, t_f | x_i)
//! ```
//!
//! ## Properties
//!
//! - **Normalization**: Σ_{x_f} χ(x_f | x_i) = 1 for all x_i
//! - **Composition**: χ_μ = χ_1 @ χ_2 (matrix multiplication)
//! - **Efficiency**: Much faster than STOK composition (no time dimension)
//!
//! ## Use Cases
//!
//! SOKs are ideal when:
//! - Time information is not needed
//! - High-level state-spaces are static (e.g., Boolean logic)
//! - Fast composition queries are required

use crate::stok::STOKKernel;
use crate::types::StokError;
use burn::prelude::*;

/// State Option Kernel: time-marginalized STOK
///
/// Represents the distribution over final states without time information.
/// Composes via matrix multiplication, making it much faster than full STOK composition.
///
/// # Mathematical Form
///
/// χ(x_f | x_i) represents the probability of terminating at state x_f
/// given initial state x_i, marginalized over all possible termination times.
///
/// # Decomposition
///
/// Like STOKs, SOKs can be decomposed into success and failure components:
/// - χ⁺(x_f | x_i): probability of goal-success termination at x_f
/// - χ⁻(x_f | x_i): probability of constraint-failure termination at x_f
/// - χ**(x_f | x_i) = χ⁺ + χ⁻
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::composition::StateOptionKernel;
///
/// // Convert STOK to SOK
/// let sok = StateOptionKernel::from_stok(&stok);
///
/// // Query termination distribution from state 0
/// let dist = sok.termination_distribution(0);
/// ```
#[derive(Clone, Debug)]
pub struct StateOptionKernel<B: Backend> {
    /// Combined termination distribution: χ**(x_f | x_i)
    ///
    /// Shape: [n_states, n_states]
    /// Row i contains distribution over final states starting from state i
    /// Each row sums to 1.0 (proper transition kernel)
    pub chi: Tensor<B, 2>,

    /// Success termination distribution: χ⁺(x_f | x_i)
    ///
    /// Probability of goal-success termination at x_f from x_i
    /// Optional: only present if constructed from decomposed STOK
    pub chi_plus: Option<Tensor<B, 2>>,

    /// Failure termination distribution: χ⁻(x_f | x_i)
    ///
    /// Probability of constraint-failure termination at x_f from x_i
    /// Optional: only present if constructed from decomposed STOK
    pub chi_minus: Option<Tensor<B, 2>>,

    /// Number of states in this kernel
    pub n_states: usize,
}

impl<B: Backend> StateOptionKernel<B> {
    /// Create SOK from a STOK by marginalizing time
    ///
    /// Implements: χ(x_f | x_i) = Σ_{t_f} η**(x_f, t_f | x_i)
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to marginalize
    ///
    /// # Returns
    ///
    /// State Option Kernel with time marginalized out
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stok = feasibility_iteration(&mdp, config)?.kernel;
    /// let sok = StateOptionKernel::from_stok(&stok);
    /// assert_eq!(sok.n_states, stok.n_states());
    /// ```
    pub fn from_stok(stok: &STOKKernel<B>) -> Self {
        let n_states = stok.n_states();

        // Marginalize time: sum over time dimension (axis 2)
        // η has shape [S, S, T], sum over T gives [S, S]

        // Combined: χ** = Σ_t (η⁺ + η⁻)
        let eta_combined = stok.eta_plus.clone() + stok.eta_minus.clone();
        let chi = eta_combined.sum_dim(2).squeeze::<2>(); // [S, S, T] -> [S, S]

        // Success component: χ⁺ = Σ_t η⁺
        let chi_plus = Some(stok.eta_plus.clone().sum_dim(2).squeeze::<2>()); // [S, S, T] -> [S, S]

        // Failure component: χ⁻ = Σ_t η⁻
        let chi_minus = Some(stok.eta_minus.clone().sum_dim(2).squeeze::<2>()); // [S, S, T] -> [S, S]

        Self {
            chi,
            chi_plus,
            chi_minus,
            n_states,
        }
    }

    /// Create SOK from raw chi matrix
    ///
    /// # Arguments
    ///
    /// * `chi` - Transition matrix [n_states, n_states]
    ///
    /// # Note
    ///
    /// This does not validate normalization. Use `validate()` to check.
    pub fn from_chi(chi: Tensor<B, 2>) -> Self {
        let dims = chi.dims();
        assert_eq!(dims.len(), 2, "Chi must be 2D tensor");
        assert_eq!(dims[0], dims[1], "Chi must be square matrix");

        Self {
            n_states: dims[0],
            chi,
            chi_plus: None,
            chi_minus: None,
        }
    }

    /// Get device this SOK is allocated on
    pub fn device(&self) -> B::Device {
        self.chi.device()
    }

    /// Validate SOK normalization
    ///
    /// Each row of χ should sum to 1.0 (it's a transition kernel)
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed deviation from 1.0
    ///
    /// # Returns
    ///
    /// Ok if normalized, Err with details if validation fails
    pub fn validate(&self, tolerance: f32) -> Result<(), StokError> {
        // Each row should sum to 1.0
        // chi is [S, S], sum_dim(1) sums over columns, giving [S]
        let row_sums: Tensor<B, 1> = self.chi.clone().sum_dim(1).squeeze::<1>(); // [S, S] -> [S]

        // Check normalization manually (validate_normalized returns bool, not Result)
        let ones = Tensor::ones([self.n_states], &self.device());
        let diff: f32 = (row_sums - ones).abs().max().into_scalar().elem();

        if diff > tolerance {
            return Err(StokError::NotNormalized {
                sum: 1.0 + diff,
                expected: 1.0,
                tolerance,
            });
        }

        // Check probability bounds [0, 1]
        self.validate_probability_bounds()?;

        // If decomposed, verify χ = χ⁺ + χ⁻
        if let (Some(ref cp), Some(ref cm)) = (&self.chi_plus, &self.chi_minus) {
            let sum = cp.clone() + cm.clone();
            let diff = (self.chi.clone() - sum)
                .abs()
                .max()
                .into_scalar()
                .elem::<f32>();

            if diff > tolerance {
                return Err(StokError::NotNormalized {
                    sum: diff,
                    expected: 0.0,
                    tolerance,
                });
            }
        }

        Ok(())
    }

    /// Check that all probabilities are in [0, 1]
    pub fn validate_probability_bounds(&self) -> Result<(), StokError> {
        let min_val: f32 = self.chi.clone().min().into_scalar().elem();
        let max_val: f32 = self.chi.clone().max().into_scalar().elem();

        if min_val < 0.0 || max_val > 1.0 {
            return Err(StokError::InvalidProbability {
                value: if min_val < 0.0 { min_val } else { max_val },
                context: "SOK chi values out of [0,1] range".into(),
            });
        }

        Ok(())
    }

    /// Validate decomposition consistency
    ///
    /// If χ⁺ and χ⁻ are present, verify χ = χ⁺ + χ⁻
    pub fn validate_decomposition(&self, tolerance: f32) -> Result<(), StokError> {
        match (&self.chi_plus, &self.chi_minus) {
            (Some(cp), Some(cm)) => {
                let sum = cp.clone() + cm.clone();
                let diff: f32 = (self.chi.clone() - sum).abs().max().into_scalar().elem();

                if diff > tolerance {
                    return Err(StokError::NotNormalized {
                        sum: diff,
                        expected: 0.0,
                        tolerance,
                    });
                }
                Ok(())
            }
            _ => Ok(()), // No decomposition to validate
        }
    }

    /// Get success probability from initial state
    ///
    /// Returns κ(x_i) = Σ_{x_f} χ⁺(x_f | x_i)
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state index
    ///
    /// # Returns
    ///
    /// Total probability of goal success, or None if decomposition not available
    pub fn success_probability(&self, initial_state: usize) -> Option<f32> {
        self.chi_plus.as_ref().map(|cp| {
            cp.clone()
                .slice([initial_state..(initial_state + 1), 0..self.n_states])
                .sum()
                .into_scalar()
                .elem()
        })
    }

    /// Get failure probability from initial state
    ///
    /// Returns 1 - κ(x_i) = Σ_{x_f} χ⁻(x_f | x_i)
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state index
    ///
    /// # Returns
    ///
    /// Total probability of constraint failure, or None if decomposition not available
    pub fn failure_probability(&self, initial_state: usize) -> Option<f32> {
        self.chi_minus.as_ref().map(|cm| {
            cm.clone()
                .slice([initial_state..(initial_state + 1), 0..self.n_states])
                .sum()
                .into_scalar()
                .elem()
        })
    }

    /// Get termination distribution from initial state
    ///
    /// Returns χ(· | x_i) as a vector of probabilities
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state index
    ///
    /// # Returns
    ///
    /// Vector of length n_states with termination probabilities
    pub fn termination_distribution(&self, initial_state: usize) -> Vec<f32> {
        self.chi
            .clone()
            .slice([initial_state..(initial_state + 1), 0..self.n_states])
            .reshape([self.n_states])
            .into_data()
            .to_vec()
            .unwrap()
    }

    /// Get full chi matrix as CPU data
    ///
    /// Useful for visualization and analysis
    pub fn to_matrix(&self) -> Vec<Vec<f32>> {
        let data: Vec<f32> = self.chi.clone().into_data().to_vec().unwrap();

        let mut matrix = Vec::with_capacity(self.n_states);
        for i in 0..self.n_states {
            let row = data[i * self.n_states..(i + 1) * self.n_states].to_vec();
            matrix.push(row);
        }

        matrix
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::default_device;

    #[test]
    fn test_sok_from_stok() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create a simple STOK manually
        // States: 0, 1, 2 with time horizon 3
        let n_states = 3;
        let max_time = 3;

        // Simple deterministic STOK: always terminate at state 2 at time 2
        let mut eta_plus: Tensor<DefaultBackend, 3> =
            Tensor::zeros([n_states, n_states, max_time], &device);

        // Set η⁺(2, t=2 | 0) = 1.0
        let value = Tensor::from_floats([[[1.0]]], &device);
        eta_plus = eta_plus.slice_assign([0..1, 2..3, 2..3], value);

        let eta_minus = Tensor::zeros([n_states, n_states, max_time], &device);
        let kappa = Tensor::from_floats([1.0, 0.0, 0.0], &device);
        let policy = Tensor::zeros([n_states], &device);

        let stok = STOKKernel {
            eta_plus,
            eta_minus,
            kappa,
            policy,
            dims: crate::types::STOKDimensions::new(n_states, max_time),
        };

        // Convert to SOK
        let sok = StateOptionKernel::from_stok(&stok);

        assert_eq!(sok.n_states, 3);

        // χ(2 | 0) should be 1.0 (sum over all times)
        let chi_20: f32 = sok.chi.clone().slice([0..1, 2..3]).into_scalar().elem();
        assert!((chi_20 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sok_normalization() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create identity matrix as SOK (always stay in same state)
        let chi: Tensor<DefaultBackend, 2> = Tensor::eye(5, &device);
        let sok = StateOptionKernel::from_chi(chi);

        // Should be normalized
        assert!(sok.validate(1e-5).is_ok());
    }

    #[test]
    fn test_sok_probability_bounds() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create SOK with invalid probabilities
        let chi: Tensor<DefaultBackend, 2> = Tensor::from_floats(
            [
                [0.5, 0.5],
                [-0.1, 1.1], // Invalid: negative and > 1
            ],
            &device,
        );
        let sok = StateOptionKernel::from_chi(chi);

        // Should fail bounds check
        assert!(sok.validate_probability_bounds().is_err());
    }

    #[test]
    fn test_sok_success_probability() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let chi: Tensor<DefaultBackend, 2> = Tensor::from_floats([[0.7, 0.3], [0.4, 0.6]], &device);
        let chi_plus: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.6, 0.2], [0.3, 0.5]], &device);
        let chi_minus: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.1, 0.1], [0.1, 0.1]], &device);

        let sok = StateOptionKernel {
            chi,
            chi_plus: Some(chi_plus),
            chi_minus: Some(chi_minus),
            n_states: 2,
        };

        // Success probability from state 0 = 0.6 + 0.2 = 0.8
        let success_prob = sok.success_probability(0).unwrap();
        assert!((success_prob - 0.8).abs() < 1e-5);

        // Failure probability from state 0 = 0.1 + 0.1 = 0.2
        let failure_prob = sok.failure_probability(0).unwrap();
        assert!((failure_prob - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_termination_distribution() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let chi: Tensor<DefaultBackend, 2> = Tensor::from_floats([[0.3, 0.7], [0.9, 0.1]], &device);
        let sok = StateOptionKernel::from_chi(chi);

        let dist = sok.termination_distribution(0);
        assert_eq!(dist.len(), 2);
        assert!((dist[0] - 0.3).abs() < 1e-5);
        assert!((dist[1] - 0.7).abs() < 1e-5);

        let dist1 = sok.termination_distribution(1);
        assert!((dist1[0] - 0.9).abs() < 1e-5);
        assert!((dist1[1] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_to_matrix() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let chi: Tensor<DefaultBackend, 2> = Tensor::from_floats([[0.5, 0.5], [0.3, 0.7]], &device);
        let sok = StateOptionKernel::from_chi(chi);

        let matrix = sok.to_matrix();
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert!((matrix[0][0] - 0.5).abs() < 1e-5);
        assert!((matrix[1][1] - 0.7).abs() < 1e-5);
    }
}
