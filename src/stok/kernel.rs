//! State-Time Option Kernel (STOK) definition.
//!
//! A STOK is a predictive probability kernel over termination events:
//!   η(x_f, t_f | x_i) = P(terminate at state x_f, time t_f | started at x_i)
//!
//! STOKs decompose into success and failure components (Eq. [17]):
//!   η** = η⁺ + η⁻
//!
//! Where:
//! - η⁺(x_f, t_f | x_i): probability of FIRST reaching goal at (x_f, t_f)
//! - η⁻(x_f, t_f | x_i): probability of FIRST violating constraint at (x_f, t_f)
//!
//! Key invariant: Σ_{x_f, t_f} η**(x_f, t_f | x) = 1 for all x

use burn::prelude::*;
use crate::types::{MDPDimensions, StokError, DEFAULT_PROBABILITY_TOLERANCE};

/// State-Time Option Kernel.
///
/// Tensor index conventions:
/// - `eta_plus[x_i, x_f, t_f]` = η⁺(x_f, t_f | x_i)
/// - `eta_minus[x_i, x_f, t_f]` = η⁻(x_f, t_f | x_i)
/// - `kappa[x]` = κ(x) = cumulative feasibility from x
/// - `policy[x]` = π*(x) = optimal action at x
#[derive(Debug)]
pub struct STOKKernel<B: Backend> {
    /// Success termination kernel η⁺(x_f, t_f | x_i), shape [S, S, T]
    /// Probability of first reaching goal at (x_f, t_f) starting from x_i
    pub eta_plus: Tensor<B, 3>,

    /// Failure termination kernel η⁻(x_f, t_f | x_i), shape [S, S, T]
    /// Probability of first constraint violation at (x_f, t_f) starting from x_i
    pub eta_minus: Tensor<B, 3>,

    /// Cumulative feasibility function κ(x), shape [S]
    /// κ(x) = max probability of reaching goal without constraint violation
    /// Invariant: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    pub kappa: Tensor<B, 1>,

    /// Optimal policy π*(x), shape [S] (integer tensor)
    /// Action that achieves κ(x)
    pub policy: Tensor<B, 1, Int>,

    /// Cached dimensions
    pub dims: MDPDimensions,
}

impl<B: Backend> STOKKernel<B> {
    /// Create a new zero-initialized STOK kernel.
    ///
    /// Zero initialization is critical per paper Appendix 7:
    /// non-zero init corrupts values in unreachable states.
    pub fn new(n_states: usize, max_time: usize, device: &B::Device) -> Self {
        let dims = MDPDimensions::new(n_states, 1, max_time);

        Self {
            eta_plus: Tensor::zeros([n_states, n_states, max_time], device),
            eta_minus: Tensor::zeros([n_states, n_states, max_time], device),
            kappa: Tensor::zeros([n_states], device),
            policy: Tensor::zeros([n_states], device),
            dims,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    pub fn n_states(&self) -> usize {
        self.dims.n_states
    }

    pub fn max_time(&self) -> usize {
        self.dims.max_time
    }

    pub fn dims(&self) -> &MDPDimensions {
        &self.dims
    }

    pub fn device(&self) -> B::Device {
        self.kappa.device()
    }

    // ========================================================================
    // Derived Quantities
    // ========================================================================

    /// Combined STOK η** = η⁺ + η⁻ (Eq. [17])
    ///
    /// This is the full termination kernel - probability of terminating
    /// (either by success or failure) at each (x_f, t_f).
    pub fn combined_stok(&self) -> Tensor<B, 3> {
        self.eta_plus.clone() + self.eta_minus.clone()
    }

    /// State Option Kernel χ(x_f | x_i) = Σ_{t_f} η**(x_f, t_f | x_i)
    ///
    /// Time-marginalized termination kernel, shape [S, S].
    /// Useful when timing doesn't matter.
    pub fn state_option_kernel(&self) -> Tensor<B, 2> {
        // Sum over time dimension (2) then squeeze to remove that dimension
        self.combined_stok().sum_dim(2).squeeze::<2>()
    }

    /// Success SOK: χ⁺(x_f | x_i) = Σ_{t_f} η⁺(x_f, t_f | x_i)
    pub fn success_sok(&self) -> Tensor<B, 2> {
        self.eta_plus.clone().sum_dim(2).squeeze::<2>()
    }

    /// Failure SOK: χ⁻(x_f | x_i) = Σ_{t_f} η⁻(x_f, t_f | x_i)
    pub fn failure_sok(&self) -> Tensor<B, 2> {
        self.eta_minus.clone().sum_dim(2).squeeze::<2>()
    }

    /// Compute κ from η⁺ (for validation)
    ///
    /// κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    pub fn kappa_from_eta(&self) -> Tensor<B, 1> {
        // Sum over x_f (dim 1) and t_f (dim 2), squeeze both
        self.eta_plus.clone()
            .sum_dim(2).squeeze::<2>()  // [S, S, T] -> [S, S]
            .sum_dim(1).squeeze::<1>()  // [S, S] -> [S]
    }

    /// Compute (1-κ) from η⁻ (for validation)
    ///
    /// 1 - κ(x) = Σ_{x_f, t_f} η⁻(x_f, t_f | x)
    pub fn one_minus_kappa_from_eta(&self) -> Tensor<B, 1> {
        self.eta_minus.clone()
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>()
    }

    // ========================================================================
    // Validation Methods
    // ========================================================================

    /// Validate STOK normalization: Σ_{x_f, t_f} η**(x_f, t_f | x) = 1
    pub fn validate_normalization(&self, tolerance: f32) -> Result<(), StokError> {
        let combined = self.combined_stok();
        // Sum over x_f (dim 1) and t_f (dim 2)
        let row_sums: Tensor<B, 1> = combined
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>();
        
        let device = row_sums.device();
        let ones: Tensor<B, 1> = Tensor::ones([self.n_states()], &device);
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
        let tol = DEFAULT_PROBABILITY_TOLERANCE;

        // Check eta_plus
        let ep_min: f32 = self.eta_plus.clone().min().into_scalar().elem();
        let ep_max: f32 = self.eta_plus.clone().max().into_scalar().elem();
        if ep_min < -tol || ep_max > 1.0 + tol {
            return Err(StokError::InvalidProbability {
                value: if ep_min < 0.0 { ep_min } else { ep_max },
                context: "eta_plus".into(),
            });
        }

        // Check eta_minus
        let em_min: f32 = self.eta_minus.clone().min().into_scalar().elem();
        let em_max: f32 = self.eta_minus.clone().max().into_scalar().elem();
        if em_min < -tol || em_max > 1.0 + tol {
            return Err(StokError::InvalidProbability {
                value: if em_min < 0.0 { em_min } else { em_max },
                context: "eta_minus".into(),
            });
        }

        // Check kappa
        let k_min: f32 = self.kappa.clone().min().into_scalar().elem();
        let k_max: f32 = self.kappa.clone().max().into_scalar().elem();
        if k_min < -tol || k_max > 1.0 + tol {
            return Err(StokError::InvalidProbability {
                value: if k_min < 0.0 { k_min } else { k_max },
                context: "kappa".into(),
            });
        }

        Ok(())
    }

    /// Validate κ equals sum of η⁺
    pub fn validate_kappa_consistency(&self, tolerance: f32) -> Result<(), StokError> {
        let kappa_computed = self.kappa_from_eta();
        let diff = (self.kappa.clone() - kappa_computed).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        if max_diff > tolerance {
            return Err(StokError::NotNormalized {
                sum: max_diff,
                expected: 0.0,
                tolerance,
            });
        }
        Ok(())
    }

    /// Run all validation checks.
    ///
    /// Only meaningful after feasibility iteration has converged.
    pub fn validate(&self, tolerance: f32) -> Result<(), StokError> {
        self.validate_normalization(tolerance)?;
        self.validate_probability_bounds()?;
        self.validate_kappa_consistency(tolerance)?;
        Ok(())
    }

    // ========================================================================
    // Data Extraction (for debugging)
    // ========================================================================

    /// Extract all data to CPU for debugging/visualization
    pub fn to_data(&self) -> STOKData {
        STOKData {
            eta_plus: self.eta_plus.clone().into_data().to_vec().unwrap(),
            eta_minus: self.eta_minus.clone().into_data().to_vec().unwrap(),
            kappa: self.kappa.clone().into_data().to_vec().unwrap(),
            policy: self.policy.clone().into_data().to_vec().unwrap(),
            dims: self.dims,
        }
    }
}

/// CPU-side STOK data for debugging and serialization
#[derive(Debug, Clone)]
pub struct STOKData {
    pub eta_plus: Vec<f32>,
    pub eta_minus: Vec<f32>,
    pub kappa: Vec<f32>,
    pub policy: Vec<i32>,
    pub dims: MDPDimensions,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DefaultBackend, default_device};

    #[test]
    fn test_kernel_initialization() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 5, &device);

        assert_eq!(kernel.n_states(), 10);
        assert_eq!(kernel.max_time(), 5);
    }

    #[test]
    fn test_kernel_shapes() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 5, &device);

        assert_eq!(kernel.eta_plus.dims(), [10, 10, 5]);
        assert_eq!(kernel.eta_minus.dims(), [10, 10, 5]);
        assert_eq!(kernel.kappa.dims(), [10]);
        assert_eq!(kernel.policy.dims(), [10]);
    }

    #[test]
    fn test_zero_initialization() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(5, 3, &device);

        let kappa_data: Vec<f32> = kernel.kappa.clone().into_data().to_vec().unwrap();
        assert!(kappa_data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_combined_stok() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Manually set some values using Vec
        let eta_plus_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0];
        let eta_minus_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        
        kernel.eta_plus = Tensor::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        kernel.eta_minus = Tensor::from_floats(eta_minus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let combined = kernel.combined_stok();
        let data: Vec<f32> = combined.into_data().to_vec().unwrap();

        assert_eq!(data[0], 0.1); // eta_plus[0,0,0] + eta_minus[0,0,0]
        assert_eq!(data[4], 0.5); // eta_plus[1,0,0] + eta_minus[1,0,0]
    }

    #[test]
    fn test_state_option_kernel_shape() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 5, &device);

        let sok = kernel.state_option_kernel();
        assert_eq!(sok.dims(), [10, 10]);
    }

    #[test]
    fn test_kappa_from_eta() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Set eta_plus such that sum over (x_f, t_f) gives known κ
        // For state 0: η⁺ entries sum to 0.7
        // For state 1: η⁺ entries sum to 1.0
        let eta_plus_data: Vec<f32> = vec![
            0.2, 0.1, // x_i=0, x_f=0, t_f=0,1
            0.3, 0.1, // x_i=0, x_f=1, t_f=0,1
            0.25, 0.25, // x_i=1, x_f=0, t_f=0,1
            0.25, 0.25, // x_i=1, x_f=1, t_f=0,1
        ];
        kernel.eta_plus = Tensor::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let kappa_computed = kernel.kappa_from_eta();
        let data: Vec<f32> = kappa_computed.into_data().to_vec().unwrap();

        assert!((data[0] - 0.7).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_to_data() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(3, 2, &device);

        let data = kernel.to_data();
        assert_eq!(data.kappa.len(), 3);
        assert_eq!(data.policy.len(), 3);
        assert_eq!(data.eta_plus.len(), 3 * 3 * 2);
    }
}