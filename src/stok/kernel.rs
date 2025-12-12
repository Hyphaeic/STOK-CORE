//! State-Time Option Kernel (STOK) data structure.
//!
//! STOKs are predictive probability kernels over termination events,
//! replacing scalar value functions with full distributions.
//!
//! # Mathematical Definition
//!
//! A STOK η**(x_f, t_f | x_i) represents the probability distribution over
//! termination events (final state x_f, final time t_f) given initial state x_i.
//!
//! # Decomposition (Equations [15-17])
//!
//! The combined STOK decomposes into success and failure components:
//! ```text
//! η**(x_f, t_f | x) = η⁺(x_f, t_f | x) + η⁻(x_f, t_f | x)
//! ```
//!
//! Where:
//! - η⁺: Success termination (goal reached without constraint violation)
//! - η⁻: Failure termination (constraint violated before goal)
//!
//! # Key Invariants
//!
//! 1. **Normalization**: Σ_{x_f, t_f} η**(x_f, t_f | x) = 1 for all x
//! 2. **κ-η consistency**: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
//! 3. **Probability bounds**: All values in [0, 1]
//!
//! # Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Section G.2: "State-Time Option Kernel"

use burn::prelude::*;
use crate::types::{MDPDimensions, StokError, DEFAULT_PROBABILITY_TOLERANCE};

/// State-Time Option Kernel with success/failure decomposition.
///
/// GPU-resident tensor structure storing the full STOK distribution
/// along with the cumulative feasibility function κ and optimal policy π.
///
/// # Tensor Conventions
///
/// - `eta_plus[x_i, x_f, t_f]` = η⁺(x_f, t_f | x_i) - success termination
/// - `eta_minus[x_i, x_f, t_f]` = η⁻(x_f, t_f | x_i) - failure termination
/// - `kappa[x]` = κ(x) - cumulative feasibility (Equation [7])
/// - `policy[x]` = π**(x) - optimal action index
///
/// # Initialization
///
/// Per paper Appendix 7, zero initialization is critical for correct
/// κ values in unreachable states. Use `STOKKernel::new()` which
/// guarantees zero initialization.
///
/// # Example
///
/// ```rust,ignore
/// use stok_core::prelude::*;
///
/// let device = default_device();
/// let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 20, &device);
///
/// assert_eq!(kernel.n_states(), 10);
/// assert_eq!(kernel.max_time(), 20);
/// ```
#[derive(Debug)]
pub struct STOKKernel<B: Backend> {
    /// Success termination probabilities η⁺(x_f, t_f | x_i).
    ///
    /// Shape: [n_states, n_states, max_time]
    ///
    /// η⁺ records the probability of reaching goal state x_f at time t_f
    /// starting from x_i, without violating constraints.
    /// See Equation [15] in Ringstrom & Schrater (2025).
    pub eta_plus: Tensor<B, 3>,

    /// Failure termination probabilities η⁻(x_f, t_f | x_i).
    ///
    /// Shape: [n_states, n_states, max_time]
    ///
    /// η⁻ records the probability of first constraint violation at (x_f, t_f)
    /// starting from x_i.
    /// See Equation [16] in Ringstrom & Schrater (2025).
    pub eta_minus: Tensor<B, 3>,

    /// Cumulative feasibility function κ(x).
    ///
    /// Shape: [n_states]
    ///
    /// κ(x) = P(eventually reach goal without constraint violation | start at x)
    /// Computed via κ-OKBE (Equation [7]).
    ///
    /// Invariant: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    pub kappa: Tensor<B, 1>,

    /// Optimal policy π**(x).
    ///
    /// Shape: [n_states]
    ///
    /// π**(x) = argmax_a Q(x, a) where Q is the feasibility Q-function.
    /// Computed via π-OKBE (Equation [8]).
    pub policy: Tensor<B, 1, Int>,

    /// Cached dimension information.
    pub dims: MDPDimensions,
}

impl<B: Backend> STOKKernel<B> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a new zero-initialized STOK kernel.
    ///
    /// All tensors are initialized to zero, which is critical per
    /// paper Appendix 7 for correct handling of unreachable states.
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states |S|
    /// * `max_time` - Maximum time horizon T
    /// * `device` - Backend device for tensor allocation
    ///
    /// # Panics
    ///
    /// Panics if `n_states == 0` or `max_time == 0`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(10, 20, &device);
    /// ```
    pub fn new(n_states: usize, max_time: usize, device: &B::Device) -> Self {
        let dims = MDPDimensions::new(n_states, 0, max_time);
        dims.validate().expect("Invalid dimensions for STOKKernel");

        // Zero initialization is critical (see paper Appendix 7)
        let eta_plus = Tensor::zeros([n_states, n_states, max_time], device);
        let eta_minus = Tensor::zeros([n_states, n_states, max_time], device);
        let kappa = Tensor::zeros([n_states], device);
        let policy = Tensor::zeros([n_states], device);

        Self {
            eta_plus,
            eta_minus,
            kappa,
            policy,
            dims,
        }
    }

    /// Create STOK kernel from pre-computed κ and π.
    ///
    /// Used when only feasibility (not full STOK) is needed.
    /// η⁺ and η⁻ are left as zeros.
    ///
    /// # Arguments
    ///
    /// * `kappa` - Pre-computed cumulative feasibility
    /// * `policy` - Pre-computed optimal policy
    /// * `max_time` - Time horizon for STOK dimensions
    /// * `device` - Backend device
    pub fn from_kappa_policy(
        kappa: Tensor<B, 1>,
        policy: Tensor<B, 1, Int>,
        max_time: usize,
        device: &B::Device,
    ) -> Self {
        let n_states = kappa.dims()[0];
        let dims = MDPDimensions::new(n_states, 0, max_time);

        let eta_plus = Tensor::zeros([n_states, n_states, max_time], device);
        let eta_minus = Tensor::zeros([n_states, n_states, max_time], device);

        Self {
            eta_plus,
            eta_minus,
            kappa,
            policy,
            dims,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Number of states in the STOK.
    pub fn n_states(&self) -> usize {
        self.dims.n_states
    }

    /// Maximum time horizon.
    pub fn max_time(&self) -> usize {
        self.dims.max_time
    }

    /// Get the device where tensors are allocated.
    pub fn device(&self) -> B::Device {
        self.kappa.device()
    }

    // ========================================================================
    // Derived Quantities
    // ========================================================================

    /// Compute κ from η⁺ by summation.
    ///
    /// κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    ///
    /// This should equal `self.kappa` if invariants hold.
    /// Used for validation.
    ///
    /// # Reference
    ///
    /// Equation [15]: κ*_g(x) = Σ_{x_f} Σ_{t_f} η⁺_{π_g}(x_f, t_f | x)
    pub fn kappa_from_eta(&self) -> Tensor<B, 1> {
        // Sum over x_f (dim 1) and t_f (dim 2)
        self.eta_plus.clone()
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>()
    }

    /// Compute infeasibility 1 - κ from η⁻.
    ///
    /// 1 - κ(x) = Σ_{x_f, t_f} η⁻(x_f, t_f | x)
    ///
    /// # Reference
    ///
    /// Equation [16]: 1 - κ*_g(x) = Σ_{x_f} Σ_{t_f} η⁻_{π_g}(x_f, t_f | x)
    pub fn infeasibility(&self) -> Tensor<B, 1> {
        self.eta_minus.clone()
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>()
    }

    /// Compute combined STOK η** = η⁺ + η⁻.
    ///
    /// The combined STOK gives the full termination distribution,
    /// regardless of success or failure.
    ///
    /// # Reference
    ///
    /// Equation [17]: η**_{o_g}(x_f, t_f | x) = η⁺_{π_g}(x_f, t_f | x) + η⁻_{π_g}(x_f, t_f | x)
    pub fn combined_stok(&self) -> Tensor<B, 3> {
        self.eta_plus.clone() + self.eta_minus.clone()
    }

    /// Compute State Option Kernel (SOK) by marginalizing time.
    ///
    /// χ(x_f | x) = Σ_{t_f} η**(x_f, t_f | x)
    ///
    /// SOK composition reduces to matrix multiplication (Equation [19]).
    pub fn state_option_kernel(&self) -> Tensor<B, 2> {
        self.combined_stok().sum_dim(2).squeeze::<2>()
    }

    // ========================================================================
    // Validation Methods
    // ========================================================================

    /// Validate STOK normalization.
    ///
    /// Combined STOK should sum to 1 for each initial state:
    /// Σ_{x_f, t_f} η**(x_f, t_f | x) = 1 for all x
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum deviation from 1.0
    ///
    /// # Note
    ///
    /// This is only meaningful after feasibility iteration converges
    /// and STOK construction completes.
    pub fn validate_normalization(&self, tolerance: f32) -> Result<(), StokError> {
        let combined = self.combined_stok();
        // Sum over x_f and t_f for each x_i
        let sums: Tensor<B, 1> = combined
            .sum_dim(2).squeeze::<2>()
            .sum_dim(1).squeeze::<1>();

        let device = sums.device();
        let ones: Tensor<B, 1> = Tensor::ones(sums.dims(), &device);
        let diff = (sums - ones).abs();
        let max_diff: f32 = diff.max().into_scalar().elem();

        if max_diff > tolerance {
            return Err(StokError::NotNormalized {
                sum: 1.0 + max_diff,
                expected: 1.0,
                tolerance,
            });
        }
        Ok(())
    }

    /// Validate all probability values are in [0, 1].
    ///
    /// Checks η⁺, η⁻, and κ for valid bounds.
    pub fn validate_probability_bounds(&self) -> Result<(), StokError> {
        // Check eta_plus
        let ep_min: f32 = self.eta_plus.clone().min().into_scalar().elem();
        let ep_max: f32 = self.eta_plus.clone().max().into_scalar().elem();
        if ep_min < -DEFAULT_PROBABILITY_TOLERANCE || ep_max > 1.0 + DEFAULT_PROBABILITY_TOLERANCE {
            return Err(StokError::InvalidProbability {
                value: if ep_min < 0.0 { ep_min } else { ep_max },
                context: "eta_plus".into(),
            });
        }

        // Check eta_minus
        let em_min: f32 = self.eta_minus.clone().min().into_scalar().elem();
        let em_max: f32 = self.eta_minus.clone().max().into_scalar().elem();
        if em_min < -DEFAULT_PROBABILITY_TOLERANCE || em_max > 1.0 + DEFAULT_PROBABILITY_TOLERANCE {
            return Err(StokError::InvalidProbability {
                value: if em_min < 0.0 { em_min } else { em_max },
                context: "eta_minus".into(),
            });
        }

        // Check kappa
        let k_min: f32 = self.kappa.clone().min().into_scalar().elem();
        let k_max: f32 = self.kappa.clone().max().into_scalar().elem();
        if k_min < -DEFAULT_PROBABILITY_TOLERANCE || k_max > 1.0 + DEFAULT_PROBABILITY_TOLERANCE {
            return Err(StokError::InvalidProbability {
                value: if k_min < 0.0 { k_min } else { k_max },
                context: "kappa".into(),
            });
        }

        Ok(())
    }

    /// Validate κ-η consistency.
    ///
    /// κ(x) should equal sum of η⁺ over (x_f, t_f):
    /// κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    ///
    /// Also checks that (1-κ) equals sum of η⁻.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Maximum allowed discrepancy
    pub fn validate_kappa_consistency(&self, tolerance: f32) -> Result<(), StokError> {
        // Check κ = Σ η⁺
        let kappa_computed = self.kappa_from_eta();
        let diff_plus = (self.kappa.clone() - kappa_computed).abs();
        let max_diff_plus: f32 = diff_plus.max().into_scalar().elem();

        if max_diff_plus > tolerance {
            return Err(StokError::NotNormalized {
                sum: max_diff_plus,
                expected: 0.0,
                tolerance,
            });
        }

        // Check (1-κ) = Σ η⁻
        let device = self.kappa.device();
        let ones: Tensor<B, 1> = Tensor::ones(self.kappa.dims(), &device);
        let one_minus_kappa = ones - self.kappa.clone();
        let infeasibility = self.infeasibility();
        let diff_minus = (one_minus_kappa - infeasibility).abs();
        let max_diff_minus: f32 = diff_minus.max().into_scalar().elem();

        if max_diff_minus > tolerance {
            return Err(StokError::NotNormalized {
                sum: max_diff_minus,
                expected: 0.0,
                tolerance,
            });
        }

        Ok(())
    }

    /// Run all validation checks.
    ///
    /// Validates normalization, probability bounds, and κ-η consistency.
    ///
    /// # Note
    ///
    /// Only meaningful after feasibility iteration has converged
    /// and STOK construction is complete.
    pub fn validate(&self, tolerance: f32) -> Result<(), StokError> {
        self.validate_normalization(tolerance)?;
        self.validate_probability_bounds()?;
        self.validate_kappa_consistency(tolerance)?;
        Ok(())
    }

    // ========================================================================
    // Data Extraction (for debugging)
    // ========================================================================

    /// Extract all data to CPU for debugging/visualization.
    ///
    /// Transfers tensor data to CPU-resident vectors.
    /// Useful for inspection, serialization, or plotting.
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

/// CPU-side STOK data for debugging and serialization.
///
/// Contains flattened vectors of all STOK tensors, suitable for
/// inspection, saving to disk, or transferring between systems.
#[derive(Debug, Clone)]
pub struct STOKData {
    /// Flattened η⁺ data, row-major [x_i, x_f, t_f]
    pub eta_plus: Vec<f32>,
    /// Flattened η⁻ data, row-major [x_i, x_f, t_f]
    pub eta_minus: Vec<f32>,
    /// κ values indexed by state
    pub kappa: Vec<f32>,
    /// Policy action indices by state
    pub policy: Vec<i32>,
    /// Dimension specification
    pub dims: MDPDimensions,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DefaultBackend, default_device};

    // ========================================================================
    // Initialization Tests
    // ========================================================================

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

        let eta_plus_data: Vec<f32> = kernel.eta_plus.clone().into_data().to_vec().unwrap();
        assert!(eta_plus_data.iter().all(|&v| v == 0.0));
    }

    // ========================================================================
    // Derived Quantity Tests
    // ========================================================================

    #[test]
    fn test_combined_stok() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Manually set some values
        let eta_plus_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0];
        let eta_minus_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        kernel.eta_minus = Tensor::<DefaultBackend, 1>::from_floats(eta_minus_data.as_slice(), &device)
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
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let kappa_computed = kernel.kappa_from_eta();
        let data: Vec<f32> = kappa_computed.into_data().to_vec().unwrap();

        assert!((data[0] - 0.7).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_infeasibility() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Set eta_minus such that sum gives known infeasibility
        let eta_minus_data: Vec<f32> = vec![
            0.1, 0.1, // x_i=0, x_f=0, t_f=0,1
            0.05, 0.05, // x_i=0, x_f=1, t_f=0,1
            0.0, 0.0, // x_i=1, x_f=0, t_f=0,1
            0.0, 0.0, // x_i=1, x_f=1, t_f=0,1
        ];
        kernel.eta_minus = Tensor::<DefaultBackend, 1>::from_floats(eta_minus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let infeas = kernel.infeasibility();
        let data: Vec<f32> = infeas.into_data().to_vec().unwrap();

        assert!((data[0] - 0.3).abs() < 1e-5); // 0.1+0.1+0.05+0.05
        assert!((data[1] - 0.0).abs() < 1e-5);
    }

    // ========================================================================
    // Validation Tests - Valid Kernels
    // ========================================================================

    #[test]
    fn test_validate_probability_bounds_zero_kernel() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(5, 3, &device);
        
        // Zero-initialized kernel should pass bounds check
        assert!(kernel.validate_probability_bounds().is_ok());
    }

    #[test]
    fn test_validate_normalized_valid_stok() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Create a properly normalized STOK
        // For each initial state, η⁺ + η⁻ must sum to 1.0
        let eta_plus_data: Vec<f32> = vec![
            0.2, 0.1, // x_i=0: contributions to state 0
            0.3, 0.1, // x_i=0: contributions to state 1  (total η⁺ from x_i=0 = 0.7)
            0.5, 0.0, // x_i=1: contributions to state 0
            0.5, 0.0, // x_i=1: contributions to state 1  (total η⁺ from x_i=1 = 1.0)
        ];
        let eta_minus_data: Vec<f32> = vec![
            0.1, 0.1, // x_i=0: failure contributions (total η⁻ from x_i=0 = 0.3)
            0.05, 0.05,
            0.0, 0.0, // x_i=1: no failures (total η⁻ from x_i=1 = 0.0)
            0.0, 0.0,
        ];

        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        kernel.eta_minus = Tensor::<DefaultBackend, 1>::from_floats(eta_minus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        // Should pass normalization check
        assert!(kernel.validate_normalization(1e-5).is_ok());
    }

    #[test]
    fn test_validate_kappa_consistency_valid() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Set up consistent κ and η
        let eta_plus_data: Vec<f32> = vec![
            0.2, 0.1, 0.3, 0.1, // x_i=0: sum = 0.7
            0.25, 0.25, 0.25, 0.25, // x_i=1: sum = 1.0
        ];
        let eta_minus_data: Vec<f32> = vec![
            0.1, 0.1, 0.05, 0.05, // x_i=0: sum = 0.3
            0.0, 0.0, 0.0, 0.0, // x_i=1: sum = 0.0
        ];
        let kappa_data: Vec<f32> = vec![0.7, 1.0]; // κ = Σ η⁺

        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        kernel.eta_minus = Tensor::<DefaultBackend, 1>::from_floats(eta_minus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        kernel.kappa = Tensor::<DefaultBackend, 1>::from_floats(kappa_data.as_slice(), &device);

        // Should pass consistency check
        assert!(kernel.validate_kappa_consistency(1e-5).is_ok());
    }

    // ========================================================================
    // Validation Tests - Invalid Kernels (should fail)
    // ========================================================================

    #[test]
    fn test_validate_probability_bounds_negative_eta_plus() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Set negative value in eta_plus
        let eta_plus_data: Vec<f32> = vec![-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let result = kernel.validate_probability_bounds();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value < 0.0);
                assert_eq!(context, "eta_plus");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_validate_probability_bounds_eta_plus_over_one() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Set value > 1.0 in eta_plus
        let eta_plus_data: Vec<f32> = vec![1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let result = kernel.validate_probability_bounds();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value > 1.0);
                assert_eq!(context, "eta_plus");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_validate_probability_bounds_negative_eta_minus() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        let eta_minus_data: Vec<f32> = vec![-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        kernel.eta_minus = Tensor::<DefaultBackend, 1>::from_floats(eta_minus_data.as_slice(), &device)
            .reshape([2, 2, 2]);

        let result = kernel.validate_probability_bounds();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::InvalidProbability { context, .. } => {
                assert_eq!(context, "eta_minus");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_validate_probability_bounds_negative_kappa() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        let kappa_data: Vec<f32> = vec![-0.1, 0.5];
        kernel.kappa = Tensor::<DefaultBackend, 1>::from_floats(kappa_data.as_slice(), &device);

        let result = kernel.validate_probability_bounds();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::InvalidProbability { context, .. } => {
                assert_eq!(context, "kappa");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_validate_probability_bounds_kappa_over_one() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        let kappa_data: Vec<f32> = vec![0.5, 1.2]; // 1.2 > 1.0
        kernel.kappa = Tensor::<DefaultBackend, 1>::from_floats(kappa_data.as_slice(), &device);

        let result = kernel.validate_probability_bounds();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::InvalidProbability { value, context } => {
                assert!(value > 1.0);
                assert_eq!(context, "kappa");
            }
            _ => panic!("Expected InvalidProbability error"),
        }
    }

    #[test]
    fn test_validate_normalization_not_normalized() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Combined STOK doesn't sum to 1
        let eta_plus_data: Vec<f32> = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        // eta_minus stays zero, so sum = 0.4 per initial state (not 1.0)

        let result = kernel.validate_normalization(1e-5);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StokError::NotNormalized { expected, .. } => {
                assert_eq!(expected, 1.0);
            }
            _ => panic!("Expected NotNormalized error"),
        }
    }

    #[test]
    fn test_validate_kappa_consistency_inconsistent() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // η⁺ sums to 0.4 per state, but κ is set to 0.9
        let eta_plus_data: Vec<f32> = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        kernel.eta_plus = Tensor::<DefaultBackend, 1>::from_floats(eta_plus_data.as_slice(), &device)
            .reshape([2, 2, 2]);
        
        let kappa_data: Vec<f32> = vec![0.9, 0.9]; // Inconsistent with η⁺ sum of 0.4
        kernel.kappa = Tensor::<DefaultBackend, 1>::from_floats(kappa_data.as_slice(), &device);

        let result = kernel.validate_kappa_consistency(1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_validate_catches_first_error() {
        let device = default_device();
        let mut kernel: STOKKernel<DefaultBackend> = STOKKernel::new(2, 2, &device);

        // Multiple issues: not normalized + negative κ
        let kappa_data: Vec<f32> = vec![-0.1, 0.5];
        kernel.kappa = Tensor::<DefaultBackend, 1>::from_floats(kappa_data.as_slice(), &device);

        // Full validate should fail
        let result = kernel.validate(1e-5);
        assert!(result.is_err());
    }

    // ========================================================================
    // Data Extraction Tests
    // ========================================================================

    #[test]
    fn test_to_data() {
        let device = default_device();
        let kernel: STOKKernel<DefaultBackend> = STOKKernel::new(3, 2, &device);

        let data = kernel.to_data();
        assert_eq!(data.kappa.len(), 3);
        assert_eq!(data.policy.len(), 3);
        assert_eq!(data.eta_plus.len(), 3 * 3 * 2);
        assert_eq!(data.eta_minus.len(), 3 * 3 * 2);
        assert_eq!(data.dims.n_states, 3);
        assert_eq!(data.dims.max_time, 2);
    }

    #[test]
    fn test_from_kappa_policy() {
        let device = default_device();
        
        let kappa_data: Vec<f32> = vec![0.5, 0.8, 1.0];
        let kappa: Tensor<DefaultBackend, 1> = Tensor::<DefaultBackend, 1>::from_floats(
            kappa_data.as_slice(),
            &device,
        );
        
        let policy_data: Vec<i32> = vec![1, 0, 2];
        let policy: Tensor<DefaultBackend, 1, Int> = Tensor::<DefaultBackend, 1, Int>::from_ints(
            policy_data.as_slice(),
            &device,
        );

        let kernel = STOKKernel::<DefaultBackend>::from_kappa_policy(kappa, policy, 10, &device);

        assert_eq!(kernel.n_states(), 3);
        assert_eq!(kernel.max_time(), 10);
        
        // η tensors should be zero
        let eta_data: Vec<f32> = kernel.eta_plus.into_data().to_vec().unwrap();
        assert!(eta_data.iter().all(|&v| v == 0.0));
    }
}