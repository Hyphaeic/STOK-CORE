//! # Chapman-Kolmogorov Composition
//!
//! Implements composition of STOKs and SOKs via Chapman-Kolmogorov equations.
//!
//! ## STOK Composition (Equation [18])
//!
//! ```text
//! η_μ(x_μ, t_μ | x) = Σ_{x_{f1}} Σ_{t_{f1}} η_{o2}(x_μ, t_μ - t_{f1} | x_{f1}) · η_{o1}(x_{f1}, t_{f1} | x)
//! ```
//!
//! This composes two option kernels by:
//! 1. Convolving in time: t_μ = t_1 + t_2
//! 2. Marginalizing over intermediate states: x_{f1}
//!
//! ## SOK Composition (Equation [19])
//!
//! ```text
//! χ_μ(x_μ | x) = Σ_{x_{f1}} χ_{o2}(x_μ | x_{f1}) · χ_{o1}(x_{f1} | x)
//! ```
//!
//! This reduces to matrix multiplication: χ_μ = χ_1 @ χ_2

use super::sok::StateOptionKernel;
use crate::stok::STOKKernel;
use crate::types::{STOKDimensions, StokError};
use burn::prelude::*;

/// Result of composing two or more STOKs
///
/// Represents the combined termination distribution for a sequence of options.
///
/// # Properties
///
/// - **Normalization**: Σ_{x_f, t_f} η(x_f, t_f | x) = 1
/// - **Time Horizon**: max_time = T₁ + T₂ - 1 for two options
/// - **Feasibility**: κ_μ ≤ κ_1 (composition cannot increase feasibility)
#[derive(Clone, Debug)]
pub struct ComposedSTOK<B: Backend> {
    /// Combined termination distribution: η**(x_f, t_f | x_i)
    ///
    /// Shape: [n_states, n_states, composed_max_time]
    /// Contains probability of terminating at (x_f, t_f) from x_i
    pub eta: Tensor<B, 3>,

    /// Success termination distribution: η⁺(x_f, t_f | x_i)
    ///
    /// Only present if composed with decomposition tracking
    pub eta_plus: Option<Tensor<B, 3>>,

    /// Failure termination distribution: η⁻(x_f, t_f | x_i)
    ///
    /// Only present if composed with decomposition tracking
    pub eta_minus: Option<Tensor<B, 3>>,

    /// Cumulative feasibility: κ_μ(x)
    ///
    /// Derived from η⁺: κ(x) = Σ_{x_f, t_f} η⁺(x_f, t_f | x)
    pub kappa: Tensor<B, 1>,

    /// Metadata
    pub n_states: usize,
    pub max_time: usize,

    /// Source option names (for debugging/tracing)
    pub source_options: Vec<String>,
}

impl<B: Backend> ComposedSTOK<B> {
    /// Convert to STOKKernel for further composition
    ///
    /// # Returns
    ///
    /// STOKKernel with same tensors (policy set to zeros as placeholder)
    pub fn to_stok_kernel(&self) -> Result<STOKKernel<B>, StokError> {
        let device = self.eta.device();

        // Get or create decomposition
        let (eta_plus, eta_minus) = match (&self.eta_plus, &self.eta_minus) {
            (Some(ep), Some(em)) => (ep.clone(), em.clone()),
            _ => {
                // No decomposition available - put all mass in eta_plus
                // This is technically incorrect but allows chaining
                let eta_plus = self.eta.clone();
                let eta_minus = Tensor::zeros_like(&self.eta);
                (eta_plus, eta_minus)
            }
        };

        // Policy is undefined for composed options
        let policy = Tensor::zeros([self.n_states], &device);

        Ok(STOKKernel {
            eta_plus,
            eta_minus,
            kappa: self.kappa.clone(),
            policy,
            dims: STOKDimensions::new(self.n_states, self.max_time),
        })
    }

    /// Get device this composed STOK is allocated on
    pub fn device(&self) -> B::Device {
        self.eta.device()
    }
}

/// Compute composed time horizon
///
/// When composing two options with time horizons T₁ and T₂,
/// the convolution extends the time dimension to T₁ + T₂ - 1.
///
/// # Arguments
///
/// * `t1` - Maximum time horizon of first option
/// * `t2` - Maximum time horizon of second option
///
/// # Returns
///
/// Maximum time horizon of composed option
///
/// # Example
///
/// ```rust,ignore
/// assert_eq!(composed_time_horizon(5, 3), 7);  // 5 + 3 - 1
/// assert_eq!(composed_time_horizon(1, 1), 1);
/// ```
pub fn composed_time_horizon(t1: usize, t2: usize) -> usize {
    // Convolution of two sequences of length T₁ and T₂
    // produces sequence of length T₁ + T₂ - 1
    t1 + t2 - 1
}

/// Compose two SOKs via matrix multiplication
///
/// Implements Equation [19]: χ_μ(x_μ | x) = Σ_{x_{f1}} χ_{o2}(x_μ | x_{f1}) · χ_{o1}(x_{f1} | x)
///
/// # Arguments
///
/// * `sok1` - First option (executed first)
/// * `sok2` - Second option (executed after sok1 terminates)
///
/// # Returns
///
/// Composed SOK representing sequential execution
///
/// # Errors
///
/// Returns error if state spaces don't match
///
/// # Example
///
/// ```rust,ignore
/// let sok1 = StateOptionKernel::from_stok(&stok1);
/// let sok2 = StateOptionKernel::from_stok(&stok2);
/// let composed = compose_soks(&sok1, &sok2)?;
/// ```
pub fn compose_soks<B: Backend>(
    sok1: &StateOptionKernel<B>,
    sok2: &StateOptionKernel<B>,
) -> Result<StateOptionKernel<B>, StokError> {
    // Validate compatibility
    if sok1.n_states != sok2.n_states {
        return Err(StokError::DimensionMismatch {
            expected: vec![sok1.n_states],
            got: vec![sok2.n_states],
        });
    }

    // χ_μ(x_f | x_i) = Σ_{x_m} χ_2(x_f | x_m) · χ_1(x_m | x_i)
    //                = (χ_1 @ χ_2)[i, f]
    //
    // Note: χ_1[i, m] = P(reach x_m | start x_i) under o1
    //       χ_2[m, f] = P(reach x_f | start x_m) under o2
    //       χ_μ[i, f] = Σ_m χ_1[i, m] * χ_2[m, f]

    let chi_composed = sok1.chi.clone().matmul(sok2.chi.clone());

    // Compose success/failure if both have decomposition
    let (chi_plus, chi_minus) = match (
        &sok1.chi_plus,
        &sok1.chi_minus,
        &sok2.chi_plus,
        &sok2.chi_minus,
    ) {
        (Some(cp1), Some(cm1), Some(cp2), Some(cm2)) => {
            // Success: o1 succeeds AND o2 succeeds
            // χ⁺_μ = χ⁺_1 @ χ⁺_2
            let cp_composed = cp1.clone().matmul(cp2.clone());

            // Failure: o1 fails OR (o1 succeeds AND o2 fails)
            // χ⁻_μ = χ⁻_1 @ (χ⁺_2 + χ⁻_2) + χ⁺_1 @ χ⁻_2
            //      = χ⁻_1 @ I + χ⁺_1 @ χ⁻_2
            //      = χ⁻_1 + χ⁺_1 @ χ⁻_2
            //
            // Actually for SOK: if o1 fails, sequence fails (χ⁻_1 contribution)
            // If o1 succeeds, we propagate through o2
            let cm_composed =
                cm1.clone().matmul(cp2.clone() + cm2.clone()) + cp1.clone().matmul(cm2.clone());

            (Some(cp_composed), Some(cm_composed))
        }
        _ => (None, None),
    };

    Ok(StateOptionKernel {
        chi: chi_composed,
        chi_plus,
        chi_minus,
        n_states: sok1.n_states,
    })
}

/// Compose two STOKs via Chapman-Kolmogorov equation
///
/// Implements Equation [18]:
/// ```text
/// η_μ(x_μ, t_μ | x) = Σ_{x_{f1}} Σ_{t_{f1}} η_{o2}(x_μ, t_μ - t_{f1} | x_{f1}) · η_{o1}(x_{f1}, t_{f1} | x)
/// ```
///
/// # Arguments
///
/// * `stok1` - First option (executed first)
/// * `stok2` - Second option (executed after stok1 terminates)
///
/// # Returns
///
/// Composed STOK with extended time horizon
///
/// # Time Complexity
///
/// O(S³ · T₁ · T₂) where S is state space size, T₁ and T₂ are time horizons
///
/// # Example
///
/// ```rust,ignore
/// let composed = compose_stoks(&stok1, &stok2)?;
/// assert_eq!(composed.max_time, stok1.max_time() + stok2.max_time() - 1);
/// ```
pub fn compose_stoks<B: Backend>(
    stok1: &STOKKernel<B>,
    stok2: &STOKKernel<B>,
) -> Result<ComposedSTOK<B>, StokError> {
    // Validate compatibility
    if stok1.n_states() != stok2.n_states() {
        return Err(StokError::DimensionMismatch {
            expected: vec![stok1.n_states()],
            got: vec![stok2.n_states()],
        });
    }

    let s = stok1.n_states();
    let t1 = stok1.max_time();
    let t2 = stok2.max_time();
    let t_composed = composed_time_horizon(t1, t2);
    let device = stok1.device();

    // Get combined STOKs: η** = η⁺ + η⁻
    let eta1 = stok1.eta_plus.clone() + stok1.eta_minus.clone(); // [S, S, T1]
    let eta2 = stok2.eta_plus.clone() + stok2.eta_minus.clone(); // [S, S, T2]

    // Initialize composed tensor
    let mut eta_composed = Tensor::zeros([s, s, t_composed], &device);

    // Compute Chapman-Kolmogorov convolution
    // For each output time t_μ, sum over all valid (t_1, t_2) pairs where t_1 + t_2 = t_μ
    for t_mu in 0..t_composed {
        // Valid range for t_1:
        // - t_1 >= 0
        // - t_1 < t1 (must be valid index in η_1)
        // - t_mu - t_1 >= 0 (t_2 must be non-negative)
        // - t_mu - t_1 < t2 (must be valid index in η_2)

        let t1_min = t_mu.saturating_sub(t2 - 1);
        let t1_max = t_mu.min(t1 - 1);

        // Skip if no valid t_1 values
        if t1_min > t1_max {
            continue;
        }

        // Accumulate contributions from all valid (t_1, t_2) pairs
        let mut slice_sum = Tensor::zeros([s, s], &device);

        for t_1 in t1_min..=t1_max {
            let t_2 = t_mu - t_1;

            // Get time slices
            // η_1[:, :, t_1] has shape [S, S]
            // η_1[x_i, x_m, t_1] = P(terminate at x_m, time t_1 | start x_i)
            let eta1_t = get_time_slice(&eta1, t_1); // [S, S]

            // η_2[:, :, t_2] has shape [S, S]
            // η_2[x_m, x_f, t_2] = P(terminate at x_f, time t_2 | start x_m)
            let eta2_t = get_time_slice(&eta2, t_2); // [S, S]

            // Compose: marginalize over intermediate state x_m
            // Result[x_i, x_f] = Σ_{x_m} η_1[x_i, x_m] * η_2[x_m, x_f]
            // This is matrix multiplication: η_1 @ η_2
            let contribution = eta1_t.matmul(eta2_t);

            slice_sum = slice_sum + contribution;
        }

        // Store composed slice
        eta_composed = set_time_slice(&eta_composed, &slice_sum, t_mu);
    }

    // Compute derived quantities
    // κ_μ(x) = Σ_{x_f, t_f} η_μ(x_f, t_f | x)
    // eta_composed [S, S, T] -> sum over time (dim 2) -> [S, S] -> sum over final states (dim 1) -> [S]
    let kappa = eta_composed
        .clone()
        .sum_dim(2)
        .squeeze::<2>() // [S, S, T] -> [S, S]
        .sum_dim(1)
        .squeeze::<1>(); // [S, S] -> [S]

    Ok(ComposedSTOK {
        eta: eta_composed,
        eta_plus: None, // Basic composition doesn't track decomposition
        eta_minus: None,
        kappa,
        n_states: s,
        max_time: t_composed,
        source_options: vec!["o1".to_string(), "o2".to_string()],
    })
}

/// Compose two STOKs while preserving success/failure decomposition
///
/// Tracks η⁺ and η⁻ through composition to maintain event interpretability.
///
/// # Decomposition Rules
///
/// - **Sequence succeeds**: o1 succeeds AND o2 succeeds
/// - **Sequence fails**: o1 fails OR (o1 succeeds AND o2 fails)
///
/// # Arguments
///
/// * `stok1` - First option
/// * `stok2` - Second option
///
/// # Returns
///
/// Composed STOK with η⁺/η⁻ decomposition preserved
pub fn compose_stoks_with_decomposition<B: Backend>(
    stok1: &STOKKernel<B>,
    stok2: &STOKKernel<B>,
) -> Result<ComposedSTOK<B>, StokError> {
    // Validate compatibility
    if stok1.n_states() != stok2.n_states() {
        return Err(StokError::DimensionMismatch {
            expected: vec![stok1.n_states()],
            got: vec![stok2.n_states()],
        });
    }

    let s = stok1.n_states();
    let t1 = stok1.max_time();
    let t2 = stok2.max_time();
    let t_composed = composed_time_horizon(t1, t2);
    let device = stok1.device();

    // Initialize decomposed outputs
    let mut eta_plus_composed = Tensor::zeros([s, s, t_composed], &device);
    let mut eta_minus_composed = Tensor::zeros([s, s, t_composed], &device);

    // Case 1: o1 fails (η_1⁻)
    // The sequence fails at the same time and state o1 fails
    // No composition needed - just copy η_1⁻ to output
    for t in 0..t1 {
        let eta1_minus_t = get_time_slice(&stok1.eta_minus, t);
        eta_minus_composed = set_time_slice(&eta_minus_composed, &eta1_minus_t, t);
    }

    // Case 2: o1 succeeds (η_1⁺), then o2 executes
    // Need to compose η_1⁺ with both η_2⁺ and η_2⁻
    for t_1 in 0..t1 {
        let eta1_plus_t = get_time_slice(&stok1.eta_plus, t_1); // [S, S]

        for t_2 in 0..t2 {
            let t_mu = t_1 + t_2;

            // Sub-case 2a: o2 succeeds -> full sequence succeeds
            let eta2_plus_t = get_time_slice(&stok2.eta_plus, t_2);
            let success_contribution = eta1_plus_t.clone().matmul(eta2_plus_t);

            let current_plus = get_time_slice(&eta_plus_composed, t_mu);
            eta_plus_composed = set_time_slice(
                &eta_plus_composed,
                &(current_plus + success_contribution),
                t_mu,
            );

            // Sub-case 2b: o2 fails -> sequence fails
            let eta2_minus_t = get_time_slice(&stok2.eta_minus, t_2);
            let failure_contribution = eta1_plus_t.clone().matmul(eta2_minus_t);

            let current_minus = get_time_slice(&eta_minus_composed, t_mu);
            eta_minus_composed = set_time_slice(
                &eta_minus_composed,
                &(current_minus + failure_contribution),
                t_mu,
            );
        }
    }

    // Combined η
    let eta_composed = eta_plus_composed.clone() + eta_minus_composed.clone();

    // κ from η⁺: sum over time (dim 2) then final states (dim 1)
    let kappa = eta_plus_composed
        .clone()
        .sum_dim(2)
        .squeeze::<2>() // [S, S, T] -> [S, S]
        .sum_dim(1)
        .squeeze::<1>(); // [S, S] -> [S]

    Ok(ComposedSTOK {
        eta: eta_composed,
        eta_plus: Some(eta_plus_composed),
        eta_minus: Some(eta_minus_composed),
        kappa,
        n_states: s,
        max_time: t_composed,
        source_options: vec!["o1".to_string(), "o2".to_string()],
    })
}

// ============================================================================
// Tensor Slice Utilities
// ============================================================================

/// Extract time slice from 3D tensor
///
/// # Arguments
///
/// * `tensor` - 3D tensor with shape [S, S, T]
/// * `t` - Time index to extract
///
/// # Returns
///
/// 2D tensor with shape [S, S] representing slice at time t
fn get_time_slice<B: Backend>(tensor: &Tensor<B, 3>, t: usize) -> Tensor<B, 2> {
    let dims = tensor.dims();
    let s1 = dims[0];
    let s2 = dims[1];

    tensor
        .clone()
        .slice([0..s1, 0..s2, t..(t + 1)])
        .reshape([s1, s2])
}

/// Set time slice in 3D tensor
///
/// # Arguments
///
/// * `tensor` - 3D tensor with shape [S, S, T]
/// * `slice` - 2D tensor with shape [S, S] to insert
/// * `t` - Time index to set
///
/// # Returns
///
/// Updated tensor with slice inserted at time t
fn set_time_slice<B: Backend>(
    tensor: &Tensor<B, 3>,
    slice: &Tensor<B, 2>,
    t: usize,
) -> Tensor<B, 3> {
    let dims = tensor.dims();
    let s1 = dims[0];
    let s2 = dims[1];

    // Reshape slice to [S, S, 1] for assignment
    let slice_3d = slice.clone().reshape([s1, s2, 1]);

    tensor
        .clone()
        .slice_assign([0..s1, 0..s2, t..(t + 1)], slice_3d)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::default_device;

    #[test]
    fn test_composed_time_horizon() {
        assert_eq!(composed_time_horizon(5, 3), 7); // 5 + 3 - 1
        assert_eq!(composed_time_horizon(1, 1), 1);
        assert_eq!(composed_time_horizon(10, 10), 19);
        assert_eq!(composed_time_horizon(1, 5), 5);
    }

    #[test]
    fn test_get_set_time_slice() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create 3D tensor
        let tensor = Tensor::<DefaultBackend, 3>::zeros([3, 3, 5], &device);

        // Create slice to insert
        let slice: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &device);

        // Set slice at time 2
        let updated = set_time_slice(&tensor, &slice, 2);

        // Verify slice was set correctly
        let retrieved = get_time_slice(&updated, 2);

        let diff: f32 = (retrieved - slice).abs().max().into_scalar().elem();
        assert!(diff < 1e-5);

        // Other time slices should still be zero
        let slice_0 = get_time_slice(&updated, 0);
        let sum: f32 = slice_0.sum().into_scalar().elem();
        assert!(sum.abs() < 1e-5);
    }

    #[test]
    fn test_compose_soks_matrix_mult() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create two simple SOKs
        let chi1: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.5, 0.5], [0.3, 0.7]], &device);
        let chi2: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.6, 0.4], [0.2, 0.8]], &device);

        let sok1 = StateOptionKernel {
            chi: chi1.clone(),
            chi_plus: None,
            chi_minus: None,
            n_states: 2,
        };

        let sok2 = StateOptionKernel {
            chi: chi2.clone(),
            chi_plus: None,
            chi_minus: None,
            n_states: 2,
        };

        // Compose
        let composed = compose_soks(&sok1, &sok2).unwrap();

        // Should equal matrix multiplication
        let expected = chi1.matmul(chi2);

        let diff: f32 = (composed.chi - expected).abs().max().into_scalar().elem();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_compose_soks_normalized() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create normalized SOKs
        let chi1: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.4, 0.6], [0.9, 0.1]], &device);
        let chi2: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.3, 0.7], [0.5, 0.5]], &device);

        let sok1 = StateOptionKernel::from_chi(chi1);
        let sok2 = StateOptionKernel::from_chi(chi2);

        // Compose
        let composed = compose_soks(&sok1, &sok2).unwrap();

        // Should be normalized
        assert!(composed.validate(1e-5).is_ok());
    }

    #[test]
    fn test_compose_stoks_simple() {
        use crate::backend::DefaultBackend;
        use crate::types::STOKDimensions;
        let device = default_device();

        // Create simple STOKs manually
        let n_states = 2;
        let max_time = 2;

        // STOK 1: deterministic transition 0 -> 1 at time 1
        let mut eta1_plus = Tensor::zeros([n_states, n_states, max_time], &device);
        eta1_plus = set_time_slice(
            &eta1_plus,
            &Tensor::from_floats([[0.0, 1.0], [0.0, 0.0]], &device),
            1,
        );

        let stok1 = STOKKernel {
            eta_plus: eta1_plus,
            eta_minus: Tensor::zeros([n_states, n_states, max_time], &device),
            kappa: Tensor::from_floats([1.0, 0.0], &device),
            policy: Tensor::zeros([n_states], &device),
            dims: STOKDimensions::new(n_states, max_time),
        };

        // STOK 2: identity (stay in same state at time 0)
        let mut eta2_plus = Tensor::zeros([n_states, n_states, max_time], &device);
        eta2_plus = set_time_slice(&eta2_plus, &Tensor::eye(n_states, &device), 0);

        let stok2 = STOKKernel {
            eta_plus: eta2_plus,
            eta_minus: Tensor::zeros([n_states, n_states, max_time], &device),
            kappa: Tensor::ones([n_states], &device),
            policy: Tensor::zeros([n_states], &device),
            dims: STOKDimensions::new(n_states, max_time),
        };

        // Compose
        let composed = compose_stoks(&stok1, &stok2).unwrap();

        // Verify time horizon
        assert_eq!(composed.max_time, 3); // 2 + 2 - 1

        // Verify normalization for state 0 (state 1 has no η in inputs, so won't normalize)
        let eta_sums = composed
            .eta
            .clone()
            .sum_dim(2)
            .squeeze::<2>()
            .sum_dim(1)
            .squeeze::<1>();

        // Check state 0 (which has valid η in stok1)
        let sum_tensor: Tensor<DefaultBackend, 1> = eta_sums.clone().slice([0..1]);
        let sum: f32 = sum_tensor.into_scalar().elem();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "State 0 should normalize to 1.0, got {}",
            sum
        );

        // Note: State 1 won't normalize because stok1 has no η for state 1
    }

    #[test]
    fn test_composed_stok_to_kernel() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let eta: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 3, 5], &device);
        let kappa: Tensor<DefaultBackend, 1> = Tensor::zeros([3], &device);

        let composed = ComposedSTOK {
            eta,
            eta_plus: None,
            eta_minus: None,
            kappa,
            n_states: 3,
            max_time: 5,
            source_options: vec![],
        };

        let kernel = composed.to_stok_kernel().unwrap();
        assert_eq!(kernel.n_states(), 3);
        assert_eq!(kernel.max_time(), 5);
    }
}
