//! Numerical stability utilities for STOK computation.
//!
//! Provides floating-point comparison, tensor norms, and probability
//! clamping functions essential for convergence detection and
//! maintaining mathematical invariants.
//!
//! # Reference
//! These utilities support the numerical stability requirements
//! described in Ringstrom & Schrater (2025) Appendix E.

use burn::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Default convergence tolerance for feasibility iteration.
/// L∞ norm threshold below which we consider κ converged.
pub const DEFAULT_CONVERGENCE_TOLERANCE: f32 = 1e-6;

/// Tolerance for probability comparisons and validation.
pub const DEFAULT_PROBABILITY_TOLERANCE: f32 = 1e-7;

/// Maximum probability value (slightly less than 1.0 for numerical safety).
pub const MAX_PROBABILITY: f32 = 1.0 - 1e-7;

/// Minimum probability value (slightly more than 0.0 for numerical safety).
pub const MIN_PROBABILITY: f32 = 1e-7;

/// Default maximum iterations for feasibility iteration.
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Default interval for checking convergence (every N iterations).
pub const DEFAULT_CHECK_INTERVAL: usize = 1;

// ============================================================================
// Scalar Utilities
// ============================================================================

/// Check if two floating-point values are approximately equal.
///
/// # Arguments
/// * `a` - First value
/// * `b` - Second value  
/// * `tolerance` - Maximum allowed absolute difference
///
/// # Returns
/// `true` if `|a - b| < tolerance`
#[inline]
pub fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() < tolerance
}

/// Check if a floating-point value is approximately zero.
///
/// # Arguments
/// * `a` - Value to check
/// * `tolerance` - Maximum allowed absolute value
///
/// # Returns
/// `true` if `|a| < tolerance`
#[inline]
pub fn approx_zero(a: f32, tolerance: f32) -> bool {
    a.abs() < tolerance
}

/// Check if a value is a valid probability (in [0, 1]).
#[inline]
pub fn is_valid_probability(p: f32) -> bool {
    p >= 0.0 && p <= 1.0
}

/// Clamp a scalar to valid probability range [0, 1].
#[inline]
pub fn clamp_probability(p: f32) -> f32 {
    p.clamp(0.0, 1.0)
}

// ============================================================================
// Tensor Utilities
// ============================================================================

/// Compute L∞ norm (maximum absolute value) of a 1D tensor.
///
/// This is the primary convergence metric:
/// `||v||_∞ = max_i |v_i|`
///
/// # Arguments
/// * `tensor` - 1D tensor
///
/// # Returns
/// Maximum absolute value as f32 (requires GPU→CPU sync)
pub fn linf_norm<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor.clone().abs().max().into_scalar().elem()
}

/// Compute L∞ distance between two 1D tensors.
///
/// Primary convergence check:
/// `||a - b||_∞ = max_i |a_i - b_i|`
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor (must have same shape as `a`)
///
/// # Returns
/// Maximum absolute element-wise difference (requires GPU→CPU sync)
///
/// # Note
/// This function synchronizes GPU→CPU for the scalar extraction.
/// For performance-critical code, consider batching convergence checks.
pub fn linf_distance<B: Backend>(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
    (a.clone() - b.clone()).abs().max().into_scalar().elem()
}

/// Compute L2 norm of a 1D tensor.
///
/// `||v||_2 = sqrt(Σ v_i²)`
pub fn l2_norm<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    let squared = tensor.clone() * tensor.clone();
    squared.sum().into_scalar().elem::<f32>().sqrt()
}

/// Clamp all tensor values to valid probability range [0, 1].
///
/// Guards against numerical drift that could produce invalid probabilities.
///
/// # Arguments
/// * `tensor` - Tensor of any dimensionality
///
/// # Returns
/// Tensor with all values clamped to [0, 1]
pub fn clamp_probabilities<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clamp(0.0, 1.0)
}

/// Check if all tensor values are valid probabilities.
///
/// # Returns
/// `true` if all values are in [0, 1] within tolerance
pub fn validate_probability_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    tolerance: f32,
) -> bool {
    let min_val: f32 = tensor.clone().min().into_scalar().elem();
    let max_val: f32 = tensor.clone().max().into_scalar().elem();
    min_val >= -tolerance && max_val <= 1.0 + tolerance
}

/// Check if a tensor sums to approximately 1.0 (for normalization checks).
pub fn validate_normalized<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    tolerance: f32,
) -> bool {
    let sum: f32 = tensor.clone().sum().into_scalar().elem();
    approx_eq(sum, 1.0, tolerance)
}

/// Extract tensor data to Vec<f32> for debugging/inspection.
///
/// # Warning
/// This transfers data from GPU to CPU. Use sparingly in hot paths.
pub fn tensor_to_vec<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
    tensor.clone().into_data().to_vec().unwrap()
}

/// Safe division that returns 0 when dividing by zero.
///
/// Useful for probability calculations where division by zero
/// should yield zero rather than NaN/Inf.
#[inline]
pub fn safe_div(a: f32, b: f32) -> f32 {
    if b.abs() < f32::EPSILON {
        0.0
    } else {
        a / b
    }
}

// ============================================================================
// Tensor Creation Utilities
// ============================================================================

/// Create a diagonal matrix from a 1D tensor.
///
/// # Arguments
/// * `diag` - 1D tensor of diagonal values, shape [N]
///
/// # Returns
/// 2D tensor of shape [N, N] with `diag` on the diagonal, zeros elsewhere
pub fn diag_matrix<B: Backend>(diag: &Tensor<B, 1>) -> Tensor<B, 2> {
    let n = diag.dims()[0];
    let device = diag.device();

    // Create identity matrix and multiply by diagonal values
    let eye: Tensor<B, 2> = Tensor::eye(n, &device);
    let diag_expanded = diag.clone().unsqueeze_dim(1); // [N, 1]

    eye * diag_expanded
}

/// Create an identity-like STOK (for composition identity tests).
///
/// Returns a kernel where starting from state x, you end at state x
/// at time 0 with probability 1.
pub fn identity_stok_slice<B: Backend>(n_states: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::eye(n_states, device)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};

    #[test]
    fn test_approx_eq() {
        assert!(approx_eq(1.0, 1.0, 1e-6));
        assert!(approx_eq(1.0, 1.0 + 1e-7, 1e-6));
        assert!(!approx_eq(1.0, 1.1, 1e-6));
    }

    #[test]
    fn test_approx_zero() {
        assert!(approx_zero(0.0, 1e-6));
        assert!(approx_zero(1e-7, 1e-6));
        assert!(!approx_zero(1e-5, 1e-6));
    }

    #[test]
    fn test_is_valid_probability() {
        assert!(is_valid_probability(0.0));
        assert!(is_valid_probability(0.5));
        assert!(is_valid_probability(1.0));
        assert!(!is_valid_probability(-0.1));
        assert!(!is_valid_probability(1.1));
    }

    #[test]
    fn test_linf_norm() {
        let device = default_device();
        let tensor: Tensor<DefaultBackend, 1> = Tensor::from_floats([1.0, -3.0, 2.0], &device);

        let norm = linf_norm(&tensor);
        assert!(approx_eq(norm, 3.0, 1e-6));
    }

    #[test]
    fn test_linf_distance() {
        let device = default_device();
        let a: Tensor<DefaultBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let b: Tensor<DefaultBackend, 1> = Tensor::from_floats([1.0, 2.5, 3.0], &device);

        let dist = linf_distance(&a, &b);
        assert!(approx_eq(dist, 0.5, 1e-6));
    }

    #[test]
    fn test_linf_distance_zero() {
        let device = default_device();
        let a: Tensor<DefaultBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
        let b = a.clone();

        let dist = linf_distance(&a, &b);
        assert!(approx_zero(dist, 1e-6));
    }

    #[test]
    fn test_clamp_probabilities() {
        let device = default_device();
        let tensor: Tensor<DefaultBackend, 1> = Tensor::from_floats([-0.1, 0.5, 1.2], &device);

        let clamped = clamp_probabilities(tensor);
        let data: Vec<f32> = clamped.into_data().to_vec().unwrap();

        assert!(approx_eq(data[0], 0.0, 1e-6));
        assert!(approx_eq(data[1], 0.5, 1e-6));
        assert!(approx_eq(data[2], 1.0, 1e-6));
    }

    #[test]
    fn test_validate_probability_tensor() {
        let device = default_device();

        let valid: Tensor<DefaultBackend, 1> = Tensor::from_floats([0.0, 0.5, 1.0], &device);
        assert!(validate_probability_tensor(&valid, 1e-6));

        let invalid: Tensor<DefaultBackend, 1> = Tensor::from_floats([-0.1, 0.5, 1.0], &device);
        assert!(!validate_probability_tensor(&invalid, 1e-6));
    }

    #[test]
    fn test_diag_matrix() {
        let device = default_device();
        let diag: Tensor<DefaultBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

        let mat = diag_matrix(&diag);
        let data: Vec<f32> = mat.into_data().to_vec().unwrap();

        // Should be [[1,0,0], [0,2,0], [0,0,3]]
        assert!(approx_eq(data[0], 1.0, 1e-6)); // [0,0]
        assert!(approx_eq(data[1], 0.0, 1e-6)); // [0,1]
        assert!(approx_eq(data[4], 2.0, 1e-6)); // [1,1]
        assert!(approx_eq(data[8], 3.0, 1e-6)); // [2,2]
    }

    #[test]
    fn test_l2_norm() {
        let device = default_device();
        let tensor: Tensor<DefaultBackend, 1> = Tensor::from_floats([3.0, 4.0], &device);

        let norm = l2_norm(&tensor);
        assert!(approx_eq(norm, 5.0, 1e-5)); // 3² + 4² = 25, √25 = 5
    }
}
