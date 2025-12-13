//! Numerical stability utilities for STOK computation.
//!
//! Provides floating-point comparison, tensor norms, and probability
//! clamping functions essential for convergence detection and
//! maintaining mathematical invariants.

use burn::prelude::*;

// ============================================================================
// Constants
// ============================================================================

/// Default convergence tolerance for feasibility iteration (L∞ norm)
pub const DEFAULT_CONVERGENCE_TOLERANCE: f32 = 1e-6;

/// Default tolerance for probability comparisons
pub const DEFAULT_PROBABILITY_TOLERANCE: f32 = 1e-7;

/// Maximum probability value (for clamping)
pub const MAX_PROBABILITY: f32 = 1.0 - 1e-7;

/// Minimum probability value (for clamping)
pub const MIN_PROBABILITY: f32 = 1e-7;

// ============================================================================
// Scalar Utilities
// ============================================================================

/// Approximate equality for floating-point comparison.
///
/// Returns true if `|a - b| < tolerance`.
///
/// # Arguments
/// * `a` - First value
/// * `b` - Second value  
/// * `tolerance` - Maximum allowed difference
///
/// # Example
/// ```ignore
/// assert!(approx_eq(1.0, 1.0 + 1e-8, 1e-6));
/// ```
#[inline]
pub fn approx_eq(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() < tolerance
}

/// Check if value is approximately zero.
///
/// Convenience wrapper around `approx_eq(a, 0.0, tolerance)`.
#[inline]
pub fn approx_zero(a: f32, tolerance: f32) -> bool {
    a.abs() < tolerance
}

/// Check if value is in valid probability range [0, 1].
#[inline]
pub fn is_valid_probability(p: f32) -> bool {
    p >= 0.0 && p <= 1.0
}

// ============================================================================
// Tensor Utilities
// ============================================================================

/// Compute L∞ norm (maximum absolute value) of a 1D tensor.
///
/// Used for convergence detection: `‖κ_new - κ_old‖_∞`
///
/// # Arguments
/// * `tensor` - Input tensor of shape [N]
///
/// # Returns
/// Maximum absolute value as f32
pub fn linf_norm<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    // TODO: Implement
    // Hint: tensor.clone().abs().max().into_scalar()
    todo!("Implement linf_norm")
}

/// Compute L∞ distance between two 1D tensors.
///
/// Primary convergence metric: `δ = ‖a - b‖_∞`
///
/// # Arguments
/// * `a` - First tensor of shape [N]
/// * `b` - Second tensor of shape [N]
///
/// # Returns
/// Maximum absolute element-wise difference
pub fn linf_distance<B: Backend>(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> f32 {
    // TODO: Implement
    // Hint: (a.clone() - b.clone()).abs().max().into_scalar()
    todo!("Implement linf_distance")
}

/// Clamp tensor values to valid probability range [0, 1].
///
/// Guards against numerical drift that could produce invalid probabilities.
///
/// # Arguments
/// * `tensor` - Input tensor of any shape
///
/// # Returns
/// Tensor with all values clamped to [0, 1]
pub fn clamp_probabilities<B: Backend, const N: usize>(
    tensor: Tensor<B, N>,
) -> Tensor<B, N> {
    // TODO: Implement
    // Hint: tensor.clamp(0.0, 1.0)
    todo!("Implement clamp_probabilities")
}

/// Check if all tensor values are in valid probability range.
///
/// # Returns
/// `true` if all values in [0, 1], `false` otherwise
pub fn all_valid_probabilities<B: Backend, const N: usize>(
    tensor: &Tensor<B, N>,
) -> bool {
    // TODO: Implement
    // Hint: Check min >= 0 and max <= 1
    todo!("Implement all_valid_probabilities")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_eq() {
        assert!(approx_eq(1.0, 1.0, 1e-6));
        assert!(approx_eq(1.0, 1.0 + 1e-8, 1e-6));
        assert!(!approx_eq(1.0, 1.1, 1e-6));
    }

    #[test]
    fn test_approx_zero() {
        assert!(approx_zero(0.0, 1e-6));
        assert!(approx_zero(1e-8, 1e-6));
        assert!(!approx_zero(0.1, 1e-6));
    }

    #[test]
    fn test_is_valid_probability() {
        assert!(is_valid_probability(0.0));
        assert!(is_valid_probability(0.5));
        assert!(is_valid_probability(1.0));
        assert!(!is_valid_probability(-0.1));
        assert!(!is_valid_probability(1.1));
    }

    // TODO: Add tensor tests after implementing functions
    // These require a Backend, so use #[cfg(test)] with test device
}
