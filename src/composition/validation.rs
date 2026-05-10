//! # Composition Validation Utilities
//!
//! Functions for validating composition correctness and consistency.

use super::chapman_kolmogorov::{compose_stoks, ComposedSTOK};
use super::sok::StateOptionKernel;
use crate::stok::STOKKernel;
use crate::types::StokError;
use burn::prelude::*;

/// Validate a composed STOK
///
/// Checks normalization, probability bounds, and decomposition consistency.
///
/// # Arguments
///
/// * `composed` - Composed STOK to validate
/// * `tolerance` - Maximum allowed deviation
///
/// # Returns
///
/// Ok if all checks pass, Err with details otherwise
///
/// # Checks Performed
///
/// 1. **Normalization**: Σ_{x_f, t_f} η(x_f, t_f | x) ≈ 1 for all x
/// 2. **Probability bounds**: All values in [0, 1]
/// 3. **Decomposition**: If η⁺ and η⁻ present, verify η = η⁺ + η⁻
/// 4. **κ consistency**: κ = Σ_{x_f, t_f} η⁺
pub fn validate_composed_stok<B: Backend>(
    composed: &ComposedSTOK<B>,
    tolerance: f32,
) -> Result<(), StokError> {
    // Check 1: Normalization - η sums to 1 for each initial state
    let eta_sums = composed
        .eta
        .clone()
        .sum_dim(2)
        .squeeze::<2>() // [S, S, T] -> [S, S]
        .sum_dim(1)
        .squeeze::<1>(); // [S, S] -> [S]
    let ones = Tensor::ones([composed.n_states], &composed.eta.device());
    let diff: f32 = (eta_sums - ones).abs().max().into_scalar().elem();

    if diff > tolerance {
        return Err(StokError::NotNormalized {
            sum: 1.0 + diff,
            expected: 1.0,
            tolerance,
        });
    }

    // Check 2: All probabilities in [0, 1]
    let min_val: f32 = composed.eta.clone().min().into_scalar().elem();
    let max_val: f32 = composed.eta.clone().max().into_scalar().elem();

    if min_val < -tolerance || max_val > 1.0 + tolerance {
        return Err(StokError::InvalidProbability {
            value: if min_val < 0.0 { min_val } else { max_val },
            context: "Composed STOK values out of bounds".to_string(),
        });
    }

    // Check 3: η = η⁺ + η⁻ (Eq [17]). After PP-201 decomposition is always present.
    let sum = composed.eta_plus.clone() + composed.eta_minus.clone();
    let decomp_diff: f32 = (composed.eta.clone() - sum).abs().max().into_scalar().elem();
    if decomp_diff > tolerance {
        return Err(StokError::NotNormalized {
            sum: decomp_diff,
            expected: 0.0,
            tolerance,
        });
    }

    // Check 4: κ = Σ η⁺ (Eq [15]).
    let kappa_from_eta = composed
        .eta_plus
        .clone()
        .sum_dim(2)
        .squeeze::<2>()
        .sum_dim(1)
        .squeeze::<1>();
    let kappa_diff: f32 = (composed.kappa.clone() - kappa_from_eta)
        .abs()
        .max()
        .into_scalar()
        .elem();
    if kappa_diff > tolerance {
        return Err(StokError::NotNormalized {
            sum: kappa_diff,
            expected: 0.0,
            tolerance,
        });
    }

    Ok(())
}

/// Validate composition consistency between source and result
///
/// Checks that composition produces reasonable results relative to inputs.
///
/// # Arguments
///
/// * `stok1` - First source STOK
/// * `stok2` - Second source STOK
/// * `composed` - Result of composition
/// * `tolerance` - Maximum allowed deviation
///
/// # Returns
///
/// Ok if consistent, Err otherwise
///
/// # Checks Performed
///
/// 1. **Feasibility non-increasing**: κ_μ(x) ≤ κ_1(x) for all x
/// 2. **State space consistency**: Same number of states
pub fn validate_composition_consistency<B: Backend>(
    stok1: &STOKKernel<B>,
    stok2: &STOKKernel<B>,
    composed: &ComposedSTOK<B>,
    tolerance: f32,
) -> Result<(), StokError> {
    // Check state space consistency
    if composed.n_states != stok1.n_states() || composed.n_states != stok2.n_states() {
        return Err(StokError::DimensionMismatch {
            expected: vec![stok1.n_states(), stok2.n_states()],
            got: vec![composed.n_states],
        });
    }

    // Check: Composed κ should be <= source κ_1 element-wise
    // (sequence can't be more feasible than first option alone)
    let kappa1 = stok1.kappa.clone();
    let kappa_composed = composed.kappa.clone();

    let violations: f32 = kappa_composed
        .clone()
        .greater(kappa1.clone() + tolerance)
        .float()
        .sum()
        .into_scalar()
        .elem();

    if violations > 0.0 {
        return Err(StokError::CompositionError {
            message: format!(
                "Composed κ exceeds source κ in {} states (should be non-increasing)",
                violations
            ),
        });
    }

    Ok(())
}

/// Validate SOK-STOK consistency
///
/// Verifies that marginalizing a STOK produces the same result as
/// composing the marginalized SOKs.
///
/// # Arguments
///
/// * `stok` - Source STOK
/// * `sok` - SOK derived from stok
/// * `tolerance` - Maximum allowed difference
///
/// # Returns
///
/// Ok if consistent, Err otherwise
///
/// # Property Verified
///
/// SOK(STOK) should equal time-marginalized STOK
pub fn validate_sok_stok_consistency<B: Backend>(
    stok: &STOKKernel<B>,
    sok: &StateOptionKernel<B>,
    tolerance: f32,
) -> Result<(), StokError> {
    // SOK should equal time-marginalized STOK
    let stok_marginalized = stok.state_option_kernel(); // [S, S]
    let diff: f32 = (stok_marginalized - sok.chi.clone())
        .abs()
        .max()
        .into_scalar()
        .elem();

    if diff > tolerance {
        return Err(StokError::CompositionError {
            message: format!("SOK differs from marginalized STOK by {}", diff),
        });
    }

    Ok(())
}

/// Validate that composition is associative (within tolerance)
///
/// Verifies: (o1 ∘ o2) ∘ o3 ≈ o1 ∘ (o2 ∘ o3)
///
/// # Arguments
///
/// * `stok1`, `stok2`, `stok3` - Three STOKs to compose
/// * `tolerance` - Maximum allowed difference
///
/// # Returns
///
/// Ok if associative, Err otherwise
pub fn validate_associativity<B: Backend>(
    stok1: &STOKKernel<B>,
    stok2: &STOKKernel<B>,
    stok3: &STOKKernel<B>,
    tolerance: f32,
) -> Result<(), StokError> {
    // (o1 ∘ o2) ∘ o3
    let left_first = {
        let temp = compose_stoks(stok1, stok2)?;
        let temp_kernel = temp.to_stok_kernel()?;
        compose_stoks(&temp_kernel, stok3)?
    };

    // o1 ∘ (o2 ∘ o3)
    let right_first = {
        let temp = compose_stoks(stok2, stok3)?;
        let temp_kernel = temp.to_stok_kernel()?;
        compose_stoks(stok1, &temp_kernel)?
    };

    // Compare results
    let kappa_diff: f32 = (left_first.kappa - right_first.kappa)
        .abs()
        .max()
        .into_scalar()
        .elem();

    if kappa_diff > tolerance {
        return Err(StokError::CompositionError {
            message: format!(
                "Associativity violated: κ differs by {} (tolerance: {})",
                kappa_diff, tolerance
            ),
        });
    }

    let eta_diff: f32 = (left_first.eta - right_first.eta)
        .abs()
        .max()
        .into_scalar()
        .elem();

    if eta_diff > tolerance {
        return Err(StokError::CompositionError {
            message: format!(
                "Associativity violated: η differs by {} (tolerance: {})",
                eta_diff, tolerance
            ),
        });
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::default_device;

    #[test]
    fn test_validate_composed_stok_identity() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        // Create identity STOK
        let n_states = 3;
        let max_time = 1;

        let mut eta_plus: Tensor<DefaultBackend, 3> =
            Tensor::zeros([n_states, n_states, max_time], &device);
        eta_plus = eta_plus.slice_assign(
            [0..n_states, 0..n_states, 0..1],
            Tensor::eye(n_states, &device).reshape([n_states, n_states, 1]),
        );

        let composed = ComposedSTOK {
            eta: eta_plus.clone(),
            eta_plus,
            eta_minus: Tensor::zeros([n_states, n_states, max_time], &device),
            kappa: Tensor::ones([n_states], &device),
            n_states,
            max_time,
            source_options: vec!["identity".to_string()],
        };

        // Should pass validation
        assert!(validate_composed_stok(&composed, 1e-5).is_ok());
    }

    #[test]
    fn test_validate_composition_consistency_feasibility() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let stok1: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        let stok2: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);

        // Create composed with higher κ than source (invalid)
        let composed: ComposedSTOK<DefaultBackend> = ComposedSTOK {
            eta_plus: Tensor::zeros([3, 3, 9], &device),
            eta_minus: Tensor::zeros([3, 3, 9], &device),
            eta: Tensor::zeros([3, 3, 9], &device),
            kappa: Tensor::ones([3], &device), // Higher than stok1.kappa (zeros)
            n_states: 3,
            max_time: 9,
            source_options: vec![],
        };

        // Should fail (composed κ > source κ)
        let result = validate_composition_consistency(&stok1, &stok2, &composed, 1e-5);
        assert!(result.is_err());
    }
}
