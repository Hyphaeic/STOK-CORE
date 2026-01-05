//! # Multi-Option Sequence Composition
//!
//! Utilities for composing sequences of more than two options.
//!
//! ## Associativity
//!
//! STOK composition is associative:
//! ```text
//! (η_1 ∘ η_2) ∘ η_3 = η_1 ∘ (η_2 ∘ η_3)
//! ```
//!
//! This allows flexible composition order.

use super::chapman_kolmogorov::{compose_stoks, ComposedSTOK};
use super::sok::StateOptionKernel;
use crate::stok::STOKKernel;
use crate::types::StokError;
use burn::prelude::Backend;

/// Sequence of options to be composed
///
/// Builder pattern for constructing option sequences.
///
/// # Example
///
/// ```rust,ignore
/// let sequence = OptionSequence::new()
///     .then(stok1, Some("navigate"))
///     .then(stok2, Some("pickup"))
///     .then(stok3, Some("deliver"));
///
/// let composed = compose_sequence(&sequence)?;
/// ```
#[derive(Clone)]
pub struct OptionSequence<B: Backend> {
    /// Ordered list of STOKs to compose
    pub options: Vec<STOKKernel<B>>,

    /// Optional names for debugging/tracing
    pub names: Vec<String>,
}

impl<B: Backend> OptionSequence<B> {
    /// Create empty option sequence
    pub fn new() -> Self {
        Self {
            options: vec![],
            names: vec![],
        }
    }

    /// Append an option to the sequence
    ///
    /// # Arguments
    ///
    /// * `stok` - STOK kernel to add
    /// * `name` - Optional name for debugging
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn then(mut self, stok: STOKKernel<B>, name: Option<&str>) -> Self {
        self.options.push(stok);
        self.names.push(name.unwrap_or("unnamed").to_string());
        self
    }

    /// Get number of options in sequence
    pub fn len(&self) -> usize {
        self.options.len()
    }

    /// Check if sequence is empty
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Get option at index
    pub fn get(&self, index: usize) -> Option<&STOKKernel<B>> {
        self.options.get(index)
    }
}

impl<B: Backend> Default for OptionSequence<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compose a sequence of STOKs pairwise
///
/// Composes options left-to-right: ((o1 ∘ o2) ∘ o3) ∘ ...
///
/// # Arguments
///
/// * `sequence` - Ordered sequence of options
///
/// # Returns
///
/// Composed STOK representing sequential execution
///
/// # Errors
///
/// - Returns error if sequence is empty
/// - Returns error if state spaces don't match
/// - Propagates composition errors
///
/// # Example
///
/// ```rust,ignore
/// let sequence = OptionSequence::new()
///     .then(stok1, Some("step1"))
///     .then(stok2, Some("step2"))
///     .then(stok3, Some("step3"));
///
/// let composed = compose_sequence(&sequence)?;
/// assert_eq!(composed.source_options.len(), 3);
/// ```
pub fn compose_sequence<B: Backend>(
    sequence: &OptionSequence<B>,
) -> Result<ComposedSTOK<B>, StokError> {
    if sequence.is_empty() {
        return Err(StokError::EmptySequence);
    }

    if sequence.len() == 1 {
        // Single option: convert to ComposedSTOK format
        let stok = &sequence.options[0];
        let _device = stok.device();

        let eta = stok.eta_plus.clone() + stok.eta_minus.clone();
        let kappa = stok.kappa.clone();

        return Ok(ComposedSTOK {
            eta,
            eta_plus: Some(stok.eta_plus.clone()),
            eta_minus: Some(stok.eta_minus.clone()),
            kappa,
            n_states: stok.n_states(),
            max_time: stok.max_time(),
            source_options: vec![sequence.names[0].clone()],
        });
    }

    // Compose pairwise: ((o1 ∘ o2) ∘ o3) ∘ ...
    // Composition is associative, so left-to-right is valid
    let mut result = compose_stoks(&sequence.options[0], &sequence.options[1])?;

    for i in 2..sequence.len() {
        // Convert intermediate result to STOKKernel for next composition
        let intermediate_stok = result.to_stok_kernel()?;
        result = compose_stoks(&intermediate_stok, &sequence.options[i])?;
    }

    // Update source information
    result.source_options = sequence.names.clone();

    Ok(result)
}

/// Compose a sequence of SOKs efficiently
///
/// Since SOK composition is matrix multiplication, we can chain:
/// χ_μ = χ_1 @ χ_2 @ χ_3 @ ...
///
/// # Arguments
///
/// * `soks` - Ordered slice of SOKs
///
/// # Returns
///
/// Composed SOK representing sequential execution
///
/// # Errors
///
/// - Returns error if sequence is empty
/// - Returns error if state spaces don't match
///
/// # Performance
///
/// More efficient than STOK sequence composition (no time dimension)
pub fn compose_sok_sequence<B: Backend>(
    soks: &[StateOptionKernel<B>],
) -> Result<StateOptionKernel<B>, StokError> {
    if soks.is_empty() {
        return Err(StokError::EmptySequence);
    }

    if soks.len() == 1 {
        return Ok(soks[0].clone());
    }

    // Validate all have same state space
    let n_states = soks[0].n_states;
    for sok in soks.iter().skip(1) {
        if sok.n_states != n_states {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_states],
                got: vec![sok.n_states],
            });
        }
    }

    // χ_μ = χ_1 @ χ_2 @ χ_3 @ ...
    // Matrix multiplication is associative
    let mut result = soks[0].chi.clone();

    for sok in soks.iter().skip(1) {
        result = result.matmul(sok.chi.clone());
    }

    Ok(StateOptionKernel {
        chi: result,
        chi_plus: None, // Decomposition lost in sequence
        chi_minus: None,
        n_states,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::default_device;

    #[test]
    fn test_option_sequence_builder() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 10, &device);

        let sequence = OptionSequence::new()
            .then(stok.clone(), Some("first"))
            .then(stok.clone(), Some("second"))
            .then(stok, Some("third"));

        assert_eq!(sequence.len(), 3);
        assert_eq!(sequence.names[0], "first");
        assert_eq!(sequence.names[2], "third");
    }

    #[test]
    fn test_compose_sequence_single() {
        use crate::backend::DefaultBackend;
        let device = default_device();

        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 10, &device);
        let sequence = OptionSequence::new().then(stok, Some("only"));

        let composed = compose_sequence(&sequence).unwrap();

        assert_eq!(composed.n_states, 5);
        assert_eq!(composed.max_time, 10);
        assert_eq!(composed.source_options.len(), 1);
    }

    #[test]
    fn test_compose_sequence_empty() {
        use crate::backend::DefaultBackend;
        let sequence: OptionSequence<DefaultBackend> = OptionSequence::new();

        let result = compose_sequence(&sequence);
        assert!(result.is_err());

        match result {
            Err(StokError::EmptySequence) => {} // Expected
            _ => panic!("Should return EmptySequence error"),
        }
    }

    #[test]
    fn test_compose_sok_sequence_identity_chain() {
        use crate::backend::DefaultBackend;
        use burn::prelude::{ElementConversion, Tensor};
        let device = default_device();

        // Three identity SOKs
        let chi_identity: Tensor<DefaultBackend, 2> = Tensor::eye(3, &device);
        let sok = StateOptionKernel::from_chi(chi_identity);

        let soks = vec![sok.clone(), sok.clone(), sok.clone()];

        // Composition of identities is identity
        let composed = compose_sok_sequence(&soks).unwrap();

        // Should still be identity matrix
        let identity = Tensor::eye(3, &device);
        let diff: f32 = (composed.chi - identity).abs().max().into_scalar().elem();
        assert!(diff < 1e-5);
    }

    #[test]
    fn test_compose_sok_sequence_associativity() {
        use crate::backend::DefaultBackend;
        use burn::prelude::{ElementConversion, Tensor};
        let device = default_device();

        let chi1: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.5, 0.5], [0.3, 0.7]], &device);
        let chi2: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.6, 0.4], [0.2, 0.8]], &device);
        let chi3: Tensor<DefaultBackend, 2> =
            Tensor::from_floats([[0.4, 0.6], [0.7, 0.3]], &device);

        let sok1 = StateOptionKernel::from_chi(chi1.clone());
        let sok2 = StateOptionKernel::from_chi(chi2.clone());
        let sok3 = StateOptionKernel::from_chi(chi3.clone());

        // Compose as sequence
        let sequence_result = compose_sok_sequence(&[sok1, sok2, sok3]).unwrap();

        // Compose manually: χ_1 @ χ_2 @ χ_3
        let manual_result = chi1.matmul(chi2).matmul(chi3);

        let diff: f32 = (sequence_result.chi - manual_result)
            .abs()
            .max()
            .into_scalar()
            .elem();
        assert!(diff < 1e-5);
    }
}
