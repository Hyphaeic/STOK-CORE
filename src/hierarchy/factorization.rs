//! # Factorized STOK for High-Dimensional Planning
//!
//! Implements STOK factorization (Theorem 2.1, Equation [23]):
//!
//! ```text
//! η̃(z_f, x_f, t_f | z, x) = ξ(t_f | z, x) · ρ_π(x_f | x, t_f) · ∏_k ρ_k(z_k,f | z_k, t_f)
//! ```
//!
//! This factorization avoids the curse of dimensionality by decomposing
//! a product-space STOK into:
//! - Base-level STOK (computed via feasibility iteration)
//! - High-level SPKs (default dynamics for each HL space)
//! - Temporal Event Function (when does first event occur?)
//!
//! ## Key Insight
//!
//! Instead of computing η̃ on |X × Z| states (intractable),
//! we factor it into |X| + n·|Zₖ| components (tractable).

use super::product_space::{HLState, ProductSpaceDims, ProductState};
use crate::prediction::{CumulativeEventFunction, StatePredictionKernel, TemporalEventFunction};
use crate::stok::STOKKernel;
use crate::types::StokError;
use burn::prelude::*;
use rand::Rng;

/// Factorized STOK representation
///
/// Stores components of the factorization rather than full product-space STOK.
///
/// # Memory
///
/// - Full product-space: O(|X|² × |Z₁|² × ... × |Zₙ|² × T)
/// - Factorized: O(|X|² × T + Σₖ |Zₖ|² × T)
///
/// For honey badger (|X|=100, |Σ|=8, |Y|=10, T=30):
/// - Full: 100² × 8² × 10² × 30 = 1.92 billion values
/// - Factorized: 100² × 30 + 8² × 30 + 10² × 30 = 304,920 values
/// - **Reduction: 6,300×**
pub struct FactorizedSTOK<B: Backend> {
    /// Base-level STOK: η_π(x_f, t_f | x)
    ///
    /// Computed via standard feasibility iteration on X
    pub base_stok: STOKKernel<B>,

    /// High-level SPKs: ρ_k(z_k,f | z_k, t_f) for each space Z_k
    ///
    /// Predicts HL state evolution under default dynamics
    pub hl_spks: Vec<StatePredictionKernel<B>>,

    /// Temporal event function: ξ(t_f | z, x)
    ///
    /// Probability that first event (goal/constraint/region-exit) occurs at t_f
    /// Optional: if None, assumes no HL events (κ̄_z = 1)
    pub tef: Option<TemporalEventFunction<B>>,

    /// Cumulative event functions for each HL space
    ///
    /// Used to compute TEF: ξ(t) = (1 - ∏ κ̄ₖ(t)) - (1 - ∏ κ̄ₖ(t-1))
    pub cefs: Vec<CumulativeEventFunction<B>>,

    /// Product-space dimensions
    pub dims: ProductSpaceDims,

    /// Device
    device: B::Device,
}

impl<B: Backend> FactorizedSTOK<B> {
    /// Evaluate factorized STOK at a product-space state-time pair
    ///
    /// Implements Equation [23]:
    /// ```text
    /// η̃(z_f, x_f, t_f | z_i, x_i) = ξ(t_f | z_i, x_i) · ρ_π(x_f | x_i, t_f) · ∏_k ρ_k(z_k,f | z_k,i, t_f)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial product-space state
    /// * `final_state` - Final product-space state
    /// * `time` - Time horizon
    ///
    /// # Returns
    ///
    /// Probability of terminating at (z_f, x_f, t_f) from (z_i, x_i)
    pub fn evaluate(
        &self,
        initial: &ProductState,
        final_state: &ProductState,
        time: usize,
    ) -> f32 {
        assert!(time > 0 && time <= self.base_stok.max_time());

        // Component 1: Base-level STOK
        // η_π(x_f, t | x_i) from base STOK
        let eta_base: f32 = self
            .base_stok
            .combined_stok()
            .slice([
                initial.base..(initial.base + 1),
                final_state.base..(final_state.base + 1),
                time..(time + 1),
            ])
            .into_scalar()
            .elem();

        // Component 2: High-level SPKs
        // ∏_k ρ_k(z_k,f | z_k,i, t)
        let mut hl_prob = 1.0f32;
        for (k, spk) in self.hl_spks.iter().enumerate() {
            let z_i = initial.hl_states[k].as_discrete();
            let z_f = final_state.hl_states[k].as_discrete();

            let rho_k_dist = spk.predict(z_i, time);
            let rho_k: f32 = rho_k_dist
                .slice([z_f..(z_f + 1)])
                .into_scalar()
                .elem();

            hl_prob *= rho_k;

            if hl_prob == 0.0 {
                return 0.0; // Early termination
            }
        }

        // Component 3: Temporal Event Function
        // ξ(t | z_i, x_i) - if available
        let tef_prob = if let Some(ref tef) = self.tef {
            // Need flat index of initial state for TEF lookup
            let initial_flat = initial.to_flat_index(&self.dims);
            tef.event_at_time(initial_flat, time)
        } else {
            1.0 // No HL events case (Corollary 6.1)
        };

        eta_base * hl_prob * tef_prob
    }

    /// Sample from factorized STOK
    ///
    /// # Arguments
    ///
    /// * `initial` - Initial product-space state
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (final product-space state, time)
    pub fn sample(&self, initial: &ProductState, rng: &mut impl Rng) -> (ProductState, usize) {
        // 1. Sample base-level (x_f, t_f) from base STOK
        let (x_f, t_f) = crate::planning::STOKSampler::sample_termination(
            &self.base_stok,
            initial.base,
            rng,
        );

        // 2. For each HL space, predict z_f using SPK at sampled time
        let mut hl_final = Vec::with_capacity(self.hl_spks.len());
        for (k, spk) in self.hl_spks.iter().enumerate() {
            let z_i = initial.hl_states[k].as_discrete();
            let z_f = spk.sample(z_i, t_f, rng);
            hl_final.push(HLState::Discrete(z_f));
        }

        let final_state = ProductState {
            base: x_f,
            hl_states: hl_final,
        };

        (final_state, t_f)
    }

    /// Get feasibility at product-space state
    ///
    /// Approximation: κ̃(s) ≈ κ_base(x) (ignores HL event probability)
    ///
    /// For exact computation, would need to integrate over time with TEF weighting.
    pub fn feasibility_approx(&self, state: &ProductState) -> f32 {
        self.base_stok
            .kappa
            .clone()
            .slice([state.base..(state.base + 1)])
            .into_scalar()
            .elem()
    }

    /// Get base-level STOK component
    pub fn base(&self) -> &STOKKernel<B> {
        &self.base_stok
    }

    /// Get HL SPK for specific space
    pub fn hl_spk(&self, space_id: usize) -> Option<&StatePredictionKernel<B>> {
        self.hl_spks.get(space_id)
    }

    /// Get device
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Assemble factorized STOK from components
///
/// # Arguments
///
/// * `base_stok` - STOK for base-level space X (from feasibility iteration)
/// * `hl_kernels` - Transition kernels P_z_k(z'|z,α) for each HL space
/// * `default_actions` - Default HL action for each space (region-specific)
/// * `dims` - Product-space dimensions
///
/// # Returns
///
/// Factorized STOK ready for evaluation/sampling
///
/// # Example
///
/// ```rust,ignore
/// // Honey badger example
/// let base_stok = solve_task_mdp(&gridworld_mdp)?.kernel;
///
/// let P_hydration = create_hydration_dynamics(10);  // 10 hydration levels
/// let P_logic = create_logic_dynamics(8);           // 2³ binary states
///
/// let factorized = assemble_factorized_stok(
///     base_stok,
///     vec![P_logic, P_hydration],
///     vec![ALPHA_NO_CHANGE, ALPHA_DEHYDRATE],  // Default actions
///     ProductSpaceDims::new(100, vec![8, 10]),
/// )?;
/// ```
pub fn assemble_factorized_stok<B: Backend>(
    base_stok: STOKKernel<B>,
    hl_kernels: Vec<Tensor<B, 3>>, // P_z_k(z'|z,α) for each space
    default_actions: Vec<usize>,    // Default α^ℓ for each space
    dims: ProductSpaceDims,
) -> Result<FactorizedSTOK<B>, StokError> {
    assert_eq!(
        hl_kernels.len(),
        default_actions.len(),
        "Must have default action for each HL kernel"
    );
    assert_eq!(
        hl_kernels.len(),
        dims.n_hl_spaces(),
        "HL kernels must match dimensions"
    );

    let device = base_stok.device();
    let max_time = base_stok.max_time();

    // 1. Build SPKs from HL kernels with default actions
    let mut hl_spks = Vec::with_capacity(hl_kernels.len());

    for (k, (P_z_k, &alpha_default)) in hl_kernels.iter().zip(&default_actions).enumerate() {
        // Validate kernel is square
        let kernel_dims = P_z_k.dims();
        assert_eq!(
            kernel_dims[0], kernel_dims[2],
            "HL kernel {} must be square",
            k
        );
        assert_eq!(
            kernel_dims[0], dims.hl_size(k),
            "HL kernel {} size doesn't match dims",
            k
        );

        // Extract P_z_k[:, :, α_default] as default Markov chain
        let n_z = kernel_dims[0];
        let P_default = P_z_k
            .clone()
            .slice([0..n_z, alpha_default..(alpha_default + 1), 0..n_z])
            .reshape([n_z, n_z]);

        // Create SPK with precomputed powers
        let spk = StatePredictionKernel::new(P_default, max_time);
        hl_spks.push(spk);
    }

    // 2. CEFs placeholder (would compute from HL-only STOKs)
    // For now, assume no HL events (Corollary 6.1 case)
    let cefs = vec![];

    // 3. TEF placeholder
    // For no HL events: ξ(t | s) reduces to base-level only
    let tef = None;

    Ok(FactorizedSTOK {
        base_stok,
        hl_spks,
        tef,
        cefs,
        dims,
        device,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::types::STOKDimensions;

    #[test]
    fn test_assemble_factorized_stok_simple() {
        let device = default_device();

        // Base STOK: 3-state chain
        let base_stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);

        // Single HL space: 2-state chain with deterministic transition
        let P_hl_data: &[f32] = &[
            0.0, 1.0, // State 0, action 0: always go to state 1
            1.0, 0.0, // State 0, action 1: always go to state 0
            1.0, 0.0, // State 1, action 0: always go to state 0
            0.0, 1.0, // State 1, action 1: always go to state 1
        ];
        let P_hl: Tensor<DefaultBackend, 1> = Tensor::from_floats(P_hl_data, &device);
        let P_hl: Tensor<DefaultBackend, 3> = P_hl.reshape([2, 2, 2]);

        let dims = ProductSpaceDims::new(3, vec![2]);

        // Assemble with default action 0 (0→1)
        let factorized = assemble_factorized_stok(base_stok, vec![P_hl], vec![0], dims).unwrap();

        assert_eq!(factorized.hl_spks.len(), 1);
        assert_eq!(factorized.dims.total_size, 6); // 3 × 2
    }

    #[test]
    fn test_factorized_evaluation_deterministic() {
        let device = default_device();

        // Create simple deterministic base STOK
        // Always terminate at state 2 at time 1
        let mut base_stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);

        let mut eta_plus: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 3, 5], &device);
        // Set η⁺(x_f=2, t=1 | x_i=0) = 1.0
        eta_plus = eta_plus.slice_assign(
            [0..1, 2..3, 1..2],
            Tensor::from_floats([[[1.0]]], &device),
        );
        base_stok.eta_plus = eta_plus;
        base_stok.kappa = Tensor::from_floats([1.0, 0.0, 0.0], &device);

        // HL: identity (stay in same state)
        let P_hl_identity: Tensor<DefaultBackend, 3> = Tensor::eye(2, &device)
            .reshape([2, 1, 2])
            .repeat_dim(1, 2); // Repeat for action dimension

        let dims = ProductSpaceDims::new(3, vec![2]);
        let factorized =
            assemble_factorized_stok(base_stok, vec![P_hl_identity], vec![0], dims.clone())
                .unwrap();

        // Evaluate: (x_i=0, z_i=0) → (x_f=2, z_f=0, t=1)
        let initial = ProductState::new(0).with_hl_discrete(0);
        let final_state = ProductState::new(2).with_hl_discrete(0);

        let prob = factorized.evaluate(&initial, &final_state, 1);

        // Should be 1.0 (deterministic base, identity HL)
        assert!((prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_factorized_sampling() {
        let device = default_device();
        let mut rng = rand::thread_rng();

        // Simple base STOK (terminates at t=1)
        let mut base_stok: STOKKernel<DefaultBackend> = STOKKernel::new(2, 3, &device);
        let mut eta_plus: Tensor<DefaultBackend, 3> = Tensor::zeros([2, 2, 3], &device);
        // Set η⁺(x_f=same, t=1 | x_i) = 1.0 (identity at t=1)
        eta_plus = eta_plus.slice_assign(
            [0..2, 0..2, 1..2],
            Tensor::eye(2, &device).reshape([2, 2, 1]),
        );
        base_stok.eta_plus = eta_plus;
        base_stok.kappa = Tensor::ones([2], &device);

        // HL: identity transition
        let P_hl: Tensor<DefaultBackend, 3> =
            Tensor::eye(3, &device).reshape([3, 1, 3]).repeat_dim(1, 2);

        let dims = ProductSpaceDims::new(2, vec![3]);
        let factorized = assemble_factorized_stok(base_stok, vec![P_hl], vec![0], dims).unwrap();

        // Sample from (x=0, z=1)
        let initial = ProductState::new(0).with_hl_discrete(1);
        let (final_state, time) = factorized.sample(&initial, &mut rng);

        // Should stay in same state (identity dynamics at t=1)
        assert_eq!(final_state.base, 0);
        assert_eq!(final_state.hl_states[0].as_discrete(), 1);
        assert_eq!(time, 1); // t=1 for this identity setup
    }

    #[test]
    fn test_factorized_multi_hl() {
        let device = default_device();

        // Base: 5 states
        let base_stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 10, &device);

        // HL space 1: 3 states
        let P_hl1: Tensor<DefaultBackend, 3> =
            Tensor::eye(3, &device).reshape([3, 1, 3]).repeat_dim(1, 2);

        // HL space 2: 4 states
        let P_hl2: Tensor<DefaultBackend, 3> =
            Tensor::eye(4, &device).reshape([4, 1, 4]).repeat_dim(1, 2);

        let dims = ProductSpaceDims::new(5, vec![3, 4]);

        let factorized =
            assemble_factorized_stok(base_stok, vec![P_hl1, P_hl2], vec![0, 0], dims.clone())
                .unwrap();

        assert_eq!(factorized.hl_spks.len(), 2);
        assert_eq!(factorized.dims.total_size, 60); // 5 × 3 × 4
    }

    #[test]
    fn test_feasibility_approx() {
        let device = default_device();

        let mut base_stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        base_stok.kappa = Tensor::from_floats([0.8, 0.5, 1.0], &device);

        let P_hl: Tensor<DefaultBackend, 3> =
            Tensor::eye(2, &device).reshape([2, 1, 2]).repeat_dim(1, 2);

        let dims = ProductSpaceDims::new(3, vec![2]);
        let factorized = assemble_factorized_stok(base_stok, vec![P_hl], vec![0], dims).unwrap();

        // Feasibility should match base κ
        let state_0 = ProductState::new(0).with_hl_discrete(0);
        assert!((factorized.feasibility_approx(&state_0) - 0.8).abs() < 1e-5);

        let state_2 = ProductState::new(2).with_hl_discrete(1);
        assert!((factorized.feasibility_approx(&state_2) - 1.0).abs() < 1e-5);
    }
}
