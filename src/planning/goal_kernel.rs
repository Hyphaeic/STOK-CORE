//! # Goal Kernel Manager
//!
//! Two distinct abstractions live here:
//!
//! - **`GoalKernel`** — a *reduced* low-dimensional manager that stores a flat
//!   `GoalId -> STOKKernel` map. Useful for problems where the planning state
//!   is just `x ∈ X` (no HL component) and Eq [24]'s boundary-action update
//!   isn't needed. This is example-grade scaffolding from the pre-audit
//!   implementation.
//!
//! - **`FactorizedGoalKernel`** — the paper-faithful Goal Kernel `G` from
//!   Eq [24] (PP-501..505). Carries per-goal `FactorizedSTOK`s plus the
//!   shared affordance and HL kernels needed for the one-step boundary-action
//!   update that advances time and HL state across an option boundary.
//!
//! New code targeting Theorem 2.1 / Eq [24] / Algorithm 2 should use
//! `FactorizedGoalKernel`. The reduced `GoalKernel` is retained for the
//! existing low-dim tests and example callers — see PP-504.

use crate::composition::StateOptionKernel;
use crate::hierarchy::{FactorizedAffordance, FactorizedSTOK, ProductSpaceDims, ProductState};
use crate::stok::STOKKernel;
use crate::types::{GoalId, StokError};
use burn::prelude::*;
use std::collections::HashMap;

/// Metadata about a goal
#[derive(Clone, Debug)]
pub struct GoalInfo {
    /// Unique goal identifier
    pub id: GoalId,

    /// Human-readable name
    pub name: String,

    /// Target state (if goal is to reach specific state)
    pub target_state: Option<usize>,

    /// Maximum time horizon for this goal's STOK
    pub max_time: usize,
}

/// Goal Kernel: manages collection of STOKs for planning
///
/// Implements the goal kernel G from Equation [24]:
/// ```text
/// G(z', x', t_f + t + 1 | (z, x)^ℓ, t, o_{ℓ,g}) = ...
/// ```
///
/// Provides:
/// - Storage and retrieval of goal-conditioned STOKs
/// - Feasibility queries
/// - Affordance computation (which goals are reachable)
///
/// # Example
///
/// ```rust,ignore
/// let mut kernel = GoalKernel::new(n_states, device);
/// kernel.add_goal(GoalId(0), stok1, "waypoint", Some(5))?;
/// kernel.add_goal(GoalId(1), stok2, "goal", Some(10))?;
///
/// let feasible = kernel.feasible_goals(current_state);
/// let kappa = kernel.query_feasibility(GoalId(0), current_state);
/// ```
pub struct GoalKernel<B: Backend> {
    /// STOKs indexed by goal ID
    stoks: HashMap<GoalId, STOKKernel<B>>,

    /// SOKs (precomputed from STOKs for fast queries)
    soks: HashMap<GoalId, StateOptionKernel<B>>,

    /// Goal metadata
    goal_info: HashMap<GoalId, GoalInfo>,

    /// Shared state space size
    n_states: usize,

    /// Device
    device: B::Device,
}

impl<B: Backend> GoalKernel<B> {
    /// Create empty goal kernel
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states in state space
    /// * `device` - Device for tensor operations
    pub fn new(n_states: usize, device: B::Device) -> Self {
        Self {
            stoks: HashMap::new(),
            soks: HashMap::new(),
            goal_info: HashMap::new(),
            n_states,
            device,
        }
    }

    /// Add a goal with its STOK
    ///
    /// # Arguments
    ///
    /// * `id` - Unique goal identifier
    /// * `stok` - STOK kernel for this goal
    /// * `name` - Human-readable name
    /// * `target_state` - Optional target state index
    ///
    /// # Returns
    ///
    /// Ok if added successfully
    ///
    /// # Errors
    ///
    /// Returns error if state spaces don't match
    pub fn add_goal(
        &mut self,
        id: GoalId,
        stok: STOKKernel<B>,
        name: impl Into<String>,
        target_state: Option<usize>,
    ) -> Result<(), StokError> {
        // Validate state space compatibility
        if stok.n_states() != self.n_states {
            return Err(StokError::DimensionMismatch {
                expected: vec![self.n_states],
                got: vec![stok.n_states()],
            });
        }

        // Precompute SOK for fast queries
        let sok = StateOptionKernel::from_stok(&stok);

        // Store metadata
        let info = GoalInfo {
            id,
            name: name.into(),
            target_state,
            max_time: stok.max_time(),
        };

        self.stoks.insert(id, stok);
        self.soks.insert(id, sok);
        self.goal_info.insert(id, info);

        Ok(())
    }

    /// Get STOK for a goal
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    ///
    /// # Returns
    ///
    /// Reference to STOK if it exists
    pub fn get_stok(&self, goal: GoalId) -> Option<&STOKKernel<B>> {
        self.stoks.get(&goal)
    }

    /// Get SOK for a goal
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    ///
    /// # Returns
    ///
    /// Reference to SOK if it exists
    pub fn get_sok(&self, goal: GoalId) -> Option<&StateOptionKernel<B>> {
        self.soks.get(&goal)
    }

    /// Get goal info
    pub fn get_info(&self, goal: GoalId) -> Option<&GoalInfo> {
        self.goal_info.get(&goal)
    }

    /// Check if goal exists in kernel
    pub fn has_goal(&self, goal: GoalId) -> bool {
        self.stoks.contains_key(&goal)
    }

    /// Get number of goals in kernel
    pub fn n_goals(&self) -> usize {
        self.stoks.len()
    }

    /// Get device
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Get all goal IDs
    pub fn goal_ids(&self) -> Vec<GoalId> {
        self.stoks.keys().copied().collect()
    }

    /// Query feasibility of reaching goal from state
    ///
    /// Returns κ_g(x)
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID to query
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Feasibility value, or 0.0 if goal doesn't exist
    pub fn query_feasibility(&self, goal: GoalId, state: usize) -> f32 {
        self.stoks
            .get(&goal)
            .map(|stok| {
                stok.kappa
                    .clone()
                    .slice([state..(state + 1)])
                    .into_scalar()
                    .elem()
            })
            .unwrap_or(0.0)
    }

    /// Batch query: feasibility for all goals from state
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// HashMap mapping goal IDs to feasibility values
    pub fn query_all_feasibilities(&self, state: usize) -> HashMap<GoalId, f32> {
        self.stoks
            .iter()
            .map(|(id, stok)| {
                let kappa: f32 = stok
                    .kappa
                    .clone()
                    .slice([state..(state + 1)])
                    .into_scalar()
                    .elem();
                (*id, kappa)
            })
            .collect()
    }

    /// Get feasible goals from a state
    ///
    /// Returns all goals with κ_g(x) > 0
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Vector of feasible goal IDs
    pub fn feasible_goals(&self, state: usize) -> Vec<GoalId> {
        let mut goal_ids: Vec<GoalId> = self.stoks.keys().copied().collect();
        goal_ids.sort();

        goal_ids
            .into_iter()
            .filter(|id| self.query_feasibility(*id, state) > 0.0)
            .collect()
    }

    /// Check if a specific goal is feasible from state
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// true if κ_g(x) > 0
    pub fn is_feasible(&self, goal: GoalId, state: usize) -> bool {
        self.query_feasibility(goal, state) > 0.0
    }

    /// Query expected termination time for goal from state
    ///
    /// Computes E[t_f | x] = Σ_{x_f, t_f} t_f · η(x_f, t_f | x)
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID
    /// * `state` - Initial state
    ///
    /// # Returns
    ///
    /// Expected time, or None if goal doesn't exist
    pub fn query_expected_time(&self, goal: GoalId, state: usize) -> Option<f32> {
        let stok = self.stoks.get(&goal)?;

        // E[t_f | x] = Σ_{x_f, t_f} t_f * η(x_f, t_f | x)
        let eta = stok.combined_stok(); // [S, S, T]
        let t_max = stok.max_time();

        let mut expected_time = 0.0f32;
        for t in 0..t_max {
            let eta_t: f32 = eta
                .clone()
                .slice([state..(state + 1), 0..self.n_states, t..(t + 1)])
                .sum()
                .into_scalar()
                .elem();
            expected_time += (t as f32) * eta_t;
        }

        Some(expected_time)
    }
}

// ============================================================================
// PP-501: FactorizedGoalKernel — paper-faithful Goal Kernel `G` from Eq [24]
// ============================================================================

/// Per-goal entry in a `FactorizedGoalKernel`. Owns the `FactorizedSTOK` for
/// that goal plus `GoalInfo` metadata.
pub struct FactorizedGoalEntry<B: Backend> {
    pub option: FactorizedSTOK<B>,
    pub info: GoalInfo,
}

/// Paper-faithful Goal Kernel implementing Eq [24]:
///
/// ```text
/// G(z', x', t_f + t + 1 | (z, x)^ℓ, t, o_{ℓ,g})
///   = Σ_{z_f, α_f, x_f, a_f}
///       P_z(z' | z_f, α_f) · F(α' | x_f, a_f) · ρ_e(z_f | z, t_f)
///       · η_{o-e,g}(α_f, a_f, x_f, t_f | x)
/// ```
///
/// Carries:
/// - per-goal `FactorizedSTOK`s (the `η_{o,g}` × `ρ_z_k` factorization),
/// - the shared affordance `F(α | x, a)` for the boundary action that triggers
///   the next HL transition,
/// - the HL kernels `P_z_k(z' | z_f, α_k)` used to apply that boundary HL
///   transition,
/// - the product-space dimensions `(|X|, |Z_1|, ..., |Z_n|)` shared across
///   all goals.
///
/// Distinct from the reduced `GoalKernel` (above) which only holds flat
/// `STOKKernel`s. Use this struct for high-dimensional / Algorithm-2 planning.
pub struct FactorizedGoalKernel<B: Backend> {
    options: HashMap<GoalId, FactorizedGoalEntry<B>>,
    affordance: FactorizedAffordance<B>,
    hl_kernels: Vec<Tensor<B, 3>>,
    dims: ProductSpaceDims,
    device: B::Device,
}

impl<B: Backend> FactorizedGoalKernel<B> {
    /// Create an empty Goal Kernel.
    ///
    /// # Arguments
    ///
    /// * `affordance` — F(α | x, a) shared across all goals.
    /// * `hl_kernels` — `P_z_k(z' | z, α_k)` per HL space (same order as
    ///   `dims.hl_dims`).
    /// * `dims` — product-space dimensions.
    pub fn new(
        affordance: FactorizedAffordance<B>,
        hl_kernels: Vec<Tensor<B, 3>>,
        dims: ProductSpaceDims,
    ) -> Self {
        assert_eq!(
            hl_kernels.len(),
            dims.n_hl_spaces(),
            "must supply one HL kernel per HL space (got {} kernels, dims has {})",
            hl_kernels.len(),
            dims.n_hl_spaces()
        );
        let device = affordance.device();
        Self {
            options: HashMap::new(),
            affordance,
            hl_kernels,
            dims,
            device,
        }
    }

    /// Add a goal-conditioned option (a `FactorizedSTOK` for that goal).
    pub fn add_option(
        &mut self,
        goal_id: GoalId,
        option: FactorizedSTOK<B>,
        info: GoalInfo,
    ) -> Result<(), StokError> {
        if option.dims.base_size != self.dims.base_size
            || option.dims.hl_sizes != self.dims.hl_sizes
        {
            return Err(StokError::DimensionMismatch {
                expected: vec![self.dims.base_size]
                    .into_iter()
                    .chain(self.dims.hl_sizes.iter().copied())
                    .collect(),
                got: vec![option.dims.base_size]
                    .into_iter()
                    .chain(option.dims.hl_sizes.iter().copied())
                    .collect(),
            });
        }
        self.options.insert(goal_id, FactorizedGoalEntry { option, info });
        Ok(())
    }

    /// Number of goals in the kernel.
    pub fn n_goals(&self) -> usize {
        self.options.len()
    }

    /// Whether this goal exists.
    pub fn has_goal(&self, goal: GoalId) -> bool {
        self.options.contains_key(&goal)
    }

    /// Borrow the option for a goal (the `FactorizedSTOK`).
    pub fn option(&self, goal: GoalId) -> Option<&FactorizedSTOK<B>> {
        self.options.get(&goal).map(|e| &e.option)
    }

    /// Borrow goal metadata.
    pub fn info(&self, goal: GoalId) -> Option<&GoalInfo> {
        self.options.get(&goal).map(|e| &e.info)
    }

    /// All goal IDs in the kernel.
    pub fn goal_ids(&self) -> Vec<GoalId> {
        self.options.keys().copied().collect()
    }

    /// Borrow the shared affordance.
    pub fn affordance(&self) -> &FactorizedAffordance<B> {
        &self.affordance
    }

    /// Borrow the HL kernels.
    pub fn hl_kernels(&self) -> &[Tensor<B, 3>] {
        &self.hl_kernels
    }

    /// Borrow the product-space dimensions.
    pub fn dims(&self) -> &ProductSpaceDims {
        &self.dims
    }

    /// Query the (possibly approximate) feasibility of a goal from a product state.
    /// Routes to `FactorizedSTOK::feasibility_approx` — exact under Cor 6.1,
    /// upper-bound under the general Eq [23] path.
    pub fn query_feasibility(&self, goal: GoalId, state: &ProductState) -> f32 {
        self.options
            .get(&goal)
            .map(|e| e.option.feasibility_approx(state))
            .unwrap_or(0.0)
    }

    /// All goals with κ̃ ≥ threshold from the given product state.
    pub fn feasible_goals(&self, state: &ProductState, threshold: f32) -> Vec<GoalId> {
        let mut ids: Vec<GoalId> = self.options.keys().copied().collect();
        ids.sort();
        ids.into_iter()
            .filter(|g| self.query_feasibility(*g, state) >= threshold)
            .collect()
    }

    // ------------------------------------------------------------------------
    // PP-502: One-step boundary-action update (Eq [24])
    // ------------------------------------------------------------------------

    /// PP-502: one-step boundary-action update under option `o_{ℓ,g}`, given
    /// the option's terminal `(x_f, t_f)` and the boundary action `(α_f, a_f)`.
    ///
    /// Implements the inner part of Eq [24]:
    ///
    /// ```text
    /// next_z'_k = Σ_{z_f_k} P_z_k(z'_k | z_f_k, α_f_k) · ρ_z_k(z_f_k | z, t_f)
    /// ```
    ///
    /// Combined with the affordance `F(α_f | x_f, a_f)` factor (which the
    /// caller has already chosen by selecting `α_f`), this returns the
    /// distribution over the next HL state vector after the boundary HL
    /// transition fires.
    ///
    /// `time_after = t + t_f + 1` (the +1 accounts for the boundary step
    /// itself per Algorithm 2 line 18).
    ///
    /// # Arguments
    ///
    /// * `goal` — which option's data to use for the HL SPK
    /// * `initial_hl` — initial HL state vector `(z_1, ..., z_n)`
    /// * `boundary_hl_action` — `α_f` per HL space (one action index per space)
    /// * `t_f` — option terminal time
    ///
    /// # Returns
    ///
    /// Per-HL-space distributions `Vec<Tensor<B, 1>>` over the next HL state.
    /// One tensor per HL space; entry k has shape `[|Z_k|]`.
    pub fn boundary_hl_distribution(
        &self,
        goal: GoalId,
        initial_hl: &[usize],
        boundary_hl_action: &[usize],
        t_f: usize,
    ) -> Result<Vec<Tensor<B, 1>>, StokError> {
        assert_eq!(
            initial_hl.len(),
            self.dims.n_hl_spaces(),
            "initial_hl length must equal n_hl_spaces"
        );
        assert_eq!(
            boundary_hl_action.len(),
            self.dims.n_hl_spaces(),
            "boundary_hl_action length must equal n_hl_spaces"
        );

        let entry = self
            .options
            .get(&goal)
            .ok_or_else(|| StokError::DimensionMismatch {
                expected: vec![1],
                got: vec![0],
            })?;
        let option = &entry.option;

        let mut out = Vec::with_capacity(self.dims.n_hl_spaces());
        for k in 0..self.dims.n_hl_spaces() {
            let n_z_k = self.dims.hl_size(k);
            // ρ_z_k(z_f | z_k, t_f) — distribution over HL state after t_f steps under default.
            let rho_dist = option.hl_spks[k].predict(initial_hl[k], t_f); // shape [n_z_k]

            // P_z_k(z' | z_f, α_f_k) — boundary HL transition under boundary action.
            let alpha_k = boundary_hl_action[k];
            let n_a_z_k = self.hl_kernels[k].dims()[1];
            assert!(
                alpha_k < n_a_z_k,
                "boundary_hl_action[{}] = {} out of range for n_a_z = {}",
                k,
                alpha_k,
                n_a_z_k
            );
            // Slice P_z_k[:, α_f_k, :] → shape [n_z_k, n_z_k] mapping z_f → z'.
            let p_z_alpha: Tensor<B, 2> = self.hl_kernels[k]
                .clone()
                .slice([0..n_z_k, alpha_k..(alpha_k + 1), 0..n_z_k])
                .reshape([n_z_k, n_z_k]);

            // Marginalize z_f: out_dist[z'] = Σ_{z_f} ρ(z_f | z_k, t_f) · P_z(z' | z_f, α).
            // = ρ_dist (row vector) · p_z_alpha
            let next_dist =
                rho_dist.reshape([1, n_z_k]).matmul(p_z_alpha).reshape([n_z_k]);
            out.push(next_dist);
        }
        Ok(out)
    }

    /// PP-502 helper: marginalize the boundary BL action probability via the
    /// affordance — `F(α_f | x_f, a_f)` — to get the joint probability of a
    /// chosen `(α_f, a_f)` triple at the option's terminal `x_f`.
    ///
    /// The caller typically uses this to weight contributions to the Goal
    /// Kernel sum in Eq [24].
    pub fn boundary_affordance_prob(
        &self,
        x_f: usize,
        a_f: usize,
        boundary_hl_action: &[usize],
    ) -> f32 {
        use crate::hierarchy::{AffordanceFunction, HLAction};
        let hl = HLAction::Vector(boundary_hl_action.to_vec());
        self.affordance.probability(&hl, x_f, a_f)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use burn::prelude::Tensor;

    #[test]
    fn test_goal_kernel_creation() {
        let device = default_device();
        let kernel: GoalKernel<DefaultBackend> = GoalKernel::new(10, device);

        assert_eq!(kernel.n_goals(), 0);
        assert_eq!(kernel.n_states, 10);
    }

    #[test]
    fn test_add_goal() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 10, &device);
        kernel
            .add_goal(GoalId(0), stok, "test_goal", Some(4))
            .unwrap();

        assert_eq!(kernel.n_goals(), 1);
        assert!(kernel.has_goal(GoalId(0)));
        assert!(!kernel.has_goal(GoalId(1)));
    }

    #[test]
    fn test_add_goal_dimension_mismatch() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        // Create STOK with different state space
        let stok: STOKKernel<DefaultBackend> = STOKKernel::new(10, 10, &device);
        let result = kernel.add_goal(GoalId(0), stok, "mismatched", None);

        assert!(result.is_err());
        match result {
            Err(StokError::DimensionMismatch { .. }) => {} // Expected
            _ => panic!("Should return DimensionMismatch error"),
        }
    }

    #[test]
    fn test_query_feasibility() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(3, device.clone());

        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok.kappa = Tensor::from_floats([0.5, 0.8, 1.0], &device);

        kernel.add_goal(GoalId(0), stok, "goal", None).unwrap();

        assert!((kernel.query_feasibility(GoalId(0), 0) - 0.5).abs() < 1e-5);
        assert!((kernel.query_feasibility(GoalId(0), 1) - 0.8).abs() < 1e-5);
        assert!((kernel.query_feasibility(GoalId(0), 2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_feasible_goals() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(3, device.clone());

        // Goal 0: feasible from states 1, 2
        let mut stok0: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok0.kappa = Tensor::from_floats([0.0, 0.8, 1.0], &device);
        kernel.add_goal(GoalId(0), stok0, "goal0", None).unwrap();

        // Goal 1: feasible from all states
        let mut stok1: STOKKernel<DefaultBackend> = STOKKernel::new(3, 5, &device);
        stok1.kappa = Tensor::from_floats([1.0, 1.0, 1.0], &device);
        kernel.add_goal(GoalId(1), stok1, "goal1", None).unwrap();

        // From state 0: only goal 1 is feasible
        let feasible_from_0 = kernel.feasible_goals(0);
        assert_eq!(feasible_from_0.len(), 1);
        assert!(feasible_from_0.contains(&GoalId(1)));

        // From state 1: both goals feasible
        let feasible_from_1 = kernel.feasible_goals(1);
        assert_eq!(feasible_from_1.len(), 2);
    }
}
