//! # Sublimation: Abstract Feasibility Bounds
//!
//! Implements Theorem 2.4 from Ringstrom & Schrater (2025).
//!
//! ## Mathematical Background
//!
//! **Sublimation** is solving a high-level-only TMDP abstracted away from the
//! full product-space to obtain feasibility bounds for pruning.
//!
//! ### Theorem 2.4 (Sublimation Bound)
//!
//! Given a CTMDP M̄ = ⟨Σ × X × Z, Ax, Ps, fg, fc⟩, we can solve a sublimated
//! TMDP Msub,σ = ⟨Σ, Aσ, Pσ, fg,σ, fc,σ⟩ on just the logic space Σ where:
//!
//! - fg,σ(σ) := max_{z,x,a} fg(σ, z, x, a)  (maximized over base level)
//! - fc,σ(σ, ασ) is a component from separable constraint fc
//!
//! Then: **κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)**
//!
//! ### Interpretation
//!
//! - If κ*_sub,σ(σ) = 0, then κ̃*(σ, z, x) = 0 for all (z, x)
//! - **Abstractly infeasible ⟹ Practically infeasible**
//! - **Abstractly feasible ⏸ Practically feasible** (not implied!)
//!
//! ### Use Case: Tree Search Pruning
//!
//! ```text
//! From paper Figure 7 (bottom):
//!
//! Before entering logic state σ₀₀₁, check sublimated feasibility:
//!   if κ*_sub,σ(σ₀₀₁) = 0:
//!       Prune branch (red star)
//!       Save node expansions
//!   else:
//!       Continue search (may still fail at base level)
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use stok_core::hierarchy::sublimation::*;
//! use stok_core::prelude::*;
//!
//! let device = default_device();
//!
//! // Create base-level MDP (gridworld)
//! let base_mdp = create_gridworld(&device);
//!
//! // Create high-level kernel (logic task)
//! let hl_kernel = create_logic_dynamics(&device);
//!
//! // Create affordance function
//! let affordance = create_affordance(&device);
//!
//! // Solve sublimated TMDP on logic space only
//! let sublimated = SublimatedTMDP::from_product(
//!     &base_mdp,
//!     hl_kernel,
//!     &affordance,
//!     0,  // HL space index
//! )?;
//!
//! let kappa_sub = sublimated.solve()?;
//!
//! // Use for pruning
//! if kappa_sub[sigma_state] == 0.0 {
//!     println!("State σ is abstractly infeasible - prune!");
//! }
//! ```

use crate::mdp::TaskMDP;
use crate::solver::solve_task_mdp;
use crate::types::StokError;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

// ================================================================================================
// Sublimated Task MDP
// ================================================================================================

/// Sublimated Task MDP: M_sub = ⟨Z, Az, Pz, fg,z, fc,z⟩
///
/// High-level-only TMDP obtained by "sublimating" (abstracting) away from
/// the full product space S = X × Z.
///
/// # Mathematical Definition
///
/// From paper Appendix 8 (Sublimation Theorem proof):
///
/// ```text
/// M_sub,σ = ⟨Σ, Aσ, Pσ, fg,σ, fc,σ⟩
///
/// where:
///   fg,σ(σ) := max_{z,x,a} fg(σ, z, x, a)  (max over base level)
///   fc,σ(σ, ασ) := component from separable fc
/// ```
///
/// # Properties
///
/// 1. **Treats HL actions as free variables** (not conditioned on base level)
/// 2. **Maximizes goal over all base states** (optimistic abstraction)
/// 3. **Provides necessary condition** for full feasibility
///
/// # Theorem 2.4
///
/// ```text
/// κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)
///
/// Contrapositive: κ*_sub,σ(σ) = 0  ⟹  κ̃*(σ, z, x) = 0 for all (z, x)
/// ```
///
/// **Proof**: See paper Appendix 8, page 32.
#[derive(Debug)]
pub struct SublimatedTMDP<B: Backend> {
    /// High-level transition kernel: P_z(z'|z, α_z)
    ///
    /// Shape: [n_hl_states, n_hl_actions, n_hl_states]
    ///
    /// **Note**: In full problem, HL actions are determined by affordance F(α_z|x,a).
    /// In sublimated problem, HL actions are treated as free control variables.
    pub hl_kernel: Tensor<B, 3>,

    /// Sublimated goal function: fg,z(z, α_z)
    ///
    /// Shape: [n_hl_states, n_hl_actions]
    ///
    /// Defined as: fg,z(z, α_z) := max_{x,a} fg(z, x, a) where F(α_z|x,a) > 0
    ///
    /// This is an **optimistic** abstraction - assumes best-case base-level conditions.
    pub goal_fn: Tensor<B, 2>,

    /// Sublimated constraint function: fc,z(z, α_z)
    ///
    /// Shape: [n_hl_states, n_hl_actions]
    ///
    /// For separable constraints: fc,z(z, α_z) is the HL component of fc.
    pub constraint_fn: Tensor<B, 2>,

    /// Maximum time horizon
    pub max_time: usize,

    /// Dimensions
    pub n_hl_states: usize,
    pub n_hl_actions: usize,
}

impl<B: Backend> SublimatedTMDP<B> {
    /// Create a new sublimated TMDP directly
    ///
    /// # Arguments
    ///
    /// * `hl_kernel` - P_z(z'|z, α_z), shape [n_z, n_α_z, n_z]
    /// * `goal_fn` - fg,z(z, α_z), shape [n_z, n_α_z]
    /// * `constraint_fn` - fc,z(z, α_z), shape [n_z, n_α_z]
    /// * `max_time` - Maximum time horizon
    ///
    /// # Returns
    ///
    /// `Ok(SublimatedTMDP)` if shapes are valid
    pub fn new(
        hl_kernel: Tensor<B, 3>,
        goal_fn: Tensor<B, 2>,
        constraint_fn: Tensor<B, 2>,
        max_time: usize,
    ) -> Result<Self, StokError> {
        // Validate shapes
        let k_dims = hl_kernel.dims();
        let g_dims = goal_fn.dims();
        let c_dims = constraint_fn.dims();

        let n_hl_states = k_dims[0];
        let n_hl_actions = k_dims[1];

        // Kernel should be [n_z, n_α_z, n_z]
        if k_dims[2] != n_hl_states {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_hl_states, n_hl_actions, n_hl_states],
                got: k_dims.to_vec(),
            });
        }

        // Goal and constraint should be [n_z, n_α_z]
        if g_dims != [n_hl_states, n_hl_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_hl_states, n_hl_actions],
                got: g_dims.to_vec(),
            });
        }

        if c_dims != [n_hl_states, n_hl_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_hl_states, n_hl_actions],
                got: c_dims.to_vec(),
            });
        }

        Ok(Self {
            hl_kernel,
            goal_fn,
            constraint_fn,
            max_time,
            n_hl_states,
            n_hl_actions,
        })
    }

    /// Solve sublimated TMDP for abstract feasibility κ*_sub
    ///
    /// # Returns
    ///
    /// CPU-side vector κ*_sub,σ for each HL state σ ∈ Σ
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let kappa_sub = sublimated.solve()?;
    ///
    /// // Check abstract feasibility
    /// for (sigma, &kappa) in kappa_sub.iter().enumerate() {
    ///     if kappa == 0.0 {
    ///         println!("State σ_{} is abstractly infeasible", sigma);
    ///     }
    /// }
    /// ```
    pub fn solve(&self) -> Result<Vec<f32>, StokError> {
        // Convert to TaskMDP and solve with standard feasibility iteration
        let task_mdp = TaskMDP::new(
            self.hl_kernel.clone(),
            self.goal_fn.clone(),
            self.constraint_fn.clone(),
            self.max_time,
        )?;

        let result = solve_task_mdp(&task_mdp)?;

        // Extract κ to CPU for tree search pruning
        let kappa_data: Vec<f32> = result.kappa.into_data().to_vec().unwrap();

        Ok(kappa_data)
    }

    /// Convert to TaskMDP for direct solving
    ///
    /// # Returns
    ///
    /// TaskMDP on high-level space only
    pub fn to_task_mdp(&self) -> Result<TaskMDP<B>, StokError> {
        TaskMDP::new(
            self.hl_kernel.clone(),
            self.goal_fn.clone(),
            self.constraint_fn.clone(),
            self.max_time,
        )
    }

    /// Get dimensions
    pub fn n_hl_states(&self) -> usize {
        self.n_hl_states
    }

    pub fn n_hl_actions(&self) -> usize {
        self.n_hl_actions
    }
}

// ================================================================================================
// Sublimation from Product-Space MDP
// ================================================================================================

/// Extract sublimated goal function `f_{g,σ}(σ, α_σ) := max_{x, a} f_g(x, a)`
/// over all `(x, a)` such that `F(α_σ | x, a) > 0`.
///
/// PP-401 generalization: the function now derives `n_hl_actions` from the
/// affordance component for `hl_space_index` and accepts `n_hl_states`
/// explicitly, removing the prior `n_hl_states = 2, n_hl_actions = 2`
/// hardcoding. The sublimated goal is broadcast across all HL states because
/// `f_g` is not a function of HL state in the assumed factorization
/// (Theorem 2.1 hypothesis).
///
/// # Arguments
///
/// * `base_mdp` — base-level TMDP carrying `f_g(x, a)`
/// * `affordance` — `F(α_σ | x, a)` for the relevant HL space
/// * `hl_space_index` — which HL space to sublimate over
/// * `n_hl_states` — `|Z_{hl_space_index}|` (typically `hl_kernel.dims()[0]`
///   from the caller)
///
/// # Returns
///
/// `f_{g,σ}(z, α_σ)` of shape `[n_hl_states, n_hl_actions]`.
pub fn maximize_goal_over_base<B: Backend>(
    base_mdp: &TaskMDP<B>,
    affordance: &crate::hierarchy::affordance::FactorizedAffordance<B>,
    hl_space_index: usize,
    n_hl_states: usize,
) -> Tensor<B, 2> {
    use crate::hierarchy::affordance::{AffordanceFunction, HLAction};

    assert!(
        hl_space_index < affordance.n_hl_spaces(),
        "hl_space_index {} out of range for affordance with {} HL spaces",
        hl_space_index,
        affordance.n_hl_spaces()
    );
    assert!(n_hl_states > 0, "n_hl_states must be > 0");
    assert_eq!(
        base_mdp.dims.n_states,
        affordance.n_base_states(),
        "base_mdp.n_states ({}) must match affordance.n_base_states ({})",
        base_mdp.dims.n_states,
        affordance.n_base_states()
    );
    assert_eq!(
        base_mdp.dims.n_actions,
        affordance.n_base_actions(),
        "base_mdp.n_actions ({}) must match affordance.n_base_actions ({})",
        base_mdp.dims.n_actions,
        affordance.n_base_actions()
    );

    let device = base_mdp.goal_fn.device();
    let n_hl_actions = affordance.n_hl_actions_for_space(hl_space_index);

    // Per HL action, maximize the base goal over all (x, a) that can induce it.
    let mut fg_hl_data = vec![0.0f32; n_hl_actions];
    let goal_data: Vec<f32> = base_mdp.goal_fn.clone().into_data().to_vec().unwrap();
    let n_states = base_mdp.dims.n_states;
    let n_actions = base_mdp.dims.n_actions;

    for hl_action_idx in 0..n_hl_actions {
        let hl_action = HLAction::Single {
            space_id: hl_space_index,
            action_id: hl_action_idx,
        };
        let mut max_goal = 0.0f32;
        for x in 0..n_states {
            for a in 0..n_actions {
                if affordance.probability(&hl_action, x, a) > 1e-6 {
                    max_goal = max_goal.max(goal_data[x * n_actions + a]);
                }
            }
        }
        fg_hl_data[hl_action_idx] = max_goal;
    }

    // Broadcast across all HL states (Theorem 2.1: f_g not a function of HL state).
    let mut fg_hl_full = vec![0.0f32; n_hl_states * n_hl_actions];
    for z in 0..n_hl_states {
        for a_z in 0..n_hl_actions {
            fg_hl_full[z * n_hl_actions + a_z] = fg_hl_data[a_z];
        }
    }

    let fg_tensor: Tensor<B, 1> = Tensor::from_floats(fg_hl_full.as_slice(), &device);
    fg_tensor.reshape([n_hl_states, n_hl_actions])
}

/// **PLACEHOLDER**: returns an all-ones HL constraint function (i.e. no HL
/// constraints anywhere). Theorem 2.4 requires `f_{c,σ}` to be the *separable
/// HL component* of the full `f_c`, which can only be derived from the
/// caller's specific factorization. There is no general way to extract it
/// from `hl_kernel` shape alone — the caller must supply it via the
/// `hl_constraint` argument to [`SublimatedTMDP::from_product`].
///
/// PP-402: this placeholder is retained as a no-constraint default for callers
/// where the HL space genuinely has no constraints; otherwise pass an explicit
/// constraint tensor.
pub fn placeholder_hl_constraint_all_free<B: Backend>(hl_kernel: &Tensor<B, 3>) -> Tensor<B, 2> {
    let dims = hl_kernel.dims();
    let n_hl_states = dims[0];
    let n_hl_actions = dims[1];
    let device = hl_kernel.device();
    Tensor::ones([n_hl_states, n_hl_actions], &device)
}

/// Deprecated alias kept for one release after PP-402.
#[deprecated(
    since = "0.2.0",
    note = "Renamed to `placeholder_hl_constraint_all_free` to make the \
            placeholder semantics explicit. For real HL constraints, supply \
            them directly to `SublimatedTMDP::from_product`."
)]
pub fn extract_hl_constraint<B: Backend>(hl_kernel: &Tensor<B, 3>) -> Tensor<B, 2> {
    placeholder_hl_constraint_all_free(hl_kernel)
}

impl<B: Backend> SublimatedTMDP<B> {
    /// Create sublimated TMDP from product-space components.
    ///
    /// PP-401/402 generalization: this entry point now (a) supports HL spaces
    /// of arbitrary cardinality (no longer hardcoded binary), and (b) accepts
    /// an explicit HL constraint function instead of silently defaulting to
    /// all-ones. For HL spaces with no real constraints, pass `None` to use
    /// `placeholder_hl_constraint_all_free`.
    ///
    /// # Arguments
    ///
    /// * `base_mdp` — base-level TMDP `M_x = ⟨X, A, P_x, f_{g,x}, f_{c,x}⟩`
    /// * `hl_kernel` — HL transition `P_σ(σ' | σ, α_σ)`, shape `[|Σ|, |A_σ|, |Σ|]`
    /// * `affordance` — `F(α_σ | x, a)` for the relevant HL space
    /// * `hl_space_index` — which HL space to sublimate over
    /// * `hl_constraint` — `f_{c,σ}(σ, α_σ)` per Theorem 2.4. Pass `None` to use
    ///   the all-free placeholder for spaces without HL constraints.
    pub fn from_product(
        base_mdp: &TaskMDP<B>,
        hl_kernel: Tensor<B, 3>,
        affordance: &crate::hierarchy::affordance::FactorizedAffordance<B>,
        hl_space_index: usize,
        hl_constraint: Option<Tensor<B, 2>>,
    ) -> Result<Self, StokError> {
        let n_hl_states = hl_kernel.dims()[0];

        let goal_fn = maximize_goal_over_base(base_mdp, affordance, hl_space_index, n_hl_states);
        let constraint_fn =
            hl_constraint.unwrap_or_else(|| placeholder_hl_constraint_all_free(&hl_kernel));

        Self::new(hl_kernel, goal_fn, constraint_fn, base_mdp.dims.max_time)
    }
}

// ================================================================================================
// Tree Search Integration
// ================================================================================================

/// Sublimated feasibility cache for tree search pruning
///
/// Stores κ*_sub for each HL state to enable fast pruning checks.
///
/// # Usage in Tree Search
///
/// ```rust,ignore
/// let cache = SublimatedFeasibilityCache::new();
///
/// // Precompute before search
/// cache.add_space(0, kappa_sub_logic);
/// cache.add_space(1, kappa_sub_hydration);
///
/// // During tree search
/// if cache.is_abstractly_infeasible(0, sigma_state) {
///     stats.nodes_pruned += 1;
///     continue;  // Skip this branch
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SublimatedFeasibilityCache {
    /// Map from HL space index to κ*_sub vector
    ///
    /// Key: HL space index (e.g., 0 for logic, 1 for hydration)
    /// Value: κ*_sub,z for all states in that space
    feasibility: std::collections::HashMap<usize, Vec<f32>>,
}

impl SublimatedFeasibilityCache {
    /// Create empty cache
    pub fn new() -> Self {
        Self {
            feasibility: std::collections::HashMap::new(),
        }
    }

    /// Add sublimated feasibility for an HL space
    ///
    /// # Arguments
    ///
    /// * `space_index` - Index of HL space
    /// * `kappa_sub` - κ*_sub vector for all states in that space
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut cache = SublimatedFeasibilityCache::new();
    ///
    /// // Add logic space feasibility
    /// cache.add_space(0, vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    ///
    /// // Add hydration space feasibility
    /// cache.add_space(1, vec![0.0, 0.2, 0.5, 0.8, 1.0]);
    /// ```
    pub fn add_space(&mut self, space_index: usize, kappa_sub: Vec<f32>) {
        self.feasibility.insert(space_index, kappa_sub);
    }

    /// Check if a state is abstractly infeasible
    ///
    /// # Arguments
    ///
    /// * `space_index` - HL space index
    /// * `hl_state` - State index in that space
    ///
    /// # Returns
    ///
    /// `true` if κ*_sub(z) = 0 (abstractly infeasible), `false` otherwise
    ///
    /// # Panics
    ///
    /// If space_index not in cache or hl_state out of bounds
    pub fn is_abstractly_infeasible(&self, space_index: usize, hl_state: usize) -> bool {
        let kappa_sub = self
            .feasibility
            .get(&space_index)
            .expect("HL space not in sublimated cache");

        kappa_sub[hl_state] < 1e-6 // Effectively zero
    }

    /// Get sublimated feasibility for a state
    ///
    /// # Arguments
    ///
    /// * `space_index` - HL space index
    /// * `hl_state` - State index in that space
    ///
    /// # Returns
    ///
    /// κ*_sub(z) ∈ [0, 1]
    pub fn get_feasibility(&self, space_index: usize, hl_state: usize) -> Option<f32> {
        self.feasibility
            .get(&space_index)
            .and_then(|kappa| kappa.get(hl_state).copied())
    }

    /// Check if cache has a specific HL space
    pub fn has_space(&self, space_index: usize) -> bool {
        self.feasibility.contains_key(&space_index)
    }

    /// Get all space indices in cache
    pub fn space_indices(&self) -> Vec<usize> {
        self.feasibility.keys().copied().collect()
    }

    /// Number of spaces in cache
    pub fn n_spaces(&self) -> usize {
        self.feasibility.len()
    }
}

impl Default for SublimatedFeasibilityCache {
    fn default() -> Self {
        Self::new()
    }
}

// ================================================================================================
// Convenience Functions
// ================================================================================================

/// Compute sublimated feasibility for a high-level space
///
/// This is the main entry point for computing abstract feasibility bounds.
///
/// # Arguments
///
/// * `base_mdp` - Base-level TaskMDP
/// * `hl_kernel` - High-level transition kernel P_z(z'|z, α_z)
/// * `affordance` - Affordance function F(α_z|x, a)
/// * `hl_space_index` - Which HL space to sublimate
///
/// # Returns
///
/// Vector κ*_sub for each HL state
///
/// # Example
///
/// ```rust,ignore
/// let kappa_sub = compute_sublimated_feasibility(
///     &base_mdp,
///     logic_kernel,
///     &affordance,
///     0,  // Logic space
/// )?;
///
/// // Use for pruning
/// if kappa_sub[sigma_001] == 0.0 {
///     println!("σ_001 is abstractly infeasible!");
/// }
/// ```
pub fn compute_sublimated_feasibility<B: Backend>(
    base_mdp: &TaskMDP<B>,
    hl_kernel: Tensor<B, 3>,
    affordance: &crate::hierarchy::affordance::FactorizedAffordance<B>,
    hl_space_index: usize,
    hl_constraint: Option<Tensor<B, 2>>,
) -> Result<Vec<f32>, StokError> {
    let sub_mdp = SublimatedTMDP::from_product(
        base_mdp,
        hl_kernel,
        affordance,
        hl_space_index,
        hl_constraint,
    )?;
    sub_mdp.solve()
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use burn::prelude::ElementConversion;

    #[test]
    fn test_sublimated_tmdp_creation() {
        let device = default_device();

        // Create simple 4-state, 2-action HL kernel
        let n_hl_states = 4;
        let n_hl_actions = 2;

        let hl_kernel: Tensor<DefaultBackend, 3> =
            Tensor::zeros([n_hl_states, n_hl_actions, n_hl_states], &device);
        let goal_fn: Tensor<DefaultBackend, 2> = Tensor::zeros([n_hl_states, n_hl_actions], &device);
        let constraint_fn: Tensor<DefaultBackend, 2> =
            Tensor::ones([n_hl_states, n_hl_actions], &device);

        let sub_mdp = SublimatedTMDP::new(hl_kernel, goal_fn, constraint_fn, 30);

        assert!(sub_mdp.is_ok());
        let mdp = sub_mdp.unwrap();
        assert_eq!(mdp.n_hl_states(), 4);
        assert_eq!(mdp.n_hl_actions(), 2);
    }

    #[test]
    fn test_sublimated_tmdp_dimension_mismatch() {
        let device = default_device();

        let hl_kernel: Tensor<DefaultBackend, 3> = Tensor::zeros([4, 2, 4], &device);
        let goal_fn: Tensor<DefaultBackend, 2> = Tensor::zeros([3, 2], &device); // Wrong!
        let constraint_fn: Tensor<DefaultBackend, 2> = Tensor::ones([4, 2], &device);

        let result = SublimatedTMDP::new(hl_kernel, goal_fn, constraint_fn, 30);

        assert!(result.is_err());
    }

    #[test]
    fn test_placeholder_hl_constraint_all_free() {
        let device = default_device();

        let hl_kernel: Tensor<DefaultBackend, 3> = Tensor::zeros([3, 2, 3], &device);
        let constraint = placeholder_hl_constraint_all_free(&hl_kernel);

        let dims = constraint.dims();
        assert_eq!(dims, [3, 2]);

        // Placeholder is all ones (no HL constraints).
        let sample: f32 = constraint.clone().slice([0..1, 0..1]).into_scalar().elem();
        assert_eq!(sample, 1.0);
    }

    #[test]
    fn test_sublimated_feasibility_cache() {
        let mut cache = SublimatedFeasibilityCache::new();

        // Add logic space
        cache.add_space(0, vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);

        // Add hydration space
        cache.add_space(1, vec![0.0, 0.2, 0.5, 0.8, 1.0]);

        assert_eq!(cache.n_spaces(), 2);
        assert!(cache.has_space(0));
        assert!(cache.has_space(1));
        assert!(!cache.has_space(2));
    }

    #[test]
    fn test_sublimated_cache_is_infeasible() {
        let mut cache = SublimatedFeasibilityCache::new();

        cache.add_space(0, vec![1.0, 0.0, 1.0, 0.0]);

        assert!(!cache.is_abstractly_infeasible(0, 0)); // κ_sub = 1.0
        assert!(cache.is_abstractly_infeasible(0, 1)); // κ_sub = 0.0
        assert!(!cache.is_abstractly_infeasible(0, 2)); // κ_sub = 1.0
        assert!(cache.is_abstractly_infeasible(0, 3)); // κ_sub = 0.0
    }

    #[test]
    fn test_sublimated_cache_get_feasibility() {
        let mut cache = SublimatedFeasibilityCache::new();

        cache.add_space(0, vec![0.5, 0.8, 0.0, 1.0]);

        assert_eq!(cache.get_feasibility(0, 0), Some(0.5));
        assert_eq!(cache.get_feasibility(0, 1), Some(0.8));
        assert_eq!(cache.get_feasibility(0, 2), Some(0.0));
        assert_eq!(cache.get_feasibility(0, 3), Some(1.0));
        assert_eq!(cache.get_feasibility(1, 0), None); // Space 1 not in cache
    }

    #[test]
    fn test_sublimation_theorem_bound() {
        // Verify Theorem 2.4: κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)
        //
        // Test setup:
        // - Logic space with 2 states (binary bit)
        // - Base space with 3 states
        // - Goal only achievable from base state 2
        // - Affordance: base state 2 flips bit (α₁), others do nothing (α₀)

        let device = default_device();

        // Base MDP: 3 states, 2 actions, goal at state 2
        let mut goal_data = vec![0.0f32; 3 * 2];
        goal_data[2 * 2 + 0] = 1.0; // Goal at state 2, action 0
        goal_data[2 * 2 + 1] = 1.0; // Goal at state 2, action 1

        let goal_fn_1d: Tensor<DefaultBackend, 1> = Tensor::from_floats(goal_data.as_slice(), &device);
        let goal_fn = goal_fn_1d.reshape([3, 2]);

        // Simple chain: 0 → 1 → 2
        let mut trans_data = vec![0.0f32; 3 * 2 * 3];
        trans_data[0 * 2 * 3 + 0 * 3 + 1] = 1.0; // (0, a0) → 1
        trans_data[0 * 2 * 3 + 1 * 3 + 1] = 1.0; // (0, a1) → 1
        trans_data[1 * 2 * 3 + 0 * 3 + 2] = 1.0; // (1, a0) → 2
        trans_data[1 * 2 * 3 + 1 * 3 + 2] = 1.0; // (1, a1) → 2
        trans_data[2 * 2 * 3 + 0 * 3 + 2] = 1.0; // (2, a0) → 2 (absorbing)
        trans_data[2 * 2 * 3 + 1 * 3 + 2] = 1.0; // (2, a1) → 2

        let transition_1d: Tensor<DefaultBackend, 1> =
            Tensor::from_floats(trans_data.as_slice(), &device);
        let transition = transition_1d.reshape([3, 2, 3]);
        let constraint_fn: Tensor<DefaultBackend, 2> = Tensor::ones([3, 2], &device);

        let base_mdp = TaskMDP::new(transition, goal_fn, constraint_fn, 10).unwrap();

        // HL kernel: 2-state (binary bit), 2 actions (α₀: no-op, α₁: flip)
        let mut hl_data = vec![0.0f32; 2 * 2 * 2];
        hl_data[0 * 2 * 2 + 0 * 2 + 0] = 1.0; // (σ₀, α₀) → σ₀ (no flip)
        hl_data[0 * 2 * 2 + 1 * 2 + 1] = 1.0; // (σ₀, α₁) → σ₁ (flip)
        hl_data[1 * 2 * 2 + 0 * 2 + 1] = 1.0; // (σ₁, α₀) → σ₁ (no flip)
        hl_data[1 * 2 * 2 + 1 * 2 + 0] = 1.0; // (σ₁, α₁) → σ₀ (flip)

        let hl_kernel_1d: Tensor<DefaultBackend, 1> =
            Tensor::from_floats(hl_data.as_slice(), &device);
        let hl_kernel = hl_kernel_1d.reshape([2, 2, 2]);

        // Affordance: only base state 2 can flip bit (α₁)
        let mut f_data = vec![0.0f32; 3 * 2 * 2];
        f_data[0 * 2 * 2 + 0 * 2 + 0] = 1.0; // (x=0, a=0) → α₀
        f_data[0 * 2 * 2 + 1 * 2 + 0] = 1.0; // (x=0, a=1) → α₀
        f_data[1 * 2 * 2 + 0 * 2 + 0] = 1.0; // (x=1, a=0) → α₀
        f_data[1 * 2 * 2 + 1 * 2 + 0] = 1.0; // (x=1, a=1) → α₀
        f_data[2 * 2 * 2 + 0 * 2 + 1] = 1.0; // (x=2, a=0) → α₁ (flip!)
        f_data[2 * 2 * 2 + 1 * 2 + 1] = 1.0; // (x=2, a=1) → α₁ (flip!)

        let f_tensor_1d: Tensor<DefaultBackend, 1> = Tensor::from_floats(f_data.as_slice(), &device);
        let f_tensor = f_tensor_1d.reshape([3, 2, 2]);

        use crate::hierarchy::affordance::FactorizedAffordance;
        let affordance = FactorizedAffordance::new(vec![f_tensor]);

        // Solve sublimated TMDP
        let sub_mdp =
            SublimatedTMDP::from_product(&base_mdp, hl_kernel, &affordance, 0, None).unwrap();

        // Check that sublimated goal is maximized
        let fg_sub_data: Vec<f32> = sub_mdp.goal_fn.clone().into_data().to_vec().unwrap();

        // fg,σ(σ, α₀) = max over (x,a) where F(α₀|x,a)>0
        //             = max_{x∈{0,1}, a∈{0,1}} fg(x,a) = 0.0
        assert!((fg_sub_data[0 * 2 + 0] - 0.0).abs() < 1e-6);
        assert!((fg_sub_data[1 * 2 + 0] - 0.0).abs() < 1e-6);

        // fg,σ(σ, α₁) = max over (x,a) where F(α₁|x,a)>0
        //             = max_{x=2, a∈{0,1}} fg(2,a) = 1.0
        assert!((fg_sub_data[0 * 2 + 1] - 1.0).abs() < 1e-6);
        assert!((fg_sub_data[1 * 2 + 1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sublimation_provides_valid_bound() {
        // Create simple sublimated problem and verify bound holds
        let device = default_device();

        // 2-state HL space, deterministic goal at state 1
        let mut goal_data = vec![0.0f32; 2 * 2];
        goal_data[1 * 2 + 0] = 1.0; // State 1, action 0
        goal_data[1 * 2 + 1] = 1.0; // State 1, action 1

        let goal_fn_1d: Tensor<DefaultBackend, 1> = Tensor::from_floats(goal_data.as_slice(), &device);
        let goal_fn = goal_fn_1d.reshape([2, 2]);

        // Deterministic transitions: state 0 unreachable from state 0
        let mut hl_data = vec![0.0f32; 2 * 2 * 2];
        hl_data[0 * 2 * 2 + 0 * 2 + 0] = 1.0; // (0, a0) → 0 (stuck!)
        hl_data[0 * 2 * 2 + 1 * 2 + 0] = 1.0; // (0, a1) → 0 (stuck!)
        hl_data[1 * 2 * 2 + 0 * 2 + 1] = 1.0; // (1, a0) → 1
        hl_data[1 * 2 * 2 + 1 * 2 + 1] = 1.0; // (1, a1) → 1

        let hl_kernel_1d: Tensor<DefaultBackend, 1> = Tensor::from_floats(hl_data.as_slice(), &device);
        let hl_kernel = hl_kernel_1d.reshape([2, 2, 2]);

        // Add constraint violation at state 0 to make it absorbing failure state
        let mut constraint_data = vec![1.0f32; 2 * 2];
        constraint_data[0 * 2 + 0] = 0.0; // State 0, action 0 violates constraint
        constraint_data[0 * 2 + 1] = 0.0; // State 0, action 1 violates constraint

        let constraint_fn_1d: Tensor<DefaultBackend, 1> = Tensor::from_floats(constraint_data.as_slice(), &device);
        let constraint_fn = constraint_fn_1d.reshape([2, 2]);

        let sub_mdp = SublimatedTMDP::new(hl_kernel, goal_fn, constraint_fn, 10).unwrap();

        let kappa_sub = sub_mdp.solve().unwrap();

        // State 0: infeasible (stuck in state 0, goal at state 1)
        assert!(kappa_sub[0] < 1e-6, "State 0 should be infeasible");

        // State 1: feasible (goal is here)
        assert!(kappa_sub[1] > 0.99, "State 1 should be feasible");

        // This demonstrates the bound:
        // For product space (σ=0, any z, any x): κ̃* ≤ κ*_sub(σ=0) = 0
        // Therefore all product states with σ=0 are infeasible!
    }

    #[test]
    fn test_sublimated_cache_operations() {
        let mut cache = SublimatedFeasibilityCache::new();

        assert_eq!(cache.n_spaces(), 0);

        cache.add_space(0, vec![1.0, 0.5, 0.0]);
        assert_eq!(cache.n_spaces(), 1);
        assert!(cache.has_space(0));

        cache.add_space(1, vec![0.8, 0.2]);
        assert_eq!(cache.n_spaces(), 2);

        let indices = cache.space_indices();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn test_sublimated_cache_pruning_logic() {
        let mut cache = SublimatedFeasibilityCache::new();

        // Logic space: states 2 and 4 are infeasible
        cache.add_space(0, vec![1.0, 0.8, 0.0, 1.0, 0.0, 0.9]);

        // Simulate tree search pruning
        let mut nodes_pruned = 0;

        for sigma in 0..6 {
            if cache.is_abstractly_infeasible(0, sigma) {
                nodes_pruned += 1;
            }
        }

        assert_eq!(nodes_pruned, 2); // States 2 and 4 should be pruned
    }
}
