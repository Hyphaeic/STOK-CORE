//! # Affordance Function
//!
//! Implements affordance functions that link base-level state-actions
//! to high-level state transformations.
//!
//! ## Mathematical Definition (Definition 0.1)
//!
//! ```text
//! F: (X × A) × A_z → [0,1]
//! F(α_z | x, a) = ∏_k F_k(α_{z_k} | x, a)
//! ```
//!
//! ## Example (Honey Badger)
//!
//! - Drinking at lake: F(α_hydrate | x_lake, a_drink) = 1.0
//! - Picking up honey: F(α_set_honey_bit | x_honey, a_pickup) = 1.0
//! - Elsewhere: F(α_default | x, a) = 1.0 (no effect on HL state)

use burn::prelude::*;
use rand::Rng;
use crate::planning::sample_categorical;

/// High-level action
///
/// Represents an action in a high-level state space Zₖ.
///
/// # Example
///
/// ```rust,ignore
/// // Hydration space actions
/// let alpha_hydrate = HLAction::single(0, 1);  // Space 0, action "hydrate"
/// let alpha_dehydrate = HLAction::single(0, 0); // Space 0, action "dehydrate"
///
/// // Multi-space action vector
/// let alpha_vec = HLAction::vector(vec![1, 0, 2]);  // Actions for Z₁, Z₂, Z₃
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HLAction {
    /// Single component action
    Single {
        /// Which HL space this acts on
        space_id: usize,
        /// Action ID within that space
        action_id: usize,
    },

    /// Vector of component actions (α_{z₁}, α_{z₂}, ...)
    /// One action per HL space
    Vector(Vec<usize>),
}

impl HLAction {
    /// Create single-component HL action
    pub fn single(space_id: usize, action_id: usize) -> Self {
        Self::Single {
            space_id,
            action_id,
        }
    }

    /// Create multi-component HL action vector
    pub fn vector(actions: Vec<usize>) -> Self {
        Self::Vector(actions)
    }

    /// Get action ID for specific space (if this is a vector)
    pub fn action_for_space(&self, space_id: usize) -> Option<usize> {
        match self {
            Self::Single {
                space_id: sid,
                action_id,
            } => {
                if *sid == space_id {
                    Some(*action_id)
                } else {
                    None
                }
            }
            Self::Vector(actions) => actions.get(space_id).copied(),
        }
    }

    /// Convert to vector form (padding with defaults if needed)
    pub fn to_vector(&self, n_spaces: usize, default: usize) -> Vec<usize> {
        match self {
            Self::Vector(v) => v.clone(),
            Self::Single {
                space_id,
                action_id,
            } => {
                let mut vec = vec![default; n_spaces];
                vec[*space_id] = *action_id;
                vec
            }
        }
    }
}

/// Set of high-level actions (Cartesian product of component action spaces)
#[derive(Clone, Debug)]
pub struct HLActionSet {
    /// Action space sizes for each HL space: [|A_{z₁}|, |A_{z₂}|, ...]
    action_sizes: Vec<usize>,

    /// Total number of action vectors: ∏ᵢ |A_{zᵢ}|
    total_actions: usize,
}

impl HLActionSet {
    /// Create HL action set
    pub fn new(action_sizes: Vec<usize>) -> Self {
        let total_actions = action_sizes.iter().product();
        Self {
            action_sizes,
            total_actions,
        }
    }

    /// Enumerate all HL action vectors
    pub fn enumerate(&self) -> Vec<HLAction> {
        let mut actions = Vec::with_capacity(self.total_actions);

        for flat_index in 0..self.total_actions {
            let action_vec = self.flat_to_action_vector(flat_index);
            actions.push(HLAction::Vector(action_vec));
        }

        actions
    }

    /// Convert flat index to action vector
    fn flat_to_action_vector(&self, mut index: usize) -> Vec<usize> {
        let mut actions = Vec::with_capacity(self.action_sizes.len());

        for &size in &self.action_sizes {
            actions.push(index % size);
            index /= size;
        }

        actions
    }
}

/// Affordance function trait
///
/// Links base-level state-actions to high-level actions.
///
/// # Paper Definition (0.1)
///
/// F(α_z | x, a) outputs probability that (x, a) induces HL action α_z
pub trait AffordanceFunction {
    /// Evaluate F(α | x, a)
    ///
    /// # Arguments
    ///
    /// * `hl_action` - High-level action α
    /// * `base_state` - Base-level state x
    /// * `base_action` - Base-level action a
    ///
    /// # Returns
    ///
    /// Probability in [0, 1]
    fn probability(&self, hl_action: &HLAction, base_state: usize, base_action: usize) -> f32;

    /// Get support: which HL actions have non-zero probability?
    ///
    /// # Arguments
    ///
    /// * `base_state` - Base-level state x
    /// * `base_action` - Base-level action a
    ///
    /// # Returns
    ///
    /// Vector of HL actions with F(α | x, a) > 0
    fn support(&self, base_state: usize, base_action: usize) -> Vec<HLAction>;

    /// Sample HL action from distribution F(· | x, a)
    ///
    /// # Arguments
    ///
    /// * `base_state` - Base-level state x
    /// * `base_action` - Base-level action a
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Sampled HL action
    fn sample(&self, base_state: usize, base_action: usize, rng: &mut impl Rng) -> HLAction;
}

/// Factorized affordance function
///
/// Implements F(α_z | x, a) = ∏_k F_k(α_{z_k} | x, a)
///
/// Each component F_k is stored as a tensor [n_base_states, n_base_actions, n_hl_actions_k]
///
/// # Example
///
/// ```rust,ignore
/// // Honey badger hydration affordance
/// let mut F_hydration = Tensor::zeros([25, 4, 2], &device);  // 25 states, 4 actions, 2 HL actions
///
/// // At lake (state 12) with drink action (action 2): hydrate
/// F_hydration[12][2][ALPHA_HYDRATE] = 1.0;
///
/// // Everywhere else: dehydrate (default)
/// for x in 0..25 {
///     for a in 0..4 {
///         if x != 12 || a != 2 {
///             F_hydration[x][a][ALPHA_DEHYDRATE] = 1.0;
///         }
///     }
/// }
///
/// let affordance = FactorizedAffordance::new(vec![F_hydration]);
/// ```
pub struct FactorizedAffordance<B: Backend> {
    /// Component affordances F_k(α_{z_k} | x, a)
    ///
    /// Each tensor has shape [n_base_states, n_base_actions, n_hl_actions_k]
    components: Vec<Tensor<B, 3>>,

    /// Base-level dimensions
    base_dims: (usize, usize), // (n_states, n_actions)

    /// HL action space sizes per component
    hl_action_sizes: Vec<usize>,
}

impl<B: Backend> FactorizedAffordance<B> {
    /// Create factorized affordance from component tensors
    ///
    /// # Arguments
    ///
    /// * `components` - Vector of F_k tensors
    ///
    /// # Returns
    ///
    /// Factorized affordance function
    ///
    /// # Panics
    ///
    /// Panics if components have inconsistent base dimensions
    pub fn new(components: Vec<Tensor<B, 3>>) -> Self {
        assert!(!components.is_empty(), "Must have at least one component");

        // Extract base dimensions from first component
        let dims = components[0].dims();
        let base_dims = (dims[0], dims[1]);

        // Validate all components have same base dimensions
        for (i, comp) in components.iter().enumerate() {
            let comp_dims = comp.dims();
            assert_eq!(
                (comp_dims[0], comp_dims[1]),
                base_dims,
                "Component {} has inconsistent base dimensions",
                i
            );
        }

        // Extract HL action sizes
        let hl_action_sizes: Vec<usize> = components.iter().map(|c| c.dims()[2]).collect();

        Self {
            components,
            base_dims,
            hl_action_sizes,
        }
    }

    /// Evaluate F(α | x, a) = ∏_k F_k(α_k | x, a)
    ///
    /// # Arguments
    ///
    /// * `hl_action` - High-level action vector
    /// * `x` - Base state
    /// * `a` - Base action
    ///
    /// # Returns
    ///
    /// Product of component probabilities
    pub fn evaluate(&self, hl_action: &HLAction, x: usize, a: usize) -> f32 {
        let action_vec = hl_action.to_vector(self.components.len(), 0);

        let mut prob = 1.0f32;
        for (k, &alpha_k) in action_vec.iter().enumerate() {
            let component_prob: f32 = self.components[k]
                .clone()
                .slice([x..(x + 1), a..(a + 1), alpha_k..(alpha_k + 1)])
                .into_scalar()
                .elem();
            prob *= component_prob;

            if prob == 0.0 {
                break; // Early termination
            }
        }

        prob
    }

    /// Get number of base states
    pub fn n_base_states(&self) -> usize {
        self.base_dims.0
    }

    /// Get number of base actions
    pub fn n_base_actions(&self) -> usize {
        self.base_dims.1
    }

    /// Get number of HL spaces
    pub fn n_hl_spaces(&self) -> usize {
        self.components.len()
    }

    /// Get device
    pub fn device(&self) -> B::Device {
        self.components[0].device()
    }
}

impl<B: Backend> AffordanceFunction for FactorizedAffordance<B> {
    fn probability(&self, hl_action: &HLAction, base_state: usize, base_action: usize) -> f32 {
        self.evaluate(hl_action, base_state, base_action)
    }

    fn support(&self, base_state: usize, base_action: usize) -> Vec<HLAction> {
        // Get support for each component
        let mut component_supports = vec![];

        for (_k, component) in self.components.iter().enumerate() {
            let n_actions = component.dims()[2];
            let mut support_k = vec![];

            for alpha in 0..n_actions {
                let prob: f32 = component
                    .clone()
                    .slice([
                        base_state..(base_state + 1),
                        base_action..(base_action + 1),
                        alpha..(alpha + 1),
                    ])
                    .into_scalar()
                    .elem();

                if prob > 0.0 {
                    support_k.push(alpha);
                }
            }

            component_supports.push(support_k);
        }

        // Cartesian product of component supports
        cartesian_product_actions(&component_supports)
    }

    fn sample(&self, x: usize, a: usize, rng: &mut impl Rng) -> HLAction {
        // Sample from each component independently
        let mut sampled = Vec::with_capacity(self.components.len());

        for component in &self.components {
            let n_actions = component.dims()[2];

            // Extract distribution F_k(· | x, a)
            let dist: Vec<f32> = component
                .clone()
                .slice([x..(x + 1), a..(a + 1), 0..n_actions])
                .reshape([n_actions])
                .into_data()
                .to_vec()
                .unwrap();

            let alpha_k = sample_categorical(&dist, rng);
            sampled.push(alpha_k);
        }

        HLAction::Vector(sampled)
    }
}

/// Compute Cartesian product of action supports
fn cartesian_product_actions(component_supports: &[Vec<usize>]) -> Vec<HLAction> {
    if component_supports.is_empty() {
        return vec![];
    }

    if component_supports.len() == 1 {
        return component_supports[0]
            .iter()
            .map(|&a| HLAction::Vector(vec![a]))
            .collect();
    }

    // Recursive Cartesian product
    let mut result = vec![];
    let first = &component_supports[0];
    let rest_product = cartesian_product_actions(&component_supports[1..]);

    for &action_0 in first {
        for rest in &rest_product {
            if let HLAction::Vector(rest_vec) = rest {
                let mut combined = vec![action_0];
                combined.extend(rest_vec);
                result.push(HLAction::Vector(combined));
            }
        }
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};

    #[test]
    fn test_hl_action_single() {
        let action = HLAction::single(0, 1);
        assert_eq!(action.action_for_space(0), Some(1));
        assert_eq!(action.action_for_space(1), None);
    }

    #[test]
    fn test_hl_action_vector() {
        let action = HLAction::vector(vec![1, 2, 3]);
        assert_eq!(action.action_for_space(0), Some(1));
        assert_eq!(action.action_for_space(1), Some(2));
        assert_eq!(action.action_for_space(2), Some(3));
    }

    #[test]
    fn test_hl_action_to_vector() {
        let single = HLAction::single(1, 5);
        let vec = single.to_vector(3, 0); // 3 spaces, default 0
        assert_eq!(vec, vec![0, 5, 0]);
    }

    #[test]
    fn test_hl_action_set() {
        let action_set = HLActionSet::new(vec![2, 3]); // 2 actions in Z₁, 3 in Z₂
        assert_eq!(action_set.total_actions, 6); // 2 × 3

        let all_actions = action_set.enumerate();
        assert_eq!(all_actions.len(), 6);
    }

    #[test]
    fn test_factorized_affordance_simple() {
        let device = default_device();

        // Single HL space with 2 actions (hydrate, dehydrate)
        // 3 base states, 2 base actions
        let mut F_data = vec![0.0f32; 3 * 2 * 2];

        // State 0, action 0: hydrate with prob 1.0
        F_data[0 * 2 * 2 + 0 * 2 + 1] = 1.0; // [x=0][a=0][α=1]

        // State 0, action 1: dehydrate with prob 1.0
        F_data[0 * 2 * 2 + 1 * 2 + 0] = 1.0;

        // State 1, all actions: dehydrate
        F_data[1 * 2 * 2 + 0 * 2 + 0] = 1.0;
        F_data[1 * 2 * 2 + 1 * 2 + 0] = 1.0;

        // State 2, all actions: dehydrate
        F_data[2 * 2 * 2 + 0 * 2 + 0] = 1.0;
        F_data[2 * 2 * 2 + 1 * 2 + 0] = 1.0;

        let F_component: Tensor<DefaultBackend, 1> =
            Tensor::from_floats(F_data.as_slice(), &device);
        let F_component: Tensor<DefaultBackend, 3> = F_component.reshape([3, 2, 2]);

        let affordance = FactorizedAffordance::new(vec![F_component]);

        // Test evaluation
        let alpha_hydrate = HLAction::vector(vec![1]);
        assert!((affordance.evaluate(&alpha_hydrate, 0, 0) - 1.0).abs() < 1e-5);

        let alpha_dehydrate = HLAction::vector(vec![0]);
        assert!((affordance.evaluate(&alpha_dehydrate, 0, 1) - 1.0).abs() < 1e-5);
        assert!((affordance.evaluate(&alpha_dehydrate, 1, 0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_factorized_affordance_product() {
        let device = default_device();

        // Two HL spaces, each with 2 actions
        // Test F(α₁, α₂ | x, a) = F₁(α₁ | x, a) × F₂(α₂ | x, a)

        // F₁: state 0 → action 0, state 1 → action 1
        let F1_data: &[f32] = &[
            1.0, 0.0, // State 0, base action 0: α₁=0
            0.0, 0.0, // State 0, base action 1
            0.0, 1.0, // State 1, base action 0: α₁=1
            0.0, 0.0, // State 1, base action 1
        ];
        let F1: Tensor<DefaultBackend, 1> = Tensor::from_floats(F1_data, &device);
        let F1: Tensor<DefaultBackend, 3> = F1.reshape([2, 2, 2]);

        // F₂: always action 0
        let F2_data: &[f32] = &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let F2: Tensor<DefaultBackend, 1> = Tensor::from_floats(F2_data, &device);
        let F2: Tensor<DefaultBackend, 3> = F2.reshape([2, 2, 2]);

        let affordance = FactorizedAffordance::new(vec![F1, F2]);

        // F(α=(0,0) | x=0, a=0) = F₁(0|0,0) × F₂(0|0,0) = 1 × 1 = 1
        let action_00 = HLAction::vector(vec![0, 0]);
        assert!((affordance.evaluate(&action_00, 0, 0) - 1.0).abs() < 1e-5);

        // F(α=(1,0) | x=0, a=0) = F₁(1|0,0) × F₂(0|0,0) = 0 × 1 = 0
        let action_10 = HLAction::vector(vec![1, 0]);
        assert!(affordance.evaluate(&action_10, 0, 0).abs() < 1e-5);
    }

    #[test]
    fn test_affordance_support() {
        let device = default_device();

        // Single component with sparse support
        let F_data: &[f32] = &[
            0.7, 0.3, 0.0, // State 0, action 0: supports α=0,1
            1.0, 0.0, 0.0, // State 0, action 1: supports α=0 only
            0.0, 0.0, 1.0, // State 1, action 0: supports α=2 only
            0.0, 1.0, 0.0, // State 1, action 1: supports α=1 only
        ];
        let F: Tensor<DefaultBackend, 1> = Tensor::from_floats(F_data, &device);
        let F: Tensor<DefaultBackend, 3> = F.reshape([2, 2, 3]);

        let affordance = FactorizedAffordance::new(vec![F]);

        // State 0, action 0: should support α=0,1
        let support = affordance.support(0, 0);
        assert_eq!(support.len(), 2);

        // State 0, action 1: should support α=0 only
        let support = affordance.support(0, 1);
        assert_eq!(support.len(), 1);
    }

    #[test]
    fn test_affordance_sampling() {
        let device = default_device();
        let mut rng = rand::thread_rng();

        // Deterministic affordance: always returns α=1
        let F_data: &[f32] = &[0.0, 1.0, 0.0, 1.0];
        let F: Tensor<DefaultBackend, 1> = Tensor::from_floats(F_data, &device);
        let F: Tensor<DefaultBackend, 3> = F.reshape([1, 2, 2]);

        let affordance = FactorizedAffordance::new(vec![F]);

        // Should always sample α=1
        for _ in 0..10 {
            let sampled = affordance.sample(0, 0, &mut rng);
            assert_eq!(sampled, HLAction::vector(vec![1]));
        }
    }

    #[test]
    fn test_cartesian_product() {
        let supports = vec![vec![0, 1], vec![0, 2]]; // Z₁: {0,1}, Z₂: {0,2}
        let product = cartesian_product_actions(&supports);

        assert_eq!(product.len(), 4); // 2 × 2
        assert!(product.contains(&HLAction::vector(vec![0, 0])));
        assert!(product.contains(&HLAction::vector(vec![0, 2])));
        assert!(product.contains(&HLAction::vector(vec![1, 0])));
        assert!(product.contains(&HLAction::vector(vec![1, 2])));
    }
}
