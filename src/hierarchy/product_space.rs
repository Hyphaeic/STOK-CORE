//! # Product-Space State Representation
//!
//! Implements state representation for S = X × Z₁ × ... × Zₙ
//!
//! ## Mathematical Background
//!
//! Product spaces arise when an agent's world has multiple coupled subsystems:
//! - X: Base-level (e.g., 2D position in gridworld)
//! - Z₁: High-level space 1 (e.g., hydration level)
//! - Z₂: High-level space 2 (e.g., task completion flags)
//!
//! Total states: |S| = |X| × |Z₁| × ... × |Zₙ|
//!
//! The factorization (Theorem 2.1) avoids explicitly constructing this
//! exponentially large space.

use std::fmt;

/// High-level state component
///
/// Represents a state in one of the high-level spaces Z_k.
///
/// # Variants
///
/// - **Discrete**: Categorical state (e.g., room number, logic state)
/// - **Continuous**: Discretized continuous value (e.g., hydration level)
/// - **BinaryVector**: Boolean flags (e.g., task completion: [honey, flowers, key])
#[derive(Clone, Debug, PartialEq)]
pub enum HLState {
    /// Discrete categorical state
    Discrete(usize),

    /// Continuous value (stored as discretized index)
    Continuous { index: usize, value: f32 },

    /// Binary vector (e.g., task flags)
    BinaryVector(Vec<bool>),
}

impl HLState {
    /// Get as discrete index (panics if not discrete or continuous)
    pub fn as_discrete(&self) -> usize {
        match self {
            HLState::Discrete(i) => *i,
            HLState::Continuous { index, .. } => *index,
            HLState::BinaryVector(bits) => binary_vec_to_index(bits),
        }
    }

    /// Get as binary vector (panics if not BinaryVector)
    pub fn as_binary_vec(&self) -> &[bool] {
        match self {
            HLState::BinaryVector(bits) => bits,
            _ => panic!("Not a binary vector"),
        }
    }

    /// Get continuous value (panics if not Continuous)
    pub fn as_continuous(&self) -> f32 {
        match self {
            HLState::Continuous { value, .. } => *value,
            _ => panic!("Not a continuous state"),
        }
    }

    /// Get state space size
    pub fn space_size(&self) -> usize {
        match self {
            HLState::Discrete(i) => *i + 1, // Assumes 0-indexed
            HLState::Continuous { index, .. } => *index + 1,
            HLState::BinaryVector(bits) => 1 << bits.len(), // 2^n
        }
    }
}

impl fmt::Display for HLState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HLState::Discrete(i) => write!(f, "D{}", i),
            HLState::Continuous { value, index } => write!(f, "C{}({:.2})", index, value),
            HLState::BinaryVector(bits) => {
                write!(f, "B[")?;
                for (i, &b) in bits.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", if b { "1" } else { "0" })?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Product-space state: s = (x, z₁, z₂, ..., zₙ)
///
/// Represents a point in the Cartesian product S = X × Z₁ × ... × Zₙ.
///
/// # Example
///
/// ```rust,ignore
/// // Honey badger at position (3,4) with full hydration, no items
/// let state = ProductState::new(34)  // Grid position
///     .with_hl_discrete(9)           // Hydration level 9/10
///     .with_hl_binary_vector(vec![false, false, false]);  // [honey, flowers, key]
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ProductState {
    /// Base-level state (X)
    pub base: usize,

    /// High-level state components (Z₁, ..., Zₙ)
    pub hl_states: Vec<HLState>,
}

impl ProductState {
    /// Create new product-space state with only base component
    pub fn new(base: usize) -> Self {
        Self {
            base,
            hl_states: vec![],
        }
    }

    /// Add discrete high-level state component
    pub fn with_hl_discrete(mut self, state: usize) -> Self {
        self.hl_states.push(HLState::Discrete(state));
        self
    }

    /// Add continuous high-level state component
    pub fn with_hl_continuous(mut self, value: f32, index: usize) -> Self {
        self.hl_states
            .push(HLState::Continuous { index, value });
        self
    }

    /// Add binary vector high-level state component
    pub fn with_hl_binary_vector(mut self, bits: Vec<bool>) -> Self {
        self.hl_states.push(HLState::BinaryVector(bits));
        self
    }

    /// Get number of high-level components
    pub fn n_hl_components(&self) -> usize {
        self.hl_states.len()
    }

    /// Convert to flat index in product space
    ///
    /// Maps (x, z₁, ..., zₙ) → single index for tensor operations
    ///
    /// # Arguments
    ///
    /// * `dims` - Product space dimensions
    ///
    /// # Returns
    ///
    /// Flat index in [0, |S|)
    ///
    /// # Formula
    ///
    /// index = x + z₁·|X| + z₂·|X|·|Z₁| + ... + zₙ·|X|·|Z₁|·...·|Zₙ₋₁|
    pub fn to_flat_index(&self, dims: &ProductSpaceDims) -> usize {
        assert_eq!(
            self.hl_states.len(),
            dims.hl_sizes.len(),
            "State components must match dimensions"
        );

        let mut index = self.base;
        let mut stride = dims.base_size;

        for (hl_state, &hl_size) in self.hl_states.iter().zip(&dims.hl_sizes) {
            let hl_idx = hl_state.as_discrete();
            assert!(
                hl_idx < hl_size,
                "HL state {} exceeds space size {}",
                hl_idx,
                hl_size
            );
            index += hl_idx * stride;
            stride *= hl_size;
        }

        assert!(
            index < dims.total_size,
            "Flat index {} exceeds total size {}",
            index,
            dims.total_size
        );

        index
    }

    /// Convert from flat index to product-space state
    ///
    /// Inverse of `to_flat_index()`
    ///
    /// # Arguments
    ///
    /// * `index` - Flat index
    /// * `dims` - Product space dimensions
    ///
    /// # Returns
    ///
    /// ProductState decoded from index
    pub fn from_flat_index(index: usize, dims: &ProductSpaceDims) -> Self {
        assert!(
            index < dims.total_size,
            "Index {} exceeds total size {}",
            index,
            dims.total_size
        );

        let mut remaining = index;

        // Extract base state (lowest stride)
        let base = remaining % dims.base_size;
        remaining /= dims.base_size;

        // Extract HL states
        let mut hl_states = Vec::with_capacity(dims.hl_sizes.len());
        for &hl_size in &dims.hl_sizes {
            let hl_idx = remaining % hl_size;
            hl_states.push(HLState::Discrete(hl_idx));
            remaining /= hl_size;
        }

        Self { base, hl_states }
    }

    /// Clone with modified base state
    pub fn with_base(&self, new_base: usize) -> Self {
        Self {
            base: new_base,
            hl_states: self.hl_states.clone(),
        }
    }

    /// Clone with modified HL state component
    pub fn with_hl_component(&self, component_idx: usize, new_state: HLState) -> Self {
        let mut new_hl = self.hl_states.clone();
        new_hl[component_idx] = new_state;
        Self {
            base: self.base,
            hl_states: new_hl,
        }
    }
}

impl fmt::Display for ProductState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S(x={}", self.base)?;
        for hl in &self.hl_states {
            write!(f, ", {}", hl)?;
        }
        write!(f, ")")
    }
}

/// Product-space dimensions
///
/// Tracks sizes of each component space in S = X × Z₁ × ... × Zₙ
///
/// # Example
///
/// ```rust,ignore
/// // Honey badger: 100-state grid × 8 logic states × 10 hydration levels
/// let dims = ProductSpaceDims::new(100, vec![8, 10]);
/// assert_eq!(dims.total_size, 8000);  // 100 × 8 × 10
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ProductSpaceDims {
    /// Base-level state space size (|X|)
    pub base_size: usize,

    /// High-level state space sizes (|Z₁|, |Z₂|, ...)
    pub hl_sizes: Vec<usize>,

    /// Total product-space size: |X| × ∏ᵢ |Zᵢ|
    pub total_size: usize,
}

impl ProductSpaceDims {
    /// Create new product-space dimensions
    ///
    /// # Arguments
    ///
    /// * `base_size` - Size of base-level space |X|
    /// * `hl_sizes` - Sizes of high-level spaces [|Z₁|, |Z₂|, ...]
    ///
    /// # Returns
    ///
    /// Product-space dimensions with computed total size
    pub fn new(base_size: usize, hl_sizes: Vec<usize>) -> Self {
        assert!(base_size > 0, "Base size must be > 0");
        for (i, &size) in hl_sizes.iter().enumerate() {
            assert!(size > 0, "HL space {} size must be > 0", i);
        }

        let hl_product: usize = hl_sizes.iter().product();
        let total_size = base_size * hl_product;

        Self {
            base_size,
            hl_sizes,
            total_size,
        }
    }

    /// Get number of high-level spaces
    pub fn n_hl_spaces(&self) -> usize {
        self.hl_sizes.len()
    }

    /// Get size of specific HL space
    pub fn hl_size(&self, index: usize) -> usize {
        self.hl_sizes[index]
    }

    /// Check if this is a pure base-level space (no HL components)
    pub fn is_base_only(&self) -> bool {
        self.hl_sizes.is_empty()
    }

    /// Get HL product size: ∏ᵢ |Zᵢ|
    pub fn hl_product_size(&self) -> usize {
        if self.hl_sizes.is_empty() {
            1
        } else {
            self.hl_sizes.iter().product()
        }
    }
}

impl fmt::Display for ProductSpaceDims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProductSpace(X:{}", self.base_size)?;
        for (i, &size) in self.hl_sizes.iter().enumerate() {
            write!(f, ", Z{}:{}", i + 1, size)?;
        }
        write!(f, " | total:{})", self.total_size)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert binary vector to integer index
///
/// # Arguments
///
/// * `bits` - Binary vector (LSB first)
///
/// # Returns
///
/// Integer representation
///
/// # Example
///
/// ```rust,ignore
/// assert_eq!(binary_vec_to_index(&[true, false, true]), 5);  // 101₂ = 5
/// ```
pub fn binary_vec_to_index(bits: &[bool]) -> usize {
    bits.iter()
        .enumerate()
        .fold(0, |acc, (i, &b)| acc + if b { 1 << i } else { 0 })
}

/// Convert integer index to binary vector
///
/// # Arguments
///
/// * `index` - Integer to convert
/// * `n_bits` - Number of bits
///
/// # Returns
///
/// Binary vector (LSB first)
pub fn index_to_binary_vec(index: usize, n_bits: usize) -> Vec<bool> {
    (0..n_bits).map(|i| (index >> i) & 1 == 1).collect()
}

/// Discretize continuous value to index
///
/// # Arguments
///
/// * `value` - Continuous value
/// * `n_bins` - Number of discrete bins
/// * `min_val` - Minimum value
/// * `max_val` - Maximum value
///
/// # Returns
///
/// Discrete index in [0, n_bins)
pub fn discretize(value: f32, n_bins: usize, min_val: f32, max_val: f32) -> usize {
    assert!(n_bins > 0, "Must have at least one bin");
    assert!(max_val > min_val, "Invalid range");

    let clamped = value.clamp(min_val, max_val);
    let normalized = (clamped - min_val) / (max_val - min_val);
    let index = (normalized * (n_bins as f32)).floor() as usize;

    index.min(n_bins - 1) // Clamp to valid range
}

/// Convert discrete index back to continuous value (bin center)
pub fn undiscretize(index: usize, n_bins: usize, min_val: f32, max_val: f32) -> f32 {
    assert!(index < n_bins, "Index {} out of range [0, {})", index, n_bins);

    let bin_width = (max_val - min_val) / (n_bins as f32);
    min_val + (index as f32 + 0.5) * bin_width
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_vec_conversion() {
        let bits = vec![true, false, true]; // 101₂ = 5
        let index = binary_vec_to_index(&bits);
        assert_eq!(index, 5);

        let recovered = index_to_binary_vec(index, 3);
        assert_eq!(recovered, bits);
    }

    #[test]
    fn test_binary_vec_all_combinations() {
        for n_bits in 1..=4 {
            let max_index = 1 << n_bits;
            for index in 0..max_index {
                let bits = index_to_binary_vec(index, n_bits);
                let recovered = binary_vec_to_index(&bits);
                assert_eq!(recovered, index);
            }
        }
    }

    #[test]
    fn test_discretize_continuous() {
        // 10 bins for [0.0, 1.0]
        assert_eq!(discretize(0.0, 10, 0.0, 1.0), 0);
        assert_eq!(discretize(0.05, 10, 0.0, 1.0), 0);
        assert_eq!(discretize(0.15, 10, 0.0, 1.0), 1);
        assert_eq!(discretize(0.95, 10, 0.0, 1.0), 9);
        assert_eq!(discretize(1.0, 10, 0.0, 1.0), 9); // Edge case

        // Out of bounds should clamp
        assert_eq!(discretize(-0.5, 10, 0.0, 1.0), 0);
        assert_eq!(discretize(1.5, 10, 0.0, 1.0), 9);
    }

    #[test]
    fn test_undiscretize() {
        // Bin centers for 10 bins in [0, 1]
        assert!((undiscretize(0, 10, 0.0, 1.0) - 0.05).abs() < 1e-5);
        assert!((undiscretize(5, 10, 0.0, 1.0) - 0.55).abs() < 1e-5);
        assert!((undiscretize(9, 10, 0.0, 1.0) - 0.95).abs() < 1e-5);
    }

    #[test]
    fn test_product_state_creation() {
        let state = ProductState::new(5);
        assert_eq!(state.base, 5);
        assert_eq!(state.n_hl_components(), 0);
    }

    #[test]
    fn test_product_state_with_hl() {
        let state = ProductState::new(10)
            .with_hl_discrete(3)
            .with_hl_binary_vector(vec![true, false, true]);

        assert_eq!(state.base, 10);
        assert_eq!(state.n_hl_components(), 2);
        assert_eq!(state.hl_states[0].as_discrete(), 3);
        assert_eq!(state.hl_states[1].as_discrete(), 5); // Binary 101 = 5
    }

    #[test]
    fn test_product_dims_creation() {
        let dims = ProductSpaceDims::new(25, vec![8, 10]);

        assert_eq!(dims.base_size, 25);
        assert_eq!(dims.n_hl_spaces(), 2);
        assert_eq!(dims.total_size, 25 * 8 * 10); // 2000
    }

    #[test]
    fn test_product_dims_base_only() {
        let dims = ProductSpaceDims::new(100, vec![]);

        assert!(dims.is_base_only());
        assert_eq!(dims.total_size, 100);
        assert_eq!(dims.hl_product_size(), 1);
    }

    #[test]
    fn test_flat_index_conversion_simple() {
        // 2×3 product space: X has 2 states, Z has 3 states
        let dims = ProductSpaceDims::new(2, vec![3]);

        // State (x=0, z=0) → index 0
        let s00 = ProductState::new(0).with_hl_discrete(0);
        assert_eq!(s00.to_flat_index(&dims), 0);

        // State (x=1, z=0) → index 1
        let s10 = ProductState::new(1).with_hl_discrete(0);
        assert_eq!(s10.to_flat_index(&dims), 1);

        // State (x=0, z=1) → index 2 (stride = 2)
        let s01 = ProductState::new(0).with_hl_discrete(1);
        assert_eq!(s01.to_flat_index(&dims), 2);

        // State (x=1, z=2) → index 5
        let s12 = ProductState::new(1).with_hl_discrete(2);
        assert_eq!(s12.to_flat_index(&dims), 5);
    }

    #[test]
    fn test_flat_index_conversion_round_trip() {
        let dims = ProductSpaceDims::new(5, vec![3, 4]);

        for index in 0..dims.total_size {
            let state = ProductState::from_flat_index(index, &dims);
            let recovered = state.to_flat_index(&dims);
            assert_eq!(
                recovered, index,
                "Round-trip failed for index {}",
                index
            );
        }
    }

    #[test]
    fn test_flat_index_honey_badger_example() {
        // Honey badger: 100 grid states × 8 logic × 10 hydration
        let dims = ProductSpaceDims::new(100, vec![8, 10]);

        // Agent at grid(34), logic(5), hydration(7)
        let state = ProductState::new(34)
            .with_hl_discrete(5)
            .with_hl_discrete(7);

        let flat_index = state.to_flat_index(&dims);

        // Verify: index = 34 + 5·100 + 7·100·8 = 34 + 500 + 5600 = 6134
        assert_eq!(flat_index, 6134);

        // Round-trip
        let recovered = ProductState::from_flat_index(flat_index, &dims);
        assert_eq!(recovered.base, 34);
        assert_eq!(recovered.hl_states[0].as_discrete(), 5);
        assert_eq!(recovered.hl_states[1].as_discrete(), 7);
    }

    #[test]
    fn test_product_state_display() {
        let state = ProductState::new(5)
            .with_hl_discrete(3)
            .with_hl_binary_vector(vec![true, false, true]);

        let display = format!("{}", state);
        assert!(display.contains("x=5"));
        assert!(display.contains("D3"));
        assert!(display.contains("B[1,0,1]"));
    }

    #[test]
    fn test_hl_state_space_size() {
        let discrete = HLState::Discrete(5);
        assert_eq!(discrete.space_size(), 6); // 0-5 inclusive

        let binary = HLState::BinaryVector(vec![false, false, false]);
        assert_eq!(binary.space_size(), 8); // 2³
    }
}
