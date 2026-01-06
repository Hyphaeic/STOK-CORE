//! # Mode Functions: ζ: Z → E
//!
//! Implements environment mode switching based on high-level state.
//!
//! ## Mathematical Background
//!
//! From Ringstrom & Schrater (2025), Section 2:
//!
//! A mode function ζ: Z → E deterministically sets the environment mode e ∈ E
//! based on the high-level state z ∈ Z. This allows high-level state changes
//! to affect base-level dynamics.
//!
//! ### Example: Locked Door
//!
//! When a key is obtained (registered in logic state σ), the door mode changes:
//!
//! ```text
//! σ(key_bit) = 0  →  e = MODE_CLOSED  →  P_x(x'|x, a, e_closed)  [door blocks]
//! σ(key_bit) = 1  →  e = MODE_OPEN    →  P_x(x'|x, a, e_open)    [door passable]
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use stok_core::hierarchy::{modes::*, product_space::*};
//!
//! // Define mode function: key controls door
//! let mode_fn = KeyDoorMode {
//!     key_space_index: 0,  // First HL space contains key bit
//!     key_bit_index: 2,    // Third bit is the key
//! };
//!
//! // Check mode based on HL state
//! let hl_states = vec![HLState::BinaryVector(vec![true, false, true])];
//! let mode = mode_fn.mode(&hl_states);
//! assert_eq!(mode, MODE_OPEN);  // Key bit is 1, door is open
//! ```

use crate::hierarchy::product_space::HLState;
use crate::mdp::TaskMDP;
use crate::types::StokError;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

// ================================================================================================
// Constants
// ================================================================================================

/// Environment mode: door is closed (blocking)
pub const MODE_CLOSED: usize = 0;

/// Environment mode: door is open (passable)
pub const MODE_OPEN: usize = 1;

// ================================================================================================
// Mode Function Trait
// ================================================================================================

/// Mode function: ζ: Z → E
///
/// Deterministically maps high-level state to environment mode.
///
/// # Mathematical Definition
///
/// Given high-level state z = (z₁, ..., zₙ), computes mode e ∈ {0, 1, ..., m-1}.
///
/// # Properties
///
/// - **Deterministic**: Same input always produces same output
/// - **Stateless**: Pure function of current HL state
/// - **Discrete**: Finite number of modes
///
/// # Examples
///
/// - Key bit → door mode (closed/open)
/// - Temperature → weather mode (normal/extreme)
/// - Quest flags → NPC behavior mode
pub trait ModeFunction: std::fmt::Debug {
    /// Compute environment mode from high-level states
    ///
    /// # Arguments
    ///
    /// * `hl_states` - Vector of high-level state components z = (z₁, ..., zₙ)
    ///
    /// # Returns
    ///
    /// Mode index e ∈ {0, 1, ..., n_modes-1}
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mode = mode_fn.mode(&hl_states);
    /// let dynamics = &transition_kernels[mode];  // Select P_x(·|·,·,e)
    /// ```
    fn mode(&self, hl_states: &[HLState]) -> usize;

    /// Number of possible modes
    ///
    /// # Returns
    ///
    /// Total number of modes m = |E|
    fn n_modes(&self) -> usize;

    /// Get mode name (optional, for debugging)
    fn mode_name(&self, mode: usize) -> String {
        format!("Mode {}", mode)
    }
}

// ================================================================================================
// Identity Mode (No Mode Switching)
// ================================================================================================

/// Identity mode function: always returns mode 0
///
/// Used when there is no environment mode switching. Equivalent to ζ(z) = 0 for all z.
///
/// # Example
///
/// ```rust,ignore
/// let no_mode = NoMode;
/// assert_eq!(no_mode.mode(&any_hl_states), 0);
/// assert_eq!(no_mode.n_modes(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct NoMode;

impl ModeFunction for NoMode {
    fn mode(&self, _hl_states: &[HLState]) -> usize {
        0
    }

    fn n_modes(&self) -> usize {
        1
    }

    fn mode_name(&self, _mode: usize) -> String {
        "NoMode".to_string()
    }
}

// ================================================================================================
// Key-Controlled Door Mode
// ================================================================================================

/// Key-controlled door mode: ζ: Σ → {closed, open}
///
/// Maps a binary key bit in logic space to door mode.
///
/// # Mathematical Form
///
/// ```text
/// ζ(σ) = {
///     MODE_CLOSED (0)  if σ(key_bit) = 0
///     MODE_OPEN   (1)  if σ(key_bit) = 1
/// }
/// ```
///
/// # Example: Mountain Pass Door (Honey Badger)
///
/// From the paper (Figure 1), a sleeping bear guards a key. When the agent obtains
/// the key, σ(3) = 1, allowing passage through the mountain door.
///
/// ```rust,ignore
/// use stok_core::hierarchy::{modes::*, product_space::*};
///
/// // Key is third bit (index 2) in first HL space (index 0)
/// let door_mode = KeyDoorMode {
///     key_space_index: 0,
///     key_bit_index: 2,
/// };
///
/// // Before obtaining key: σ = [0,0,0]
/// let no_key = vec![HLState::BinaryVector(vec![false, false, false])];
/// assert_eq!(door_mode.mode(&no_key), MODE_CLOSED);
///
/// // After obtaining key: σ = [0,0,1]
/// let has_key = vec![HLState::BinaryVector(vec![false, false, true])];
/// assert_eq!(door_mode.mode(&has_key), MODE_OPEN);
/// ```
#[derive(Clone, Debug)]
pub struct KeyDoorMode {
    /// Which high-level state component contains the key bit
    ///
    /// Index into `hl_states` vector (typically 0 for single logic space)
    pub key_space_index: usize,

    /// Which bit in the binary vector represents the key
    ///
    /// Index into binary vector (e.g., 2 means third bit: σ(3) in 1-indexed notation)
    pub key_bit_index: usize,
}

impl KeyDoorMode {
    /// Create a new key-door mode function
    ///
    /// # Arguments
    ///
    /// * `key_space_index` - Index of HL space containing key bit (usually 0)
    /// * `key_bit_index` - Index of bit representing key (0-indexed)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Key is the third bit in the first HL space
    /// let mode_fn = KeyDoorMode::new(0, 2);
    /// ```
    pub fn new(key_space_index: usize, key_bit_index: usize) -> Self {
        Self {
            key_space_index,
            key_bit_index,
        }
    }
}

impl ModeFunction for KeyDoorMode {
    fn mode(&self, hl_states: &[HLState]) -> usize {
        // Extract the key state component
        let key_state = &hl_states[self.key_space_index];

        // Get binary vector
        let bits = key_state.as_binary_vec();

        // Check if key bit is set
        if bits[self.key_bit_index] {
            MODE_OPEN
        } else {
            MODE_CLOSED
        }
    }

    fn n_modes(&self) -> usize {
        2 // {closed, open}
    }

    fn mode_name(&self, mode: usize) -> String {
        match mode {
            MODE_CLOSED => "DoorClosed".to_string(),
            MODE_OPEN => "DoorOpen".to_string(),
            _ => format!("UnknownMode({})", mode),
        }
    }
}

// ================================================================================================
// Multi-Bit Composite Mode
// ================================================================================================

/// Multi-bit composite mode: ζ: Σ → E where |E| = 2^k
///
/// Uses k bits to determine mode, allowing 2^k different modes.
///
/// # Example: Multiple Keys
///
/// ```rust,ignore
/// // Two keys control different door systems
/// let multi_mode = MultiBitMode {
///     space_index: 0,
///     bit_indices: vec![2, 3],  // Bits 2 and 3
/// };
///
/// // σ = [_, _, 0, 0] → mode 0 (both doors closed)
/// // σ = [_, _, 1, 0] → mode 1 (door 1 open)
/// // σ = [_, _, 0, 1] → mode 2 (door 2 open)
/// // σ = [_, _, 1, 1] → mode 3 (both doors open)
/// ```
#[derive(Clone, Debug)]
pub struct MultiBitMode {
    /// High-level state component containing mode bits
    pub space_index: usize,

    /// Bit indices used to compute mode (LSB first)
    pub bit_indices: Vec<usize>,
}

impl MultiBitMode {
    /// Create a new multi-bit mode function
    ///
    /// # Arguments
    ///
    /// * `space_index` - Index of HL space containing mode bits
    /// * `bit_indices` - Indices of bits (LSB to MSB order)
    pub fn new(space_index: usize, bit_indices: Vec<usize>) -> Self {
        Self {
            space_index,
            bit_indices,
        }
    }
}

impl ModeFunction for MultiBitMode {
    fn mode(&self, hl_states: &[HLState]) -> usize {
        let state = &hl_states[self.space_index];
        let bits = state.as_binary_vec();

        // Compute mode as binary number from selected bits
        let mut mode = 0;
        for (i, &bit_idx) in self.bit_indices.iter().enumerate() {
            if bits[bit_idx] {
                mode |= 1 << i;
            }
        }

        mode
    }

    fn n_modes(&self) -> usize {
        1 << self.bit_indices.len() // 2^k
    }
}

// ================================================================================================
// Threshold-Based Mode (Continuous HL State)
// ================================================================================================

/// Threshold-based mode: ζ: R → E
///
/// Maps continuous high-level state to discrete mode via thresholds.
///
/// # Example: Temperature Regulation
///
/// ```rust,ignore
/// // Temperature controls weather severity
/// let temp_mode = ThresholdMode {
///     space_index: 0,
///     thresholds: vec![32.0, 100.0],  // Freezing, normal, extreme heat
/// };
///
/// // temp < 32   → mode 0 (freezing)
/// // 32 ≤ temp < 100 → mode 1 (normal)
/// // temp ≥ 100  → mode 2 (extreme heat)
/// ```
#[derive(Clone, Debug)]
pub struct ThresholdMode {
    /// High-level state component (must be Continuous)
    pub space_index: usize,

    /// Threshold values (sorted ascending)
    ///
    /// Creates n_thresholds + 1 modes:
    /// - mode 0: x < thresholds[0]
    /// - mode 1: thresholds[0] ≤ x < thresholds[1]
    /// - ...
    /// - mode n: x ≥ thresholds[n-1]
    pub thresholds: Vec<f32>,
}

impl ThresholdMode {
    /// Create a new threshold-based mode function
    ///
    /// # Arguments
    ///
    /// * `space_index` - Index of HL space (must be Continuous)
    /// * `thresholds` - Threshold values (will be sorted)
    pub fn new(space_index: usize, mut thresholds: Vec<f32>) -> Self {
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            space_index,
            thresholds,
        }
    }
}

impl ModeFunction for ThresholdMode {
    fn mode(&self, hl_states: &[HLState]) -> usize {
        let state = &hl_states[self.space_index];
        let value = state.as_continuous();

        // Find first threshold exceeded
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            if value < threshold {
                return i;
            }
        }

        // Value exceeds all thresholds
        self.thresholds.len()
    }

    fn n_modes(&self) -> usize {
        self.thresholds.len() + 1
    }
}

// ================================================================================================
// Mode-Conditioned Task MDP
// ================================================================================================

/// Mode-conditioned Task MDP: P_x(x'|x,a,e) where e = ζ(z)
///
/// Extends TaskMDP to support mode-dependent base-level dynamics.
///
/// # Mathematical Background
///
/// From Ringstrom & Schrater (2025), Section 2.B:
///
/// The composition function λ can include a mode function ζ: Z → E that
/// changes the base-level dynamics based on high-level state:
///
/// ```text
/// P_s(s'|s,a) = Σ_{α_z,e} P_z(z'|z,α_z) · F(α_z|x,a) · P_x(x'|x,a,e) · ζ(e|z)
/// ```
///
/// Where ζ(e|z) is deterministic: ζ(e|z) = 1 if e = mode(z), else 0.
///
/// # Example: Mountain Pass Door
///
/// ```rust,ignore
/// use stok_core::hierarchy::modes::*;
/// use stok_core::prelude::*;
///
/// let device = default_device();
///
/// // Two transition kernels: closed and open
/// let p_closed = create_gridworld_with_wall(&device);  // Door blocks passage
/// let p_open = create_gridworld_no_wall(&device);      // Door allows passage
///
/// let mode_mdp = ModeConditionedMDP::new(
///     vec![p_closed, p_open],
///     goal_fn,
///     constraint_fn,
///     Box::new(KeyDoorMode::new(0, 2)),
///     30,  // max_time
/// )?;
///
/// // When key bit = 0, uses p_closed
/// // When key bit = 1, uses p_open
/// ```
#[derive(Debug)]
pub struct ModeConditionedMDP<B: Backend> {
    /// Transition kernels per mode: P_x(x'|x,a,e) for each e ∈ E
    ///
    /// Vector of transition tensors, one per mode.
    /// Shape: Vec<Tensor<[n_states, n_actions, n_states]>>
    pub transitions: Vec<Tensor<B, 3>>,

    /// Goal function f_g(x,a), shape [n_states, n_actions]
    ///
    /// Same for all modes (goal doesn't change with environment mode)
    pub goal_fn: Tensor<B, 2>,

    /// Constraint function f_c(x,a), shape [n_states, n_actions]
    ///
    /// Could be mode-dependent if constraints change with mode
    pub constraint_fn: Tensor<B, 2>,

    /// Mode function ζ: Z → E
    ///
    /// Computes environment mode from high-level state
    pub mode_fn: Box<dyn ModeFunction>,

    /// Maximum time horizon for STOK computation
    pub max_time: usize,

    /// Cached dimensions
    pub n_states: usize,
    pub n_actions: usize,
    pub n_modes: usize,
}

impl<B: Backend> ModeConditionedMDP<B> {
    /// Create a new mode-conditioned TaskMDP
    ///
    /// # Arguments
    ///
    /// * `transitions` - Vector of transition kernels, one per mode
    /// * `goal_fn` - Goal function f_g(x,a)
    /// * `constraint_fn` - Constraint function f_c(x,a)
    /// * `mode_fn` - Mode function ζ: Z → E
    /// * `max_time` - Maximum time horizon
    ///
    /// # Returns
    ///
    /// `Ok(ModeConditionedMDP)` if valid, `Err(StokError)` otherwise
    ///
    /// # Errors
    ///
    /// - Number of transitions must equal mode_fn.n_modes()
    /// - All transitions must have same shape [S, A, S]
    /// - Goal and constraint must match [S, A]
    pub fn new(
        transitions: Vec<Tensor<B, 3>>,
        goal_fn: Tensor<B, 2>,
        constraint_fn: Tensor<B, 2>,
        mode_fn: Box<dyn ModeFunction>,
        max_time: usize,
    ) -> Result<Self, StokError> {
        // Validate number of modes
        if transitions.len() != mode_fn.n_modes() {
            return Err(StokError::DimensionMismatch {
                expected: vec![mode_fn.n_modes()],
                got: vec![transitions.len()],
            });
        }

        if transitions.is_empty() {
            return Err(StokError::DimensionMismatch {
                expected: vec![1],
                got: vec![0],
            });
        }

        // Get dimensions from first transition kernel
        let dims = transitions[0].dims();
        let n_states = dims[0];
        let n_actions = dims[1];

        // Validate all transitions have same shape
        for (_i, trans) in transitions.iter().enumerate() {
            let trans_dims = trans.dims();
            if trans_dims != dims {
                return Err(StokError::DimensionMismatch {
                    expected: dims.to_vec(),
                    got: trans_dims.to_vec(),
                });
            }

            // Validate square transition matrix
            if trans_dims[0] != trans_dims[2] {
                return Err(StokError::DimensionMismatch {
                    expected: vec![n_states, n_actions, n_states],
                    got: trans_dims.to_vec(),
                });
            }
        }

        // Validate goal and constraint shapes
        let g_dims = goal_fn.dims();
        let c_dims = constraint_fn.dims();

        if g_dims != [n_states, n_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_states, n_actions],
                got: g_dims.to_vec(),
            });
        }

        if c_dims != [n_states, n_actions] {
            return Err(StokError::DimensionMismatch {
                expected: vec![n_states, n_actions],
                got: c_dims.to_vec(),
            });
        }

        let n_modes = mode_fn.n_modes();

        Ok(Self {
            transitions,
            goal_fn,
            constraint_fn,
            mode_fn,
            max_time,
            n_states,
            n_actions,
            n_modes,
        })
    }

    /// Get transition kernel for a specific mode
    ///
    /// # Arguments
    ///
    /// * `mode` - Mode index e ∈ {0, ..., n_modes-1}
    ///
    /// # Returns
    ///
    /// Reference to P_x(x'|x,a,e) for the given mode
    pub fn get_transition(&self, mode: usize) -> &Tensor<B, 3> {
        &self.transitions[mode]
    }

    /// Get transition kernel based on high-level state
    ///
    /// # Arguments
    ///
    /// * `hl_states` - High-level state components
    ///
    /// # Returns
    ///
    /// Reference to P_x(x'|x,a,e) where e = ζ(z)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Get dynamics based on whether agent has key
    /// let hl_states = vec![HLState::BinaryVector(vec![false, false, true])];
    /// let p_x = mode_mdp.get_transition_for_hl_state(&hl_states);
    /// // p_x is P_x(·|·,·,e_open) if key bit is 1
    /// ```
    pub fn get_transition_for_hl_state(&self, hl_states: &[HLState]) -> &Tensor<B, 3> {
        let mode = self.mode_fn.mode(hl_states);
        self.get_transition(mode)
    }

    /// Convert to regular TaskMDP for a specific mode
    ///
    /// Creates a standard TaskMDP using the transition kernel for mode e.
    ///
    /// # Arguments
    ///
    /// * `mode` - Mode index e ∈ {0, ..., n_modes-1}
    ///
    /// # Returns
    ///
    /// TaskMDP with P_x(·|·,·,e) as its transition kernel
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Solve OKBE with door open
    /// let mdp_open = mode_mdp.to_task_mdp(MODE_OPEN)?;
    /// let result = solve_task_mdp(&mdp_open)?;
    /// ```
    pub fn to_task_mdp(&self, mode: usize) -> Result<TaskMDP<B>, StokError> {
        TaskMDP::new(
            self.transitions[mode].clone(),
            self.goal_fn.clone(),
            self.constraint_fn.clone(),
            self.max_time,
        )
    }

    /// Convert to regular TaskMDP based on high-level state
    ///
    /// # Arguments
    ///
    /// * `hl_states` - High-level state components
    ///
    /// # Returns
    ///
    /// TaskMDP with P_x(·|·,·,ζ(z)) as its transition kernel
    pub fn to_task_mdp_for_hl_state(
        &self,
        hl_states: &[HLState],
    ) -> Result<TaskMDP<B>, StokError> {
        let mode = self.mode_fn.mode(hl_states);
        self.to_task_mdp(mode)
    }

    /// Get dimensions
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    pub fn n_modes(&self) -> usize {
        self.n_modes
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::ElementConversion;

    #[test]
    fn test_no_mode() {
        let no_mode = NoMode;

        let hl_states = vec![HLState::Discrete(5)];
        assert_eq!(no_mode.mode(&hl_states), 0);
        assert_eq!(no_mode.n_modes(), 1);
    }

    #[test]
    fn test_key_door_mode_closed() {
        let door_mode = KeyDoorMode::new(0, 2);

        // Key bit (index 2) is false → door closed
        let hl_states = vec![HLState::BinaryVector(vec![true, false, false])];
        assert_eq!(door_mode.mode(&hl_states), MODE_CLOSED);
    }

    #[test]
    fn test_key_door_mode_open() {
        let door_mode = KeyDoorMode::new(0, 2);

        // Key bit (index 2) is true → door open
        let hl_states = vec![HLState::BinaryVector(vec![true, false, true])];
        assert_eq!(door_mode.mode(&hl_states), MODE_OPEN);
    }

    #[test]
    fn test_key_door_mode_names() {
        let door_mode = KeyDoorMode::new(0, 0);

        assert_eq!(door_mode.mode_name(MODE_CLOSED), "DoorClosed");
        assert_eq!(door_mode.mode_name(MODE_OPEN), "DoorOpen");
        assert_eq!(door_mode.n_modes(), 2);
    }

    #[test]
    fn test_multi_bit_mode_single_bit() {
        let mode_fn = MultiBitMode::new(0, vec![1]);

        let closed = vec![HLState::BinaryVector(vec![false, false])];
        assert_eq!(mode_fn.mode(&closed), 0);

        let open = vec![HLState::BinaryVector(vec![false, true])];
        assert_eq!(mode_fn.mode(&open), 1);

        assert_eq!(mode_fn.n_modes(), 2);
    }

    #[test]
    fn test_multi_bit_mode_two_bits() {
        let mode_fn = MultiBitMode::new(0, vec![2, 3]);

        // Binary: 00 → mode 0
        let mode_00 = vec![HLState::BinaryVector(vec![false, false, false, false])];
        assert_eq!(mode_fn.mode(&mode_00), 0b00);

        // Binary: 01 → mode 1 (bit 2 set)
        let mode_01 = vec![HLState::BinaryVector(vec![false, false, true, false])];
        assert_eq!(mode_fn.mode(&mode_01), 0b01);

        // Binary: 10 → mode 2 (bit 3 set)
        let mode_10 = vec![HLState::BinaryVector(vec![false, false, false, true])];
        assert_eq!(mode_fn.mode(&mode_10), 0b10);

        // Binary: 11 → mode 3 (both set)
        let mode_11 = vec![HLState::BinaryVector(vec![false, false, true, true])];
        assert_eq!(mode_fn.mode(&mode_11), 0b11);

        assert_eq!(mode_fn.n_modes(), 4);
    }

    #[test]
    fn test_threshold_mode_single_threshold() {
        let mode_fn = ThresholdMode::new(0, vec![50.0]);

        let below = vec![HLState::Continuous {
            index: 10,
            value: 25.0,
        }];
        assert_eq!(mode_fn.mode(&below), 0);

        let above = vec![HLState::Continuous {
            index: 75,
            value: 75.0,
        }];
        assert_eq!(mode_fn.mode(&above), 1);

        assert_eq!(mode_fn.n_modes(), 2);
    }

    #[test]
    fn test_threshold_mode_multiple_thresholds() {
        let mode_fn = ThresholdMode::new(0, vec![32.0, 100.0]);

        // Freezing: temp < 32
        let freezing = vec![HLState::Continuous {
            index: 5,
            value: 20.0,
        }];
        assert_eq!(mode_fn.mode(&freezing), 0);

        // Normal: 32 ≤ temp < 100
        let normal = vec![HLState::Continuous {
            index: 50,
            value: 72.0,
        }];
        assert_eq!(mode_fn.mode(&normal), 1);

        // Extreme: temp ≥ 100
        let extreme = vec![HLState::Continuous {
            index: 80,
            value: 120.0,
        }];
        assert_eq!(mode_fn.mode(&extreme), 2);

        assert_eq!(mode_fn.n_modes(), 3);
    }

    #[test]
    fn test_threshold_mode_auto_sorts() {
        // Provide unsorted thresholds
        let mode_fn = ThresholdMode::new(0, vec![100.0, 32.0, 75.0]);

        let value = vec![HLState::Continuous {
            index: 40,
            value: 50.0,
        }];

        // Should be in range [32, 75) → mode 1
        assert_eq!(mode_fn.mode(&value), 1);
    }

    #[test]
    fn test_mode_function_with_multiple_hl_spaces() {
        // Test with multiple HL components
        let door_mode = KeyDoorMode::new(1, 0); // Key in second space, first bit

        let hl_states = vec![
            HLState::Discrete(5),                         // Space 0: ignored
            HLState::BinaryVector(vec![true, false]),     // Space 1: key bit is true
            HLState::Continuous {
                index: 3,
                value: 0.7,
            }, // Space 2: ignored
        ];

        assert_eq!(door_mode.mode(&hl_states), MODE_OPEN);
    }

    // ========================================================================
    // ModeConditionedMDP Tests
    // ========================================================================

    #[test]
    fn test_mode_conditioned_mdp_creation() {
        use crate::backend::{default_device, DefaultBackend};

        let device = default_device();

        // Create two simple 3-state, 2-action transition kernels
        let n_states = 3;
        let n_actions = 2;

        // Mode 0 (closed): simple transitions
        let p_closed = Tensor::<DefaultBackend, 3>::zeros([n_states, n_actions, n_states], &device);

        // Mode 1 (open): different transitions
        let p_open = Tensor::<DefaultBackend, 3>::zeros([n_states, n_actions, n_states], &device);

        let goal_fn = Tensor::<DefaultBackend, 2>::zeros([n_states, n_actions], &device);
        let constraint_fn = Tensor::<DefaultBackend, 2>::ones([n_states, n_actions], &device);

        let mode_mdp = ModeConditionedMDP::new(
            vec![p_closed, p_open],
            goal_fn,
            constraint_fn,
            Box::new(KeyDoorMode::new(0, 0)),
            30,
        );

        assert!(mode_mdp.is_ok());
        let mdp = mode_mdp.unwrap();
        assert_eq!(mdp.n_modes(), 2);
        assert_eq!(mdp.n_states(), 3);
        assert_eq!(mdp.n_actions(), 2);
    }

    #[test]
    fn test_mode_conditioned_mdp_wrong_number_of_modes() {
        use crate::backend::{default_device, DefaultBackend};

        let device = default_device();

        let n_states = 3;
        let n_actions = 2;

        // Only provide 1 transition kernel
        let p = Tensor::<DefaultBackend, 3>::zeros([n_states, n_actions, n_states], &device);
        let goal_fn = Tensor::<DefaultBackend, 2>::zeros([n_states, n_actions], &device);
        let constraint_fn = Tensor::<DefaultBackend, 2>::ones([n_states, n_actions], &device);

        // But mode function expects 2 modes
        let result = ModeConditionedMDP::new(
            vec![p],
            goal_fn,
            constraint_fn,
            Box::new(KeyDoorMode::new(0, 0)), // Expects 2 modes
            30,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_mode_conditioned_get_transition_for_hl_state() {
        use crate::backend::{default_device, DefaultBackend};

        let device = default_device();

        let n_states = 3;
        let n_actions = 2;

        let p_closed = Tensor::<DefaultBackend, 3>::zeros([n_states, n_actions, n_states], &device);
        let p_open = Tensor::<DefaultBackend, 3>::ones([n_states, n_actions, n_states], &device);

        let goal_fn = Tensor::<DefaultBackend, 2>::zeros([n_states, n_actions], &device);
        let constraint_fn = Tensor::<DefaultBackend, 2>::ones([n_states, n_actions], &device);

        let mode_mdp = ModeConditionedMDP::new(
            vec![p_closed, p_open],
            goal_fn,
            constraint_fn,
            Box::new(KeyDoorMode::new(0, 0)),
            30,
        )
        .unwrap();

        // Test with key bit = 0 (door closed)
        let no_key = vec![HLState::BinaryVector(vec![false])];
        let trans_closed = mode_mdp.get_transition_for_hl_state(&no_key);

        // Should get first tensor (all zeros)
        let sample = trans_closed.clone().slice([0..1, 0..1, 0..1]).into_scalar().elem::<f32>();
        assert_eq!(sample, 0.0);

        // Test with key bit = 1 (door open)
        let has_key = vec![HLState::BinaryVector(vec![true])];
        let trans_open = mode_mdp.get_transition_for_hl_state(&has_key);

        // Should get second tensor (all ones)
        let sample = trans_open.clone().slice([0..1, 0..1, 0..1]).into_scalar().elem::<f32>();
        assert_eq!(sample, 1.0);
    }

    #[test]
    fn test_mode_conditioned_to_task_mdp() {
        use crate::backend::{default_device, DefaultBackend};

        let device = default_device();

        let n_states = 3;
        let n_actions = 2;

        let p_closed = Tensor::<DefaultBackend, 3>::zeros([n_states, n_actions, n_states], &device);
        let p_open = Tensor::<DefaultBackend, 3>::ones([n_states, n_actions, n_states], &device);

        let goal_fn = Tensor::<DefaultBackend, 2>::zeros([n_states, n_actions], &device);
        let constraint_fn = Tensor::<DefaultBackend, 2>::ones([n_states, n_actions], &device);

        let mode_mdp = ModeConditionedMDP::new(
            vec![p_closed.clone(), p_open.clone()],
            goal_fn,
            constraint_fn,
            Box::new(KeyDoorMode::new(0, 0)),
            30,
        )
        .unwrap();

        // Convert to TaskMDP for MODE_CLOSED
        let task_mdp_closed = mode_mdp.to_task_mdp(MODE_CLOSED).unwrap();

        // Convert to TaskMDP for MODE_OPEN
        let task_mdp_open = mode_mdp.to_task_mdp(MODE_OPEN).unwrap();

        // Verify transitions are different
        let sample_closed = task_mdp_closed.transition.clone()
            .slice([0..1, 0..1, 0..1]).into_scalar().elem::<f32>();
        let sample_open = task_mdp_open.transition.clone()
            .slice([0..1, 0..1, 0..1]).into_scalar().elem::<f32>();

        assert_eq!(sample_closed, 0.0);
        assert_eq!(sample_open, 1.0);
    }
}

