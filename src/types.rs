//! Core type definitions for STOK computation.
//!
//! This module provides type-safe wrappers for indices and dimensions,
//! ensuring compile-time prevention of index confusion.
//!
//! # Reference
//!
//! Ringstrom, T., & Schrater, P. (2025). Compositionality and Bounds for
//! Optimal Value Functions in Reinforcement Learning.
//!
//! # Type Safety
//!
//! The newtype wrappers (`StateIdx`, `ActionIdx`, `TimeIdx`) prevent
//! accidental mixing of different index types at compile time:
//!
//! ```rust,ignore
//! let state: StateIdx = StateIdx(5);
//! let action: ActionIdx = ActionIdx(2);
//! // These are different types - cannot be confused!
//! ```

use std::fmt;

// ============================================================================
// Index Newtypes - Compile-time index safety
// ============================================================================

/// State index newtype wrapper.
///
/// Wraps a `usize` to prevent accidental mixing with action or time indices.
/// Used for indexing into state-dependent tensors like κ, η, P.
///
/// # Example
///
/// ```rust
/// use stok_core::types::StateIdx;
///
/// let state = StateIdx::from(5);
/// let raw: usize = state.into();
/// assert_eq!(raw, 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateIdx(pub usize);

impl From<usize> for StateIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<StateIdx> for usize {
    fn from(v: StateIdx) -> Self {
        v.0
    }
}

impl fmt::Display for StateIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

/// Action index newtype wrapper.
///
/// Wraps a `usize` to prevent accidental mixing with state or time indices.
/// Used for indexing into action-dependent tensors like f_g, f_c, π.
///
/// # Example
///
/// ```rust
/// use stok_core::types::ActionIdx;
///
/// let action = ActionIdx::from(2);
/// let raw: usize = action.into();
/// assert_eq!(raw, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionIdx(pub usize);

impl From<usize> for ActionIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<ActionIdx> for usize {
    fn from(v: ActionIdx) -> Self {
        v.0
    }
}

impl fmt::Display for ActionIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A{}", self.0)
    }
}

/// Time index newtype wrapper.
///
/// Wraps a `usize` representing discrete time steps in STOK computation.
/// Used for indexing into time-dependent tensors like η⁺, η⁻.
///
/// Per Equations [9-12] in Ringstrom & Schrater (2025), time indices
/// are used in the recursive STOK construction.
///
/// # Example
///
/// ```rust
/// use stok_core::types::TimeIdx;
///
/// let t = TimeIdx(5);
/// let t_prev = t.saturating_sub(1);
/// assert_eq!(t_prev, TimeIdx(4));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimeIdx(pub usize);

impl TimeIdx {
    /// Saturating subtraction for time indices.
    ///
    /// Returns `TimeIdx(0)` if the subtraction would underflow.
    /// Useful for backward time iteration in STOK construction.
    pub fn saturating_sub(&self, other: usize) -> TimeIdx {
        TimeIdx(self.0.saturating_sub(other))
    }

    /// Returns the next time index, if within bounds.
    pub fn checked_add(&self, other: usize) -> Option<TimeIdx> {
        self.0.checked_add(other).map(TimeIdx)
    }
}

impl From<usize> for TimeIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<TimeIdx> for usize {
    fn from(v: TimeIdx) -> Self {
        v.0
    }
}

impl fmt::Display for TimeIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={}", self.0)
    }
}

/// Goal identifier for multi-goal planning (Phase 4).
///
/// Uniquely identifies a goal in the Goal Kernel manager.
/// See Equation [24] in Ringstrom & Schrater (2025) for Goal Kernel definition.
///
/// # Example
///
/// ```rust
/// use stok_core::types::GoalId;
///
/// let goal = GoalId::new(0);
/// assert_eq!(format!("{}", goal), "Goal(0)");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GoalId(pub u32);

impl GoalId {
    /// Create a new goal identifier.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw identifier value.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl From<u32> for GoalId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl fmt::Display for GoalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Goal({})", self.0)
    }
}

// ============================================================================
// Dimension Tracking
// ============================================================================

/// MDP dimension specification.
///
/// Tracks the sizes of the state space, action space, and time horizon
/// for consistent tensor shape validation throughout STOK computation.
///
/// # Fields
///
/// * `n_states` - Number of states |S| in the MDP
/// * `n_actions` - Number of actions |A| available at each state
/// * `max_time` - Maximum time horizon T for STOK computation
///
/// # Tensor Shapes (Reference)
///
/// | Tensor | Shape | Description |
/// |--------|-------|-------------|
/// | P | [S, A, S] | Transition dynamics P(x'│x,a) |
/// | f_g, f_c | [S, A] | Goal/constraint functions |
/// | κ | [S] | Cumulative feasibility (Eq. [7]) |
/// | η⁺, η⁻ | [S, S, T] | STOK components (Eq. [15-16]) |
/// | χ | [S, S] | State Option Kernel (Eq. [19]) |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MDPDimensions {
    /// Number of states in the MDP
    pub n_states: usize,
    /// Number of actions available at each state
    pub n_actions: usize,
    /// Maximum time horizon for STOK computation
    pub max_time: usize,
}

impl MDPDimensions {
    /// Create new dimension specification.
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of states (must be > 0)
    /// * `n_actions` - Number of actions (must be > 0)
    /// * `max_time` - Time horizon (must be > 0)
    pub fn new(n_states: usize, n_actions: usize, max_time: usize) -> Self {
        Self {
            n_states,
            n_actions,
            max_time,
        }
    }

    /// Validate dimensions are non-zero.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all dimensions are positive, `Err(StokError)` otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use stok_core::types::MDPDimensions;
    ///
    /// let valid = MDPDimensions::new(10, 4, 20);
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = MDPDimensions::new(0, 4, 20);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), StokError> {
        if self.n_states == 0 {
            return Err(StokError::InvalidDimension {
                name: "n_states".into(),
                value: 0,
                reason: "must be > 0".into(),
            });
        }
        if self.n_actions == 0 {
            return Err(StokError::InvalidDimension {
                name: "n_actions".into(),
                value: 0,
                reason: "must be > 0".into(),
            });
        }
        if self.max_time == 0 {
            return Err(StokError::InvalidDimension {
                name: "max_time".into(),
                value: 0,
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Total size of state-time tensor slice.
    ///
    /// Returns `n_states * max_time`, useful for buffer allocation.
    pub fn state_time_size(&self) -> usize {
        self.n_states * self.max_time
    }

    /// Total size of transition tensor P[S, A, S].
    ///
    /// Returns `n_states * n_actions * n_states`.
    pub fn transition_size(&self) -> usize {
        self.n_states * self.n_actions * self.n_states
    }

    /// Total size of STOK tensor η[S, S, T].
    ///
    /// Returns `n_states * n_states * max_time`.
    pub fn stok_size(&self) -> usize {
        self.n_states * self.n_states * self.max_time
    }
}

// ============================================================================
// STOK Dimension Tracking (no action dimension - policy already determined)
// ============================================================================

/// STOK dimension specification.
///
/// Unlike MDPDimensions, STOKs don't have an action dimension because
/// the policy is already fixed. Per Eq. [17] of Ringstrom & Schrater (2025),
/// a STOK is "a transition kernel with one action, o_g" — the option itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct STOKDimensions {
    pub n_states: usize,
    pub max_time: usize,
}

impl STOKDimensions {
    pub fn new(n_states: usize, max_time: usize) -> Self {
        Self { n_states, max_time }
    }

    /// Validate dimensions are non-zero
    pub fn validate(&self) -> Result<(), StokError> {
        if self.n_states == 0 {
            return Err(StokError::InvalidDimension {
                name: "n_states".into(),
                value: 0,
                reason: "must be > 0".into(),
            });
        }
        if self.max_time == 0 {
            return Err(StokError::InvalidDimension {
                name: "max_time".into(),
                value: 0,
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }

    /// Create from MDP dimensions (drops n_actions)
    pub fn from_mdp(mdp_dims: &MDPDimensions) -> Self {
        Self {
            n_states: mdp_dims.n_states,
            max_time: mdp_dims.max_time,
        }
    }

    /// Total size of STOK tensor (η⁺ or η⁻)
    pub fn stok_size(&self) -> usize {
        self.n_states * self.n_states * self.max_time
    }

    /// Total size of state-time tensor slice
    pub fn state_time_size(&self) -> usize {
        self.n_states * self.max_time
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// STOK computation errors.
///
/// Comprehensive error type covering all failure modes in STOK computation,
/// from dimension mismatches to convergence failures.
///
/// # Variants
///
/// * `DimensionMismatch` - Tensor shapes don't match expected dimensions
/// * `InvalidDimension` - A dimension parameter is invalid (e.g., zero)
/// * `InvalidProbability` - Probability value outside [0, 1]
/// * `NotNormalized` - Probability distribution doesn't sum to expected value
/// * `DeviceError` - GPU/backend initialization or execution error
/// * `ConvergenceFailure` - Feasibility iteration didn't converge
/// * `CompositionError` - Error during STOK composition (Phase 3)
#[derive(Debug, Clone)]
pub enum StokError {
    /// Tensor dimension mismatch.
    ///
    /// Occurs when tensor shapes don't match expected dimensions
    /// for operations like matrix multiplication or element-wise ops.
    DimensionMismatch {
        /// Expected tensor dimensions
        expected: Vec<usize>,
        /// Actual tensor dimensions
        got: Vec<usize>,
    },

    /// Numerical instability detected during computation.
    ///
    /// Occurs when values become NaN, infinite, or violate expected bounds
    /// during iterative computation.
    NumericalInstability {
        /// Location/context where instability was detected
        location: String,
        /// The problematic value that triggered the error
        value: f32,
    },

    /// Invalid dimension value.
    ///
    /// Occurs when a dimension parameter is invalid (e.g., zero states).
    InvalidDimension {
        /// Name of the dimension parameter
        name: String,
        /// The invalid value
        value: usize,
        /// Explanation of why it's invalid
        reason: String,
    },

    /// Probability value out of bounds [0, 1].
    ///
    /// Occurs when goal, constraint, or STOK values fall outside
    /// valid probability range.
    InvalidProbability {
        /// The out-of-bounds value
        value: f32,
        /// Context describing which tensor/field
        context: String,
    },

    /// Tensor not properly normalized.
    ///
    /// Occurs when transition probabilities don't sum to 1,
    /// or when STOK normalization invariants are violated.
    NotNormalized {
        /// Actual sum
        sum: f32,
        /// Expected sum (usually 1.0)
        expected: f32,
        /// Tolerance used for comparison
        tolerance: f32,
    },

    /// Device/backend error.
    ///
    /// Occurs during GPU initialization or tensor operations.
    DeviceError(String),

    /// Feasibility iteration failed to converge.
    ///
    /// Occurs when the κ-OKBE (Equation [7]) doesn't reach
    /// the convergence tolerance within max iterations.
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Final convergence delta ||κ_new - κ_old||_∞
        delta: f32,
    },

    /// Composition error (Phase 3).
    ///
    /// Occurs during Chapman-Kolmogorov composition (Equation [18]).
    CompositionError {
        /// Description of the composition error
        message: String,
    },

    /// Empty option sequence provided.
    ///
    /// Occurs when attempting to compose an empty sequence of options.
    EmptySequence,
}

impl fmt::Display for StokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "Dimension mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            }
            Self::InvalidDimension {
                name,
                value,
                reason,
            } => {
                write!(f, "Invalid dimension '{}' = {}: {}", name, value, reason)
            }
            Self::NumericalInstability { location, value } => {
                write!(
                    f,
                    "Numerical instability at {}: value = {}",
                    location, value
                )
            }
            Self::InvalidProbability { value, context } => {
                write!(f, "Invalid probability {} in {}", value, context)
            }
            Self::NotNormalized {
                sum,
                expected,
                tolerance,
            } => {
                write!(
                    f,
                    "Not normalized: sum = {}, expected = {} (±{})",
                    sum, expected, tolerance
                )
            }
            Self::DeviceError(msg) => write!(f, "Device error: {}", msg),
            Self::ConvergenceFailure { iterations, delta } => {
                write!(
                    f,
                    "Convergence failed after {} iterations (delta = {})",
                    iterations, delta
                )
            }
            Self::CompositionError { message } => {
                write!(f, "Composition error: {}", message)
            }
            Self::EmptySequence => {
                write!(f, "Empty option sequence provided for composition")
            }
        }
    }
}

impl std::error::Error for StokError {}

// ============================================================================
// Numerical Constants
// ============================================================================

/// Default convergence tolerance for feasibility iteration.
///
/// Used in κ-OKBE (Equation [7]) convergence check:
/// `||κ_new - κ_old||_∞ < tolerance`
pub const DEFAULT_CONVERGENCE_TOLERANCE: f32 = 1e-6;

/// Default tolerance for probability validation.
///
/// Used when checking row-stochastic property of transitions
/// and normalization of STOK distributions.
pub const DEFAULT_PROBABILITY_TOLERANCE: f32 = 1e-5;

/// Minimum probability value to avoid log(0).
///
/// Used for numerical stability in log-probability computations.
pub const MIN_PROBABILITY: f32 = 1e-7;

/// Maximum probability value.
///
/// `1.0 - ε` to avoid floating-point issues at boundary.
pub const MAX_PROBABILITY: f32 = 1.0 - 1e-7;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // StateIdx Tests
    // ========================================================================

    #[test]
    fn test_state_idx_conversion() {
        let idx = StateIdx::from(42);
        assert_eq!(usize::from(idx), 42);
    }

    #[test]
    fn test_state_idx_display() {
        let idx = StateIdx(7);
        assert_eq!(format!("{}", idx), "S7");
    }

    #[test]
    fn test_state_idx_ordering() {
        let s1 = StateIdx(5);
        let s2 = StateIdx(10);
        assert!(s1 < s2);
    }

    // ========================================================================
    // ActionIdx Tests
    // ========================================================================

    #[test]
    fn test_action_idx_conversion() {
        let idx = ActionIdx::from(3);
        assert_eq!(usize::from(idx), 3);
    }

    #[test]
    fn test_action_idx_display() {
        let idx = ActionIdx(2);
        assert_eq!(format!("{}", idx), "A2");
    }

    // ========================================================================
    // TimeIdx Tests
    // ========================================================================

    #[test]
    fn test_time_idx_saturating_sub() {
        let t = TimeIdx(5);
        assert_eq!(t.saturating_sub(3), TimeIdx(2));
        assert_eq!(t.saturating_sub(10), TimeIdx(0));
    }

    #[test]
    fn test_time_idx_checked_add() {
        let t = TimeIdx(5);
        assert_eq!(t.checked_add(3), Some(TimeIdx(8)));
        assert_eq!(TimeIdx(usize::MAX).checked_add(1), None);
    }

    #[test]
    fn test_time_idx_display() {
        let t = TimeIdx(10);
        assert_eq!(format!("{}", t), "t=10");
    }

    // ========================================================================
    // GoalId Tests
    // ========================================================================

    #[test]
    fn test_goal_id_display() {
        let g = GoalId::new(7);
        assert_eq!(format!("{}", g), "Goal(7)");
    }

    #[test]
    fn test_goal_id_as_u32() {
        let g = GoalId::new(42);
        assert_eq!(g.as_u32(), 42);
    }

    // ========================================================================
    // MDPDimensions Tests
    // ========================================================================

    #[test]
    fn test_dimensions_validation_valid() {
        let valid = MDPDimensions::new(10, 4, 20);
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_dimensions_validation_zero_states() {
        let invalid = MDPDimensions::new(0, 4, 20);
        let err = invalid.validate().unwrap_err();
        match err {
            StokError::InvalidDimension { name, .. } => {
                assert_eq!(name, "n_states");
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_dimensions_validation_zero_actions() {
        let invalid = MDPDimensions::new(10, 0, 20);
        let err = invalid.validate().unwrap_err();
        match err {
            StokError::InvalidDimension { name, .. } => {
                assert_eq!(name, "n_actions");
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_dimensions_validation_zero_time() {
        let invalid = MDPDimensions::new(10, 4, 0);
        let err = invalid.validate().unwrap_err();
        match err {
            StokError::InvalidDimension { name, .. } => {
                assert_eq!(name, "max_time");
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_dimensions_sizes() {
        let dims = MDPDimensions::new(10, 4, 20);
        assert_eq!(dims.state_time_size(), 200);
        assert_eq!(dims.transition_size(), 400);
        assert_eq!(dims.stok_size(), 2000);
    }

    // ========================================================================
    // StokError Tests
    // ========================================================================

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = StokError::DimensionMismatch {
            expected: vec![10, 4, 10],
            got: vec![10, 3, 10],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Dimension mismatch"));
        assert!(msg.contains("[10, 4, 10]"));
    }

    #[test]
    fn test_error_display_invalid_probability() {
        let err = StokError::InvalidProbability {
            value: 1.5,
            context: "goal_fn".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1.5"));
        assert!(msg.contains("goal_fn"));
    }

    #[test]
    fn test_error_display_not_normalized() {
        let err = StokError::NotNormalized {
            sum: 0.95,
            expected: 1.0,
            tolerance: 1e-5,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0.95"));
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_error_display_convergence_failure() {
        let err = StokError::ConvergenceFailure {
            iterations: 1000,
            delta: 0.01,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1000"));
        assert!(msg.contains("0.01"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let err = StokError::DeviceError("test".into());
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &err;
    }

    // ========================================================================
    // STOKDimensions Tests
    // ========================================================================

    #[test]
    fn test_stok_dimensions_new() {
        let dims = STOKDimensions::new(10, 5);
        assert_eq!(dims.n_states, 10);
        assert_eq!(dims.max_time, 5);
    }

    #[test]
    fn test_stok_dimensions_validate_valid() {
        let dims = STOKDimensions::new(10, 5);
        assert!(dims.validate().is_ok());
    }

    #[test]
    fn test_stok_dimensions_validate_zero_states() {
        let dims = STOKDimensions::new(0, 5);
        assert!(dims.validate().is_err());
    }

    #[test]
    fn test_stok_dimensions_validate_zero_time() {
        let dims = STOKDimensions::new(10, 0);
        assert!(dims.validate().is_err());
    }

    #[test]
    fn test_stok_dimensions_from_mdp() {
        let mdp_dims = MDPDimensions::new(10, 3, 5);
        let stok_dims = STOKDimensions::from_mdp(&mdp_dims);
        assert_eq!(stok_dims.n_states, 10);
        assert_eq!(stok_dims.max_time, 5);
        // n_actions is intentionally dropped
    }

    #[test]
    fn test_stok_dimensions_sizes() {
        let dims = STOKDimensions::new(4, 3);
        assert_eq!(dims.stok_size(), 4 * 4 * 3); // S * S * T
        assert_eq!(dims.state_time_size(), 4 * 3); // S * T
    }
}
