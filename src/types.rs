
//! Core type definitions for STOK implementation.
//!
//! Provides newtype wrappers to prevent index confusion at compile time,
//! dimension tracking, and error types.

use std::fmt;

// ============================================================================
// Index Newtypes - Compile-time safety against mixing indices
// ============================================================================

/// State space index wrapper
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateIdx(pub usize);

impl From<usize> for StateIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<StateIdx> for usize {
    fn from(s: StateIdx) -> usize {
        s.0
    }
}

/// Action space index wrapper
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ActionIdx(pub usize);

impl From<usize> for ActionIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<ActionIdx> for usize {
    fn from(a: ActionIdx) -> usize {
        a.0
    }
}

/// Time step index wrapper
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimeIdx(pub usize);

impl From<usize> for TimeIdx {
    fn from(v: usize) -> Self {
        Self(v)
    }
}

impl From<TimeIdx> for usize {
    fn from(t: TimeIdx) -> usize {
        t.0
    }
}

impl TimeIdx {
    /// Saturating subtraction for time indices
    pub fn saturating_sub(&self, other: usize) -> TimeIdx {
        TimeIdx(self.0.saturating_sub(other))
    }
}

/// Goal identifier for GoalKernel management (Phase 4)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GoalId(pub u32);

impl GoalId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

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

/// MDP dimension specification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MDPDimensions {
    pub n_states: usize,
    pub n_actions: usize,
    pub max_time: usize,
}

impl MDPDimensions {
    pub fn new(n_states: usize, n_actions: usize, max_time: usize) -> Self {
        Self {
            n_states,
            n_actions,
            max_time,
        }
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

    /// Total size of state-time tensor slice
    pub fn state_time_size(&self) -> usize {
        self.n_states * self.max_time
    }

    /// Total size of transition tensor
    pub fn transition_size(&self) -> usize {
        self.n_states * self.n_actions * self.n_states
    }

    /// Total size of STOK tensor (η⁺ or η⁻)
    pub fn stok_size(&self) -> usize {
        self.n_states * self.n_states * self.max_time
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// STOK computation errors
#[derive(Debug, Clone)]
pub enum StokError {
    /// Tensor dimension mismatch
    DimensionMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid dimension value
    InvalidDimension {
        name: String,
        value: usize,
        reason: String,
    },

    /// Probability value out of bounds
    InvalidProbability {
        value: f32,
        context: String,
    },

    /// Tensor not properly normalized
    NotNormalized {
        sum: f32,
        expected: f32,
        tolerance: f32,
    },

    /// Device/backend error
    DeviceError(String),

    /// Feasibility iteration failed to converge
    ConvergenceFailure {
        iterations: usize,
        delta: f32,
    },

    /// Composition error
    CompositionError {
        message: String,
    },
}

impl fmt::Display for StokError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::InvalidDimension { name, value, reason } => {
                write!(f, "Invalid dimension '{}' = {}: {}", name, value, reason)
            }
            Self::InvalidProbability { value, context } => {
                write!(f, "Invalid probability {} in {}", value, context)
            }
            Self::NotNormalized { sum, expected, tolerance } => {
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
        }
    }
}

impl std::error::Error for StokError {}

// ============================================================================
// Numerical Constants
// ============================================================================

/// Default convergence tolerance for feasibility iteration
pub const DEFAULT_CONVERGENCE_TOLERANCE: f32 = 1e-6;

/// Default tolerance for probability validation
pub const DEFAULT_PROBABILITY_TOLERANCE: f32 = 1e-5;

/// Minimum probability value (avoid log(0))
pub const MIN_PROBABILITY: f32 = 1e-7;

/// Maximum probability value
pub const MAX_PROBABILITY: f32 = 1.0 - 1e-7;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_idx_conversion() {
        let idx = StateIdx::from(42);
        assert_eq!(usize::from(idx), 42);
    }

    #[test]
    fn test_time_idx_saturating_sub() {
        let t = TimeIdx(5);
        assert_eq!(t.saturating_sub(3), TimeIdx(2));
        assert_eq!(t.saturating_sub(10), TimeIdx(0));
    }

    #[test]
    fn test_dimensions_validation() {
        let valid = MDPDimensions::new(10, 4, 20);
        assert!(valid.validate().is_ok());

        let invalid = MDPDimensions::new(0, 4, 20);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_dimensions_sizes() {
        let dims = MDPDimensions::new(10, 4, 20);
        assert_eq!(dims.state_time_size(), 200);
        assert_eq!(dims.transition_size(), 400);
        assert_eq!(dims.stok_size(), 2000);
    }

    #[test]
    fn test_goal_id_display() {
        let g = GoalId::new(7);
        assert_eq!(format!("{}", g), "Goal(7)");
    }
}