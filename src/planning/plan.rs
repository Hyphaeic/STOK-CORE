//! # Plan Representation and Validation
//!
//! Defines the Plan struct for representing sequences of options
//! and utilities for validation and simulation.

use crate::types::{GoalId, StateIdx};
use burn::prelude::Backend;
use rand::Rng;
use std::fmt;

/// A sequence of options forming a plan
///
/// Represents an ordered sequence of goal-conditioned options to execute.
///
/// # Example
///
/// ```rust,ignore
/// let plan = Plan {
///     options: vec![GoalId(0), GoalId(1), GoalId(2)],
///     feasibility: 0.85,
///     expected_time: Some(12.5),
///     initial_state: StateIdx(0),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct Plan {
    /// Ordered sequence of option IDs to execute
    pub options: Vec<GoalId>,

    /// Expected cumulative feasibility (product of individual κ values)
    pub feasibility: f32,

    /// Expected total time (if computed)
    pub expected_time: Option<f32>,

    /// Initial state this plan is valid from
    pub initial_state: StateIdx,
}

impl Plan {
    /// Create empty plan
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Starting state
    ///
    /// # Returns
    ///
    /// Empty plan with feasibility 1.0
    pub fn empty(initial_state: StateIdx) -> Self {
        Self {
            options: vec![],
            feasibility: 1.0,
            expected_time: Some(0.0),
            initial_state,
        }
    }

    /// Append an option to the plan
    ///
    /// Updates cumulative feasibility and expected time.
    ///
    /// # Arguments
    ///
    /// * `goal` - Goal ID to append
    /// * `step_feasibility` - κ for this step
    /// * `step_time` - Expected time for this step
    pub fn append(&mut self, goal: GoalId, step_feasibility: f32, step_time: Option<f32>) {
        self.options.push(goal);
        self.feasibility *= step_feasibility;

        if let (Some(ref mut total), Some(step)) = (&mut self.expected_time, step_time) {
            *total += step;
        } else {
            self.expected_time = None;
        }
    }

    /// Get number of options in plan
    pub fn len(&self) -> usize {
        self.options.len()
    }

    /// Check if plan is empty
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plan[")?;
        for (i, goal) in self.options.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{}", goal)?;
        }
        write!(
            f,
            "] (κ={:.3}, t={:?})",
            self.feasibility, self.expected_time
        )
    }
}

/// Plan validation errors
#[derive(Debug, Clone)]
pub enum PlanError {
    /// Plan has no options
    EmptyPlan,

    /// Goal ID not found in kernel
    UnknownGoal(GoalId),

    /// Step in plan is infeasible from current state
    InfeasibleStep {
        /// Goal that cannot be reached
        goal: GoalId,
        /// State from which it's infeasible
        state: StateIdx,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPlan => write!(f, "Plan has no options"),
            Self::UnknownGoal(goal) => write!(f, "Unknown goal: {}", goal),
            Self::InfeasibleStep { goal, state } => {
                write!(f, "Infeasible step: goal {} from state {:?}", goal, state)
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Result of simulating a plan
#[derive(Clone, Debug)]
pub struct SimulationResult {
    /// Success rate across all simulations
    pub success_rate: f32,

    /// Mean time for successful runs
    pub mean_time: f32,

    /// All simulated trajectories
    pub trajectories: Vec<SimulatedTrajectory>,
}

/// Single simulated trajectory
#[derive(Clone, Debug)]
pub struct SimulatedTrajectory {
    /// States visited
    pub states: Vec<usize>,

    /// Times at each state
    pub times: Vec<usize>,

    /// Whether trajectory succeeded
    pub success: bool,
}

/// Simulate plan execution with Monte Carlo
///
/// Samples multiple trajectories through the plan to estimate success rate.
///
/// # Arguments
///
/// * `plan` - Plan to simulate
/// * `goal_kernel` - Goal kernel containing STOKs
/// * `n_simulations` - Number of trajectories to sample
/// * `rng` - Random number generator
///
/// # Returns
///
/// Simulation results with success rate and trajectories
///
/// # Example
///
/// ```rust,ignore
/// let mut rng = rand::thread_rng();
/// let result = simulate_plan(&plan, &goal_kernel, 1000, &mut rng);
/// println!("Success rate: {:.1}%", result.success_rate * 100.0);
/// ```
pub fn simulate_plan<B: Backend>(
    plan: &Plan,
    goal_kernel: &super::GoalKernel<B>,
    n_simulations: usize,
    rng: &mut impl Rng,
) -> SimulationResult {
    use super::sampling::{STOKSampler, TerminationOutcome};

    let mut successes = 0;
    let mut total_time = 0.0f32;
    let mut trajectories = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        let mut state = plan.initial_state.0;
        let mut time = 0usize;
        let mut states = vec![state];
        let mut times = vec![time];
        let mut success = true;

        for &goal in &plan.options {
            // Get STOK for this goal
            let stok = match goal_kernel.get_stok(goal) {
                Some(s) => s,
                None => {
                    success = false;
                    break;
                }
            };

            // Sample outcome
            let outcome = STOKSampler::sample_outcome(stok, state, rng);

            match outcome {
                TerminationOutcome::Success => {
                    let (next_state, duration) = STOKSampler::sample_termination(stok, state, rng);
                    state = next_state;
                    time += duration;
                }
                TerminationOutcome::Failure => {
                    success = false;
                    break;
                }
            }

            states.push(state);
            times.push(time);
        }

        if success {
            successes += 1;
            total_time += time as f32;
        }

        trajectories.push(SimulatedTrajectory {
            states,
            times,
            success,
        });
    }

    SimulationResult {
        success_rate: successes as f32 / n_simulations as f32,
        mean_time: if successes > 0 {
            total_time / successes as f32
        } else {
            f32::INFINITY
        },
        trajectories,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_empty() {
        let plan = Plan::empty(StateIdx(0));
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        assert_eq!(plan.feasibility, 1.0);
        assert_eq!(plan.expected_time, Some(0.0));
    }

    #[test]
    fn test_plan_append() {
        let mut plan = Plan::empty(StateIdx(0));
        plan.append(GoalId(1), 0.9, Some(5.0));
        plan.append(GoalId(2), 0.8, Some(3.0));

        assert_eq!(plan.len(), 2);
        assert!((plan.feasibility - 0.72).abs() < 1e-5); // 0.9 * 0.8
        assert_eq!(plan.expected_time, Some(8.0)); // 5 + 3
    }

    #[test]
    fn test_plan_display() {
        let plan = Plan {
            options: vec![GoalId(0), GoalId(1)],
            feasibility: 0.85,
            expected_time: Some(10.5),
            initial_state: StateIdx(0),
        };

        let display = format!("{}", plan);
        assert!(display.contains("Goal(0)"));
        assert!(display.contains("Goal(1)"));
        assert!(display.contains("0.850"));
    }
}
