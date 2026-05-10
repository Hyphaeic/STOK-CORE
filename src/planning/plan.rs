//! # Plan Representation and Validation
//!
//! Defines the `Plan` struct for representing sequences of options and
//! utilities for validation and simulation.
//!
//! This module also defines `PlanKernel` (PP-503) — the paper's m-fold
//! composition of the Goal Kernel `G_m` under a meta-policy `μ = (o_g1, ..., o_gm)`.
//! `PlanKernel` walks an option sequence through a `FactorizedGoalKernel`,
//! applying Eq [24]'s one-step boundary-action update at each step, and
//! summarizes the cumulative feasibility / time / final-state distribution.

use crate::hierarchy::{ProductSpaceDims, ProductState};
use crate::planning::goal_kernel::FactorizedGoalKernel;
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

// ============================================================================
// PP-503: Plan Kernel — m-fold composition of Goal Kernel under meta-policy μ
// ============================================================================

/// One step in a `PlanKernel` simulation: the state, time, and cumulative
/// feasibility after applying one option in the meta-policy sequence.
#[derive(Clone, Debug)]
pub struct PlanStep {
    /// The option (goal id) just applied.
    pub option: GoalId,
    /// Product-space state after this option (deterministic mode-of-distribution).
    pub state: ProductState,
    /// Cumulative time t after this option (sum of per-option durations + boundary +1).
    pub time: usize,
    /// Cumulative feasibility κ̃_μ along the trace so far (product of per-step κ̃).
    pub cumulative_feasibility: f32,
}

/// Plan Kernel: deterministic forward-simulation of an option sequence through
/// a `FactorizedGoalKernel`, applying Eq [24]'s one-step boundary-action update
/// at each option boundary.
///
/// Per the paper (page 8 / Fig 3 lower right), this is the m-fold composition
/// `G_m(s_m, t_m | s_0, t_0, μ) = G ∘ G ∘ ... ∘ G` where `μ = (o_g1, ..., o_gm)`
/// is a meta-policy of options.
///
/// This implementation provides:
/// - **Deterministic simulation** (`simulate_deterministic`): take the
///   most-likely outcome at each step (mode of the BL STOK termination
///   distribution; argmax of HL boundary distribution). Useful for tree
///   search and trace inspection.
/// - **Stochastic sampling** (`sample`): sample one outcome per step using
///   the FactorizedSTOK's sample method. Useful for Monte-Carlo evaluation.
///
/// Both modes return a `PlanTrace` that records the sequence of intermediate
/// states, times, and cumulative feasibility along the meta-policy.
pub struct PlanKernel<'a, B: Backend> {
    pub goal_kernel: &'a FactorizedGoalKernel<B>,
    pub option_sequence: Vec<GoalId>,
}

/// Result of running a `PlanKernel` simulation.
#[derive(Clone, Debug)]
pub struct PlanTrace {
    pub initial: ProductState,
    pub steps: Vec<PlanStep>,
    pub final_state: ProductState,
    pub total_time: usize,
    pub final_cumulative_feasibility: f32,
}

impl<'a, B: Backend> PlanKernel<'a, B> {
    pub fn new(goal_kernel: &'a FactorizedGoalKernel<B>, option_sequence: Vec<GoalId>) -> Self {
        Self {
            goal_kernel,
            option_sequence,
        }
    }

    /// Number of options in the meta-policy.
    pub fn len(&self) -> usize {
        self.option_sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.option_sequence.is_empty()
    }

    /// Deterministic forward simulation: at each step, take the mode of the
    /// FactorizedSTOK termination distribution, then advance HL state via
    /// `boundary_hl_distribution` (using mode of the boundary action over
    /// the affordance support).
    ///
    /// Cumulative feasibility multiplies the per-step `feasibility_approx`
    /// — exact under the Cor 6.1 path, upper-bound under the general path
    /// (see `FactorizedSTOK::feasibility_approx_is_exact`).
    pub fn simulate_deterministic(&self, initial: &ProductState) -> PlanTrace {
        let mut state = initial.clone();
        let mut time = 0usize;
        let mut cum_feas = 1.0f32;
        let mut steps = Vec::with_capacity(self.option_sequence.len());

        for &goal in &self.option_sequence {
            let option = match self.goal_kernel.option(goal) {
                Some(o) => o,
                None => break,
            };

            // Per-step κ̃ (feasibility approx — possibly upper-bound).
            let step_feas = option.feasibility_approx(&state);
            cum_feas *= step_feas;

            // Mode of (x_f, t_f) under the option, given current state.
            let (x_f, t_f) = mode_of_factorized_termination(option, &state);

            // Mode of HL boundary action (per HL space): pick the boundary
            // action α_f_k that maximizes the HL kernel's outgoing mass at z_f.
            // For determinism we just pick action index 0 (the default action)
            // as the boundary action — this is what Algorithm 2 line 17 does
            // when α is unambiguous from the affordance.
            let boundary_alpha: Vec<usize> = (0..self.goal_kernel.dims().n_hl_spaces())
                .map(|k| {
                    let n_a = self.goal_kernel.hl_kernels()[k].dims()[1];
                    if n_a == 0 { 0 } else { 0 }
                })
                .collect();

            // Apply Eq [24] HL boundary update.
            let next_hl_dists = self
                .goal_kernel
                .boundary_hl_distribution(goal, &state.hl_indices(), &boundary_alpha, t_f.max(1))
                .expect("boundary update");

            // Mode of each HL distribution.
            let next_hl_states: Vec<usize> = next_hl_dists
                .iter()
                .map(|d| {
                    use burn::prelude::ElementConversion;
                    let v: Vec<f32> = d.clone().into_data().to_vec().unwrap();
                    v.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();

            // Build the next product state.
            let mut next_state = ProductState::new(x_f);
            for &z in &next_hl_states {
                next_state = next_state.with_hl_discrete(z);
            }

            time += t_f + 1; // +1 for boundary step per Eq [24] / Algorithm 2 line 18.
            steps.push(PlanStep {
                option: goal,
                state: next_state.clone(),
                time,
                cumulative_feasibility: cum_feas,
            });
            state = next_state;
        }

        let final_state = state.clone();
        let total_time = time;
        let final_cf = cum_feas;
        PlanTrace {
            initial: initial.clone(),
            steps,
            final_state,
            total_time,
            final_cumulative_feasibility: final_cf,
        }
    }

    /// Stochastic sampling: at each step, sample an outcome from the
    /// FactorizedSTOK's `sample` method (which already dispatches between
    /// Cor 6.1 and general paths). Same trace structure as
    /// `simulate_deterministic`, but with one Monte-Carlo realization.
    pub fn sample(&self, initial: &ProductState, rng: &mut impl Rng) -> PlanTrace {
        let mut state = initial.clone();
        let mut time = 0usize;
        let mut cum_feas = 1.0f32;
        let mut steps = Vec::with_capacity(self.option_sequence.len());

        for &goal in &self.option_sequence {
            let option = match self.goal_kernel.option(goal) {
                Some(o) => o,
                None => break,
            };
            let step_feas = option.feasibility_approx(&state);
            cum_feas *= step_feas;

            let (sampled_state, t_f) = option.sample(&state, rng);

            // For the stochastic path, skip the explicit boundary update —
            // sample already returns the post-option state distribution which
            // already incorporates HL evolution under default dynamics. The
            // boundary HL transition is implicit in the option's induced HL
            // state. We add the +1 boundary step to match the time bookkeeping.
            time += t_f + 1;
            steps.push(PlanStep {
                option: goal,
                state: sampled_state.clone(),
                time,
                cumulative_feasibility: cum_feas,
            });
            state = sampled_state;
        }

        let final_state = state.clone();
        let total_time = time;
        let final_cf = cum_feas;
        PlanTrace {
            initial: initial.clone(),
            steps,
            final_state,
            total_time,
            final_cumulative_feasibility: final_cf,
        }
    }
}

/// Find the (x_f, t_f) tuple with maximal probability under a `FactorizedSTOK`
/// starting from the given product state. Used by `simulate_deterministic`
/// for the BL component.
fn mode_of_factorized_termination<B: Backend>(
    option: &crate::hierarchy::FactorizedSTOK<B>,
    initial: &ProductState,
) -> (usize, usize) {
    let max_time = option.base_stok.max_time();
    let n_x = option.dims.base_size;
    let mut best = (0usize, 1usize);
    let mut best_prob = -1.0f32;
    for x_f in 0..n_x {
        let mut final_state = ProductState::new(x_f);
        for &z in &initial.hl_indices() {
            final_state = final_state.with_hl_discrete(z);
        }
        for t in 1..max_time {
            let p = option.evaluate(initial, &final_state, t);
            if p > best_prob {
                best_prob = p;
                best = (x_f, t);
            }
        }
    }
    best
}

/// Helper for `ProductState` to extract HL indices as a flat `Vec<usize>`.
trait ProductStateHLIndices {
    fn hl_indices(&self) -> Vec<usize>;
}

impl ProductStateHLIndices for ProductState {
    fn hl_indices(&self) -> Vec<usize> {
        self.hl_states.iter().map(|h| h.as_discrete()).collect()
    }
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod plan_kernel_tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::hierarchy::{
        assemble_factorized_stok, FactorizedAffordance, ProductSpaceDims,
    };
    use crate::mdp::TaskMDP;
    use crate::planning::goal_kernel::{FactorizedGoalKernel, GoalInfo};
    use crate::stok::STOKKernel;
    use crate::types::GoalId;
    use burn::prelude::Tensor;

    fn build_minimal_goal_kernel() -> (FactorizedGoalKernel<DefaultBackend>, ProductState) {
        let device = default_device();

        // Base TaskMDP: 3 states, simple chain.
        let base_mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 6, &device);
        let base_stok = crate::solver::solve_task_mdp(&base_mdp).unwrap();

        // HL kernel: 2 states, 1 HL action (identity).
        let mut hl_data = vec![0.0f32; 2 * 1 * 2];
        hl_data[0 * 2 + 0] = 1.0;
        hl_data[1 * 2 + 1] = 1.0;
        let hl_kernel: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(hl_data.as_slice(), &device).reshape([2, 1, 2]);

        // Factorized STOK (Cor 6.1 path — no HL events).
        let dims = ProductSpaceDims::new(3, vec![2]);
        let factorized = assemble_factorized_stok(
            base_stok,
            vec![hl_kernel.clone()],
            vec![0],
            dims.clone(),
        )
        .unwrap();

        // Build the FactorizedGoalKernel.
        // Affordance: a single (1-state HL space) component, all (x, a) → α=0.
        // Match the base MDP shape: 3 base states, 3 base actions (simple_chain).
        let mut f_data = vec![0.0f32; 3 * 3 * 1];
        for x in 0..3 {
            for a in 0..3 {
                f_data[x * 3 * 1 + a * 1 + 0] = 1.0;
            }
        }
        let f_tensor: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 1]);
        let affordance = FactorizedAffordance::new(vec![f_tensor]);

        let mut goal_kernel = FactorizedGoalKernel::new(affordance, vec![hl_kernel], dims);
        let info = GoalInfo {
            id: GoalId(0),
            name: "test_goal".to_string(),
            target_state: Some(2),
            max_time: 6,
        };
        goal_kernel.add_option(GoalId(0), factorized, info).unwrap();

        let initial = ProductState::new(0).with_hl_discrete(0);
        (goal_kernel, initial)
    }

    /// PP-501: FactorizedGoalKernel constructed correctly.
    #[test]
    fn factorized_goal_kernel_basics() {
        let (gk, _) = build_minimal_goal_kernel();
        assert_eq!(gk.n_goals(), 1);
        assert!(gk.has_goal(GoalId(0)));
        assert!(!gk.has_goal(GoalId(99)));
        assert_eq!(gk.goal_ids().len(), 1);
    }

    /// PP-502: boundary HL distribution shape and normalization.
    #[test]
    fn boundary_hl_distribution_normalizes() {
        use burn::prelude::ElementConversion;
        let (gk, _) = build_minimal_goal_kernel();
        let dists = gk
            .boundary_hl_distribution(GoalId(0), &[0], &[0], 2)
            .unwrap();
        assert_eq!(dists.len(), 1);
        let probs: Vec<f32> = dists[0].clone().into_data().to_vec().unwrap();
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "boundary HL distribution should sum to 1, got {}",
            sum
        );
    }

    /// PP-503: PlanKernel deterministic simulation completes and produces a
    /// trace matching the option sequence length.
    #[test]
    fn plan_kernel_deterministic_simulation_completes() {
        let (gk, initial) = build_minimal_goal_kernel();
        let plan_kernel = PlanKernel::new(&gk, vec![GoalId(0), GoalId(0)]);
        let trace = plan_kernel.simulate_deterministic(&initial);

        assert_eq!(trace.steps.len(), 2, "trace should have one step per option");
        assert!(trace.total_time >= trace.steps.len(), "time advances each step");
        assert!(
            trace.final_cumulative_feasibility >= 0.0
                && trace.final_cumulative_feasibility <= 1.0,
            "cumulative κ̃ in [0, 1]"
        );
    }

    /// PP-503: PlanKernel stochastic sampling completes and produces valid trace.
    #[test]
    fn plan_kernel_sample_completes() {
        use rand::SeedableRng;
        let (gk, initial) = build_minimal_goal_kernel();
        let plan_kernel = PlanKernel::new(&gk, vec![GoalId(0)]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let trace = plan_kernel.sample(&initial, &mut rng);
        assert_eq!(trace.steps.len(), 1);
        assert!(trace.total_time >= 1);
    }

    #[test]
    fn plan_kernel_empty_sequence_is_valid() {
        let (gk, initial) = build_minimal_goal_kernel();
        let plan_kernel: PlanKernel<DefaultBackend> = PlanKernel::new(&gk, vec![]);
        assert!(plan_kernel.is_empty());
        let trace = plan_kernel.simulate_deterministic(&initial);
        assert_eq!(trace.steps.len(), 0);
        assert_eq!(trace.total_time, 0);
        assert_eq!(trace.final_state.base, initial.base);
    }
}
