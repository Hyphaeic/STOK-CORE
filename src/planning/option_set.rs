//! # Option-Set Builders for Theorems 2.2 and 2.3
//!
//! PP-701: paper-faithful builders for the two option-set sufficiency results
//! from the paper:
//!
//! - **Theorem 2.2 (State-Action Option Set):** for a deterministic CTMDP,
//!   the size-`|X|` set `O_X,A` (one option per BL state, with any final
//!   action in `A`) is sufficient for tree search to find an optimal open-loop
//!   policy. Each option's goal is "reach state x" — `f_g(x', a) = 1` iff
//!   `x' = x`, independent of `a`.
//!
//! - **Theorem 2.3 (Affordance Option Set):** when `P_α` is static, the size-
//!   `|A_α|` set `O_Xα` (one option per affordance HL action) is sufficient.
//!   Each option's goal is "induce HL action α" — `f_g(x, a) = 1` iff
//!   `F(α | x, a) > 0`, where F is the affordance function.
//!
//! Both builders consume a base TaskMDP "template" (its transitions and
//! constraint function are reused across goals; only `f_g` varies per option)
//! and return a `Vec<STOKKernel<B>>` — one STOK per goal in the option set.
//!
//! These builders supersede the example-grade ad-hoc construction in
//! `examples/honey_badger_*` etc., which are now consumers of these APIs.

use crate::hierarchy::{AffordanceFunction, FactorizedAffordance, HLAction};
use crate::mdp::TaskMDP;
use crate::solver::solve_task_mdp;
use crate::stok::STOKKernel;
use crate::types::StokError;
use burn::prelude::*;

/// Build `O_X` (Theorem 2.2): one option per BL state. Each option's goal is
/// "reach state x" — `f_g(x', a) = 1` iff `x' = x`, regardless of `a`.
///
/// Returns a vector of STOKs of length `base_mdp_template.n_states()`,
/// in BL state order.
///
/// The base MDP's transition and constraint functions are reused across
/// options; only the goal function changes per option.
pub fn build_state_option_set<B: Backend>(
    base_mdp_template: &TaskMDP<B>,
) -> Result<Vec<STOKKernel<B>>, StokError> {
    let device = base_mdp_template.device();
    let n_x = base_mdp_template.n_states();
    let n_a = base_mdp_template.n_actions();
    let max_time = base_mdp_template.max_time();

    let mut out = Vec::with_capacity(n_x);
    for goal_state in 0..n_x {
        // f_g(x', a) = 1 iff x' = goal_state.
        let mut g_data = vec![0.0f32; n_x * n_a];
        for a in 0..n_a {
            g_data[goal_state * n_a + a] = 1.0;
        }
        let goal: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(g_data.as_slice(), &device).reshape([n_x, n_a]);

        let mdp_g = TaskMDP::new(
            base_mdp_template.transition.clone(),
            goal,
            base_mdp_template.constraint_fn.clone(),
            max_time,
        )?;
        let stok = solve_task_mdp(&mdp_g)?;
        out.push(stok);
    }
    Ok(out)
}

/// Build `O_Xα` (Theorem 2.3): one option per affordance HL action in the
/// specified HL space. Each option's goal is "induce HL action α" —
/// `f_g(x, a) = 1` iff `F(α | x, a) > 0`.
///
/// Returns a vector of STOKs of length `affordance.n_hl_actions_for_space(hl_space_index)`,
/// in HL action order.
///
/// Per Theorem 2.3, the assumption `P_α static` (HL transitions don't depend
/// on the BL state) is the caller's responsibility — the builder does not
/// validate it. If `P_α` is non-static, the resulting option set may not be
/// sufficient for optimal tree search (but the STOKs themselves are still
/// well-defined).
pub fn build_affordance_option_set<B: Backend>(
    base_mdp_template: &TaskMDP<B>,
    affordance: &FactorizedAffordance<B>,
    hl_space_index: usize,
) -> Result<Vec<STOKKernel<B>>, StokError> {
    assert!(
        hl_space_index < affordance.n_hl_spaces(),
        "hl_space_index {} out of range",
        hl_space_index
    );
    assert_eq!(
        base_mdp_template.n_states(),
        affordance.n_base_states(),
        "base MDP n_states must match affordance"
    );
    assert_eq!(
        base_mdp_template.n_actions(),
        affordance.n_base_actions(),
        "base MDP n_actions must match affordance"
    );

    let device = base_mdp_template.device();
    let n_x = base_mdp_template.n_states();
    let n_a = base_mdp_template.n_actions();
    let max_time = base_mdp_template.max_time();
    let n_alpha = affordance.n_hl_actions_for_space(hl_space_index);

    let mut out = Vec::with_capacity(n_alpha);
    for alpha in 0..n_alpha {
        let hl_action = HLAction::Single {
            space_id: hl_space_index,
            action_id: alpha,
        };
        // f_g(x, a) = 1 iff F(α | x, a) > 0.
        let mut g_data = vec![0.0f32; n_x * n_a];
        for x in 0..n_x {
            for a in 0..n_a {
                let p = affordance.probability(&hl_action, x, a);
                if p > 1e-6 {
                    g_data[x * n_a + a] = 1.0;
                }
            }
        }
        let goal: Tensor<B, 2> =
            Tensor::<B, 1>::from_floats(g_data.as_slice(), &device).reshape([n_x, n_a]);

        let mdp_a = TaskMDP::new(
            base_mdp_template.transition.clone(),
            goal,
            base_mdp_template.constraint_fn.clone(),
            max_time,
        )?;
        let stok = solve_task_mdp(&mdp_a)?;
        out.push(stok);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};

    /// PP-701 / CHK-T2.2: state option set has exactly |X| options, one per state.
    #[test]
    fn state_option_set_has_one_option_per_state() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(5, 8, &device);
        let options = build_state_option_set(&mdp).expect("build state option set");
        assert_eq!(options.len(), 5);
        for stok in &options {
            assert_eq!(stok.n_states(), 5);
        }
    }

    /// PP-701 / CHK-T2.2: each state-option's STOK is feasible from at least
    /// the goal state itself (κ at the goal state is ≈ 1).
    #[test]
    fn state_option_set_each_option_is_feasible_at_its_goal() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(4, 8, &device);
        let options = build_state_option_set(&mdp).expect("build options");

        for (g, stok) in options.iter().enumerate() {
            let kappa: Vec<f32> = stok.kappa.clone().into_data().to_vec().unwrap();
            assert!(
                kappa[g] > 0.99,
                "option for goal state {} should have κ ≈ 1 at its goal, got {}",
                g,
                kappa[g]
            );
        }
    }

    /// PP-701 / CHK-T2.3: affordance option set has exactly |A_α| options,
    /// one per affordance HL action.
    #[test]
    fn affordance_option_set_has_one_option_per_hl_action() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 6, &device);

        // 3 base states × 3 base actions × 2 HL actions. Default: α=0.
        // (x=2, a=*) induces α=1.
        let mut f_data = vec![0.0f32; 3 * 3 * 2];
        for x in 0..3 {
            for a in 0..3 {
                f_data[x * 3 * 2 + a * 2 + 0] = 1.0;
            }
        }
        for a in 0..3 {
            f_data[2 * 3 * 2 + a * 2 + 0] = 0.0;
            f_data[2 * 3 * 2 + a * 2 + 1] = 1.0;
        }
        let f_tensor: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 2]);
        let affordance = FactorizedAffordance::new(vec![f_tensor]);

        let options = build_affordance_option_set(&mdp, &affordance, 0)
            .expect("build affordance option set");
        assert_eq!(options.len(), 2);
    }

    /// PP-701: builder validates dimension match between base MDP and affordance.
    #[test]
    #[should_panic(expected = "base MDP n_actions must match affordance")]
    fn affordance_option_set_panics_on_dim_mismatch() {
        let device = default_device();
        let mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 6, &device); // 3 actions

        // Affordance with WRONG n_actions = 2.
        let f_data = vec![1.0f32; 3 * 2 * 2];
        let f_tensor: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 2, 2]);
        let affordance = FactorizedAffordance::new(vec![f_tensor]);

        let _ = build_affordance_option_set(&mdp, &affordance, 0);
    }
}
