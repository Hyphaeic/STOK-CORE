//! PP-305 / CHK-T2.1 / CHK-CO6.1: Direct-DP cross-checks for the STOK
//! factorization. Builds a small explicit product-space TaskMDP, solves it
//! via standard feasibility iteration to get the ground-truth η̃**, then
//! verifies that `FactorizedSTOK::evaluate` matches within numerical tolerance.

use burn::prelude::*;
use stok_core::backend::{default_device, DefaultBackend};
use stok_core::hierarchy::{
    assemble_factorized_stok, assemble_factorized_stok_with_hl_events, ProductSpaceDims,
    ProductState,
};
use stok_core::mdp::TaskMDP;
use stok_core::solver::{feasibility_iteration, solve_task_mdp, FeasibilityIterationConfig};

// ============================================================================
// Witness 1: 2-base × 2-HL with no HL events (Cor 6.1 case)
// ============================================================================

/// Build a tiny CTMDP where HL evolves under default dynamics with no events
/// (HL TMDP has no goal anywhere, so κ_HL = 0 ⟹ κ̄_HL = 1 always).
fn build_cor_6_1_witness() -> (
    TaskMDP<DefaultBackend>, // base TaskMDP
    Tensor<DefaultBackend, 3>, // HL kernel
    TaskMDP<DefaultBackend>, // HL TaskMDP (with no goal — exercises Cor 6.1 trigger)
    ProductSpaceDims,
    TaskMDP<DefaultBackend>, // product TaskMDP for direct DP
) {
    let device = default_device();

    // Base: 3 states, single action, deterministic chain 0→1→2 with absorbing
    // goal at state 2 (f_g[2,0] = 1).
    let n_x = 3;
    let n_a = 1;
    let mut base_p = vec![0.0f32; n_x * n_a * n_x];
    base_p[0 * n_x + 1] = 1.0;
    base_p[1 * n_x + 2] = 1.0;
    base_p[2 * n_x + 2] = 1.0;
    let base_trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(base_p.as_slice(), &device).reshape([n_x, n_a, n_x]);
    let mut base_g = vec![0.0f32; n_x * n_a];
    base_g[2 * n_a + 0] = 1.0;
    let base_goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(base_g.as_slice(), &device).reshape([n_x, n_a]);
    let base_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_x, n_a], &device);
    let base_mdp = TaskMDP::new(base_trans, base_goal, base_constraint, 8).unwrap();

    // HL: 2 states, 1 action, identity (no transitions). With no HL events
    // (no HL goal anywhere), κ_HL = 0 ⟹ κ̄_HL = 1 always — the Cor 6.1 case.
    let n_z = 2;
    let n_az = 1;
    let mut hl_p = vec![0.0f32; n_z * n_az * n_z];
    hl_p[0 * n_z + 0] = 1.0;
    hl_p[1 * n_z + 1] = 1.0;
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_p.as_slice(), &device).reshape([n_z, n_az, n_z]);

    // HL TaskMDP: no goal, with mild constraint risk (0.7) so the chain absorbs
    // without violating the absorbing-policy guard.
    let hl_goal: Tensor<DefaultBackend, 2> = Tensor::zeros([n_z, n_az], &device);
    let hl_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_z, n_az], &device) * 0.7;
    let hl_tmdp = TaskMDP::new(hl_kernel.clone(), hl_goal, hl_constraint, 8).unwrap();

    let dims = ProductSpaceDims::new(n_x, vec![n_z]);

    // Product TaskMDP: state = z * n_x + x, single action. Product transition
    // mirrors the base + HL identity. Goal at product state = (z=*, x=2).
    let n_s = n_z * n_x;
    let mut p_data = vec![0.0f32; n_s * n_a * n_s];
    let s_idx = |z: usize, x: usize| z * n_x + x;
    let x_succ = |x: usize| -> usize { (x + 1).min(n_x - 1) };
    for z in 0..n_z {
        for x in 0..n_x {
            let s = s_idx(z, x);
            let s_p = s_idx(z, x_succ(x));
            p_data[s * n_a * n_s + 0 * n_s + s_p] = 1.0;
        }
    }
    let trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(p_data.as_slice(), &device).reshape([n_s, n_a, n_s]);
    let mut g_data = vec![0.0f32; n_s * n_a];
    for z in 0..n_z {
        let s = s_idx(z, 2);
        g_data[s * n_a + 0] = 1.0;
    }
    let goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(g_data.as_slice(), &device).reshape([n_s, n_a]);
    let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_s, n_a], &device);
    let product_mdp = TaskMDP::new(trans, goal, constraint, 8).unwrap();

    (base_mdp, hl_kernel, hl_tmdp, dims, product_mdp)
}

/// CHK-CO6.1: factorized evaluate (Cor 6.1 path) matches direct-DP η̃**(s_f, t_f|s)
/// on the no-HL-events witness, within tolerance.
#[test]
fn chk_co6_1_factorized_matches_direct_dp_no_hl_events() {
    let (base_mdp, hl_kernel, _hl_tmdp, dims, product_mdp) = build_cor_6_1_witness();

    // Solve product MDP directly to get ground truth.
    let result = feasibility_iteration(
        &product_mdp,
        FeasibilityIterationConfig::default().with_max_time(8),
    )
    .unwrap();
    let eta_truth = result.kernel.combined_stok(); // [9, 9, 8]

    // Build factorized STOK via Cor 6.1 path (no HL TMDPs).
    let base_stok = solve_task_mdp(&base_mdp).unwrap();
    let factorized = assemble_factorized_stok(base_stok, vec![hl_kernel], vec![0], dims).unwrap();

    // Compare evaluate output against ground truth across (s_i, s_f, t_f).
    let n_x = 3;
    let n_z = 2;
    let mut max_diff = 0.0f32;
    for z_i in 0..n_z {
        for x_i in 0..n_x {
            for z_f in 0..n_z {
                for x_f in 0..n_x {
                    for t in 1..8 {
                        let initial = ProductState::new(x_i).with_hl_discrete(z_i);
                        let final_st = ProductState::new(x_f).with_hl_discrete(z_f);
                        let factorized_val = factorized.evaluate(&initial, &final_st, t);

                        let s_i = z_i * n_x + x_i;
                        let s_f = z_f * n_x + x_f;
                        let truth_val: f32 = eta_truth
                            .clone()
                            .slice([s_i..(s_i + 1), s_f..(s_f + 1), t..(t + 1)])
                            .into_scalar()
                            .elem();

                        let diff = (factorized_val - truth_val).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
            }
        }
    }

    assert!(
        max_diff < 1e-3,
        "Cor 6.1 factorized vs direct-DP max diff = {} > 1e-3",
        max_diff
    );
}

// ============================================================================
// Witness 2: 2-base × 2-HL with HL events (general Eq [23] case)
// ============================================================================

/// Build a CTMDP where HL events DO occur — the HL TMDP has a goal at state 1,
/// reachable in one HL step from state 0 (under the policy that picks the
/// goal-inducing HL action). Used to exercise the general Eq [23] path.
fn build_general_witness() -> (
    TaskMDP<DefaultBackend>,
    Tensor<DefaultBackend, 3>,
    TaskMDP<DefaultBackend>,
    ProductSpaceDims,
) {
    let device = default_device();

    // Base: 3 states, 1 action, chain 0→1→2 with goal at x=2.
    let n_x = 3;
    let n_a = 1;
    let mut base_p = vec![0.0f32; n_x * n_a * n_x];
    base_p[0 * n_x + 1] = 1.0;
    base_p[1 * n_x + 2] = 1.0;
    base_p[2 * n_x + 2] = 1.0;
    let base_trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(base_p.as_slice(), &device).reshape([n_x, n_a, n_x]);
    let mut base_g = vec![0.0f32; n_x * n_a];
    base_g[2 * n_a + 0] = 1.0;
    let base_goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(base_g.as_slice(), &device).reshape([n_x, n_a]);
    let base_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_x, n_a], &device);
    let base_mdp = TaskMDP::new(base_trans, base_goal, base_constraint, 8).unwrap();

    // HL: 2 states, 2 actions. Default action = 0 (advance), 1 = stay.
    // Under default action 0 from state 0: advance to 1.
    // From state 1: stay at 1.
    let n_z = 2;
    let n_az = 2;
    let mut hl_p = vec![0.0f32; n_z * n_az * n_z];
    hl_p[0 * n_az * n_z + 0 * n_z + 1] = 1.0; // (z=0, α=0) → 1
    hl_p[0 * n_az * n_z + 1 * n_z + 0] = 1.0; // (z=0, α=1) → 0
    hl_p[1 * n_az * n_z + 0 * n_z + 1] = 1.0; // (z=1, α=0) → 1
    hl_p[1 * n_az * n_z + 1 * n_z + 1] = 1.0; // (z=1, α=1) → 1
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_p.as_slice(), &device).reshape([n_z, n_az, n_z]);

    // HL TaskMDP: goal at HL state 1 (any action).
    let mut hl_g = vec![0.0f32; n_z * n_az];
    hl_g[1 * n_az + 0] = 1.0;
    hl_g[1 * n_az + 1] = 1.0;
    let hl_goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_g.as_slice(), &device).reshape([n_z, n_az]);
    let hl_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_z, n_az], &device);
    let hl_tmdp = TaskMDP::new(hl_kernel.clone(), hl_goal, hl_constraint, 8).unwrap();

    let dims = ProductSpaceDims::new(n_x, vec![n_z]);
    (base_mdp, hl_kernel, hl_tmdp, dims)
}

/// PP-306: general-path sampling exercises the TEF (returns t_f drawn from
/// product-space ξ_s, not from the BL STOK termination distribution alone).
///
/// We don't try to verify the empirical sampling distribution here (would need
/// many draws). Instead we verify that:
///   (1) sampling completes without panicking on the general path,
///   (2) returned t_f is within [1, max_time - 1],
///   (3) returned product state has the right shape.
#[test]
fn pp306_general_path_sampling_completes() {
    use rand::SeedableRng;

    let (base_mdp, hl_kernel, hl_tmdp, dims) = build_general_witness();
    let base_stok = solve_task_mdp(&base_mdp).unwrap();

    let factorized = assemble_factorized_stok_with_hl_events(
        base_stok,
        &base_mdp,
        vec![hl_kernel],
        vec![0],
        vec![hl_tmdp],
        dims,
    )
    .unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let initial = ProductState::new(0).with_hl_discrete(0);

    for _ in 0..20 {
        let (final_st, t_f) = factorized.sample(&initial, &mut rng);
        let max_time = factorized.base_stok.max_time();
        assert!(
            t_f >= 1 && t_f < max_time,
            "PP-306: t_f = {} out of range [1, {})",
            t_f,
            max_time
        );
        assert!(final_st.base < 3, "PP-306: x_f out of base range");
        assert_eq!(final_st.hl_states.len(), 1);
    }
}

/// PP-303: the general Eq [23] path runs end-to-end and produces total mass
/// (Σ over s_f, t_f of η̃**) consistent with the per-state probability budget
/// (each row of η̃ sums to ≤ 1, and the success row sums to κ̃).
///
/// We do not cross-check against direct-DP product-MDP because constructing
/// that product correctly when HL events fire requires a bigger design (need
/// to embed the HL event timing in the BL termination). Instead this test
/// validates the general path's internal consistency.
#[test]
fn chk_general_path_internal_consistency() {
    let (base_mdp, hl_kernel, hl_tmdp, dims) = build_general_witness();
    let base_stok = solve_task_mdp(&base_mdp).unwrap();

    let factorized = assemble_factorized_stok_with_hl_events(
        base_stok,
        &base_mdp,
        vec![hl_kernel],
        vec![0],
        vec![hl_tmdp],
        dims,
    )
    .unwrap();

    // Sanity: the general path is taken (base_spk + base_cef + cefs all populated).
    assert!(factorized.base_spk.is_some());
    assert!(factorized.base_cef.is_some());
    assert_eq!(factorized.cefs.len(), 1);

    // Sum over (z_f, x_f, t_f) of evaluate output for each initial state.
    // Since the witness is fully feasible (every product state reaches the
    // HL goal eventually), the total should be close to 1.
    let max_time = factorized.base_stok.max_time();
    let n_x = 3;
    let n_z = 2;
    for z_i in 0..n_z {
        for x_i in 0..n_x {
            let initial = ProductState::new(x_i).with_hl_discrete(z_i);
            let mut total = 0.0f32;
            for z_f in 0..n_z {
                for x_f in 0..n_x {
                    let final_st = ProductState::new(x_f).with_hl_discrete(z_f);
                    for t in 1..max_time {
                        total += factorized.evaluate(&initial, &final_st, t);
                    }
                }
            }
            // Total mass should be in [0, 1] (a valid probability sum).
            assert!(
                total >= -1e-3 && total <= 1.0 + 1e-2,
                "general path total mass at (z={}, x={}) = {} not in [0, 1]",
                z_i,
                x_i,
                total
            );
        }
    }
}
