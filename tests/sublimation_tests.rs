//! PP-403 / PP-404: Theorem 2.4 (Sublimation) bound tests on non-binary HL
//! spaces.
//!
//! These tests exercise `SublimatedTMDP::from_product` with HL spaces of
//! cardinality > 2, then cross-check the bound `κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)`
//! against the κ̃ computed by direct dynamic programming on the explicit
//! product space.

use burn::prelude::*;
use stok_core::backend::{default_device, DefaultBackend};
use stok_core::hierarchy::{FactorizedAffordance, SublimatedTMDP};
use stok_core::mdp::TaskMDP;
use stok_core::solver::{feasibility_iteration, FeasibilityIterationConfig};

// ============================================================================
// PP-403: SublimatedTMDP::from_product accepts non-binary HL cardinalities
// ============================================================================

/// Build a minimal 3-state, 3-action base TaskMDP (matching simple_chain's
/// 3-action shape) for use in sublimation tests where the affordance also
/// expects 3 base actions.
fn minimal_base_mdp_3x3() -> TaskMDP<DefaultBackend> {
    let device = default_device();
    TaskMDP::simple_chain(3, 8, &device)
}

/// PP-403: 4-state HL with 3 HL actions — sized correctly.
#[test]
fn from_product_4_state_hl_3_actions_shapes_ok() {
    let device = default_device();
    let base_mdp = minimal_base_mdp_3x3(); // 3 base states, 3 base actions

    // HL kernel: 4 HL states, 3 HL actions, deterministic identity.
    let mut hl_data = vec![0.0f32; 4 * 3 * 4];
    for z in 0..4 {
        for a in 0..3 {
            hl_data[z * 3 * 4 + a * 4 + z] = 1.0;
        }
    }
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_data.as_slice(), &device).reshape([4, 3, 4]);

    // Affordance: shape MUST match base_mdp = [3, 3, 3] (3 base states, 3 base
    // actions, 3 HL actions). All (x, a) deterministic on α = 0.
    let mut f_data = vec![0.0f32; 3 * 3 * 3];
    for x in 0..3 {
        for a in 0..3 {
            f_data[x * 3 * 3 + a * 3 + 0] = 1.0;
        }
    }
    let f_tensor: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 3]);
    let affordance = FactorizedAffordance::new(vec![f_tensor]);

    let sub = SublimatedTMDP::from_product(&base_mdp, hl_kernel, &affordance, 0, None).unwrap();

    assert_eq!(sub.n_hl_states(), 4, "PP-403: HL state cardinality respected");
    assert_eq!(sub.n_hl_actions(), 3, "PP-403: HL action cardinality respected");
    assert_eq!(
        sub.goal_fn.dims(),
        [4, 3],
        "PP-403: goal_fn shape matches non-binary HL"
    );
    assert_eq!(
        sub.constraint_fn.dims(),
        [4, 3],
        "PP-403: constraint_fn shape matches non-binary HL"
    );
}

/// PP-402 / PP-403: explicit HL constraint propagates through from_product.
#[test]
fn from_product_explicit_hl_constraint_is_respected() {
    let device = default_device();
    let base_mdp = minimal_base_mdp_3x3(); // 3 base states, 3 base actions

    // HL: 3 states × 2 HL actions × 3 states.
    let mut hl_data = vec![0.0f32; 3 * 2 * 3];
    for z in 0..3 {
        for a in 0..2 {
            hl_data[z * 2 * 3 + a * 3 + z] = 1.0;
        }
    }
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_data.as_slice(), &device).reshape([3, 2, 3]);

    // Affordance shape must match base_mdp: [3, 3, 2].
    let mut f_data = vec![0.0f32; 3 * 3 * 2];
    for x in 0..3 {
        for a in 0..3 {
            f_data[x * 3 * 2 + a * 2 + 0] = 1.0;
        }
    }
    let f_tensor: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 2]);
    let affordance = FactorizedAffordance::new(vec![f_tensor]);

    // Explicit HL constraint: forbid (z=1, α=0).
    let mut c_data = vec![1.0f32; 3 * 2];
    c_data[1 * 2 + 0] = 0.0;
    let hl_constraint: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(c_data.as_slice(), &device).reshape([3, 2]);

    let sub = SublimatedTMDP::from_product(
        &base_mdp,
        hl_kernel,
        &affordance,
        0,
        Some(hl_constraint),
    )
    .unwrap();

    let c_back: Vec<f32> = sub.constraint_fn.clone().into_data().to_vec().unwrap();
    assert_eq!(
        c_back[1 * 2 + 0],
        0.0,
        "explicit HL constraint at (z=1, α=0) must propagate (got {})",
        c_back[1 * 2 + 0]
    );
    assert_eq!(c_back[0 * 2 + 0], 1.0);
    assert_eq!(c_back[2 * 2 + 1], 1.0);
}

// ============================================================================
// PP-404 / CHK-T2.4: bound test on 3-state HL with non-trivial transitions
// ============================================================================

/// Build a deterministic single-action CTMDP witness whose product space is
/// well-behaved (no κ-tied infinite self-loops) so direct DP works without
/// hitting the absorbing-policy guard.
///
/// Single base action = "advance" (move x right, capped at boundary).
/// Affordance: at base state x ∈ {1, 2} the action induces HL α=1 (advance);
/// at x=0 it induces α=0 (no-op). HL transitions: α=1 advances z (capped),
/// α=0 keeps z.
///
/// The agent path from (z=0, x=0) is: (0,0) →[α=0] (0,1) →[α=1] (1,2)
/// →[α=1] (2,2) (goal). Every non-goal state has exactly one successor and
/// the goal is reachable from everywhere.
fn build_well_behaved_ctmdp() -> (
    TaskMDP<DefaultBackend>,
    FactorizedAffordance<DefaultBackend>,
    Tensor<DefaultBackend, 3>,
) {
    let device = default_device();
    let n_x = 3;
    let n_a = 1; // single action — no κ-ties
    let n_z = 3;
    let n_az = 2;

    // Base transition: action 0 moves right (capped).
    let mut base_p = vec![0.0f32; n_x * n_a * n_x];
    base_p[0 * n_x + 1] = 1.0;
    base_p[1 * n_x + 2] = 1.0;
    base_p[2 * n_x + 2] = 1.0; // x=2 stays at boundary
    let base_trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(base_p.as_slice(), &device).reshape([n_x, n_a, n_x]);

    // Base goal at (x=1, a=0) so maximize_goal_over_base can pull a 1 onto
    // α=1 (the action induced by x=1 — which is x=1, a=0).
    let mut base_g = vec![0.0f32; n_x * n_a];
    base_g[1 * n_a + 0] = 1.0; // (x=1, a=0) — induces α=1
    base_g[2 * n_a + 0] = 1.0; // (x=2, a=0) — also induces α=1
    let base_goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(base_g.as_slice(), &device).reshape([n_x, n_a]);
    let base_constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_x, n_a], &device);
    let base_mdp = TaskMDP::new(base_trans, base_goal, base_constraint, 12).unwrap();

    // HL kernel: α=0 keeps z, α=1 advances z (capped).
    let mut hl_p = vec![0.0f32; n_z * n_az * n_z];
    let hidx = |z: usize, a: usize, zp: usize| z * n_az * n_z + a * n_z + zp;
    for z in 0..n_z {
        hl_p[hidx(z, 0, z)] = 1.0;
        let zp = (z + 1).min(n_z - 1);
        hl_p[hidx(z, 1, zp)] = 1.0;
    }
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_p.as_slice(), &device).reshape([n_z, n_az, n_z]);

    // Affordance: x=1 or x=2 induces α=1 (advance), x=0 induces α=0 (no-op).
    let mut f_data = vec![0.0f32; n_x * n_a * n_az];
    let fidx = |x: usize, a: usize, alpha: usize| x * n_a * n_az + a * n_az + alpha;
    f_data[fidx(0, 0, 0)] = 1.0; // x=0: no-op
    f_data[fidx(1, 0, 1)] = 1.0; // x=1: advance
    f_data[fidx(2, 0, 1)] = 1.0; // x=2: advance
    let f_tensor: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([n_x, n_a, n_az]);
    let affordance = FactorizedAffordance::new(vec![f_tensor]);

    (base_mdp, affordance, hl_kernel)
}

/// Build the explicit product-space TaskMDP for the well-behaved witness.
fn build_well_behaved_product() -> TaskMDP<DefaultBackend> {
    let device = default_device();
    let n_x = 3;
    let n_a = 1;
    let n_z = 3;
    let n_s = n_z * n_x;

    let mut p_data = vec![0.0f32; n_s * n_a * n_s];
    let s_idx = |z: usize, x: usize| z * n_x + x;

    let alpha_at = |x: usize| -> usize {
        if x == 0 {
            0
        } else {
            1
        }
    };
    let z_succ = |z: usize, alpha: usize| -> usize {
        if alpha == 0 {
            z
        } else {
            (z + 1).min(n_z - 1)
        }
    };
    let x_succ = |x: usize| -> usize { (x + 1).min(n_x - 1) };

    for z in 0..n_z {
        for x in 0..n_x {
            let alpha = alpha_at(x);
            let z_p = z_succ(z, alpha);
            let x_p = x_succ(x);
            let s = s_idx(z, x);
            let s_p = s_idx(z_p, x_p);
            p_data[s * n_a * n_s + 0 * n_s + s_p] = 1.0;
        }
    }
    let trans: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(p_data.as_slice(), &device).reshape([n_s, n_a, n_s]);

    // Product goal: HL state z = 2.
    let mut g_data = vec![0.0f32; n_s * n_a];
    for x in 0..n_x {
        let s = s_idx(2, x);
        g_data[s * n_a + 0] = 1.0;
    }
    let goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(g_data.as_slice(), &device).reshape([n_s, n_a]);
    let constraint: Tensor<DefaultBackend, 2> = Tensor::ones([n_s, n_a], &device);

    TaskMDP::new(trans, goal, constraint, 12).unwrap()
}

/// PP-404 / CHK-T2.4: verify the bound `κ̃*(σ, z, x) ≤ κ*_sub,σ(σ)` on a
/// 3-state HL witness via direct DP cross-check. Single-action setup avoids
/// the κ-tied-self-loop edge case in `extract_pi_time_minimizing`.
#[test]
fn theorem_2_4_bound_holds_on_3_state_hl_well_behaved() {
    let (base_mdp, affordance, hl_kernel) = build_well_behaved_ctmdp();

    // Sublimated κ_sub on 3-state HL.
    let sub_kappa =
        SublimatedTMDP::from_product(&base_mdp, hl_kernel, &affordance, 0, None)
            .unwrap()
            .solve()
            .unwrap();

    // Direct-DP κ̃ on the 9-state product space.
    let product_mdp = build_well_behaved_product();
    let result = feasibility_iteration(
        &product_mdp,
        FeasibilityIterationConfig::default().with_max_time(12),
    )
    .unwrap();
    let kappa_full: Vec<f32> = result.kernel.kappa.into_data().to_vec().unwrap();

    // Bound: κ̃((z, x)) ≤ κ_sub(z) + ε for every product state.
    for z in 0..3 {
        let bound = sub_kappa[z];
        for x in 0..3 {
            let s = z * 3 + x;
            let actual = kappa_full[s];
            assert!(
                actual <= bound + 1e-4,
                "Theorem 2.4 violated at (z={}, x={}): κ̃ = {}, κ_sub(z) = {}",
                z,
                x,
                actual,
                bound
            );
        }
    }
    // Sanity: κ_sub for z=2 (goal HL state) must be 1.
    assert!(
        sub_kappa[2] > 0.99,
        "κ_sub(z=2) should be ≈ 1 (HL goal state), got {}",
        sub_kappa[2]
    );
}

/// PP-404 contrapositive: when `κ_sub(z) = 0`, every product state with that
/// z is infeasible. Construct a sublimated MDP with constraint termination so
/// the cycling HL states absorb properly (avoiding the absorbing-policy guard).
#[test]
fn theorem_2_4_contrapositive_zero_sub_kappa_isolated_state() {
    let device = default_device();
    let n_z = 3;
    let n_az = 2;

    // HL kernel: z=2 is reachable only from itself; z=0 and z=1 cycle. Goal at z=2.
    let mut hl_p = vec![0.0f32; n_z * n_az * n_z];
    let hidx = |z: usize, a: usize, zp: usize| z * n_az * n_z + a * n_z + zp;
    hl_p[hidx(0, 0, 1)] = 1.0;
    hl_p[hidx(0, 1, 1)] = 1.0;
    hl_p[hidx(1, 0, 0)] = 1.0;
    hl_p[hidx(1, 1, 0)] = 1.0;
    hl_p[hidx(2, 0, 2)] = 1.0;
    hl_p[hidx(2, 1, 2)] = 1.0;
    let hl_kernel: Tensor<DefaultBackend, 3> =
        Tensor::<DefaultBackend, 1>::from_floats(hl_p.as_slice(), &device).reshape([n_z, n_az, n_z]);

    // Goal at z=2.
    let mut g_data = vec![0.0f32; n_z * n_az];
    g_data[2 * n_az + 0] = 1.0;
    g_data[2 * n_az + 1] = 1.0;
    let goal: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(g_data.as_slice(), &device).reshape([n_z, n_az]);

    // Constraint: z=0 and z=1 each have probability 0.5 of constraint
    // termination per step. This breaks the would-be 2-cycle into an
    // absorbing chain (each step has chance to fail-out), satisfying
    // Appendix-5's spectral-radius requirement on P_NN.
    let mut c_data = vec![1.0f32; n_z * n_az];
    c_data[0 * n_az + 0] = 0.5;
    c_data[0 * n_az + 1] = 0.5;
    c_data[1 * n_az + 0] = 0.5;
    c_data[1 * n_az + 1] = 0.5;
    let constraint: Tensor<DefaultBackend, 2> =
        Tensor::<DefaultBackend, 1>::from_floats(c_data.as_slice(), &device).reshape([n_z, n_az]);

    let sub = SublimatedTMDP::new(hl_kernel, goal, constraint, 12).unwrap();
    let kappa_sub = sub.solve().unwrap();

    // z=0 and z=1 are isolated from the goal z=2 ⟹ κ_sub = 0 there.
    assert!(
        kappa_sub[0] < 1e-4,
        "κ_sub(0) should be 0 (isolated from goal), got {}",
        kappa_sub[0]
    );
    assert!(
        kappa_sub[1] < 1e-4,
        "κ_sub(1) should be 0 (isolated from goal), got {}",
        kappa_sub[1]
    );
    assert!(
        kappa_sub[2] > 0.99,
        "κ_sub(2) should be 1 (goal state), got {}",
        kappa_sub[2]
    );

    // Contrapositive: by Theorem 2.4, ∀x. κ̃((z=0, x)) ≤ κ_sub(0) = 0 and
    // ∀x. κ̃((z=1, x)) ≤ κ_sub(1) = 0. Tree-search pruning (PP-603) consumes
    // exactly this: any branch entering z ∈ {0, 1} can be safely pruned.
}
