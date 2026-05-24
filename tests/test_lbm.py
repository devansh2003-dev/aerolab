"""Invariants for the D2Q9 lattice constants.

These are physics-derived invariants -- breaking any of them means the solver
won't recover Navier-Stokes correctly, regardless of how clean the Python looks.
"""
import sys
from pathlib import Path

# Add project root to sys.path so `from src.lbm import ...` works without an
# editable install. Same pattern we use in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.lbm import (
    CS2,
    LATTICE_VELOCITIES,
    LATTICE_WEIGHTS,
    OPPOSITE,
    bounce_back,
    collide,
    equilibrium,
    macroscopic,
    step_njit_mrt_no_force,
    step_njit_mrt_with_force,
    step_njit_with_force,
    stream,
    zou_he_inflow,
    zou_he_outflow_pressure,
)


def test_q9_shape():
    """D2Q9 has exactly 9 velocity directions, each 2D."""
    assert LATTICE_VELOCITIES.shape == (9, 2)
    assert LATTICE_WEIGHTS.shape == (9,)
    assert OPPOSITE.shape == (9,)


def test_weights_sum_to_one():
    """Lattice weights normalize to 1.0 (mass/probability conservation)."""
    assert np.isclose(LATTICE_WEIGHTS.sum(), 1.0)


def test_weight_values():
    """Standard D2Q9: 4/9 rest, 1/9 cardinals, 1/36 diagonals."""
    assert np.isclose(LATTICE_WEIGHTS[0], 4 / 9)
    assert np.allclose(LATTICE_WEIGHTS[1:5], 1 / 9)
    assert np.allclose(LATTICE_WEIGHTS[5:9], 1 / 36)


def test_velocity_set_is_centrally_symmetric():
    """For every direction i, -c_i is also in the set, and OPPOSITE points to it.
    This is what makes bounce-back possible: a wall can always reflect any
    incoming direction onto a discrete outgoing one."""
    for i, c in enumerate(LATTICE_VELOCITIES):
        opp = OPPOSITE[i]
        assert np.array_equal(LATTICE_VELOCITIES[opp], -c), (
            f"OPPOSITE[{i}]={opp} but c[{opp}]={LATTICE_VELOCITIES[opp]} "
            f"is not the negation of c[{i}]={c}"
        )


def test_speed_of_sound_squared():
    """c_s^2 = sum_i (w_i * cx_i^2) = 1/3. This is the lattice's built-in
    isothermal speed of sound, used in the equilibrium and equation of state."""
    cs2_computed_x = float(np.sum(LATTICE_WEIGHTS * LATTICE_VELOCITIES[:, 0] ** 2))
    cs2_computed_y = float(np.sum(LATTICE_WEIGHTS * LATTICE_VELOCITIES[:, 1] ** 2))
    assert np.isclose(cs2_computed_x, CS2)
    assert np.isclose(cs2_computed_y, CS2)  # symmetry: x and y must agree
    assert np.isclose(CS2, 1 / 3)


def test_first_moment_of_weights_is_zero():
    """sum_i (w_i * c_i) must equal zero. Otherwise the lattice has a built-in
    drift even at rest, and the equilibrium f^eq won't recover zero velocity."""
    moment = np.sum(LATTICE_WEIGHTS[:, None] * LATTICE_VELOCITIES, axis=0)
    assert np.allclose(moment, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Equilibrium distribution (Step 2b)
# ---------------------------------------------------------------------------

def test_equilibrium_at_rest_returns_weights_times_density():
    """f_eq(rho, u=0) = w_i * rho. With zero velocity all the u-dependent terms
    vanish and only the leading w_i * rho remains."""
    rho = 1.5
    u = np.zeros(2)
    feq = equilibrium(rho, u)
    assert feq.shape == (9,)
    assert np.allclose(feq, LATTICE_WEIGHTS * rho)


def test_equilibrium_point_recovers_density():
    """sum_i f_eq_i = rho for any (rho, u). Mass is conserved exactly by the
    Hermite expansion regardless of velocity."""
    rho = 1.2
    u = np.array([0.05, -0.03])  # |u| ~ 0.06 << cs ~ 0.577, well within stable LBM regime
    feq = equilibrium(rho, u)
    assert np.isclose(feq.sum(), rho)


def test_equilibrium_point_recovers_momentum():
    """sum_i f_eq_i * c_i = rho * u for any (rho, u). Momentum conservation,
    also exact at second order."""
    rho = 1.2
    u = np.array([0.08, 0.04])
    feq = equilibrium(rho, u)
    mom_x = float(np.sum(feq * LATTICE_VELOCITIES[:, 0]))
    mom_y = float(np.sum(feq * LATTICE_VELOCITIES[:, 1]))
    assert np.isclose(mom_x, rho * u[0])
    assert np.isclose(mom_y, rho * u[1])


def test_equilibrium_field_shape():
    """Field case returns shape (9, Nx, Ny)."""
    Nx, Ny = 8, 6
    rho = np.ones((Nx, Ny))
    u = np.zeros((2, Nx, Ny))
    feq = equilibrium(rho, u)
    assert feq.shape == (9, Nx, Ny)


def test_equilibrium_field_mass_conservation():
    """sum over directions equals rho at every grid point. Same invariant as
    the point case but verified across a non-uniform field."""
    rng = np.random.default_rng(seed=0)
    Nx, Ny = 8, 6
    rho = rng.uniform(0.8, 1.3, size=(Nx, Ny))
    u = rng.normal(0, 0.05, size=(2, Nx, Ny))
    feq = equilibrium(rho, u)
    assert np.allclose(feq.sum(axis=0), rho)


def test_equilibrium_field_momentum_conservation():
    """sum over directions of f_eq * c equals rho * u at every grid point."""
    rng = np.random.default_rng(seed=1)
    Nx, Ny = 8, 6
    rho = np.ones((Nx, Ny))
    u = rng.normal(0, 0.05, size=(2, Nx, Ny))
    feq = equilibrium(rho, u)
    mom_x = np.sum(feq * LATTICE_VELOCITIES[:, 0, None, None], axis=0)
    mom_y = np.sum(feq * LATTICE_VELOCITIES[:, 1, None, None], axis=0)
    assert np.allclose(mom_x, rho * u[0])
    assert np.allclose(mom_y, rho * u[1])


# ---------------------------------------------------------------------------
# Macroscopic moments + BGK collision (Step 2c)
# ---------------------------------------------------------------------------

def test_macroscopic_round_trip_point():
    """macroscopic(equilibrium(rho, u)) returns (rho, u). Sanity-checks that
    moments and equilibrium agree -- if either is wrong, this fails."""
    rho = 1.2
    u = np.array([0.05, -0.03])
    feq = equilibrium(rho, u)
    rho_back, u_back = macroscopic(feq)
    assert np.isclose(rho_back, rho)
    assert np.allclose(u_back, u)


def test_macroscopic_round_trip_field():
    """Field version of the round-trip across a non-uniform grid."""
    rng = np.random.default_rng(seed=2)
    Nx, Ny = 8, 6
    rho = rng.uniform(0.8, 1.3, size=(Nx, Ny))
    u = rng.normal(0, 0.05, size=(2, Nx, Ny))
    feq = equilibrium(rho, u)
    rho_back, u_back = macroscopic(feq)
    assert np.allclose(rho_back, rho)
    assert np.allclose(u_back, u)


def test_collide_equilibrium_is_fixed_point():
    """If f is already f_eq, collision returns f unchanged. The "no perturbation
    to relax" boundary case."""
    rng = np.random.default_rng(seed=3)
    Nx, Ny = 8, 6
    rho = rng.uniform(0.8, 1.3, size=(Nx, Ny))
    u = rng.normal(0, 0.05, size=(2, Nx, Ny))
    feq = equilibrium(rho, u)
    f_post = collide(feq, tau=1.0)
    assert np.allclose(f_post, feq)


def test_collide_preserves_mass():
    """rho is invariant under collision at every cell -- the cornerstone of
    the BGK construction."""
    rng = np.random.default_rng(seed=4)
    Nx, Ny = 8, 6
    f = rng.uniform(0.05, 0.15, size=(9, Nx, Ny))
    f_post = collide(f, tau=0.8)
    rho_pre, _ = macroscopic(f)
    rho_post, _ = macroscopic(f_post)
    assert np.allclose(rho_post, rho_pre)


def test_collide_preserves_momentum():
    """rho * u is invariant under collision at every cell."""
    rng = np.random.default_rng(seed=5)
    Nx, Ny = 8, 6
    f = rng.uniform(0.05, 0.15, size=(9, Nx, Ny))
    f_post = collide(f, tau=0.8)
    rho_pre, u_pre = macroscopic(f)
    rho_post, u_post = macroscopic(f_post)
    assert np.allclose(rho_post * u_post, rho_pre * u_pre)


def test_collide_relaxation_rate_matches_one_over_tau():
    """One BGK step shrinks (f - f_eq) by exactly factor (1 - 1/tau).

    Construction: take an equilibrium f_eq, perturb a few directions by amounts
    that cancel in the mass and momentum moments. Then macroscopic(f) returns
    the same (rho, u), so equilibrium(rho, u) returns the same f_eq, and the
    deviation (f - f_eq) is purely "non-equilibrium" -- exactly what BGK
    relaxes."""
    Nx, Ny = 4, 4
    rho = np.ones((Nx, Ny))
    u = np.zeros((2, Nx, Ny))
    feq = equilibrium(rho, u)

    # Perturbation that conserves both mass and momentum:
    # +d to direction 1 (east), +d to direction 3 (west), -2d to direction 0 (rest).
    # mass shift = 0, x-momentum shift = (+1 - 1)*d = 0, y-momentum shift = 0.
    f = feq.copy()
    d = 0.01
    f[1] += d
    f[3] += d
    f[0] -= 2 * d

    # Sanity: f and feq share the same macroscopic moments.
    rho_check, u_check = macroscopic(f)
    assert np.allclose(rho_check, 1.0)
    assert np.allclose(u_check, 0.0)

    tau = 2.0
    f_post = collide(f, tau)
    deviation_pre = f - feq
    deviation_post = f_post - feq
    expected_post = deviation_pre * (1.0 - 1.0 / tau)
    assert np.allclose(deviation_post, expected_post)


# ---------------------------------------------------------------------------
# Streaming (Step 2d)
# ---------------------------------------------------------------------------

def test_stream_rest_direction_unchanged():
    """Direction 0 (rest, cx=cy=0) doesn't move under streaming."""
    f = np.zeros((9, 5, 5))
    f[0, 2, 2] = 1.0
    f_post = stream(f)
    assert np.array_equal(f_post[0], f[0])


def test_stream_east_pulse_moves_plus_x():
    """A unit pulse in direction 1 (east) at (2, 2) ends up at (3, 2)."""
    f = np.zeros((9, 5, 5))
    f[1, 2, 2] = 1.0
    f_post = stream(f)
    assert f_post[1, 3, 2] == 1.0
    assert f_post[1, 2, 2] == 0.0


def test_stream_wraps_periodically_at_edge():
    """A pulse moving east off the right edge wraps to the left edge.
    This is intentional: walls will be imposed by bounce_back, not by stream."""
    f = np.zeros((9, 5, 5))
    f[1, 4, 2] = 1.0  # east-moving at right boundary
    f_post = stream(f)
    assert f_post[1, 0, 2] == 1.0


def test_stream_conserves_total_mass():
    """Streaming rearranges; it doesn't create or destroy. Sum over all space
    and all directions is invariant."""
    rng = np.random.default_rng(seed=10)
    f = rng.uniform(0.05, 0.15, size=(9, 8, 6))
    f_post = stream(f)
    assert np.isclose(f.sum(), f_post.sum())


def test_stream_all_directions_land_correctly():
    """For each of the 9 directions, a unit pulse at the center moves to
    (center + c_i). Verifies the cx/cy lookup is consistent with np.roll's
    sign convention."""
    Nx, Ny = 7, 7
    cx_all = LATTICE_VELOCITIES[:, 0]
    cy_all = LATTICE_VELOCITIES[:, 1]
    cxc, cyc = 3, 3  # center
    for i in range(9):
        f = np.zeros((9, Nx, Ny))
        f[i, cxc, cyc] = 1.0
        f_post = stream(f)
        x_dst = (cxc + int(cx_all[i])) % Nx
        y_dst = (cyc + int(cy_all[i])) % Ny
        assert f_post[i, x_dst, y_dst] == 1.0, (
            f"Direction {i} (c={(cx_all[i], cy_all[i])}) didn't land at "
            f"({x_dst}, {y_dst})"
        )


# ---------------------------------------------------------------------------
# Bounce-back boundary (Step 2e)
# ---------------------------------------------------------------------------

def test_bounce_back_no_solid_cells_is_identity():
    """Empty mask: nothing to swap, f unchanged."""
    rng = np.random.default_rng(seed=20)
    f = rng.uniform(0.05, 0.15, size=(9, 5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    f_post = bounce_back(f, mask)
    assert np.array_equal(f_post, f)


def test_bounce_back_swaps_opposite_directions():
    """At a solid cell, the value in direction i ends up in direction
    OPPOSITE[i], and vice versa."""
    f = np.zeros((9, 5, 5))
    # Distinct values in two opposing pairs: (1, 3) east/west and (5, 7) NE/SW.
    f[1, 2, 2] = 0.70
    f[3, 2, 2] = 0.20
    f[5, 2, 2] = 0.10
    f[7, 2, 2] = 0.05

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    f_post = bounce_back(f, mask)

    # 1 <-> 3, 5 <-> 7 should be swapped.
    assert f_post[1, 2, 2] == 0.20
    assert f_post[3, 2, 2] == 0.70
    assert f_post[5, 2, 2] == 0.05
    assert f_post[7, 2, 2] == 0.10


def test_bounce_back_leaves_fluid_cells_untouched():
    """Cells outside the solid mask are not modified."""
    rng = np.random.default_rng(seed=21)
    f = rng.uniform(0.05, 0.15, size=(9, 5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    f_post = bounce_back(f, mask)

    # Compare everywhere except the solid cell.
    fluid_mask = ~mask
    assert np.array_equal(f_post[:, fluid_mask], f[:, fluid_mask])


def test_bounce_back_is_involutive():
    """Applying bounce-back twice returns the original (each pair swaps then
    unswaps). This catches off-by-one bugs in the OPPOSITE table."""
    rng = np.random.default_rng(seed=22)
    f = rng.uniform(0.05, 0.15, size=(9, 5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True  # 3x3 block of solids
    f_twice = bounce_back(bounce_back(f, mask), mask)
    assert np.allclose(f_twice, f)


def test_bounce_back_preserves_density_at_solid_cells():
    """Sum of populations at each solid cell is unchanged: bounce-back is a
    permutation of the 9 directional populations, not a sink/source."""
    rng = np.random.default_rng(seed=23)
    f = rng.uniform(0.05, 0.15, size=(9, 5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    rho_pre = f[:, 2, 2].sum()
    f_post = bounce_back(f, mask)
    rho_post = f_post[:, 2, 2].sum()
    assert np.isclose(rho_post, rho_pre)


# ---------------------------------------------------------------------------
# JIT'd fused step (Stage B): must produce identical output to the pure-NumPy
# reference path. This is the single most important test in the file -- if it
# fails, every downstream simulation will be wrong in subtle ways.
# ---------------------------------------------------------------------------

def test_step_njit_matches_pure_numpy_single_step():
    """One JIT'd step must match the pure-NumPy reference path to within
    ``atol=1e-10`` (NOT bit-identical -- ``fastmath=True`` lets the
    compiler reassociate FP ops, and parallel reductions don't preserve
    sum order). The 1e-10 tolerance is far below LBM discretization
    error, so the test catches algorithmic bugs while accepting
    round-off drift from fastmath.

    Uses a no-Bouzidi (all -1) q-field so the JIT step's Bouzidi block is a
    no-op, matching the pure-NumPy reference path which uses halfway BB.
    A separate Bouzidi-specific test verifies the JIT Bouzidi block."""
    from src.forces import momentum_exchange_force
    from src.shapes import cylinder_mask, no_bouzidi_q_field

    Nx, Ny = 60, 40
    tau = 0.6
    U = 0.05
    mask = cylinder_mask(Nx, Ny, cx=20, cy=20, radius=4)
    q_field = no_bouzidi_q_field(Nx, Ny)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    rng = np.random.default_rng(seed=999)
    u0[1] = rng.normal(0, 1e-3, size=(Nx, Ny))
    f_init = equilibrium(rho0, u0)

    f_inflow = equilibrium(1.0, np.array([U, 0.0]))
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)

    # --- Pure-NumPy reference: collide, force, bounce-back, stream, BCs ---
    f_pure = f_init.copy()
    f_post_coll_pure = collide(f_pure, tau)
    F_pure = momentum_exchange_force(f_post_coll_pure, mask)
    f_pure = bounce_back(f_post_coll_pure, mask)
    f_pure = stream(f_pure)
    zou_he_inflow(f_pure, ux_in=U, uy_in=0.0)
    zou_he_outflow_pressure(f_pure, rho_out=1.0)

    # --- JIT'd fused step ---
    f_jit, Fx_jit, Fy_jit = step_njit_with_force(
        f_init.copy(), tau, mask, q_field, f_inflow, inflow_dirs, outflow_dirs
    )

    # JIT step is compiled with fastmath=True + parallel reduction, which
    # relaxes FP associativity. Tolerance is loose enough to absorb that
    # round-off (well under 1e-10 per cell), tight enough to catch any real
    # algorithmic bug.
    max_diff = float(np.max(np.abs(f_jit - f_pure)))
    assert np.allclose(f_jit, f_pure, atol=1e-10), f"f differs from reference by {max_diff:g}"
    assert np.isclose(Fx_jit, F_pure[0], atol=1e-12), f"Fx: JIT={Fx_jit}, ref={F_pure[0]}"
    assert np.isclose(Fy_jit, F_pure[1], atol=1e-12), f"Fy: JIT={Fy_jit}, ref={F_pure[1]}"


def test_step_njit_matches_pure_numpy_after_ten_steps():
    """Same equivalence check, propagated for 10 steps. Catches accumulating
    drift that a single-step test would miss (e.g., subtle BC order bugs)."""
    from src.forces import momentum_exchange_force
    from src.shapes import cylinder_mask, no_bouzidi_q_field

    Nx, Ny = 50, 30
    tau = 0.6
    U = 0.04
    mask = cylinder_mask(Nx, Ny, cx=18, cy=15, radius=3)
    q_field = no_bouzidi_q_field(Nx, Ny)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    rng = np.random.default_rng(seed=777)
    u0[1] = rng.normal(0, 1e-3, size=(Nx, Ny))
    f_init = equilibrium(rho0, u0)

    f_inflow = equilibrium(1.0, np.array([U, 0.0]))
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)

    # Pure-NumPy: 10 steps
    f_pure = f_init.copy()
    for _ in range(10):
        f_post = collide(f_pure, tau)
        f_pure = bounce_back(f_post, mask)
        f_pure = stream(f_pure)
        zou_he_inflow(f_pure, ux_in=U, uy_in=0.0)
        zou_he_outflow_pressure(f_pure, rho_out=1.0)

    # JIT: 10 steps
    f_jit = f_init.copy()
    for _ in range(10):
        f_jit, _, _ = step_njit_with_force(
            f_jit, tau, mask, q_field, f_inflow, inflow_dirs, outflow_dirs
        )

    # 10 steps of fastmath round-off can accumulate; we still want the JIT
    # answer to agree with the reference to ~8 decimal places, far below the
    # discretization error of the LBM method itself.
    max_diff = float(np.max(np.abs(f_jit - f_pure)))
    assert np.allclose(f_jit, f_pure, atol=1e-8), f"f drift after 10 steps: {max_diff:g}"


# ---------------------------------------------------------------------------
# MRT JIT path: drives the production Streamlit Real CFD mode. These tests
# verify the two MRT variants agree, that the operator preserves the
# physical invariants (mass), and that uniform freestream stays stable.
# ---------------------------------------------------------------------------

def test_step_njit_mrt_no_force_matches_with_force():
    """The viz-path MRT (no force loop) must produce identical f to the
    validation-path MRT (with momentum-exchange force loop). Both apply
    the same collision + Smagorinsky LES + bounce-back + stream + BCs;
    only difference is whether forces on the body are summed. If this
    test fails, the two paths have desynced and validation numbers won't
    match what the Streamlit app shows."""
    from src.shapes import cylinder_mask, no_bouzidi_q_field

    Nx, Ny = 60, 40
    tau = 0.6
    U = 0.05
    mask = cylinder_mask(Nx, Ny, cx=20, cy=20, radius=4)
    q_field = no_bouzidi_q_field(Nx, Ny)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    rng = np.random.default_rng(seed=42)
    u0[1] = rng.normal(0, 1e-3, size=(Nx, Ny))
    f_init = equilibrium(rho0, u0)

    f_inflow = equilibrium(1.0, np.array([U, 0.0]))
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)

    f_with, _Fx, _Fy = step_njit_mrt_with_force(
        f_init.copy(), tau, mask, q_field, f_inflow, inflow_dirs, outflow_dirs,
    )
    f_no = step_njit_mrt_no_force(
        f_init.copy(), tau, mask, q_field, f_inflow, inflow_dirs, outflow_dirs,
    )

    # Both functions are JIT-compiled with fastmath + parallel, so identical
    # arithmetic doesn't always produce bit-identical output (parallel
    # reduction order can differ). 1e-12 covers that comfortably -- if real
    # algorithmic drift sneaks in it'll be many orders of magnitude larger.
    max_diff = float(np.max(np.abs(f_with - f_no)))
    assert np.allclose(f_with, f_no, atol=1e-12), (
        f"MRT no_force diverged from with_force: max |Δf| = {max_diff:g}"
    )


def test_step_njit_mrt_conserves_mass_closed_box():
    """One MRT step in a closed (no inflow / no outflow) box preserves total
    mass to FP precision. MRT's moment-basis construction makes m0=rho a
    conserved moment by design; bounce-back and streaming just rearrange
    populations. If this fails the MRT relaxation rates are leaking mass."""
    from src.shapes import no_bouzidi_q_field

    Nx, Ny = 40, 30
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[15:20, 12:18] = True  # arbitrary small solid block
    q_field = no_bouzidi_q_field(Nx, Ny)

    rng = np.random.default_rng(seed=100)
    rho = rng.uniform(0.95, 1.05, size=(Nx, Ny))
    u = rng.normal(0, 0.02, size=(2, Nx, Ny))
    f = equilibrium(rho, u)

    # Empty inflow/outflow direction arrays = closed box (top/bottom walls
    # bounce back, left/right walls fall back to streaming's periodic).
    no_dirs = np.empty(0, dtype=np.int32)
    f_inflow_dummy = equilibrium(1.0, np.array([0.0, 0.0]))

    mass_pre = float(f.sum())
    f_after = step_njit_mrt_no_force(
        f, 0.6, mask, q_field, f_inflow_dummy, no_dirs, no_dirs,
    )
    mass_post = float(f_after.sum())

    rel_mass_drift = abs(mass_post - mass_pre) / mass_pre
    assert rel_mass_drift < 1e-10, (
        f"MRT lost mass: {mass_pre:.6e} -> {mass_post:.6e} "
        f"(relative drift {rel_mass_drift:.3e})"
    )


def test_step_njit_mrt_uniform_freestream_stays_uniform():
    """50 MRT steps of uniform inflow over an empty domain must leave the
    bulk velocity ~ U everywhere (except a thin BL near the top/bottom
    walls). This catches numerical drift, spurious vorticity injection at
    the boundaries, or instability in the Smagorinsky correction at zero
    strain."""
    from src.shapes import no_bouzidi_q_field

    Nx, Ny = 60, 30
    U = 0.05
    mask = np.zeros((Nx, Ny), dtype=bool)
    q_field = no_bouzidi_q_field(Nx, Ny)

    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    f = equilibrium(rho0, u0)

    f_inflow = equilibrium(1.0, np.array([U, 0.0]))
    inflow_dirs = np.array([1, 5, 8], dtype=np.int32)
    outflow_dirs = np.array([3, 6, 7], dtype=np.int32)

    for _ in range(50):
        f = step_njit_mrt_no_force(
            f, 0.7, mask, q_field, f_inflow, inflow_dirs, outflow_dirs,
        )

    _, u_final = macroscopic(f)

    # Bulk window stays clear of THREE adjustment regions:
    #   * Zou-He inflow adjustment (the imposed velocity creates a small
    #     density bump near x=0 that propagates ~10-20 cells inward at 50
    #     steps and decays over hundreds of steps -- a well-documented LBM
    #     boundary feature, not a bug. Marginally larger than the previous
    #     equilibrium inflow's bump, but mass-conserving in exchange)
    #   * zero-gradient-outflow adjustment (~5 cells from x=Nx-1)
    #   * top/bottom bounce-back boundary layers (~1-2 cells at tau=0.7)
    #
    # Tolerance of 3e-3 (= 6% of U=0.05) admits the small steady-state
    # inflow bump while still catching real instabilities -- NaN/inf,
    # exponential growth, wrong-direction flow, or new spurious modes.
    # A divergent solver would blow this up by 10x+ within a handful of
    # steps; this threshold is ~30x below any real failure mode.
    bulk_ux = u_final[0, 25:Nx - 15, 8:Ny - 8]
    bulk_uy = u_final[1, 25:Nx - 15, 8:Ny - 8]

    assert np.abs(bulk_ux - U).max() < 3e-3, (
        f"Bulk u_x drifted from inflow value {U}: "
        f"max abs deviation = {np.abs(bulk_ux - U).max():.4e}"
    )
    assert np.abs(bulk_uy).max() < 3e-3, (
        f"Bulk u_y not zero: max abs = {np.abs(bulk_uy).max():.4e}"
    )
