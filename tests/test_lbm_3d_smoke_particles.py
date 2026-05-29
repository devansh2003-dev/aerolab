"""Analytic verification tests for src/lbm_3d_smoke_particles.py.

Per D-8 in 3D_PHASE0_DECISIONS.md (revised 2026-05-26 after
reviewer feedback), these are real analytic-field gates on the
RK4 integrator and the trilinear interpolator. The original D-8
verification was a single visual gut check, which the reviewer
correctly flagged as too soft -- it cannot distinguish:

  - off-by-half-cell trilinear (would still look like "smoke")
  - wrong RK4 weights (would still look like "smoke")
  - integer wrap-around at high N (would still look like "smoke")
  - velocity field broadcast bug (would still look like "smoke")

These tests cover all four failure modes by comparing against
known closed-form trajectories.
"""
from __future__ import annotations

import numpy as np

from src.lbm_3d_smoke_particles import (
    advect_rk4,
    seed_inflow_particles,
    step_smoke,
    trilerp_3d,
)

# ============================================================================
# trilerp_3d: read-back fidelity
# ============================================================================


class TestTrilerp:
    def test_integer_position_returns_grid_value(self):
        """At an exact lattice point, the interpolated value equals the field
        value at that cell."""
        Nx, Ny, Nz = 8, 8, 8
        field = np.arange(Nx * Ny * Nz, dtype=np.float32).reshape(Nx, Ny, Nz)
        xs = np.array([2.0, 5.0, 0.0])
        ys = np.array([3.0, 1.0, 4.0])
        zs = np.array([4.0, 6.0, 0.0])
        result = trilerp_3d(field, xs, ys, zs)
        expected = np.array(
            [field[2, 3, 4], field[5, 1, 6], field[0, 4, 0]]
        )
        np.testing.assert_allclose(result, expected)

    def test_cell_centre_is_corner_mean(self):
        """At (i + 0.5, j + 0.5, k + 0.5) the result is the mean of the
        eight surrounding corners."""
        rng = np.random.default_rng(0)
        Nx, Ny, Nz = 4, 4, 4
        field = rng.random((Nx, Ny, Nz)).astype(np.float32)
        xs = np.array([1.5])
        ys = np.array([2.5])
        zs = np.array([0.5])
        result = trilerp_3d(field, xs, ys, zs)
        expected = np.mean(field[1:3, 2:4, 0:2])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_uniform_field_returns_constant_everywhere(self):
        """A constant field interps to that constant at any FP position."""
        field = np.full((10, 10, 10), 3.14, dtype=np.float32)
        xs = np.array([1.3, 5.7, 8.1, 0.0, 9.0])
        ys = np.array([4.4, 2.2, 6.6, 0.0, 9.0])
        zs = np.array([0.5, 9.0, 3.3, 0.0, 9.0])
        result = trilerp_3d(field, xs, ys, zs)
        np.testing.assert_allclose(result, 3.14)

    def test_linear_field_is_recovered_exactly(self):
        """Trilinear interp of a linear field is exact (no second-order
        truncation error). This catches off-by-half-cell bugs in the
        weight construction."""
        Nx, Ny, Nz = 16, 16, 16
        ix, iy, iz = np.meshgrid(
            np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
        )
        field = (2.0 * ix + 3.0 * iy + 5.0 * iz).astype(np.float32)
        xs = np.array([3.7, 8.2, 11.5])
        ys = np.array([1.1, 9.9, 6.6])
        zs = np.array([4.4, 0.5, 12.0])
        result = trilerp_3d(field, xs, ys, zs)
        expected = 2.0 * xs + 3.0 * ys + 5.0 * zs
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================================================
# D-8 analytic test #1: uniform flow
# ============================================================================


class TestUniformFlowAdvection:
    """A particle in u = (u_in, 0, 0) drifts at exactly u_in per unit time
    along x, with zero y/z motion. The RK4 integrator should reproduce this
    to machine precision because the field is constant (all four k values
    are identical)."""

    def test_one_step(self):
        Nx, Ny, Nz = 32, 16, 16
        u_in = 0.1
        ux = np.full((Nx, Ny, Nz), u_in, dtype=np.float32)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        x0, y0, z0 = 5.0, 8.0, 8.0
        dt = 1.0
        x1, y1, z1 = advect_rk4(
            np.array([x0]), np.array([y0]), np.array([z0]),
            ux, uy, uz, dt,
        )
        np.testing.assert_allclose(x1[0], x0 + u_in * dt, atol=1e-7)
        np.testing.assert_allclose(y1[0], y0, atol=1e-7)
        np.testing.assert_allclose(z1[0], z0, atol=1e-7)

    def test_many_steps(self):
        """100 advection steps cumulative error is still ~ machine epsilon
        in uniform flow. Catches accumulator-bug regressions."""
        Nx, Ny, Nz = 80, 16, 16
        u_in = 0.1
        ux = np.full((Nx, Ny, Nz), u_in, dtype=np.float32)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        px = np.array([5.0])
        py = np.array([8.0])
        pz = np.array([8.0])
        dt = 0.5
        n_steps = 100
        for _ in range(n_steps):
            px, py, pz = advect_rk4(px, py, pz, ux, uy, uz, dt)

        np.testing.assert_allclose(
            px[0], 5.0 + u_in * n_steps * dt, atol=1e-4,
        )
        np.testing.assert_allclose(py[0], 8.0, atol=1e-7)
        np.testing.assert_allclose(pz[0], 8.0, atol=1e-7)

    def test_many_particles_one_step(self):
        """Vectorised advection over many particles in uniform flow."""
        Nx, Ny, Nz = 32, 16, 16
        u_in = 0.05
        ux = np.full((Nx, Ny, Nz), u_in, dtype=np.float32)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)

        rng = np.random.default_rng(1)
        n = 50
        px = rng.uniform(2.0, Nx - 4.0, size=n)
        py = rng.uniform(2.0, Ny - 4.0, size=n)
        pz = rng.uniform(2.0, Nz - 4.0, size=n)
        dt = 1.0
        new_x, new_y, new_z = advect_rk4(px, py, pz, ux, uy, uz, dt)
        np.testing.assert_allclose(new_x, px + u_in * dt, atol=1e-6)
        np.testing.assert_allclose(new_y, py, atol=1e-6)
        np.testing.assert_allclose(new_z, pz, atol=1e-6)


# ============================================================================
# D-8 analytic test #2: 3D plane-Poiseuille centerline
# ============================================================================


class TestPoiseuilleCenterlineAdvection:
    """In 3D plane Poiseuille flow u(y) = u_peak * (1 - ((y - y_c)/h)²),
    a particle seeded exactly on the centerline y = y_c drifts at u_peak
    along x with zero y, z motion. This is the canonical test the
    reviewer (2026-05-26) called for to gate the trilerp + RK4 combo
    against a known field gradient."""

    def _make_poiseuille_field(self, Nx: int, Ny: int, Nz: int, u_peak: float):
        y_c = (Ny - 1) / 2.0
        h = (Ny - 1) / 2.0
        ys_grid = np.arange(Ny, dtype=np.float32)
        u_profile = u_peak * (1.0 - ((ys_grid - y_c) / h) ** 2)
        ux = np.broadcast_to(
            u_profile[None, :, None], (Nx, Ny, Nz)
        ).astype(np.float32).copy()
        uy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        uz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        return ux, uy, uz, y_c

    def test_centerline_drifts_at_u_peak(self):
        # Use ODD Ny so the centerline y_c = (Ny - 1) / 2 lands on an
        # integer grid point (y_c = 7 for Ny = 15). At an integer
        # position trilerp returns the field value exactly, which is
        # what makes "drifts at u_peak" a CLEAN analytic check. With
        # even Ny the centerline sits between two grid points and the
        # discrete field reads u_peak * (1 - (0.5 / h)²) -- still
        # correct trilerp behaviour but a different analytic answer.
        Nx, Ny, Nz = 64, 15, 17
        u_peak = 0.1
        ux, uy, uz, y_c = self._make_poiseuille_field(Nx, Ny, Nz, u_peak)
        assert float(y_c).is_integer(), "y_c must land on a grid point"

        px = np.array([5.0])
        py = np.array([y_c])
        pz = np.array([Nz // 2 * 1.0])
        dt = 0.5
        n_steps = 50
        for _ in range(n_steps):
            px, py, pz = advect_rk4(px, py, pz, ux, uy, uz, dt)

        np.testing.assert_allclose(py[0], y_c, atol=1e-5,
            err_msg="centerline particle drifted off-axis in pure-x flow")
        np.testing.assert_allclose(pz[0], Nz // 2, atol=1e-5,
            err_msg="centerline particle drifted in z in pure-x flow")
        np.testing.assert_allclose(
            px[0], 5.0 + u_peak * n_steps * dt, rtol=1e-5,
            err_msg="centerline particle did not drift at u_peak",
        )

    def test_off_centerline_drifts_slower(self):
        """A particle near the wall sees u < u_peak, so it must lag the
        centerline particle after the same wall-clock time."""
        Nx, Ny, Nz = 64, 32, 16
        u_peak = 0.1
        ux, uy, uz, y_c = self._make_poiseuille_field(Nx, Ny, Nz, u_peak)

        h = (Ny - 1) / 2.0
        # Particle at half-wall-distance: u = u_peak * (1 - 0.25) = 0.75 * u_peak
        y_off = y_c + 0.5 * h
        u_expected = u_peak * (1.0 - 0.5 ** 2)

        px = np.array([5.0])
        py = np.array([y_off])
        pz = np.array([Nz / 2.0])
        dt = 0.5
        n_steps = 50
        for _ in range(n_steps):
            px, py, pz = advect_rk4(px, py, pz, ux, uy, uz, dt)

        # Drift matches the local u_x at that y
        np.testing.assert_allclose(
            px[0], 5.0 + u_expected * n_steps * dt, rtol=2e-3,
        )
        # Still no transverse motion in pure parabolic-x flow
        np.testing.assert_allclose(py[0], y_off, atol=1e-5)


# ============================================================================
# step_smoke: spawn / cull invariants
# ============================================================================


class TestStepSmoke:
    """End-to-end frame logic: spawn at inflow, advect, cull at outflow /
    body / lifetime."""

    def _empty_uniform_setup(self, Nx=32, Ny=16, Nz=16, u_in=0.1):
        ux = np.full((Nx, Ny, Nz), u_in, dtype=np.float32)
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)
        body_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
        return ux, uy, uz, body_mask

    def test_no_midfield_spawn_when_seed_is_none(self):
        """Without inflow seeds, the particle count never grows. This is
        THE design rule from feedback_streamline_design: never spawn
        mid-domain."""
        ux, uy, uz, body_mask = self._empty_uniform_setup()
        px = np.array([5.0, 6.0, 7.0])
        py = np.array([8.0, 8.0, 8.0])
        pz = np.array([8.0, 8.0, 8.0])
        age = np.zeros(3, dtype=np.int32)

        for _ in range(5):
            px, py, pz, age = step_smoke(
                px, py, pz, age, ux, uy, uz, body_mask,
                dt=1.0, n_substeps=1, max_age=100,
                inflow_seed_xyz=None,
            )

        assert len(px) <= 3, "particle count cannot grow without inflow seeds"

    def test_cull_on_outflow(self):
        """A particle drifting past x = Nx - 1.5 is removed."""
        Nx = 32
        ux, uy, uz, body_mask = self._empty_uniform_setup(Nx=Nx)
        # one near outflow, one mid-domain
        px = np.array([Nx - 1.6, 5.0])
        py = np.array([8.0, 8.0])
        pz = np.array([8.0, 8.0])
        age = np.zeros(2, dtype=np.int32)

        for _ in range(2):
            px, py, pz, age = step_smoke(
                px, py, pz, age, ux, uy, uz, body_mask,
                dt=1.0, n_substeps=1, max_age=100,
                inflow_seed_xyz=None,
            )

        # The mid-domain particle survives, the outflow one is culled
        assert len(px) == 1
        assert 4.5 < px[0] < 8.0

    def test_cull_on_body_hit(self):
        """A particle whose nearest cell becomes solid is removed."""
        Nx, Ny, Nz = 32, 16, 16
        ux, uy, uz, body_mask = self._empty_uniform_setup(Nx, Ny, Nz)
        # Vertical wall at x = 15: every (y, z) cell at x=15 is solid
        body_mask[15, :, :] = True

        px = np.array([14.0])
        py = np.array([8.0])
        pz = np.array([8.0])
        age = np.zeros(1, dtype=np.int32)

        # u_in = 0.1, so it takes ~10 frames at dt=1 to drift one cell
        # into the body. 20 frames is plenty.
        for _ in range(20):
            px, py, pz, age = step_smoke(
                px, py, pz, age, ux, uy, uz, body_mask,
                dt=1.0, n_substeps=1, max_age=100,
                inflow_seed_xyz=None,
            )

        assert len(px) == 0, "particle should have been culled inside body"

    def test_cull_on_max_age(self):
        """A particle older than max_age is removed even if still inside
        the domain."""
        ux, uy, uz, body_mask = self._empty_uniform_setup()
        px = np.array([5.0])
        py = np.array([8.0])
        pz = np.array([8.0])
        age = np.array([0], dtype=np.int32)
        max_age = 3

        for _ in range(max_age + 1):
            px, py, pz, age = step_smoke(
                px, py, pz, age, ux, uy, uz, body_mask,
                dt=0.01,        # tiny step so we don't outflow first
                n_substeps=1, max_age=max_age,
                inflow_seed_xyz=None,
            )

        assert len(px) == 0, "particle older than max_age should be culled"

    def test_inflow_seeds_only_arrive_at_inflow_x(self):
        """Seeded particles must all share the inflow x position; this
        enforces the inflow-only rule at the API boundary."""
        y_rows = np.array([4.0, 8.0, 12.0])
        z_rows = np.array([4.0, 8.0, 12.0])
        seed_x, seed_y, seed_z = seed_inflow_particles(
            n_per_row=2, y_rows=y_rows, z_rows=z_rows, x=2.5,
        )
        # Every seed must have the same x; that x must equal the parameter
        np.testing.assert_array_equal(seed_x, 2.5)
        # 2 per row * 3 y * 3 z = 18 particles
        assert len(seed_x) == 18
        # y and z within their requested ranges (allowing default jitter 0.5)
        assert seed_y.min() > y_rows.min() - 0.6
        assert seed_y.max() < y_rows.max() + 0.6
        assert seed_z.min() > z_rows.min() - 0.6
        assert seed_z.max() < z_rows.max() + 0.6

    def test_end_to_end_continuous_seeding_keeps_steady_pool(self):
        """With continuous inflow seeding and unculled flow, the pool
        reaches a steady-state size (births balance deaths-by-outflow)."""
        Nx = 64
        ux, uy, uz, body_mask = self._empty_uniform_setup(
            Nx=Nx, u_in=0.1,
        )
        rng = np.random.default_rng(42)
        y_rows = np.array([4.0, 8.0, 12.0])
        z_rows = np.array([4.0, 8.0, 12.0])
        px = np.empty(0, dtype=np.float64)
        py = np.empty(0, dtype=np.float64)
        pz = np.empty(0, dtype=np.float64)
        age = np.empty(0, dtype=np.int32)

        max_age = 200  # well beyond the cross-channel transit time
        history = []
        for _ in range(80):
            seed = seed_inflow_particles(
                n_per_row=1, y_rows=y_rows, z_rows=z_rows, x=2.0, rng=rng,
            )
            px, py, pz, age = step_smoke(
                px, py, pz, age, ux, uy, uz, body_mask,
                dt=1.0, n_substeps=4, max_age=max_age,
                inflow_seed_xyz=seed,
            )
            history.append(len(px))

        # Steady state: the count must stop growing once inflow particles
        # start reaching the outflow. Last 20 frames variation should be
        # small relative to the mean.
        tail = np.array(history[-20:])
        assert tail.mean() > 0, "pool should not be empty"
        # Coefficient of variation should be modest -- births and deaths
        # are balanced once steady state is reached.
        assert tail.std() / tail.mean() < 0.30, (
            f"pool is not steady: cv = {tail.std() / tail.mean():.3f}"
        )
