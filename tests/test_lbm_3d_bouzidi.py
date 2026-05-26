"""Tests for src/lbm_3d_bouzidi.py — analytic Bouzidi q-field generator.

Per D-4 in 3D_PHASE0_DECISIONS.md the q-field is derived from the
quadratic |d + q c_i|² = R². These tests pin the quadratic solver
against hand-computed cases AND the sphere wall-link generator
against analytic invariants (cell-count scaling, q-bounds, mirror
symmetry).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.lbm_3d_bouzidi import (
    WallLinkList,
    make_sphere_mask,
    solve_bouzidi_q,
    sphere_wall_links,
)


# ============================================================================
# solve_bouzidi_q: closed-form quadratic checks
# ============================================================================


class TestSolveBouzidiQ:
    def test_rest_direction_returns_invalid(self):
        """The rest vector c = (0, 0, 0) has |c| = 0, so the quadratic
        is degenerate. Must return -1 (sentinel for "not a wall link")."""
        q = solve_bouzidi_q((-5.0, 0.0, 0.0), (0, 0, 0), R=4.5)
        assert q == -1.0

    def test_axis_direction_R_integer_gives_q_one(self):
        """R = 4 (integer), fluid cell at offset (-5, 0, 0) from centre,
        direction +x. The wall is at x_f + 1 * c = (-4, 0, 0), exactly
        on the sphere surface. q = 1."""
        q = solve_bouzidi_q((-5.0, 0.0, 0.0), (1, 0, 0), R=4.0)
        assert q == pytest.approx(1.0)

    def test_axis_direction_R_half_gives_q_half(self):
        """R = 4.5 (between two grid points), fluid cell at offset
        (-5, 0, 0) along +x. The wall is at (-4.5, 0, 0), halfway between
        the fluid cell and the next cell (which is solid). q = 0.5."""
        q = solve_bouzidi_q((-5.0, 0.0, 0.0), (1, 0, 0), R=4.5)
        assert q == pytest.approx(0.5, abs=1e-6)

    def test_axis_direction_q_third(self):
        """R = 4.667 = 14/3, fluid cell at (-5, 0, 0) along +x: the wall
        is at (-14/3, 0, 0), q = 5 - 14/3 = 1/3."""
        q = solve_bouzidi_q((-5.0, 0.0, 0.0), (1, 0, 0), R=14.0 / 3.0)
        assert q == pytest.approx(1.0 / 3.0, abs=1e-6)

    def test_no_intersection_returns_invalid(self):
        """A fluid cell far away from the sphere along a direction that
        misses the surface returns -1."""
        # Cell at (-10, -10, 0), going +x. The closest point along the
        # ray (-10 + t, -10, 0) to origin is (0, -10, 0), distance 10,
        # which is way bigger than R = 4. No intersection in (0, 1].
        q = solve_bouzidi_q((-10.0, -10.0, 0.0), (1, 0, 0), R=4.0)
        assert q == -1.0

    def test_edge_direction_quadratic(self):
        """Edge direction c = (1, 1, 0), |c|² = 2. Use R chosen so the
        wall is exactly mid-link (q = 0.5).

        Fluid cell at (-3, -3, 0); the wall position is x_f + q c.
        At q = 0.5: wall = (-2.5, -2.5, 0), distance from origin =
        sqrt(2 * 6.25) = sqrt(12.5) ~ 3.5355. So R = sqrt(12.5)
        should give q = 0.5."""
        R = float(np.sqrt(12.5))
        q = solve_bouzidi_q((-3.0, -3.0, 0.0), (1, 1, 0), R=R)
        assert q == pytest.approx(0.5, abs=1e-6)

    def test_q_outside_unit_interval_returned_as_invalid(self):
        """If the smaller positive root is > 1 (wall is beyond the
        neighbour cell), this is not a Bouzidi link. The caller must
        treat the neighbour as fluid (no link)."""
        # Fluid cell at (-10, 0, 0), sphere R = 4 centred at origin.
        # Wall crossings at q = 6, q = 14. Both > 1, so not a wall link.
        q = solve_bouzidi_q((-10.0, 0.0, 0.0), (1, 0, 0), R=4.0)
        assert q == -1.0

    def test_q_strictly_positive(self):
        """A fluid cell already on the surface (degenerate edge case)
        should NOT report q = 0 -- the convention is the cell centre
        of a SOLID cell sits at or inside the surface, so a "fluid"
        cell whose distance from centre equals R is a degenerate
        boundary case that the caller should handle by treating the
        cell as solid before reaching this helper."""
        # Fluid cell at (-4, 0, 0), c = (1, 0, 0), R = 4. The cell is
        # exactly on the surface. q = 0 (degenerate) -- not in (0, 1].
        q = solve_bouzidi_q((-4.0, 0.0, 0.0), (1, 0, 0), R=4.0)
        # The "smaller positive root in (0, 1]" gate filters q = 0 out.
        # The next root (q = 8) is also outside (0, 1]. Returns -1.
        assert q == -1.0


# ============================================================================
# sphere_wall_links: structural invariants on the generated list
# ============================================================================


class TestSphereWallLinks:
    def test_empty_when_sphere_is_entirely_inside_solid_region(self):
        """A sphere entirely outside the grid (negative coords) emits
        no wall links because no fluid cell has a solid neighbour."""
        links = sphere_wall_links(
            Nx=32, Ny=32, Nz=32, cx=-50.0, cy=-50.0, cz=-50.0, R=4.0,
        )
        assert isinstance(links, WallLinkList)
        assert links.n_links == 0

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="R must be positive"):
            sphere_wall_links(Nx=32, Ny=32, Nz=32, cx=16, cy=16, cz=16, R=0.0)
        with pytest.raises(ValueError, match="R must be positive"):
            sphere_wall_links(Nx=32, Ny=32, Nz=32, cx=16, cy=16, cz=16, R=-1.0)

    def test_all_q_in_unit_interval(self):
        """Every emitted q must be in (0, 1] (the Bouzidi valid range)."""
        links = sphere_wall_links(
            Nx=48, Ny=48, Nz=48, cx=24, cy=24, cz=24, R=8.5,
        )
        assert links.n_links > 0
        assert links.q.min() > 0.0
        assert links.q.max() <= 1.0

    def test_all_directions_are_non_rest(self):
        """The rest vector (i=0) must never appear -- it has |c| = 0
        and cannot be a wall link by construction."""
        links = sphere_wall_links(
            Nx=48, Ny=48, Nz=48, cx=24, cy=24, cz=24, R=8.5,
        )
        assert links.n_links > 0
        assert links.dir.min() >= 1
        assert links.dir.max() <= 18

    def test_emitted_cells_are_fluid(self):
        """The fluid cell of every wall link must be OUTSIDE the
        sphere mask (otherwise it's a solid-to-solid link, which the
        Bouzidi kernel must never see)."""
        Nx = Ny = Nz = 48
        cx = cy = cz = 24
        R = 8.5
        links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        assert links.n_links > 0
        mask = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
        assert not mask[links.x, links.y, links.z].any(), (
            "wall-link fluid cells must NOT be inside the body mask"
        )

    def test_neighbour_cells_are_solid(self):
        """For each wall link (x, y, z, i), the cell at (x+c_i) must
        be solid."""
        from src.lbm_3d_bouzidi import LATTICE_VELOCITIES_3D
        Nx = Ny = Nz = 48
        cx = cy = cz = 24
        R = 8.5
        links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        mask = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
        for k in range(links.n_links):
            i = int(links.dir[k])
            c = LATTICE_VELOCITIES_3D[i]
            xn = int(links.x[k]) + int(c[0])
            yn = int(links.y[k]) + int(c[1])
            zn = int(links.z[k]) + int(c[2])
            assert mask[xn, yn, zn], (
                f"link {k}: neighbour ({xn}, {yn}, {zn}) of fluid "
                f"({links.x[k]}, {links.y[k]}, {links.z[k]}) along "
                f"direction {i} = {tuple(c)} should be solid but isn't"
            )

    def test_link_count_scales_as_surface_area(self):
        """Wall-link count scales as ~ 4πR² (surface area in cells)
        times the average number of solid neighbours per surface fluid
        cell (some constant > 1, < 18). For a moderate-R sphere far
        from the domain edges we expect the count to land within a
        factor of 2 of 4πR² in the centred limit."""
        # Run twice at different R, check the ratio of link counts
        # follows ~ R₂² / R₁².
        Nx = Ny = Nz = 64
        cx = cy = cz = 32
        links_small = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R=4.5)
        links_big = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R=9.0)
        ratio_expected = (9.0 / 4.5) ** 2  # = 4
        ratio_actual = links_big.n_links / links_small.n_links
        assert 0.7 * ratio_expected < ratio_actual < 1.3 * ratio_expected, (
            f"link count ratio {ratio_actual:.2f} not within 30 % of "
            f"surface-area scaling prediction {ratio_expected:.2f}"
        )

    def test_link_distribution_is_symmetric_under_reflection(self):
        """A sphere centred at the domain centre should produce equal
        numbers of wall links in +x and -x (and likewise +y / -y,
        +z / -z). This catches asymmetry bugs in the cell iteration
        or the direction enumeration."""
        Nx = Ny = Nz = 32
        cx = cy = cz = 16  # exact centre
        R = 6.0
        links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        assert links.n_links > 0
        # Direction indices: +x = 1, -x = 2; +y = 3, -y = 4; +z = 5, -z = 6
        # (per LATTICE_VELOCITIES_3D in lbm_3d).
        n_plus_x = int((links.dir == 1).sum())
        n_minus_x = int((links.dir == 2).sum())
        n_plus_y = int((links.dir == 3).sum())
        n_minus_y = int((links.dir == 4).sum())
        n_plus_z = int((links.dir == 5).sum())
        n_minus_z = int((links.dir == 6).sum())
        assert n_plus_x == n_minus_x, (
            f"+x links ({n_plus_x}) != -x links ({n_minus_x})"
        )
        assert n_plus_y == n_minus_y, (
            f"+y links ({n_plus_y}) != -y links ({n_minus_y})"
        )
        assert n_plus_z == n_minus_z, (
            f"+z links ({n_plus_z}) != -z links ({n_minus_z})"
        )

    def test_q_distribution_matches_analytic_for_known_link(self):
        """For a sphere of R = 4.5 centred at (10, 10, 10), the fluid
        cell (5, 10, 10) has its +x neighbour (6, 10, 10) inside the
        sphere (distance = 4 < 4.5). The wall is halfway between
        x = 5 and x = 6, so q = 0.5. Locate that specific link in the
        generated list and confirm its q."""
        Nx = Ny = Nz = 32
        cx = cy = cz = 10
        R = 4.5
        links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        # Find the link at fluid cell (5, 10, 10), direction +x (i = 1).
        mask = (
            (links.x == 5)
            & (links.y == 10)
            & (links.z == 10)
            & (links.dir == 1)
        )
        assert mask.any(), "expected wall link at (5, 10, 10) along +x"
        q = float(links.q[mask][0])
        assert q == pytest.approx(0.5, abs=1e-5)


# ============================================================================
# make_sphere_mask: sanity that it matches src.lbm_3d._make_sphere_mask
# ============================================================================


class TestMakeSphereMask:
    def test_matches_lbm_3d_helper(self):
        """The mask helper in this module must agree with the one in
        src/lbm_3d.py -- they share the surface convention
        (|d|² <= R² is solid)."""
        from src.lbm_3d import _make_sphere_mask as ref_mask
        Nx = Ny = Nz = 32
        cx = cy = cz = 16
        R = 5.0
        a = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
        b = ref_mask(Nx, Ny, Nz, cx, cy, cz, R)
        assert np.array_equal(a, b)


# ============================================================================
# Bouzidi correction kernel: integration-level checks on run_channel_smoke
# ============================================================================


class TestBouzidiCorrection:
    """The full-way bounce-back inside `step_bgk_3d` is corrected to
    Bouzidi linear interpolation when `wall_links` is passed to
    `run_channel_smoke`. The invariants below pin the kernel against
    physics:

      1. At q = 0.5 the Bouzidi formula reduces to f_tilde_i(x_f),
         which equals what full-way BB already writes. So q=0.5
         everywhere must be a no-op.
      2. At q != 0.5 the corrected output must differ from full-way.
      3. Velocity inside the body stays zero under Bouzidi too
         (no leakage via the corrected populations).
    """

    def _build_sphere_setup(self, q_override=None):
        """Return (Nx, Ny, Nz, body, wall_links) -- optionally with all
        q's overridden to a specific value for sanity tests."""
        from src.lbm_3d_bouzidi import (
            sphere_wall_links,
            make_sphere_mask,
        )
        Nx, Ny, Nz = 32, 24, 24
        cx, cy, cz = 12, 12, 12
        R = 4.5
        body = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
        wall_links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
        if q_override is not None:
            wall_links.q[:] = np.float32(q_override)
        return Nx, Ny, Nz, body, wall_links

    @pytest.mark.slow
    def test_bouzidi_at_q_half_matches_full_way(self):
        """At q = 0.5 the Bouzidi linear formula collapses to
        f_tilde_i(x_f). Running with all-q=0.5 wall-links must produce
        the SAME velocity field (to float32 precision) as running
        without Bouzidi (full-way BB only). This is the canonical
        sanity check that the correction kernel is wired correctly."""
        from src.lbm_3d import run_channel_smoke
        Nx, Ny, Nz, body, wall_links = self._build_sphere_setup(q_override=0.5)
        u_in, nu, n_steps = 0.04, 0.02, 200

        _, ux_full, _, _, _ = run_channel_smoke(
            Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=n_steps,
            body=body,
        )
        _, ux_bz, _, _, _ = run_channel_smoke(
            Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=n_steps,
            body=body, wall_links=wall_links,
        )
        # float32 precision over 200 steps allows ~1e-4 relative drift
        # from non-associative summation order, but the velocity field
        # values are O(0.01-0.1), so an absolute tolerance of 1e-5 is
        # a tight pin. If this fails the formulas at q=0.5 are
        # diverging from full-way BB and something is wrong.
        max_abs_diff = float(np.max(np.abs(ux_full - ux_bz)))
        assert max_abs_diff < 1e-5, (
            f"q=0.5 Bouzidi differs from full-way BB by "
            f"{max_abs_diff:.2e} -- formula reduction broken"
        )

    @pytest.mark.slow
    def test_bouzidi_at_real_q_differs_from_full_way(self):
        """With the ANALYTIC q-field (most q's not equal to 0.5),
        Bouzidi must produce a measurably different velocity field
        from full-way BB. This is the dual of the q=0.5 test --
        catches a kernel that silently does nothing."""
        from src.lbm_3d import run_channel_smoke
        Nx, Ny, Nz, body, wall_links = self._build_sphere_setup()
        # Confirm the wall-link list has q values away from 0.5
        # (otherwise the test would be vacuous on a corner-case mask).
        assert np.any(np.abs(wall_links.q - 0.5) > 0.05), (
            "wall_link q-field has no q != 0.5 entries; test "
            "cannot distinguish Bouzidi from full-way for this case"
        )

        u_in, nu, n_steps = 0.04, 0.02, 200
        _, ux_full, _, _, _ = run_channel_smoke(
            Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=n_steps,
            body=body,
        )
        _, ux_bz, _, _, _ = run_channel_smoke(
            Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=n_steps,
            body=body, wall_links=wall_links,
        )
        max_abs_diff = float(np.max(np.abs(ux_full - ux_bz)))
        # Effect size should be at least ~1 % of u_in somewhere -- the
        # near-wall populations carry the Bouzidi correction directly.
        assert max_abs_diff > 0.01 * u_in, (
            f"Bouzidi field is indistinguishable from full-way "
            f"(max diff = {max_abs_diff:.2e}, expected > {0.01 * u_in:.2e}). "
            f"Correction kernel may be silently skipping wall links."
        )

    @pytest.mark.slow
    def test_bouzidi_preserves_solid_cell_zero_velocity(self):
        """With Bouzidi corrections active, the post-pass-zero in
        `run_channel_smoke` (which sets ux/uy/uz to 0 inside the body
        mask) must still hold. Catches a regression where the
        correction kernel writes into solid cells."""
        from src.lbm_3d import run_channel_smoke
        Nx, Ny, Nz, body, wall_links = self._build_sphere_setup()
        _, ux, uy, uz, _ = run_channel_smoke(
            Nx=Nx, Ny=Ny, Nz=Nz, u_in=0.04, nu=0.02, n_steps=100,
            body=body, wall_links=wall_links,
        )
        assert float(np.max(np.abs(ux[body]))) == 0.0
        assert float(np.max(np.abs(uy[body]))) == 0.0
        assert float(np.max(np.abs(uz[body]))) == 0.0
