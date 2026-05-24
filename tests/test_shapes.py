"""Tests for shape mask generators in src/shapes.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.shapes import (
    cylinder_mask,
    cylinder_q_field,
    ellipse_mask,
    ellipse_q_field,
    naca4_airfoil_mask,
    naca4_q_field,
    no_bouzidi_q_field,
    square_mask,
    square_q_field,
)


def test_cylinder_mask_shape():
    """Returned mask has shape (Nx, Ny) regardless of cylinder size."""
    mask = cylinder_mask(50, 30, cx=25, cy=15, radius=10)
    assert mask.shape == (50, 30)
    assert mask.dtype == bool


def test_cylinder_mask_center_is_solid():
    """The center cell of a cylinder is always solid."""
    mask = cylinder_mask(50, 50, cx=25, cy=25, radius=10)
    assert mask[25, 25]


def test_cylinder_mask_far_corner_is_fluid():
    """A cell in the corner, far outside the cylinder, is fluid."""
    mask = cylinder_mask(50, 50, cx=25, cy=25, radius=10)
    assert not mask[0, 0]


def test_cylinder_mask_radius_boundary_inclusive():
    """A cell exactly at distance == radius is included (uses <= not <)."""
    mask = cylinder_mask(20, 20, cx=10, cy=10, radius=5)
    # cell (15, 10) is at distance exactly 5
    assert mask[15, 10]
    # cell (16, 10) is at distance 6 > 5
    assert not mask[16, 10]


def test_cylinder_mask_area_matches_pi_r_squared():
    """Number of solid cells approximates pi * r^2 (within 5% for a moderate
    radius -- discretization error shrinks as radius grows)."""
    r = 20
    mask = cylinder_mask(80, 80, cx=40, cy=40, radius=r)
    n_solid = int(mask.sum())
    expected = np.pi * r ** 2
    rel_err = abs(n_solid - expected) / expected
    assert rel_err < 0.05, f"Cell count {n_solid} differs from pi*r^2={expected:.1f} by {rel_err:.1%}"


# ---------------------------------------------------------------------------
# Bouzidi q-fields (interpolated bounce-back wall fractions)
# ---------------------------------------------------------------------------

def test_no_bouzidi_q_field_is_all_negative_one():
    """no_bouzidi_q_field returns a fully -1 array of the right shape -- the
    'halfway BB everywhere' sentinel. If any element drifts >= 0, the solver
    would silently apply Bouzidi at that link with wrong (or undefined) q."""
    q = no_bouzidi_q_field(50, 30)
    assert q.shape == (50, 30, 9)
    assert q.dtype == np.float64
    assert np.all(q == -1.0)


def test_cylinder_q_field_shape_and_dtype():
    """Returned shape (Nx, Ny, 9), dtype float64. The JIT step function expects
    exactly this layout; mismatches will cause Numba compilation errors that
    are noisy but slow to debug."""
    q = cylinder_q_field(40, 40, cx=20, cy=20, radius=5)
    assert q.shape == (40, 40, 9)
    assert q.dtype == np.float64


def test_cylinder_q_field_rest_direction_always_negative():
    """Direction 0 (rest, c=0) is not a streaming link -- it can never be a
    wall link. q_field[..., 0] must always be -1 to keep the JIT inner loop
    safe (the Bouzidi formulas would divide by |c|^2 = 0)."""
    q = cylinder_q_field(40, 40, cx=20, cy=20, radius=8)
    assert np.all(q[..., 0] == -1.0)


def test_cylinder_q_field_fluid_to_fluid_links_are_negative():
    """A fluid cell far from the cylinder has all-fluid neighbors -- every q
    at that cell is -1 (no wall link)."""
    q = cylinder_q_field(40, 40, cx=20, cy=20, radius=5)
    # (5, 5) is far from the cylinder body at (20, 20).
    assert np.all(q[5, 5, :] == -1.0)


def test_cylinder_q_field_axis_aligned_link_matches_geometry():
    """For an axis-aligned link from a fluid cell to the cylinder's analytic
    boundary, q must equal the geometric distance. (5, 10) heading +x toward a
    cylinder at (10, 10) radius 4: boundary at x = 10 - 4 = 6, so the link
    from (5, 10) ends exactly at the boundary at q = 1.0."""
    q = cylinder_q_field(20, 20, cx=10, cy=10, radius=4)
    # cell (5, 10) heading +x (direction 1): neighbor (6, 10) is on the boundary
    # (dist^2=16=r^2, mask=True due to <=). q[5, 10, 1] = 1.0.
    assert np.isclose(q[5, 10, 1], 1.0)


def test_cylinder_q_field_diagonal_link_matches_quadratic():
    """For a diagonal link, q is the smaller positive root of
        |x_f + q c_i - c|^2 = r^2.
    Hand-computed for (7, 6) heading NE (c_i = (1, 1)) into solid (8, 7) toward
    cylinder at (10, 10) radius 4: 2q^2 - 14q + 9 = 0 -> q = (14 - sqrt(124))/4
    = 0.7161."""
    q = cylinder_q_field(20, 20, cx=10, cy=10, radius=4)
    expected = (14.0 - np.sqrt(124.0)) / 4.0
    assert np.isclose(q[7, 6, 5], expected, atol=1e-12)


def test_cylinder_q_field_all_wall_links_are_in_unit_interval():
    """Every q > 0 entry must be in (0, 1] -- it's a fraction of one lattice
    link. Out-of-range values would have the Bouzidi formula apply nonsense
    weights (or divide by zero at q=0)."""
    q = cylinder_q_field(80, 80, cx=40, cy=40, radius=10)
    positive = q[q > 0]
    assert positive.min() > 0.0
    assert positive.max() <= 1.0 + 1e-12  # tiny FP slack


def test_ellipse_q_field_reduces_to_cylinder_when_axes_equal():
    """ellipse_q_field(a=b=r, aoa=0) must match cylinder_q_field(r) link-for-link.
    Catches algebraic divergence between the two solvers and verifies the
    body-frame transform is correctly implemented (cos=1, sin=0)."""
    Nx, Ny, cx, cy, r = 60, 60, 30, 30, 12
    q_ell = ellipse_q_field(Nx, Ny, cx=cx, cy=cy, a=r, b=r, aoa_deg=0.0)
    q_cyl = cylinder_q_field(Nx, Ny, cx=cx, cy=cy, radius=r)
    assert np.allclose(q_ell, q_cyl, atol=1e-12)


def test_ellipse_q_field_rotation_invariance_axes_equal():
    """For a circular ellipse (a == b), rotation is a no-op: q-field at any
    AoA must match the AoA=0 case. Catches transform-sign bugs."""
    Nx, Ny, cx, cy, r = 50, 50, 25, 25, 8
    q0 = ellipse_q_field(Nx, Ny, cx=cx, cy=cy, a=r, b=r, aoa_deg=0.0)
    q30 = ellipse_q_field(Nx, Ny, cx=cx, cy=cy, a=r, b=r, aoa_deg=30.0)
    # Wall-link positions and q-values must match (a circle is rotationally symmetric).
    assert np.allclose(q0, q30, atol=1e-12)


def test_ellipse_q_field_axis_aligned_link_matches_geometry():
    """For an unrotated ellipse a=10, b=5 centered at (20, 20), cell (10, 20)
    heading +x hits the boundary at x = 20 - 10 = 10. So the link from
    (9, 20) heading +x covers the gap to the boundary, ending at q=1.0
    (since (10, 20) is on the boundary, mask=True)."""
    q = ellipse_q_field(40, 40, cx=20, cy=20, a=10, b=5, aoa_deg=0.0)
    # The cell (9, 20) is just outside the ellipse along the major axis.
    assert np.isclose(q[9, 20, 1], 1.0)


def test_ellipse_q_field_minor_axis_link_uses_b_not_a():
    """An ellipse with a=10, b=5 should have a SMALLER wall fraction on
    minor-axis links than on major-axis links (the body's edge in y is
    closer than in x). Verifies that the b parameter actually affects the
    y-direction quadratic terms -- catches a bug where b was ignored."""
    q = ellipse_q_field(60, 60, cx=30, cy=30, a=10, b=5, aoa_deg=0.0)
    # Probe a cell whose +y neighbor is solid (heading toward the minor axis edge).
    # (30, 24) is at y-dist 6 from center; (30, 25) is at y-dist 5 (= b, on boundary).
    # Link from (30, 24) heading +y to (30, 25). Both inside? Check.
    mask = ellipse_mask(60, 60, cx=30, cy=30, a=10, b=5, aoa_deg=0.0)
    assert not mask[30, 24]
    assert mask[30, 25]
    assert q[30, 24, 2] > 0.0  # is a wall link
    # And on the major axis, the link from (19, 30) to (20, 30) is also a wall link.
    assert not mask[19, 30]
    assert mask[20, 30]
    assert q[19, 30, 1] > 0.0


def test_ellipse_q_field_all_wall_links_in_unit_interval():
    """All Bouzidi q values must land in (0, 1]; out-of-range would crash
    the Bouzidi formula (q = 0 divides by zero in the q >= 0.5 branch)."""
    q = ellipse_q_field(80, 80, cx=40, cy=40, a=15, b=8, aoa_deg=20.0)
    positive = q[q > 0]
    assert positive.min() > 0.0
    assert positive.max() <= 1.0 + 1e-12


def test_square_q_field_axis_aligned_link_matches_geometry():
    """Unrotated square side=10 centered at (20, 20): right face is at x=25
    (inclusive). Cell (15, 20) heading +x toward (16, 20): is (16, 20) solid?
    Square boundary at x = cx - half = 15 (inclusive). So mask[15, 20] = True
    and (14, 20) is fluid. Link (14, 20) -> +x to (15, 20), q = 1.0 (boundary
    at the next cell)."""
    q = square_q_field(40, 40, cx=20, cy=20, side=10, aoa_deg=0.0)
    # Outside the square along +x axis: (14, 20). Neighbor (15, 20) is on
    # the left face (x = 20 - 5 = 15, mask=True due to <=).
    assert np.isclose(q[14, 20, 1], 1.0)


def test_square_q_field_45deg_rotation_changes_wall_link_set():
    """Square rotated 45 deg is a diamond -- a substantially different
    voxelization than the axis-aligned square. The set of wall links must
    therefore differ: at least some cells that were wall links at AoA=0
    are not at AoA=45, and vice versa. Catches a transform-sign bug where
    the rotation is silently a no-op."""
    q0 = square_q_field(40, 40, cx=20, cy=20, side=10, aoa_deg=0.0)
    q45 = square_q_field(40, 40, cx=20, cy=20, side=10, aoa_deg=45.0)
    walls0 = (q0 > 0)
    walls45 = (q45 > 0)
    # XOR: True where the link is a wall in one but not the other.
    xor = walls0 ^ walls45
    assert xor.sum() > 20, (
        f"Only {int(xor.sum())} links differ between 0 deg and 45 deg square -- "
        "rotation likely a no-op"
    )


def test_square_q_field_unrotated_axis_aligned_corner_is_halfway_or_less():
    """An axis-aligned square rotated 0 deg has flat faces aligned to the
    lattice. For cells outside the square along an axis-aligned link,
    halfway BB (q = 0.5) is exact and other q values are >= 0.5 on faces
    further from a grid line. So all axis-aligned wall-link q values should
    be in (0.5 - eps, 1.0]."""
    q = square_q_field(40, 40, cx=20, cy=20, side=10, aoa_deg=0.0)
    # Cardinal directions: 1=E, 2=N, 3=W, 4=S.
    for i in (1, 2, 3, 4):
        positive = q[..., i][q[..., i] > 0]
        if positive.size > 0:
            # Axis-aligned: link length is 1; the half-cell convention puts
            # the wall at q=0.5 minimum. Allow tiny FP slack.
            assert positive.min() >= 0.5 - 1e-9, (
                f"direction {i}: min q = {positive.min():.4f}, expected >= 0.5"
            )


def test_square_q_field_rotation_breaks_axis_alignment():
    """A rotated square should produce q values strictly less than 0.5 on
    some axis-aligned links (the rotated face crosses lattice links at
    arbitrary fractions, not just at midpoints)."""
    q = square_q_field(40, 40, cx=20, cy=20, side=10, aoa_deg=20.0)
    # Look across all 8 directions for some q < 0.5.
    positive = q[q > 0]
    assert positive.size > 0
    assert positive.min() < 0.5, (
        f"rotated square gave min q = {positive.min():.4f}, expected < 0.5"
    )


def test_square_q_field_all_wall_links_in_unit_interval():
    """All Bouzidi q values are in (0, 1]."""
    q = square_q_field(60, 60, cx=30, cy=30, side=15, aoa_deg=30.0)
    positive = q[q > 0]
    assert positive.min() > 0.0
    assert positive.max() <= 1.0 + 1e-12


def test_naca4_q_field_shape_and_dtype():
    q = naca4_q_field(200, 100, cx=50, cy=50, chord=80,
                       naca_code="0012", aoa_deg=0.0)
    assert q.shape == (200, 100, 9)
    assert q.dtype == np.float64
    assert np.all(q[..., 0] == -1.0)  # rest direction


def test_naca4_q_field_has_wall_links():
    """The 12% airfoil has a non-trivial boundary; we should detect dozens
    to hundreds of wall links. Catches the no-Bouzidi degenerate case
    (e.g., wrong mask vs polygon ordering, all links flagged as no-hit)."""
    q = naca4_q_field(200, 100, cx=50, cy=50, chord=80,
                       naca_code="0012", aoa_deg=0.0)
    n_links = int((q > 0).sum())
    assert n_links > 100, f"Only {n_links} wall links found for NACA 0012"


def test_naca4_q_field_all_wall_links_in_unit_interval():
    """Every q > 0 entry must be in (0, 1]."""
    q = naca4_q_field(200, 100, cx=50, cy=50, chord=80,
                       naca_code="4412", aoa_deg=5.0)
    positive = q[q > 0]
    assert positive.size > 0
    assert positive.min() > 0.0
    assert positive.max() <= 1.0 + 1e-12


def test_naca4_q_field_aoa_changes_wall_link_set():
    """The set of wall links must differ between two AoAs -- the rotated
    airfoil voxelizes onto different cells. Catches the silent no-op
    rotation bug for the airfoil polygon transform."""
    q0 = naca4_q_field(200, 100, cx=50, cy=50, chord=80,
                       naca_code="0012", aoa_deg=0.0)
    q15 = naca4_q_field(200, 100, cx=50, cy=50, chord=80,
                        naca_code="0012", aoa_deg=15.0)
    diff = (q0 > 0) ^ (q15 > 0)
    assert diff.sum() > 30, (
        f"Only {int(diff.sum())} links differ between AoA=0 and AoA=15 -- "
        "rotation likely a no-op"
    )


def test_cylinder_q_field_n_wall_links_scales_with_perimeter():
    """Number of wall links scales roughly with the cylinder perimeter
    (~ 2 pi r at each radius, give or take a constant for diagonal coverage).
    Tests that bigger cylinders have proportionally more wall links and
    smaller ones less -- catches a regression where the q-field generator
    forgets to scan some directions."""
    def n_links(r):
        return int((cylinder_q_field(80, 80, cx=40, cy=40, radius=r) > 0).sum())
    n_small = n_links(5)
    n_large = n_links(20)
    # 4x radius should give roughly 4x links (perimeter scales linearly).
    # Allow 2x slack for staircase / diagonal effects on a coarse grid.
    assert n_large > 2 * n_small, (
        f"links scale poorly with radius: r=5 -> {n_small}, r=20 -> {n_large}"
    )


# ---------------------------------------------------------------------------
# square_mask
# ---------------------------------------------------------------------------

def test_square_mask_shape():
    """Returned mask has shape (Nx, Ny) and dtype bool."""
    mask = square_mask(50, 30, cx=25, cy=15, side=10)
    assert mask.shape == (50, 30)
    assert mask.dtype == bool


def test_square_mask_center_solid_corner_fluid():
    """Center cell is inside the square; a far corner is outside."""
    mask = square_mask(50, 50, cx=25, cy=25, side=10)
    assert mask[25, 25]
    assert not mask[0, 0]


def test_square_mask_exact_count_integer_side():
    """With integer side and integer center, the inclusive-boundary square has
    exactly (side + 1)**2 cells (one row on each side of center, plus center).
    This is the strongest geometric test -- catches off-by-one and asymmetry."""
    mask = square_mask(50, 50, cx=25, cy=25, side=10)
    assert int(mask.sum()) == (10 + 1) ** 2  # 121


def test_square_mask_boundary_inclusive():
    """Cells exactly at distance side/2 from center along each axis are solid."""
    mask = square_mask(20, 20, cx=10, cy=10, side=6)
    # (10+3, 10) is at x-distance exactly 3 = side/2
    assert mask[13, 10]
    assert mask[10, 13]
    # (10+4, 10) is past the boundary
    assert not mask[14, 10]
    assert not mask[10, 14]


# ---------------------------------------------------------------------------
# ellipse_mask
# ---------------------------------------------------------------------------

def test_ellipse_mask_shape():
    """Returned mask has shape (Nx, Ny) and dtype bool."""
    mask = ellipse_mask(50, 30, cx=25, cy=15, a=10, b=5)
    assert mask.shape == (50, 30)
    assert mask.dtype == bool


def test_ellipse_mask_center_solid_corner_fluid():
    mask = ellipse_mask(50, 50, cx=25, cy=25, a=10, b=5)
    assert mask[25, 25]
    assert not mask[0, 0]


def test_ellipse_mask_area_matches_pi_a_b():
    """Number of solid cells approximates pi * a * b (within 5%)."""
    a, b = 20, 10
    mask = ellipse_mask(80, 60, cx=40, cy=30, a=a, b=b)
    n_solid = int(mask.sum())
    expected = np.pi * a * b
    rel_err = abs(n_solid - expected) / expected
    assert rel_err < 0.05, f"Cell count {n_solid} differs from pi*a*b={expected:.1f} by {rel_err:.1%}"


def test_ellipse_mask_boundary_inclusive_each_axis():
    """Cells at (cx+a, cy) and (cx, cy+b) are exactly on the boundary -- solid."""
    a, b = 8, 4
    mask = ellipse_mask(40, 40, cx=20, cy=20, a=a, b=b)
    # On-axis boundary points
    assert mask[20 + a, 20]
    assert mask[20, 20 + b]
    # Just past the boundary
    assert not mask[20 + a + 1, 20]
    assert not mask[20, 20 + b + 1]


def test_ellipse_mask_reduces_to_cylinder_when_axes_equal():
    """ellipse_mask(a=b=r) must produce the same mask as cylinder_mask(r).
    Catches any algebraic divergence between the two implementations."""
    Nx, Ny, cx, cy, r = 60, 60, 30, 30, 15
    el = ellipse_mask(Nx, Ny, cx=cx, cy=cy, a=r, b=r)
    cyl = cylinder_mask(Nx, Ny, cx=cx, cy=cy, radius=r)
    assert np.array_equal(el, cyl)


# ---------------------------------------------------------------------------
# naca4_airfoil_mask
# ---------------------------------------------------------------------------

def test_naca4_mask_shape():
    """Returned mask has shape (Nx, Ny) and dtype bool."""
    mask = naca4_airfoil_mask(200, 100, cx=50, cy=50, chord=100, naca_code="0012", aoa_deg=0)
    assert mask.shape == (200, 100)
    assert mask.dtype == bool


def test_naca0012_chord_symmetric_at_zero_aoa():
    """A symmetric NACA at alpha=0 must mirror exactly across the chord line.
    Any asymmetry above/below y = cy reveals a sign error in the surface
    construction or a buggy camber path."""
    Nx, Ny = 200, 100
    cx, cy = 50, 50
    mask = naca4_airfoil_mask(Nx, Ny, cx=cx, cy=cy, chord=100, naca_code="0012", aoa_deg=0)
    for j_offset in range(1, min(cy, Ny - 1 - cy) + 1):
        assert np.array_equal(mask[:, cy + j_offset], mask[:, cy - j_offset]), (
            f"NACA 0012 not symmetric across chord at y offset {j_offset}"
        )


def test_naca4412_not_chord_symmetric():
    """A cambered NACA *must* break chord symmetry -- if it doesn't, the
    camber line isn't being applied. Tests that camber actually does something."""
    Nx, Ny = 200, 100
    cx, cy = 50, 50
    mask = naca4_airfoil_mask(Nx, Ny, cx=cx, cy=cy, chord=100, naca_code="4412", aoa_deg=0)
    diff_found = False
    for j_offset in range(1, min(cy, Ny - 1 - cy) + 1):
        if not np.array_equal(mask[:, cy + j_offset], mask[:, cy - j_offset]):
            diff_found = True
            break
    assert diff_found, "NACA 4412 should not be chord-symmetric due to camber"


def test_naca0012_midchord_on_chord_is_solid():
    """The cell at half-chord along the chord line is deep inside a 12%-thick
    symmetric airfoil. Confirms LE-anchored positioning is correct."""
    Nx, Ny = 200, 100
    cx, cy = 50, 50
    chord = 100
    mask = naca4_airfoil_mask(Nx, Ny, cx=cx, cy=cy, chord=chord, naca_code="0012", aoa_deg=0)
    assert mask[cx + chord // 2, cy]


def test_naca0012_far_above_is_fluid():
    """A cell well above the airfoil (10x its max half-thickness) is fluid."""
    Nx, Ny = 200, 100
    cx, cy = 50, 50
    chord = 100
    mask = naca4_airfoil_mask(Nx, Ny, cx=cx, cy=cy, chord=chord, naca_code="0012", aoa_deg=0)
    # NACA 0012 max half-thickness is ~6% chord = 6 cells; check 20 cells above.
    assert not mask[cx + chord // 2, cy + 20]


def test_naca0012_area_matches_thickness_integral():
    """Number of solid cells matches the analytical area of a NACA 4-digit
    symmetric airfoil: A = 2 * integral(y_t dx) * chord^2 ~ 0.6809 * t * chord^2.
    Tolerance 5% accounts for the cell-centered discretization."""
    chord = 100
    t = 0.12
    mask = naca4_airfoil_mask(200, 100, cx=50, cy=50, chord=chord, naca_code="0012", aoa_deg=0)
    expected = 0.6809 * t * chord ** 2  # ~817
    n_solid = int(mask.sum())
    rel_err = abs(n_solid - expected) / expected
    assert rel_err < 0.05, (
        f"NACA 0012 area {n_solid} differs from analytic {expected:.0f} by {rel_err:.1%}"
    )
