"""Tests for shape mask generators in src/shapes.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.shapes import cylinder_mask, ellipse_mask, naca4_airfoil_mask, square_mask


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
