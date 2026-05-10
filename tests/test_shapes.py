"""Tests for shape mask generators in src/shapes.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.shapes import cylinder_mask


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
