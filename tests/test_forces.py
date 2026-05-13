"""Tests for the momentum-exchange force calculator in src/forces.py.

Per the principle "synthetic distributions where the answer is known by hand":
each test isolates one failure mode -- zero baseline, symmetry, direction sign,
diagonal handling, linearity, and a sim-scale sanity check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.forces import momentum_exchange_force
from src.lbm import LATTICE_VELOCITIES, equilibrium
from src.shapes import cylinder_mask


def test_force_zero_f_is_zero():
    """Trivial baseline: f = 0 everywhere implies zero force."""
    f = np.zeros((9, 10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    F = momentum_exchange_force(f, mask)
    assert np.allclose(F, [0.0, 0.0])


def test_force_uniform_f_around_symmetric_solid_is_zero():
    """A single solid cell with uniform f in all 8 neighbors: 8 momentum
    contributions cancel pairwise (east cancels west, NE cancels SW, etc.).
    Net force = 0. Catches summation-direction bugs."""
    f = np.ones((9, 10, 10)) * 0.1
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    F = momentum_exchange_force(f, mask)
    assert np.allclose(F, [0.0, 0.0])


def test_force_east_population_at_west_neighbor():
    """One fluid cell at (3, 5) has only f[1]=0.5 (east). Its east neighbor
    (4, 5) is solid. Single wall-link, single direction.
    Expected: F = 2 * c_1 * f_1 = 2 * (1, 0) * 0.5 = (1.0, 0.0).
    Catches sign / direction-index bugs."""
    f = np.zeros((9, 10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 5] = True
    f[1, 3, 5] = 0.5
    F = momentum_exchange_force(f, mask)
    assert np.allclose(F, [1.0, 0.0])


def test_force_ne_population_at_ne_neighbor():
    """f[5]=0.7 (NE) at fluid cell (3, 3); cell (4, 4) is solid.
    Expected: F = 2 * c_5 * f_5 = 2 * (1, 1) * 0.7 = (1.4, 1.4).
    Catches diagonal direction handling."""
    f = np.zeros((9, 10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    f[5, 3, 3] = 0.7
    F = momentum_exchange_force(f, mask)
    assert np.allclose(F, [1.4, 1.4])


def test_force_is_linear_in_f():
    """F(f1 + f2) = F(f1) + F(f2). Catches non-linear contamination from any
    refactor that tries to be clever (e.g., conditional handling)."""
    rng = np.random.default_rng(seed=100)
    Nx, Ny = 20, 20
    mask = cylinder_mask(Nx, Ny, cx=10, cy=10, radius=3)
    f1 = rng.uniform(0, 0.1, size=(9, Nx, Ny))
    f2 = rng.uniform(0, 0.1, size=(9, Nx, Ny))
    F1 = momentum_exchange_force(f1, mask)
    F2 = momentum_exchange_force(f2, mask)
    F_combined = momentum_exchange_force(f1 + f2, mask)
    assert np.allclose(F_combined, F1 + F2)


def test_force_on_cylinder_in_uniform_eastward_flow():
    """Sanity at simulation scale: uniform east-flowing equilibrium f over a
    centered cylinder should produce positive drag (Fx > 0) and zero net side
    force (Fy = 0 to machine precision, by symmetry). The actual magnitude
    isn't validated here (that's the Re=100 simulation's job) -- this just
    catches sign flips and broken symmetry."""
    Nx, Ny = 80, 60
    cx, cy, r = 40, 30, 5  # symmetric about y = 30
    U = 0.05

    mask = cylinder_mask(Nx, Ny, cx, cy, r)
    rho0 = np.ones((Nx, Ny))
    u0 = np.zeros((2, Nx, Ny))
    u0[0] = U
    f = equilibrium(rho0, u0)

    F = momentum_exchange_force(f, mask)
    assert F[0] > 0, f"Drag should be positive for east-flowing fluid, got Fx = {F[0]}"
    assert abs(F[1]) < 1e-10, f"Symmetric setup should give Fy = 0, got {F[1]}"
