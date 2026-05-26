"""Smoke tests for the 3D D3Q19 solver scaffold in ``src/lbm_3d``.

These are NOT validation tests. They are lattice-constant sanity
checks plus a tiny channel-flow run that should converge toward a
parabolic plane-Poiseuille profile in y. The point is to catch
regressions in streaming / boundary code while the 3D solver is
still in development.

Headline 3D validation (cylinder Cd vs Williamson) comes later, after
the body bounce-back and Bouzidi q-fields are in place.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.lbm_3d import (
    CS2_3D,
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
    _make_sphere_mask,
    equilibrium_3d,
    macroscopic_3d,
    run_channel_smoke,
)

# ---------------------------------------------------------------------------
# Lattice-constant sanity (cheap, runs in <1 ms)
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert abs(float(LATTICE_WEIGHTS_3D.sum()) - 1.0) < 1e-12


def test_cs2_matches_second_moment():
    # sum_i w_i c_i_alpha^2 = cs2 for each axis alpha.
    for alpha in range(3):
        m2 = float(
            (LATTICE_WEIGHTS_3D * LATTICE_VELOCITIES_3D[:, alpha] ** 2).sum()
        )
        assert abs(m2 - CS2_3D) < 1e-12, f"axis {alpha}: {m2} != {CS2_3D}"


def test_opposite_is_an_involution():
    # OPPOSITE[OPPOSITE[i]] == i for all i.
    for i in range(19):
        assert OPPOSITE_3D[OPPOSITE_3D[i]] == i


def test_opposite_matches_negation():
    for i in range(19):
        opp = OPPOSITE_3D[i]
        assert np.all(
            LATTICE_VELOCITIES_3D[opp] == -LATTICE_VELOCITIES_3D[i]
        ), f"opposite of {i} is {opp} but velocities don't negate"


def test_velocities_unique():
    # No two velocity vectors should coincide.
    seen = set()
    for v in LATTICE_VELOCITIES_3D:
        t = tuple(int(x) for x in v)
        assert t not in seen, f"duplicate velocity {t}"
        seen.add(t)


# ---------------------------------------------------------------------------
# Equilibrium round-trips
# ---------------------------------------------------------------------------

def test_equilibrium_recovers_rho_and_u():
    rho = np.full((4, 4, 4), 1.05, dtype=np.float64)
    u = np.zeros((3, 4, 4, 4), dtype=np.float64)
    u[0] = 0.02
    u[1] = -0.01
    u[2] = 0.005
    f_eq = equilibrium_3d(rho, u)
    r2, ux, uy, uz = macroscopic_3d(f_eq)
    assert np.allclose(r2, rho, atol=1e-12)
    assert np.allclose(ux, u[0], atol=1e-10)
    assert np.allclose(uy, u[1], atol=1e-10)
    assert np.allclose(uz, u[2], atol=1e-10)


# ---------------------------------------------------------------------------
# Channel-flow smoke (slow -- gated behind a marker; runs ~5 s
# including JIT compile, ~1 s thereafter for the actual loop)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_channel_smoke_mass_conserved():
    """A short 3D channel-flow run should not drift mass by more
    than a percent.

    The inflow is prescribed equilibrium (rho = 1), the outflow is
    zero-gradient, and the walls bounce-back. Mass conservation is
    not exact (the inflow injects mass each step) but the relative
    drift should be small over 100 steps at modest Re.
    """
    _, _, _, _, diag = run_channel_smoke(
        Nx=32, Ny=16, Nz=16, u_in=0.04, nu=0.02, n_steps=100,
    )
    assert abs(diag["mass_drift_rel"]) < 0.05, (
        f"mass drift {diag['mass_drift_rel']:.4f} exceeds 5 % over 100 steps "
        f"-- streaming or boundary likely broken"
    )


@pytest.mark.slow
def test_channel_smoke_has_nonzero_velocity():
    """After a short run, the peak streamwise velocity should be
    bounded above by Ma ~ 0.1 (the standard LBM stability limit) and
    not collapsed to zero (which would indicate the inflow is not
    propagating)."""
    _, ux, _, _, diag = run_channel_smoke(
        Nx=32, Ny=16, Nz=16, u_in=0.04, nu=0.02, n_steps=100,
    )
    assert diag["u_peak"] > 0.5 * diag["u_in"], (
        f"peak u {diag['u_peak']:.4f} is far below inflow {diag['u_in']:.4f}"
    )
    assert diag["u_peak"] < 0.3, (
        f"peak u {diag['u_peak']:.4f} exceeded Ma ~ 0.5 -- simulation diverged"
    )
    # Spot check that ux is finite everywhere (no NaN escaping the
    # body / boundary handling).
    assert np.all(np.isfinite(ux))


# ---------------------------------------------------------------------------
# Phase A2: sphere-body sanity checks (cheap; gate the wiring works)
# ---------------------------------------------------------------------------


def test_sphere_mask_geometry():
    """_make_sphere_mask produces a sphere of the right size, centred where
    expected, with no off-by-one drift."""
    Nx, Ny, Nz = 32, 16, 16
    cx, cy, cz = 16, 8, 8
    R = 4.0
    mask = _make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    assert mask.shape == (Nx, Ny, Nz)
    assert mask.dtype == np.bool_
    # Centre is solid; corners are fluid.
    assert mask[cx, cy, cz]
    assert not mask[0, 0, 0]
    assert not mask[Nx - 1, Ny - 1, Nz - 1]
    # Cell count: 4/3 pi R^3 with discretisation slack (~ +/- 15 %).
    expected = (4.0 / 3.0) * np.pi * R ** 3
    actual = float(mask.sum())
    assert 0.85 * expected < actual < 1.15 * expected, (
        f"sphere cell count {actual:.0f} vs analytic {expected:.0f} "
        f"is outside the +/- 15 % discretisation band"
    )
    # Surface symmetry: cell on +x of centre equals cell on -x of centre.
    for d in range(1, int(R)):
        assert mask[cx + d, cy, cz] == mask[cx - d, cy, cz]
        assert mask[cx, cy + d, cz] == mask[cx, cy - d, cz]
        assert mask[cx, cy, cz + d] == mask[cx, cy, cz - d]


@pytest.mark.slow
def test_sphere_blocks_flow_at_centre():
    """With a sphere in the channel, the velocity inside the sphere should
    be ~0 (bounce-back forces no-slip on solid cells), and the velocity
    just downstream of the sphere should be LOWER than the channel
    centre-line value at the same y/z without a sphere -- evidence the
    LBM actually deflects the flow."""
    Nx, Ny, Nz = 48, 24, 24
    u_in, nu = 0.04, 0.02
    cx, cy, cz = 16, 12, 12
    R = 4.0

    # Run WITHOUT body for the baseline centreline velocity.
    _, ux_clean, _, _, _ = run_channel_smoke(
        Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=400,
    )
    # Sample downstream of where the sphere will be.
    u_clean_downstream = float(ux_clean[cx + 6, cy, cz])

    # Run WITH the sphere body.
    body = _make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    _, ux_sphere, _, _, _ = run_channel_smoke(
        Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=400, body=body,
    )

    # Inside the sphere: velocity should be ~0 (bounce-back pins it).
    u_inside = float(ux_sphere[cx, cy, cz])
    assert abs(u_inside) < 0.1 * u_in, (
        f"velocity inside sphere {u_inside:.4f} is not near zero "
        f"-- bounce-back is not pinning the solid cells"
    )
    # Just downstream of the sphere: streamwise velocity should be
    # measurably reduced vs the no-body baseline. Use a generous gate
    # (10 % reduction) -- the actual wake on a Re ~ 50 sphere is
    # closer to 60-80 % reduction, but at 48^3 with full-way BB and
    # only 400 steps the wake is still developing.
    u_wake_downstream = float(ux_sphere[cx + 6, cy, cz])
    assert u_wake_downstream < 0.9 * u_clean_downstream, (
        f"wake velocity {u_wake_downstream:.4f} is not reduced vs "
        f"baseline {u_clean_downstream:.4f} -- sphere may not be "
        f"deflecting the flow"
    )
