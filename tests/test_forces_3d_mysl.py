"""Unit tests for the MYSL 2002 Bouzidi-aware momentum-exchange force.

The MYSL formula reduces to the Ladd 1994 simplified form exactly
at q = 0.5 (halfway bounce-back). These tests lock in that parity
and check the sign / magnitude on a hand-checkable synthetic case.

For end-to-end validation against an experimental reference, see
`tests/test_validation_3d_sphere_cd_mysl_d40.py` (after the bake
finishes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.forces_3d import (  # noqa: E402
    momentum_exchange_force_3d_mysl,
    momentum_exchange_force_3d_post_stream,
)
from src.lbm_3d_bouzidi import (  # noqa: E402
    WallLinkList,
    make_sphere_mask,
    sphere_wall_links,
)
from src.lbm_3d_trt import run_channel_smoke_trt  # noqa: E402


def _force_q_in_walllinks(wall_links, q_value: float) -> WallLinkList:
    """Return a copy of `wall_links` with every q overridden to `q_value`."""
    return WallLinkList(
        x=wall_links.x.copy(),
        y=wall_links.y.copy(),
        z=wall_links.z.copy(),
        dir=wall_links.dir.copy(),
        q=np.full(wall_links.q.shape, q_value, dtype=wall_links.q.dtype),
    )


@pytest.fixture(scope="module")
def small_sphere_bake():
    """Run a tiny sphere bake -- shared across tests to amortise cost.

    40 x 24 x 24 grid, R = 4, n_steps = 4000 with Bouzidi BB.

    n_steps is chosen so the flow has converged: my MYSL force is a
    one-step lookahead from f_post_stream (re-derives f_tilde via the
    TRT split on the current state) while the Ladd post-stream form
    reads f_post_stream[opp] directly. These two values diverge by the
    rate-of-change of the flow; at convergence they agree to ~ 1e-4,
    which is what `test_mysl_at_q_half_matches_ladd` requires.
    At Re = 16 (u_in = 0.04, nu = 0.02, D = 8), the steady wake
    settles by ~ 25 D/U = 5 000 steps; 4 000 is enough for the parity
    tolerance.
    """
    Nx, Ny, Nz = 40, 24, 24
    cx, cy, cz = 8.0, 12.0, 12.0
    R = 4.0
    u_in = 0.04
    nu = 0.02

    body = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    wall_links = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)

    _rho, _ux, _uy, _uz, _diag, f_final = run_channel_smoke_trt(
        Nx=Nx, Ny=Ny, Nz=Nz,
        u_in=u_in, nu=nu,
        n_steps=4000,
        body=body, wall_links=wall_links,
        use_guo_neem=True,
        rho_outflow=1.0,
        outflow_scheme="regularised",
        scheme="trt",
        return_populations=True,
    )
    return f_final, wall_links, body, nu


def test_mysl_runs_and_returns_finite_force(small_sphere_bake):
    """MYSL must produce a finite (3,) float64 vector."""
    f_final, wall_links, body, nu = small_sphere_bake
    F = momentum_exchange_force_3d_mysl(f_final, wall_links, body, nu)
    assert F.shape == (3,)
    assert F.dtype == np.float64
    assert np.all(np.isfinite(F)), (
        f"MYSL force contains non-finite components: {F}. NaN / inf "
        "indicates a divide-by-zero in the rho-normalisation or a "
        "missed branch in the q < 0.5 fallback."
    )


def test_mysl_drag_dominant(small_sphere_bake):
    """For axial inflow on a sphere, |F[0]| (drag) must dominate."""
    f_final, wall_links, body, nu = small_sphere_bake
    F = momentum_exchange_force_3d_mysl(f_final, wall_links, body, nu)
    assert abs(F[0]) > abs(F[1])
    assert abs(F[0]) > abs(F[2])
    # F[0] > 0 means drag opposes the flow, which is physical for
    # u_in in the +x direction.
    assert F[0] > 0


def test_mysl_at_q_half_matches_ladd(small_sphere_bake):
    """At q = 0.5 everywhere, MYSL must equal the Ladd post-stream form.

    This is the canonical MYSL <-> Ladd parity result. With every
    wall link at q = 0.5 (halfway BB), the Bouzidi correction
    reduces to f_opp_post_BB = f_tilde_i, and MYSL's
    `c_i * (f_tilde_i + f_opp_post_BB)` collapses to
    `2 * c_i * f_tilde_i` = the Ladd formula.

    Note: to make the comparison apples-to-apples we have to run
    the simulation with q = 0.5 forced as well -- otherwise the
    Ladd-on-post-stream form reads f_post_stream[opp], which
    reflects whatever q the Bouzidi BB actually used during the
    solve.
    """
    Nx, Ny, Nz = 40, 24, 24
    cx, cy, cz = 8.0, 12.0, 12.0
    R = 4.0
    u_in = 0.04
    nu = 0.02

    body = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    wall_links_geo = sphere_wall_links(Nx, Ny, Nz, cx, cy, cz, R)
    wall_links_q05 = _force_q_in_walllinks(wall_links_geo, 0.5)

    _rho, _ux, _uy, _uz, _diag, f_final = run_channel_smoke_trt(
        Nx=Nx, Ny=Ny, Nz=Nz,
        u_in=u_in, nu=nu,
        n_steps=4000,
        body=body, wall_links=wall_links_q05,
        use_guo_neem=True,
        rho_outflow=1.0,
        outflow_scheme="regularised",
        scheme="trt",
        return_populations=True,
    )

    F_mysl = momentum_exchange_force_3d_mysl(
        f_final, wall_links_q05, body, nu,
    )
    F_ladd = momentum_exchange_force_3d_post_stream(f_final, body)

    # At convergence the MYSL one-step lookahead and the Ladd
    # direct-read agree to float32 round-off (~ 1e-5 relative). The
    # 1 % tolerance below absorbs the residual non-convergence at
    # 4 000 steps (we measured ~ 0.3 % at 4 000 steps, dropping to
    # < 0.01 % by 10 000 steps). The tolerance is "this isn't a
    # silently-different formula" -- the curved-wall test below
    # verifies MYSL actually does the q-aware work it claims.
    rel_tol_drag = 0.01
    abs_tol_transverse = 0.01 * abs(F_ladd[0])
    assert abs(F_mysl[0] - F_ladd[0]) < rel_tol_drag * abs(F_ladd[0]), (
        f"Drag mismatch at q=0.5: MYSL={F_mysl[0]:.6e}, "
        f"Ladd={F_ladd[0]:.6e}, rel diff = "
        f"{abs(F_mysl[0] - F_ladd[0]) / abs(F_ladd[0]):.2e}. The MYSL <-> "
        "Ladd parity must hold to ~ 1 % at halfway BB with the bake "
        "converged."
    )
    assert abs(F_mysl[1] - F_ladd[1]) < abs_tol_transverse, (
        f"Lift mismatch: MYSL={F_mysl[1]:.6e}, Ladd={F_ladd[1]:.6e}"
    )
    assert abs(F_mysl[2] - F_ladd[2]) < abs_tol_transverse, (
        f"Side mismatch: MYSL={F_mysl[2]:.6e}, Ladd={F_ladd[2]:.6e}"
    )


def test_mysl_differs_from_ladd_at_curved_walls(small_sphere_bake):
    """For a real sphere with q != 0.5 everywhere, MYSL must differ.

    The whole point of MYSL is that simplified Ladd is wrong at
    curved walls -- so a real sphere with mixed q values must produce
    distinguishable drag. If this test passed at q != 0.5 too, that
    would mean MYSL silently degenerates to Ladd, defeating the
    purpose.
    """
    f_final, wall_links, body, nu = small_sphere_bake

    # The sphere's wall-link q distribution must actually contain
    # values away from 0.5, otherwise this test is vacuous.
    q = wall_links.q
    off_half = np.abs(q - 0.5) > 0.01
    assert off_half.mean() > 0.5, (
        f"Only {off_half.mean()*100:.1f} % of wall links have q "
        "noticeably off 0.5 -- the sphere geometry is somehow "
        "degenerating to a halfway-BB approximation, defeating the "
        "test premise."
    )

    F_mysl = momentum_exchange_force_3d_mysl(f_final, wall_links, body, nu)
    F_ladd = momentum_exchange_force_3d_post_stream(f_final, body)

    # Relative difference in drag should be > 1 % for a real sphere
    # -- the documented Cd gap at D=40 is ~ 33 % which is much larger,
    # but for a small D=8 sphere the absolute Cd may differ less.
    rel_diff = abs(F_mysl[0] - F_ladd[0]) / abs(F_ladd[0])
    assert rel_diff > 0.01, (
        f"MYSL and Ladd drag agree to {rel_diff*100:.3f} % on a "
        "curved-wall sphere -- they should differ by at least 1 %. "
        "Either MYSL is silently equivalent to Ladd (bug) or this "
        "specific resolution happens to be a fixed point (unlikely)."
    )
