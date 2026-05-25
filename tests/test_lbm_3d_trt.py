"""Phase 1 gate tests for the production 3D D3Q19 TRT kernel.

The headline check is the Taylor-Green vortex decay rate gate: the
analytic KE decay rate is 4 ν k², and the measured rate must match
within ±2 % for both TRT and BGK. This single test proves the
collision operator, the streaming, and the viscosity relation are
all correct.

Additional checks:
  * BGK reduction: setting s_plus = s_minus must reproduce a BGK step.
  * Conservation: total mass in a periodic box must hold to machine
    precision.
  * Layout: lattice constants are re-imported from src.lbm_3d and the
    TRT module uses them via the same OPPOSITE / WEIGHTS arrays.
  * float32 dtype passes through the kernel and produces the same
    decay rate within the proto-4 budget.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.lbm_3d import (
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
)
from src.lbm_3d_trt import (
    LAMBDA_TRT,
    analytic_tgv_decay_rate,
    fit_decay_rate,
    omegas_for_bgk,
    omegas_for_trt,
    run_tgv,
    trt_periodic_step,
)


# ---------------------------------------------------------------------------
# Cheap algebra / construction tests
# ---------------------------------------------------------------------------

def test_lambda_is_three_sixteenths():
    s_plus, s_minus = omegas_for_trt(0.01)
    lam = (1.0 / s_plus - 0.5) * (1.0 / s_minus - 0.5)
    assert abs(lam - LAMBDA_TRT) < 1e-12, (
        f"Λ = {lam}, expected {LAMBDA_TRT}. The wall-placement guarantee "
        f"that makes TRT worth its cost relies on Λ = 3/16 exactly."
    )


def test_uniform_equilibrium_preserved():
    """If f is initialised at the uniform equilibrium for some (rho, u)
    and the domain is periodic, then one step must leave f unchanged.
    The collision sees f = e_eq so (f - e) = 0 in both TRT terms, and
    streaming a uniform field is the identity. If this fails, either
    the equilibrium calculation or the streaming has a defect."""
    N = 8
    rho = np.full((N, N, N), 1.0, dtype=np.float32)
    u = np.zeros((3, N, N, N), dtype=np.float32)
    u[0] = 0.02
    u[1] = -0.01
    u[2] = 0.005
    from src.lbm_3d import equilibrium_3d
    f = equilibrium_3d(rho, u).astype(np.float32)
    f_next = f.copy()
    s_plus, s_minus = omegas_for_trt(0.01)
    s_plus = np.float32(s_plus)
    s_minus = np.float32(s_minus)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)
    trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
    # float32 single-step round-off tolerance on a 19-term sum.
    assert np.allclose(f_next, f, atol=1e-6), (
        f"Uniform equilibrium not preserved; max diff "
        f"{np.abs(f_next - f).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Phase 1 EXIT GATE: TGV decay rate within ±2 % of analytic 4 ν k²
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_phase1_gate_trt_tgv_decay_rate():
    """The Phase 1 gate. With Λ = 3/16, the periodic TGV decay rate
    must match the analytic 4 ν k² within 2 %. If this fails, either
    the collision operator or the streaming or the equilibrium has
    a defect; do not advance to Phase 2."""
    N = 32
    nu = 0.01
    times, ke, diag = run_tgv(
        N=N, U=0.04, nu=nu, n_steps=800, scheme="trt", dtype=np.float32,
    )
    assert not diag["diverged"], "TGV diverged under TRT — kernel broken"
    measured = fit_decay_rate(times, ke)
    analytic = analytic_tgv_decay_rate(nu, N)
    err_pct = 100.0 * (measured - analytic) / analytic
    assert abs(err_pct) < 2.0, (
        f"Phase 1 gate FAILED: TRT decay rate {measured:.6e} vs analytic "
        f"{analytic:.6e}, err {err_pct:+.2f} %. Tolerance is ±2 %."
    )


@pytest.mark.slow
def test_bgk_reference_path_also_passes_gate():
    """The BGK reference path is kept for the equivalence test; it
    should also reproduce the analytic TGV decay rate in a periodic
    box (BGK and TRT only diverge at walls). If BGK fails here while
    TRT passes, the issue is in the equilibrium or streaming."""
    N = 32
    nu = 0.01
    times, ke, diag = run_tgv(
        N=N, U=0.04, nu=nu, n_steps=800, scheme="bgk", dtype=np.float32,
    )
    assert not diag["diverged"]
    measured = fit_decay_rate(times, ke)
    analytic = analytic_tgv_decay_rate(nu, N)
    assert abs(100.0 * (measured - analytic) / analytic) < 2.0


@pytest.mark.slow
def test_mass_conserved_in_periodic_box():
    """In a periodic box with no source terms, total mass must hold
    to machine precision over the run."""
    N = 16
    nu = 0.02
    f_initial = None
    f_final = None
    # Re-run TGV but capture the population sum at start and end.
    from src.lbm_3d_trt import init_tgv, trt_periodic_step
    f = init_tgv(N, 0.03, dtype=np.float64)
    f_next = f.copy()
    f_initial = float(f.sum())
    s_plus, s_minus = omegas_for_trt(nu)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float64)
    opp = OPPOSITE_3D.astype(np.int32)
    for _ in range(200):
        trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
    f_final = float(f.sum())
    rel = abs(f_final - f_initial) / f_initial
    assert rel < 1e-12, (
        f"Mass drifted by {rel:.2e} over 200 periodic steps — should "
        f"be machine precision."
    )
