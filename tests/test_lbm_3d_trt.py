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
        f"Lambda = {lam}, expected {LAMBDA_TRT}. The wall-placement "
        f"guarantee that makes TRT worth its cost relies on Lambda = "
        f"3/16 exactly."
    )


def test_s_plus_equals_one_over_tau_exactly():
    """Reviewer 2026-05-28 P3 sanity: s_plus must equal 1/tau exactly,
    i.e. 1/s_plus = 3 nu + 1/2. A bug here would silently make the
    kernel compute the wrong viscosity, with the TGV decay rate
    masking it (because the rate depends on the actual viscosity the
    kernel implements, not the nu the user thinks they set).
    """
    for nu in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        s_plus, _ = omegas_for_trt(nu)
        tau = 3.0 * nu + 0.5
        # exact float arithmetic: 1/s_plus must equal tau within
        # a few ulps.
        assert abs(1.0 / s_plus - tau) < 1e-15 * tau, (
            f"nu={nu}: 1/s_plus = {1.0/s_plus}, expected tau = {tau}"
        )


def test_trt_split_is_non_degenerate():
    """Reviewer 2026-05-28 P3 sanity: confirm the symmetric /
    antisymmetric split is actually doing different work on the two
    parts. Run two cases that differ only in (s_plus, s_minus) and
    verify the results differ when the initial f has BOTH symmetric
    AND antisymmetric off-equilibrium content. If the kernel
    collapsed the split (e.g. silently used s_plus for both), the
    two cases would produce identical output.
    """
    N = 6
    rho = np.full((N, N, N), 1.0, dtype=np.float32)
    u = np.zeros((3, N, N, N), dtype=np.float32)
    u[0] = 0.02
    from src.lbm_3d import equilibrium_3d
    f_eq = equilibrium_3d(rho, u).astype(np.float32)
    # Asymmetric perturbation: +1e-3 on i=1 (east) and -1e-3 on
    # i=2 (west) creates pure antisymmetric content (f^+ = 0,
    # f^- != 0). +1e-3 on both i=3 (north) and i=4 (south) creates
    # pure symmetric content (f^+ != 0, f^- = 0).
    f = f_eq.copy()
    f[1] += np.float32(1e-3)
    f[2] -= np.float32(1e-3)
    f[3] += np.float32(1e-3)
    f[4] += np.float32(1e-3)
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)

    # Case A: TRT with separated rates (s_plus != s_minus).
    f_a = f.copy()
    f_next_a = f_a.copy()
    s_plus_a, s_minus_a = omegas_for_trt(0.01)
    trt_periodic_step(f_a, f_next_a, np.float32(s_plus_a),
                       np.float32(s_minus_a), vel, weights, opp)

    # Case B: same s_plus but flipped s_minus (use the BGK value
    # of s_minus instead of the TRT one).
    f_b = f.copy()
    f_next_b = f_b.copy()
    s_plus_b, _ = omegas_for_bgk(0.01)
    s_minus_b = np.float32(s_plus_b)  # forced BGK collapse
    trt_periodic_step(f_b, f_next_b, np.float32(s_plus_b), s_minus_b,
                       vel, weights, opp)

    diff = float(np.abs(f_next_a - f_next_b).max())
    # If the split were collapsed, both runs would give identical
    # results. A meaningful spread (much larger than float32 noise)
    # confirms s_minus is wired through the antisymmetric part.
    assert diff > 1e-5, (
        f"TRT and BGK-via-collapsed-split gave indistinguishable "
        f"results (max diff {diff:.2e}); the antisymmetric split "
        f"is not actually using s_minus."
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


# ---------------------------------------------------------------------------
# Channel-flow TRT driver (run_channel_smoke_trt)
#
# The kernels are unit-tested above and via the periodic TGV decay gate.
# These tests pin the channel composition: TRT collision + full-way wall
# BB + body BB + Guo NEEM inflow/outflow + optional Bouzidi correction.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_trt_channel_runs_finite_and_mass_bounded():
    """A 100-step TRT channel run produces finite output and bounded
    mass drift. Same gate as the BGK channel (< 5 % over 100 steps).
    """
    from src.lbm_3d_trt import run_channel_smoke_trt
    _, ux, _, _, diag = run_channel_smoke_trt(
        Nx=32, Ny=16, Nz=16, u_in=0.04, nu=0.02, n_steps=100,
    )
    assert np.all(np.isfinite(ux)), "TRT channel produced NaN/Inf"
    assert abs(diag["mass_drift_rel"]) < 0.05, (
        f"TRT mass drift {diag['mass_drift_rel']:.4f} exceeds 5 % "
        f"over 100 steps -- channel driver likely broken"
    )
    assert diag["u_peak"] > 0.5 * diag["u_in"], (
        f"TRT peak u {diag['u_peak']:.4f} far below inflow "
        f"{diag['u_in']:.4f} -- flow isn't developing"
    )
    assert diag["u_peak"] < 0.3, (
        f"TRT peak u {diag['u_peak']:.4f} exceeded Ma ~ 0.5 -- diverged"
    )


@pytest.mark.slow
def test_trt_channel_matches_bgk_at_unit_lambda():
    """The strongest invariant for the TRT channel: with
    ``scheme="bgk"`` (which sets s_plus = s_minus = omega), the
    TRT channel must reproduce the proven BGK pure-bulk + Guo NEEM
    path to float32 precision.

    At s_plus = s_minus = omega the TRT collision split reduces to
    BGK exactly (algebraic identity). Combined with the same boundary
    handling and the same (collision-agnostic) Guo NEEM passes, the
    full velocity field must agree.
    """
    from src.lbm_3d import run_channel_smoke
    from src.lbm_3d_trt import run_channel_smoke_trt
    args = dict(Nx=24, Ny=12, Nz=12, u_in=0.04, nu=0.02, n_steps=60)
    _, ux_bgk, uy_bgk, uz_bgk, _ = run_channel_smoke(**args, use_guo_neem=True)
    _, ux_trt, uy_trt, uz_trt, _ = run_channel_smoke_trt(**args, scheme="bgk")
    # Float32 accumulation differences are O(1e-7); 1e-5 is a tight pin.
    max_diff = max(
        float(np.max(np.abs(ux_trt - ux_bgk))),
        float(np.max(np.abs(uy_trt - uy_bgk))),
        float(np.max(np.abs(uz_trt - uz_bgk))),
    )
    assert max_diff < 1e-5, (
        f"TRT channel with s_plus = s_minus = omega should reduce to "
        f"BGK pure-bulk + Guo NEEM, but max abs diff is {max_diff:.2e}"
    )


@pytest.mark.slow
def test_trt_channel_sphere_wake_reduction():
    """A sphere placed in the TRT channel produces a measurable
    velocity reduction downstream, same direction as the BGK channel.
    This catches a kernel that silently bypasses the body mask.
    """
    from src.lbm_3d_bouzidi import make_sphere_mask, sphere_wall_links
    from src.lbm_3d_trt import run_channel_smoke_trt
    Nx, Ny, Nz = 48, 24, 24
    u_in, nu = 0.04, 0.02
    cx, cy, cz = 16, 12, 12
    R = 4.0

    # No-body baseline.
    _, ux_clean, _, _, _ = run_channel_smoke_trt(
        Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=400,
    )
    u_clean_downstream = float(ux_clean[cx + 6, cy, cz])

    body = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    wall_links = sphere_wall_links(Nx, Ny, Nz, float(cx), float(cy), float(cz), R)
    _, ux_sphere, _, _, _ = run_channel_smoke_trt(
        Nx=Nx, Ny=Ny, Nz=Nz, u_in=u_in, nu=nu, n_steps=400,
        body=body, wall_links=wall_links,
    )

    # Velocity inside sphere ~ 0.
    u_inside = float(ux_sphere[cx, cy, cz])
    assert abs(u_inside) < 0.1 * u_in, (
        f"TRT velocity inside sphere {u_inside:.4f} is not near zero "
        f"-- body bounce-back not pinning solid cells"
    )
    # Downstream velocity measurably reduced (~10 % gate, generous --
    # at this Re and grid the wake is still developing).
    u_wake = float(ux_sphere[cx + 6, cy, cz])
    assert u_wake < 0.9 * u_clean_downstream, (
        f"TRT wake velocity {u_wake:.4f} is not reduced vs baseline "
        f"{u_clean_downstream:.4f} -- sphere not deflecting flow"
    )


@pytest.mark.slow
def test_trt_channel_invalid_scheme_raises():
    """``scheme`` only accepts 'trt' and 'bgk'; anything else is a
    typo and we'd rather fail fast than silently pick one.
    """
    from src.lbm_3d_trt import run_channel_smoke_trt
    with pytest.raises(ValueError, match="unknown scheme"):
        run_channel_smoke_trt(
            Nx=12, Ny=8, Nz=8, n_steps=2, scheme="srt",
        )
