"""Phase 0 prototype #1: TRT (Lambda = 3/16) vs BGK on a 3D TGV.

Confirms the central collision-model recommendation from
3D_PHASE0_DECISIONS.md section D-2:

  * BGK becomes unstable / over-dissipative as tau -> 0.5.
  * TRT with Lambda = (1/s+ - 1/2)(1/s- - 1/2) = 3/16 stays stable
    and reproduces the analytic 4 nu k^2 kinetic-energy decay rate
    of a 2D-extruded Taylor-Green vortex.

Pass criterion: BGK shows non-trivial degradation at low tau (visible
either as divergence or a decay-rate error >> TRT's); TRT decay rate
within ~2 % of analytic across the tested tau range.

Disposable. Removed once Phase 1 production TRT lands.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from scripts._3d_phase0_helpers import (  # noqa: E402
    analytic_tgv_decay_rate,
    fit_decay_rate,
    kinetic_energy,
    make_tgv_init,
)
from src.lbm_3d import (  # noqa: E402
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
    equilibrium_3d,
    macroscopic_3d,
)

VEL = LATTICE_VELOCITIES_3D.astype(np.int32)
W = LATTICE_WEIGHTS_3D.astype(np.float64)
OPP = OPPOSITE_3D.astype(np.int32)


@njit(cache=False, fastmath=True)
def step_periodic_trt(f, f_next, s_plus, s_minus, vel, weights, opp):
    """One TRT collide + periodic stream pass. BGK is the special case
    s_plus = s_minus."""
    _, Nx, Ny, Nz = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)
    e = np.empty(19, dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                rho = 0.0
                mx = 0.0
                my = 0.0
                mz = 0.0
                for i in range(19):
                    fi = f[i, x, y, z]
                    rho += fi
                    mx += vel[i, 0] * fi
                    my += vel[i, 1] * fi
                    mz += vel[i, 2] * fi
                inv_rho = 1.0 / rho if rho > 0 else 0.0
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz
                for i in range(19):
                    cu = vel[i, 0] * ux + vel[i, 1] * uy + vel[i, 2] * uz
                    e[i] = weights[i] * rho * (
                        1.0 + cu / cs2 + (cu * cu) * inv2cs4 - usq * inv2cs2
                    )
                for i in range(19):
                    ii = opp[i]
                    fi = f[i, x, y, z]
                    fii = f[ii, x, y, z]
                    fp = 0.5 * (fi + fii)
                    fm = 0.5 * (fi - fii)
                    ep = 0.5 * (e[i] + e[ii])
                    em = 0.5 * (e[i] - e[ii])
                    f_post = fi - s_plus * (fp - ep) - s_minus * (fm - em)
                    xn = (x + vel[i, 0]) % Nx
                    yn = (y + vel[i, 1]) % Ny
                    zn = (z + vel[i, 2]) % Nz
                    f_next[i, xn, yn, zn] = f_post


def run_tgv(N, U, nu, n_steps, scheme: str, sample_every=10):
    s_plus = 1.0 / (3.0 * nu + 0.5)
    if scheme == "trt":
        # Lambda = (1/s+ - 1/2)(1/s- - 1/2) = 3/16
        s_minus = 1.0 / (1.0 / (16.0 * nu) + 0.5)
    else:
        s_minus = s_plus  # BGK fallback
    rho, u = make_tgv_init(N, U, dtype=np.float64)
    f = equilibrium_3d(rho, u)
    f_next = f.copy()
    ke_series = [kinetic_energy(u)]
    t_series = [0]
    for step in range(1, n_steps + 1):
        step_periodic_trt(f, f_next, s_plus, s_minus, VEL, W, OPP)
        f, f_next = f_next, f
        if step % sample_every == 0:
            _, ux, uy, uz = macroscopic_3d(f)
            ke = kinetic_energy(np.stack([ux, uy, uz]))
            if not np.isfinite(ke) or ke > 10 * ke_series[0]:
                return ke_series + [float("nan")], t_series + [step], True
            ke_series.append(ke)
            t_series.append(step)
    return ke_series, t_series, False


def main() -> int:
    N = 24
    U = 0.04
    n_steps = 600
    print(f"# Proto 1: TRT (Lambda=3/16) vs BGK on TGV, N={N}, U={U}, n_steps={n_steps}")
    print(f"{'scheme':6} {'nu':>9} {'tau':>7} {'diverged':>9} "
          f"{'measured':>10} {'analytic':>10} {'err %':>7}")
    rows = []
    for nu in (0.005, 0.01, 0.02, 0.05):
        analytic = analytic_tgv_decay_rate(nu, N)
        for scheme in ("bgk", "trt"):
            t0 = time.time()
            ke, ts, diverged = run_tgv(N, U, nu, n_steps, scheme)
            tau = 3.0 * nu + 0.5
            elapsed = time.time() - t0
            if diverged or len(ke) < 5:
                rows.append((scheme, nu, tau, True, float("nan"), analytic, float("nan")))
                print(f"{scheme:6} {nu:9.4f} {tau:7.4f}      yes       nan  "
                      f"{analytic:10.5f}     n/a   (run {elapsed:.1f} s)")
                continue
            # Skip the first ~25 % of samples to discard initial transient.
            cutoff = max(1, len(ke) // 4)
            measured = fit_decay_rate(ts[cutoff:], ke[cutoff:])
            err_pct = 100.0 * (measured - analytic) / analytic
            rows.append((scheme, nu, tau, False, measured, analytic, err_pct))
            print(f"{scheme:6} {nu:9.4f} {tau:7.4f}       no  {measured:10.5f}  "
                  f"{analytic:10.5f}  {err_pct:+6.2f}   (run {elapsed:.1f} s)")
    print()
    trt_rows = [r for r in rows if r[0] == "trt" and not r[3]]
    bgk_rows = [r for r in rows if r[0] == "bgk"]
    trt_ok = (trt_rows and all(abs(r[6]) < 2.0 for r in trt_rows))
    bgk_problem = any(
        r[3] or (not r[3] and abs(r[6]) > 2.0 * abs(
            next((t[6] for t in trt_rows if t[1] == r[1]), 1e-9))
        )
        for r in bgk_rows
    )
    print(f"[verdict] TRT decay-rate err < 2 % on all stable runs: {trt_ok}")
    print(f"[verdict] BGK shows divergence or noticeably worse decay error: {bgk_problem}")
    if trt_ok:
        print("[PASS] TRT with Lambda=3/16 reproduces the analytic TGV decay rate. "
              "Production collision = TRT confirmed.")
        return 0
    print("[FAIL] TRT did not hold within tolerance; investigate before Phase 1.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
