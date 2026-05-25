"""Phase 0 prototype #2: Guo NEEM inflow/outflow mass closure.

Confirms the boundary-condition decision from 3D_PHASE0_DECISIONS.md
section D-3: the Guo, Zheng, Shi (2002) non-equilibrium extrapolation
method closes mass on a 3D empty channel, dodging the awkward 3D
edge-and-corner closures that Zou-He requires.

Setup: a 32 x 16 x 16 channel, periodic in y and z (so the only
non-periodic faces are x = 0 inflow and x = Nx - 1 outflow). Inflow
prescribed u = (U, 0, 0) at rho = 1; outflow prescribes rho = 1 and
extrapolates u from the interior neighbour. BGK collision at moderate
tau (well-conditioned for a closure test).

Pass criterion: the relative drift of total mass after 500 steps is
< 1 % (in 2D this kind of test is well under 0.1 %, so 1 % is a
loose-but-honest bar for a first 3D pass).

Disposable: when the Phase 1 production BC lands the throwaway
kernel here is deleted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from src.lbm_3d import (  # noqa: E402
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    equilibrium_3d,
)

VEL = LATTICE_VELOCITIES_3D.astype(np.int32)
W = LATTICE_WEIGHTS_3D.astype(np.float64)


@njit(cache=False, fastmath=True)
def step_guo_channel(f, f_next, omega, u_in, vel, weights):
    """Single BGK collide + stream + Guo NEEM inflow/outflow pass.

    Periodic in y and z so the only non-periodic closures are at
    x = 0 (velocity inflow) and x = Nx - 1 (pressure outflow).
    """
    _, Nx, Ny, Nz = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)
    e_local = np.empty(19, dtype=np.float64)
    # --- Collide + stream (periodic in all three for the moment;
    #     boundary closures applied next) ---
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
                    e_local[i] = weights[i] * rho * (
                        1.0 + cu / cs2 + (cu * cu) * inv2cs4 - usq * inv2cs2
                    )
                for i in range(19):
                    f_post = f[i, x, y, z] - omega * (f[i, x, y, z] - e_local[i])
                    xn = x + vel[i, 0]
                    yn = (y + vel[i, 1]) % Ny
                    zn = (z + vel[i, 2]) % Nz
                    if 0 <= xn < Nx:
                        f_next[i, xn, yn, zn] = f_post
                    # populations leaving via x get overwritten by the
                    # Guo NEEM step below; we drop them here.
    # --- Guo NEEM at x = 0 (inflow, prescribed velocity u_in) ---
    for y in range(Ny):
        for z in range(Nz):
            # Interior neighbour at x = 1
            rho_n = 0.0
            mxn = 0.0
            myn = 0.0
            mzn = 0.0
            for i in range(19):
                fn = f_next[i, 1, y, z]
                rho_n += fn
                mxn += vel[i, 0] * fn
                myn += vel[i, 1] * fn
                mzn += vel[i, 2] * fn
            inv_rn = 1.0 / rho_n if rho_n > 0 else 0.0
            un_x = mxn * inv_rn
            un_y = myn * inv_rn
            un_z = mzn * inv_rn
            # Boundary macroscopics: prescribed u = (u_in, 0, 0),
            # rho extrapolated from neighbour (no penetration of any
            # density wave from outside the domain).
            rho_b = rho_n
            ub_x = u_in
            ub_y = 0.0
            ub_z = 0.0
            usq_b = ub_x * ub_x + ub_y * ub_y + ub_z * ub_z
            usq_n = un_x * un_x + un_y * un_y + un_z * un_z
            for i in range(19):
                cu_b = vel[i, 0] * ub_x + vel[i, 1] * ub_y + vel[i, 2] * ub_z
                cu_n = vel[i, 0] * un_x + vel[i, 1] * un_y + vel[i, 2] * un_z
                e_b = weights[i] * rho_b * (
                    1.0 + cu_b / cs2 + (cu_b * cu_b) * inv2cs4 - usq_b * inv2cs2
                )
                e_n = weights[i] * rho_n * (
                    1.0 + cu_n / cs2 + (cu_n * cu_n) * inv2cs4 - usq_n * inv2cs2
                )
                f_next[i, 0, y, z] = e_b + (f_next[i, 1, y, z] - e_n)
    # --- Guo NEEM at x = Nx - 1 (outflow, prescribed rho = 1) ---
    for y in range(Ny):
        for z in range(Nz):
            xn = Nx - 2
            rho_n = 0.0
            mxn = 0.0
            myn = 0.0
            mzn = 0.0
            for i in range(19):
                fn = f_next[i, xn, y, z]
                rho_n += fn
                mxn += vel[i, 0] * fn
                myn += vel[i, 1] * fn
                mzn += vel[i, 2] * fn
            inv_rn = 1.0 / rho_n if rho_n > 0 else 0.0
            un_x = mxn * inv_rn
            un_y = myn * inv_rn
            un_z = mzn * inv_rn
            rho_b = 1.0
            ub_x = un_x
            ub_y = un_y
            ub_z = un_z
            usq_b = ub_x * ub_x + ub_y * ub_y + ub_z * ub_z
            usq_n = un_x * un_x + un_y * un_y + un_z * un_z
            for i in range(19):
                cu_b = vel[i, 0] * ub_x + vel[i, 1] * ub_y + vel[i, 2] * ub_z
                cu_n = vel[i, 0] * un_x + vel[i, 1] * un_y + vel[i, 2] * un_z
                e_b = weights[i] * rho_b * (
                    1.0 + cu_b / cs2 + (cu_b * cu_b) * inv2cs4 - usq_b * inv2cs2
                )
                e_n = weights[i] * rho_n * (
                    1.0 + cu_n / cs2 + (cu_n * cu_n) * inv2cs4 - usq_n * inv2cs2
                )
                f_next[i, Nx - 1, y, z] = e_b + (f_next[i, xn, y, z] - e_n)


def main() -> int:
    Nx, Ny, Nz = 32, 16, 16
    U = 0.04
    nu = 0.02
    n_steps = 500
    omega = 1.0 / (3.0 * nu + 0.5)
    rho = np.ones((Nx, Ny, Nz), dtype=np.float64)
    u = np.zeros((3, Nx, Ny, Nz), dtype=np.float64)
    u[0] = U
    f = equilibrium_3d(rho, u)
    f_next = f.copy()
    mass_initial = float(f.sum())
    print(f"# Proto 2: Guo NEEM mass closure, grid={Nx}x{Ny}x{Nz}, U={U}, nu={nu}, "
          f"n_steps={n_steps}")
    print(f"# omega = {omega:.4f}, mass_initial = {mass_initial:.4f}")
    t0 = time.time()
    samples = []
    for step in range(1, n_steps + 1):
        step_guo_channel(f, f_next, omega, U, VEL, W)
        f, f_next = f_next, f
        if step % 50 == 0:
            m = float(f.sum())
            samples.append((step, m, (m - mass_initial) / mass_initial))
    elapsed = time.time() - t0
    print(f"{'step':>6} {'mass':>14} {'drift_rel %':>12}")
    for step, m, drift in samples:
        print(f"{step:>6} {m:>14.4f} {drift*100:>+12.5f}")
    final_drift = samples[-1][2]
    print(f"\n[time] {elapsed:.2f} s for {n_steps} steps "
          f"({n_steps/elapsed:.1f} steps/s)")
    print(f"[verdict] final mass drift = {final_drift*100:+.4f} %")
    if abs(final_drift) < 0.01:
        print("[PASS] Guo NEEM closes mass within 1 % on the empty channel. "
              "Production inflow/outflow = Guo NEEM confirmed.")
        return 0
    print("[FAIL] Mass drift exceeds 1 %; investigate before Phase 1. "
          "Fallback: regularized BC (Latt & Chopard 2006).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
