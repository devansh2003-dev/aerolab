"""Phase 0 prototype #4: float32 vs float64 accuracy on the TGV decay rate.

Confirms the float32 decision from 3D_PHASE0_DECISIONS.md section D-1:
the population array is float32 throughout. The justification is that
LBM's incompressible-flow recovery is second-order in Mach and float32
gives ~7 decimal digits, which at Ma ~ 0.07 leaves ample headroom.
This proto puts a number on "ample" by measuring the TGV decay rate
at both dtypes against the analytic 4 nu k^2.

Pass criterion: float32 decay rate within 1 % of the float64 rate
(both should sit close to the analytic; a 1 % spread between dtypes
means float32 is not introducing dtype-induced error materially
above the discretisation error).
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
    equilibrium_3d,
    macroscopic_3d,
)

VEL = LATTICE_VELOCITIES_3D.astype(np.int32)


@njit(cache=False, fastmath=True)
def step_bgk_periodic_f32(f, f_next, omega, vel, weights):
    """BGK in a periodic box, dtype-agnostic via Numba's type
    inference on f.dtype. Passing a float32 f gives a float32 kernel."""
    _, Nx, Ny, Nz = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)
    e_local = np.empty(19, dtype=f.dtype)
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                rho = f.dtype.type(0.0)
                mx = f.dtype.type(0.0)
                my = f.dtype.type(0.0)
                mz = f.dtype.type(0.0)
                for i in range(19):
                    fi = f[i, x, y, z]
                    rho += fi
                    mx += vel[i, 0] * fi
                    my += vel[i, 1] * fi
                    mz += vel[i, 2] * fi
                inv_rho = f.dtype.type(1.0) / rho if rho > 0 else f.dtype.type(0.0)
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
                    xn = (x + vel[i, 0]) % Nx
                    yn = (y + vel[i, 1]) % Ny
                    zn = (z + vel[i, 2]) % Nz
                    f_next[i, xn, yn, zn] = f_post


def run_tgv_at_dtype(dtype, N, U, nu, n_steps, sample_every=10):
    omega = dtype(1.0 / (3.0 * nu + 0.5))
    rho, u = make_tgv_init(N, U, dtype=dtype)
    f = equilibrium_3d(rho, u).astype(dtype)
    f_next = f.copy()
    weights = LATTICE_WEIGHTS_3D.astype(dtype)
    ke_series = [kinetic_energy(u)]
    t_series = [0]
    for step in range(1, n_steps + 1):
        step_bgk_periodic_f32(f, f_next, omega, VEL, weights)
        f, f_next = f_next, f
        if step % sample_every == 0:
            _, ux, uy, uz = macroscopic_3d(f)
            ke = kinetic_energy(np.stack([ux, uy, uz]))
            ke_series.append(ke)
            t_series.append(step)
    cutoff = max(1, len(ke_series) // 4)
    rate = fit_decay_rate(t_series[cutoff:], ke_series[cutoff:])
    return rate


def main() -> int:
    N = 24
    U = 0.04
    nu = 0.01
    n_steps = 600
    print(f"# Proto 4: float32 vs float64, TGV N={N}, U={U}, nu={nu}, "
          f"n_steps={n_steps}")
    analytic = analytic_tgv_decay_rate(nu, N)
    rows = []
    for dt in (np.float64, np.float32):
        t0 = time.time()
        rate = run_tgv_at_dtype(dt, N, U, nu, n_steps)
        elapsed = time.time() - t0
        err_pct = 100.0 * (rate - analytic) / analytic
        rows.append((dt.__name__, rate, err_pct, elapsed))
        print(f"  {dt.__name__:>7}  measured={rate:.6f}  analytic={analytic:.6f}  "
              f"err={err_pct:+6.2f} %  ({elapsed:.1f} s)")
    r64 = rows[0][1]
    r32 = rows[1][1]
    spread_pct = 100.0 * abs(r32 - r64) / r64
    print()
    print(f"[verdict] f32 vs f64 spread = {spread_pct:.4f} %")
    if spread_pct < 1.0:
        print(f"[PASS] float32 reproduces the float64 decay rate to within 1 %. "
              f"Production dtype = float32 confirmed (memory halved at "
              f"validation grids; ~622 MB single buffer at 320 x 160 x 160).")
        return 0
    print("[FAIL] float32 deviates from float64 by > 1 %. Reconsider the "
          "memory budget at float64 (validation grid would jump to 1.24 GB "
          "single buffer).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
