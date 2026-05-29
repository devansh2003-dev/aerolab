"""Shared helpers for the Phase 0 confirmation prototypes.

Disposable: this file exists only to dedupe the TGV initial condition
and the decay-rate fit across protos 1 and 4. Removed when the four
Phase 0 prototypes have served their purpose. See 3D_PHASE0_DECISIONS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def make_tgv_init(N: int, U: float, dtype=np.float32):
    """2D-extruded Taylor-Green vortex on a periodic N x N x N box.

    u_x = -U cos(k x) sin(k y),  u_y =  U sin(k x) cos(k y),  u_z = 0.
    k = 2 pi / N. Analytic kinetic-energy decay rate is 4 nu k^2.
    """
    k = 2.0 * np.pi / N
    xs = (np.arange(N, dtype=np.float64)[:, None, None] + 0.5)
    ys = (np.arange(N, dtype=np.float64)[None, :, None] + 0.5)
    u = np.zeros((3, N, N, N), dtype=dtype)
    u[0] = (-U * np.cos(k * xs) * np.sin(k * ys)).astype(dtype)
    u[1] = (U * np.sin(k * xs) * np.cos(k * ys)).astype(dtype)
    rho = np.ones((N, N, N), dtype=dtype)
    return rho, u


def kinetic_energy(u: np.ndarray) -> float:
    return float(0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sum())


def fit_decay_rate(times, ke):
    """Linear fit ln(KE) = a + b t; return -b."""
    ln_ke = np.log(np.asarray(ke, dtype=np.float64))
    b = np.polyfit(np.asarray(times, dtype=np.float64), ln_ke, 1)[0]
    return -float(b)


def analytic_tgv_decay_rate(nu: float, N: int) -> float:
    k = 2.0 * np.pi / N
    return 4.0 * nu * k * k
