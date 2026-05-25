"""Phase 1 production 3D D3Q19 TRT kernel — periodic box only, no geometry.

This module is the production target written from the locked Phase 0
decisions in ``3D_PHASE0_DECISIONS.md`` and the throughput finding in
``3D_PHASE0_FINDINGS.md``:

  * **D3Q19** lattice, **float32** populations, layout ``(19, Nx, Ny, Nz)``.
    Constants are re-imported from ``src.lbm_3d`` (the Phase 0 prototype
    module) so the lattice is the single source of truth across the
    project.
  * **TRT collision** (Ginzburg, Verhaeghe, d'Humières 2008) with the
    "magic parameter" Λ = (1/s_plus − 1/2)(1/s_minus − 1/2) = **3/16**,
    which places the bounce-back wall at the exact mid-link independent
    of viscosity. **BGK is the special case s_plus = s_minus = 1/τ**
    and is implemented by the same kernel — see ``omegas_for_bgk``.
  * **Pull streaming** in a periodic box. Each destination cell pulls
    from 19 source cells with explicit signed-wrap (no Python ``%``,
    which Numba lowers to expensive int-div). This makes the kernel
    parallel-safe by construction; ``parallel=True`` is a one-flag
    switch for the offline ``Validation3D`` path.
  * **Branch-free hot loop.** No inflow / outflow / wall / body
    conditionals inside the inner-most loop. That isolation is the
    direct fix for Phase 0 prototype #3's throughput shortfall.
    Boundaries are applied by a separate pass (added in Phase 2),
    keeping this kernel straight-line code.

Phase 1 exit gate: the 2D-extruded 3D Taylor-Green vortex decays at
the analytic rate 4 ν k² within ±2 %.

Phase 2 will add Bouzidi analytic-q bounce-back on the sphere and
Guo NEEM inflow/outflow as a separate boundary pass that runs after
``trt_periodic_step``. The kernel API will not change.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.lbm_3d import (
    CS2_3D,
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
    equilibrium_3d,
    macroscopic_3d,
)


# ---------------------------------------------------------------------------
# Relaxation-rate helpers (TRT magic parameter Λ = 3/16)
# ---------------------------------------------------------------------------

LAMBDA_TRT = 3.0 / 16.0


def omegas_for_trt(nu: float) -> tuple[float, float]:
    """Return (s_plus, s_minus) for TRT with Λ = 3/16.

    s_plus sets viscosity: nu = cs²(1/s_plus − 1/2), so 1/s_plus = 3 nu + 1/2.
    s_minus is determined by Λ = (1/s_plus − 1/2)(1/s_minus − 1/2) = 3/16.
    """
    s_plus = 1.0 / (3.0 * nu + 0.5)
    # 1/s_minus - 1/2 = Λ / (1/s_plus - 1/2) = Λ / (3 ν) = 1 / (16 ν)
    s_minus = 1.0 / (1.0 / (16.0 * nu) + 0.5)
    return s_plus, s_minus


def omegas_for_bgk(nu: float) -> tuple[float, float]:
    """BGK as the special case s_plus = s_minus = ω. Kept as the
    reference path for equivalence tests."""
    omega = 1.0 / (3.0 * nu + 0.5)
    return omega, omega


# ---------------------------------------------------------------------------
# Production kernel: pull-streaming TRT collision, periodic box
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, parallel=False)
def trt_periodic_step(f, f_next, s_plus, s_minus, vel, weights, opp):
    """One periodic-box TRT collide + push-stream pass (serial).

    Layout: ``f``, ``f_next`` are shape (19, Nx, Ny, Nz). The caller
    swaps the ``f`` and ``f_next`` references between calls.

    Algorithm: classic "collide then push-stream", with all scratch
    state kept in scalar locals — no per-cell ``np.empty`` allocations
    (Numba lowers those to heap calls inside parallel kernels, which
    dominated the cost of the first draft of this file).

    Per cell:

        1. Sum the 19 populations to (rho, ux, uy, uz).
        2. For each direction i, compute ``e_i`` and ``e_ii`` (the
           opposite-direction equilibrium) fresh from those moments,
           pair with ``f[i]`` and ``f[opp[i]]``, apply the TRT
           collision, and push the result to the destination cell
           with periodic wrap.

        The cost is 19 reads of ``f[i]`` plus a second read of
        ``f[opp[i]]`` per direction — TRT is structurally ~2× a
        comparable BGK kernel because the symmetric/antisymmetric
        split needs the paired population.

    Streaming pattern: PUSH (write to neighbour). Push is faster than
    pull for serial execution because reads stay on one cell. It is
    NOT parallel-safe — two source cells could write to the same
    destination. For the offline ``parallel=True`` path see
    ``trt_periodic_step_parallel`` below, which uses pull.
    """
    _, Nx, Ny, Nz = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # --- Macroscopic moments (one read of f[i,x,y,z]) ---
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
                inv_rho = f.dtype.type(1.0) / rho
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz

                # --- TRT collide + push-stream per direction ---
                for i in range(19):
                    ii = opp[i]
                    cu_i = vel[i, 0] * ux + vel[i, 1] * uy + vel[i, 2] * uz
                    cu_ii = vel[ii, 0] * ux + vel[ii, 1] * uy + vel[ii, 2] * uz
                    e_i = weights[i] * rho * (
                        1.0 + cu_i / cs2 + (cu_i * cu_i) * inv2cs4 - usq * inv2cs2
                    )
                    e_ii = weights[ii] * rho * (
                        1.0 + cu_ii / cs2 + (cu_ii * cu_ii) * inv2cs4 - usq * inv2cs2
                    )
                    f_i = f[i, x, y, z]
                    f_ii = f[ii, x, y, z]
                    fp = 0.5 * (f_i + f_ii)
                    fm = 0.5 * (f_i - f_ii)
                    ep = 0.5 * (e_i + e_ii)
                    em = 0.5 * (e_i - e_ii)
                    f_post = f_i - s_plus * (fp - ep) - s_minus * (fm - em)

                    xn = x + vel[i, 0]
                    if xn < 0:
                        xn += Nx
                    elif xn >= Nx:
                        xn -= Nx
                    yn = y + vel[i, 1]
                    if yn < 0:
                        yn += Ny
                    elif yn >= Ny:
                        yn -= Ny
                    zn = z + vel[i, 2]
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    f_next[i, xn, yn, zn] = f_post


@njit(cache=True, fastmath=True, parallel=True)
def trt_periodic_step_parallel(f, f_next, s_plus, s_minus, vel, weights, opp):
    """Pull-streaming parallel variant for the offline ``Validation3D`` path.

    Pull-streaming reads from 19 source cells per destination cell and
    writes to exactly one — this makes the kernel parallel-safe by
    construction. It is ~1.3-2× slower per step than the serial
    push-stream kernel ``trt_periodic_step``, but the ``prange`` over
    the outer x-loop recovers that and more (4-8× on a typical laptop
    with `NUMBA_NUM_THREADS` set).

    Cloud uses the serial push kernel, NOT this one, because of the
    documented `NUMBA_NUM_THREADS` race-condition (see
    [[project_aerolab_3d_phase]] memory entry).
    """
    _, Nx, Ny, Nz = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)
    for x in prange(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # Pull stream into scalar locals — no per-cell allocs.
                # Compute moments in the same pass.
                rho = f.dtype.type(0.0)
                mx = f.dtype.type(0.0)
                my = f.dtype.type(0.0)
                mz = f.dtype.type(0.0)
                # First pass: collect rho and momentum from pulled f.
                # Pulled values are recomputed in the second pass —
                # that costs us 19 extra index ops but avoids the
                # 19-vec allocation.
                for i in range(19):
                    xs = x - vel[i, 0]
                    if xs < 0:
                        xs += Nx
                    elif xs >= Nx:
                        xs -= Nx
                    ys = y - vel[i, 1]
                    if ys < 0:
                        ys += Ny
                    elif ys >= Ny:
                        ys -= Ny
                    zs = z - vel[i, 2]
                    if zs < 0:
                        zs += Nz
                    elif zs >= Nz:
                        zs -= Nz
                    fi = f[i, xs, ys, zs]
                    rho += fi
                    mx += vel[i, 0] * fi
                    my += vel[i, 1] * fi
                    mz += vel[i, 2] * fi
                inv_rho = f.dtype.type(1.0) / rho
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz
                # Second pass: re-pull, compute equilibria, TRT collide,
                # write to f_next[i, x, y, z].
                for i in range(19):
                    ii = opp[i]
                    xs = x - vel[i, 0]
                    if xs < 0:
                        xs += Nx
                    elif xs >= Nx:
                        xs -= Nx
                    ys = y - vel[i, 1]
                    if ys < 0:
                        ys += Ny
                    elif ys >= Ny:
                        ys -= Ny
                    zs = z - vel[i, 2]
                    if zs < 0:
                        zs += Nz
                    elif zs >= Nz:
                        zs -= Nz
                    xs_ii = x - vel[ii, 0]
                    if xs_ii < 0:
                        xs_ii += Nx
                    elif xs_ii >= Nx:
                        xs_ii -= Nx
                    ys_ii = y - vel[ii, 1]
                    if ys_ii < 0:
                        ys_ii += Ny
                    elif ys_ii >= Ny:
                        ys_ii -= Ny
                    zs_ii = z - vel[ii, 2]
                    if zs_ii < 0:
                        zs_ii += Nz
                    elif zs_ii >= Nz:
                        zs_ii -= Nz
                    f_i = f[i, xs, ys, zs]
                    f_ii = f[ii, xs_ii, ys_ii, zs_ii]
                    cu_i = vel[i, 0] * ux + vel[i, 1] * uy + vel[i, 2] * uz
                    cu_ii = vel[ii, 0] * ux + vel[ii, 1] * uy + vel[ii, 2] * uz
                    e_i = weights[i] * rho * (
                        1.0 + cu_i / cs2 + (cu_i * cu_i) * inv2cs4 - usq * inv2cs2
                    )
                    e_ii = weights[ii] * rho * (
                        1.0 + cu_ii / cs2 + (cu_ii * cu_ii) * inv2cs4 - usq * inv2cs2
                    )
                    fp = 0.5 * (f_i + f_ii)
                    fm = 0.5 * (f_i - f_ii)
                    ep = 0.5 * (e_i + e_ii)
                    em = 0.5 * (e_i - e_ii)
                    f_next[i, x, y, z] = (
                        f_i - s_plus * (fp - ep) - s_minus * (fm - em)
                    )


# ---------------------------------------------------------------------------
# AoS (array-of-structures) layout variant: shape (Nx, Ny, Nz, 19)
#
# The plan §3.D-6 named this as the fallback if the SoA layout's cache
# behaviour disappointed in the throughput prototype. Phase 0 proto 3
# and Phase 1 benchmarking both showed throughput collapsing as N grew
# (12 → 7 → 3 Mcell/s as N went 48 → 64 → 96), the unmistakable
# signature of a memory-bandwidth regression: at 96³ the per-direction
# stride is Nx·Ny·Nz·4 = 3.4 MB, so the 19 reads per cell each miss
# L2/L3.
#
# AoS layout puts the 19 populations of one cell contiguous in memory.
# Per-cell reads of all 19 directions become unit-stride (one cache
# line). Push writes to neighbour cells still scatter, but writes are
# cheaper than reads (write buffer + write-combining).
#
# Math is identical; only the layout changes. The (rho, u, equilibrium,
# TRT split) computation is bit-for-bit the same as the SoA kernel.
# ---------------------------------------------------------------------------

def equilibrium_3d_aos(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """D3Q19 equilibrium for AoS layout. ``u`` has shape (3, Nx, Ny, Nz).
    Returns f with shape (Nx, Ny, Nz, 19)."""
    cs2 = CS2_3D
    Nx, Ny, Nz = rho.shape
    ux, uy, uz = u[0], u[1], u[2]
    usq = ux * ux + uy * uy + uz * uz
    f_eq = np.empty((Nx, Ny, Nz, 19), dtype=rho.dtype)
    for i in range(19):
        cx, cy, cz = LATTICE_VELOCITIES_3D[i]
        cu = cx * ux + cy * uy + cz * uz
        f_eq[..., i] = LATTICE_WEIGHTS_3D[i] * rho * (
            1.0 + cu / cs2 + (cu * cu) / (2.0 * cs2 * cs2) - usq / (2.0 * cs2)
        )
    return f_eq


def macroscopic_3d_aos(f: np.ndarray):
    """AoS macroscopic moments. f shape (Nx, Ny, Nz, 19)."""
    rho = f.sum(axis=-1)
    inv = np.where(rho > 0, 1.0 / rho, 0.0)
    cx = LATTICE_VELOCITIES_3D[:, 0].astype(f.dtype)
    cy = LATTICE_VELOCITIES_3D[:, 1].astype(f.dtype)
    cz = LATTICE_VELOCITIES_3D[:, 2].astype(f.dtype)
    ux = (f * cx).sum(axis=-1) * inv
    uy = (f * cy).sum(axis=-1) * inv
    uz = (f * cz).sum(axis=-1) * inv
    return rho, ux, uy, uz


def init_tgv_aos(N: int, U: float, dtype=np.float32) -> np.ndarray:
    """2D-extruded TGV initial condition in AoS layout."""
    k = 2.0 * np.pi / N
    xs = (np.arange(N, dtype=np.float64)[:, None, None] + 0.5)
    ys = (np.arange(N, dtype=np.float64)[None, :, None] + 0.5)
    rho = np.ones((N, N, N), dtype=dtype)
    u = np.zeros((3, N, N, N), dtype=dtype)
    u[0] = (-U * np.cos(k * xs) * np.sin(k * ys)).astype(dtype)
    u[1] = (+U * np.sin(k * xs) * np.cos(k * ys)).astype(dtype)
    return equilibrium_3d_aos(rho, u).astype(dtype)


@njit(cache=True, fastmath=True, parallel=False)
def trt_periodic_step_aos(f, f_next, s_plus, s_minus, vel, weights, opp):
    """AoS-layout periodic-box TRT collide + push-stream (serial).

    Layout: ``f``, ``f_next`` are shape (Nx, Ny, Nz, 19) — last axis
    contiguous. All 19 populations of one cell sit in a single 76-byte
    span, so the per-cell read pattern hits one cache line regardless
    of grid size.

    Math: identical to ``trt_periodic_step``. Only the axis order of
    f and f_next changes.
    """
    Nx, Ny, Nz, _ = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # --- Macroscopic moments: unit-stride reads ---
                rho = f.dtype.type(0.0)
                mx = f.dtype.type(0.0)
                my = f.dtype.type(0.0)
                mz = f.dtype.type(0.0)
                for i in range(19):
                    fi = f[x, y, z, i]
                    rho += fi
                    mx += vel[i, 0] * fi
                    my += vel[i, 1] * fi
                    mz += vel[i, 2] * fi
                inv_rho = f.dtype.type(1.0) / rho
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz

                # --- TRT collide + push-stream per direction ---
                for i in range(19):
                    ii = opp[i]
                    cu_i = vel[i, 0] * ux + vel[i, 1] * uy + vel[i, 2] * uz
                    cu_ii = vel[ii, 0] * ux + vel[ii, 1] * uy + vel[ii, 2] * uz
                    e_i = weights[i] * rho * (
                        1.0 + cu_i / cs2 + (cu_i * cu_i) * inv2cs4 - usq * inv2cs2
                    )
                    e_ii = weights[ii] * rho * (
                        1.0 + cu_ii / cs2 + (cu_ii * cu_ii) * inv2cs4 - usq * inv2cs2
                    )
                    f_i = f[x, y, z, i]
                    f_ii = f[x, y, z, ii]
                    fp = 0.5 * (f_i + f_ii)
                    fm = 0.5 * (f_i - f_ii)
                    ep = 0.5 * (e_i + e_ii)
                    em = 0.5 * (e_i - e_ii)
                    f_post = f_i - s_plus * (fp - ep) - s_minus * (fm - em)

                    xn = x + vel[i, 0]
                    if xn < 0:
                        xn += Nx
                    elif xn >= Nx:
                        xn -= Nx
                    yn = y + vel[i, 1]
                    if yn < 0:
                        yn += Ny
                    elif yn >= Ny:
                        yn -= Ny
                    zn = z + vel[i, 2]
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    f_next[xn, yn, zn, i] = f_post


@njit(cache=True, fastmath=True, parallel=True)
def trt_periodic_step_aos_parallel(f, f_next, s_plus, s_minus, vel, weights, opp):
    """AoS parallel variant for the offline ``Validation3D`` path.

    Note this uses PUSH streaming and is technically not parallel-safe
    in the strict sense (two source cells could write to the same
    destination on the same step). For a typical D3Q19 push-stream
    on a periodic box this is benign because each (source, direction)
    maps to a unique destination cell — the i index disambiguates.
    Verified empirically against the serial variant in the gate tests.
    """
    Nx, Ny, Nz, _ = f.shape
    cs2 = 1.0 / 3.0
    inv2cs2 = 1.0 / (2.0 * cs2)
    inv2cs4 = 1.0 / (2.0 * cs2 * cs2)
    for x in prange(Nx):
        for y in range(Ny):
            for z in range(Nz):
                rho = f.dtype.type(0.0)
                mx = f.dtype.type(0.0)
                my = f.dtype.type(0.0)
                mz = f.dtype.type(0.0)
                for i in range(19):
                    fi = f[x, y, z, i]
                    rho += fi
                    mx += vel[i, 0] * fi
                    my += vel[i, 1] * fi
                    mz += vel[i, 2] * fi
                inv_rho = f.dtype.type(1.0) / rho
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz
                for i in range(19):
                    ii = opp[i]
                    cu_i = vel[i, 0] * ux + vel[i, 1] * uy + vel[i, 2] * uz
                    cu_ii = vel[ii, 0] * ux + vel[ii, 1] * uy + vel[ii, 2] * uz
                    e_i = weights[i] * rho * (
                        1.0 + cu_i / cs2 + (cu_i * cu_i) * inv2cs4 - usq * inv2cs2
                    )
                    e_ii = weights[ii] * rho * (
                        1.0 + cu_ii / cs2 + (cu_ii * cu_ii) * inv2cs4 - usq * inv2cs2
                    )
                    f_i = f[x, y, z, i]
                    f_ii = f[x, y, z, ii]
                    fp = 0.5 * (f_i + f_ii)
                    fm = 0.5 * (f_i - f_ii)
                    ep = 0.5 * (e_i + e_ii)
                    em = 0.5 * (e_i - e_ii)
                    f_post = f_i - s_plus * (fp - ep) - s_minus * (fm - em)
                    xn = x + vel[i, 0]
                    if xn < 0:
                        xn += Nx
                    elif xn >= Nx:
                        xn -= Nx
                    yn = y + vel[i, 1]
                    if yn < 0:
                        yn += Ny
                    elif yn >= Ny:
                        yn -= Ny
                    zn = z + vel[i, 2]
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    f_next[xn, yn, zn, i] = f_post


def run_tgv_aos(
    N: int = 32,
    U: float = 0.04,
    nu: float = 0.01,
    n_steps: int = 800,
    sample_every: int = 20,
    scheme: str = "trt",
    parallel: bool = False,
    dtype=np.float32,
):
    """AoS analogue of run_tgv. Same gate, same API, faster at 96³."""
    if scheme == "trt":
        s_plus, s_minus = omegas_for_trt(nu)
    elif scheme == "bgk":
        s_plus, s_minus = omegas_for_bgk(nu)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    s_plus = dtype(s_plus)
    s_minus = dtype(s_minus)

    f = init_tgv_aos(N, U, dtype=dtype)
    f_next = f.copy()
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(dtype)
    opp = OPPOSITE_3D.astype(np.int32)

    step_fn = (
        trt_periodic_step_aos_parallel if parallel else trt_periodic_step_aos
    )

    times = [0]
    _, ux0, uy0, uz0 = macroscopic_3d_aos(f)
    ke = [_kinetic_energy(ux0, uy0, uz0)]
    for step in range(1, n_steps + 1):
        step_fn(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
        if step % sample_every == 0:
            _, ux_t, uy_t, uz_t = macroscopic_3d_aos(f)
            ke_t = _kinetic_energy(ux_t, uy_t, uz_t)
            if not np.isfinite(ke_t) or ke_t > 10 * ke[0]:
                return times + [step], ke + [float("nan")], {"diverged": True}
            times.append(step)
            ke.append(ke_t)
    diag = {
        "diverged": False,
        "ke_initial": ke[0],
        "ke_final": ke[-1],
        "n_steps": n_steps,
        "scheme": scheme,
        "parallel": parallel,
        "dtype": np.dtype(dtype).name,
        "layout": "aos",
    }
    return times, ke, diag


# ---------------------------------------------------------------------------
# Taylor-Green vortex driver (the Phase 1 exit gate)
# ---------------------------------------------------------------------------

def init_tgv(N: int, U: float, dtype=np.float32) -> np.ndarray:
    """2D-extruded Taylor-Green vortex on a periodic N×N×N box.

    u_x = −U cos(k x) sin(k y),  u_y = +U sin(k x) cos(k y),  u_z = 0.
    k = 2π / N. Analytic kinetic-energy decay rate is 4 ν k².
    """
    k = 2.0 * np.pi / N
    xs = (np.arange(N, dtype=np.float64)[:, None, None] + 0.5)
    ys = (np.arange(N, dtype=np.float64)[None, :, None] + 0.5)
    rho = np.ones((N, N, N), dtype=dtype)
    u = np.zeros((3, N, N, N), dtype=dtype)
    u[0] = (-U * np.cos(k * xs) * np.sin(k * ys)).astype(dtype)
    u[1] = (+U * np.sin(k * xs) * np.cos(k * ys)).astype(dtype)
    f = equilibrium_3d(rho, u).astype(dtype)
    return f


def analytic_tgv_decay_rate(nu: float, N: int) -> float:
    """KE(t) = KE(0) exp(−4 ν k² t); decay rate is 4 ν k²."""
    k = 2.0 * np.pi / N
    return 4.0 * nu * k * k


def run_tgv(
    N: int = 32,
    U: float = 0.04,
    nu: float = 0.01,
    n_steps: int = 800,
    sample_every: int = 20,
    scheme: str = "trt",
    parallel: bool = False,
    dtype=np.float32,
):
    """Run a 3D TGV for n_steps, returning (times, ke, diag).

    scheme = "trt" (production, Λ = 3/16) or "bgk" (reference path,
    s_plus = s_minus). parallel switches the kernel variant.
    """
    if scheme == "trt":
        s_plus, s_minus = omegas_for_trt(nu)
    elif scheme == "bgk":
        s_plus, s_minus = omegas_for_bgk(nu)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    s_plus = dtype(s_plus)
    s_minus = dtype(s_minus)

    f = init_tgv(N, U, dtype=dtype)
    f_next = f.copy()
    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(dtype)
    opp = OPPOSITE_3D.astype(np.int32)

    step_fn = (
        trt_periodic_step_parallel if parallel else trt_periodic_step
    )

    times = [0]
    rho0, ux0, uy0, uz0 = macroscopic_3d(f)
    ke = [_kinetic_energy(ux0, uy0, uz0)]
    for step in range(1, n_steps + 1):
        step_fn(f, f_next, s_plus, s_minus, vel, weights, opp)
        f, f_next = f_next, f
        if step % sample_every == 0:
            rho_t, ux_t, uy_t, uz_t = macroscopic_3d(f)
            ke_t = _kinetic_energy(ux_t, uy_t, uz_t)
            if not np.isfinite(ke_t) or ke_t > 10 * ke[0]:
                return times + [step], ke + [float("nan")], {"diverged": True}
            times.append(step)
            ke.append(ke_t)
    diag = {
        "diverged": False,
        "ke_initial": ke[0],
        "ke_final": ke[-1],
        "n_steps": n_steps,
        "scheme": scheme,
        "parallel": parallel,
        "dtype": np.dtype(dtype).name,
    }
    return times, ke, diag


def _kinetic_energy(ux, uy, uz) -> float:
    return float(0.5 * (ux * ux + uy * uy + uz * uz).sum())


def fit_decay_rate(times, ke) -> float:
    """Linear fit ln(KE) = a + b t; return −b. Discards the first
    quarter of samples to avoid the brief initial transient."""
    ts = np.asarray(times, dtype=np.float64)
    es = np.asarray(ke, dtype=np.float64)
    cutoff = max(1, len(ts) // 4)
    b = np.polyfit(ts[cutoff:], np.log(es[cutoff:]), 1)[0]
    return -float(b)


__all__ = [
    "LAMBDA_TRT",
    "omegas_for_trt",
    "omegas_for_bgk",
    "trt_periodic_step",
    "trt_periodic_step_parallel",
    "init_tgv",
    "analytic_tgv_decay_rate",
    "run_tgv",
    "fit_decay_rate",
]
