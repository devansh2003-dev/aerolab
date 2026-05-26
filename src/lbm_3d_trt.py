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
    span, so per-cell reads hit one cache line regardless of N.

    Known-but-deferred optimisation (reviewer 2026-05-28 P1.3):
    every equilibrium gets computed TWICE in the per-direction loop —
    once as ``e_i`` when ``i`` is the direction itself, once as
    ``e_ii`` when ``opp[i]`` points back to it. The math is doubled
    by construction, so in principle precomputing all 19 equilibria
    into a buffer once should be faster. In practice on Numba 0.65,
    the two reasonable buffer choices both backfired:

      * ``np.empty(19, dtype=f.dtype)`` measured ~35 % SLOWER at
        96^3 (1.19 vs 1.94 Mcell/s). Numba does not stack-allocate
        this reliably and the heap-allocation cost per cell dwarfs
        the saved equilibrium work.
      * Passing a preallocated 19-vector buffer through the kernel
        signature breaks parallel safety (one buffer, many
        prange-iterations writing to it).

    The correct fix is 19 named scalars (``e0, e1, ..., e18``),
    matching what the 2D MRT kernel does. That is ~150 lines per
    kernel variant × 4 variants = 600 lines of nearly-identical
    code, and is queued as a focused Phase 1.5 optimisation pass
    once geometry (Phase 2) is in. Doing it now would mask Phase 2
    bugs behind a kernel that is hard to debug.

    For Phase 1, the kernel stays in its visually clear form and
    pays the structural ~2x TRT overhead. Production grid sizes
    are decided in Phase 5 once the gate-test Cd numbers from
    Phase 4 reveal whether the wall-placement advantage was worth
    the throughput cost.
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
                # NOTE: e_i and e_ii are computed fresh each iteration.
                # Every equilibrium therefore evaluates twice across
                # the 19-iteration loop. See the docstring for why this
                # duplication is deferred rather than fixed in place.
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

    Parallel safety (reviewer 2026-05-28 P5):

        For push streaming on a periodic D3Q19 box the destination
        is determined by (source_cell, direction_i) via
        (x + c_i_x, y + c_i_y, z + c_i_z, i) — the direction index
        is preserved as the destination's last axis. The map
        (source, i) → (destination, i) is therefore a BIJECTION:

          * Different sources to the same destination would require
            two source cells (xs1, ys1, zs1) and (xs2, ys2, zs2) with
            xs1 + c_i_x == xs2 + c_i_x (and analogously y, z) for the
            SAME direction i — which collapses to xs1 == xs2, ys1 ==
            ys2, zs1 == zs2. No collision.
          * Different directions to the same destination would write
            to different last-axis slots f_next[..., i], so they touch
            disjoint memory.

        Therefore every f_next entry has exactly one writer per step,
        regardless of how prange schedules the outer x-loop. This is
        a structural property of D3Q19 push-streaming on the
        (Nx, Ny, Nz, 19) layout, not a runtime check; the earlier
        "verified empirically" claim understated what the lattice
        algebra already guarantees.

    Cloud must continue to use the serial variant, see
    [[project_aerolab_3d_phase]] memory entry for the
    `NUMBA_NUM_THREADS` race-condition rationale.
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


# ---------------------------------------------------------------------------
# Channel-flow TRT driver
#
# Composes:
#   * ``trt_channel_step``: TRT collide + push-stream + full-way wall BB
#     at y faces + full-way body BB. x faces are DROPPED (no streaming
#     contribution out of the domain). Mirrors ``step_bgk_3d_pure_bulk``
#     in ``src/lbm_3d.py`` for boundary handling; only the collision math
#     differs.
#   * ``apply_guo_inflow`` and ``apply_guo_outflow`` from ``src/lbm_3d.py``
#     -- those are collision-agnostic (they only need post-stream
#     f_next entries and the prescribed boundary state), so the BGK
#     and TRT paths share the inflow/outflow code.
#   * ``apply_bouzidi_correction_trt`` from ``src/lbm_3d_bouzidi.py`` when
#     wall_links are provided.
#
# At ``s_plus = s_minus = omega`` the TRT split formula reduces to BGK
# exactly. The corresponding driver-level test pins this against the
# proven ``step_bgk_3d_pure_bulk`` + Guo NEEM path.
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def trt_channel_step(
    f: np.ndarray,
    f_next: np.ndarray,
    body: np.ndarray,
    s_plus: np.float32,
    s_minus: np.float32,
    vel: np.ndarray,
    weights: np.ndarray,
    opp: np.ndarray,
) -> None:
    """One channel-flow TRT collide + push-stream pass.

    Boundary handling: ``z`` is periodic, ``y`` is full-way bounce-back
    on both faces (channel walls), ``x`` boundaries are DROPPED (the
    caller runs ``apply_guo_inflow`` / ``apply_guo_outflow`` as
    post-passes to fill them). Solid cells (``body == True``) bounce-back
    via opposite-direction write to the source.

    Same shape and convention as ``step_bgk_3d_pure_bulk``: pure-bulk
    plus body + walls, no inline inflow/outflow.
    """
    _, Nx, Ny, Nz = f.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if body[x, y, z]:
                    continue

                rho = np.float32(0.0)
                mx = np.float32(0.0)
                my = np.float32(0.0)
                mz = np.float32(0.0)
                for i in range(19):
                    fi = f[i, x, y, z]
                    rho += fi
                    mx += np.float32(vel[i, 0]) * fi
                    my += np.float32(vel[i, 1]) * fi
                    mz += np.float32(vel[i, 2]) * fi
                if rho > np.float32(0.0):
                    inv_rho = np.float32(1.0) / rho
                else:
                    inv_rho = np.float32(0.0)
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz

                for i in range(19):
                    ii = opp[i]
                    cu_i = (
                        np.float32(vel[i, 0]) * ux
                        + np.float32(vel[i, 1]) * uy
                        + np.float32(vel[i, 2]) * uz
                    )
                    cu_ii = (
                        np.float32(vel[ii, 0]) * ux
                        + np.float32(vel[ii, 1]) * uy
                        + np.float32(vel[ii, 2]) * uz
                    )
                    e_i = weights[i] * rho * (
                        np.float32(1.0)
                        + cu_i / cs2
                        + (cu_i * cu_i) * inv_2cs4
                        - usq * inv_2cs2
                    )
                    e_ii = weights[ii] * rho * (
                        np.float32(1.0)
                        + cu_ii / cs2
                        + (cu_ii * cu_ii) * inv_2cs4
                        - usq * inv_2cs2
                    )
                    f_i = f[i, x, y, z]
                    f_ii = f[ii, x, y, z]
                    fp = np.float32(0.5) * (f_i + f_ii)
                    fm = np.float32(0.5) * (f_i - f_ii)
                    ep = np.float32(0.5) * (e_i + e_ii)
                    em = np.float32(0.5) * (e_i - e_ii)
                    f_post = f_i - s_plus * (fp - ep) - s_minus * (fm - em)

                    xn = x + vel[i, 0]
                    yn = y + vel[i, 1]
                    zn = z + vel[i, 2]
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    if 0 <= xn < Nx and 0 <= yn < Ny:
                        if body[xn, yn, zn]:
                            f_next[opp[i], x, y, z] = f_post
                        else:
                            f_next[i, xn, yn, zn] = f_post
                    elif yn < 0 or yn >= Ny:
                        f_next[opp[i], x, y, z] = f_post
                    # xn out of [0, Nx): drop. Guo NEEM fills the
                    # boundary cells in their own post-pass.


def run_channel_smoke_trt(
    Nx: int = 80,
    Ny: int = 32,
    Nz: int = 32,
    u_in: float = 0.05,
    nu: float = 0.01,
    n_steps: int = 400,
    body: np.ndarray | None = None,
    wall_links=None,
    use_guo_neem: bool = True,
    rho_outflow: float = 1.0,
    scheme: str = "trt",
    progress_callback=None,
):
    """3D channel-flow with TRT collision + (optional) Bouzidi + (default)
    Guo NEEM. Returns ``(rho, ux, uy, uz, diag)`` -- same shape as
    ``run_channel_smoke``.

    Differences from ``run_channel_smoke``:

      * ``trt_channel_step`` instead of ``step_bgk_3d``. The TRT magic
        parameter Λ = 3/16 places the no-slip wall at the on-link
        midpoint INDEPENDENT of viscosity -- the property that buys
        Cd accuracy in the validation track.
      * Default ``use_guo_neem=True``: TRT without Guo NEEM is not a
        sensible production configuration.
      * ``scheme="trt"`` (default) uses Λ = 3/16; ``scheme="bgk"`` uses
        ``s_plus = s_minus = omega`` for equivalence pinning against
        ``run_channel_smoke``.
      * Bouzidi (if ``wall_links`` is provided) uses
        ``apply_bouzidi_correction_trt`` (NOT the BGK variant).
    """
    from src.lbm_3d import (
        apply_guo_inflow,
        apply_guo_outflow,
        init_population,
        macroscopic_3d,
    )

    if body is None:
        body = np.zeros((Nx, Ny, Nz), dtype=np.bool_)

    if scheme == "trt":
        s_plus_v, s_minus_v = omegas_for_trt(nu)
    elif scheme == "bgk":
        s_plus_v, s_minus_v = omegas_for_bgk(nu)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    s_plus = np.float32(s_plus_v)
    s_minus = np.float32(s_minus_v)

    f = init_population(Nx, Ny, Nz, u_in)
    f_next = f.copy()

    vel = LATTICE_VELOCITIES_3D.astype(np.int32)
    weights = LATTICE_WEIGHTS_3D.astype(np.float32)
    opp = OPPOSITE_3D.astype(np.int32)

    mass_initial = float(f.sum())

    if wall_links is not None:
        from src.lbm_3d_bouzidi import apply_bouzidi_correction_trt
    else:
        apply_bouzidi_correction_trt = None

    u_in_f32 = np.float32(u_in)
    rho_outflow_f32 = np.float32(rho_outflow)

    for step in range(n_steps):
        trt_channel_step(
            f, f_next, body, s_plus, s_minus, vel, weights, opp
        )
        if use_guo_neem:
            apply_guo_inflow(f_next, body, u_in_f32)
            apply_guo_outflow(f_next, body, rho_outflow_f32)

        if apply_bouzidi_correction_trt is not None:
            apply_bouzidi_correction_trt(
                f, f_next, body,
                wall_links.x, wall_links.y, wall_links.z,
                wall_links.dir, wall_links.q,
                s_plus, s_minus,
            )
        f, f_next = f_next, f
        if progress_callback is not None and (
            step % max(1, n_steps // 20) == 0
        ):
            progress_callback(
                step / n_steps, f"3D TRT step {step}/{n_steps}"
            )

    rho, ux, uy, uz = macroscopic_3d(f)
    if body is not None:
        ux = np.where(body, np.float32(0.0), ux)
        uy = np.where(body, np.float32(0.0), uy)
        uz = np.where(body, np.float32(0.0), uz)
    mass_final = float(f.sum())

    mid_x = Nx // 2
    mid_z = Nz // 2
    y_profile = ux[mid_x, :, mid_z]
    u_mean = float(np.mean(y_profile[1:-1]))
    u_peak = float(np.max(y_profile))
    centerline_ratio = u_peak / u_mean if u_mean > 0 else float("nan")

    diag = {
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_drift_rel": (mass_final - mass_initial) / mass_initial,
        "u_peak": u_peak,
        "u_mean": u_mean,
        "centerline_ratio": centerline_ratio,
        "u_in": u_in,
        "nu": nu,
        "scheme": scheme,
        "use_guo_neem": use_guo_neem,
    }
    return rho, ux, uy, uz, diag


__all__ = [
    "LAMBDA_TRT",
    "omegas_for_trt",
    "omegas_for_bgk",
    "trt_periodic_step",
    "trt_periodic_step_parallel",
    "trt_channel_step",
    "run_channel_smoke_trt",
    "init_tgv",
    "analytic_tgv_decay_rate",
    "run_tgv",
    "fit_decay_rate",
]
