"""3D Lattice Boltzmann Method (LBM) solver -- scaffold.

D3Q19 lattice, BGK collision, full-way bounce-back on solid + walls,
Zou-He-style equilibrium inflow, zero-gradient outflow, periodic
spanwise. NumPy reference + Numba ``@njit`` fused step.

This is the WORK-IN-PROGRESS 3D extension of the 2D solver in
``src/lbm.py``. It is intentionally minimal:

  * BGK (single relaxation time), not MRT. Once the streaming and
    boundaries pass the channel-flow smoke at low Re, we can layer
    MRT on top -- it is purely a collision-step change.
  * Full-way bounce-back on the body, not Bouzidi. The 2D project's
    Bouzidi q-fields are a third-order correction that requires
    geometric ray-cell intersection; bringing them to 3D needs its
    own diagnostic pass, so this scaffold keeps the simpler
    on-link-midpoint bounce-back.
  * float32 populations. D3Q19 at 200 x 100 x 100 is already
    ~150 MB in float32 and ~300 MB in float64 (per buffer, and we
    keep two). float32 is the right default for development on a
    laptop without GPU.

The headline 2D validation lives in VALIDATION.md. 3D validation is
deferred until the smoke test passes; the immediate first target is
3D plane Poiseuille (channel flow, no body) where the analytic
profile gives a known-answer check.

Reference: Kruger et al., "The Lattice Boltzmann Method: Principles
and Practice" (Springer, 2017), chapter 3 (velocity sets) and
chapter 5 (boundary conditions). D3Q19 weights and moment basis
follow d'Humieres et al. 2002.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Threading: deliberately disabled to match src/lbm.py. The Cloud
# container's NUMBA_NUM_THREADS race-condition only matters for the
# deployed 2D solver, but keeping the 3D solver single-threaded too
# means a future merge or shared-helper extraction does not introduce
# a new threading boundary. Local users who want parallel can flip
# this to numba.prange and rebuild.
prange = range


# ---------------------------------------------------------------------------
# D3Q19 lattice
#
# Velocity index convention:
#   0      : rest
#   1-6    : six face neighbours (+/- x, y, z)
#   7-18   : twelve edge neighbours (the cube edges meeting the origin)
#
# The exact ordering below matches the standard d'Humieres / Kruger
# textbook layout, so the moment basis (when we add MRT) can be
# transcribed without re-permuting.
# ---------------------------------------------------------------------------

LATTICE_VELOCITIES_3D = np.array(
    [
        [ 0,  0,  0],   # 0: rest
        [ 1,  0,  0],   # 1: +x
        [-1,  0,  0],   # 2: -x
        [ 0,  1,  0],   # 3: +y
        [ 0, -1,  0],   # 4: -y
        [ 0,  0,  1],   # 5: +z
        [ 0,  0, -1],   # 6: -z
        [ 1,  1,  0],   # 7
        [-1, -1,  0],   # 8
        [ 1, -1,  0],   # 9
        [-1,  1,  0],   # 10
        [ 1,  0,  1],   # 11
        [-1,  0, -1],   # 12
        [ 1,  0, -1],   # 13
        [-1,  0,  1],   # 14
        [ 0,  1,  1],   # 15
        [ 0, -1, -1],   # 16
        [ 0,  1, -1],   # 17
        [ 0, -1,  1],   # 18
    ],
    dtype=np.int8,
)

# Weights: w0 = 1/3, axis = 1/18, edge = 1/36. They sum to 1 and
# reproduce the second-order moments of Maxwell-Boltzmann required
# for Navier-Stokes recovery.
LATTICE_WEIGHTS_3D = np.array(
    [1.0 / 3.0]
    + [1.0 / 18.0] * 6
    + [1.0 / 36.0] * 12,
    dtype=np.float64,
)

# OPPOSITE[i] = index of -c_i. Verified against
# LATTICE_VELOCITIES_3D[OPPOSITE[i]] == -LATTICE_VELOCITIES_3D[i]
# in the smoke test (`tests/test_lbm_3d_smoke.py`).
OPPOSITE_3D = np.array(
    [0,
     2, 1, 4, 3, 6, 5,           # axis pairs
     8, 7, 10, 9,                # xy edges
     12, 11, 14, 13,             # xz edges
     16, 15, 18, 17],            # yz edges
    dtype=np.int8,
)

# Lattice sound speed squared. Same as 2D -- it is set by the lattice
# normalization sum_i w_i * c_i_x^2 = 1/3, which holds for both D2Q9
# and D3Q19 by construction.
CS2_3D = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Macroscopic moments + equilibrium
# ---------------------------------------------------------------------------

def macroscopic_3d(f: np.ndarray):
    """Return (rho, ux, uy, uz) from population field ``f`` shape (19, Nx, Ny, Nz).

    Reference (NumPy) implementation. Used for tests and small grids;
    the Numba step does the same arithmetic in-place per cell.
    """
    rho = f.sum(axis=0)
    # Avoid divide-by-zero in test fixtures that may seed rho = 0
    # somewhere. Real runs keep rho ~ 1.
    inv = np.where(rho > 0, 1.0 / rho, 0.0)
    cx = LATTICE_VELOCITIES_3D[:, 0].astype(f.dtype)
    cy = LATTICE_VELOCITIES_3D[:, 1].astype(f.dtype)
    cz = LATTICE_VELOCITIES_3D[:, 2].astype(f.dtype)
    ux = np.tensordot(cx, f, axes=([0], [0])) * inv
    uy = np.tensordot(cy, f, axes=([0], [0])) * inv
    uz = np.tensordot(cz, f, axes=([0], [0])) * inv
    return rho, ux, uy, uz


def equilibrium_3d(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """D3Q19 equilibrium f^eq for (rho, u). ``u`` has shape (3, Nx, Ny, Nz).

    Second-order Hermite expansion:
      f^eq_i = w_i rho [ 1 + (c.u)/cs2 + (c.u)^2 / (2 cs2^2) - u.u/(2 cs2) ]
    """
    cs2 = CS2_3D
    Nx, Ny, Nz = rho.shape
    ux, uy, uz = u[0], u[1], u[2]
    usq = ux * ux + uy * uy + uz * uz
    f_eq = np.empty((19, Nx, Ny, Nz), dtype=rho.dtype)
    for i in range(19):
        cx, cy, cz = LATTICE_VELOCITIES_3D[i]
        cu = cx * ux + cy * uy + cz * uz
        f_eq[i] = LATTICE_WEIGHTS_3D[i] * rho * (
            1.0 + cu / cs2 + (cu * cu) / (2.0 * cs2 * cs2) - usq / (2.0 * cs2)
        )
    return f_eq


# ---------------------------------------------------------------------------
# Fused Numba step: collide + stream + boundary application
#
# Layout: ``f`` is shape (19, Nx, Ny, Nz). Indexing convention is
# f[i, x, y, z]. The streaming step copies f_i[x, y, z] -> f_i[x+cx,
# y+cy, z+cz], implemented out-of-place with a scratch buffer ``f_next``.
#
# Boundaries handled in-place AFTER the stream into f_next:
#   * x = 0     : equilibrium-velocity inflow at ux=u_in, uy=uz=0
#   * x = Nx-1  : zero-gradient outflow (copy from x = Nx-2)
#   * y = 0, Ny-1: no-slip via full-way bounce-back (mirror unknowns)
#   * z direction: periodic (handled by index wrap in stream loop)
#   * body[x,y,z] == True: full-way bounce-back (post-stream swap of opposites)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def step_bgk_3d(
    f: np.ndarray,
    f_next: np.ndarray,
    body: np.ndarray,
    omega: np.float32,
    u_in: np.float32,
) -> None:
    """Fused BGK collision + streaming + boundaries, in-place into ``f_next``.

    Caller swaps the f / f_next references between calls so allocation
    is amortized. Both arrays must be C-contiguous float32.

    body is a bool array of shape (Nx, Ny, Nz). True = solid.
    """
    Nx, Ny, Nz = body.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    # Collision + streaming in one pass. Standard "pull" formulation
    # would read neighbours; this "push" formulation writes to
    # neighbours, which makes solid-cell handling marginally simpler.
    for x in prange(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if body[x, y, z]:
                    # Inside solid: do not collide. Bounce-back handled
                    # below as a post-stream swap.
                    continue

                # --- macroscopic moments ---
                rho = np.float32(0.0)
                mx = np.float32(0.0)
                my = np.float32(0.0)
                mz = np.float32(0.0)
                for i in range(19):
                    fi = f[i, x, y, z]
                    rho += fi
                    mx += np.float32(LATTICE_VELOCITIES_3D[i, 0]) * fi
                    my += np.float32(LATTICE_VELOCITIES_3D[i, 1]) * fi
                    mz += np.float32(LATTICE_VELOCITIES_3D[i, 2]) * fi
                # Inflow override at x = 0: prescribed velocity, density
                # equilibrated to match. Plain equilibrium BC -- good
                # enough at moderate Re; replace with Zou-He when we
                # care about pressure recovery at the inlet.
                if x == 0:
                    rho = np.float32(1.0)
                    ux = u_in
                    uy = np.float32(0.0)
                    uz = np.float32(0.0)
                else:
                    inv_rho = np.float32(1.0) / rho if rho > 0 else np.float32(0.0)
                    ux = mx * inv_rho
                    uy = my * inv_rho
                    uz = mz * inv_rho

                usq = ux * ux + uy * uy + uz * uz

                # --- collide ---
                for i in range(19):
                    cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                    cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                    cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                    w = np.float32(LATTICE_WEIGHTS_3D[i])
                    cu = cx * ux + cy * uy + cz * uz
                    f_eq = w * rho * (
                        np.float32(1.0)
                        + cu / cs2
                        + (cu * cu) * inv_2cs4
                        - usq * inv_2cs2
                    )
                    f_post = f[i, x, y, z] + omega * (f_eq - f[i, x, y, z])
                    if x == 0:
                        # At the inlet, write equilibrium directly:
                        # populations downstream of the inlet are
                        # consistent with the prescribed velocity.
                        f_post = f_eq

                    # --- stream to (x+cx, y+cy, z+cz) ---
                    xn = x + LATTICE_VELOCITIES_3D[i, 0]
                    yn = y + LATTICE_VELOCITIES_3D[i, 1]
                    zn = z + LATTICE_VELOCITIES_3D[i, 2]
                    # z is periodic
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    # x and y are bounded; out-of-range stays at the
                    # source cell (effectively held by the outflow
                    # copy / wall bounce-back applied next).
                    if 0 <= xn < Nx and 0 <= yn < Ny:
                        if body[xn, yn, zn]:
                            # Body bounce-back: redirect this population
                            # back as the opposite at the source.
                            opp = OPPOSITE_3D[i]
                            f_next[opp, x, y, z] = f_post
                        else:
                            f_next[i, xn, yn, zn] = f_post
                    elif yn < 0 or yn >= Ny:
                        # Channel wall bounce-back: opposite at source.
                        opp = OPPOSITE_3D[i]
                        f_next[opp, x, y, z] = f_post
                    elif xn >= Nx:
                        # Outflow: keep the population at the source so
                        # the outflow copy step picks it up.
                        f_next[i, x, y, z] = f_post

    # --- Outflow: zero-gradient copy from x = Nx-2 to x = Nx-1 ---
    for y in range(Ny):
        for z in range(Nz):
            if not body[Nx - 1, y, z]:
                for i in range(19):
                    f_next[i, Nx - 1, y, z] = f_next[i, Nx - 2, y, z]


# ---------------------------------------------------------------------------
# Guo non-equilibrium extrapolation (NEEM) boundary path
#
# References: Guo, Zheng, Shi (2002), "Non-equilibrium extrapolation method
# for velocity and pressure boundary conditions in the lattice Boltzmann
# method", Chinese Physics 11(4), 366-374.
#
# The classical equilibrium-inflow + zero-gradient-outflow used by
# ``step_bgk_3d`` is the cheap approximation. Guo NEEM is the principled
# upgrade: at a boundary node the distribution is decomposed into an
# equilibrium part (evaluated at the prescribed boundary macros) and a
# non-equilibrium part (copied from the nearest interior neighbour). The
# non-equilibrium part is what carries the shear / pressure gradient
# information across the boundary, so the wake near the outlet stops
# accumulating spurious reflections.
#
# Architectural pattern: TWO post-passes after a pure-bulk collide+stream
# kernel. Mirrors the Bouzidi correction post-pass design (see
# ``src/lbm_3d_bouzidi.py``). The interaction with Bouzidi is benign --
# Bouzidi only writes specific ``f_next[opp, x_f]`` entries; Guo NEEM
# writes every entry at x=0 / x=Nx-1. When ordered Guo→Bouzidi, Bouzidi
# wins at wall-link cells (rare overlap, only if the body intersects the
# inlet or outlet plane, which is not a real geometry).
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def step_bgk_3d_pure_bulk(
    f: np.ndarray,
    f_next: np.ndarray,
    body: np.ndarray,
    omega: np.float32,
) -> None:
    """Same as ``step_bgk_3d`` minus the inflow override and outflow copy.

    Used by the Guo NEEM path: the boundary passes (``apply_guo_inflow``
    and ``apply_guo_outflow``) override f_next at x=0 / x=Nx-1 after this
    kernel runs, so any value this kernel writes there is discarded. We
    skip the inline boundary handling to keep the kernel branch-free at
    the boundary planes and to make the responsibility clear: this kernel
    is pure bulk + walls + body, nothing more.

    Channel-wall BB (y boundaries) and body BB (solid neighbours) stay
    inline because they would otherwise require their own boundary
    passes for the same per-step coverage.
    """
    Nx, Ny, Nz = body.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for x in prange(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if body[x, y, z]:
                    continue

                # --- macroscopic moments (no inflow override) ---
                rho = np.float32(0.0)
                mx = np.float32(0.0)
                my = np.float32(0.0)
                mz = np.float32(0.0)
                for i in range(19):
                    fi = f[i, x, y, z]
                    rho += fi
                    mx += np.float32(LATTICE_VELOCITIES_3D[i, 0]) * fi
                    my += np.float32(LATTICE_VELOCITIES_3D[i, 1]) * fi
                    mz += np.float32(LATTICE_VELOCITIES_3D[i, 2]) * fi
                inv_rho = np.float32(1.0) / rho if rho > 0 else np.float32(0.0)
                ux = mx * inv_rho
                uy = my * inv_rho
                uz = mz * inv_rho
                usq = ux * ux + uy * uy + uz * uz

                # --- collide + stream ---
                for i in range(19):
                    cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                    cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                    cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                    w = np.float32(LATTICE_WEIGHTS_3D[i])
                    cu = cx * ux + cy * uy + cz * uz
                    f_eq = w * rho * (
                        np.float32(1.0)
                        + cu / cs2
                        + (cu * cu) * inv_2cs4
                        - usq * inv_2cs2
                    )
                    f_post = f[i, x, y, z] + omega * (f_eq - f[i, x, y, z])

                    xn = x + LATTICE_VELOCITIES_3D[i, 0]
                    yn = y + LATTICE_VELOCITIES_3D[i, 1]
                    zn = z + LATTICE_VELOCITIES_3D[i, 2]
                    if zn < 0:
                        zn += Nz
                    elif zn >= Nz:
                        zn -= Nz
                    if 0 <= xn < Nx and 0 <= yn < Ny:
                        if body[xn, yn, zn]:
                            opp = OPPOSITE_3D[i]
                            f_next[opp, x, y, z] = f_post
                        else:
                            f_next[i, xn, yn, zn] = f_post
                    elif yn < 0 or yn >= Ny:
                        # Channel wall: full-way BB.
                        opp = OPPOSITE_3D[i]
                        f_next[opp, x, y, z] = f_post
                    # xn out of domain (x < 0 or x >= Nx): drop the
                    # population. Guo NEEM will populate the boundary
                    # cells in their own pass.


@njit(cache=True, fastmath=True)
def apply_guo_inflow(
    f_next: np.ndarray,
    body: np.ndarray,
    u_in: np.float32,
) -> None:
    """Guo NEEM inflow at x = 0: prescribed velocity, extrapolated density.

    For each fluid cell at x = 0:
      1. Compute the interior neighbour's macroscopic moments from
         f_next[*, 1, y, z] (the post-stream values).
      2. Set the boundary state: u_b = (u_in, 0, 0), rho_b = rho_neighbor
         (mass-preserving density extrapolation -- using rho = 1.0
         instead would force a constant-density inlet which slowly
         drains mass at finite Re).
      3. For each direction i:
            f_next[i, 0, y, z] = f_eq_i(rho_b, u_b)
                               + (f_next[i, 1, y, z] - f_eq_i(rho_n, u_n))

    The non-equilibrium part (the bracket term) carries the local shear
    information from the neighbour into the boundary; that is what makes
    the wake recover cleanly compared to the plain equilibrium-write
    inflow used by ``step_bgk_3d``.
    """
    _, Nx, Ny, Nz = f_next.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for y in range(Ny):
        for z in range(Nz):
            if body[0, y, z]:
                continue

            # Neighbour macroscopic (x = 1) from f_next.
            rho_n = np.float32(0.0)
            mx = np.float32(0.0)
            my = np.float32(0.0)
            mz = np.float32(0.0)
            for i in range(19):
                fi = f_next[i, 1, y, z]
                rho_n += fi
                mx += np.float32(LATTICE_VELOCITIES_3D[i, 0]) * fi
                my += np.float32(LATTICE_VELOCITIES_3D[i, 1]) * fi
                mz += np.float32(LATTICE_VELOCITIES_3D[i, 2]) * fi
            if rho_n > np.float32(0.0):
                inv_rho_n = np.float32(1.0) / rho_n
            else:
                inv_rho_n = np.float32(0.0)
            ux_n = mx * inv_rho_n
            uy_n = my * inv_rho_n
            uz_n = mz * inv_rho_n
            usq_n = ux_n * ux_n + uy_n * uy_n + uz_n * uz_n

            # Boundary state: prescribed velocity, density extrapolated.
            rho_b = rho_n
            ux_b = u_in
            uy_b = np.float32(0.0)
            uz_b = np.float32(0.0)
            usq_b = ux_b * ux_b

            for i in range(19):
                cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                w = np.float32(LATTICE_WEIGHTS_3D[i])
                cu_n = cx * ux_n + cy * uy_n + cz * uz_n
                cu_b = cx * ux_b + cy * uy_b + cz * uz_b
                f_eq_n = w * rho_n * (
                    np.float32(1.0)
                    + cu_n / cs2
                    + (cu_n * cu_n) * inv_2cs4
                    - usq_n * inv_2cs2
                )
                f_eq_b = w * rho_b * (
                    np.float32(1.0)
                    + cu_b / cs2
                    + (cu_b * cu_b) * inv_2cs4
                    - usq_b * inv_2cs2
                )
                f_next[i, 0, y, z] = f_eq_b + (f_next[i, 1, y, z] - f_eq_n)


@njit(cache=True, fastmath=True)
def apply_guo_outflow(
    f_next: np.ndarray,
    body: np.ndarray,
    rho_target: np.float32,
) -> None:
    """Guo NEEM outflow at x = Nx - 1: prescribed pressure, extrapolated velocity.

    For each fluid cell at x = Nx - 1:
      1. Compute the interior neighbour's macroscopic moments from
         f_next[*, Nx - 2, y, z].
      2. Set the boundary state: rho_b = rho_target (1.0 = atmospheric),
         u_b = u_neighbor (velocity extrapolation).
      3. For each direction i:
            f_next[i, Nx-1, y, z] = f_eq_i(rho_b, u_b)
                                  + (f_next[i, Nx-2, y, z] - f_eq_i(rho_n, u_n))

    Replaces the zero-gradient COPY used by ``step_bgk_3d``. Zero-gradient
    forces both rho and u to be uniform across the outlet plane, which is
    unphysical and pollutes the wake. Pressure-prescribed + velocity-
    extrapolated lets the flow leave the domain naturally.
    """
    _, Nx, Ny, Nz = f_next.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for y in range(Ny):
        for z in range(Nz):
            if body[Nx - 1, y, z]:
                continue

            # Neighbour macroscopic (x = Nx - 2) from f_next.
            rho_n = np.float32(0.0)
            mx = np.float32(0.0)
            my = np.float32(0.0)
            mz = np.float32(0.0)
            for i in range(19):
                fi = f_next[i, Nx - 2, y, z]
                rho_n += fi
                mx += np.float32(LATTICE_VELOCITIES_3D[i, 0]) * fi
                my += np.float32(LATTICE_VELOCITIES_3D[i, 1]) * fi
                mz += np.float32(LATTICE_VELOCITIES_3D[i, 2]) * fi
            if rho_n > np.float32(0.0):
                inv_rho_n = np.float32(1.0) / rho_n
            else:
                inv_rho_n = np.float32(0.0)
            ux_n = mx * inv_rho_n
            uy_n = my * inv_rho_n
            uz_n = mz * inv_rho_n
            usq_n = ux_n * ux_n + uy_n * uy_n + uz_n * uz_n

            # Boundary state: prescribed pressure (rho), extrapolated velocity.
            rho_b = rho_target
            ux_b = ux_n
            uy_b = uy_n
            uz_b = uz_n
            usq_b = usq_n

            for i in range(19):
                cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                w = np.float32(LATTICE_WEIGHTS_3D[i])
                cu_n = cx * ux_n + cy * uy_n + cz * uz_n
                cu_b = cx * ux_b + cy * uy_b + cz * uz_b
                f_eq_n = w * rho_n * (
                    np.float32(1.0)
                    + cu_n / cs2
                    + (cu_n * cu_n) * inv_2cs4
                    - usq_n * inv_2cs2
                )
                f_eq_b = w * rho_b * (
                    np.float32(1.0)
                    + cu_b / cs2
                    + (cu_b * cu_b) * inv_2cs4
                    - usq_b * inv_2cs2
                )
                f_next[i, Nx - 1, y, z] = f_eq_b + (f_next[i, Nx - 2, y, z] - f_eq_n)


@njit(cache=True, fastmath=True)
def apply_regularised_outflow(
    f_next: np.ndarray,
    body: np.ndarray,
    rho_target: np.float32,
) -> None:
    """Latt-Chopard 2008 regularised outflow at x = Nx - 1.

    Drop-in replacement for ``apply_guo_outflow`` that survives at
    higher Re (~ 100 and above on the dev grid). Same prescribed-
    pressure / extrapolated-velocity boundary state; the difference
    is the reconstruction of the non-equilibrium populations.

    **The Guo NEEM failure mode it fixes.** Guo NEEM sets
    ``f_outflow = f_eq_b + (f_interior - f_eq_interior)``. The
    ``f_interior - f_eq_interior`` term carries the FULL non-
    equilibrium content of the interior, including non-hydrodynamic
    ghost moments. At higher Re the ghosts grow rapidly; eventually
    some ``f_i`` flips negative and the simulation blows up. This is
    the populations-go-negative signature documented in
    ``scripts/bake_3d_field.py`` for sphere_re100 (the bake is
    deferred for exactly this reason).

    **The regularisation.** Per Latt & Chopard 2008
    *Comp. Fluids* **37**, 159, the populations admit a Hermite
    expansion ``f_i = f_eq_i + f_i^neq``, and the leading-order
    non-equilibrium is exactly the second moment
    ``Pi^neq_{ab} = sum_i c_ia c_ib (f_i - f_eq_i)``. The regularised
    reconstruction projects ``f^neq`` onto this stress moment only,
    discarding everything else:

    .. math::

        f_i^{neq,reg} = \\frac{w_i}{2 c_s^4} (c_{ia} c_{ib} - c_s^2
        \\delta_{ab}) \\, \\Pi^{neq}_{ab}

    Then we set ``f_outflow = f_eq_b + f^neq_reg`` (with ``f_eq_b``
    at the prescribed boundary state). Off-diagonal cross-terms
    (``c_ia c_ib`` for ``a != b``) appear with the factor 2 from the
    Einstein-sum symmetry.

    The result: only the physical hydrodynamic stress survives at
    the outlet. Ghost moments cannot accumulate, populations stay
    positive at higher Re, and the bulk wake exits cleanly the same
    way it does with Guo NEEM at low Re (low-Re equivalence is
    pinned by ``tests/test_lbm_3d_outflow.py``).

    See also: Coreixas et al. 2017 *Phys. Rev. E* **96**, 033306
    (regularisation as a noise filter, lifts the Mach-number ceiling
    too) and Latt et al. 2020 *Phil. Trans. R. Soc. A* **378**
    20190559 (overview of stabilisation schemes including
    regularised LBM).
    """
    _, Nx, Ny, Nz = f_next.shape
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * 1.0 / 3.0))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for y in range(Ny):
        for z in range(Nz):
            if body[Nx - 1, y, z]:
                continue

            # Pass 1: interior moments rho_n, u_n from f_next[..., Nx-2, y, z].
            rho_n = np.float32(0.0)
            mx = np.float32(0.0)
            my = np.float32(0.0)
            mz = np.float32(0.0)
            for i in range(19):
                fi = f_next[i, Nx - 2, y, z]
                rho_n += fi
                mx += np.float32(LATTICE_VELOCITIES_3D[i, 0]) * fi
                my += np.float32(LATTICE_VELOCITIES_3D[i, 1]) * fi
                mz += np.float32(LATTICE_VELOCITIES_3D[i, 2]) * fi
            if rho_n > np.float32(0.0):
                inv_rho_n = np.float32(1.0) / rho_n
            else:
                inv_rho_n = np.float32(0.0)
            ux_n = mx * inv_rho_n
            uy_n = my * inv_rho_n
            uz_n = mz * inv_rho_n
            usq_n = ux_n * ux_n + uy_n * uy_n + uz_n * uz_n

            # Pass 2: build Pi^neq at the interior neighbour. Six
            # independent components (symmetric 3x3 stress tensor).
            Pi_xx = np.float32(0.0)
            Pi_yy = np.float32(0.0)
            Pi_zz = np.float32(0.0)
            Pi_xy = np.float32(0.0)
            Pi_xz = np.float32(0.0)
            Pi_yz = np.float32(0.0)
            for i in range(19):
                cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                w = np.float32(LATTICE_WEIGHTS_3D[i])
                cu_n = cx * ux_n + cy * uy_n + cz * uz_n
                f_eq_n = w * rho_n * (
                    np.float32(1.0)
                    + cu_n / cs2
                    + (cu_n * cu_n) * inv_2cs4
                    - usq_n * inv_2cs2
                )
                f_neq_n = f_next[i, Nx - 2, y, z] - f_eq_n
                Pi_xx += cx * cx * f_neq_n
                Pi_yy += cy * cy * f_neq_n
                Pi_zz += cz * cz * f_neq_n
                Pi_xy += cx * cy * f_neq_n
                Pi_xz += cx * cz * f_neq_n
                Pi_yz += cy * cz * f_neq_n

            # Pass 3: boundary equilibrium + regularised non-equilibrium.
            rho_b = rho_target
            ux_b = ux_n
            uy_b = uy_n
            uz_b = uz_n
            usq_b = usq_n

            for i in range(19):
                cx = np.float32(LATTICE_VELOCITIES_3D[i, 0])
                cy = np.float32(LATTICE_VELOCITIES_3D[i, 1])
                cz = np.float32(LATTICE_VELOCITIES_3D[i, 2])
                w = np.float32(LATTICE_WEIGHTS_3D[i])
                cu_b = cx * ux_b + cy * uy_b + cz * uz_b
                f_eq_b = w * rho_b * (
                    np.float32(1.0)
                    + cu_b / cs2
                    + (cu_b * cu_b) * inv_2cs4
                    - usq_b * inv_2cs2
                )
                # H_i^(2) : Pi^neq, with the diagonal terms carrying
                # the (c_ia c_ib - cs2 delta_ab) Hermite trace
                # subtraction and the off-diagonal terms doubled via
                # the symmetry c_ia c_ib + c_ib c_ia = 2 c_ia c_ib.
                H_dd_Pi = (
                    (cx * cx - cs2) * Pi_xx
                    + (cy * cy - cs2) * Pi_yy
                    + (cz * cz - cs2) * Pi_zz
                    + np.float32(2.0) * cx * cy * Pi_xy
                    + np.float32(2.0) * cx * cz * Pi_xz
                    + np.float32(2.0) * cy * cz * Pi_yz
                )
                f_neq_reg = w * inv_2cs4 * H_dd_Pi
                f_next[i, Nx - 1, y, z] = f_eq_b + f_neq_reg


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def _make_sphere_mask(Nx: int, Ny: int, Nz: int,
                       cx: int, cy: int, cz: int,
                       radius: float) -> np.ndarray:
    """Boolean mask shape (Nx, Ny, Nz), True inside the sphere."""
    xs = np.arange(Nx)[:, None, None]
    ys = np.arange(Ny)[None, :, None]
    zs = np.arange(Nz)[None, None, :]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) <= radius * radius


def init_population(Nx: int, Ny: int, Nz: int,
                    u_in: float, dtype=np.float32) -> np.ndarray:
    """Initial f at uniform velocity u_in along +x, rho = 1 everywhere."""
    rho = np.ones((Nx, Ny, Nz), dtype=dtype)
    u = np.zeros((3, Nx, Ny, Nz), dtype=dtype)
    u[0] = u_in
    return equilibrium_3d(rho, u).astype(dtype)


def run_channel_smoke(
    Nx: int = 80,
    Ny: int = 32,
    Nz: int = 32,
    u_in: float = 0.05,
    nu: float = 0.01,
    n_steps: int = 400,
    body: np.ndarray | None = None,
    wall_links=None,
    use_guo_neem: bool = False,
    rho_outflow: float = 1.0,
    outflow_scheme: str = "guo",
    progress_callback=None,
):
    """Run the 3D channel-flow smoke and return (rho, ux, uy, uz, diag).

    No body unless one is passed. The default geometry is a plain
    channel: bounce-back on y = 0 / Ny-1, periodic in z, inflow at
    x = 0, zero-gradient at x = Nx-1. Expected steady solution is
    a parabolic plane-Poiseuille profile in y, uniform in z.

    If ``wall_links`` is provided (a ``WallLinkList`` from
    ``src/lbm_3d_bouzidi.py``), the full-way bounce-back inside
    ``step_bgk_3d`` is corrected to Bouzidi linear interpolation in a
    post-pass per step. At q = 0.5 the Bouzidi correction is a no-op
    so passing wall_links with all-q=0.5 yields identical output to
    full-way -- the q=0.5 sanity test pins this.

    If ``use_guo_neem`` is True, swap the inline equilibrium-inflow +
    zero-gradient-outflow path for the Guo non-equilibrium extrapolation
    pair (``apply_guo_inflow`` + ``apply_guo_outflow``) running as post-
    passes after the pure-bulk kernel. ``rho_outflow`` is the prescribed
    outlet density (defaults to 1.0 = atmospheric). The two boundary
    pathways are independent: the legacy path stays bit-for-bit
    untouched when ``use_guo_neem`` is False, which is what every
    existing test relies on.

    diag is a dict with mass-conservation, peak velocity, and the
    measured centreline-to-mean ratio (~ 1.5 for fully-developed
    plane Poiseuille).
    """
    if body is None:
        body = np.zeros((Nx, Ny, Nz), dtype=np.bool_)
    omega = np.float32(1.0 / (3.0 * nu + 0.5))
    f = init_population(Nx, Ny, Nz, u_in)
    f_next = f.copy()

    mass_initial = float(f.sum())

    # Pre-import the Bouzidi correction so the import cost is paid once
    # per call (not per step) AND the function only loads when actually
    # using Bouzidi -- the full-way path stays free of numba JIT for
    # the correction kernel.
    if wall_links is not None:
        from src.lbm_3d_bouzidi import apply_bouzidi_correction
    else:
        apply_bouzidi_correction = None

    u_in_f32 = np.float32(u_in)
    rho_outflow_f32 = np.float32(rho_outflow)

    # Choose the outflow post-pass once outside the hot loop. The
    # regularised scheme survives higher Re (filters ghost moments at
    # the outlet plane); Guo NEEM stays the default for legacy bake
    # presets. See ``apply_regularised_outflow`` docstring for the
    # Latt-Chopard 2008 reference and the Re-100 motivation.
    if outflow_scheme not in ("guo", "regularised"):
        raise ValueError(
            f"outflow_scheme must be 'guo' or 'regularised', "
            f"got {outflow_scheme!r}."
        )
    _apply_outflow = (
        apply_regularised_outflow if outflow_scheme == "regularised"
        else apply_guo_outflow
    )

    for step in range(n_steps):
        if use_guo_neem:
            # Pure-bulk step (no inline boundary handling at x faces),
            # then Guo NEEM post-passes for inflow and outflow.
            step_bgk_3d_pure_bulk(f, f_next, body, omega)
            apply_guo_inflow(f_next, body, u_in_f32)
            _apply_outflow(f_next, body, rho_outflow_f32)
        else:
            # Legacy path: inline equilibrium-inflow + zero-gradient
            # outflow inside step_bgk_3d. Bit-for-bit unchanged.
            step_bgk_3d(f, f_next, body, omega, u_in_f32)

        if apply_bouzidi_correction is not None:
            # Bouzidi reads PRE-step populations (still in f) to
            # recompute f_tilde locally at wall links, then overrides
            # specific f_next[opp, x_f] entries. Runs AFTER the Guo
            # NEEM passes so that at the rare wall link sitting at
            # x=0 / x=Nx-1, Bouzidi's wall-aware override wins.
            apply_bouzidi_correction(
                f, f_next, body,
                wall_links.x, wall_links.y, wall_links.z,
                wall_links.dir, wall_links.q,
                omega, u_in_f32,
            )
        f, f_next = f_next, f
        if progress_callback is not None and (step % max(1, n_steps // 20) == 0):
            progress_callback(step / n_steps, f"3D step {step}/{n_steps}")

    rho, ux, uy, uz = macroscopic_3d(f)
    # Zero out macroscopic velocities inside the body. The kernel skips
    # collision on solid cells, so their populations stay at the initial
    # f_eq(rho=1, u=u_in)  -- read back through macroscopic_3d this looks
    # like the inflow velocity is leaking through the solid. Physically
    # solid cells have no fluid flow; for the downstream consumers
    # (smoke-particle trilerp interpolation, slice plots) we need that
    # to be reflected in the returned arrays. Bouzidi BB would handle
    # this implicitly by tracking wall positions; for the simpler
    # full-way BB scaffold a post-pass zero is the right fix.
    if body is not None:
        ux = np.where(body, np.float32(0.0), ux)
        uy = np.where(body, np.float32(0.0), uy)
        uz = np.where(body, np.float32(0.0), uz)
    mass_final = float(f.sum())

    # Centerline (mid-z, mid-y) profile through x; centre y-profile at
    # mid-channel-x. Plane-Poiseuille between bounce-back walls in y:
    # the profile is parabolic with u_peak = 1.5 * u_mean.
    mid_x = Nx // 2
    mid_z = Nz // 2
    y_profile = ux[mid_x, :, mid_z]
    # Exclude wall cells (y = 0 and y = Ny-1) from the mean.
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
        "omega": float(omega),
        "n_steps": n_steps,
        "shape": (Nx, Ny, Nz),
    }
    return rho, ux, uy, uz, diag


__all__ = [
    "LATTICE_VELOCITIES_3D",
    "LATTICE_WEIGHTS_3D",
    "OPPOSITE_3D",
    "CS2_3D",
    "macroscopic_3d",
    "equilibrium_3d",
    "step_bgk_3d",
    "init_population",
    "run_channel_smoke",
    "_make_sphere_mask",
]
