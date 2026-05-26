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
    progress_callback=None,
):
    """Run the 3D channel-flow smoke and return (rho, ux, uy, uz, diag).

    No body unless one is passed. The default geometry is a plain
    channel: bounce-back on y = 0 / Ny-1, periodic in z, inflow at
    x = 0, zero-gradient at x = Nx-1. Expected steady solution is
    a parabolic plane-Poiseuille profile in y, uniform in z.

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

    for step in range(n_steps):
        step_bgk_3d(f, f_next, body,
                    omega, np.float32(u_in))
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
