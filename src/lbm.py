"""2D Lattice Boltzmann Method (LBM) solver core.

D2Q9 lattice, BGK collision, bounce-back boundaries. Pure-NumPy reference
implementations PLUS a Numba ``@njit`` fused-step function for production runs.

Reference: Kruger et al., "The Lattice Boltzmann Method: Principles and Practice"
(Springer, 2017) -- the standard textbook for this method.
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# D2Q9 lattice
#
# Index convention (compass + rest):
#
#       6   2   5
#         \ | /
#       3 - 0 - 1
#         / | \
#       7   4   8
# ---------------------------------------------------------------------------

# Discrete velocity vectors c_i = (cx_i, cy_i) in lattice units (one cell per
# step per unit speed). Shape (9, 2). int8 is enough -- components are -1, 0, 1.
LATTICE_VELOCITIES = np.array(
    [
        [0, 0],    # 0: rest
        [1, 0],    # 1: east
        [0, 1],    # 2: north
        [-1, 0],   # 3: west
        [0, -1],   # 4: south
        [1, 1],    # 5: NE
        [-1, 1],   # 6: NW
        [-1, -1],  # 7: SW
        [1, -1],   # 8: SE
    ],
    dtype=np.int8,
)

# Lattice weights w_i. Chosen so the second-order moments of the equilibrium
# distribution match the Maxwell-Boltzmann distribution -- which is what makes
# the recovered macroscopic equations be Navier-Stokes (to second order in Mach).
# They sum to 1, and the weights for symmetric directions are equal.
LATTICE_WEIGHTS = np.array(
    [
        4 / 9,                            # 0: rest
        1 / 9, 1 / 9, 1 / 9, 1 / 9,       # 1-4: cardinals
        1 / 36, 1 / 36, 1 / 36, 1 / 36,   # 5-8: diagonals
    ],
    dtype=np.float64,
)

# For every velocity i, OPPOSITE[i] is the index of -c_i. Used by bounce-back
# boundaries: a particle moving toward a wall in direction i is reflected back
# along OPPOSITE[i]. Pairs:
#   1 (east)  <-> 3 (west)
#   2 (north) <-> 4 (south)
#   5 (NE)    <-> 7 (SW)
#   6 (NW)    <-> 8 (SE)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)

# Speed of sound squared in lattice units. Falls out of the lattice normalization:
# c_s^2 = sum_i (w_i * cx_i^2) = 1/3. Appears throughout the equilibrium relation
# and the equation of state (pressure P = rho * c_s^2 in this isothermal model).
CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# Equilibrium distribution (Step 2b)
# ---------------------------------------------------------------------------

def equilibrium(rho, u):
    """D2Q9 equilibrium distribution f_eq for given density and velocity.

    Implements the standard second-order Hermite expansion:

        f_eq_i = w_i * rho * (1
                              + (c_i . u) / cs^2
                              + (c_i . u)^2 / (2 * cs^4)
                              - u . u / (2 * cs^2))

    The truncation at second order in u is what makes the recovered macroscopic
    equation be Navier-Stokes (to leading order in Mach number). LBM is only
    valid for |u| << cs (typically |u| < 0.1 in lattice units) -- exceeding this
    causes negative populations and the simulation diverges.

    Parameters
    ----------
    rho : float or ndarray of shape (Nx, Ny)
        Macroscopic density. Scalar for single-point use, 2-D field for solver.
    u : ndarray of shape (2,) or (2, Nx, Ny)
        Macroscopic velocity (x, y). Same dimensionality choice as ``rho``.

    Returns
    -------
    f_eq : ndarray of shape (9,) or (9, Nx, Ny)
        Equilibrium distribution. First axis is the velocity direction.
    """
    u = np.asarray(u, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    # Lattice velocity components as float arrays for arithmetic with floats.
    cx = LATTICE_VELOCITIES[:, 0].astype(np.float64)
    cy = LATTICE_VELOCITIES[:, 1].astype(np.float64)

    if u.ndim == 1:
        # Single-point case: u has shape (2,).
        u_dot_c = cx * u[0] + cy * u[1]    # shape (9,)
        u_sq = u[0] ** 2 + u[1] ** 2       # scalar
        f_eq = LATTICE_WEIGHTS * rho * (
            1.0
            + u_dot_c / CS2
            + 0.5 * (u_dot_c / CS2) ** 2
            - 0.5 * u_sq / CS2
        )
    else:
        # Field case: u has shape (2, Nx, Ny). Reshape lattice arrays for broadcasting.
        cx_b = cx[:, None, None]   # shape (9, 1, 1)
        cy_b = cy[:, None, None]
        u_dot_c = cx_b * u[0] + cy_b * u[1]  # shape (9, Nx, Ny)
        u_sq = u[0] ** 2 + u[1] ** 2          # shape (Nx, Ny)
        f_eq = (
            LATTICE_WEIGHTS[:, None, None]
            * rho[None, :, :]
            * (
                1.0
                + u_dot_c / CS2
                + 0.5 * (u_dot_c / CS2) ** 2
                - 0.5 * u_sq[None, :, :] / CS2
            )
        )
    return f_eq


# ---------------------------------------------------------------------------
# Macroscopic moments (helper for everything: collision, viz, force calc)
# ---------------------------------------------------------------------------

def macroscopic(f):
    """Compute density and velocity from a distribution f.

    These are the discrete versions of the kinetic-theory moments:
        rho = sum_i f_i
        rho * u = sum_i f_i * c_i

    Parameters
    ----------
    f : ndarray of shape (9,) or (9, Nx, Ny)
        Distribution function.

    Returns
    -------
    rho : ndarray of shape () or (Nx, Ny)
    u   : ndarray of shape (2,) or (2, Nx, Ny)
    """
    cx = LATTICE_VELOCITIES[:, 0].astype(np.float64)
    cy = LATTICE_VELOCITIES[:, 1].astype(np.float64)

    if f.ndim == 1:
        rho = f.sum()
        ux = np.sum(f * cx) / rho
        uy = np.sum(f * cy) / rho
        u = np.array([ux, uy], dtype=np.float64)
    else:
        rho = f.sum(axis=0)                                   # shape (Nx, Ny)
        ux = np.sum(f * cx[:, None, None], axis=0) / rho      # shape (Nx, Ny)
        uy = np.sum(f * cy[:, None, None], axis=0) / rho
        u = np.stack([ux, uy], axis=0)                        # shape (2, Nx, Ny)
    return rho, u


# ---------------------------------------------------------------------------
# BGK collision step (Step 2c)
# ---------------------------------------------------------------------------

def collide(f, tau):
    """BGK collision: relax f toward equilibrium with relaxation time tau.

        f_post_i = f_i  -  (1 / tau) * (f_i  -  f_eq_i)

    The relaxation time controls kinematic viscosity in lattice units:
        nu = cs^2 * (tau - 0.5)

    LBM is stable for tau > 0.5; values near 0.5 give low viscosity (high
    Reynolds, may need stabilization or finer grids), tau >= 1 is comfortably
    stable. Collision is local (no neighbor lookups) and exactly conserves
    mass and momentum at every cell -- those are the moments matched by f_eq.

    Parameters
    ----------
    f   : ndarray of shape (9, Nx, Ny)  -- distribution function before collision
    tau : float                         -- relaxation time

    Returns
    -------
    f_post : ndarray of shape (9, Nx, Ny) -- distribution after collision
    """
    rho, u = macroscopic(f)
    f_eq = equilibrium(rho, u)
    return f - (f - f_eq) / tau


# ---------------------------------------------------------------------------
# Streaming step (Step 2d)
# ---------------------------------------------------------------------------

def stream(f):
    """Streaming: each f_i is shifted by one cell along velocity c_i.

        f_post[i, x, y] = f_pre[i, x - cx[i], y - cy[i]]

    In words: the new value at (x, y) in direction i comes from the upstream
    neighbor in direction i. Equivalently, the old value at (x, y) moving in
    direction i lands at (x + cx[i], y + cy[i]).

    Boundaries here are periodic (``np.roll`` wraps). Solid walls are imposed
    separately by ``bounce_back`` applied AFTER streaming -- that two-stage
    pattern keeps each function single-purpose and is the standard LBM idiom.

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny)

    Returns
    -------
    f_post : ndarray of shape (9, Nx, Ny)
    """
    f_post = np.empty_like(f)
    # axis=0 of f[i] is x, axis=1 is y. np.roll with positive shift moves entries
    # in the positive direction along that axis.
    for i in range(9):
        cx_i = int(LATTICE_VELOCITIES[i, 0])
        cy_i = int(LATTICE_VELOCITIES[i, 1])
        f_post[i] = np.roll(f[i], shift=(cx_i, cy_i), axis=(0, 1))
    return f_post


# ---------------------------------------------------------------------------
# Bounce-back boundary on solid cells (Step 2e)
# ---------------------------------------------------------------------------

def bounce_back(f, solid_mask):
    """Halfway bounce-back: at each solid cell, swap f_i with f_OPPOSITE[i].

    Apply between collision and streaming:
        f = collide(f, tau)
        f = bounce_back(f, solid_mask)
        f = stream(f)

    The wall is conceptually at the half-cell between the solid and the
    adjacent fluid -- that gives second-order spatial accuracy at flat walls
    and is the standard choice for cylinder-class validation problems.

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny)
    solid_mask : ndarray of bool, shape (Nx, Ny)
        True at cells inside the obstacle.

    Returns
    -------
    f_post : ndarray of shape (9, Nx, Ny)
    """
    f_post = f.copy()
    # Snapshot of the original values at solid cells. Without this, the swap
    # would read partially-overwritten values and corrupt half the directions.
    # f[:, solid_mask] uses NumPy's bool-mask fancy indexing: shape (9, n_solid).
    f_solid_pre = f[:, solid_mask].copy()
    for i in range(9):
        opp = int(OPPOSITE[i])
        f_post[i, solid_mask] = f_solid_pre[opp]
    return f_post


# ---------------------------------------------------------------------------
# JIT-compiled fused step function (production hot path)
# ---------------------------------------------------------------------------
# All of the above functions are kept as pure-NumPy reference implementations
# (used by unit tests, easy to read, fast enough for ad-hoc work).
#
# For long runs (cylinder validation: 30-50k timesteps) we use the fused step
# below. It performs the entire timestep -- collide, force calc, bounce-back,
# streaming, inflow/outflow overrides -- in one Numba ``@njit`` function with
# explicit loops. Numpy's ``np.roll`` and bool-mask fancy indexing aren't
# JIT-compatible, so streaming and bounce-back are rewritten as explicit
# indexed shifts here.
#
# Equivalence with the pure-NumPy reference is checked by a unit test in
# tests/test_lbm.py -- if you change either, the test must still pass.

@njit(cache=True)
def step_njit_with_force(f, tau, solid_mask, f_inflow, inflow_dirs, outflow_dirs):
    """One fused LBM timestep + momentum-exchange force calculation.

    Performs (equivalent to pure-NumPy reference path):
        f_post_coll = collide(f, tau)
        (Fx, Fy)    = momentum_exchange_force(f_post_coll, solid_mask)
        f_bounced   = bounce_back(f_post_coll, solid_mask)
        f_new       = stream(f_bounced)
        f_new[inflow_dirs, 0, :] = f_inflow[inflow_dirs, None]
        f_new[outflow_dirs, -1, :] = f_new[outflow_dirs, -2, :]

    First call triggers JIT compilation (~3-5 sec). Subsequent calls run at
    roughly ~0.5 ms/step on a 300x100 grid -- about 25x faster than the pure-
    NumPy version.

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny), float64
        Distribution function.
    tau : float
        BGK relaxation time. Must be > 0.5.
    solid_mask : ndarray of shape (Nx, Ny), bool
        True at obstacle cells.
    f_inflow : ndarray of shape (9,), float64
        Equilibrium distribution at the inflow state.
    inflow_dirs : ndarray of int, e.g. ``np.array([1, 5, 8], dtype=np.int32)``
        Directions to override at the left boundary (x=0).
    outflow_dirs : ndarray of int, e.g. ``np.array([3, 6, 7], dtype=np.int32)``
        Directions to zero-gradient-extrapolate at the right boundary (x=Nx-1).

    Returns
    -------
    f_new : ndarray of shape (9, Nx, Ny)
    Fx, Fy : float
        Force on the obstacle in lattice units.
    """
    Nx = f.shape[1]
    Ny = f.shape[2]

    # D2Q9 constants -- defined locally as float64 arrays so Numba sees clean types.
    cx_arr = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    cy_arr = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    w_arr = np.array([
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    ])
    opp_arr = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
    inv_tau = 1.0 / tau

    # === Step 1: collide (BGK) ===
    # Per-cell inner loop computes rho, u, equilibrium, and relaxes f toward it.
    f_post_coll = np.empty_like(f)
    for x in range(Nx):
        for y in range(Ny):
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[i, x, y]
                rho += fi
                mx += fi * cx_arr[i]
                my += fi * cy_arr[i]
            ux = mx / rho
            uy = my / rho
            u2 = ux * ux + uy * uy
            for i in range(9):
                cu = cx_arr[i] * ux + cy_arr[i] * uy
                # cs^2 = 1/3, so 1/cs^2 = 3, 1/(2*cs^4) = 4.5, 1/(2*cs^2) = 1.5.
                feq = w_arr[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
                f_post_coll[i, x, y] = f[i, x, y] - (f[i, x, y] - feq) * inv_tau

    # === Step 2: momentum-exchange force from f_post_coll ===
    # F = sum over wall-links of 2 * c_i * f_i (post-collision, at fluid cell).
    Fx = 0.0
    Fy = 0.0
    for i in range(1, 9):  # skip rest direction
        cxi = cx_arr[i]
        cyi = cy_arr[i]
        cxi_int = int(cxi)
        cyi_int = int(cyi)
        for x in range(Nx):
            for y in range(Ny):
                if not solid_mask[x, y]:
                    xn = (x + cxi_int) % Nx
                    yn = (y + cyi_int) % Ny
                    if solid_mask[xn, yn]:
                        contrib = 2.0 * f_post_coll[i, x, y]
                        Fx += cxi * contrib
                        Fy += cyi * contrib

    # === Step 3: bounce-back at solid cells ===
    # At solid cells: f_new[i] = f_post_coll[opp[i]] (swap with opposite direction).
    # At fluid cells: f_new[i] = f_post_coll[i] (unchanged).
    f_after_bb = np.empty_like(f_post_coll)
    for i in range(9):
        opp_i = opp_arr[i]
        for x in range(Nx):
            for y in range(Ny):
                if solid_mask[x, y]:
                    f_after_bb[i, x, y] = f_post_coll[opp_i, x, y]
                else:
                    f_after_bb[i, x, y] = f_post_coll[i, x, y]

    # === Step 4: streaming with periodic wrap ===
    # f_new[i, x, y] = f_after_bb[i, x - cx_i, y - cy_i] (mod Nx/Ny).
    f_new = np.empty_like(f_after_bb)
    for i in range(9):
        cxi_int = int(cx_arr[i])
        cyi_int = int(cy_arr[i])
        for x in range(Nx):
            for y in range(Ny):
                xs = (x - cxi_int) % Nx
                ys = (y - cyi_int) % Ny
                f_new[i, x, y] = f_after_bb[i, xs, ys]

    # === Step 5: inflow override at left boundary (x = 0) ===
    for k in range(inflow_dirs.shape[0]):
        i = int(inflow_dirs[k])
        for y in range(Ny):
            f_new[i, 0, y] = f_inflow[i]

    # === Step 6: outflow override at right boundary (x = Nx-1, zero-gradient) ===
    for k in range(outflow_dirs.shape[0]):
        i = int(outflow_dirs[k])
        for y in range(Ny):
            f_new[i, Nx - 1, y] = f_new[i, Nx - 2, y]

    return f_new, Fx, Fy
