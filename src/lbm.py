"""2D Lattice Boltzmann Method (LBM) solver core.

D2Q9 lattice, BGK collision, bounce-back boundaries. Pure-NumPy reference
implementations PLUS a Numba ``@njit`` fused-step function for production runs.

Reference: Kruger et al., "The Lattice Boltzmann Method: Principles and Practice"
(Springer, 2017) -- the standard textbook for this method.
"""
import numpy as np
from numba import njit

# prange aliased to range so the loop syntax below stays the same. We
# dropped numba parallel=True after Streamlit Cloud's container produced
# RuntimeError("Cannot set NUMBA_NUM_THREADS to a [...]") at JIT-compile
# time -- something in Cloud's environment mutates Numba's thread config
# between import and compile, and the cleanest fix is to remove threading
# from the equation entirely. Local loses ~2-3x parallel speedup on the
# LBM step; Cloud's 1 vCPU was already serial in practice.
prange = range

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
# Zou-He velocity inflow at the left boundary (x = 0)
# ---------------------------------------------------------------------------
# Replaces the previous "equilibrium inflow" trick
#     f[inflow_dirs, 0, :] = f_inflow[inflow_dirs, None]
# which sets the unknown populations to their equilibrium values but does NOT
# enforce the prescribed velocity exactly. The mismatch leaks mass into / out
# of the domain at a few percent per 1000 steps (measured: 2.7% / 8000 steps
# on the production cylinder Re=100 case before this BC).
#
# Zou & He (Phys. Fluids 1997) close the system at a velocity-prescribed
# boundary by:
#   1. Solving for boundary density from rho = (f0+f2+f4+2(f3+f6+f7))/(1-ux),
#      which comes from substituting the prescribed ux into the rho+momentum
#      sum identities.
#   2. Using non-equilibrium bounce-back on the normal pair (f1, f3) to pin
#      one of the three unknowns: f1 = f3 + (2/3) rho ux.
#   3. Closing the remaining two unknowns (f5, f8) from the y-momentum and
#      stress constraints.
#
# Result: exact mass conservation at the inflow boundary, second-order
# accurate in space (matches the rest of the LBM order), no anisotropy from
# the f_inflow ghost values feeding into the next collision.

def zou_he_inflow(f, ux_in, uy_in):
    """Apply Zou-He velocity inflow at x = 0 to a pure-NumPy f-field.

    Used by the test reference path and as documentation for the JIT
    implementations. Modifies f in place at x = 0 (column 0 only).

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny)
        Post-streaming populations. Reads f[0,2,3,4,6,7] at x=0,
        overwrites f[1,5,8] at x=0.
    ux_in, uy_in : float
        Prescribed inflow velocity at x = 0.
    """
    f0 = f[0, 0, :]
    f2 = f[2, 0, :]
    f3 = f[3, 0, :]
    f4 = f[4, 0, :]
    f6 = f[6, 0, :]
    f7 = f[7, 0, :]
    rho_w = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_in)
    f[1, 0, :] = f3 + (2.0 / 3.0) * rho_w * ux_in
    f[5, 0, :] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * ux_in + 0.5 * rho_w * uy_in
    f[8, 0, :] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_w * ux_in - 0.5 * rho_w * uy_in


# Bouzidi interpolated bounce-back (Bouzidi, Firdaouss, Lallemand 2001).
# ---------------------------------------------------------------------------
# Halfway bounce-back assumes the wall sits exactly at the half-cell distance
# between every fluid cell and its solid neighbor. For curved bodies (cylinder,
# ellipse, NACA) this voxelizes the boundary to a staircase, shifting the
# effective wall position by up to half a cell and inflating Cd by tens of
# percent. Bouzidi BB replaces the per-cell swap with a per-LINK interpolation
# that uses the exact analytic wall fraction q for each wall-crossing link.
#
# The wall fraction q in [0, 1] is the fluid-to-wall distance along c_i,
# normalized to |c_i|:
#   * q = 0.5 reduces to halfway bounce-back (the swap-based formula is exact)
#   * q < 0.5 wall is closer to the fluid cell than half-cell
#   * q > 0.5 wall is farther into the link than half-cell
#
# Linear interpolation (sufficient for second-order LBM, ~10x cheaper than
# quadratic; cf. Yu/Mei/Luo/Shyy 2003 review):
#   q >= 0.5: f_{-i}^new(x_f) = (1/2q) f_i_post(x_f) + ((2q-1)/2q) f_{-i}_post(x_f)
#   q  < 0.5: f_{-i}^new(x_f) = 2q  f_i_post(x_f) + (1-2q)        f_i_post(x_f - c_i)
#
# Per-shape Bouzidi q-fields are computed in src/shapes.py and passed into the
# JIT step functions. A q-field of all -1 (returned by no_bouzidi_q_field)
# selects halfway BB at every link -- bit-for-bit identical to the pre-Bouzidi
# solver, no performance penalty beyond the conditional check.

def bouzidi_correction(f_after_bb, f_new, solid_mask, q_field):
    """Pure-NumPy Bouzidi linear-interpolation correction at wall links.

    Mirrors the JIT block inlined into each step_njit_* function. Used by
    the test reference path (so JIT-vs-NumPy equivalence holds with Bouzidi
    too) and as documentation.

    Parameters
    ----------
    f_after_bb : ndarray (9, Nx, Ny)
        Post-collision populations (with halfway BB swap applied on solid
        cells). At fluid cells this equals f_post_coll.
    f_new : ndarray (9, Nx, Ny)
        Already-streamed populations. This function OVERWRITES the wall-link
        entries in place.
    solid_mask : ndarray (Nx, Ny) bool
    q_field : ndarray (Nx, Ny, 9) float64
        See src/shapes.py for the structure. q <= 0 means "no Bouzidi" for
        that link (halfway BB result, already produced by streaming, is kept).
    """
    Nx, Ny = solid_mask.shape
    cx_arr = LATTICE_VELOCITIES[:, 0]
    cy_arr = LATTICE_VELOCITIES[:, 1]
    for i in range(1, 9):
        cxi = int(cx_arr[i])
        cyi = int(cy_arr[i])
        opp_i = int(OPPOSITE[i])
        for x in range(Nx):
            for y in range(Ny):
                if solid_mask[x, y]:
                    continue
                q_val = q_field[x, y, i]
                if q_val <= 0.0:
                    continue
                f_i = f_after_bb[i, x, y]
                f_opp_at_xf = f_after_bb[opp_i, x, y]
                if q_val >= 0.5:
                    inv_2q = 1.0 / (2.0 * q_val)
                    f_new[opp_i, x, y] = (
                        inv_2q * f_i
                        + (2.0 * q_val - 1.0) * inv_2q * f_opp_at_xf
                    )
                else:
                    xb = x - cxi
                    yb = y - cyi
                    if 0 <= xb < Nx and 0 <= yb < Ny:
                        f_i_back = f_after_bb[i, xb, yb]
                        f_new[opp_i, x, y] = (
                            2.0 * q_val * f_i
                            + (1.0 - 2.0 * q_val) * f_i_back
                        )
                    else:
                        f_new[opp_i, x, y] = f_i


# Zou-He pressure outflow at x = Nx-1, paired with the velocity inflow above.
# Prescribes rho_w = 1 (freestream reference pressure). The two BCs together
# close the mass balance: inflow injection rate matches outflow extraction
# rate at steady state, because both boundaries respond to the same physical
# constraints. Replaces the previous "zero-gradient outflow"
#     f[outflow_dirs, -1, :] = f[outflow_dirs, -2, :]
# which under-extracts when the channel develops back-pressure (e.g. behind
# a body), leaving the inflow over-injecting relative to outflow extraction.
# uy at the outflow is zero-gradient extrapolated from x=Nx-2 so the wake
# can pass through without being forced to zero transverse velocity.

def zou_he_outflow_pressure(f, rho_out=1.0):
    """Apply Zou-He pressure outflow at x = Nx-1 to a pure-NumPy f-field.

    Modifies f in place at x = Nx-1 (last column only).

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny)
        Post-streaming populations. Reads f[0,1,2,4,5,8] at x=Nx-1,
        overwrites f[3,6,7] at x=Nx-1. uy at the boundary is extrapolated
        from the previous column (zero-gradient on the transverse velocity).
    rho_out : float, default 1.0
        Prescribed outflow density (freestream reference).
    """
    f0 = f[0, -1, :]
    f1 = f[1, -1, :]
    f2 = f[2, -1, :]
    f4 = f[4, -1, :]
    f5 = f[5, -1, :]
    f8 = f[8, -1, :]
    # ux from prescribed rho_w at outflow: derived from mass + x-momentum balance.
    ux_w = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out - 1.0
    # uy extrapolated from interior so transverse-velocity waves pass through
    # instead of reflecting back.
    rho_interior = f[:, -2, :].sum(axis=0)
    uy_w = (f[2, -2, :] - f[4, -2, :] + f[5, -2, :] + f[6, -2, :]
            - f[7, -2, :] - f[8, -2, :]) / rho_interior
    f[3, -1, :] = f1 - (2.0 / 3.0) * rho_out * ux_w
    f[6, -1, :] = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux_w - 0.5 * rho_out * uy_w
    f[7, -1, :] = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux_w + 0.5 * rho_out * uy_w


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

@njit(fastmath=True)
def step_njit_with_force(f, tau, solid_mask, q_field, f_inflow, inflow_dirs, outflow_dirs):
    """One fused LBM timestep + momentum-exchange force calculation.

    Performs (equivalent to pure-NumPy reference path):
        f_post_coll = collide(f, tau)
        (Fx, Fy)    = momentum_exchange_force(f_post_coll, solid_mask)
        f_bounced   = bounce_back(f_post_coll, solid_mask)
        f_new       = stream(f_bounced)
        zou_he_inflow(f_new, ux_in, uy_in)       # mass-conserving left BC
        f_new[outflow_dirs, -1, :] = f_new[outflow_dirs, -2, :]

    Compiled with ``fastmath=True`` only -- ``parallel=True`` was stripped after
    Streamlit Cloud's container produced a NUMBA_NUM_THREADS RuntimeError at
    JIT-compile time. ``prange`` in this file is aliased to ``range`` (top of
    module) so the loop syntax stays unchanged. Warm-step throughput on a
    240x80 grid is ~0.2 ms/step serial; local loses the ~2-3x parallel speedup
    but Cloud's 1 vCPU was already serial in practice.

    First call triggers JIT compilation (~5-8 sec). Subsequent calls reuse the
    cached binary.

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

    # === Step 1: collide (BGK) -- per-cell, embarrassingly parallel over x ===
    f_post_coll = np.empty_like(f)
    for x in prange(Nx):
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

    # === Step 2: momentum-exchange force (Bouzidi-aware) -- kept serial ===
    # For halfway BB (q <= 0) the standard formula F_link = 2 c_i f_i_post(x_f)
    # applies. For Bouzidi (q > 0) the correct momentum transfer is
    #   F_link = c_i [f_i_post(x_f) + f_-i_new(x_f)]
    # where f_-i_new is the Bouzidi-interpolated population that will land at
    # x_f after the BC step (Mei, Yu, Shyy, Luo 2002 -- momentum exchange for
    # Bouzidi BB). At q = 0.5 the Bouzidi result reduces to f_i_post and the
    # formula collapses back to 2 c_i f_i_post, so this is a strict
    # generalization. Tried prange on x with scalar reduction: Numba couldn't
    # infer the conditional += pattern and silently returned 0.0. Force calc
    # is ~5% of step time; serial is fine.
    Fx = 0.0
    Fy = 0.0
    for i in range(1, 9):
        cxi = cx_arr[i]
        cyi = cy_arr[i]
        cxi_int = int(cxi)
        cyi_int = int(cyi)
        opp_i = opp_arr[i]
        for x in range(Nx):
            for y in range(Ny):
                if not solid_mask[x, y]:
                    xn = (x + cxi_int) % Nx
                    yn = (y + cyi_int) % Ny
                    if solid_mask[xn, yn]:
                        f_i_post = f_post_coll[i, x, y]
                        q_val = q_field[x, y, i]
                        if q_val <= 0.0:
                            contrib = 2.0 * f_i_post
                        elif q_val >= 0.5:
                            inv_2q = 1.0 / (2.0 * q_val)
                            f_minus_i_new = (
                                inv_2q * f_i_post
                                + (2.0 * q_val - 1.0) * inv_2q * f_post_coll[opp_i, x, y]
                            )
                            contrib = f_i_post + f_minus_i_new
                        else:
                            xb = x - cxi_int
                            yb = y - cyi_int
                            if 0 <= xb < Nx and 0 <= yb < Ny:
                                f_minus_i_new = (
                                    2.0 * q_val * f_i_post
                                    + (1.0 - 2.0 * q_val) * f_post_coll[i, xb, yb]
                                )
                                contrib = f_i_post + f_minus_i_new
                            else:
                                contrib = 2.0 * f_i_post
                        Fx += cxi * contrib
                        Fy += cyi * contrib

    # === Step 3: bounce-back -- per-cell, parallel over x ===
    f_after_bb = np.empty_like(f_post_coll)
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[opp_arr[i], x, y]
            else:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[i, x, y]

    # === Step 4: streaming -- per-cell, parallel over x ===
    # f_new[i, x, y] = f_after_bb[i, x - cx_i, y - cy_i]  (mod Nx/Ny).
    f_new = np.empty_like(f_after_bb)
    for x in prange(Nx):
        for y in range(Ny):
            for i in range(9):
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                xs = (x - cxi_int) % Nx
                ys = (y - cyi_int) % Ny
                f_new[i, x, y] = f_after_bb[i, xs, ys]

    # === Step 5: Zou-He velocity inflow at left boundary (x = 0) ===
    # Mass-conserving inflow that exactly enforces the prescribed velocity.
    # ux_in / uy_in are recovered from the f_inflow vector the caller supplied
    # (built as equilibrium(rho=1, u=(U,0)) -- so ux=U, uy=0 in practice).
    # Gated on inflow_dirs being non-empty so the closed-box test (which passes
    # empty dirs) still skips inflow entirely. The `inflow_dirs` array's
    # contents are not used -- the formulas are hardcoded for the left wall.
    if inflow_dirs.shape[0] > 0:
        rho_in = (f_inflow[0] + f_inflow[1] + f_inflow[2] + f_inflow[3]
                  + f_inflow[4] + f_inflow[5] + f_inflow[6] + f_inflow[7]
                  + f_inflow[8])
        ux_in = (f_inflow[1] - f_inflow[3] + f_inflow[5] - f_inflow[6]
                 - f_inflow[7] + f_inflow[8]) / rho_in
        uy_in = (f_inflow[2] - f_inflow[4] + f_inflow[5] + f_inflow[6]
                 - f_inflow[7] - f_inflow[8]) / rho_in
        for y in range(Ny):
            f0_w = f_new[0, 0, y]
            f2_w = f_new[2, 0, y]
            f3_w = f_new[3, 0, y]
            f4_w = f_new[4, 0, y]
            f6_w = f_new[6, 0, y]
            f7_w = f_new[7, 0, y]
            rho_w = (f0_w + f2_w + f4_w + 2.0 * (f3_w + f6_w + f7_w)) / (1.0 - ux_in)
            f_new[1, 0, y] = f3_w + (2.0 / 3.0) * rho_w * ux_in
            f_new[5, 0, y] = f7_w - 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in + 0.5 * rho_w * uy_in
            f_new[8, 0, y] = f6_w + 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in - 0.5 * rho_w * uy_in

    # === Step 6: Zou-He pressure outflow at right boundary (x = Nx-1) ===
    # Prescribes rho_w = 1.0 (freestream reference) so mass balance closes
    # against the Zou-He velocity inflow above. uy is extrapolated from the
    # interior cell so wake structures pass through instead of reflecting.
    # See zou_he_outflow_pressure docstring at module top for derivation.
    if outflow_dirs.shape[0] > 0:
        for y in range(Ny):
            f0_w = f_new[0, Nx - 1, y]
            f1_w = f_new[1, Nx - 1, y]
            f2_w = f_new[2, Nx - 1, y]
            f4_w = f_new[4, Nx - 1, y]
            f5_w = f_new[5, Nx - 1, y]
            f8_w = f_new[8, Nx - 1, y]
            ux_w = (f0_w + f2_w + f4_w + 2.0 * (f1_w + f5_w + f8_w)) - 1.0
            rho_int = (f_new[0, Nx - 2, y] + f_new[1, Nx - 2, y] + f_new[2, Nx - 2, y]
                       + f_new[3, Nx - 2, y] + f_new[4, Nx - 2, y] + f_new[5, Nx - 2, y]
                       + f_new[6, Nx - 2, y] + f_new[7, Nx - 2, y] + f_new[8, Nx - 2, y])
            uy_w = (f_new[2, Nx - 2, y] - f_new[4, Nx - 2, y]
                    + f_new[5, Nx - 2, y] + f_new[6, Nx - 2, y]
                    - f_new[7, Nx - 2, y] - f_new[8, Nx - 2, y]) / rho_int
            f_new[3, Nx - 1, y] = f1_w - (2.0 / 3.0) * ux_w
            f_new[6, Nx - 1, y] = f8_w - 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w - 0.5 * uy_w
            f_new[7, Nx - 1, y] = f5_w + 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w + 0.5 * uy_w

    # === Step 7: Bouzidi interpolated bounce-back at wall links ===
    # Override the streamed-in population at fluid cells whose direction-i
    # neighbor is solid, using q_field[x, y, i] to interpolate to the exact
    # wall position. q <= 0 means "no Bouzidi": the previously-streamed value
    # (which equals halfway-BB at that link) is kept. See bouzidi_correction
    # docstring at module top for the derivation.
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                continue
            for i in range(1, 9):
                q_val = q_field[x, y, i]
                if q_val <= 0.0:
                    continue
                opp_i = opp_arr[i]
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                f_i = f_after_bb[i, x, y]
                f_opp_at_xf = f_after_bb[opp_i, x, y]
                if q_val >= 0.5:
                    inv_2q = 1.0 / (2.0 * q_val)
                    f_new[opp_i, x, y] = (
                        inv_2q * f_i
                        + (2.0 * q_val - 1.0) * inv_2q * f_opp_at_xf
                    )
                else:
                    xb = x - cxi_int
                    yb = y - cyi_int
                    if 0 <= xb < Nx and 0 <= yb < Ny:
                        f_i_back = f_after_bb[i, xb, yb]
                        f_new[opp_i, x, y] = (
                            2.0 * q_val * f_i
                            + (1.0 - 2.0 * q_val) * f_i_back
                        )
                    else:
                        f_new[opp_i, x, y] = f_i

    return f_new, Fx, Fy


# ---------------------------------------------------------------------------
# MRT collision (multi-relaxation-time) -- structural fix for high-Re BGK
# ---------------------------------------------------------------------------
# BGK collapses to one relaxation rate (1/tau) for every moment. As tau ->
# 0.5 (high Re for a given grid) the higher-order non-hydrodynamic moments
# relax too quickly and the simulation blows up via the well-documented
# tau-near-stability-limit instability.
#
# MRT (Lallemand & Luo 2000; d'Humieres et al. 2002) projects f onto a moment
# basis (rho, energy, energy-squared, momentum, energy-flux, stresses) and
# relaxes EACH moment with its own rate. The conserved moments (rho, jx, jy)
# don't relax. The viscous-stress moments (m7, m8) relax with s_nu = 1/tau,
# giving the SAME kinematic viscosity as BGK. The remaining "free" moments
# (m1, m2, m4, m6) get relaxation rates we choose -- decoupling bulk
# viscosity and ghost-moment damping from kinematic viscosity. That's what
# makes the solver stable at tau -> 0.5 where BGK fails.
#
# Relaxation rate choice: see d'Humieres et al. (2002) Phil. Trans. R. Soc. A
# 360, 437-451. The constraint is 0 < s_i < 2 for stability; values near 1.0
# maximize stability, values near 1/tau minimize accuracy loss vs BGK at low
# Re. We pick the middle ground here -- empirically stable from Re=50 up
# through at least Re=1500 on the 240x80 standard preset.

# MRT free relaxation rates (applied to the non-conserved, non-viscous
# moments). Tuned to 1.4 -- the Lallemand-Luo middle ground validated for
# cylinder flow at Re=100..1500. Lower values (s -> 1.0) over-damp the
# higher-order moments that MRT relies on for stability and actually make
# things WORSE; higher values (s -> 1.9) trade stability for accuracy.
# Sharp-edged bluff bodies still need the Smagorinsky LES adjustment below
# to stay stable above Re~500.
#
# These constants are the single source of truth. Both the pure-NumPy
# reference (``collide_mrt``) and the two JIT-compiled hot paths
# (``step_njit_mrt_with_force``, ``step_njit_mrt_no_force``) reference them
# directly. Numba constant-folds them at compile time. If you change a
# value here, delete ``__pycache__/`` to force a JIT recompile so the new
# value takes effect.
S_E = 1.4    # energy moment (m1)            -- bulk viscosity damping
S_EPS = 1.4  # energy-squared moment (m2)    -- higher-order damping
S_Q = 1.4    # energy-flux moments (m4, m6)  -- ghost-moment damping

# Smagorinsky-LES sub-grid eddy-viscosity constant. The LBM-Smagorinsky
# adjustment scales the effective relaxation time per-cell based on local
# strain rate magnitude (computed from the non-equilibrium part of the
# stress moments m7 and m8). 0.17 is Lilly's theoretical value (Lilly 1967);
# common engineering range is 0.10-0.20 (Smagorinsky 1963 ~0.16,
# Deardorff 0.10-0.14).
#
# Stability survey (5000 steps, then-Standard 320x100, every shape+AoA,
# Re=200,500,1000,1500) on 2026-05-18:
#   * Re<=500: all 9 (shape,AoA) combos stable at 0.14, 0.17, and 0.20.
#   * Re>=1000 + sharp-corner geometry (Square 45 deg, Ellipse 45 deg): NaN
#     at all three values WHEN USING halfway bounce-back -- the BC-driven
#     instability traced to knife-edge voxelized corners injecting spurious
#     pressure pulses no amount of subgrid damping could absorb. This was
#     the motivation for adding Bouzidi-Firdaouss-Lallemand interpolated
#     bounce-back, which is now the production path (square_q_field /
#     ellipse_q_field in src/shapes.py); the sharp-corner cases are stable
#     since that change.
#   * 0.14 ADDS three new unstable cases at moderate Re (Cylinder@Re=1000,
#     Square 45@Re=1000, Ellipse 30@Re=1500) -- too thin a margin.
#   * 0.20 was previously claimed to be "stable to Re=1500" but actually
#     silently produces NaN frames on Ellipse 45 deg @ Re>=1000 (matplotlib
#     renders NaN as background so the user never saw the divergence).
#
# 0.17 dominates both: same set of unstable cases as 0.20, but 28% less
# eddy viscosity in the bulk (= sharper vortex cores, crisper shear layers
# at low Re where the user lives by default).
C_SMAG = 0.17
C_SMAG_SQ = C_SMAG * C_SMAG    # pre-computed so the JIT path doesn't square it per cell

# Moment-basis transformation matrix M (rows = moments, cols = velocity indices
# matching the convention at the top of this file: [rest, E, N, W, S, NE, NW, SW, SE]).
# Reference: Lallemand & Luo (2000). The 9 rows are:
#   m0 = rho       m1 = e          m2 = epsilon
#   m3 = jx        m4 = qx         m5 = jy           m6 = qy
#   m7 = pxx       m8 = pxy
_M_MRT = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1],
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1],
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1],
], dtype=np.float64)


def collide_mrt(f, tau):
    """MRT collision (pure-NumPy reference). Relaxes f via 9 moment rates.

    Same kinematic viscosity as ``collide(f, tau)`` (BGK) -- m7 and m8 use
    s = 1/tau -- but additional free rates on the non-hydrodynamic moments
    decouple bulk viscosity and stabilize the solver near tau = 0.5 where
    BGK diverges.

    Parameters
    ----------
    f : ndarray of shape (9, Nx, Ny)
    tau : float, > 0.5

    Returns
    -------
    f_post : ndarray of shape (9, Nx, Ny)
    """
    s_nu = 1.0 / tau
    # Per-moment relaxation rates. Order: [m0, m1, m2, m3, m4, m5, m6, m7, m8].
    # Conserved moments (rho=m0, jx=m3, jy=m5) get s=0 -- they don't change.
    S = np.array([0.0, S_E, S_EPS, 0.0, S_Q, 0.0, S_Q, s_nu, s_nu])

    # f -> moments via tensor contraction over the velocity axis.
    # einsum 'ij,jxy->ixy' is the natural way to write M @ f for each (x, y).
    m = np.einsum("ij,jxy->ixy", _M_MRT, f)

    rho = m[0]
    jx = m[3]
    jy = m[5]
    ux = jx / rho
    uy = jy / rho
    u2 = ux * ux + uy * uy

    # Equilibrium moments (Lallemand & Luo 2000, eq. 2.13).
    m_eq = np.empty_like(m)
    m_eq[0] = rho
    m_eq[1] = rho * (-2.0 + 3.0 * u2)
    m_eq[2] = rho * (1.0 - 3.0 * u2)
    m_eq[3] = jx
    m_eq[4] = -jx
    m_eq[5] = jy
    m_eq[6] = -jy
    m_eq[7] = rho * (ux * ux - uy * uy)
    m_eq[8] = rho * ux * uy

    # Smagorinsky LES adjusts s_nu (m7, m8 rates) per cell based on local
    # non-equilibrium stress magnitude -- adds eddy viscosity in high-strain
    # regions (corners, shear layers) for stability at high Re.
    dPxx = m[7] - m_eq[7]
    dPxy = m[8] - m_eq[8]
    Q = np.sqrt(dPxx ** 2 + 4.0 * dPxy ** 2)
    c_smag_sq = C_SMAG ** 2
    tau_eff = 0.5 * (tau + np.sqrt(tau ** 2 + 18.0 * c_smag_sq * Q / rho))
    s_nu_field = 1.0 / tau_eff

    # Relax each moment. m7 / m8 get per-cell rates; everything else uses S.
    m_post = m.copy()
    for k in range(9):
        if k in (7, 8):
            continue
        if S[k] == 0.0:
            continue
        m_post[k] = m[k] - S[k] * (m[k] - m_eq[k])
    m_post[7] = m[7] - s_nu_field * dPxx
    m_post[8] = m[8] - s_nu_field * dPxy
    M_inv = np.linalg.inv(_M_MRT)
    return np.einsum("ij,jxy->ixy", M_inv, m_post)


@njit(fastmath=True)
def step_njit_mrt_with_force(f, tau, solid_mask, q_field, f_inflow, inflow_dirs, outflow_dirs):
    """One fused MRT-LBM timestep + momentum-exchange force calculation.

    Drop-in replacement for ``step_njit_with_force`` (BGK) with identical
    signature and identical kinematic viscosity. Uses MRT collision so the
    solver stays stable as tau approaches 0.5 (high Re for a given grid).

    Inlines the M and M^-1 transforms as explicit per-cell algebra. The
    moments and equilibrium moments are computed cell-by-cell from the 9
    populations, relaxed individually, and reassembled into post-collision
    populations -- no matrix multiplies inside the hot loop.

    Compile time: ~6-10 sec on first call, cached thereafter. Warm-step
    throughput on 240x80 serial: ~0.4 ms/step (~2x slower than BGK, acceptable
    for the much larger stable Re envelope).
    """
    Nx = f.shape[1]
    Ny = f.shape[2]

    cx_arr = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    cy_arr = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    opp_arr = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

    # MRT relaxation rates + Smagorinsky-squared from the module-level
    # constants. Numba constant-folds these at compile time; the JIT cache
    # picks up changes via the source-hash, but if you've already compiled
    # once with the old values, delete __pycache__/ to force a rebuild.
    s_e = S_E
    s_eps = S_EPS
    s_q = S_Q
    c_smag_sq = C_SMAG_SQ

    # === Step 1: MRT collision -- per-cell, parallel over x ===
    f_post_coll = np.empty_like(f)
    for x in prange(Nx):
        for y in range(Ny):
            f0 = f[0, x, y]
            f1 = f[1, x, y]
            f2 = f[2, x, y]
            f3 = f[3, x, y]
            f4 = f[4, x, y]
            f5 = f[5, x, y]
            f6 = f[6, x, y]
            f7 = f[7, x, y]
            f8 = f[8, x, y]

            # Moments m = M @ f, inlined (no matrix multiply).
            m0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            m1 = -4.0 * f0 - f1 - f2 - f3 - f4 + 2.0 * (f5 + f6 + f7 + f8)
            m2 = 4.0 * f0 - 2.0 * (f1 + f2 + f3 + f4) + f5 + f6 + f7 + f8
            m3 = f1 - f3 + f5 - f6 - f7 + f8
            m4 = -2.0 * f1 + 2.0 * f3 + f5 - f6 - f7 + f8
            m5 = f2 - f4 + f5 + f6 - f7 - f8
            m6 = -2.0 * f2 + 2.0 * f4 + f5 + f6 - f7 - f8
            m7 = f1 - f2 + f3 - f4
            m8 = f5 - f6 + f7 - f8

            rho = m0
            jx = m3
            jy = m5
            ux = jx / rho
            uy = jy / rho
            u2 = ux * ux + uy * uy

            # Equilibrium moments.
            m1_eq = rho * (-2.0 + 3.0 * u2)
            m2_eq = rho * (1.0 - 3.0 * u2)
            m4_eq = -jx
            m6_eq = -jy
            m7_eq = rho * (ux * ux - uy * uy)
            m8_eq = rho * ux * uy

            # Smagorinsky LES: compute local effective tau from the
            # non-equilibrium stress moments. Q^2 = 2 * Pi_neq:Pi_neq with
            # 2D-incompressible deviatoric assumption (Pi_xx = -Pi_yy).
            dPxx = m7 - m7_eq
            dPxy = m8 - m8_eq
            Q = np.sqrt(dPxx * dPxx + 4.0 * dPxy * dPxy)
            tau_eff = 0.5 * (tau + np.sqrt(tau * tau + 18.0 * c_smag_sq * Q / rho))
            s_nu_eff = 1.0 / tau_eff

            # Relax non-conserved moments individually.
            m1_p = m1 - s_e * (m1 - m1_eq)
            m2_p = m2 - s_eps * (m2 - m2_eq)
            m4_p = m4 - s_q * (m4 - m4_eq)
            m6_p = m6 - s_q * (m6 - m6_eq)
            m7_p = m7 - s_nu_eff * dPxx
            m8_p = m8 - s_nu_eff * dPxy
            # Conserved moments unchanged: rho, jx, jy.

            # Inverse transform f = M^-1 @ m (inlined; coefficients precomputed
            # from M^-1 = M^T / norms with norms = [9,36,36,6,12,6,12,4,4]).
            inv9 = 1.0 / 9.0
            inv36 = 1.0 / 36.0
            inv18 = 1.0 / 18.0
            inv6 = 1.0 / 6.0
            inv12 = 1.0 / 12.0
            inv4 = 1.0 / 4.0
            r9 = rho * inv9
            f_post_coll[0, x, y] = r9 - m1_p * inv9 + m2_p * inv9
            f_post_coll[1, x, y] = r9 - m1_p * inv36 - m2_p * inv18 + jx * inv6 - m4_p * inv6 + m7_p * inv4
            f_post_coll[2, x, y] = r9 - m1_p * inv36 - m2_p * inv18 + jy * inv6 - m6_p * inv6 - m7_p * inv4
            f_post_coll[3, x, y] = r9 - m1_p * inv36 - m2_p * inv18 - jx * inv6 + m4_p * inv6 + m7_p * inv4
            f_post_coll[4, x, y] = r9 - m1_p * inv36 - m2_p * inv18 - jy * inv6 + m6_p * inv6 - m7_p * inv4
            f_post_coll[5, x, y] = r9 + m1_p * inv18 + m2_p * inv36 + jx * inv6 + m4_p * inv12 + jy * inv6 + m6_p * inv12 + m8_p * inv4
            f_post_coll[6, x, y] = r9 + m1_p * inv18 + m2_p * inv36 - jx * inv6 - m4_p * inv12 + jy * inv6 + m6_p * inv12 - m8_p * inv4
            f_post_coll[7, x, y] = r9 + m1_p * inv18 + m2_p * inv36 - jx * inv6 - m4_p * inv12 - jy * inv6 - m6_p * inv12 + m8_p * inv4
            f_post_coll[8, x, y] = r9 + m1_p * inv18 + m2_p * inv36 + jx * inv6 + m4_p * inv12 - jy * inv6 - m6_p * inv12 - m8_p * inv4

    # === Step 2: momentum-exchange force (Bouzidi-aware, identical to BGK path) ===
    # See the BGK with-force step function for the rationale; same formulas.
    Fx = 0.0
    Fy = 0.0
    for i in range(1, 9):
        cxi = cx_arr[i]
        cyi = cy_arr[i]
        cxi_int = int(cxi)
        cyi_int = int(cyi)
        opp_i = opp_arr[i]
        for x in range(Nx):
            for y in range(Ny):
                if not solid_mask[x, y]:
                    xn = (x + cxi_int) % Nx
                    yn = (y + cyi_int) % Ny
                    if solid_mask[xn, yn]:
                        f_i_post = f_post_coll[i, x, y]
                        q_val = q_field[x, y, i]
                        if q_val <= 0.0:
                            contrib = 2.0 * f_i_post
                        elif q_val >= 0.5:
                            inv_2q = 1.0 / (2.0 * q_val)
                            f_minus_i_new = (
                                inv_2q * f_i_post
                                + (2.0 * q_val - 1.0) * inv_2q * f_post_coll[opp_i, x, y]
                            )
                            contrib = f_i_post + f_minus_i_new
                        else:
                            xb = x - cxi_int
                            yb = y - cyi_int
                            if 0 <= xb < Nx and 0 <= yb < Ny:
                                f_minus_i_new = (
                                    2.0 * q_val * f_i_post
                                    + (1.0 - 2.0 * q_val) * f_post_coll[i, xb, yb]
                                )
                                contrib = f_i_post + f_minus_i_new
                            else:
                                contrib = 2.0 * f_i_post
                        Fx += cxi * contrib
                        Fy += cyi * contrib

    # === Step 3: bounce-back -- per-cell, parallel over x ===
    f_after_bb = np.empty_like(f_post_coll)
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[opp_arr[i], x, y]
            else:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[i, x, y]

    # === Step 4: streaming -- per-cell, parallel over x ===
    f_new = np.empty_like(f_after_bb)
    for x in prange(Nx):
        for y in range(Ny):
            for i in range(9):
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                xs = (x - cxi_int) % Nx
                ys = (y - cyi_int) % Ny
                f_new[i, x, y] = f_after_bb[i, xs, ys]

    # === Step 4.5: bounce-back top/bottom -- kill periodic y-wraparound ===
    # Streaming uses modulo Ny so the box is periodic vertically -- air
    # exiting the bottom physically re-enters at the top, which looks
    # unphysical for "uniform flow past a body". We overwrite the wraparound
    # junk with halfway bounce-back at the walls: the population that "tried"
    # to leave through the wall reflects back at the same cell, reversed.
    # Tried free-slip first; at tau approaching 0.5 it produced wall
    # instabilities that diverged the simulation. Full bounce-back is more
    # dissipative (adds a thin wall boundary layer) but rock-stable from
    # Re=50 through Re=1500.
    for x in prange(Nx):
        # Top wall (y = Ny-1): N->S, NE->SW, NW->SE -- reverse the velocity.
        f_new[4, x, Ny - 1] = f_after_bb[2, x, Ny - 1]
        f_new[7, x, Ny - 1] = f_after_bb[5, x, Ny - 1]
        f_new[8, x, Ny - 1] = f_after_bb[6, x, Ny - 1]
        # Bottom wall (y = 0): S->N, SW->NE, SE->NW.
        f_new[2, x, 0] = f_after_bb[4, x, 0]
        f_new[5, x, 0] = f_after_bb[7, x, 0]
        f_new[6, x, 0] = f_after_bb[8, x, 0]

    # === Step 5: Zou-He velocity inflow at left boundary (x = 0) ===
    # See step_njit_with_force for the derivation; identical formulas reused.
    if inflow_dirs.shape[0] > 0:
        rho_in = (f_inflow[0] + f_inflow[1] + f_inflow[2] + f_inflow[3]
                  + f_inflow[4] + f_inflow[5] + f_inflow[6] + f_inflow[7]
                  + f_inflow[8])
        ux_in = (f_inflow[1] - f_inflow[3] + f_inflow[5] - f_inflow[6]
                 - f_inflow[7] + f_inflow[8]) / rho_in
        uy_in = (f_inflow[2] - f_inflow[4] + f_inflow[5] + f_inflow[6]
                 - f_inflow[7] - f_inflow[8]) / rho_in
        for y in range(Ny):
            f0_w = f_new[0, 0, y]
            f2_w = f_new[2, 0, y]
            f3_w = f_new[3, 0, y]
            f4_w = f_new[4, 0, y]
            f6_w = f_new[6, 0, y]
            f7_w = f_new[7, 0, y]
            rho_w = (f0_w + f2_w + f4_w + 2.0 * (f3_w + f6_w + f7_w)) / (1.0 - ux_in)
            f_new[1, 0, y] = f3_w + (2.0 / 3.0) * rho_w * ux_in
            f_new[5, 0, y] = f7_w - 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in + 0.5 * rho_w * uy_in
            f_new[8, 0, y] = f6_w + 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in - 0.5 * rho_w * uy_in

    # === Step 6: Zou-He pressure outflow at right boundary (x = Nx-1) ===
    # Prescribes rho_w = 1.0 (freestream reference) so mass balance closes
    # against the Zou-He velocity inflow above. uy is extrapolated from the
    # interior cell so wake structures pass through instead of reflecting.
    # See zou_he_outflow_pressure docstring at module top for derivation.
    if outflow_dirs.shape[0] > 0:
        for y in range(Ny):
            f0_w = f_new[0, Nx - 1, y]
            f1_w = f_new[1, Nx - 1, y]
            f2_w = f_new[2, Nx - 1, y]
            f4_w = f_new[4, Nx - 1, y]
            f5_w = f_new[5, Nx - 1, y]
            f8_w = f_new[8, Nx - 1, y]
            ux_w = (f0_w + f2_w + f4_w + 2.0 * (f1_w + f5_w + f8_w)) - 1.0
            rho_int = (f_new[0, Nx - 2, y] + f_new[1, Nx - 2, y] + f_new[2, Nx - 2, y]
                       + f_new[3, Nx - 2, y] + f_new[4, Nx - 2, y] + f_new[5, Nx - 2, y]
                       + f_new[6, Nx - 2, y] + f_new[7, Nx - 2, y] + f_new[8, Nx - 2, y])
            uy_w = (f_new[2, Nx - 2, y] - f_new[4, Nx - 2, y]
                    + f_new[5, Nx - 2, y] + f_new[6, Nx - 2, y]
                    - f_new[7, Nx - 2, y] - f_new[8, Nx - 2, y]) / rho_int
            f_new[3, Nx - 1, y] = f1_w - (2.0 / 3.0) * ux_w
            f_new[6, Nx - 1, y] = f8_w - 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w - 0.5 * uy_w
            f_new[7, Nx - 1, y] = f5_w + 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w + 0.5 * uy_w

    # === Step 7: Bouzidi interpolated bounce-back at wall links ===
    # Override the streamed-in population at fluid cells whose direction-i
    # neighbor is solid, using q_field[x, y, i] to interpolate to the exact
    # wall position. q <= 0 means "no Bouzidi": the previously-streamed value
    # (which equals halfway-BB at that link) is kept. See bouzidi_correction
    # docstring at module top for the derivation.
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                continue
            for i in range(1, 9):
                q_val = q_field[x, y, i]
                if q_val <= 0.0:
                    continue
                opp_i = opp_arr[i]
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                f_i = f_after_bb[i, x, y]
                f_opp_at_xf = f_after_bb[opp_i, x, y]
                if q_val >= 0.5:
                    inv_2q = 1.0 / (2.0 * q_val)
                    f_new[opp_i, x, y] = (
                        inv_2q * f_i
                        + (2.0 * q_val - 1.0) * inv_2q * f_opp_at_xf
                    )
                else:
                    xb = x - cxi_int
                    yb = y - cyi_int
                    if 0 <= xb < Nx and 0 <= yb < Ny:
                        f_i_back = f_after_bb[i, xb, yb]
                        f_new[opp_i, x, y] = (
                            2.0 * q_val * f_i
                            + (1.0 - 2.0 * q_val) * f_i_back
                        )
                    else:
                        f_new[opp_i, x, y] = f_i

    return f_new, Fx, Fy


@njit(fastmath=True)
def step_njit_mrt_no_force(f, tau, solid_mask, q_field, f_inflow, inflow_dirs, outflow_dirs):
    """MRT timestep without the momentum-exchange force calculation.

    Drop-in for the Streamlit visualization path (it doesn't consume Fx/Fy).
    Identical physics to ``step_njit_mrt_with_force`` minus the
    8-direction force-accumulator loop. On a 240x80 grid this saves
    ~5-8% per step -- small per step, useful across 2000+ record steps.
    Validation scripts that need Fx/Fy keep using the with-force variant;
    this is a pure performance refinement for the viz path.
    """
    Nx = f.shape[1]
    Ny = f.shape[2]

    cx_arr = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    cy_arr = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    opp_arr = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

    # Pull MRT rates + C_SMAG^2 from the module-level constants -- single
    # source of truth shared with the with-force variant and the pure-NumPy
    # reference. See lbm.py top of MRT section for the values.
    s_e = S_E
    s_eps = S_EPS
    s_q = S_Q
    c_smag_sq = C_SMAG_SQ

    # MRT collision -- inlined moment transform, identical to with-force path.
    f_post_coll = np.empty_like(f)
    for x in prange(Nx):
        for y in range(Ny):
            f0 = f[0, x, y]
            f1 = f[1, x, y]
            f2 = f[2, x, y]
            f3 = f[3, x, y]
            f4 = f[4, x, y]
            f5 = f[5, x, y]
            f6 = f[6, x, y]
            f7 = f[7, x, y]
            f8 = f[8, x, y]

            m0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            m1 = -4.0 * f0 - f1 - f2 - f3 - f4 + 2.0 * (f5 + f6 + f7 + f8)
            m2 = 4.0 * f0 - 2.0 * (f1 + f2 + f3 + f4) + f5 + f6 + f7 + f8
            m3 = f1 - f3 + f5 - f6 - f7 + f8
            m4 = -2.0 * f1 + 2.0 * f3 + f5 - f6 - f7 + f8
            m5 = f2 - f4 + f5 + f6 - f7 - f8
            m6 = -2.0 * f2 + 2.0 * f4 + f5 + f6 - f7 - f8
            m7 = f1 - f2 + f3 - f4
            m8 = f5 - f6 + f7 - f8

            rho = m0
            jx = m3
            jy = m5
            ux = jx / rho
            uy = jy / rho
            u2 = ux * ux + uy * uy

            m1_eq = rho * (-2.0 + 3.0 * u2)
            m2_eq = rho * (1.0 - 3.0 * u2)
            m4_eq = -jx
            m6_eq = -jy
            m7_eq = rho * (ux * ux - uy * uy)
            m8_eq = rho * ux * uy

            dPxx = m7 - m7_eq
            dPxy = m8 - m8_eq
            Q = np.sqrt(dPxx * dPxx + 4.0 * dPxy * dPxy)
            tau_eff = 0.5 * (tau + np.sqrt(tau * tau + 18.0 * c_smag_sq * Q / rho))
            s_nu_eff = 1.0 / tau_eff

            m1_p = m1 - s_e * (m1 - m1_eq)
            m2_p = m2 - s_eps * (m2 - m2_eq)
            m4_p = m4 - s_q * (m4 - m4_eq)
            m6_p = m6 - s_q * (m6 - m6_eq)
            m7_p = m7 - s_nu_eff * dPxx
            m8_p = m8 - s_nu_eff * dPxy

            inv9 = 1.0 / 9.0
            inv36 = 1.0 / 36.0
            inv18 = 1.0 / 18.0
            inv6 = 1.0 / 6.0
            inv12 = 1.0 / 12.0
            inv4 = 1.0 / 4.0
            r9 = rho * inv9
            f_post_coll[0, x, y] = r9 - m1_p * inv9 + m2_p * inv9
            f_post_coll[1, x, y] = r9 - m1_p * inv36 - m2_p * inv18 + jx * inv6 - m4_p * inv6 + m7_p * inv4
            f_post_coll[2, x, y] = r9 - m1_p * inv36 - m2_p * inv18 + jy * inv6 - m6_p * inv6 - m7_p * inv4
            f_post_coll[3, x, y] = r9 - m1_p * inv36 - m2_p * inv18 - jx * inv6 + m4_p * inv6 + m7_p * inv4
            f_post_coll[4, x, y] = r9 - m1_p * inv36 - m2_p * inv18 - jy * inv6 + m6_p * inv6 - m7_p * inv4
            f_post_coll[5, x, y] = r9 + m1_p * inv18 + m2_p * inv36 + jx * inv6 + m4_p * inv12 + jy * inv6 + m6_p * inv12 + m8_p * inv4
            f_post_coll[6, x, y] = r9 + m1_p * inv18 + m2_p * inv36 - jx * inv6 - m4_p * inv12 + jy * inv6 + m6_p * inv12 - m8_p * inv4
            f_post_coll[7, x, y] = r9 + m1_p * inv18 + m2_p * inv36 - jx * inv6 - m4_p * inv12 - jy * inv6 - m6_p * inv12 + m8_p * inv4
            f_post_coll[8, x, y] = r9 + m1_p * inv18 + m2_p * inv36 + jx * inv6 + m4_p * inv12 - jy * inv6 - m6_p * inv12 - m8_p * inv4

    # Bounce-back on solid cells.
    f_after_bb = np.empty_like(f_post_coll)
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[opp_arr[i], x, y]
            else:
                for i in range(9):
                    f_after_bb[i, x, y] = f_post_coll[i, x, y]

    # Streaming.
    f_new = np.empty_like(f_after_bb)
    for x in prange(Nx):
        for y in range(Ny):
            for i in range(9):
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                xs = (x - cxi_int) % Nx
                ys = (y - cyi_int) % Ny
                f_new[i, x, y] = f_after_bb[i, xs, ys]

    # Top/bottom wall bounce-back (kills periodic y-wraparound).
    for x in prange(Nx):
        f_new[4, x, Ny - 1] = f_after_bb[2, x, Ny - 1]
        f_new[7, x, Ny - 1] = f_after_bb[5, x, Ny - 1]
        f_new[8, x, Ny - 1] = f_after_bb[6, x, Ny - 1]
        f_new[2, x, 0] = f_after_bb[4, x, 0]
        f_new[5, x, 0] = f_after_bb[7, x, 0]
        f_new[6, x, 0] = f_after_bb[8, x, 0]

    # Zou-He inflow (left, x=0) + zero-gradient outflow (right, x=Nx-1).
    # See step_njit_with_force for the Zou-He derivation; identical formulas
    # reused. Gated on inflow_dirs being non-empty so the closed-box test
    # (which passes empty dirs) still skips inflow entirely.
    if inflow_dirs.shape[0] > 0:
        rho_in = (f_inflow[0] + f_inflow[1] + f_inflow[2] + f_inflow[3]
                  + f_inflow[4] + f_inflow[5] + f_inflow[6] + f_inflow[7]
                  + f_inflow[8])
        ux_in = (f_inflow[1] - f_inflow[3] + f_inflow[5] - f_inflow[6]
                 - f_inflow[7] + f_inflow[8]) / rho_in
        uy_in = (f_inflow[2] - f_inflow[4] + f_inflow[5] + f_inflow[6]
                 - f_inflow[7] - f_inflow[8]) / rho_in
        for y in range(Ny):
            f0_w = f_new[0, 0, y]
            f2_w = f_new[2, 0, y]
            f3_w = f_new[3, 0, y]
            f4_w = f_new[4, 0, y]
            f6_w = f_new[6, 0, y]
            f7_w = f_new[7, 0, y]
            rho_w = (f0_w + f2_w + f4_w + 2.0 * (f3_w + f6_w + f7_w)) / (1.0 - ux_in)
            f_new[1, 0, y] = f3_w + (2.0 / 3.0) * rho_w * ux_in
            f_new[5, 0, y] = f7_w - 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in + 0.5 * rho_w * uy_in
            f_new[8, 0, y] = f6_w + 0.5 * (f2_w - f4_w) + (1.0 / 6.0) * rho_w * ux_in - 0.5 * rho_w * uy_in
    # Zou-He pressure outflow (rho_w = 1.0) -- see step_njit_with_force for derivation.
    if outflow_dirs.shape[0] > 0:
        for y in range(Ny):
            f0_w = f_new[0, Nx - 1, y]
            f1_w = f_new[1, Nx - 1, y]
            f2_w = f_new[2, Nx - 1, y]
            f4_w = f_new[4, Nx - 1, y]
            f5_w = f_new[5, Nx - 1, y]
            f8_w = f_new[8, Nx - 1, y]
            ux_w = (f0_w + f2_w + f4_w + 2.0 * (f1_w + f5_w + f8_w)) - 1.0
            rho_int = (f_new[0, Nx - 2, y] + f_new[1, Nx - 2, y] + f_new[2, Nx - 2, y]
                       + f_new[3, Nx - 2, y] + f_new[4, Nx - 2, y] + f_new[5, Nx - 2, y]
                       + f_new[6, Nx - 2, y] + f_new[7, Nx - 2, y] + f_new[8, Nx - 2, y])
            uy_w = (f_new[2, Nx - 2, y] - f_new[4, Nx - 2, y]
                    + f_new[5, Nx - 2, y] + f_new[6, Nx - 2, y]
                    - f_new[7, Nx - 2, y] - f_new[8, Nx - 2, y]) / rho_int
            f_new[3, Nx - 1, y] = f1_w - (2.0 / 3.0) * ux_w
            f_new[6, Nx - 1, y] = f8_w - 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w - 0.5 * uy_w
            f_new[7, Nx - 1, y] = f5_w + 0.5 * (f2_w - f4_w) - (1.0 / 6.0) * ux_w + 0.5 * uy_w

    # Bouzidi interpolated bounce-back at wall links -- see derivation at
    # module-level bouzidi_correction docstring. Mirrors the block in
    # step_njit_with_force / step_njit_mrt_with_force.
    for x in prange(Nx):
        for y in range(Ny):
            if solid_mask[x, y]:
                continue
            for i in range(1, 9):
                q_val = q_field[x, y, i]
                if q_val <= 0.0:
                    continue
                opp_i = opp_arr[i]
                cxi_int = int(cx_arr[i])
                cyi_int = int(cy_arr[i])
                f_i = f_after_bb[i, x, y]
                f_opp_at_xf = f_after_bb[opp_i, x, y]
                if q_val >= 0.5:
                    inv_2q = 1.0 / (2.0 * q_val)
                    f_new[opp_i, x, y] = (
                        inv_2q * f_i
                        + (2.0 * q_val - 1.0) * inv_2q * f_opp_at_xf
                    )
                else:
                    xb = x - cxi_int
                    yb = y - cyi_int
                    if 0 <= xb < Nx and 0 <= yb < Ny:
                        f_i_back = f_after_bb[i, xb, yb]
                        f_new[opp_i, x, y] = (
                            2.0 * q_val * f_i
                            + (1.0 - 2.0 * q_val) * f_i_back
                        )
                    else:
                        f_new[opp_i, x, y] = f_i

    return f_new
