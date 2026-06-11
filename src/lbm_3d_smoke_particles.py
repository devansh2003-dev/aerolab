"""3D smoke-particle advection (Phase A1, locked by D-8).

The consumer-product v1 viz: massless tracers advected through a
steady (or quasi-steady) velocity field via RK4 with trilinear
interpolation. The expensive LBM solve produces (ux, uy, uz); this
module turns it into the wind-tunnel-smoke look AeroLab's audience
expects.

Design rules (carried over from the 2D streakline lesson — see
the memory note feedback_streamline_design, which records THREE
prior regressions of this rule across May 2026):

  - Seed only from a stable upstream column. NEVER inject fresh
    tracers downstream of the body. The user reads mid-domain
    spawn as "smoke appearing from nowhere behind the shape" and
    has pushed back on it every time.
  - Forward-only RK4 integration. No bidirectional streamline
    draw.
  - Cull on (a) leaving the domain, (b) entering the body mask,
    (c) age > max_age. If the back half of the channel looks
    empty, raise max_age, NOT seed density mid-domain.

Algorithm (D-8 in 3D_PHASE0_DECISIONS.md):

  for each frame:
    spawn N new particles at the inflow column
    for each particle: RK4 step through (ux, uy, uz) using trilerp
    age += 1
    cull (outside / inside body / age expired)
    return live particles for rendering

Why pure NumPy and not Numba: per-frame cost is ~5-15 ms for ~500
particles at ~4 RK4 substeps. The 2D streakline path
(`src/lbm_render.py`) is also pure NumPy for the same reason —
keeps the code uniform and trivial to test against analytic fields
(D-8 verification, reviewer 2026-05-26).
"""
from __future__ import annotations

import numpy as np

# --- Trilinear interpolation ------------------------------------------------


def trilerp_3d(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """Vectorised trilinear interpolation of a 3D scalar field at FP positions.

    Parameters
    ----------
    field : (Nx, Ny, Nz) ndarray
        Scalar field on the LBM lattice. Indices are lattice cells.
    xs, ys, zs : 1D arrays of float
        Particle positions in lattice coordinates. Values outside
        ``[0, N-1]`` are clipped to the boundary cell pair to avoid
        out-of-bounds reads; the caller is responsible for culling
        truly out-of-domain particles BEFORE calling this.

    Returns
    -------
    1D array, same length as xs/ys/zs, of the interpolated value.

    Notes
    -----
    The 3D analogue of `_bilerp` in `src/lbm_render.py`. Used
    inside the RK4 advector to read (ux, uy, uz) at sub-cell
    particle positions.
    """
    Nx, Ny, Nz = field.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, Nx - 2)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, Ny - 2)
    z0 = np.clip(np.floor(zs).astype(np.int32), 0, Nz - 2)
    fx = xs - x0
    fy = ys - y0
    fz = zs - z0

    c000 = field[x0,     y0,     z0]
    c100 = field[x0 + 1, y0,     z0]
    c010 = field[x0,     y0 + 1, z0]
    c110 = field[x0 + 1, y0 + 1, z0]
    c001 = field[x0,     y0,     z0 + 1]
    c101 = field[x0 + 1, y0,     z0 + 1]
    c011 = field[x0,     y0 + 1, z0 + 1]
    c111 = field[x0 + 1, y0 + 1, z0 + 1]

    return (c000 * (1 - fx) * (1 - fy) * (1 - fz)
            + c100 *      fx  * (1 - fy) * (1 - fz)
            + c010 * (1 - fx) *      fy  * (1 - fz)
            + c110 *      fx  *      fy  * (1 - fz)
            + c001 * (1 - fx) * (1 - fy) *      fz
            + c101 *      fx  * (1 - fy) *      fz
            + c011 * (1 - fx) *      fy  *      fz
            + c111 *      fx  *      fy  *      fz)


# --- RK4 advection ----------------------------------------------------------


def advect_rk4(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One RK4 advection step through a steady velocity field.

    p(t + dt) = p(t) + dt * (k1 + 2 k2 + 2 k3 + k4) / 6,
    where each k_i is u evaluated (trilinearly) at a Runge-Kutta
    intermediate position.

    Returns new (xs, ys, zs) arrays; inputs are not modified.
    """
    k1x = trilerp_3d(ux, xs, ys, zs)
    k1y = trilerp_3d(uy, xs, ys, zs)
    k1z = trilerp_3d(uz, xs, ys, zs)

    xm = xs + 0.5 * dt * k1x
    ym = ys + 0.5 * dt * k1y
    zm = zs + 0.5 * dt * k1z
    k2x = trilerp_3d(ux, xm, ym, zm)
    k2y = trilerp_3d(uy, xm, ym, zm)
    k2z = trilerp_3d(uz, xm, ym, zm)

    xm = xs + 0.5 * dt * k2x
    ym = ys + 0.5 * dt * k2y
    zm = zs + 0.5 * dt * k2z
    k3x = trilerp_3d(ux, xm, ym, zm)
    k3y = trilerp_3d(uy, xm, ym, zm)
    k3z = trilerp_3d(uz, xm, ym, zm)

    xm = xs + dt * k3x
    ym = ys + dt * k3y
    zm = zs + dt * k3z
    k4x = trilerp_3d(ux, xm, ym, zm)
    k4y = trilerp_3d(uy, xm, ym, zm)
    k4z = trilerp_3d(uz, xm, ym, zm)

    new_x = xs + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    new_y = ys + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    new_z = zs + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return new_x, new_y, new_z


# --- Inflow seeding ---------------------------------------------------------


def seed_inflow_particles(
    n_per_row: int,
    y_rows: np.ndarray,
    z_rows: np.ndarray,
    x: float = 2.0,
    jitter: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spawn new particles at the inflow column on a (y, z) grid.

    The seeded particles all share ``x``; their y, z positions sit
    on the cartesian product of ``y_rows`` x ``z_rows``, repeated
    ``n_per_row`` times, with small uniform jitter in y/z to break
    the regularity (so the streamline visualisation does not look
    like a grid pattern).

    NEVER seed off the inflow column. See feedback_streamline_design
    in memory — three prior regressions of this rule.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    yy, zz = np.meshgrid(y_rows, z_rows, indexing="ij")
    n_per_frame = yy.size * n_per_row
    new_x = np.full(n_per_frame, x, dtype=np.float64)
    new_y = np.tile(yy.ravel(), n_per_row).astype(np.float64)
    new_z = np.tile(zz.ravel(), n_per_row).astype(np.float64)
    new_y += rng.uniform(-jitter, jitter, size=n_per_frame)
    new_z += rng.uniform(-jitter, jitter, size=n_per_frame)
    return new_x, new_y, new_z


# --- One frame: spawn + advect + cull ---------------------------------------


def step_smoke(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, age: np.ndarray,
    ux: np.ndarray, uy: np.ndarray, uz: np.ndarray,
    body_mask: np.ndarray | None,
    dt: float,
    n_substeps: int,
    max_age: int,
    inflow_seed_xyz: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame of smoke: optionally seed at inflow, advect, cull.

    Parameters
    ----------
    px, py, pz, age
        Current particle state. ``age`` is in frames (incremented
        once per call); use ``max_age * dt`` to think about
        lifetime in lattice time.
    ux, uy, uz : (Nx, Ny, Nz) ndarrays
        Velocity field, treated as time-frozen for the frame.
    body_mask : (Nx, Ny, Nz) bool or None
        Solid cells. A particle whose nearest cell is solid is
        culled. ``None`` means no body (channel flow).
    dt
        Total advection time for this frame, in lattice units.
        Split internally into ``n_substeps`` RK4 steps of size
        ``dt / n_substeps``.
    n_substeps
        RK4 substeps per frame. 4 matches the 2D code.
    max_age
        Frames after which a particle is culled regardless of
        position. Should be sized so a single inflow particle can
        drift past the outflow before fading: roughly
        ``Nx / (u_in * dt)`` frames, with a margin for slowdown
        in the wake. Same lesson as 2D's `max_age_local`.
    inflow_seed_xyz
        Optional ``(new_x, new_y, new_z)`` from
        ``seed_inflow_particles``. ``None`` means no spawn (useful
        for tests).

    Returns
    -------
    Updated (px, py, pz, age) with culled particles removed.
    """
    # 1. Spawn new inflow particles
    if inflow_seed_xyz is not None:
        seed_x, seed_y, seed_z = inflow_seed_xyz
        px = np.concatenate([px, seed_x])
        py = np.concatenate([py, seed_y])
        pz = np.concatenate([pz, seed_z])
        age = np.concatenate(
            [age, np.zeros(len(seed_x), dtype=np.int32)]
        )

    if len(px) == 0:
        return px, py, pz, age

    # D-10: n_substeps=0 would crash inside dt/float(n_substeps) with
    # ZeroDivisionError -- the test would otherwise have caught it but
    # production never passes 0 today. One-line guard so a future caller
    # gets a clear ValueError.
    if int(n_substeps) <= 0:
        raise ValueError(
            f"n_substeps must be >= 1, got {n_substeps!r}."
        )

    # 2. Advect via n_substeps RK4 steps
    dt_sub = dt / float(n_substeps)
    for _ in range(int(n_substeps)):
        px, py, pz = advect_rk4(px, py, pz, ux, uy, uz, dt_sub)

    age = age + 1

    # 3. Cull
    Nx, Ny, Nz = ux.shape
    in_x = (px >= 1.0) & (px < Nx - 1.5)
    in_y = (py >= 1.0) & (py < Ny - 1.5)
    in_z = (pz >= 1.0) & (pz < Nz - 1.5)
    if body_mask is not None:
        # D-10: body_mask shape mismatch with ux used to dump a deep
        # IndexError from body_mask[xi, yi, zi] when xi exceeded the
        # mask's bounds. Cheaper to fail at the boundary with the actual
        # mismatch info than chase it from the index error.
        if body_mask.shape != ux.shape:
            raise ValueError(
                f"body_mask.shape {body_mask.shape} != ux.shape "
                f"{ux.shape}; pass a body_mask sized to the velocity "
                f"field."
            )
        xi = np.clip(np.round(px).astype(np.int32), 0, Nx - 1)
        yi = np.clip(np.round(py).astype(np.int32), 0, Ny - 1)
        zi = np.clip(np.round(pz).astype(np.int32), 0, Nz - 1)
        in_body = body_mask[xi, yi, zi]
    else:
        in_body = np.zeros(len(px), dtype=bool)
    keep = in_x & in_y & in_z & (~in_body) & (age < max_age)

    return px[keep], py[keep], pz[keep], age[keep]
