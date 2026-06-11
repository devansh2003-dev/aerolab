"""Q-criterion isosurface extraction for 3D LBM velocity fields.

Q = (1/2) (|Ω|² − |S|²) where S = sym(∇u) is the strain rate tensor and
Ω = antisym(∇u) is the vorticity tensor. A positive Q means rotation
dominates strain locally; the connected regions with Q > Q_threshold are
the "vortex tubes" that Q-criterion is designed to highlight.

Reference: Hunt, Wray, Moin (1988), "Eddies, streams, and convergence
zones in turbulent flows", CTR-S88. Q-criterion has remained the most
common vortex identifier in CFD post-processing for thirty years
because it correctly disqualifies pure shear (which would be confused
with rotation by the simpler |Ω| threshold).

This module is the **first Phase A3 visual primitive**: takes the
(ux, uy, uz, body) the LBM solver returned, computes Q on the lattice,
extracts an isosurface at a user-chosen level via marching cubes, and
returns vertices and faces ready for ``plotly.graph_objects.Mesh3d``.

The Q array is computed via NumPy central differences (``np.gradient``).
For a 96³ grid that is ~5 ms; small compared to the simulation cost,
so we do not Numba-JIT it. Marching cubes from scikit-image then
extracts the triangulated surface.
"""
from __future__ import annotations

import numpy as np


def compute_q_field(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    body: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Q-criterion field on the lattice.

    Parameters
    ----------
    ux, uy, uz : (Nx, Ny, Nz) float32 or float64
        Velocity components on the lattice.
    body : (Nx, Ny, Nz) bool, optional
        If provided, Q is zeroed inside solid cells (where the velocity
        gradients are not meaningful -- the solver pins u to 0 there
        and the discontinuity at the surface would produce spurious
        peaks in ``np.gradient``).

    Returns
    -------
    Q : (Nx, Ny, Nz) float32
        Q = (1/2) (|Ω|² − |S|²). Positive in vortex cores, negative in
        regions of pure strain, zero in quiescent flow.

    Notes
    -----
    ``np.gradient`` uses second-order accurate central differences in
    the interior and first-order one-sided at the boundaries. The
    boundary cells are NOT zeroed here -- the caller can choose to
    clip them off the isosurface render if the one-sided values
    produce visual artifacts at the inlet/outlet plane.

    The norms are Frobenius norms squared:
      |S|² = Σᵢⱼ Sᵢⱼ² = Sxx² + Syy² + Szz² + 2(Sxy² + Sxz² + Syz²)
      |Ω|² = Σᵢⱼ Ωᵢⱼ² = 2(Ωxy² + Ωxz² + Ωyz²)
    """
    # Central differences. np.gradient axis order matches array axes.
    dux_dx, dux_dy, dux_dz = np.gradient(ux)
    duy_dx, duy_dy, duy_dz = np.gradient(uy)
    duz_dx, duz_dy, duz_dz = np.gradient(uz)

    # Strain rate tensor (symmetric).
    S_xx = dux_dx
    S_yy = duy_dy
    S_zz = duz_dz
    S_xy = 0.5 * (dux_dy + duy_dx)
    S_xz = 0.5 * (dux_dz + duz_dx)
    S_yz = 0.5 * (duy_dz + duz_dy)

    # Vorticity tensor (antisymmetric); only the upper-triangular
    # components are independent.
    W_xy = 0.5 * (dux_dy - duy_dx)
    W_xz = 0.5 * (dux_dz - duz_dx)
    W_yz = 0.5 * (duy_dz - duz_dy)

    # Frobenius norms squared. The off-diagonal terms get a factor of
    # 2 because both Sᵢⱼ and Sⱼᵢ contribute -- they are equal by
    # symmetry, so we square once and double.
    S_norm_sq = (
        S_xx * S_xx + S_yy * S_yy + S_zz * S_zz
        + 2.0 * (S_xy * S_xy + S_xz * S_xz + S_yz * S_yz)
    )
    W_norm_sq = 2.0 * (W_xy * W_xy + W_xz * W_xz + W_yz * W_yz)

    Q = 0.5 * (W_norm_sq - S_norm_sq)
    Q = Q.astype(np.float32, copy=False)

    if body is not None:
        # Inside the body, the LBM driver pins velocity to zero, so
        # ∇u right at the surface has a discontinuity that produces
        # a spurious Q halo. Clear the body interior. (We do NOT clear
        # the surface cells -- the user often wants to see Q close to
        # the body, just not the artificial halo from inside.)
        Q = np.where(body, np.float32(0.0), Q)

    return Q


def extract_q_isosurface(
    Q: np.ndarray,
    level: float,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the Q = level isosurface via marching cubes.

    Returns ``(vertices, faces)`` ready for ``plotly.graph_objects.Mesh3d``
    (``x = verts[:, 0]``, ``y = verts[:, 1]``, ``z = verts[:, 2]``, and
    ``i / j / k`` from the three columns of ``faces``).

    Returns ``None`` when the chosen level lies entirely outside the
    range of Q (``level >= Q.max()`` or ``level <= Q.min()``) -- nothing
    to render. The dev-bench treats this as "no vortex structure at
    this threshold, lower it".

    Parameters
    ----------
    Q : (Nx, Ny, Nz) float
        Q-criterion field from :func:`compute_q_field`.
    level : float
        Q threshold defining the isosurface. Positive values isolate
        vortex tubes; choosing the right level is empirical and
        flow-dependent. A reasonable starting point is
        ``level = 0.05 * (Q.max() - Q.min()) + Q.min()`` clipped to
        positive values.
    spacing : 3-tuple
        Physical grid spacing along each axis. Defaults to unit cells
        (lattice units). Plotly will render in the same units.
    """
    from skimage.measure import marching_cubes

    # D-10: a NaN-contaminated Q field (rare but possible after a
    # divergent solve or a bake with NaN holes) used to dump an opaque
    # marching_cubes internal error. Bail to None so the caller treats
    # it the same as "no isosurface" and the chart just omits the shell.
    if not np.all(np.isfinite(Q)):
        return None

    q_min = float(Q.min())
    q_max = float(Q.max())
    if level >= q_max or level <= q_min:
        # No isosurface at this level: marching_cubes would raise.
        return None

    verts, faces, _, _ = marching_cubes(Q, level=level, spacing=spacing)
    return verts.astype(np.float32), faces.astype(np.int32)


def suggest_q_level(Q: np.ndarray, fraction: float = 0.10) -> float:
    """Suggest a sensible Q-isosurface level for a given field.

    Returns ``fraction * Q_max`` clipped to a strictly positive value
    (negative-Q isosurfaces correspond to strain-dominated regions, not
    vortex cores, and are rarely what the user wants to see by
    default). Empirical default is 10 % of the maximum -- captures the
    main tube structure on most channel-flow wakes we've seen.
    """
    q_max = float(Q.max())
    if q_max <= 0.0:
        # No positive-Q region (flow is purely strain). Returning a
        # tiny positive number ensures marching cubes returns None
        # rather than an isosurface inside the strain region.
        return 1e-12
    return max(fraction * q_max, 1e-12)


__all__ = [
    "compute_q_field",
    "extract_q_isosurface",
    "suggest_q_level",
]
