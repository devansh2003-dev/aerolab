"""Bouzidi interpolated bounce-back for 3D LBM — analytic q-fields.

Phase A2-FULL Part 1 (per D-4 / D-5 in 3D_PHASE0_DECISIONS.md). The
Phase A2-MVP uses full-way bounce-back: every wall link is treated as
if the boundary sits at the on-link midpoint (q = 0.5), regardless of
where the analytic surface actually is. That works for visuals but
introduces a viscosity-dependent error in Cd at the validation level.

Bouzidi 2001 ("A momentum-exchange-based ..." but the relevant chunk
is Eqs. 2-5) interpolates between the fluid-side and solid-side
populations using the actual wall fraction q. Combined with the
TRT magic parameter Λ = 3/16, this places the no-slip wall at the
mid-link position INDEPENDENT of viscosity -- the property that
directly serves Cd accuracy.

This module computes the q-field analytically for a sphere (and is
easy to extend for a cylinder). For uploaded meshes, the q-field is
NOT analytic — that path uses a sparse wall-link list per D-5 with
q = 0.5 initially and Bouzidi-against-voxelised-mesh as a v1.1
follow-on.

Output format: sparse wall-link list (per D-5). Five 1-D arrays
indexed by wall-link number:
    wall_x, wall_y, wall_z : int32   -- fluid cell coordinates
    wall_dir               : int32   -- lattice direction index (1..18)
    wall_q                 : float32 -- wall fraction in (0, 1]

The Bouzidi kernel reads each link, looks up the post-stream
population, and applies the interpolation. We keep the generator
pure-NumPy here; the inner-loop application belongs in a Numba step
function (Part 2 of this phase, separate commit).

Algorithm reference: D-4 in 3D_PHASE0_DECISIONS.md derives the
quadratic. d = x_fluid - x_centre. Wall crossing satisfies
|d + q c_i|² = R², expanding to A q² + B q + C = 0 with
A = |c_i|², B = 2 (d · c_i), C = |d|² - R². Smaller positive root
in (0, 1] is q.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from src.lbm_3d import (
    LATTICE_VELOCITIES_3D,
    LATTICE_WEIGHTS_3D,
    OPPOSITE_3D,
)


@dataclass
class WallLinkList:
    """Sparse wall-link list (per D-5).

    Each entry is a (fluid cell, lattice direction) pair where the
    neighbour in that direction is solid. The Bouzidi kernel iterates
    these once per step.

    Attributes are 1-D arrays of the same length `n_links`.
    """
    x: np.ndarray          # int32 -- fluid cell x
    y: np.ndarray          # int32 -- fluid cell y
    z: np.ndarray          # int32 -- fluid cell z
    dir: np.ndarray        # int32 -- lattice direction (1..18)
    q: np.ndarray          # float32 -- wall fraction in (0, 1]

    @property
    def n_links(self) -> int:
        return len(self.x)


def solve_bouzidi_q(
    d: tuple[float, float, float] | np.ndarray,
    c: tuple[int, int, int] | np.ndarray,
    R: float,
) -> float:
    """Solve A q² + B q + C = 0 for the Bouzidi wall fraction.

    Parameters
    ----------
    d : 3-tuple or array
        ``x_fluid - x_centre``: vector from the sphere centre to the
        FLUID cell whose link we are testing.
    c : 3-tuple or array
        Lattice direction (one of ±x, ±y, ±z, or an edge vector). Must
        be the integer direction, not a normalised unit vector.
    R : float
        Sphere radius in lattice units.

    Returns
    -------
    float
        The wall fraction q ∈ (0, 1] if the link crosses the surface,
        or -1.0 if it does not (no real root in the valid range).
        Returning a sentinel instead of raising keeps the inner loop
        branchless-friendly for a future Numba pass.

    Notes
    -----
    A = |c|² (1 for face directions, 2 for edge directions in D3Q19).
    B = 2 (d · c). C = |d|² - R². Smaller positive root in (0, 1].
    """
    d = np.asarray(d, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    A = float(np.dot(c, c))
    if A == 0.0:
        # Rest direction: cannot be a wall link.
        return -1.0
    B = 2.0 * float(np.dot(d, c))
    C = float(np.dot(d, d)) - R * R
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        # No real intersection -- the link does not cross the surface.
        return -1.0
    sqrt_disc = float(np.sqrt(disc))
    # Smaller positive root in (0, 1]. The "-" branch gives the smaller
    # value when A > 0 and the discriminant is real, which is always
    # the case here.
    q_minus = (-B - sqrt_disc) / (2.0 * A)
    q_plus = (-B + sqrt_disc) / (2.0 * A)
    for q in (q_minus, q_plus):
        if 0.0 < q <= 1.0:
            return float(q)
    return -1.0


def make_sphere_mask(
    Nx: int, Ny: int, Nz: int,
    cx: float, cy: float, cz: float,
    R: float,
) -> np.ndarray:
    """Boolean solid mask for a sphere centred at (cx, cy, cz), radius R.

    Same convention as `src.lbm_3d._make_sphere_mask` -- cells whose
    centre lies AT or INSIDE the sphere surface are solid
    (``|x - centre|² <= R²``). Mirror copy lives here so this module
    is self-contained and the wall-link generator does not pull a
    leading-underscore helper from another file.
    """
    xs = np.arange(Nx)[:, None, None]
    ys = np.arange(Ny)[None, :, None]
    zs = np.arange(Nz)[None, None, :]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2) <= R * R


def make_cylinder_mask(
    Nx: int, Ny: int, Nz: int,
    cx: float, cy: float,
    R: float,
) -> np.ndarray:
    """Boolean solid mask for a cylinder spanning the z direction.

    Centred at (cx, cy) in the (x, y) plane, radius R, spans the
    full z extent of the domain (z is the spanwise axis for the
    standard wind-tunnel orientation). Cells inside the circle in
    every (x, y) slice are solid:

        (xs - cx) ** 2 + (ys - cy) ** 2 <= R ** 2

    independent of z. The 3D extrude of the validated 2D cylinder --
    same wake physics (steady recirculation, then von Karman
    shedding above Re ~ 47) but with z-direction perturbations now
    physically representable.

    Returns
    -------
    mask : (Nx, Ny, Nz) bool ndarray
        True where the cell is solid.
    """
    xs = np.arange(Nx)[:, None, None]
    ys = np.arange(Ny)[None, :, None]
    in_circle_2d = (xs - cx) ** 2 + (ys - cy) ** 2 <= R * R
    # Broadcast the (Nx, Ny, 1) 2D circle across the full z range.
    return np.broadcast_to(in_circle_2d, (Nx, Ny, Nz)).copy()


def _naca_thickness(x: np.ndarray, thickness: float) -> np.ndarray:
    """NACA 4-digit half-thickness y_t(x) for chord-normalised x in
    [0, 1]. Standard analytic formula -- the closed-trailing-edge
    coefficient (-0.1036) is used so the airfoil tail meets cleanly
    in the voxelisation."""
    xc = np.clip(x, 0.0, 1.0)
    return 5.0 * thickness * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1036 * xc ** 4
    )


def _naca_camber(
    x: np.ndarray, m: float, p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean camber line y_c(x) and its slope dyc/dx for a NACA 4-digit
    airfoil with max camber m at chord-fraction p. Returns
    (camber, slope) arrays. For symmetric airfoils pass m = 0 -> both
    arrays are zero."""
    xc = np.clip(x, 0.0, 1.0)
    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    if m > 0.0 and 0.0 < p < 1.0:
        front = xc <= p
        yc[front] = (m / p ** 2) * (
            2.0 * p * xc[front] - xc[front] ** 2
        )
        dyc[front] = (2.0 * m / p ** 2) * (p - xc[front])
        back = ~front
        yc[back] = (m / (1.0 - p) ** 2) * (
            1.0 - 2.0 * p + 2.0 * p * xc[back] - xc[back] ** 2
        )
        dyc[back] = (2.0 * m / (1.0 - p) ** 2) * (p - xc[back])
    return yc, dyc


def naca_outline(
    m: float, p: float, thickness: float, n_pts: int = 81,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Return a closed (n_pts, 2) polyline tracing the airfoil
    surface in chord-normalised (x/c, y/c) coordinates. Used by both
    the 3D mask voxeliser and the Plotly mesh renderer so the body
    rendering matches the simulated body exactly.

    With ``aoa_deg`` non-zero the polyline is rotated about the chord
    midpoint (x/c = 0.5) by the given angle of attack -- positive
    rotates the leading edge UP (nose-up convention, matches 2D).
    """
    # Cosine spacing on the chord so the leading edge (where the
    # surface curvature is highest) is sampled finely.
    beta = np.linspace(0.0, np.pi, max(8, n_pts // 2))
    xs = 0.5 * (1.0 - np.cos(beta))
    yt = _naca_thickness(xs, thickness)
    yc, dyc = _naca_camber(xs, m, p)
    theta = np.arctan(dyc)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    upper_x = xs - yt * sin_t
    upper_y = yc + yt * cos_t
    lower_x = xs + yt * sin_t
    lower_y = yc - yt * cos_t
    # Closed loop: TE -> upper LE -> LE -> lower TE -> TE.
    poly = np.concatenate([
        np.stack([upper_x[::-1], upper_y[::-1]], axis=1),
        np.stack([lower_x[1:], lower_y[1:]], axis=1),
        np.stack([upper_x[:1], upper_y[:1]], axis=1),  # close
    ]).astype(np.float64)
    if abs(aoa_deg) > 1e-9:
        # Rotate the chord midpoint about itself. Positive AoA tilts
        # the leading edge up, matching the 2D convention.
        ang = np.deg2rad(aoa_deg)
        c, s = np.cos(ang), np.sin(ang)
        pivot = np.array([0.5, 0.0])
        d = poly - pivot
        poly = pivot + np.stack([
            d[:, 0] * c - d[:, 1] * s,
            d[:, 0] * s + d[:, 1] * c,
        ], axis=1)
    return poly


def make_naca_mask(
    Nx: int, Ny: int, Nz: int,
    x_le: float, y_chord: float, chord: float,
    m: float, p: float, thickness: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Boolean solid mask for a NACA 4-digit airfoil, extruded along
    the spanwise z axis. Leading edge at lattice x = x_le, chord aligned
    with +x and length ``chord`` cells, mean camber line at y = y_chord.

    With ``aoa_deg`` non-zero the airfoil is rotated about its chord
    midpoint (positive = leading edge up, nose-up convention). The
    voxel test inverse-rotates each query point back to the airfoil's
    canonical frame before checking the upper/lower bounds, so the
    same analytic thickness + camber formulas apply unchanged.
    """
    pivot_x = x_le + chord * 0.5
    pivot_y = y_chord
    cos_a = float(np.cos(np.deg2rad(aoa_deg)))
    sin_a = float(np.sin(np.deg2rad(aoa_deg)))
    xs_grid = np.arange(Nx, dtype=np.float64)[:, None]
    ys_grid = np.arange(Ny, dtype=np.float64)[None, :]
    # Inverse rotation (rotate query points by -aoa) to align with the
    # un-rotated airfoil.
    dx = xs_grid - pivot_x
    dy = ys_grid - pivot_y
    x_local = pivot_x + dx * cos_a + dy * sin_a
    y_local = pivot_y + (-dx * sin_a + dy * cos_a)
    xc = (x_local - x_le) / max(chord, 1.0)
    yc_norm = (y_local - y_chord) / max(chord, 1.0)
    yt = _naca_thickness(xc, thickness)
    camber_y, _ = _naca_camber(xc, m, p)
    in_chord = (xc >= 0.0) & (xc <= 1.0)
    y_upper = camber_y + yt
    y_lower = camber_y - yt
    inside_2d = (
        in_chord
        & (yc_norm <= y_upper)
        & (yc_norm >= y_lower)
    )
    return np.broadcast_to(
        inside_2d[:, :, None], (Nx, Ny, Nz)
    ).copy()


def make_cube_mask(
    Nx: int, Ny: int, Nz: int,
    cx: float, cy: float, cz: float,
    half_extent: float,
) -> np.ndarray:
    """Boolean solid mask for an axis-aligned cube centred at
    (cx, cy, cz) with half-edge ``half_extent``. Cells whose centres
    fall inside the cube (chebyshev distance from centre <= h) are
    solid. Voxel-stair walls are handled by ``voxel_wall_links``.
    """
    xs = np.arange(Nx)[:, None, None]
    ys = np.arange(Ny)[None, :, None]
    zs = np.arange(Nz)[None, None, :]
    return (
        (np.abs(xs - cx) <= half_extent)
        & (np.abs(ys - cy) <= half_extent)
        & (np.abs(zs - cz) <= half_extent)
    )


def sphere_wall_links(
    Nx: int, Ny: int, Nz: int,
    cx: float, cy: float, cz: float,
    R: float,
) -> WallLinkList:
    """Build the analytic Bouzidi wall-link list for a sphere.

    For every fluid cell `x_f` and every non-rest lattice direction
    `c_i` (i = 1..18), if the neighbour cell `x_f + c_i` is solid,
    compute the wall fraction `q` analytically and add the entry to
    the list. Out-of-domain neighbours are NOT wall links (they are
    domain-boundary links handled by inflow / outflow / wall BC code).

    The result is a sparse representation per D-5 in the memo: zero
    storage for the dense q-field (159 MB at 128³ for D3Q19), O(L²)
    storage for the boundary cells of a body of linear size L.

    Parameters
    ----------
    Nx, Ny, Nz
        Lattice dimensions.
    cx, cy, cz
        Sphere centre in lattice coordinates. Floats; the surface need
        not coincide with cell centres.
    R
        Sphere radius in lattice units. Must be positive.

    Returns
    -------
    WallLinkList
        Sparse representation of every wall link with its analytic q.

    Raises
    ------
    ValueError
        If R is non-positive or the grid is empty.
    """
    if R <= 0.0:
        raise ValueError(f"R must be positive; got R = {R}")
    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        raise ValueError(
            f"Grid must be positive on every axis; got "
            f"({Nx}, {Ny}, {Nz})"
        )

    mask = make_sphere_mask(Nx, Ny, Nz, cx, cy, cz, R)
    # If the sphere doesn't intersect the domain at all, short-circuit
    # to empty arrays -- saves the O(Nx Ny Nz) traversal below.
    if not mask.any():
        empty_i32 = np.empty(0, dtype=np.int32)
        return WallLinkList(
            x=empty_i32, y=empty_i32, z=empty_i32,
            dir=empty_i32, q=np.empty(0, dtype=np.float32),
        )

    # Walk every fluid cell. For each non-rest direction, check whether
    # the neighbour is solid; if so, compute q. We allocate lists and
    # convert at the end -- the wall-link count is O(L²) which is tiny
    # compared to the Nx Ny Nz cell scan.
    lat = LATTICE_VELOCITIES_3D
    wall_x: list[int] = []
    wall_y: list[int] = []
    wall_z: list[int] = []
    wall_dir: list[int] = []
    wall_q: list[float] = []

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                if mask[x, y, z]:
                    continue  # solid cells emit no wall links
                # Direction 0 is the rest vector; skip it.
                for i in range(1, 19):
                    ci0 = int(lat[i, 0])
                    ci1 = int(lat[i, 1])
                    ci2 = int(lat[i, 2])
                    xn = x + ci0
                    yn = y + ci1
                    zn = z + ci2
                    # Out-of-domain neighbours are handled by the
                    # domain-boundary BC code, not Bouzidi.
                    if not (0 <= xn < Nx and 0 <= yn < Ny and 0 <= zn < Nz):
                        continue
                    if not mask[xn, yn, zn]:
                        continue
                    # Solid neighbour: compute q.
                    q = solve_bouzidi_q(
                        (x - cx, y - cy, z - cz),
                        (ci0, ci1, ci2),
                        R,
                    )
                    if q < 0.0:
                        # No real root in (0, 1] -- shouldn't happen
                        # when the mask said the neighbour is solid,
                        # but skip defensively so a numerical edge
                        # case at R = exact-cell-centre can't insert
                        # a bogus q = -1 entry into the list.
                        continue
                    wall_x.append(x)
                    wall_y.append(y)
                    wall_z.append(z)
                    wall_dir.append(i)
                    wall_q.append(q)

    return WallLinkList(
        x=np.asarray(wall_x, dtype=np.int32),
        y=np.asarray(wall_y, dtype=np.int32),
        z=np.asarray(wall_z, dtype=np.int32),
        dir=np.asarray(wall_dir, dtype=np.int32),
        q=np.asarray(wall_q, dtype=np.float32),
    )


# ===========================================================================
# Bouzidi correction kernel (Phase A2-FULL Part 2)
# ===========================================================================
#
# The full-way bounce-back path in `src/lbm_3d.step_bgk_3d` writes
# `f_next[opp, x_f] = f_tilde_i(x_f)` whenever the neighbour of fluid
# cell x_f along direction i is solid. This is correct for q = 0.5
# (wall at on-link midpoint). For other q values it introduces a
# viscosity-dependent error in Cd.
#
# Bouzidi-Firdaouss-Lallemand 2001 ("Momentum transfer of a Boltzmann-
# lattice fluid with boundaries", Phys. Fluids 13:11) defines the
# linear-interpolation correction. After full-way BB has already
# populated f_next[opp, x_f] for every wall link, we run this post-
# pass to OVERRIDE those values with the q-correct Bouzidi result.
#
# At q = 0.5 the Bouzidi formula reduces to f_tilde_i(x_f), so this
# pass is a no-op there -- the q=0.5 test in `tests/test_lbm_3d_bouzidi.py`
# pins this invariant.


@njit(cache=True, fastmath=True)
def apply_bouzidi_correction(
    f_pre: np.ndarray,            # (19, Nx, Ny, Nz) PRE-collision populations
    f_next: np.ndarray,           # (19, Nx, Ny, Nz) post-stream populations (full-way BB applied)
    body: np.ndarray,             # (Nx, Ny, Nz) bool
    wall_x: np.ndarray,           # (N,) int32
    wall_y: np.ndarray,           # (N,) int32
    wall_z: np.ndarray,           # (N,) int32
    wall_dir: np.ndarray,         # (N,) int32
    wall_q: np.ndarray,           # (N,) float32
    omega: np.float32,
    u_in: np.float32,
) -> None:
    """Apply Bouzidi linear interpolation correction at every wall link.

    Reads ``f_pre`` (the populations BEFORE the step, kept unmodified by
    ``step_bgk_3d``) to recompute the post-collision ``f_tilde`` locally
    at the wall-link source cell (and, for q < 0.5, at the upstream
    cell). Overwrites ``f_next[opp, x_f]`` in place.

    Reads the lattice constants from `src.lbm_3d` module-level arrays
    (LATTICE_VELOCITIES_3D, LATTICE_WEIGHTS_3D, OPPOSITE_3D) so kernel
    and Bouzidi correction always agree on convention.

    The math:

      q >= 0.5:
        f_next[opp, x_f] = (1/(2q)) * f_tilde_i(x_f)
                         + ((2q - 1)/(2q)) * f_tilde_opp(x_f)

      q < 0.5:
        f_next[opp, x_f] = 2q * f_tilde_i(x_f)
                         + (1 - 2q) * f_tilde_i(x_f - c_i)

    Notes
    -----
    - ``f_pre`` MUST be the unmutated input to the step (not f_next).
      ``run_channel_smoke`` calls this BEFORE the f/f_next swap.
    - Upstream cells out of the domain or inside the body force a
      fall-back to full-way (q = 0.5 equivalent): we just leave
      f_next[opp, x_f] at the value step_bgk_3d already wrote.
    - Inflow override at x = 0 matches step_bgk_3d (rho = 1, u =
      (u_in, 0, 0)). Keeping this in sync is load-bearing -- if the
      step changes its inflow rule, this must change too.
    """
    Nx, Ny, Nz = body.shape
    n_links = wall_x.shape[0]
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * (1.0 / 3.0)))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for k in range(n_links):
        x = wall_x[k]
        y = wall_y[k]
        z = wall_z[k]
        i = wall_dir[k]
        q = wall_q[k]
        opp = OPPOSITE_3D[i]

        # ---- Compute macroscopic at the fluid cell (x_f) ----
        rho_xf = np.float32(0.0)
        mx = np.float32(0.0)
        my = np.float32(0.0)
        mz = np.float32(0.0)
        for j in range(19):
            fj = f_pre[j, x, y, z]
            rho_xf += fj
            mx += np.float32(LATTICE_VELOCITIES_3D[j, 0]) * fj
            my += np.float32(LATTICE_VELOCITIES_3D[j, 1]) * fj
            mz += np.float32(LATTICE_VELOCITIES_3D[j, 2]) * fj

        if x == 0:
            # Inflow override (must match step_bgk_3d exactly).
            rho_xf = np.float32(1.0)
            ux = u_in
            uy = np.float32(0.0)
            uz = np.float32(0.0)
        else:
            if rho_xf > np.float32(0.0):
                inv_rho = np.float32(1.0) / rho_xf
            else:
                inv_rho = np.float32(0.0)
            ux = mx * inv_rho
            uy = my * inv_rho
            uz = mz * inv_rho

        usq = ux * ux + uy * uy + uz * uz

        # ---- f_tilde_i at (x, y, z) ----
        cxi = np.float32(LATTICE_VELOCITIES_3D[i, 0])
        cyi = np.float32(LATTICE_VELOCITIES_3D[i, 1])
        czi = np.float32(LATTICE_VELOCITIES_3D[i, 2])
        wi = np.float32(LATTICE_WEIGHTS_3D[i])
        cu_i = cxi * ux + cyi * uy + czi * uz
        f_eq_i = wi * rho_xf * (
            np.float32(1.0)
            + cu_i / cs2
            + (cu_i * cu_i) * inv_2cs4
            - usq * inv_2cs2
        )
        if x == 0:
            # At the inlet step writes equilibrium directly (see
            # step_bgk_3d). f_tilde IS f_eq there.
            f_tilde_i = f_eq_i
        else:
            f_tilde_i = f_pre[i, x, y, z] + omega * (f_eq_i - f_pre[i, x, y, z])

        if q >= np.float32(0.5):
            # ---- q >= 0.5: need f_tilde_opp at (x, y, z) ----
            cxo = np.float32(LATTICE_VELOCITIES_3D[opp, 0])
            cyo = np.float32(LATTICE_VELOCITIES_3D[opp, 1])
            czo = np.float32(LATTICE_VELOCITIES_3D[opp, 2])
            wo = np.float32(LATTICE_WEIGHTS_3D[opp])
            cu_opp = cxo * ux + cyo * uy + czo * uz
            f_eq_opp = wo * rho_xf * (
                np.float32(1.0)
                + cu_opp / cs2
                + (cu_opp * cu_opp) * inv_2cs4
                - usq * inv_2cs2
            )
            if x == 0:
                f_tilde_opp = f_eq_opp
            else:
                f_tilde_opp = (
                    f_pre[opp, x, y, z]
                    + omega * (f_eq_opp - f_pre[opp, x, y, z])
                )
            inv_2q = np.float32(1.0) / (np.float32(2.0) * q)
            f_next[opp, x, y, z] = (
                inv_2q * f_tilde_i
                + (np.float32(2.0) * q - np.float32(1.0)) * inv_2q * f_tilde_opp
            )
        else:
            # ---- q < 0.5: need f_tilde_i at upstream cell (x - c_i) ----
            xu = x - LATTICE_VELOCITIES_3D[i, 0]
            yu = y - LATTICE_VELOCITIES_3D[i, 1]
            zu = z - LATTICE_VELOCITIES_3D[i, 2]
            in_domain = (
                xu >= 0 and xu < Nx
                and yu >= 0 and yu < Ny
                and zu >= 0 and zu < Nz
            )
            if (not in_domain) or body[xu, yu, zu]:
                # Upstream cell unavailable -- leave the full-way BB
                # value step_bgk_3d already wrote. Slightly less
                # accurate at narrow gaps but never wrong.
                continue
            # Compute macroscopic at upstream cell
            rho_u = np.float32(0.0)
            mxu = np.float32(0.0)
            myu = np.float32(0.0)
            mzu = np.float32(0.0)
            for j in range(19):
                fj = f_pre[j, xu, yu, zu]
                rho_u += fj
                mxu += np.float32(LATTICE_VELOCITIES_3D[j, 0]) * fj
                myu += np.float32(LATTICE_VELOCITIES_3D[j, 1]) * fj
                mzu += np.float32(LATTICE_VELOCITIES_3D[j, 2]) * fj
            if xu == 0:
                rho_u = np.float32(1.0)
                uxu = u_in
                uyu = np.float32(0.0)
                uzu = np.float32(0.0)
            else:
                if rho_u > np.float32(0.0):
                    inv_rho_u = np.float32(1.0) / rho_u
                else:
                    inv_rho_u = np.float32(0.0)
                uxu = mxu * inv_rho_u
                uyu = myu * inv_rho_u
                uzu = mzu * inv_rho_u
            usq_u = uxu * uxu + uyu * uyu + uzu * uzu
            cu_iu = cxi * uxu + cyi * uyu + czi * uzu
            f_eq_iu = wi * rho_u * (
                np.float32(1.0)
                + cu_iu / cs2
                + (cu_iu * cu_iu) * inv_2cs4
                - usq_u * inv_2cs2
            )
            if xu == 0:
                f_tilde_iu = f_eq_iu
            else:
                f_tilde_iu = (
                    f_pre[i, xu, yu, zu]
                    + omega * (f_eq_iu - f_pre[i, xu, yu, zu])
                )
            f_next[opp, x, y, z] = (
                np.float32(2.0) * q * f_tilde_i
                + (np.float32(1.0) - np.float32(2.0) * q) * f_tilde_iu
            )


# ===========================================================================
# TRT variant of the Bouzidi correction kernel
# ===========================================================================
#
# The BGK kernel above is wired into `run_channel_smoke` in `src/lbm_3d.py`,
# which uses single-omega BGK collision + channel inflow/outflow. The
# production validation track uses the TRT kernels in `src/lbm_3d_trt.py`
# (Λ = 3/16 magic parameter: mid-link wall placement independent of
# viscosity, which is the property that buys Cd accuracy).
#
# The Bouzidi correction is collision-AGNOSTIC at the formula level — it
# only needs the post-collision pre-stream `f_tilde` values. The only thing
# that changes vs the BGK kernel is how those `f_tilde` get computed: TRT
# uses the symmetric/antisymmetric split with (s_plus, s_minus) instead of
# the single-omega BGK relaxation.
#
# This variant has NO inflow override. The TRT module's kernels are pure
# periodic-box; when a TRT channel driver is added it will need its own
# variant (or a parameter to enable the override) that mirrors whatever
# inflow rule the new driver picks.


@njit(cache=True, fastmath=True)
def apply_bouzidi_correction_trt(
    f_pre: np.ndarray,            # (19, Nx, Ny, Nz) PRE-step populations
    f_next: np.ndarray,           # (19, Nx, Ny, Nz) post-stream populations
    body: np.ndarray,             # (Nx, Ny, Nz) bool
    wall_x: np.ndarray,           # (N,) int32
    wall_y: np.ndarray,           # (N,) int32
    wall_z: np.ndarray,           # (N,) int32
    wall_dir: np.ndarray,         # (N,) int32
    wall_q: np.ndarray,           # (N,) float32
    s_plus: np.float32,
    s_minus: np.float32,
) -> None:
    """Apply Bouzidi linear interpolation correction with TRT collision.

    Same architecture as ``apply_bouzidi_correction`` (BGK variant above):
    runs as a post-pass over the sparse wall-link list, reads ``f_pre``
    (the unmutated input to the TRT step), recomputes ``f_tilde`` locally
    at the wall-link source cell (and at the upstream cell when q < 0.5),
    and overrides ``f_next[opp, x_f]`` in place.

    TRT post-collision per direction (Ginzburg, Verhaeghe, d'Humières 2008):

      f_tilde_i   = f_i   - s_plus * (fp - ep) - s_minus * (fm - em)
      f_tilde_opp = f_opp - s_plus * (fp - ep) + s_minus * (fm - em)

    where ``fp = (f_i + f_opp) / 2``, ``fm = (f_i - f_opp) / 2``,
    ``ep = (e_i + e_opp) / 2``, ``em = (e_i - e_opp) / 2``, and
    ``e_i, e_opp`` are the local equilibria. BGK is the special case
    ``s_plus = s_minus = omega``; substituting in the formulas recovers
    ``f_tilde_i = f_i - omega * (f_i - e_i)`` exactly.

    No inflow override: the TRT module's kernels (``trt_periodic_step``
    and variants in ``src/lbm_3d_trt.py``) are periodic-box only.
    """
    Nx, Ny, Nz = body.shape
    n_links = wall_x.shape[0]
    cs2 = np.float32(1.0 / 3.0)
    inv_2cs2 = np.float32(1.0 / (2.0 * (1.0 / 3.0)))
    inv_2cs4 = np.float32(1.0 / (2.0 * (1.0 / 3.0) * (1.0 / 3.0)))

    for k in range(n_links):
        x = wall_x[k]
        y = wall_y[k]
        z = wall_z[k]
        i = wall_dir[k]
        q = wall_q[k]
        opp = OPPOSITE_3D[i]

        # ---- Local macroscopic at the fluid cell (x_f) ----
        rho_xf = np.float32(0.0)
        mx = np.float32(0.0)
        my = np.float32(0.0)
        mz = np.float32(0.0)
        for j in range(19):
            fj = f_pre[j, x, y, z]
            rho_xf += fj
            mx += np.float32(LATTICE_VELOCITIES_3D[j, 0]) * fj
            my += np.float32(LATTICE_VELOCITIES_3D[j, 1]) * fj
            mz += np.float32(LATTICE_VELOCITIES_3D[j, 2]) * fj
        if rho_xf > np.float32(0.0):
            inv_rho = np.float32(1.0) / rho_xf
        else:
            inv_rho = np.float32(0.0)
        ux = mx * inv_rho
        uy = my * inv_rho
        uz = mz * inv_rho
        usq = ux * ux + uy * uy + uz * uz

        # ---- Equilibria at (x_f) for i and opp ----
        cxi = np.float32(LATTICE_VELOCITIES_3D[i, 0])
        cyi = np.float32(LATTICE_VELOCITIES_3D[i, 1])
        czi = np.float32(LATTICE_VELOCITIES_3D[i, 2])
        wi = np.float32(LATTICE_WEIGHTS_3D[i])
        cxo = np.float32(LATTICE_VELOCITIES_3D[opp, 0])
        cyo = np.float32(LATTICE_VELOCITIES_3D[opp, 1])
        czo = np.float32(LATTICE_VELOCITIES_3D[opp, 2])
        wo = np.float32(LATTICE_WEIGHTS_3D[opp])
        cu_i = cxi * ux + cyi * uy + czi * uz
        cu_o = cxo * ux + cyo * uy + czo * uz
        e_i = wi * rho_xf * (
            np.float32(1.0)
            + cu_i / cs2
            + (cu_i * cu_i) * inv_2cs4
            - usq * inv_2cs2
        )
        e_o = wo * rho_xf * (
            np.float32(1.0)
            + cu_o / cs2
            + (cu_o * cu_o) * inv_2cs4
            - usq * inv_2cs2
        )

        # ---- TRT post-collision split at (x_f) ----
        f_i = f_pre[i, x, y, z]
        f_o = f_pre[opp, x, y, z]
        fp = np.float32(0.5) * (f_i + f_o)
        fm = np.float32(0.5) * (f_i - f_o)
        ep = np.float32(0.5) * (e_i + e_o)
        em = np.float32(0.5) * (e_i - e_o)
        f_tilde_i = f_i - s_plus * (fp - ep) - s_minus * (fm - em)
        f_tilde_o = f_o - s_plus * (fp - ep) + s_minus * (fm - em)

        if q >= np.float32(0.5):
            inv_2q = np.float32(1.0) / (np.float32(2.0) * q)
            f_next[opp, x, y, z] = (
                inv_2q * f_tilde_i
                + (np.float32(2.0) * q - np.float32(1.0)) * inv_2q * f_tilde_o
            )
        else:
            # q < 0.5: need f_tilde_i at upstream cell (x_f - c_i)
            xu = x - LATTICE_VELOCITIES_3D[i, 0]
            yu = y - LATTICE_VELOCITIES_3D[i, 1]
            zu = z - LATTICE_VELOCITIES_3D[i, 2]
            in_domain = (
                xu >= 0 and xu < Nx
                and yu >= 0 and yu < Ny
                and zu >= 0 and zu < Nz
            )
            if (not in_domain) or body[xu, yu, zu]:
                # Fall back to full-way BB (= f_tilde_i): the value the
                # caller's wall-handling already wrote stays.
                continue

            # ---- Local macroscopic at upstream cell ----
            rho_u = np.float32(0.0)
            mxu = np.float32(0.0)
            myu = np.float32(0.0)
            mzu = np.float32(0.0)
            for j in range(19):
                fj = f_pre[j, xu, yu, zu]
                rho_u += fj
                mxu += np.float32(LATTICE_VELOCITIES_3D[j, 0]) * fj
                myu += np.float32(LATTICE_VELOCITIES_3D[j, 1]) * fj
                mzu += np.float32(LATTICE_VELOCITIES_3D[j, 2]) * fj
            if rho_u > np.float32(0.0):
                inv_rho_u = np.float32(1.0) / rho_u
            else:
                inv_rho_u = np.float32(0.0)
            uxu = mxu * inv_rho_u
            uyu = myu * inv_rho_u
            uzu = mzu * inv_rho_u
            usq_u = uxu * uxu + uyu * uyu + uzu * uzu
            cu_iu = cxi * uxu + cyi * uyu + czi * uzu
            cu_ou = cxo * uxu + cyo * uyu + czo * uzu
            e_iu = wi * rho_u * (
                np.float32(1.0)
                + cu_iu / cs2
                + (cu_iu * cu_iu) * inv_2cs4
                - usq_u * inv_2cs2
            )
            e_ou = wo * rho_u * (
                np.float32(1.0)
                + cu_ou / cs2
                + (cu_ou * cu_ou) * inv_2cs4
                - usq_u * inv_2cs2
            )
            f_iu = f_pre[i, xu, yu, zu]
            f_ou = f_pre[opp, xu, yu, zu]
            fp_u = np.float32(0.5) * (f_iu + f_ou)
            fm_u = np.float32(0.5) * (f_iu - f_ou)
            ep_u = np.float32(0.5) * (e_iu + e_ou)
            em_u = np.float32(0.5) * (e_iu - e_ou)
            f_tilde_iu = (
                f_iu - s_plus * (fp_u - ep_u) - s_minus * (fm_u - em_u)
            )
            f_next[opp, x, y, z] = (
                np.float32(2.0) * q * f_tilde_i
                + (np.float32(1.0) - np.float32(2.0) * q) * f_tilde_iu
            )


__all__ = [
    "WallLinkList",
    "make_cube_mask",
    "make_cylinder_mask",
    "make_naca_mask",
    "naca_outline",
    "solve_bouzidi_q",
    "make_sphere_mask",
    "sphere_wall_links",
    "apply_bouzidi_correction",
    "apply_bouzidi_correction_trt",
]
