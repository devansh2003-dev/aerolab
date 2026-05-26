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

from src.lbm_3d import LATTICE_VELOCITIES_3D


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


__all__ = [
    "WallLinkList",
    "solve_bouzidi_q",
    "make_sphere_mask",
    "sphere_wall_links",
]
