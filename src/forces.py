"""Force calculation for LBM solid bodies via the momentum-exchange method.

Reference
---------
Ladd, A.J.C. "Numerical simulations of particulate suspensions via a discretized
Boltzmann equation. Part 1. Theoretical foundation." J. Fluid Mech. 271,
285-309 (1994).

The simplified momentum-exchange form used here -- valid for halfway bounce-back
on a flat or piecewise-flat wall (good for cylinders to within a few percent;
curved-wall refinements come from Mei-Luo 1999 if we need them later) -- is:

    F = sum over wall-links L of  2 * c_i * f_i(fluid cell of L, post-collision)

A "wall-link" is a pair (fluid cell x, solid cell x + c_i) where the fluid cell
is adjacent to a solid cell in direction i. The factor of 2 captures the full
momentum transfer: the bouncing particle carries +c_i * f_i in, leaves with
-c_i * f_i out, so total transferred to the wall is +2 * c_i * f_i.

The `f` used must be the value AFTER collide() but BEFORE bounce_back() at the
fluid cell -- the step ordering becomes:

    f_post_coll  =  collide(f, tau)
    F_step       =  momentum_exchange_force(f_post_coll, solid_mask)
    f            =  bounce_back(f_post_coll, solid_mask)
    f            =  stream(f)
"""
import numpy as np

from src.lbm import LATTICE_VELOCITIES


def momentum_exchange_force(f_post_collision: np.ndarray, solid_mask: np.ndarray) -> np.ndarray:
    """Compute the total force on a solid body via momentum exchange.

    Parameters
    ----------
    f_post_collision : ndarray of shape (9, Nx, Ny)
        Distribution function AFTER ``collide()`` and BEFORE ``bounce_back()``.
    solid_mask : ndarray of bool, shape (Nx, Ny)
        True where the cell is solid.

    Returns
    -------
    force : ndarray of shape (2,)
        Total ``[Fx, Fy]`` on the solid body, in lattice units. Drag is +x for
        east-flowing fluid; lift is +y (perpendicular to typical inflow).
    """
    F = np.zeros(2)
    fluid_mask = ~solid_mask

    # Skip i=0 (rest direction): c_0 = (0, 0) carries no momentum.
    for i in range(1, 9):
        cx_i = int(LATTICE_VELOCITIES[i, 0])
        cy_i = int(LATTICE_VELOCITIES[i, 1])

        # Shift the solid mask by -c_i so that at position x we know whether
        # x + c_i is solid. np.roll(mask, shift=(-cx, -cy)) produces:
        #     shifted[x, y] = mask[(x + cx) % Nx, (y + cy) % Ny]
        # (the modulo wrap is harmless here -- we'll AND with fluid_mask, and
        #  domain boundaries shouldn't be marked solid unless they're actually
        #  part of the body).
        solid_in_dir_i = np.roll(solid_mask, shift=(-cx_i, -cy_i), axis=(0, 1))

        # Wall-link mask: cells that are fluid AND have a solid neighbor in direction i.
        wall_link_mask = fluid_mask & solid_in_dir_i

        # Sum f_i over those cells.
        f_sum = float(f_post_collision[i][wall_link_mask].sum())

        # Contribution to force: 2 * c_i * sum_of_f_i.
        F[0] += 2.0 * cx_i * f_sum
        F[1] += 2.0 * cy_i * f_sum

    return F
