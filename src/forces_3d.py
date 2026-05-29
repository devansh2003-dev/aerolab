"""3D force calculation for D3Q19 LBM bodies via momentum exchange.

Reference
---------
Ladd, A.J.C. "Numerical simulations of particulate suspensions via a
discretized Boltzmann equation. Part 1. Theoretical foundation."
J. Fluid Mech. 271, 285-309 (1994).

The simplified momentum-exchange form used here (valid for halfway
bounce-back; gives sphere Cd to within ~5-10% of Mei-Yu-Shyy-Luo 2002
refinements at our grid resolution and is enough to validate the
solver against the canonical Clift-Grace-Weber 1978 curve):

    F = sum over wall-links L of  2 * c_i * f_i(x_f, post-collision)

Direct 3D analogue of ``src/forces.py``. A wall-link is a pair
(fluid cell x_f, solid cell x_f + c_i) where the fluid cell is
adjacent to a solid cell in direction i.

Two API entry points -- both compute the same physical quantity but
expect different population timings:

* ``momentum_exchange_force_3d(f_post_collision, body)`` consumes the
  state AFTER collide() and BEFORE bounce_back() / streaming. This is
  the canonical Ladd 1994 input, the one the 2D production path uses,
  and the one to prefer when the caller has access to the inside of
  the step kernel.

* ``momentum_exchange_force_3d_post_stream(f_post_stream, body, opp)``
  consumes the natural exit state of ``run_channel_smoke_trt`` -- the
  post-streaming, pre-next-collision populations. Same physics, but
  the formula reads ``f_opp[i](x_f)`` instead of ``f_i(x_f)`` because
  halfway bounce-back reflects the outgoing population back into the
  opposite-direction slot at the fluid cell after streaming.
"""
import numpy as np

from src.lbm_3d import LATTICE_VELOCITIES_3D, OPPOSITE_3D


def momentum_exchange_force_3d(
    f_post_collision: np.ndarray,
    solid_mask: np.ndarray,
) -> np.ndarray:
    """Compute the total force on a 3D solid body via momentum exchange.

    Parameters
    ----------
    f_post_collision : ndarray of shape (19, Nx, Ny, Nz)
        D3Q19 distribution function AFTER ``collide()`` and BEFORE
        ``bounce_back()``.
    solid_mask : ndarray of bool, shape (Nx, Ny, Nz)
        True where the cell is solid.

    Returns
    -------
    force : ndarray of shape (3,)
        Total ``[Fx, Fy, Fz]`` on the solid body, in lattice units.
        Drag is +x for east-flowing fluid; lift / side-force are y, z.
    """
    F = np.zeros(3, dtype=np.float64)
    fluid_mask = ~solid_mask

    # Skip i=0 (rest direction): c_0 = (0,0,0) carries no momentum.
    for i in range(1, 19):
        cx_i = int(LATTICE_VELOCITIES_3D[i, 0])
        cy_i = int(LATTICE_VELOCITIES_3D[i, 1])
        cz_i = int(LATTICE_VELOCITIES_3D[i, 2])

        # Shift the solid mask by -c_i so that at position x we know
        # whether x + c_i is solid. ``np.roll(mask, shift=(-cx,-cy,-cz))``
        # gives shifted[x,y,z] = mask[(x+cx)%Nx, (y+cy)%Ny, (z+cz)%Nz];
        # we AND with fluid_mask so the boundary-wrap is harmless.
        solid_in_dir_i = np.roll(
            solid_mask,
            shift=(-cx_i, -cy_i, -cz_i),
            axis=(0, 1, 2),
        )
        wall_link_mask = fluid_mask & solid_in_dir_i

        f_sum = float(f_post_collision[i][wall_link_mask].sum())
        F[0] += 2.0 * cx_i * f_sum
        F[1] += 2.0 * cy_i * f_sum
        F[2] += 2.0 * cz_i * f_sum

    return F


def momentum_exchange_force_3d_post_stream(
    f_post_stream: np.ndarray,
    solid_mask: np.ndarray,
) -> np.ndarray:
    """Compute the total force from the POST-STREAM populations.

    ``run_channel_smoke_trt`` returns the post-streaming state -- that
    is, every population has already been (a) collided, (b) bounced
    back at solid cells, and (c) streamed. To evaluate the Ladd 1994
    formula at this point, the population that left the fluid cell
    x_f going INTO the wall (direction c_i) has been reflected and
    now occupies the OPPOSITE slot at the same fluid cell x_f:

        f_opp[i](x_f, post-stream) == f_i(x_f, post-collision)
                                            for halfway BB

    So the sum becomes ``2 * c_i * f_opp[i](x_f)`` over wall-links.
    Note the sign relative to the canonical formula: that ``c_i``
    still points from the fluid cell INTO the wall, but the population
    being multiplied is the one that has already been reflected.

    Equivalent expression (and the form that follows the standard
    direct-momentum-exchange derivation in steady flow):

        F = - sum_links 2 * c_opp[i] * f_opp[i](x_f, post-stream)

    which is what we evaluate. The result is the force ON the body in
    the +x_drag direction, the same convention as
    ``momentum_exchange_force_3d``.
    """
    F = np.zeros(3, dtype=np.float64)
    fluid_mask = ~solid_mask
    opp = OPPOSITE_3D

    for i in range(1, 19):
        cx_i = int(LATTICE_VELOCITIES_3D[i, 0])
        cy_i = int(LATTICE_VELOCITIES_3D[i, 1])
        cz_i = int(LATTICE_VELOCITIES_3D[i, 2])
        opp_i = int(opp[i])

        solid_in_dir_i = np.roll(
            solid_mask,
            shift=(-cx_i, -cy_i, -cz_i),
            axis=(0, 1, 2),
        )
        wall_link_mask = fluid_mask & solid_in_dir_i

        # f_opp[i](x_f, post-stream) corresponds to f_i(x_f, post-collide)
        # at the same fluid cell, but the direction the population is
        # carrying is c_opp[i] = -c_i. Net momentum transferred to wall:
        #     2 * c_i * f_i_post_collide
        #   = 2 * c_i * f_opp[i]_post_stream
        f_sum = float(f_post_stream[opp_i][wall_link_mask].sum())
        F[0] += 2.0 * cx_i * f_sum
        F[1] += 2.0 * cy_i * f_sum
        F[2] += 2.0 * cz_i * f_sum

    return F


def drag_coefficient_3d(
    F_drag: float,
    rho_ref: float,
    u_ref: float,
    A_proj: float,
) -> float:
    """Drag coefficient from a momentum-exchange force.

    Cd = F_drag / (0.5 * rho_ref * u_ref^2 * A_proj)

    For a sphere of radius R in axial flow, ``A_proj = pi * R**2``.
    For a cylinder of diameter D and span L, ``A_proj = D * L``.
    """
    if u_ref <= 0.0 or A_proj <= 0.0:
        raise ValueError(
            f"u_ref and A_proj must be positive (got "
            f"u_ref={u_ref}, A_proj={A_proj})."
        )
    return float(F_drag / (0.5 * rho_ref * u_ref * u_ref * A_proj))


__all__ = [
    "momentum_exchange_force_3d",
    "momentum_exchange_force_3d_post_stream",
    "drag_coefficient_3d",
]
