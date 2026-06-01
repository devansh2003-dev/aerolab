"""3D force calculation for D3Q19 LBM bodies via momentum exchange.

Reference
---------
Ladd, A.J.C. "Numerical simulations of particulate suspensions via a
discretized Boltzmann equation. Part 1. Theoretical foundation."
J. Fluid Mech. 271, 285-309 (1994).

The simplified momentum-exchange form used here (valid for halfway
bounce-back) is:

    F = sum over wall-links L of  2 * c_i * f_i(x_f, post-collision)

**Accuracy note (revised 2026-05-31).** An earlier version of this
docstring claimed this Ladd form lands "within ~5-10% of Mei-Yu-Shyy-Luo
2002 refinements." VALIDATION.md sec 8.3 / 8.3.1 has since measured
the sphere Cd at D = 20 / B = 25 % and found Cd = 1.572 - 1.645
versus Clift-Grace-Weber 1.09 (+44 % to +51 %). The low-blockage
cross-check at B = 25 % falsified the original "blockage dominates"
hypothesis; the bulk of the residual error is now attributed to
**this simplified Ladd 1994 form on a D = 20 grid** -- specifically,
not weighting each link by its q-fraction (which MYSL 2002 does).
The MYSL upgrade + D >= 40 grid (Mei-Luo-Shyy 1999) is the priority
3D-validation next step in VALIDATION.md sec 8.7, and should close
the bulk of the gap to a percent-level Cd.

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

from src.lbm_3d import LATTICE_VELOCITIES_3D, LATTICE_WEIGHTS_3D, OPPOSITE_3D


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


def momentum_exchange_force_3d_mysl(
    f_state: np.ndarray,
    wall_links,
    body: np.ndarray,
    nu: float,
    s_minus: float | None = None,
) -> np.ndarray:
    """MYSL 2002 Bouzidi-aware momentum exchange on a 3D body.

    Implements the Mei-Yu-Shyy-Luo 2002 momentum-exchange formula
    (Phys. Rev. E 65, 041203) specialised to Bouzidi quadratic-
    bounce-back walls:

        F_link = c_i * (f_tilde_i(x_f) + f_opp_post_BB(x_f))

    where ``f_tilde_i`` is the TRT post-collision i-direction
    population at the fluid cell ``x_f``, and ``f_opp_post_BB`` is
    the Bouzidi q-aware bounce-back result for the opposite
    direction. At ``q = 0.5`` the Bouzidi formula collapses to
    ``f_opp_post_BB = f_tilde_i``, recovering the Ladd 1994
    simplified form exactly -- so this function reduces to
    ``2 * c_i * f_tilde_i`` in the halfway-wall limit, and the
    parity test ``test_forces_3d_mysl.py`` locks that in.

    For ``q != 0.5`` (curved walls voxelised at finite resolution)
    MYSL accounts for the actual wall position via q. The simplified
    Ladd form used in ``momentum_exchange_force_3d`` and
    ``momentum_exchange_force_3d_post_stream`` does NOT, which is
    the dominant residual bias documented in VALIDATION.md sec 8.3.3.

    Parameters
    ----------
    f_state : ndarray of shape (19, Nx, Ny, Nz)
        Population field. For a converged steady bake this is the
        returned state of ``run_channel_smoke_trt`` (post-stream,
        post-BB). We re-derive the post-collision ``f_tilde_i`` and
        ``f_tilde_opp`` at wall-link cells via a one-step lookahead
        through the TRT split, which is exact at convergence.
    wall_links : WallLinkList
        Sparse wall-link list (``src.lbm_3d_bouzidi.WallLinkList``)
        with the per-link q-fractions used during the solve.
    body : ndarray of shape (Nx, Ny, Nz) bool
        Solid mask. Used to check whether the upstream cell for the
        ``q < 0.5`` branch is fluid (cannot recover f_tilde from a
        solid cell -- fall back to f_tilde at x_f).
    nu : float
        Kinematic viscosity, used to derive ``s_plus = 1 / (3 nu + 0.5)``
        for the TRT post-collision split.
    s_minus : float, optional
        Override for the TRT minus-mode relaxation rate. If None,
        derived from the magic-parameter constraint
        ``Lambda = (1/s_plus - 0.5)(1/s_minus - 0.5) = 3/16``
        (Ginzburg-Verhaeghe-d'Humieres 2008), which is what
        ``run_channel_smoke_trt`` uses by default.

    Returns
    -------
    F : ndarray of shape (3,)
        Total drag force on the body, in the convention of
        ``momentum_exchange_force_3d`` (positive ``F[0]`` = drag
        in the +x direction).
    """
    Nx, Ny, Nz = body.shape

    # TRT relaxation rates. Lambda = 3/16 default mirrors lbm_3d_trt.
    s_plus = 1.0 / (3.0 * float(nu) + 0.5)
    if s_minus is None:
        LAMBDA = 3.0 / 16.0
        s_minus = 1.0 / (LAMBDA / (1.0 / s_plus - 0.5) + 0.5)

    F = np.zeros(3, dtype=np.float64)
    wx, wy, wz = wall_links.x, wall_links.y, wall_links.z
    wdir = wall_links.dir
    wq = wall_links.q

    # Per-direction processing -- groups links so the rho/u/eq
    # reconstruction is one vectorised pass per direction.
    for i in range(1, 19):
        dir_mask = wdir == i
        if not dir_mask.any():
            continue
        opp = int(OPPOSITE_3D[i])
        cx_i = int(LATTICE_VELOCITIES_3D[i, 0])
        cy_i = int(LATTICE_VELOCITIES_3D[i, 1])
        cz_i = int(LATTICE_VELOCITIES_3D[i, 2])
        wt_i = float(LATTICE_WEIGHTS_3D[i])
        wt_o = float(LATTICE_WEIGHTS_3D[opp])

        x = wx[dir_mask].astype(np.int64)
        y = wy[dir_mask].astype(np.int64)
        z = wz[dir_mask].astype(np.int64)
        q = wq[dir_mask].astype(np.float64)

        f_tilde_i, f_tilde_o = _trt_post_collide_at(
            f_state, x, y, z, i, opp, wt_i, wt_o,
            cx_i, cy_i, cz_i, s_plus, s_minus,
        )

        # Bouzidi q-aware reflected population.
        f_opp_post = np.empty_like(f_tilde_i)

        # Branch 1: q >= 0.5 uses f_tilde_i(x_f) and f_tilde_opp(x_f).
        hi = q >= 0.5
        if hi.any():
            qh = q[hi]
            inv_2qh = 1.0 / (2.0 * qh)
            f_opp_post[hi] = (
                inv_2qh * f_tilde_i[hi]
                + (2.0 * qh - 1.0) * inv_2qh * f_tilde_o[hi]
            )

        # Branch 2: q < 0.5 uses f_tilde_i at the upstream fluid cell
        # (x_f - c_i). When that cell is out-of-domain or solid, fall
        # back to f_tilde_i at x_f -- same fallback the Bouzidi BB
        # implementation uses, so the force evaluation cannot disagree
        # with the underlying BB scheme.
        lo = ~hi
        if lo.any():
            x_lo, y_lo, z_lo = x[lo], y[lo], z[lo]
            q_lo = q[lo]
            xu = x_lo - cx_i
            yu = y_lo - cy_i
            zu = z_lo - cz_i
            in_dom = (
                (xu >= 0) & (xu < Nx)
                & (yu >= 0) & (yu < Ny)
                & (zu >= 0) & (zu < Nz)
            )
            valid = np.zeros(len(x_lo), dtype=bool)
            valid[in_dom] = ~body[xu[in_dom], yu[in_dom], zu[in_dom]]

            f_tilde_i_up = np.empty(len(x_lo), dtype=np.float64)
            if valid.any():
                fti_v, _ = _trt_post_collide_at(
                    f_state, xu[valid], yu[valid], zu[valid],
                    i, opp, wt_i, wt_o,
                    cx_i, cy_i, cz_i, s_plus, s_minus,
                )
                f_tilde_i_up[valid] = fti_v
            f_tilde_i_up[~valid] = f_tilde_i[lo][~valid]

            f_opp_post[lo] = (
                2.0 * q_lo * f_tilde_i[lo]
                + (1.0 - 2.0 * q_lo) * f_tilde_i_up
            )

        # MYSL link contribution: c_i * (f_tilde_i + f_opp_post_BB).
        contrib = (f_tilde_i + f_opp_post).sum()
        F[0] += cx_i * contrib
        F[1] += cy_i * contrib
        F[2] += cz_i * contrib

    return F


def _trt_post_collide_at(
    f: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    i: int, opp: int,
    wt_i: float, wt_o: float,
    cx_i: int, cy_i: int, cz_i: int,
    s_plus: float, s_minus: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f_tilde_i, f_tilde_opp) at (x, y, z) via TRT post-collision split.

    Mirrors the f_tilde reconstruction inside
    ``src.lbm_3d_bouzidi.apply_bouzidi_correction_trt`` so the MYSL
    force evaluation uses the same post-collision values the Bouzidi
    BB would use at the next step.
    """
    # Macroscopic at (x, y, z) using all 19 populations.
    rho = np.zeros(len(x), dtype=np.float64)
    mx_ = np.zeros(len(x), dtype=np.float64)
    my_ = np.zeros(len(x), dtype=np.float64)
    mz_ = np.zeros(len(x), dtype=np.float64)
    for j in range(19):
        fj = f[j, x, y, z].astype(np.float64)
        rho += fj
        mx_ += float(LATTICE_VELOCITIES_3D[j, 0]) * fj
        my_ += float(LATTICE_VELOCITIES_3D[j, 1]) * fj
        mz_ += float(LATTICE_VELOCITIES_3D[j, 2]) * fj
    inv_rho = 1.0 / np.maximum(rho, 1e-30)
    ux = mx_ * inv_rho
    uy = my_ * inv_rho
    uz = mz_ * inv_rho
    usq = ux * ux + uy * uy + uz * uz

    # Equilibria at (x, y, z) for directions i and opp.
    cu_i = cx_i * ux + cy_i * uy + cz_i * uz
    cu_o = -cu_i
    e_i = wt_i * rho * (1.0 + 3.0 * cu_i + 4.5 * cu_i * cu_i - 1.5 * usq)
    e_o = wt_o * rho * (1.0 + 3.0 * cu_o + 4.5 * cu_o * cu_o - 1.5 * usq)

    # TRT post-collision split.
    f_i = f[i, x, y, z].astype(np.float64)
    f_o = f[opp, x, y, z].astype(np.float64)
    fp = 0.5 * (f_i + f_o)
    fm = 0.5 * (f_i - f_o)
    ep = 0.5 * (e_i + e_o)
    em = 0.5 * (e_i - e_o)
    f_tilde_i = f_i - s_plus * (fp - ep) - s_minus * (fm - em)
    f_tilde_o = f_o - s_plus * (fp - ep) + s_minus * (fm - em)
    return f_tilde_i, f_tilde_o


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
    "momentum_exchange_force_3d_mysl",
    "drag_coefficient_3d",
]
