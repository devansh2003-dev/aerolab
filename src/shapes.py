"""Shape mask generators for LBM solid bodies.

A "shape mask" is a boolean ndarray of shape (Nx, Ny) where True marks cells
inside the obstacle. The mask is built once before the time loop and passed
into ``bounce_back(f, mask)`` every step.

Cell-centered convention: a cell is part of the obstacle if its center lies
inside the geometric shape. The effective wall is implicitly at the half-cell
boundary between solid and fluid (matches halfway bounce-back).
"""
import numpy as np
from matplotlib.path import Path


def cylinder_mask(Nx: int, Ny: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Boolean mask for a circular cylinder.

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Cylinder center in lattice units. Floats allowed (cylinder need not be
        on a grid point) -- cells are marked based on center-to-center distance.
    radius
        Cylinder radius in lattice units.

    Returns
    -------
    mask : ndarray of bool, shape (Nx, Ny)
        True where (x - cx)**2 + (y - cy)**2 <= radius**2.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    # indexing="ij" so X[i, j] = i and Y[i, j] = j -- matches our (Nx, Ny) layout.
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2


# ---------------------------------------------------------------------------
# Bouzidi interpolated bounce-back q-fields
# ---------------------------------------------------------------------------
# A "q-field" is a (Nx, Ny, 9) float64 array where q_field[x, y, i] gives the
# wall fraction for the lattice link starting at fluid cell (x, y) heading in
# direction i. The wall fraction is:
#
#     q = (distance from (x, y) to the analytic wall, along c_i) / |c_i|
#
# in [0, 1] for a Bouzidi-corrected wall link, or <= 0 (we use -1) for any link
# that is NOT a wall link (either fluid-to-fluid, solid-to-solid, or solid-to-fluid).
#
# The JIT step functions in src/lbm.py read this array AFTER streaming and apply
# Bouzidi's linear interpolation formula (Bouzidi, Firdaouss, Lallemand 2001) at
# each q > 0 link. Halfway bounce-back is q = 0.5 -- the formula degenerates to
# the simple swap at that value. A q-field of all -1 means "use halfway BB
# everywhere" and reproduces the pre-Bouzidi solver bit-for-bit.

_LATTICE_C = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
], dtype=np.float64)


def no_bouzidi_q_field(Nx: int, Ny: int) -> np.ndarray:
    """Return a q-field that selects halfway bounce-back at every link.

    Use when a shape doesn't yet have an analytic q-field generator (square,
    ellipse, NACA in the current iteration), so the solver falls back to its
    original halfway-BB behavior with zero performance penalty.
    """
    return np.full((Nx, Ny, 9), -1.0, dtype=np.float64)


def cylinder_q_field(
    Nx: int, Ny: int, cx: float, cy: float, radius: float,
) -> np.ndarray:
    """Bouzidi q-field for a circular cylinder.

    For every fluid cell adjacent to the cylinder in lattice direction i, computes
    the exact fraction q ∈ (0, 1] such that (x + q·c_ix, y + q·c_iy) lies on the
    analytic cylinder boundary. Solves the quadratic
        |x_f + q·c_i - c|^2 = r^2
    and picks the smaller positive root (the first wall crossing from x_f toward
    the solid). All other links get q = -1 (no Bouzidi correction, halfway BB).

    Parameters
    ----------
    Nx, Ny, cx, cy, radius
        Same as ``cylinder_mask``.

    Returns
    -------
    q_field : ndarray of shape (Nx, Ny, 9), float64
    """
    mask = cylinder_mask(Nx, Ny, cx, cy, radius)
    q = np.full((Nx, Ny, 9), -1.0, dtype=np.float64)

    # Build the per-cell (dx, dy) once.
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = X.astype(np.float64) - cx
    dy = Y.astype(np.float64) - cy
    r2 = radius * radius

    for i in range(1, 9):                # skip i=0 (rest direction, no link)
        cxi, cyi = _LATTICE_C[i]
        # Find fluid cells whose direction-i neighbor is solid AND in-bounds.
        xn = np.arange(Nx) + int(cxi)
        yn = np.arange(Ny) + int(cyi)
        in_x = (xn >= 0) & (xn < Nx)
        in_y = (yn >= 0) & (yn < Ny)
        in_xy = in_x[:, None] & in_y[None, :]
        # Build neighbor-is-solid mask. Out-of-bounds neighbors aren't wall links
        # in this sense -- those are domain-boundary links, handled separately.
        neighbor_solid = np.zeros((Nx, Ny), dtype=bool)
        xs = xn[in_x]
        ys = yn[in_y]
        neighbor_solid[np.ix_(in_x, in_y)] = mask[np.ix_(xs, ys)]
        wall_links = (~mask) & neighbor_solid & in_xy
        if not wall_links.any():
            continue

        # Quadratic in q: |c_i|^2 * q^2 + 2(dx·c_ix + dy·c_iy)·q + (|d|^2 - r^2) = 0.
        # |c_i|^2 is 1 for cardinals (i in {1,2,3,4}) and 2 for diagonals (i in {5..8}).
        A = float(cxi * cxi + cyi * cyi)
        B = 2.0 * (dx * cxi + dy * cyi)
        C = dx * dx + dy * dy - r2
        disc = B * B - 4.0 * A * C
        # disc < 0 should not occur on a wall link (fluid cell outside, neighbor
        # inside means the segment must cross the boundary). Guard anyway.
        disc_pos = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc_pos)
        # Smaller root = first crossing from x_f toward the solid.
        q_small = (-B - sqrt_disc) / (2.0 * A)
        # Clamp the tiny FP-noise excursions outside [0, 1] back into the valid range.
        q_small = np.clip(q_small, 0.0, 1.0)
        q[..., i] = np.where(wall_links, q_small, q[..., i])

    return q


def ellipse_q_field(
    Nx: int, Ny: int, cx: float, cy: float, a: float, b: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Bouzidi q-field for an ellipse (optionally rotated by aoa_deg).

    The ellipse boundary in body frame is (Xr/a)^2 + (Yr/b)^2 = 1. After
    rotating both the cell position and the lattice direction into the body
    frame, the wall-crossing parameter q is the smaller positive root of a
    quadratic identical in structure to the cylinder case:
        A q^2 + B q + C = 0
    with
        A = (X_dir/a)^2 + (Y_dir/b)^2
        B = 2 * (Xr_0 * X_dir / a^2 + Yr_0 * Y_dir / b^2)
        C = (Xr_0/a)^2 + (Yr_0/b)^2 - 1
    where (Xr_0, Yr_0) are the body-frame coords of the fluid cell and
    (X_dir, Y_dir) is the body-frame lattice direction.

    The general analytic ellipse formula naturally reduces to a circle
    when a == b, so the a==b, aoa_deg==0 special case is handled by the
    same code path -- no separate fall-through branch exists.
    """
    mask = ellipse_mask(Nx, Ny, cx, cy, a, b, aoa_deg)
    q = np.full((Nx, Ny, 9), -1.0, dtype=np.float64)

    ang = np.deg2rad(aoa_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    x = np.arange(Nx, dtype=np.float64)
    y = np.arange(Ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = X - cx
    dy = Y - cy
    # Body-frame cell coords (same convention as ellipse_mask).
    Xr0 = cos_a * dx - sin_a * dy
    Yr0 = sin_a * dx + cos_a * dy
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    C = Xr0 * Xr0 * inv_a2 + Yr0 * Yr0 * inv_b2 - 1.0  # = ((Xr/a)^2 + (Yr/b)^2) - 1

    for i in range(1, 9):
        cxi, cyi = _LATTICE_C[i]
        # Body-frame lattice direction.
        X_dir = cos_a * cxi - sin_a * cyi
        Y_dir = sin_a * cxi + cos_a * cyi

        # Wall-link mask: fluid -> in-bounds solid neighbor.
        xn = np.arange(Nx) + int(cxi)
        yn = np.arange(Ny) + int(cyi)
        in_x = (xn >= 0) & (xn < Nx)
        in_y = (yn >= 0) & (yn < Ny)
        in_xy = in_x[:, None] & in_y[None, :]
        neighbor_solid = np.zeros((Nx, Ny), dtype=bool)
        xs = xn[in_x]
        ys = yn[in_y]
        neighbor_solid[np.ix_(in_x, in_y)] = mask[np.ix_(xs, ys)]
        wall_links = (~mask) & neighbor_solid & in_xy
        if not wall_links.any():
            continue

        A = X_dir * X_dir * inv_a2 + Y_dir * Y_dir * inv_b2
        B = 2.0 * (Xr0 * X_dir * inv_a2 + Yr0 * Y_dir * inv_b2)
        disc = B * B - 4.0 * A * C
        disc_pos = np.maximum(disc, 0.0)
        sqrt_disc = np.sqrt(disc_pos)
        q_small = (-B - sqrt_disc) / (2.0 * A)
        q_small = np.clip(q_small, 0.0, 1.0)
        q[..., i] = np.where(wall_links, q_small, q[..., i])

    return q


def ellipse_mask(
    Nx: int, Ny: int, cx: float, cy: float, a: float, b: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Boolean mask for an ellipse, optionally rotated.

    Condition: ``((x' / a)**2 + (y' / b)**2 <= 1`` where (x', y') is the world
    point translated to the ellipse center and rotated INTO the body frame.
    The cylinder is the special case ``a == b == radius`` -- useful for
    streamlined-vs-bluff comparison studies.

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Ellipse center in lattice units.
    a
        Semi-axis along the body's local x (half-width before rotation).
    b
        Semi-axis along the body's local y (half-height before rotation).
    aoa_deg
        Rotation about ``(cx, cy)`` in degrees. Positive angle tilts the
        +x body axis upward in grid coords, matching the NACA convention.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    if aoa_deg == 0.0:
        return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
    ang = np.deg2rad(aoa_deg)
    c, s = np.cos(ang), np.sin(ang)
    # Inverse of the CW rotation used by naca4_airfoil_mask -- maps world coords
    # back into body-frame coords where the ellipse is axis-aligned.
    Xr = c * (X - cx) - s * (Y - cy)
    Yr = s * (X - cx) + c * (Y - cy)
    return (Xr / a) ** 2 + (Yr / b) ** 2 <= 1.0


def square_q_field(
    Nx: int, Ny: int, cx: float, cy: float, side: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Bouzidi q-field for a square (optionally rotated by aoa_deg).

    The square boundary in body frame is the union of four half-line segments:
    Xr = ±half OR Yr = ±half. For each wall link, find the smallest positive
    q ∈ (0, 1] that lands the body-frame point on one of those segments
    AND inside the orthogonal half-extent (so we don't pick the wall-line's
    extension beyond the corner).

    For each of the four candidate intersections we solve a linear equation:
        Xr_0 + q X_dir = ±half     (left/right faces, requires |Yr_0 + q Y_dir| <= half)
        Yr_0 + q Y_dir = ±half     (bottom/top faces, requires |Xr_0 + q X_dir| <= half)
    and take the smallest positive q whose orthogonal coord is in [-half, half].
    """
    mask = square_mask(Nx, Ny, cx, cy, side, aoa_deg)
    q = np.full((Nx, Ny, 9), -1.0, dtype=np.float64)

    ang = np.deg2rad(aoa_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    half = side / 2.0

    x = np.arange(Nx, dtype=np.float64)
    y = np.arange(Ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = X - cx
    dy = Y - cy
    Xr0 = cos_a * dx - sin_a * dy
    Yr0 = sin_a * dx + cos_a * dy

    # Big sentinel for "no valid crossing on this face" -- larger than 1 so
    # the np.minimum.reduce picks the smallest real q in (0, 1] if any face
    # produced one.
    NO_HIT = 10.0

    for i in range(1, 9):
        cxi, cyi = _LATTICE_C[i]
        X_dir = cos_a * cxi - sin_a * cyi
        Y_dir = sin_a * cxi + cos_a * cyi

        xn = np.arange(Nx) + int(cxi)
        yn = np.arange(Ny) + int(cyi)
        in_x = (xn >= 0) & (xn < Nx)
        in_y = (yn >= 0) & (yn < Ny)
        in_xy = in_x[:, None] & in_y[None, :]
        neighbor_solid = np.zeros((Nx, Ny), dtype=bool)
        xs = xn[in_x]
        ys = yn[in_y]
        neighbor_solid[np.ix_(in_x, in_y)] = mask[np.ix_(xs, ys)]
        wall_links = (~mask) & neighbor_solid & in_xy
        if not wall_links.any():
            continue

        # Compute candidate q for each of the four faces. If X_dir or Y_dir
        # is zero, the corresponding equation has no solution (skip that face).
        q_candidates = np.full((Nx, Ny, 4), NO_HIT, dtype=np.float64)
        if abs(X_dir) > 1e-15:
            inv_Xdir = 1.0 / X_dir
            for k, sgn in enumerate([+1.0, -1.0]):
                q_k = (sgn * half - Xr0) * inv_Xdir
                Yr_at_q = Yr0 + q_k * Y_dir
                valid = (q_k > 0.0) & (q_k <= 1.0) & (np.abs(Yr_at_q) <= half + 1e-12)
                q_candidates[..., k] = np.where(valid, q_k, NO_HIT)
        if abs(Y_dir) > 1e-15:
            inv_Ydir = 1.0 / Y_dir
            for k, sgn in enumerate([+1.0, -1.0]):
                q_k = (sgn * half - Yr0) * inv_Ydir
                Xr_at_q = Xr0 + q_k * X_dir
                valid = (q_k > 0.0) & (q_k <= 1.0) & (np.abs(Xr_at_q) <= half + 1e-12)
                q_candidates[..., 2 + k] = np.where(valid, q_k, NO_HIT)
        q_min = q_candidates.min(axis=-1)
        q_min = np.clip(q_min, 0.0, 1.0)
        # Only write valid q for wall links that actually had a hit.
        valid_link = wall_links & (q_min < NO_HIT - 1.0)
        q[..., i] = np.where(valid_link, q_min, q[..., i])

    return q


def square_mask(
    Nx: int, Ny: int, cx: float, cy: float, side: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Boolean mask for a square, optionally rotated.

    Two conditions ANDed: ``|x'| <= side/2 AND |y'| <= side/2`` where the
    primed coordinates are the world point translated to the center and
    rotated INTO the body frame. The wall is implicit at the half-cell
    boundary (consistent with halfway bounce-back convention).

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Square center in lattice units.
    side
        Edge length in lattice units.
    aoa_deg
        Rotation about ``(cx, cy)`` in degrees. Positive angle tilts the
        +x body axis upward in grid coords, matching the NACA convention.
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    half = side / 2.0
    if aoa_deg == 0.0:
        return (np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half)
    ang = np.deg2rad(aoa_deg)
    c, s = np.cos(ang), np.sin(ang)
    Xr = c * (X - cx) - s * (Y - cy)
    Yr = s * (X - cx) + c * (Y - cy)
    return (np.abs(Xr) <= half) & (np.abs(Yr) <= half)


# ---------------------------------------------------------------------------
# NACA 4-digit airfoil mask
# ---------------------------------------------------------------------------

def _naca4_decode(naca_code: str) -> tuple:
    """Decode a 4-digit NACA code into (m, p, t) as fractions of chord.

    Example: ``"4412"`` -> ``(0.04, 0.4, 0.12)`` (4% camber at 40% chord, 12% thick).
    """
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(f"naca_code must be a 4-digit string, got {naca_code!r}")
    m = int(naca_code[0]) / 100.0    # max camber as fraction
    p = int(naca_code[1]) / 10.0     # location of max camber as fraction
    t = int(naca_code[2:]) / 100.0   # max thickness as fraction
    return m, p, t


def _naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """Closed-TE NACA 4-digit thickness distribution (half-thickness y_t at chord x).

    Uses -0.1036 as the 5th coefficient (vs the original -0.1015) so the
    trailing edge closes cleanly to zero -- preferred for masking.
    """
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )


def _naca4_camber(x: np.ndarray, m: float, p: float) -> tuple:
    """NACA 4-digit camber line y_c(x) and slope dy_c/dx, piecewise at x = p."""
    if m == 0.0:
        # Symmetric airfoil -- no camber, no slope. Skip the divisions that
        # would otherwise hit p=0 zero-division for codes like "0012".
        return np.zeros_like(x), np.zeros_like(x)
    y_c = np.where(
        x < p,
        m / p ** 2 * (2 * p * x - x ** 2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2),
    )
    dyc_dx = np.where(
        x < p,
        2 * m / p ** 2 * (p - x),
        2 * m / (1 - p) ** 2 * (p - x),
    )
    return y_c, dyc_dx


def naca4_outline_xy(naca_code: str, n_pts: int = 200) -> tuple:
    """High-resolution NACA 4-digit boundary in airfoil-local coords.

    Returns ``(poly_x, poly_y)`` with the LE at the origin and the chord
    aligned with the +x axis. Same polygon construction the mask function
    uses, so an overlay drawn from this matches the mask exactly. Caller is
    responsible for the AoA rotation + grid placement.
    """
    m, p, t = _naca4_decode(naca_code)
    x_a = np.linspace(0.0, 1.0, n_pts)
    y_t = _naca4_thickness(x_a, t)
    y_c, dyc_dx = _naca4_camber(x_a, m, p)
    theta = np.arctan(dyc_dx)
    x_upper = x_a - y_t * np.sin(theta)
    y_upper = y_c + y_t * np.cos(theta)
    x_lower = x_a + y_t * np.sin(theta)
    y_lower = y_c - y_t * np.cos(theta)
    poly_x = np.concatenate([x_upper, x_lower[::-1]])
    poly_y = np.concatenate([y_upper, y_lower[::-1]])
    return poly_x, poly_y


def naca4_q_field(
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    chord: float,
    naca_code: str,
    aoa_deg: float,
    n_pts: int = 200,
) -> np.ndarray:
    """Bouzidi q-field for a NACA 4-digit airfoil.

    No closed-form distance to the airfoil boundary exists, so we use the
    same high-resolution polygon that the mask is built from (via
    naca4_outline_xy) and intersect each lattice link with every edge of
    that polygon. q is the smallest valid intersection parameter t in (0, 1]
    where the lattice link from the fluid cell crosses an edge segment.

    Cost: ~n_pts*2 edges * (n_wall_links per direction) * 8 directions.
    For a chord=44 airfoil on Standard 320x100 (~500 wall links per direction),
    that's ~3M edge tests -- runs in well under a second.
    """
    mask = naca4_airfoil_mask(Nx, Ny, cx, cy, chord, naca_code, aoa_deg)
    q = np.full((Nx, Ny, 9), -1.0, dtype=np.float64)

    # Build the polygon in world coords -- same transform as naca4_airfoil_mask.
    poly_x_local, poly_y_local = naca4_outline_xy(naca_code, n_pts=n_pts)
    ang = np.deg2rad(aoa_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    poly_x = cx + chord * (poly_x_local * cos_a + poly_y_local * sin_a)
    poly_y = cy + chord * (-poly_x_local * sin_a + poly_y_local * cos_a)

    # Edges: kth edge goes from (poly_x[k], poly_y[k]) to (poly_x[k+1], poly_y[k+1]),
    # with wrap-around closing the polygon.
    E1x = poly_x
    E1y = poly_y
    Ex = np.roll(poly_x, -1) - poly_x
    Ey = np.roll(poly_y, -1) - poly_y

    for i in range(1, 9):
        cxi, cyi = _LATTICE_C[i]

        # Wall-link mask (same construction as the cylinder/ellipse paths).
        xn = np.arange(Nx) + int(cxi)
        yn = np.arange(Ny) + int(cyi)
        in_x = (xn >= 0) & (xn < Nx)
        in_y = (yn >= 0) & (yn < Ny)
        in_xy = in_x[:, None] & in_y[None, :]
        neighbor_solid = np.zeros((Nx, Ny), dtype=bool)
        xs = xn[in_x]
        ys = yn[in_y]
        neighbor_solid[np.ix_(in_x, in_y)] = mask[np.ix_(xs, ys)]
        wall_links = (~mask) & neighbor_solid & in_xy
        if not wall_links.any():
            continue

        # Precompute direction-dependent denominator across all edges.
        # System for the intersection of line (x_f, y_f) + q*(cxi, cyi) with
        # edge E1 + s*(Ex, Ey):
        #   det = Ex*cyi - cxi*Ey  (independent of x_f)
        det = Ex * cyi - cxi * Ey
        valid_det = np.abs(det) > 1e-12
        # Safe inverse: 1/det where det is non-zero, 1.0 elsewhere; we mask
        # the result downstream so the dummy value doesn't pollute.
        inv_det = np.where(valid_det, 1.0 / np.where(valid_det, det, 1.0), 0.0)

        # Loop over wall-link cells (sparse). 200-2000 per direction in practice.
        wall_x, wall_y = np.where(wall_links)
        for x, y in zip(wall_x, wall_y, strict=True):
            dx_e = x - E1x        # shape (n_edges,)
            dy_e = y - E1y
            # q (link parameter): see derivation comment in the function header.
            q_vec = (Ex * (-dy_e) - Ey * (-dx_e)) * inv_det
            # s (edge parameter): how far along the polygon edge the intersection is.
            s_vec = (cxi * (-dy_e) - cyi * (-dx_e)) * inv_det
            # Valid: det non-zero AND q in (0, 1] AND s in [0, 1].
            valid = valid_det & (q_vec > 1e-12) & (q_vec <= 1.0 + 1e-12) \
                              & (s_vec >= -1e-12) & (s_vec <= 1.0 + 1e-12)
            if not valid.any():
                # No edge intersection found -- mask must be inconsistent with
                # the polygon (e.g., very thin features). Fall back to halfway BB.
                continue
            q_min = float(q_vec[valid].min())
            q[x, y, i] = min(max(q_min, 0.0), 1.0)

    return q


def naca4_airfoil_mask(
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    chord: float,
    naca_code: str,
    aoa_deg: float,
) -> np.ndarray:
    """Boolean mask for a NACA 4-digit airfoil at a given angle of attack.

    The leading edge is anchored at ``(cx, cy)``. Positive ``aoa_deg`` rotates
    the leading edge upward (lift-positive convention) so the trailing edge
    ends up below the chord line in the grid frame.

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Leading-edge position in lattice units.
    chord
        Chord length in lattice units.
    naca_code
        4-digit NACA designation, e.g. ``"4412"`` or ``"0012"``.
    aoa_deg
        Angle of attack in degrees (positive = leading edge up).

    Returns
    -------
    mask : ndarray of bool, shape (Nx, Ny)
    """
    poly_x, poly_y = naca4_outline_xy(naca_code, n_pts=200)

    # Transform from airfoil-local (LE at origin, chord along +x) to grid coords.
    # Positive AoA rotates LE up, which puts TE below the chord direction in
    # the global (lattice) frame -- hence the negative-sine on the y component.
    aoa_rad = np.deg2rad(aoa_deg)
    cos_a = np.cos(aoa_rad)
    sin_a = np.sin(aoa_rad)
    grid_x = cx + chord * (poly_x * cos_a + poly_y * sin_a)
    grid_y = cy + chord * (-poly_x * sin_a + poly_y * cos_a)
    vertices = np.column_stack([grid_x, grid_y])

    path = Path(vertices)
    xs = np.arange(Nx)
    ys = np.arange(Ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    return path.contains_points(points).reshape(Nx, Ny)
