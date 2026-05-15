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


def ellipse_mask(Nx: int, Ny: int, cx: float, cy: float, a: float, b: float) -> np.ndarray:
    """Boolean mask for an axis-aligned ellipse.

    Condition: ``((x - cx) / a)**2 + ((y - cy) / b)**2 <= 1``. The cylinder is
    the special case ``a == b == radius`` -- useful for streamlined-vs-bluff
    comparison studies.

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Ellipse center in lattice units.
    a
        Semi-axis along x (half-width).
    b
        Semi-axis along y (half-height).

    Returns
    -------
    mask : ndarray of bool, shape (Nx, Ny)
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0


def square_mask(Nx: int, Ny: int, cx: float, cy: float, side: float) -> np.ndarray:
    """Boolean mask for an axis-aligned square.

    Two conditions ANDed: ``|x - cx| <= side/2 AND |y - cy| <= side/2``. The
    wall is implicit at the half-cell boundary (consistent with halfway
    bounce-back convention used elsewhere). For a moderate-Re wake this gives
    the classic bluff-body separation pattern -- much wider wake than a
    cylinder of equivalent frontal area.

    Parameters
    ----------
    Nx, Ny
        Grid dimensions.
    cx, cy
        Square center in lattice units.
    side
        Edge length in lattice units (frontal area = side per unit span).

    Returns
    -------
    mask : ndarray of bool, shape (Nx, Ny)
    """
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    half = side / 2.0
    return (np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half)


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
    m, p, t = _naca4_decode(naca_code)

    # Sample airfoil-local boundary at high resolution (200 points per surface)
    # so the polygon faithfully captures the rounded LE and thin TE.
    n_pts = 200
    x_a = np.linspace(0.0, 1.0, n_pts)
    y_t = _naca4_thickness(x_a, t)
    y_c, dyc_dx = _naca4_camber(x_a, m, p)
    theta = np.arctan(dyc_dx)

    # Surface points with thickness perpendicular to camber line.
    x_upper = x_a - y_t * np.sin(theta)
    y_upper = y_c + y_t * np.cos(theta)
    x_lower = x_a + y_t * np.sin(theta)
    y_lower = y_c - y_t * np.cos(theta)

    # Closed polygon: upper LE -> TE, then lower TE -> LE.
    poly_x = np.concatenate([x_upper, x_lower[::-1]])
    poly_y = np.concatenate([y_upper, y_lower[::-1]])

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
