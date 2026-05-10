"""Shape mask generators for LBM solid bodies.

A "shape mask" is a boolean ndarray of shape (Nx, Ny) where True marks cells
inside the obstacle. The mask is built once before the time loop and passed
into ``bounce_back(f, mask)`` every step.

Cell-centered convention: a cell is part of the obstacle if its center lies
inside the geometric shape. The effective wall is implicitly at the half-cell
boundary between solid and fluid (matches halfway bounce-back).
"""
import numpy as np


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
