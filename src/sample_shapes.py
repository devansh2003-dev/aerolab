"""Bundled sample silhouettes for the "Try a sample" button.

Three real-world silhouettes called out as the Phase 2 W5 validation gate
in the README: car profile, fish, building cross-section. Each is a hand-
tuned polygon in image-pixel coords (origin top-left, y-down) so that
``polygon_to_lbm_mask`` and ``polygon_outline_xy`` can be reused unchanged.

The polygons are deliberately simple (10-40 vertices) so they look clean
after Douglas-Peucker simplification AND rasterize crisply at the custom
shape's target_extent_cells. Each one has been validated to run without
NaN at Re=200 on both Standard and Detailed presets.
"""
from __future__ import annotations

import numpy as np


def fish_polygon() -> np.ndarray:
    """A fish silhouette: streamlined body with triangular tail fin.

    ~200 px wide x 100 px tall in image coords. The mouth is on the LEFT
    (low x) so the natural orientation has the fish facing into the
    inflow (which comes from x=0 in lattice coords). polygon_to_lbm_mask
    no longer flips x, so this orientation is preserved as-is in the
    simulation -- same convention applies to user uploads.
    """
    return np.array([
        # Mouth tip (LEFT side of body, in image coords)
        (0.0, 50.0),
        # Top of head curving up to dorsal line
        (14.0, 32.0),
        (32.0, 22.0),
        (54.0, 16.0),
        (82.0, 14.0),
        (110.0, 16.0),
        # Dorsal fin small bump
        (130.0, 8.0),
        (140.0, 18.0),
        # Continue along top toward tail
        (152.0, 22.0),
        (168.0, 28.0),
        # Tail fin: triangular notch (RIGHT side of body)
        (184.0, 14.0),
        (188.0, 30.0),
        (176.0, 50.0),
        (188.0, 70.0),
        (184.0, 86.0),
        (168.0, 72.0),
        # Back along underside
        (152.0, 78.0),
        (130.0, 84.0),
        (102.0, 86.0),
        (74.0, 84.0),
        (48.0, 80.0),
        (28.0, 72.0),
        (12.0, 64.0),
    ], dtype=np.float64)


def car_profile_polygon() -> np.ndarray:
    """A car silhouette in side view: hood, cabin, trunk.

    ~300 px wide x 100 px tall. Wheels omitted because we want a closed
    body for clean LBM rasterization; the bottom is a single flat line.
    Hood is on the LEFT (low x) so it faces into the inflow -- the natural
    drag-test orientation. Matches the upload convention: orient your
    source image with the front of your shape on the left.
    """
    return np.array([
        # Front bumper bottom-left corner
        (0.0, 88.0),
        # Front bumper top
        (0.0, 70.0),
        # Hood slope going back/up
        (30.0, 56.0),
        (80.0, 48.0),
        # Windshield slope up to roof
        (105.0, 22.0),
        # Roof top-front
        (140.0, 18.0),
        # Roof top-back
        (200.0, 18.0),
        # Rear window slope down to trunk
        (228.0, 32.0),
        # Trunk top
        (255.0, 38.0),
        # Rear bumper top
        (280.0, 46.0),
        # Rear bumper top-right
        (286.0, 60.0),
        # Rear bumper bottom-right
        (286.0, 88.0),
        # Underbody (single straight line back to start)
    ], dtype=np.float64)


def building_cross_section_polygon() -> np.ndarray:
    """A tall building silhouette, lying on its side so wind blows
    across it (the classic "wind around a tower" 2D benchmark).

    Original drawing (upright, ~100 px wide x 300 px tall) is rotated
    90 deg CW into the returned polygon so the long axis sits
    horizontally: ~300 px wide x ~100 px tall. The foundation faces the
    inflow on the LEFT; the spire points away on the right. With the
    polygon scaled to 60 cells (Standard preset extent), this gives a
    ~60 x 19 lattice body = ~24 % vertical blockage, in line with the
    other bundled samples (fish, car) and well below the 50 % threshold
    where the channel walls start to dominate the wake.

    Without this rotation, the upright building runs ~75 % vertical
    blockage and reports an absurdly high Cd (~30+) -- the simulation
    is correct but unphysical for the intended "look at the wake"
    use case.
    """
    # Upright polygon, image coords (origin top-left, y down).
    # Spire tip is at low y (top of image); foundation is at high y.
    upright = np.array([
        (50.0, 6.0),    # Spire / antenna tip
        (44.0, 24.0),
        (24.0, 28.0),
        (24.0, 38.0),
        (8.0, 42.0),
        (8.0, 110.0),
        (18.0, 115.0),
        (18.0, 290.0),
        (4.0, 296.0),   # Base bottom-left
        (96.0, 296.0),  # Base bottom-right
        (82.0, 290.0),
        (82.0, 115.0),
        (92.0, 110.0),
        (92.0, 42.0),
        (76.0, 38.0),
        (76.0, 28.0),
        (56.0, 24.0),
    ], dtype=np.float64)
    # Rotate 90 deg CW: (x, y) -> (y, x_max - x). Result: foundation
    # ends up on the LEFT (low x, facing inflow), spire on the RIGHT.
    x_max = upright[:, 0].max()
    rotated = np.empty_like(upright)
    rotated[:, 0] = upright[:, 1]
    rotated[:, 1] = x_max - upright[:, 0]
    return rotated


SAMPLE_SHAPES = {
    "Fish": fish_polygon,
    "Car profile": car_profile_polygon,
    "Building cross-section": building_cross_section_polygon,
}
