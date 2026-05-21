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

    ~200 px wide x 100 px tall in image coords. The mouth is on the right
    (positive x) so the natural orientation has the fish facing into the
    flow (which comes from the left). aoa_deg=180 would flip head-on-tail.
    """
    return np.array([
        # Mouth tip (right side of body, in image coords)
        (192.0, 50.0),
        # Top of head curving up to dorsal line
        (178.0, 32.0),
        (160.0, 22.0),
        (138.0, 16.0),
        (110.0, 14.0),
        (82.0, 16.0),
        # Dorsal fin small bump
        (62.0, 8.0),
        (52.0, 18.0),
        # Continue along top toward tail
        (40.0, 22.0),
        (24.0, 28.0),
        # Tail fin: triangular notch
        (8.0, 14.0),
        (4.0, 30.0),
        (16.0, 50.0),
        (4.0, 70.0),
        (8.0, 86.0),
        (24.0, 72.0),
        # Back along underside
        (40.0, 78.0),
        (62.0, 84.0),
        (90.0, 86.0),
        (118.0, 84.0),
        (144.0, 80.0),
        (164.0, 72.0),
        (180.0, 64.0),
    ], dtype=np.float64)


def car_profile_polygon() -> np.ndarray:
    """A car silhouette in side view: hood, cabin, trunk.

    ~300 px wide x 100 px tall. Wheels omitted because we want a closed
    body for clean LBM rasterization; the bottom is a single flat line.
    Hood is on the right (positive x) so it faces into the leftward flow
    -- the natural drag-test orientation.
    """
    return np.array([
        # Front bumper bottom-right corner
        (290.0, 88.0),
        # Front bumper top
        (290.0, 70.0),
        # Hood slope going back/up
        (260.0, 56.0),
        (210.0, 48.0),
        # Windshield slope up to roof
        (185.0, 22.0),
        # Roof top-front
        (150.0, 18.0),
        # Roof top-back
        (90.0, 18.0),
        # Rear window slope down to trunk
        (62.0, 32.0),
        # Trunk top
        (35.0, 38.0),
        # Rear bumper top
        (10.0, 46.0),
        # Rear bumper top-left
        (4.0, 60.0),
        # Rear bumper bottom-left
        (4.0, 88.0),
        # Underbody (single straight line back to start)
    ], dtype=np.float64)


def building_cross_section_polygon() -> np.ndarray:
    """A tall building silhouette: stepped facade + spire on top.

    ~100 px wide x 300 px tall. Will rotate 90 deg by default in the app
    so the flow goes ACROSS the building rather than blowing it from
    above (which would be a structural-wind-load test, valid but not
    the typical 2D CFD demo orientation). Mirrors the classic
    'wind around a square tower' undergrad benchmark.
    """
    return np.array([
        # Spire / antenna at top
        (50.0, 6.0),
        (44.0, 24.0),
        # Roof corners with step
        (24.0, 28.0),
        (24.0, 38.0),
        (8.0, 42.0),
        # Down the left face with one step (setback)
        (8.0, 110.0),
        (18.0, 115.0),
        (18.0, 290.0),
        # Base bottom-left
        (4.0, 296.0),
        # Base bottom-right
        (96.0, 296.0),
        # Up the right face with mirrored step
        (82.0, 290.0),
        (82.0, 115.0),
        (92.0, 110.0),
        (92.0, 42.0),
        (76.0, 38.0),
        (76.0, 28.0),
        (56.0, 24.0),
    ], dtype=np.float64)


SAMPLE_SHAPES = {
    "Fish": fish_polygon,
    "Car profile": car_profile_polygon,
    "Building cross-section": building_cross_section_polygon,
}
