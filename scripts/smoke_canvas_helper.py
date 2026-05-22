"""Unit test for canvas_image_to_polygon.

Synthesises a "drawn circle" by rasterising a circle outline in RGBA
into a (320, 160, 4) array, then feeds it through canvas_image_to_polygon
and asserts the result is a usable polygon. Catches:
  * Module import path bugs (scipy.ndimage.binary_dilation/fill_holes)
  * Threshold + dilation + fill_holes pipeline regressions
  * The fall-through into extract_silhouette_from_image's contour finder
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.custom_shape import canvas_image_to_polygon


def synth_drawn_circle(h: int = 160, w: int = 320, radius: int = 35) -> np.ndarray:
    """Build an RGBA image with a stroked-circle drawing (open along the
    outline -- just like st_canvas would produce from a freehand circle)."""
    cx, cy = w // 2, h // 2
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - cx, yy - cy)
    # Annular stroke: pixels within stroke_width/2 of the circle.
    stroke_mask = (dist > radius - 5) & (dist < radius + 5)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[stroke_mask, :3] = 255  # white stroke
    rgba[stroke_mask, 3] = 255   # opaque
    return rgba


def synth_open_line(h: int = 160, w: int = 320) -> np.ndarray:
    """Draw a straight line -- does NOT enclose a region. Should reject."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    yy = h // 2
    rgba[yy - 3 : yy + 3, 50:270, :3] = 255
    rgba[yy - 3 : yy + 3, 50:270, 3] = 255
    return rgba


def main():
    print("[1/2] Drawn closed circle -> polygon...")
    rgba = synth_drawn_circle()
    result = canvas_image_to_polygon(rgba)
    assert result.polygon_xy.ndim == 2 and result.polygon_xy.shape[1] == 2, (
        f"Expected (N, 2) polygon; got shape {result.polygon_xy.shape}"
    )
    assert result.polygon_xy.shape[0] >= 6, (
        f"Polygon too sparse: {result.polygon_xy.shape[0]} verts"
    )
    print(f"  PASS: {result.polygon_xy.shape[0]} vertices, image {result.image_w} x {result.image_h}")

    print("[2/2] Open line (should reject)...")
    rgba = synth_open_line()
    try:
        canvas_image_to_polygon(rgba)
    except ValueError as e:
        print(f"  PASS: rejected with message: {e!s}")
    else:
        raise AssertionError("Open line should have raised ValueError")

    print()
    print("Canvas helper end-to-end validation passed.")


if __name__ == "__main__":
    main()
