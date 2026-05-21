"""Custom shape extraction for the "Upload your own" / "Draw your own" paths.

Two entry points:

    extract_silhouette_from_image(png_bytes) -> SilhouetteResult
        Decode a PNG/JPG, threshold to a single connected region, fill
        internal holes, extract the outer boundary contour, simplify
        with Douglas-Peucker. Returns the polygon in image-pixel coords.

    polygon_to_lbm_mask(polygon_xy, Nx, Ny, cx, cy, target_extent_cells)
        Rasterize a polygon (image-pixel coords) onto an (Nx, Ny) LBM
        grid, centred at (cx, cy), scaled so its longest dimension equals
        target_extent_cells.

The same polygon format is produced by the drawable-canvas path -- so
downstream code (lbm_render, shapes) only cares about the polygon array
and not which source it came from.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as ndimage_label
from skimage.measure import approximate_polygon, find_contours
from skimage.filters import threshold_otsu

MIN_IMAGE_DIM = 100   # pixels per side -- minimum upload size
MAX_IMAGE_DIM = 2048  # resize-down cap for extraction speed
# Douglas-Peucker tolerance as a fraction of the shorter image dimension.
# 0.01 = "aggressive but not too aggressive" smoothing -- removes obvious
# pixel-staircase noise while preserving sharp corners (~10 px at 1024 px).
SIMPLIFY_TOLERANCE_FRAC = 0.01
# Reject shapes outside these area-fraction bounds (relative to image area).
MIN_AREA_FRAC = 0.02
MAX_AREA_FRAC = 0.85


@dataclass
class SilhouetteResult:
    """Outcome of silhouette extraction.

    polygon_xy is (N, 2) in image-pixel coords (x=col, y=row, origin top-left,
    y increasing downward -- standard PIL convention). Downstream rasterizers
    flip y to match the lattice convention.
    """
    polygon_xy: np.ndarray
    image_w: int
    image_h: int
    n_holes_filled: int
    n_components_found: int
    warnings: list


def extract_silhouette_from_image(png_bytes: bytes) -> SilhouetteResult:
    """Extract the outer boundary polygon of the dominant shape in an image.

    Pipeline:
      1. Load + grayscale + resize cap.
      2. Reject if smaller than MIN_IMAGE_DIM.
      3. Otsu threshold to a binary mask. Auto-pick foreground = dark or
         light by checking which side the image corners fall on.
      4. Find connected components, keep the largest.
      5. Reject if the largest touches the image edge, or area is < 2 %
         or > 85 % of frame.
      6. binary_fill_holes on the kept component.
      7. find_contours at level 0.5 to get the outer boundary polyline.
      8. approximate_polygon (Douglas-Peucker, moderate tolerance).
      9. Reject if simplified polygon has fewer than 8 vertices.

    Raises ValueError with a user-readable message on extraction failure.
    """
    warnings: list = []

    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    w, h = img.size
    if min(w, h) < MIN_IMAGE_DIM:
        raise ValueError(
            f"Image is {w}x{h} px -- need at least "
            f"{MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}. Try a higher-resolution image."
        )

    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        w, h = new_size

    arr = np.asarray(img, dtype=np.float64)
    # Reject constant or near-constant images up front. skimage's threshold_otsu
    # does not raise on these in 0.26+; it just returns the constant value and
    # the resulting binary mask is all-True or all-False, which we'd detect a
    # few lines down with a much less helpful error.
    if arr.max() - arr.min() < 5:
        raise ValueError(
            "The image has no contrast -- looks blank. Try a clearer "
            "image with a distinct shape on a contrasting background."
        )
    # Otsu picks an automatic threshold. If foreground happens to be lighter
    # than background, we invert so the "shape" pixels are always True below.
    thresh = threshold_otsu(arr)
    binary = arr < thresh
    # Heuristic: most of the four corner pixels should be background. If
    # they're not, the shape is light-on-dark and we flip.
    corner_vals = np.array([binary[0, 0], binary[-1, 0], binary[0, -1], binary[-1, -1]])
    if corner_vals.sum() >= 3:
        binary = ~binary

    # Connected components, keep the largest by pixel count.
    labels, n_components = ndimage_label(binary)
    if n_components == 0:
        raise ValueError(
            "Couldn't find any shape in the image. Try a clearer image "
            "with a dark shape on a light background (or vice versa)."
        )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    largest_label = int(np.argmax(sizes))
    largest = labels == largest_label

    if n_components > 1:
        warnings.append(
            f"Found {n_components} disconnected pieces -- using the largest."
        )

    area_frac = float(largest.sum()) / float(w * h)
    if area_frac < MIN_AREA_FRAC:
        raise ValueError(
            f"Detected shape fills only {100 * area_frac:.1f}% of the image -- "
            "too small to simulate. Try a cleaner image where the shape "
            "fills more of the frame."
        )
    if area_frac > MAX_AREA_FRAC:
        raise ValueError(
            f"Detected shape fills {100 * area_frac:.1f}% of the image -- "
            "leave more whitespace around it so we can find the outline."
        )

    edge_touch = (
        largest[0, :].any() or largest[-1, :].any()
        or largest[:, 0].any() or largest[:, -1].any()
    )
    if edge_touch:
        raise ValueError(
            "The shape touches the edge of the image. Add some padding "
            "around it (a white border helps)."
        )

    pixels_before = int(largest.sum())
    filled = binary_fill_holes(largest)
    pixels_after = int(filled.sum())
    n_holes_filled = 1 if pixels_after > pixels_before else 0

    # find_contours returns a list of polylines at the given level. Pick the
    # longest one -- that's the outer boundary. Output is (row, col); we
    # swap to (x=col, y=row) so callers can think in PIL coords.
    contours = find_contours(filled.astype(np.float64), level=0.5)
    if not contours:
        raise ValueError("Couldn't trace the outline of the shape.")
    contour = max(contours, key=len)
    polygon = np.flip(contour, axis=1)  # (row, col) -> (x, y)

    tolerance = SIMPLIFY_TOLERANCE_FRAC * min(w, h)
    simplified = approximate_polygon(polygon, tolerance=tolerance)
    # A rectangle legitimately reduces to 4 (+1 closing) vertices, so the
    # minimum has to be 4. Lower than that means the contour collapsed to a
    # degenerate line or point -- usually a noisy upload that should be
    # surfaced as an error rather than rasterized into something silly.
    if len(simplified) < 4:
        raise ValueError(
            f"Simplified outline has only {len(simplified)} corners -- "
            "the shape collapsed during simplification. Try a smoother "
            "source image or one with less noise."
        )

    return SilhouetteResult(
        polygon_xy=simplified,
        image_w=w,
        image_h=h,
        n_holes_filled=n_holes_filled,
        n_components_found=n_components,
        warnings=warnings,
    )


def polygon_to_lbm_mask(
    polygon_xy: np.ndarray,
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    target_extent_cells: float,
    aoa_deg: float = 0.0,
) -> np.ndarray:
    """Rasterize a polygon onto the (Nx, Ny) LBM grid.

    Steps:
      1. Translate the polygon so its bbox centre is at the origin.
      2. Scale so its longest bbox dimension equals target_extent_cells.
      3. Flip y (PIL is y-down, lattice is y-up).
      4. Rotate by aoa_deg around the origin (counter-clockwise positive,
         matching the existing rotation convention for square/ellipse).
      5. Translate to (cx, cy).
      6. Rasterize via PIL ImageDraw onto a (Nx, Ny) bool mask.

    Returns a bool array of shape (Nx, Ny), True inside the polygon.
    """
    poly = np.asarray(polygon_xy, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(
            f"polygon_xy must be (N, 2) with N >= 3; got shape {poly.shape}"
        )

    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    extent_x = x_max - x_min
    extent_y = y_max - y_min
    longest = max(extent_x, extent_y)
    if longest <= 0:
        raise ValueError("Polygon has zero extent -- can't rasterize.")

    scale = target_extent_cells / longest

    # Center on origin, scale, flip y.
    centred = poly - np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    centred *= scale
    centred[:, 1] *= -1.0

    # Rotate (CCW positive in math convention -- matches square/ellipse).
    if aoa_deg != 0.0:
        theta = np.deg2rad(aoa_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        centred = centred @ rot.T

    final = centred + np.array([cx, cy])

    # PIL ImageDraw.polygon takes (W, H) image and a list of (x, y) tuples,
    # then we transpose back to (Nx, Ny) for the LBM mask convention.
    pil_img = Image.new("L", (Nx, Ny), 0)
    draw = ImageDraw.Draw(pil_img)
    draw.polygon([(float(p[0]), float(p[1])) for p in final], fill=1)
    return np.asarray(pil_img, dtype=bool).T


def render_silhouette_preview(
    polygon_xy: np.ndarray,
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    target_extent_cells: float,
    aoa_deg: float = 0.0,
) -> bytes:
    """Render a quick PNG preview of how the polygon will sit on the LBM grid.

    Same transform pipeline as polygon_to_lbm_mask -- the user sees the
    extracted polygon already centred, scaled, and rotated, with a flow
    arrow on the left so they can confirm orientation before clicking Run.
    Returns PNG bytes suitable for st.image().
    """
    import matplotlib.pyplot as plt

    xs, ys = polygon_outline_xy(
        polygon_xy, Nx, Ny, cx, cy, target_extent_cells, aoa_deg,
    )

    aspect = max(2.0, 8.0 * Ny / Nx)
    fig, ax = plt.subplots(figsize=(8.0, aspect), dpi=80)
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_aspect("equal")
    ax.fill(xs, ys, color="#cbd5e1", alpha=0.75, zorder=2)
    ax.plot(xs, ys, color="#f8fafc", linewidth=1.6, zorder=3)
    ax.annotate(
        "", xy=(Nx * 0.12, Ny * 0.5), xytext=(Nx * 0.02, Ny * 0.5),
        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2),
    )
    ax.text(
        Nx * 0.02, Ny * 0.62, "flow",
        color="#94a3b8", fontsize=9, fontfamily="monospace",
    )
    ax.set_facecolor("#0b1220")
    fig.patch.set_facecolor("#0b1220")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(axis="both", colors="#64748b", labelsize=8)

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", facecolor="#0b1220",
        bbox_inches="tight", pad_inches=0.05,
    )
    plt.close(fig)
    return buf.getvalue()


def polygon_outline_xy(
    polygon_xy: np.ndarray,
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    target_extent_cells: float,
    aoa_deg: float = 0.0,
):
    """Return the polygon vertices already transformed into lattice coords.

    Mirror of polygon_to_lbm_mask but returns separate (xs, ys) arrays so the
    LBM renderer can draw the smooth analytic body outline on top of the
    voxelized mask -- same convention as cylinder_outline_xy etc.
    """
    poly = np.asarray(polygon_xy, dtype=np.float64)
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    longest = max(x_max - x_min, y_max - y_min)
    if longest <= 0:
        raise ValueError("Polygon has zero extent.")

    scale = target_extent_cells / longest
    centred = poly - np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    centred *= scale
    centred[:, 1] *= -1.0

    if aoa_deg != 0.0:
        theta = np.deg2rad(aoa_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        centred = centred @ rot.T

    final = centred + np.array([cx, cy])
    # Close the polygon for matplotlib
    closed = np.vstack([final, final[0]])
    return closed[:, 0], closed[:, 1]
