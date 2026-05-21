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
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
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

    Robust to any single-colour background (white, black, grey, any tint)
    and to PNGs with alpha channels. Pipeline:
      1. Load + alpha-composite onto white (so transparent PNGs work) +
         grayscale + resize cap.
      2. Reject if smaller than MIN_IMAGE_DIM or near-constant.
      3. Sample background colour from a thin border ring. Foreground
         signal = |pixel - background|. This generalises the old "dark-
         on-light OR light-on-dark via auto-invert" approach to ANY
         uniform background colour, including sepia / mid-grey / muted
         3D-render backgrounds.
      4. Otsu threshold on the foreground signal -> binary mask.
      5. Morphological closing to bridge anti-aliased gaps and connect
         narrow-but-touching limbs (e.g. a character's wrist holding a
         bat). Closing radius scales with image size.
      6. Find connected components, keep the largest.
      7. Reject if the largest touches the image edge, or area is < 2 %
         or > 85 % of frame.
      8. binary_fill_holes on the kept component.
      9. find_contours + approximate_polygon (Douglas-Peucker).
     10. Reject if simplified polygon has fewer than 4 vertices.

    Raises ValueError with a user-readable message on extraction failure.
    """
    warnings: list = []

    pil_img = Image.open(io.BytesIO(png_bytes))
    # Alpha compositing: PNGs from many tools (transparent-background
    # exports from Photoshop, AI image generators, 3D render pipelines)
    # store their visible pixels with the subject's RGB and use the alpha
    # channel for transparency. Plain .convert("L") would read whatever
    # RGB was stored under alpha=0 -- often (0, 0, 0), which makes the
    # "white-background" subject look black-background to the extractor.
    # We composite onto white explicitly so the perceived background
    # always matches what the user sees in their image viewer.
    if (
        pil_img.mode in ("RGBA", "LA")
        or (pil_img.mode == "P" and "transparency" in pil_img.info)
    ):
        rgba = pil_img.convert("RGBA")
        white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
        white_bg.paste(rgba, mask=rgba.split()[-1])
        img = white_bg.convert("L")
    else:
        img = pil_img.convert("L")

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
    if arr.max() - arr.min() < 5:
        raise ValueError(
            "The image has no contrast -- looks blank. Try a clearer "
            "image with a distinct shape on a contrasting background."
        )

    # Sample a thin border ring to estimate the background brightness.
    # Median is robust to small amounts of shape-near-edge contamination
    # (a 5-10 % overlap doesn't move the median). border_w scales with
    # image size so we always have a statistically meaningful sample.
    border_w = max(2, min(w, h) // 40)
    border_pixels = np.concatenate([
        arr[:border_w, :].ravel(),
        arr[-border_w:, :].ravel(),
        arr[border_w:-border_w, :border_w].ravel(),
        arr[border_w:-border_w, -border_w:].ravel(),
    ])
    bg_value = float(np.median(border_pixels))

    # Foreground signal: distance from the background brightness. Pixels
    # that differ from the background -- regardless of direction -- are
    # candidate foreground. Otsu picks the threshold inside this
    # distance image, which is always non-negative and tends to be
    # cleanly bimodal even when the source image isn't.
    fg_signal = np.abs(arr - bg_value)
    try:
        fg_thresh = threshold_otsu(fg_signal)
    except ValueError as e:
        raise ValueError(
            "Couldn't separate the shape from the background. Try an "
            "image with a single solid-colour background."
        ) from e
    binary = fg_signal > fg_thresh

    # Morphological closing: bridge AA-edge gaps and reconnect narrow
    # appendages. iter scales with image size so the bridged-gap width
    # is roughly constant in fraction-of-image (~0.5 % of shorter dim).
    # We pad with False before closing so the erosion half of the
    # operation doesn't shrink shapes that touch the array boundary --
    # that would let edge-touching uploads silently pass the edge-touch
    # gate below, defeating its purpose.
    closing_iters = max(2, min(w, h) // 200)
    pad = closing_iters + 1
    padded = np.pad(binary, pad, mode="constant", constant_values=False)
    padded = binary_closing(padded, iterations=closing_iters)
    binary = padded[pad:-pad, pad:-pad]

    # Connected components, keep the largest by pixel count.
    labels, n_components = ndimage_label(binary)
    if n_components == 0:
        raise ValueError(
            "Couldn't find any shape in the image. Make sure the subject "
            "stands out clearly against its background."
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

    # Center on origin, scale, flip y. The y-flip is required because PIL
    # image coords have y increasing downward but the lattice has y
    # increasing upward -- without it the body would appear vertically
    # mirrored relative to the source. We do NOT flip x: uploaded shapes
    # should appear in the tunnel with the same left-right orientation as
    # the source image. Sample polygons in src/sample_shapes.py are
    # pre-defined with the "front" on the left side of their bbox so they
    # face the inflow naturally.
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
    mask = np.asarray(pil_img, dtype=bool).T

    # Morphological opening: erode by 1 cell then dilate by 1 cell. Removes
    # 1-cell-wide protrusions (kite tail-string, hair-thin whiskers) AND
    # isolated single-cell islands AND tight pinch-points. These features
    # are unphysical in a discrete-grid solver -- bounce-back on a 1-cell
    # appendage can't sustain mass, density collapses to ~0, and the next
    # step's MRT moment loop divides by zero. We remove them at the source
    # rather than band-aiding the solver.
    mask = binary_opening(mask, iterations=1)

    # Then keep only the largest connected component (opening may still
    # leave smaller closed regions, e.g. a wing tip that detached from the
    # main body after erosion). Single closed body -> single contiguous
    # solid region -> clean solver state.
    labels, n_labels = ndimage_label(mask)
    if n_labels > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(np.argmax(sizes))
    elif n_labels == 0:
        raise ValueError(
            "Shape vanished during morphological cleanup -- the silhouette "
            "is too thin / spindly for the LBM grid. Try a chunkier shape."
        )
    return mask


def render_outline_to_png(xs: np.ndarray, ys: np.ndarray, Nx: int, Ny: int) -> bytes:
    """Render a closed (xs, ys) outline on a dark tunnel background.

    Used by both render_silhouette_preview (uploads / samples) and by
    lbm_render.render_shape_preview (built-in cylinder / ellipse / NACA /
    square). Same look-and-feel so the user gets the same orientation
    cue regardless of which shape they pick. Returns PNG bytes suitable
    for st.image().
    """
    import matplotlib.pyplot as plt

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


def render_silhouette_preview(
    polygon_xy: np.ndarray,
    Nx: int,
    Ny: int,
    cx: float,
    cy: float,
    target_extent_cells: float,
    aoa_deg: float = 0.0,
) -> bytes:
    """Preview an uploaded / drawn polygon already placed on the LBM grid.

    Thin wrapper: transforms polygon_xy via polygon_outline_xy and hands
    the result to render_outline_to_png for the shared rendering pass.
    """
    xs, ys = polygon_outline_xy(
        polygon_xy, Nx, Ny, cx, cy, target_extent_cells, aoa_deg,
    )
    return render_outline_to_png(xs, ys, Nx, Ny)


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
