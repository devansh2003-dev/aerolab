"""Unit tests for src/custom_shape.py.

Strategy: generate synthetic PNGs in-memory with known geometry (filled
ellipse, square, triangle), run them through the extractor, assert the
result matches the input geometry to within polygon-simplification slack.
"""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.custom_shape import (
    MAX_AREA_FRAC,
    MAX_IMAGE_DIM,
    MIN_AREA_FRAC,
    MIN_IMAGE_DIM,
    SilhouetteResult,
    extract_silhouette_from_image,
    polygon_outline_xy,
    polygon_to_lbm_mask,
)


def _make_png_with_shape(draw_fn, size=(400, 300), bg=255, fg=0) -> bytes:
    """Render a shape to a PNG and return its bytes. draw_fn(ImageDraw) is the
    caller-supplied lambda that adds the shape to the image."""
    img = Image.new("L", size, bg)
    d = ImageDraw.Draw(img)
    draw_fn(d, fg)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# === extract_silhouette_from_image ===


def test_extracts_filled_ellipse():
    """A clean black ellipse on white background should give a polygon with
    bbox matching the input ellipse's bbox to within a few pixels."""
    bbox = (80, 80, 320, 220)  # x0, y0, x1, y1
    png = _make_png_with_shape(lambda d, fg: d.ellipse(bbox, fill=fg))

    result = extract_silhouette_from_image(png)
    assert isinstance(result, SilhouetteResult)
    assert result.image_w == 400 and result.image_h == 300
    assert result.n_components_found == 1

    poly = result.polygon_xy
    assert poly.ndim == 2 and poly.shape[1] == 2
    assert len(poly) >= 8

    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    # Allow ~5 px slack for Douglas-Peucker.
    assert abs(x_min - bbox[0]) < 6
    assert abs(y_min - bbox[1]) < 6
    assert abs(x_max - bbox[2]) < 6
    assert abs(y_max - bbox[3]) < 6


def test_extracts_square():
    """A black square gives a polygon with ~4 corners after DP simplification."""
    png = _make_png_with_shape(lambda d, fg: d.rectangle((100, 100, 300, 250), fill=fg))
    result = extract_silhouette_from_image(png)
    # A square should simplify to a polygon with ~4-6 vertices (slack for DP).
    assert 4 <= len(result.polygon_xy) <= 16


def test_extracts_light_shape_on_dark_background():
    """White-on-black: same shape should be detected. Background sampling
    detects the black border, foreground signal = |arr - 0| = arr peaks
    inside the shape, so the same Otsu pass extracts it without any
    explicit inversion logic."""
    png = _make_png_with_shape(
        lambda d, fg: d.ellipse((100, 80, 300, 220), fill=255),
        bg=0,
    )
    result = extract_silhouette_from_image(png)
    poly = result.polygon_xy
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    assert abs(x_min - 100) < 8
    assert abs(x_max - 300) < 8


def test_extracts_shape_on_grey_background():
    """Mid-grey background (128): old corner-Otsu heuristic would mis-
    threshold or invert wrongly. Border-ring background detection sees
    bg=128 and the foreground signal correctly peaks inside the shape."""
    img = Image.new("L", (400, 300), 128)
    d = ImageDraw.Draw(img)
    d.ellipse((100, 80, 300, 220), fill=40)  # darker shape on grey
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = extract_silhouette_from_image(buf.getvalue())
    poly = result.polygon_xy
    x_min = poly[:, 0].min()
    x_max = poly[:, 0].max()
    assert abs(x_min - 100) < 8
    assert abs(x_max - 300) < 8


def test_extracts_shape_from_rgba_with_alpha():
    """A PNG with transparent background and the shape rendered with full
    opacity. Without alpha compositing, the bytes under alpha=0 are often
    (0, 0, 0), making the perceived background black -- which would
    threshold the wrong way for a 'white-bg' subject."""
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))  # transparent everywhere
    d = ImageDraw.Draw(img)
    d.ellipse((100, 80, 300, 220), fill=(40, 40, 40, 255))  # opaque dark shape
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = extract_silhouette_from_image(buf.getvalue())
    poly = result.polygon_xy
    # Should detect the ellipse, not be confused by the alpha-stored zeros.
    assert abs(poly[:, 0].min() - 100) < 8
    assert abs(poly[:, 0].max() - 300) < 8


def test_morphological_closing_bridges_thin_gaps():
    """Two ellipses separated by a 2-px gap should merge into ONE component
    after closing, simulating a character whose arm anti-aliasing breaks
    the limb into pieces. With iter=max(2, min(w,h)//200)=2 on a 400x300
    image, gaps up to ~4 px bridge."""
    def draw(d, fg):
        d.ellipse((100, 100, 200, 200), fill=fg)
        d.ellipse((202, 100, 300, 200), fill=fg)  # 2-px horizontal gap
    png = _make_png_with_shape(draw)
    result = extract_silhouette_from_image(png)
    # After closing the two ellipses merge -> one component.
    assert result.n_components_found == 1


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "TIFF", "WEBP", "GIF"])
def test_extracts_from_common_raster_formats(fmt):
    """All common raster formats should round-trip through the extractor
    via PIL's native plugins. Same shape, same expected polygon shape."""
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((100, 80, 300, 220), fill="black")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    result = extract_silhouette_from_image(buf.getvalue())
    assert len(result.polygon_xy) >= 8
    assert abs(result.polygon_xy[:, 0].min() - 100) < 8
    assert abs(result.polygon_xy[:, 0].max() - 300) < 8


def test_extracts_first_frame_of_animated_gif():
    """Animated GIFs / multi-page TIFFs / animated WebPs have n_frames > 1;
    the extractor should pick frame 0 and surface a friendly warning."""
    frame_a = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(frame_a).ellipse((100, 80, 300, 220), fill="black")
    frame_b = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(frame_b).rectangle((50, 50, 350, 250), fill="black")
    buf = io.BytesIO()
    frame_a.save(
        buf, format="GIF", save_all=True, append_images=[frame_b],
        duration=200, loop=0,
    )
    result = extract_silhouette_from_image(buf.getvalue())
    # First frame is the ellipse, not the bigger rectangle. The polygon's
    # bbox should match the ellipse's bbox, not the rectangle's.
    assert abs(result.polygon_xy[:, 0].min() - 100) < 8
    assert abs(result.polygon_xy[:, 0].max() - 300) < 8
    assert any("frames" in w for w in result.warnings)


def test_rejects_corrupt_image():
    """Non-image bytes should raise a clean ValueError, not a PIL crash."""
    with pytest.raises(ValueError, match="Couldn't decode"):
        extract_silhouette_from_image(b"this is not an image, just text")


def test_rejects_image_too_small():
    png = _make_png_with_shape(
        lambda d, fg: d.ellipse((10, 10, 50, 50), fill=fg),
        size=(60, 60),
    )
    with pytest.raises(ValueError, match="at least"):
        extract_silhouette_from_image(png)


def test_accepts_shape_touching_edge_via_auto_padding():
    """Shape that runs to the image border used to be rejected. Now the
    extractor auto-pads with a bg-coloured border so the shape gets the
    whitespace it needs, and extraction succeeds. The returned polygon
    is in original-image coordinates (the padding is purely internal)."""
    png = _make_png_with_shape(
        lambda d, fg: d.rectangle((0, 50, 200, 250), fill=fg),  # left edge touches
    )
    result = extract_silhouette_from_image(png)
    # Polygon bbox should still anchor at x ~ 0 in original-image coords --
    # auto-padding shouldn't have shifted the result.
    assert result.polygon_xy[:, 0].min() < 5
    assert abs(result.polygon_xy[:, 0].max() - 200) < 6


def test_rejects_too_small_shape():
    """A 1 % area shape is below the MIN_AREA_FRAC=2 % gate."""
    png = _make_png_with_shape(
        lambda d, fg: d.ellipse((195, 145, 205, 155), fill=fg),  # tiny dot
    )
    with pytest.raises(ValueError, match="too small to simulate|fills"):
        extract_silhouette_from_image(png)


def test_rejects_blank_image():
    """Constant-colour image has no contrast; threshold_otsu raises and we
    translate that to a user-readable error."""
    img = Image.new("L", (400, 300), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with pytest.raises(ValueError, match="no contrast|blank"):
        extract_silhouette_from_image(buf.getvalue())


def test_keeps_largest_component_with_warning():
    """Multiple disconnected shapes: extractor keeps the largest and
    surfaces a warning about the discarded pieces."""
    def draw(d, fg):
        d.ellipse((40, 40, 360, 260), fill=fg)   # big shape
        d.ellipse((10, 10, 30, 30), fill=fg)     # small distractor
    png = _make_png_with_shape(draw)
    result = extract_silhouette_from_image(png)
    assert result.n_components_found == 2
    assert any("disconnected" in w for w in result.warnings)
    # Bbox should be the BIG ellipse, not the small one.
    poly = result.polygon_xy
    assert poly[:, 0].max() > 300


def test_fills_internal_hole():
    """Donut: outer outline traced cleanly, hole filled before contouring."""
    def draw(d, fg):
        d.ellipse((80, 60, 320, 240), fill=fg)
        d.ellipse((170, 130, 230, 170), fill=255)  # hole
    png = _make_png_with_shape(draw)
    result = extract_silhouette_from_image(png)
    assert result.n_holes_filled == 1
    # Polygon should follow the OUTER boundary, not also outline the hole.
    poly = result.polygon_xy
    assert len(poly) < 80  # if the hole was traced too, we'd get many more vertices


def test_oversize_image_resized_to_max():
    """A 4000-px-wide image gets downscaled to MAX_IMAGE_DIM."""
    png = _make_png_with_shape(
        lambda d, fg: d.ellipse((800, 800, 3200, 2200), fill=fg),
        size=(4000, 3000),
    )
    result = extract_silhouette_from_image(png)
    assert max(result.image_w, result.image_h) <= MAX_IMAGE_DIM


# === polygon_to_lbm_mask ===


def test_rasterize_square_polygon_to_mask():
    """A unit-square polygon scaled to target_extent=20 centred at (50, 40)
    in an (Nx=120, Ny=80) grid should produce a 20x20 mask block around (50, 40)."""
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    mask = polygon_to_lbm_mask(square, Nx=120, Ny=80, cx=50.0, cy=40.0, target_extent_cells=20.0)

    assert mask.shape == (120, 80)
    assert mask.dtype == bool
    # Most pixels in the 20x20 box around (50, 40) should be True.
    inside = mask[40:60, 30:50]
    assert inside.sum() > 350  # 20x20 = 400 cells, allow rasterization slack


def test_rasterize_preserves_thin_features():
    """A character-like polygon (tall body + thin arms / legs) should
    retain its appendages after rasterizing onto an 80-cell-tall grid.
    Regression: earlier versions ran binary_opening here to remove
    1-cell solver protrusions, which also erased legitimate thin
    features and caused user-visible 'parts get cut off' bugs."""
    # Stick-figure: head + body + arms + legs, with 3-px-wide limbs.
    poly = np.array([
        # Head outline (top of figure)
        (50, 0), (60, 5), (65, 15), (60, 25), (50, 30),
        # Down right side of body to right arm
        (52, 35), (75, 38), (78, 40), (75, 42), (52, 45),
        # Continue body to right leg
        (52, 70), (60, 105), (58, 108), (50, 80),
        # Across to left leg
        (50, 108), (42, 105), (50, 70),
        # Left arm
        (48, 45), (25, 42), (22, 40), (25, 38), (48, 35),
        # Back up to head
        (40, 25), (35, 15), (40, 5),
    ], dtype=np.float64)
    mask = polygon_to_lbm_mask(
        poly, Nx=320, Ny=80, cx=70.0, cy=40.0,
        target_extent_cells=60.0, aoa_deg=0.0,
    )
    # The figure spans the full 60-cell target_extent in y. Verify that
    # both the upper (head) and lower (feet) halves have solid cells --
    # if opening were erasing the limbs, the top + bottom rows of the
    # body would be sparse.
    ys = np.where(mask.any(axis=0))[0]
    assert ys.min() < 15, f"top of figure missing: y_min={ys.min()}"
    assert ys.max() > 65, f"bottom of figure missing: y_max={ys.max()}"
    # Solid-cell count should be substantial -- if features were eroded
    # we'd see < 200 cells; the figure (body + limbs) is ~400-600.
    assert mask.sum() > 200, f"too few solid cells: {mask.sum()}"


def test_rasterize_respects_rotation():
    """A square rotated 45 deg should have a smaller axis-aligned bbox in the
    grid (the diamond fits inside a wider/taller box, but the cells on the
    far edges are sparser)."""
    square = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1],
    ], dtype=np.float64)
    mask_0 = polygon_to_lbm_mask(square, 120, 80, 60.0, 40.0, 20.0, aoa_deg=0.0)
    mask_45 = polygon_to_lbm_mask(square, 120, 80, 60.0, 40.0, 20.0, aoa_deg=45.0)

    # Both produce filled regions of the same total area (within rasterization).
    assert abs(int(mask_0.sum()) - int(mask_45.sum())) < 50
    # The rotated version has a wider y-extent (diamond is taller than the
    # axis-aligned square at the same target_extent_cells).
    ys_0 = np.where(mask_0.any(axis=0))[0]
    ys_45 = np.where(mask_45.any(axis=0))[0]
    assert (ys_45.max() - ys_45.min()) > (ys_0.max() - ys_0.min())


def test_rasterize_rejects_degenerate():
    """A polygon with all-coincident vertices has zero extent."""
    degenerate = np.array([[5.0, 5.0]] * 5)
    with pytest.raises(ValueError, match="zero extent"):
        polygon_to_lbm_mask(degenerate, 100, 60, 50.0, 30.0, 20.0)


def test_rasterize_rejects_too_few_vertices():
    with pytest.raises(ValueError, match="N >= 3"):
        polygon_to_lbm_mask(np.array([[0.0, 0.0], [1.0, 0.0]]), 100, 60, 50, 30, 20)


# === polygon_outline_xy ===


def test_polygon_outline_is_closed():
    """outline_xy returns the polygon vertices closed for matplotlib drawing."""
    triangle = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float64)
    xs, ys = polygon_outline_xy(triangle, Nx=100, Ny=60, cx=50.0, cy=30.0, target_extent_cells=20.0)
    # First and last vertices should coincide (closed loop).
    assert xs[0] == pytest.approx(xs[-1])
    assert ys[0] == pytest.approx(ys[-1])
    assert len(xs) == 4   # original 3 + closing vertex


# === End-to-end ===


def test_simulate_and_render_accepts_custom_polygon():
    """End-to-end smoke: extract a polygon from a synthetic PNG, feed it to
    simulate_and_render with shape_preset='Custom', verify we get a GIF
    back with the expected metadata keys."""
    from src.lbm_render import simulate_and_render

    png = _make_png_with_shape(
        lambda d, fg: d.ellipse((80, 80, 320, 220), fill=fg),
        size=(400, 300),
    )
    result = extract_silhouette_from_image(png)

    out = simulate_and_render(
        "Custom", reynolds_target=200, aoa_deg=0.0,
        res_key="Standard (320 x 80)",
        n_frames=3,  # tiny for speed
        custom_polygon=result.polygon_xy,
    )
    assert isinstance(out["gif_bytes"], bytes) and len(out["gif_bytes"]) > 0
    assert out["lbm_nx"] == 320 and out["lbm_ny"] == 80
    assert out["label"].startswith("Custom shape")
    # char_length should equal the preset's custom_extent (Standard = 60).
    assert out["char_length"] == pytest.approx(60.0)


def test_simulate_and_render_rotates_custom_polygon():
    """Same polygon at AoA=0 vs AoA=45 should produce different labels and
    different masks (verified indirectly via label string)."""
    from src.lbm_render import simulate_and_render

    png = _make_png_with_shape(
        lambda d, fg: d.rectangle((100, 110, 300, 190), fill=fg),
        size=(400, 300),
    )
    poly = extract_silhouette_from_image(png).polygon_xy

    out_0 = simulate_and_render(
        "Custom", 200, 0.0, "Standard (320 x 80)",
        n_frames=2, custom_polygon=poly,
    )
    out_45 = simulate_and_render(
        "Custom", 200, 45.0, "Standard (320 x 80)",
        n_frames=2, custom_polygon=poly,
    )
    assert "rotation" not in out_0["label"]
    assert "+45.0" in out_45["label"]


def test_simulate_and_render_rejects_custom_without_polygon():
    from src.lbm_render import simulate_and_render
    with pytest.raises(ValueError, match="custom_polygon"):
        simulate_and_render(
            "Custom", 200, 0.0, "Standard (320 x 80)", n_frames=2,
        )


# === Phase 2 W5 validation gate ===
# README states the W5 gate as: "image-upload demo working end-to-end on
# three real-world silhouettes (car profile, fish, building cross-section)".
# These are bundled as parametric polygons in src/sample_shapes.py; each
# must run a short pipeline without producing NaN frames or zero-sized
# GIFs at Re=200 on the Standard preset.


@pytest.mark.parametrize("sample_name", ["Fish", "Car profile", "Building cross-section"])
def test_phase2_w5_gate_sample_silhouettes_run_clean(sample_name):
    """Phase 2 W5 gate: bundled sample silhouettes run end-to-end without
    NaN at Re=200 Standard. n_frames=4 is the minimum that exercises both
    JIT-compiled paths (warmup kick + record loop) -- if the polygon
    produced an unstable mask, NaN would show up here."""
    from src.lbm_render import simulate_and_render
    from src.sample_shapes import SAMPLE_SHAPES

    polygon = SAMPLE_SHAPES[sample_name]()
    out = simulate_and_render(
        "Custom", reynolds_target=200, aoa_deg=0.0,
        res_key="Standard (320 x 80)",
        n_frames=4, custom_polygon=polygon,
    )
    assert isinstance(out["gif_bytes"], bytes)
    assert len(out["gif_bytes"]) > 5000   # non-empty animation
    assert out["lbm_nx"] == 320 and out["lbm_ny"] == 80
    # tau >= 0.5 means the kinematic-viscosity setup didn't degenerate.
    assert out["tau"] > 0.5


def test_sample_shapes_module_returns_valid_polygons():
    """Each bundled sample yields a polygon meeting polygon_to_lbm_mask's
    contract: (N, 2) shape, at least 3 vertices, finite values, non-degenerate."""
    from src.sample_shapes import SAMPLE_SHAPES

    assert len(SAMPLE_SHAPES) == 3
    for name, fn in SAMPLE_SHAPES.items():
        poly = fn()
        assert poly.ndim == 2 and poly.shape[1] == 2, (
            f"{name!r}: polygon shape {poly.shape} is not (N, 2)"
        )
        assert poly.shape[0] >= 3, f"{name!r}: needs >= 3 vertices"
        assert np.isfinite(poly).all(), f"{name!r}: polygon has non-finite values"
        extent_x = poly[:, 0].max() - poly[:, 0].min()
        extent_y = poly[:, 1].max() - poly[:, 1].min()
        assert extent_x > 0 and extent_y > 0, f"{name!r}: polygon has zero extent"


def test_image_to_mask_roundtrip_preserves_shape_centred():
    """Upload a 200x150 ellipse PNG, extract polygon, rasterize back to a
    100x60 LBM grid. The mask should be centred and roughly elliptical."""
    bbox = (40, 30, 160, 120)  # ellipse in PNG coords
    png = _make_png_with_shape(
        lambda d, fg: d.ellipse(bbox, fill=fg),
        size=(200, 150),
    )
    result = extract_silhouette_from_image(png)
    mask = polygon_to_lbm_mask(
        result.polygon_xy, Nx=100, Ny=60, cx=50.0, cy=30.0, target_extent_cells=30.0,
    )
    # Roughly half the cells in a 30x22 ellipse box should be set.
    # Area of a 30x22 ellipse = pi * 15 * 11 = 518. Allow generous slack.
    assert 200 < int(mask.sum()) < 800
    # Centre of mass should be near (50, 30).
    ys_x, ys_y = np.where(mask)
    assert abs(np.mean(ys_x) - 50) < 3
    assert abs(np.mean(ys_y) - 30) < 3
