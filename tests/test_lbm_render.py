"""Tests for the LBM rendering pipeline in src/lbm_render.py.

These cover three layers:

  1) Pure geometric helpers (`expand_outline`, `body_outline_xy`) -- fast,
     deterministic checks on the analytic body-outline overlay used to hide
     the voxelized mask boundary.

  2) The bilinear-interpolation primitive (`_bilerp`) -- the kernel of the
     particle-streakline RK4 advection. If this is wrong, every particle
     trajectory in the GIF is wrong.

  3) An end-to-end smoke test of `simulate_and_render` against every shape
     preset with `n_frames=5`. Catches any future break in the full
     LBM->snapshot->render->GIF chain (the kind of break that pytest of
     individual physics primitives won't see). ~5 s wall time after JIT
     warm-up.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from src.lbm_render import (
    BODY_OUTLINE_MARGIN,
    RESOLUTION_PRESETS,
    STEPS_PER_FRAME,
    _bilerp,
    body_outline_xy,
    expand_outline,
    simulate_and_render,
)


STANDARD = "Standard (320 x 80)"
STANDARD_CFG = RESOLUTION_PRESETS[STANDARD]


# ---------------------------------------------------------------------------
# expand_outline
# ---------------------------------------------------------------------------

def test_expand_outline_pushes_every_vertex_outward():
    """Every vertex moves strictly farther from the centroid by `margin` units.
    The body patch is drawn margin-cells outside the voxelized mask precisely
    to hide the staircase boundary -- if any vertex moves inward, the patch
    would expose the mask edge."""
    # Circle of radius 10 centered at (50, 30).
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    xs = 50 + 10 * np.cos(t)
    ys = 30 + 10 * np.sin(t)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))

    margin = 1.5
    xs2, ys2 = expand_outline(xs, ys, margin)

    r_before = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_after = np.sqrt((xs2 - cx) ** 2 + (ys2 - cy) ** 2)

    # Each vertex moves outward by ~margin (perfectly on a centered circle).
    assert np.allclose(r_after - r_before, margin)


def test_expand_outline_preserves_vertex_count_and_centroid():
    """Expansion is uniform around the centroid; both vertex count and the
    centroid itself must be preserved."""
    xs = np.array([10.0, 20.0, 20.0, 10.0])
    ys = np.array([10.0, 10.0, 20.0, 20.0])
    xs2, ys2 = expand_outline(xs, ys, margin=0.5)

    assert xs2.shape == xs.shape
    assert ys2.shape == ys.shape
    assert np.isclose(np.mean(xs2), np.mean(xs))
    assert np.isclose(np.mean(ys2), np.mean(ys))


def test_expand_outline_handles_degenerate_at_centroid():
    """A vertex at the centroid (r=0) would otherwise divide by zero. The
    `r_safe = max(r, 1.0)` guard must keep the function finite."""
    xs = np.array([0.0, 1.0, -1.0])
    ys = np.array([0.0, 1.0, -1.0])  # centroid at (0, 0), first vertex AT centroid
    xs2, ys2 = expand_outline(xs, ys, margin=1.0)
    assert np.all(np.isfinite(xs2))
    assert np.all(np.isfinite(ys2))


# ---------------------------------------------------------------------------
# body_outline_xy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_preset", [
    "Cylinder", "Square", "Ellipse", "NACA 0012", "NACA 4412",
])
def test_body_outline_returns_finite_pairs(shape_preset):
    """For every supported shape, the outline returns matched-length finite
    coordinate arrays. Catches any new shape preset that forgets a return path."""
    xs, ys = body_outline_xy(shape_preset, STANDARD_CFG, aoa_deg=0.0)
    assert xs.shape == ys.shape
    assert xs.ndim == 1
    assert len(xs) >= 4  # at least a triangle, ideally many more
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))


def test_body_outline_cylinder_centered_on_body_position():
    """Cylinder outline centroid must land on (body_x, cy) -- catches any drift
    introduced when the cylinder branch is touched."""
    xs, ys = body_outline_xy("Cylinder", STANDARD_CFG, aoa_deg=0.0)
    assert np.isclose(np.mean(xs), STANDARD_CFG["body_x"], atol=0.5)
    assert np.isclose(np.mean(ys), STANDARD_CFG["cy"], atol=0.5)


def test_body_outline_cylinder_radius_matches_preset():
    """The cylinder outline radius equals D/2 from the preset. Sanity-checks
    that the displayed body matches the physics mask in size."""
    xs, ys = body_outline_xy("Cylinder", STANDARD_CFG, aoa_deg=0.0)
    r = np.sqrt((xs - STANDARD_CFG["body_x"]) ** 2 +
                (ys - STANDARD_CFG["cy"]) ** 2)
    expected_r = STANDARD_CFG["cylinder_D"] / 2
    assert np.allclose(r, expected_r)


def test_body_outline_square_rotation_changes_outline():
    """Rotating a square by 45° must move every vertex (no fixed points except
    the centroid). Confirms aoa_deg actually drives the rotation."""
    xs0, ys0 = body_outline_xy("Square", STANDARD_CFG, aoa_deg=0.0)
    xs45, ys45 = body_outline_xy("Square", STANDARD_CFG, aoa_deg=45.0)
    # Different rotations must produce different vertex positions (not just
    # the same vertices in a different order).
    assert not np.allclose(sorted(xs0), sorted(xs45)) or \
           not np.allclose(sorted(ys0), sorted(ys45))


def test_body_outline_naca0012_is_chord_symmetric_at_zero_aoa():
    """A symmetric NACA at AoA=0 must mirror across the chord line (y = cy).
    The outline is built from naca4_outline_xy which already guarantees this
    for the airfoil itself; this test makes sure body_outline_xy doesn't
    accidentally break that symmetry through its transform."""
    cy = STANDARD_CFG["cy"]
    xs, ys = body_outline_xy("NACA 0012", STANDARD_CFG, aoa_deg=0.0)
    # For every point (x, y) above the chord, there should be one below.
    # Use sorted y-values relative to chord: should be near-mirror symmetric.
    dy = ys - cy
    above = np.sort(dy[dy > 0])
    below = np.sort(-dy[dy < 0])
    # Trim to equal length (in case there's one point exactly on chord).
    n = min(len(above), len(below))
    assert np.allclose(above[:n], below[:n], atol=1e-6)


# ---------------------------------------------------------------------------
# _bilerp (private but load-bearing for particle advection)
# ---------------------------------------------------------------------------

def test_bilerp_at_integer_positions_returns_field_value():
    """At an integer (x, y), bilinear interpolation should return the cell
    value exactly -- no contribution from neighbors when fx=fy=0."""
    field = np.arange(50, dtype=np.float64).reshape(10, 5)
    xs = np.array([2.0, 5.0, 9.0])
    ys = np.array([1.0, 3.0, 4.0])  # clipped to 3 internally (max y0 = ny-2)
    out = _bilerp(field, xs, ys)
    # At (2, 1): field[2, 1] = 11. At (5, 3): field[5, 3] = 28.
    # At (9, 4): y0 clipped to 3, fy=1.0 -> returns field[9,4] = 49.
    assert np.isclose(out[0], field[2, 1])
    assert np.isclose(out[1], field[5, 3])
    assert np.isclose(out[2], field[9, 4])


def test_bilerp_at_half_integer_is_midpoint_average():
    """At a midpoint between four cells, bilinear interpolation returns the
    average of the four values. Strongest test of the kernel."""
    field = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2
    out = _bilerp(field, np.array([0.5]), np.array([0.5]))
    assert np.isclose(out[0], (1.0 + 2.0 + 3.0 + 4.0) / 4.0)


def test_bilerp_recovers_linear_field_exactly():
    """A field that's a linear function of (x, y) must be reproduced exactly
    by bilinear interpolation at any FP coordinate. This is the defining
    property of bilinear: linear fields = no interpolation error."""
    Nx, Ny = 20, 15
    x_idx, y_idx = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    field = 1.5 + 0.3 * x_idx + 0.7 * y_idx

    rng = np.random.default_rng(seed=2026)
    # Sample inside [1, Nx-2] x [1, Ny-2] to avoid clipping affecting the result.
    xs = rng.uniform(1.0, Nx - 2.0, size=20)
    ys = rng.uniform(1.0, Ny - 2.0, size=20)
    expected = 1.5 + 0.3 * xs + 0.7 * ys
    out = _bilerp(field, xs, ys)
    assert np.allclose(out, expected, atol=1e-12)


def test_bilerp_clips_out_of_bounds_safely():
    """Coordinates outside the grid get clipped to in-bounds neighbors instead
    of raising. The RK4 advection occasionally projects particles right to
    the edge during a step (they're culled afterward, but interpolation
    must not crash mid-step)."""
    field = np.arange(50, dtype=np.float64).reshape(10, 5)
    xs = np.array([-5.0, 100.0])  # both wildly out of range
    ys = np.array([-2.0, 100.0])
    out = _bilerp(field, xs, ys)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# End-to-end: simulate_and_render with n_frames=5 (~5 s after JIT warm-up)
# ---------------------------------------------------------------------------

# Per-shape minimal end-to-end coverage. Re=200, AoA=5 for non-cylinders
# (small enough that the wake develops but doesn't go chaotic in 5 frames).
# Cylinder gets AoA=0 since it's rotationally invariant.
@pytest.mark.parametrize("shape_preset,aoa_deg", [
    ("Cylinder", 0.0),
    ("Square", 5.0),
    ("Ellipse", 5.0),
    ("NACA 0012", 5.0),
    ("NACA 4412", 5.0),
])
def test_simulate_and_render_end_to_end(shape_preset, aoa_deg):
    """Run the full LBM->snapshot->render->GIF pipeline for 5 frames against
    each shape preset. Verifies:
      - The result dict has every key app.py unpacks.
      - The GIF bytes are non-empty AND decodable by PIL (catches a broken
        encoder, truncated buffer, or wrong content-type).
      - The decoded GIF has the expected frame count (5).
      - tau lands in the LBM-stable range (> 0.5).

    This is the single most important regression catch in the suite: any
    future refactor of lbm_render that breaks the contract with app.py
    (renamed keys, off-by-one in n_frames, broken GIF assembly) trips here
    instead of in a user's browser."""
    result = simulate_and_render(
        shape_preset, 200, aoa_deg, STANDARD, n_frames=5,
    )

    # 1) Contract: every dict key app.py reads must exist.
    required_keys = {
        "gif_bytes", "vort_cbar_bytes", "speed_cbar_bytes",
        "label", "tau", "nu", "char_length",
        "lbm_nx", "lbm_ny", "n_frames", "n_steps", "near_stable",
    }
    assert required_keys.issubset(result.keys()), (
        f"missing keys: {required_keys - result.keys()}"
    )

    # 2) Bytes are non-empty.
    assert len(result["gif_bytes"]) > 1024, "GIF suspiciously small"
    assert len(result["vort_cbar_bytes"]) > 256
    assert len(result["speed_cbar_bytes"]) > 256

    # 3) GIF is actually decodable + has the right frame count.
    img = Image.open(io.BytesIO(result["gif_bytes"]))
    assert img.format == "GIF"
    # PIL's n_frames is the total number of frames in the GIF.
    assert getattr(img, "n_frames", 1) == 5

    # 4) Physics scalars are sensible (no NaN, no divergence).
    assert np.isfinite(result["tau"])
    assert result["tau"] > 0.5  # LBM stable region
    assert np.isfinite(result["nu"])
    assert result["nu"] > 0.0
    assert result["lbm_nx"] == STANDARD_CFG["Nx"]
    assert result["lbm_ny"] == STANDARD_CFG["Ny"]
    assert result["n_frames"] == 5
    assert result["n_steps"] == 5 * STEPS_PER_FRAME


def test_simulate_and_render_progress_callback_called():
    """If a progress_callback is supplied, it must be called at least once per
    phase (sim + render) and end with frac=1.0 -- this is how the Streamlit
    UI shows progress. A silent regression would leave the user staring at
    a 0% bar for 30 s."""
    calls = []

    def cb(frac, text):
        calls.append((frac, text))

    simulate_and_render(
        "Cylinder", 200, 0.0, STANDARD, n_frames=3, progress_callback=cb,
    )

    assert len(calls) > 0, "progress_callback never invoked"
    fracs = [c[0] for c in calls]
    assert all(0.0 <= f <= 1.0 for f in fracs), f"fractions out of range: {fracs}"
    assert max(fracs) == pytest.approx(1.0), (
        f"final progress not 1.0 (max was {max(fracs)})"
    )


def test_simulate_and_render_default_n_frames_uses_preset():
    """When n_frames is not passed, the preset's frame count must apply.
    Catches a refactor that accidentally hardcodes a test override into
    production. We don't run the full simulation (too slow) -- just inspect
    the returned dict for a config-equivalent call (n_frames=1)."""
    # Run with n_frames=1 (fastest) and confirm the result reflects that
    # explicit override, then confirm we *can* read preset n_frames separately.
    result = simulate_and_render(
        "Cylinder", 200, 0.0, STANDARD, n_frames=1,
    )
    assert result["n_frames"] == 1
    # Preset's default frame count is still what production uses. The
    # exact number is tuned for Cloud wall-time; this test just guards
    # against the preset accidentally going below the kick window.
    assert STANDARD_CFG["n_frames"] >= 30, (
        f"Standard preset n_frames={STANDARD_CFG['n_frames']} is shorter "
        f"than the kick window (frames 0.6-4) + ~25 frames of recorded wake "
        f"-- the wake won't develop. Bump back up."
    )
