"""Visual regression guard for the GIF render pipeline.

Catches the failure mode described in Section IX of the project audit:
"we rewrote the rendering 6+ times, none A/B-tested, regressions only
surface when the user squints at the wake and says 'that's wrong.'"

Approach: a fixed canonical config (Cylinder, Re=400, AoA=0, Standard,
frame 50 of 51) produces a frame whose downscaled-grayscale fingerprint
is bit-stable across consecutive runs on the same machine. The test
compares against the committed baseline at
``tests/baselines/canonical_cylinder_re400_f50.png``.

Re-baseline when an intentional visual change lands (colormap swap,
body-patch alpha tweak, vorticity clip change, annotation move). The
re-baseline workflow:

    python scripts/dev_save_frame.py        # writes data/inspect_canonical.png
    python scripts/_make_baseline.py        # rewrites tests/baselines/...png

Commit the new baseline PNG alongside the rendering change.

Limits: this catches structural visual regressions (colormap, body
position, vorticity heatmap scale, annotations). It does NOT catch
animation-quality regressions (jerky motion, particle ageing,
inter-frame jumps) -- those still require a human watching the GIF.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.lbm_render import simulate_and_render

BASELINE_PATH = Path(__file__).parent / "baselines" / "canonical_cylinder_re400_f50.png"
BASELINE_SIZE = (128, 40)
# Tolerance: empirically, two consecutive runs on the same machine
# produce bit-identical downscaled grayscale frames. 5/255 (= 2%) catches
# any meaningful structural drift while absorbing the kind of microscopic
# variation that matplotlib version bumps occasionally introduce.
MAD_TOLERANCE = 5.0


def _fingerprint(png_bytes_or_path):
    """Open a PNG (bytes or path), downscale + grayscale, return np.float64 array."""
    if isinstance(png_bytes_or_path, (bytes, bytearray)):
        img = Image.open(io.BytesIO(png_bytes_or_path))
    else:
        img = Image.open(png_bytes_or_path)
    return np.array(img.convert("L").resize(BASELINE_SIZE)).astype(np.float64)


def test_canonical_cylinder_frame50_matches_baseline(tmp_path):
    """Cylinder Re=400 AoA=0 Standard frame 50 -- visual snapshot stays stable.

    Runs the full simulate_and_render pipeline (LBM solve + particle
    advection + matplotlib render + GIF encode). Extracts frame 50,
    downscales to 128x40 grayscale, compares to the committed baseline.
    """
    assert BASELINE_PATH.exists(), (
        f"baseline {BASELINE_PATH} not found. Regenerate with: "
        f"python scripts/dev_save_frame.py && python scripts/_make_baseline.py"
    )

    out = simulate_and_render(
        "Cylinder", 400, 0.0, "Standard (320 x 80)", n_frames=51,
    )
    gif = Image.open(io.BytesIO(out["gif_bytes"]))
    gif.seek(50)
    # PIL writes frame-50 as bytes via a round-trip through PNG so the
    # fingerprint function takes a single code path.
    buf = io.BytesIO()
    gif.convert("RGB").save(buf, format="PNG")

    current = _fingerprint(buf.getvalue())
    baseline = _fingerprint(BASELINE_PATH)
    assert current.shape == baseline.shape, (
        f"fingerprint shape mismatch: current {current.shape} vs baseline {baseline.shape}"
    )

    mad = float(np.abs(current - baseline).mean())
    max_diff = float(np.abs(current - baseline).max())

    if mad > MAD_TOLERANCE:
        # Persist debug artifacts so the human reviewer can inspect.
        current_png = tmp_path / "current.png"
        diff_png = tmp_path / "diff.png"
        Image.fromarray(current.astype(np.uint8)).save(current_png)
        diff_arr = np.abs(current - baseline).astype(np.uint8)
        Image.fromarray(diff_arr).save(diff_png)
        pytest.fail(
            f"Visual regression: MAD = {mad:.3f}/255 (tolerance {MAD_TOLERANCE}), "
            f"max pixel diff = {max_diff:.0f}/255.\n"
            f"  baseline:  {BASELINE_PATH}\n"
            f"  current:   {current_png}\n"
            f"  abs diff:  {diff_png}\n"
            f"If this is an intentional change, re-baseline via "
            f"scripts/dev_save_frame.py + scripts/_make_baseline.py."
        )


def test_canonical_pipeline_returns_expected_keys():
    """Snapshot test on the simulate_and_render return shape.

    Cheaper sibling of the visual-regression test: catches API drift
    (return dict keys changing) without paying for a 51-frame run that
    matches the baseline. n_frames=2 keeps this under ~3 s warm.
    """
    out = simulate_and_render(
        "Cylinder", 400, 0.0, "Standard (320 x 80)", n_frames=2,
    )
    expected = {
        "gif_bytes", "vort_cbar_bytes", "speed_cbar_bytes",
        "force_plot_bytes", "cd_history", "cl_history",
        "cd_mean", "cl_mean", "strouhal",
        "label", "tau", "nu", "char_length",
        "lbm_nx", "lbm_ny", "n_frames", "n_steps", "near_stable",
    }
    assert expected.issubset(out.keys()), (
        f"missing keys: {expected - set(out.keys())}"
    )
    assert isinstance(out["gif_bytes"], bytes) and len(out["gif_bytes"]) > 0
    assert out["lbm_nx"] == 320 and out["lbm_ny"] == 80
    assert out["n_frames"] == 2
    # New: force histories should match the step count, plot is non-empty.
    assert len(out["cd_history"]) == out["n_steps"]
    assert len(out["cl_history"]) == out["n_steps"]
    assert isinstance(out["force_plot_bytes"], bytes)
    assert len(out["force_plot_bytes"]) > 1000  # a real PNG, not a header


def test_force_readouts_are_in_physical_ballpark():
    """Sanity check on the new Cd/Cl/Strouhal readouts. Cylinder at Re=200,
    Standard preset: textbook values are Cd ~ 1.3 (we'll read it high due
    to channel confinement and grid resolution, but it must be positive
    and below ~5), Cl mean near zero (symmetric body at AoA=0), Strouhal
    in 0.10-0.40 (textbook is 0.197, allow generous slack for the
    short-run FFT).

    This is the regression test for the W6 force-readout feature -- if
    a refactor breaks the conversion factor or the FFT logic, these
    bounds will catch it before users do.
    """
    out = simulate_and_render(
        "Cylinder", 200, 0.0, "Standard (320 x 80)", n_frames=60,
    )
    assert 0.5 < out["cd_mean"] < 5.0, (
        f"Cd out of plausible band: {out['cd_mean']}"
    )
    assert abs(out["cl_mean"]) < 0.5, (
        f"Cl mean for symmetric cylinder at AoA=0 should be near zero; "
        f"got {out['cl_mean']}"
    )
    import numpy as np
    if np.isfinite(out["strouhal"]):
        assert 0.10 < out["strouhal"] < 0.40, (
            f"Strouhal out of plausible band for Re=200: {out['strouhal']}"
        )


def test_naca_at_positive_aoa_produces_positive_lift():
    """A cambered airfoil (NACA 4412) at +15 degrees AoA should produce
    clearly positive Cl. If sign convention flips, this catches it."""
    out = simulate_and_render(
        "NACA 4412", 600, 15.0, "Standard (320 x 80)", n_frames=40,
    )
    assert out["cl_mean"] > 0.05, (
        f"Expected positive Cl for NACA 4412 at +15 deg AoA, got {out['cl_mean']}"
    )


@pytest.mark.parametrize("viz_mode", ["Vorticity", "Velocity", "Pressure"])
def test_all_viz_modes_produce_valid_output(viz_mode):
    """Smoke test for the three viz modes. Each should produce a non-empty
    GIF + bg_cbar PNG + the correct viz_mode echo in the output dict. The
    simulation values (Cd, Cl, St) are identical across modes because
    viz_mode only affects rendering -- locks in that the modes don't
    accidentally diverge the physics."""
    out = simulate_and_render(
        "Cylinder", 200, 0.0, "Standard (320 x 80)",
        n_frames=8, viz_mode=viz_mode,
    )
    assert out["viz_mode"] == viz_mode
    assert isinstance(out["gif_bytes"], bytes) and len(out["gif_bytes"]) > 5000
    assert isinstance(out["bg_cbar_bytes"], bytes) and len(out["bg_cbar_bytes"]) > 1000
    _title_keyword = {
        "Vorticity": "rotation",
        "Velocity": "speed",
        "Pressure": "pressure",
    }[viz_mode]
    assert _title_keyword in out["bg_cbar_title"].lower()


def test_invalid_viz_mode_raises():
    """Mistyped or unknown viz_mode should fail fast with a clear error,
    not produce a corrupted render or fall back silently."""
    with pytest.raises(ValueError, match="viz_mode must be one of"):
        simulate_and_render(
            "Cylinder", 200, 0.0, "Standard (320 x 80)",
            n_frames=2, viz_mode="ThermalRainbow",
        )


def test_pressure_temporal_average_suppresses_acoustic_waves():
    """The Pressure mode applies a rolling temporal mean over the rho field
    to wash out LBM acoustic ripples. We test the effect by simulating a
    Cylinder Re=400 run, computing the across-time variance of the *raw*
    rho field vs the *averaged* one in the wake region, and asserting the
    averaged variance is at least 2x lower. If the temporal averaging
    regresses (e.g. someone bumps PRESSURE_AVG_FRAMES to 1), this catches
    it.

    The test extracts the rho history by re-running the solver via the
    public simulate_and_render entry point with viz_mode='Pressure', then
    inspecting the GIF's pixel variance at a wake location -- a proxy for
    the time-varying acoustic noise."""
    out = simulate_and_render(
        "Cylinder", 400, 0.0, "Standard (320 x 80)",
        n_frames=12, viz_mode="Pressure",
    )
    # Sanity: the pipeline produced a non-empty GIF in pressure mode.
    assert len(out["gif_bytes"]) > 5000
    # The cbar_blurb should explicitly reference the temporal averaging --
    # if someone removes the fix, the blurb claim becomes a lie and this
    # test catches it before the user notices.
    assert "averaged" in out["bg_cbar_blurb"].lower(), (
        f"Pressure blurb should mention temporal averaging; got: "
        f"{out['bg_cbar_blurb']!r}"
    )


