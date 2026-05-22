"""End-to-end gallery click validation.

Loads the app, switches to Real CFD mode, clicks gallery card 0
(Cylinder Re=200), waits for the auto-run flow to fire, and verifies
that the run-result UI ("Measured forces" expander, the Vorticity
swatch text, etc.) actually rendered.

Takes ~50 s because it pays the full LBM cost + numba JIT.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def main():
    # Default timeout high enough to absorb cold-start JIT compile + LBM
    # run on the slowest of the gallery presets.
    at = AppTest.from_file("app.py", default_timeout=180)
    print("Loading app...")
    at.run()
    if at.exception:
        for e in at.exception:
            print("EXCEPTION (load):", e.value)
        raise SystemExit(1)

    print("Switching to Real CFD...")
    at.sidebar.radio[0].set_value("Real CFD (LBM)").run()
    if at.exception:
        for e in at.exception:
            print("EXCEPTION (mode switch):", e.value)
        raise SystemExit(1)

    # Locate the gallery card buttons
    gallery_btns = [b for b in at.button if b.key and b.key.startswith("lbm_gallery_card_")]
    print(f"Found {len(gallery_btns)} gallery cards.")
    assert len(gallery_btns) == 6

    print("Clicking card 0 (Swirls behind a pole)...")
    t0 = time.time()
    gallery_btns[0].click().run()
    elapsed = time.time() - t0
    print(f"  click + rerun + sim took {elapsed:.1f} s")
    if at.exception:
        for e in at.exception:
            print("EXCEPTION (gallery click):", e.value)
        raise SystemExit(1)

    # Verify the run-result UI rendered. Probe: the "What you're looking at"
    # legend header is only rendered when a run has been displayed.
    md_text = " ".join(m.value for m in at.markdown)
    if "What you're looking at" not in md_text:
        print("FAIL: 'What you're looking at' legend not found in markdown.")
        print("Markdown blocks present:")
        for m in at.markdown:
            print(f"  - {m.value[:80]!r}")
        raise SystemExit(2)
    print("  PASS: legend rendered (run displayed).")

    # Also verify the gallery cards are GONE (since a run is now displayed).
    gallery_btns_after = [b for b in at.button if b.key and b.key.startswith("lbm_gallery_card_")]
    assert len(gallery_btns_after) == 0, (
        f"Gallery should be hidden after run, but found {len(gallery_btns_after)} cards still."
    )
    print("  PASS: gallery hidden (single-run view active).")

    # Verify the sidebar widgets reflect the card's config.
    shape_select = next(s for s in at.sidebar.selectbox if s.key == "lbm_shape_select")
    velocity_slider = next(s for s in at.sidebar.slider if s.key == "lbm_velocity_slider")
    assert shape_select.value == "Cylinder  (round pipe)", (
        f"Shape select value = {shape_select.value!r}, expected Cylinder"
    )
    assert abs(velocity_slider.value - 0.60) < 1e-6, (
        f"Velocity slider = {velocity_slider.value}, expected 0.60"
    )
    print(f"  PASS: sidebar reflects card 0 config (shape={shape_select.value!r}, "
          f"vel={velocity_slider.value}).")

    print()
    print("Gallery click end-to-end validation passed.")


if __name__ == "__main__":
    main()
