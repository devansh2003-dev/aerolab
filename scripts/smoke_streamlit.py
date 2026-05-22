"""Programmatic smoke test for app.py using Streamlit's AppTest harness.

Catches:
  - NameErrors / undefined-variable bugs from edits
  - Stale session_state references
  - Widget initialization issues
  - Any unhandled exception during the first render

We run the app twice: once in Fast mode (default), once in Real CFD mode.
We do NOT click Run -- the goal is to surface render-time bugs, not pay
the ~30s LBM cost.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

print("=== Smoke test 1: Fast mode (default) ===")
at = AppTest.from_file("app.py", default_timeout=60)
at.run()
if at.exception:
    for e in at.exception:
        print("EXCEPTION:", e.value)
    raise SystemExit(1)
print(f"  Fast mode rendered OK. {len(at.markdown)} markdown blocks, "
      f"{len(at.button)} buttons, {len(at.slider)} sliders.")

print()
print("=== Smoke test 2: Real CFD mode (no Run click) ===")
at2 = AppTest.from_file("app.py", default_timeout=60)
at2.run()
# Flip the mode radio to Real CFD
mode_radio = at2.sidebar.radio[0]
mode_radio.set_value("Real CFD (LBM)").run()
if at2.exception:
    for e in at2.exception:
        print("EXCEPTION:", e.value)
    raise SystemExit(1)
print(f"  Real CFD mode rendered OK. {len(at2.markdown)} markdown blocks, "
      f"{len(at2.button)} buttons, {len(at2.slider)} sliders.")

# Find the gallery buttons. They have keys lbm_gallery_card_0 .. _5.
_gallery_btns = [b for b in at2.button if b.key and b.key.startswith("lbm_gallery_card_")]
print(f"  Gallery cards present: {len(_gallery_btns)} buttons.")
assert len(_gallery_btns) == 6, f"expected 6 gallery cards, got {len(_gallery_btns)}"

# Don't click them in the smoke test (would trigger a 40 s LBM run).
# We verify the auto-run wiring separately in scripts/smoke_gallery_click.py.

print()
print("All smoke checks passed.")
