"""Tests for the airfoil input-validation hardening.

External review 2026-05-24 surfaced a P0 crash: typing `banana, 9999` in
the Fast mode airfoil field crashed the whole app with a traceback that
leaked the deploy path. The crash root-caused to two layers:

  1. `normalize_naca('banana')` returned 'nacabanana' unconditionally.
  2. `asb.Airfoil('nacabanana')` silently returned coordinates=None,
     which then crashed downstream `coordinates[:, 0]`.

These tests lock down both layers so a regression can't re-introduce
either failure mode. (`normalize_naca` lives in src/airfoils.py as of
the D1 split; previously it was inside app.py and had to be ast-parsed
out, since importing app.py runs the whole Streamlit UI top-level.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.airfoils import get_airfoil, normalize_naca  # noqa: E402

# ---------------------------------------------------------------------------
# normalize_naca: accepted shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("naca4412", "naca4412"),
    ("4412", "naca4412"),
    ("NACA 4412", "naca4412"),
    ("  Naca  4412 ", "naca4412"),
    ("23012", "naca23012"),
    ("naca23012", "naca23012"),
])
def test_normalize_naca_accepts_valid_codes(raw, expected):
    assert normalize_naca(raw) == expected


# ---------------------------------------------------------------------------
# normalize_naca: rejected shapes (the regression gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "banana",
    "9999.5",
    "12",
    "123",
    "123456",
    "naca-bad",
    "",
    "   ",
    "naca",
    "abc4412",
])
def test_normalize_naca_rejects_garbage(bad):
    with pytest.raises(ValueError, match="not a valid NACA code"):
        normalize_naca(bad)


# ---------------------------------------------------------------------------
# get_airfoil: defence in depth (anything that gets through normalize_naca
# still has to survive AeroSandbox returning coordinates=None)
# ---------------------------------------------------------------------------

def test_get_airfoil_accepts_valid_naca():
    af = get_airfoil("naca4412")
    assert af.coordinates is not None
    assert len(af.coordinates) > 0


def test_get_airfoil_rejects_unresolvable_string():
    """AeroSandbox accepts 'nacabanana' silently (coordinates=None); the
    hardened get_airfoil turns that into an explicit ValueError so the
    app doesn't crash downstream on coordinates[:, 0]."""
    with pytest.raises(ValueError, match="not a recognised airfoil"):
        get_airfoil("nacabanana")
