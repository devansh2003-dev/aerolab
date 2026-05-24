"""Canonical 2D bluff-body reference data + blockage-correction helpers.

Extracted from app.py during the D1 split (external review 2026-05-24).
Pure data and pure functions only -- no Streamlit dependency, so this
module can be unit-tested in isolation (`pytest tests/test_references.py`).

Two reference tables per shape:

  *_REFERENCE_*   -- expected output of OUR solver in its 33-35 % blocked
                     channel. Used by the legacy "vs textbook" delta chip
                     for a sanity check (kept for backward compat with
                     test_lbm_render and a couple of older callers).
  *_FREESTREAM_*  -- true unbounded-flow values from the canonical
                     experiments (Williamson 1996 / Norberg 1994 for
                     cylinder; Okajima 1982 / Sohankar 1998 for square).
                     This is what the Forces panel compares the
                     blockage-corrected Cd / Strouhal against.

The gap between the two is the channel-blockage inflation that the
Allen-Vincenti / West-Apelt corrections in `blockage_corrected` are
designed to remove.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Cylinder (round, broadside) -- Williamson 1996 + Norberg 1994
# ---------------------------------------------------------------------------

CYLINDER_REFERENCE_CD = {
    40: 1.55, 80: 1.40, 100: 1.40, 150: 1.32, 200: 1.30,
    300: 1.35, 500: 1.40, 800: 1.41, 1000: 1.40, 1500: 1.42,
}
CYLINDER_FREESTREAM_CD = {
    40: 1.55, 80: 1.38, 100: 1.32, 150: 1.20, 200: 1.15,
    300: 1.08, 500: 1.02, 800: 1.00, 1000: 0.99, 1500: 1.00,
}
CYLINDER_REFERENCE_ST = {
    80: 0.155, 100: 0.165, 150: 0.180, 200: 0.197, 300: 0.207,
    500: 0.215, 800: 0.215, 1000: 0.21, 1500: 0.21,
}
CYLINDER_FREESTREAM_ST = {
    # Williamson 1989: St vs Re, asymptotes near 0.21 above Re~400.
    80: 0.155, 100: 0.166, 150: 0.182, 200: 0.197, 300: 0.205,
    500: 0.207, 800: 0.210, 1000: 0.210, 1500: 0.210,
}


# ---------------------------------------------------------------------------
# Square (sharp-cornered, broadside) -- Okajima 1982 + Sohankar 1998
# ---------------------------------------------------------------------------

SQUARE_REFERENCE_CD = {
    80: 1.55, 100: 1.50, 150: 1.50, 200: 1.50, 300: 1.65,
    500: 1.95, 800: 2.05, 1000: 2.10, 1500: 2.15,
}
SQUARE_FREESTREAM_CD = {
    80: 1.50, 100: 1.50, 150: 1.55, 200: 1.60, 300: 1.85,
    500: 2.00, 800: 2.10, 1000: 2.15, 1500: 2.20,
}
SQUARE_REFERENCE_ST = {
    80: 0.130, 100: 0.135, 150: 0.140, 200: 0.143, 300: 0.140,
    500: 0.135, 800: 0.130, 1000: 0.128, 1500: 0.125,
}
SQUARE_FREESTREAM_ST = {
    80: 0.140, 100: 0.143, 150: 0.146, 200: 0.148, 300: 0.142,
    500: 0.135, 800: 0.130, 1000: 0.128, 1500: 0.125,
}


# ---------------------------------------------------------------------------
# Allen-Vincenti / Pope-Harper shape constants. Match
# scripts/validate_solver.py exactly. K is in [0.5, 1.5] per
# Barlow-Rae-Pope; diamond uses the square value since 2-D blockage
# acceleration depends on the maximum lateral extent, not corner
# orientation.
# ---------------------------------------------------------------------------

ALLEN_VINCENTI_K = {"Cylinder": 1.10, "Square": 1.00}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def interp_or_none(re_value: float, table: dict):
    """Linear-interpolated table lookup, returns None if out of range."""
    if not table:
        return None
    lo, hi = min(table), max(table)
    if re_value < lo or re_value > hi:
        return None
    keys = sorted(table)
    vals = [table[k] for k in keys]
    return float(np.interp(re_value, keys, vals))


def textbook_reference(shape_preset: str, re_value: int):
    """Linear-interpolated channel-tuned (Cd, St) reference for the
    Cylinder / Square benchmark cases.

    Returns (None, None) for shapes we don't ship a reference for, or
    for Re outside the validated band. The Square table is for
    broadside flow only -- the caller is responsible for only invoking
    this when the body presents a flat face (aoa ~ 0).
    """
    if shape_preset == "Cylinder":
        cd_ref = interp_or_none(re_value, CYLINDER_REFERENCE_CD)
        st_ref = interp_or_none(re_value, CYLINDER_REFERENCE_ST)
    elif shape_preset == "Square":
        cd_ref = interp_or_none(re_value, SQUARE_REFERENCE_CD)
        st_ref = interp_or_none(re_value, SQUARE_REFERENCE_ST)
    else:
        cd_ref, st_ref = None, None
    return cd_ref, st_ref


def freestream_reference(shape_preset: str, re_value: int):
    """True unbounded-flow (Cd, St) from Williamson / Norberg / Okajima.

    Distinct from `textbook_reference`, which is calibrated to our
    33-35 % blocked-channel solver: that one answers "is the simulation
    behaving as expected?", while this one answers "what would you
    measure in a wind tunnel?". Returns (None, None) outside the
    validated band.
    """
    if shape_preset == "Cylinder":
        cd_free = interp_or_none(re_value, CYLINDER_FREESTREAM_CD)
        st_free = interp_or_none(re_value, CYLINDER_FREESTREAM_ST)
    elif shape_preset == "Square":
        cd_free = interp_or_none(re_value, SQUARE_FREESTREAM_CD)
        st_free = interp_or_none(re_value, SQUARE_FREESTREAM_ST)
    else:
        cd_free, st_free = None, None
    return cd_free, st_free


def blockage_corrected(shape_preset: str, cd_raw: float, st_raw: float,
                       char_length: float, lbm_ny: int):
    """Apply Allen-Vincenti (Cd) + West-Apelt (St) blockage corrections.

    Returns (cd_corr, st_corr, blockage_ratio, K).

      cd_corr = cd_raw * (1 - K * B)^2        (Allen-Vincenti / Pope-Harper)
      st_corr = st_raw / (1 + 2 B + B^2)      (West & Apelt 1982, JFM 114)

    Returns (None, None, B, None) for shapes we don't ship a K for
    (e.g. Custom drawn polygons -- no blockage-correction reference
    exists in the literature for arbitrary 2-D outlines).
    """
    B = float(char_length) / max(float(lbm_ny), 1.0)
    K = ALLEN_VINCENTI_K.get(shape_preset)
    if K is None:
        return None, None, B, None
    cd_corr = cd_raw * (1.0 - K * B) ** 2
    st_corr = (
        st_raw / (1.0 + 2.0 * B + B * B)
        if np.isfinite(st_raw) else float("nan")
    )
    return cd_corr, st_corr, B, K
