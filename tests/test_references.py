"""Tests for src/references.py -- the canonical 2-D bluff-body reference
data + blockage-correction helpers extracted from app.py during the D1
split (external review 2026-05-24).

Two reasons this module gets its own test file:
  1. app.py is unimportable as a unit (top-level Streamlit calls run the
     UI), so anything we want to test in isolation has to live in a
     library module like this one.
  2. The blockage-correction K factors and formulas must stay in sync
     with scripts/validate_solver.py and tests/test_validation_benchmark.py
     -- if any of those three drift, the headline numbers in
     VALIDATION.md no longer mean what they say. These tests pin the
     contract.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.references import (  # noqa: E402
    ALLEN_VINCENTI_K,
    blockage_corrected,
    freestream_reference,
    interp_or_none,
    textbook_reference,
)

# ---------------------------------------------------------------------------
# interp_or_none
# ---------------------------------------------------------------------------

def test_interp_or_none_in_range():
    table = {100: 1.0, 200: 2.0, 400: 4.0}
    assert interp_or_none(150, table) == pytest.approx(1.5)
    assert interp_or_none(300, table) == pytest.approx(3.0)


def test_interp_or_none_at_endpoints():
    table = {100: 1.0, 200: 2.0}
    assert interp_or_none(100, table) == pytest.approx(1.0)
    assert interp_or_none(200, table) == pytest.approx(2.0)


def test_interp_or_none_returns_none_below_range():
    table = {100: 1.0, 200: 2.0}
    assert interp_or_none(50, table) is None


def test_interp_or_none_returns_none_above_range():
    table = {100: 1.0, 200: 2.0}
    assert interp_or_none(500, table) is None


def test_interp_or_none_empty_table():
    assert interp_or_none(100, {}) is None


# ---------------------------------------------------------------------------
# textbook / freestream lookups
# ---------------------------------------------------------------------------

def test_textbook_reference_cylinder_known_point():
    cd, st = textbook_reference("Cylinder", 200)
    assert cd is not None and st is not None


def test_textbook_reference_unknown_shape():
    assert textbook_reference("Banana", 200) == (None, None)


def test_freestream_reference_matches_williamson_table():
    # Re=200 is a published Williamson 1996 anchor: Cd ~ 1.15, St ~ 0.197.
    cd, st = freestream_reference("Cylinder", 200)
    assert cd == pytest.approx(1.15, abs=0.01)
    assert st == pytest.approx(0.197, abs=0.01)


def test_freestream_reference_square_known_point():
    # Okajima 1982 broadside square at Re=200: Cd ~ 1.60.
    cd, st = freestream_reference("Square", 200)
    assert cd == pytest.approx(1.60, abs=0.01)


def test_freestream_reference_out_of_band():
    assert freestream_reference("Cylinder", 10) == (None, None)
    assert freestream_reference("Cylinder", 5000) == (None, None)


# ---------------------------------------------------------------------------
# blockage_corrected -- the contract that MUST match
# scripts/validate_solver.py exactly (otherwise the headline numbers
# in VALIDATION.md no longer mean what the test gate says they mean).
# ---------------------------------------------------------------------------

def test_av_k_constants_match_validate_solver():
    """If you change K here you MUST change it in
    scripts/validate_solver.py (ALLEN_VINCENTI_K) and re-run the
    validation sweep before merging. This test pins the contract."""
    assert ALLEN_VINCENTI_K == {"Cylinder": 1.10, "Square": 1.00}


def test_blockage_corrected_cylinder_standard_preset():
    """Standard preset: D=28, Ny=80 -> B=0.35. K=1.10 -> correction
    factor (1 - 1.10*0.35)^2 = 0.385129 ~ 0.385."""
    cd_corr, st_corr, B, K = blockage_corrected(
        "Cylinder", cd_raw=3.0, st_raw=0.35,
        char_length=28, lbm_ny=80,
    )
    assert B == pytest.approx(0.35, abs=0.001)
    assert K == 1.10
    assert cd_corr == pytest.approx(3.0 * (1 - 1.10 * 0.35) ** 2, abs=1e-6)
    # St correction: 0.35 / (1 + 2*0.35 + 0.35**2) = 0.35 / 1.8225
    assert st_corr == pytest.approx(0.35 / 1.8225, abs=1e-6)


def test_blockage_corrected_square_uses_k_one():
    cd_corr, _, B, K = blockage_corrected(
        "Square", cd_raw=4.0, st_raw=0.37,
        char_length=28, lbm_ny=80,
    )
    assert K == 1.00
    assert cd_corr == pytest.approx(4.0 * (1 - 1.00 * 0.35) ** 2, abs=1e-6)


def test_blockage_corrected_unknown_shape_returns_none():
    """Custom polygons have no published K factor -- the helper returns
    (None, None, B, None) so callers can fall through to raw display."""
    cd_corr, st_corr, B, K = blockage_corrected(
        "Custom", cd_raw=2.5, st_raw=0.2,
        char_length=20, lbm_ny=80,
    )
    assert cd_corr is None
    assert st_corr is None
    assert K is None
    assert B == pytest.approx(0.25, abs=0.001)


def test_blockage_corrected_propagates_nan_strouhal():
    """If the caller has no valid Strouhal (NaN) we propagate NaN
    rather than producing a meaningless 'corrected' NaN/finite mix."""
    _, st_corr, _, _ = blockage_corrected(
        "Cylinder", cd_raw=3.0, st_raw=float("nan"),
        char_length=28, lbm_ny=80,
    )
    assert math.isnan(st_corr)
