"""Strouhal diagnostics are present on every validation row (card #5).

David Artemyev's 2026-05-27 review flagged that short FFT records produce
wide bins and falsely precise error percentages. The fix is to surface
the actual record length, bin width, and captured cycle count so a
reader can decide whether the displayed St is a measurement or a
coincidence. This test pins those fields to the committed JSON so they
cannot silently disappear.

Three properties are checked:

1. Every row that reports a finite ``st_raw`` also reports
   ``strouhal_bin_width``, ``strouhal_n_cycles``, ``strouhal_record_len``,
   and the boolean ``strouhal_insufficient_record`` (true iff
   n_cycles < 20).
2. The numbers are internally consistent:
   bin_width = (1 / record_len) * char_length / U_INFLOW
   n_cycles  = st_raw * U_INFLOW / char_length * record_len
3. The INSUFFICIENT_RECORD sentinel matches n_cycles < 20.

This is a drift gate, NOT a quality gate: it is fine (and currently
true) that EVERY benchmark row is flagged INSUFFICIENT_RECORD. The
point of the test is that the diagnostic information survives any
future rerun.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# Inflow speed in lattice units, matches src/lbm_render.py:U_INFLOW.
U_INFLOW = 0.1

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILES = [
    ROOT / "data" / "validation" / "results.json",
    ROOT / "data" / "validation" / "results_lowblockage.json",
    ROOT / "data" / "validation" / "results_resolved.json",
]


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["results"]


@pytest.mark.parametrize(
    "path", RESULTS_FILES, ids=lambda p: p.name,
)
def test_st_diagnostics_fields_present(path: Path):
    """Every St-bearing row must carry the four diagnostic fields."""
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout")
    for r in _load_rows(path):
        if r.get("st_raw") is None or not math.isfinite(float(r["st_raw"])):
            continue
        for key in (
            "strouhal_bin_width",
            "strouhal_n_cycles",
            "strouhal_record_len",
            "strouhal_insufficient_record",
        ):
            assert key in r, (
                f"{path.name}: row {r.get('shape')!r} Re={r.get('re')} "
                f"missing {key!r}; backfill via lbm_render or the "
                f"augmentation script."
            )


@pytest.mark.parametrize(
    "path", RESULTS_FILES, ids=lambda p: p.name,
)
def test_st_diagnostics_self_consistent(path: Path):
    """bin_width and n_cycles must match the standard formulas."""
    if not path.exists():
        pytest.skip(f"{path.name} not present in this checkout")
    for r in _load_rows(path):
        if r.get("st_raw") is None or not math.isfinite(float(r["st_raw"])):
            continue
        rec = int(r["strouhal_record_len"])
        L = float(r["char_length"])
        st_raw = float(r["st_raw"])
        expected_bw = (1.0 / rec) * L / U_INFLOW
        expected_nc = st_raw * U_INFLOW / L * rec
        assert math.isclose(
            float(r["strouhal_bin_width"]), expected_bw, rel_tol=1e-3,
        ), (
            f"{path.name}: bin_width drift for {r['shape']} Re={r['re']}: "
            f"json={r['strouhal_bin_width']}, expected={expected_bw}"
        )
        assert math.isclose(
            float(r["strouhal_n_cycles"]), expected_nc, abs_tol=0.05,
        ), (
            f"{path.name}: n_cycles drift for {r['shape']} Re={r['re']}: "
            f"json={r['strouhal_n_cycles']}, expected={expected_nc}"
        )
        assert (
            bool(r["strouhal_insufficient_record"])
            == (float(r["strouhal_n_cycles"]) < 20)
        ), (
            f"{path.name}: insufficient_record flag inconsistent for "
            f"{r['shape']} Re={r['re']}"
        )
