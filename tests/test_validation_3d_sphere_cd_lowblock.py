"""CI gate for the LOW-BLOCKAGE sphere Re=100 3D drag validation.

Companion to ``tests/test_validation_3d_sphere_cd.py``. Reads the
JSON written by ``scripts/validate_3d_sphere_cd_lowblock.py`` (same
Re=100 sphere setup but on a 160 x 80 x 80 grid where blockage drops
from 42 % to 25 %) and gates the value against the same
Clift-Grace-Weber 1978 reference.

The two gates together let the reviewer see that the +44 % error on
the shipped bake is dominated by blockage: at B = 25 % the residual
error should be ~+15-25 % instead of ~+44 %.

Like the high-blockage test, the simulation is too expensive to
re-run on every push (~20 min); the gate is on the committed JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESULTS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "validation_3d_sphere_re100_lowblock.json"
)

CD_PHYSICAL_MIN = 0.4
CD_PHYSICAL_MAX = 3.0
EXPECTED_BLOCKAGE_PCT = 25.0  # Sanity check on the grid that was run

# C-11: hard-coded reference for independent recomputation (see sibling
# test_validation_3d_sphere_cd.py for full rationale -- the previous
# test_drag_in_clift_grace_weber_band asserted result["Cd_in_band"]
# which the JSON's own script already wrote, making the gate circular).
CGW_CD_REF = 1.09
CD_TOLERANCE_BAND = 0.7  # absolute Cd units, matches data file


@pytest.fixture(scope="module")
def result() -> dict:
    if not _RESULTS.exists():
        pytest.skip(
            f"{_RESULTS.name} not found -- regenerate via "
            "`python scripts/validate_3d_sphere_cd_lowblock.py` "
            "(takes ~20 min on a laptop CPU)."
        )
    return json.loads(_RESULTS.read_text())


def test_drag_in_clift_grace_weber_band(result):
    """C-11: recompute the verdict from Cd_raw against test-side constants
    instead of asserting result["Cd_in_band"] (circular)."""
    cd = float(result["Cd_raw"])
    assert abs(result["Cd_ref_clift_grace_weber"] - CGW_CD_REF) < 1e-9
    assert abs(result["Cd_tolerance_band"] - CD_TOLERANCE_BAND) < 1e-9
    err_abs = abs(cd - CGW_CD_REF)
    assert err_abs <= CD_TOLERANCE_BAND, (
        f"sphere Re=100 (low-blockage) Cd = {cd:.3f} is outside the "
        f"+/-{CD_TOLERANCE_BAND:.2f} band around Clift-Grace-Weber "
        f"{CGW_CD_REF:.3f} (|Cd - ref| = {err_abs:.3f}). "
        "Re-run validate_3d_sphere_cd_lowblock.py."
    )


def test_drag_is_physically_signed(result):
    F_drag = result["F_drag_lattice"]
    assert F_drag > 0.0, (
        f"low-blockage sphere drag is negative ({F_drag:+.4f}); "
        "a sign flip or formula bug."
    )


def test_drag_magnitude_is_physical(result):
    cd = result["Cd_raw"]
    assert CD_PHYSICAL_MIN < cd < CD_PHYSICAL_MAX, (
        f"low-blockage sphere Cd = {cd:.3f} is outside "
        f"[{CD_PHYSICAL_MIN}, {CD_PHYSICAL_MAX}]."
    )


def test_axisymmetric_forces_vanish(result):
    """Axisymmetric flow should drive F_lift and F_side to zero. On the
    160 x 80 x 80 grid the sphere centre at cy = cz = 40 sits between
    cells 39 and 40 (half-cell offset on an even-count axis), and the
    voxelised wall picks a slightly asymmetric ring of cells. With
    R = 10 cells across, a one-cell wall-link asymmetry is ~10 % of
    the cross-section, so the residual lift can creep up to ~5-7 %
    of drag here even though the underlying physics is axisymmetric.
    Tolerance is loosened to 8 % to absorb that, vs 5 % on the high-
    blockage test where the centre at cy = cz = 24 with R = 10 has a
    cleaner half-cell offset and the residual was 1.8 %."""
    F_drag = result["F_drag_lattice"]
    F_lift = result["F_lift_lattice"]
    F_side = result["F_side_lattice"]
    assert abs(F_lift) / abs(F_drag) < 0.08, (
        f"|F_lift| / |F_drag| = {abs(F_lift)/abs(F_drag):.3%}"
    )
    assert abs(F_side) / abs(F_drag) < 0.08, (
        f"|F_side| / |F_drag| = {abs(F_side)/abs(F_drag):.3%}"
    )


def test_mass_drift_bounded(result):
    drift = abs(result["mass_drift_rel"])
    assert drift < 0.01, (
        f"|mass drift| = {drift:.2%} exceeds the 1 % budget."
    )


def test_advective_times_settled(result):
    adv = result["advective_times"]
    assert adv >= 4.0, (
        f"advective times {adv:.1f} D/u below the 4 D/u settling rule."
    )


def test_blockage_is_lower_than_shipped_bake(result):
    """The whole point: blockage on this run must actually be lower
    than the 42 % shipped sphere_re100 bake. Guards against someone
    re-running with the wrong grid."""
    blk = result["blockage_pct"]
    assert blk < 30.0, (
        f"blockage = {blk:.1f} % is not meaningfully lower than the "
        "42 % shipped bake; check the CONFIG grid in "
        "scripts/validate_3d_sphere_cd_lowblock.py."
    )
    assert abs(blk - EXPECTED_BLOCKAGE_PCT) < 1.0, (
        f"blockage = {blk:.1f} % does not match the configured "
        f"{EXPECTED_BLOCKAGE_PCT:.0f} % grid; CONFIG drift."
    )


def test_blockage_is_not_dominant_bias(result):
    """Halving blockage (42 % -> 25 %) was the experiment proposed in
    VALIDATION.md to verify that the +44 % Cd gap at B = 42 % was
    blockage-driven. The measurement shows the two Cd values land
    within ~5 % of each other -- blockage is NOT the dominant bias,
    so the error budget in §8.3 points at grid + momentum exchange
    instead. This gate locks that conclusion in: if a future change
    pushes the two Cd values apart, the budget needs to be re-derived
    from scratch."""
    high_block_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "validation_3d_sphere_re100.json"
    )
    if not high_block_path.exists():
        pytest.skip("high-blockage reference JSON missing")
    high = json.loads(high_block_path.read_text())
    rel_diff = abs(result["Cd_raw"] - high["Cd_raw"]) / high["Cd_raw"]
    # 15 % envelope. The measured spread is 4.6 % (1.572 -> 1.645).
    # If a re-run lands outside the envelope, blockage might be more
    # important than we currently think and §8.3 needs revisiting.
    assert rel_diff < 0.15, (
        f"Cd_high = {high['Cd_raw']:.3f}, Cd_low = {result['Cd_raw']:.3f}: "
        f"relative difference {rel_diff:.1%} is larger than the 15 % "
        "envelope -- blockage may be more dominant than §8.3 claims, "
        "revisit the error budget."
    )
