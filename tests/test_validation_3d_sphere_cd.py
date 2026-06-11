"""CI gate for the sphere Re=100 3D drag validation.

Reads the JSON written by ``scripts/validate_3d_sphere_cd.py`` (which
runs the D3Q19 TRT sphere simulation for ~5 D/u and computes Cd via
momentum exchange) and checks that the measured Cd lands in a band
around the Clift-Grace-Weber 1978 reference value.

The simulation is too expensive to re-run on every push (~250 s); the
gate is on the committed result file, identical in pattern to
``tests/test_doc_validation_consistency.py`` which gates the 2D
Williamson / Okajima numbers without re-running them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESULTS = Path(__file__).resolve().parent.parent / "data" / "validation_3d_sphere_re100.json"

# Sanity bounds. The script's tolerance band already encodes the
# Cd-in-band PASS/FAIL; these are belt-and-braces guards that catch
# accidental sign flips, mis-scaled forces, or a stale JSON.
CD_PHYSICAL_MIN = 0.4
CD_PHYSICAL_MAX = 3.0

# C-11: hard-coded reference values for an independent recomputation.
# The previous test asserted result["Cd_in_band"] -- a verdict the
# script that produced the JSON already wrote -- which is circular: a
# bug that mis-computes Cd_in_band passes the test. Pinning the
# reference + tolerance here lets the test recompute the verdict from
# Cd_raw alone, mirroring the d40 / mysl_d40 sibling tests.
CGW_CD_REF = 1.09
CD_TOLERANCE_BAND = 0.7  # absolute Cd units, matches data file


@pytest.fixture(scope="module")
def result() -> dict:
    if not _RESULTS.exists():
        pytest.skip(
            f"{_RESULTS.name} not found -- regenerate via "
            "`python scripts/validate_3d_sphere_cd.py` "
            "(takes ~250 s on a laptop CPU)."
        )
    return json.loads(_RESULTS.read_text())


def test_drag_in_clift_grace_weber_band(result):
    """Cd is within the tolerance band of CGW 1978 sphere Re=100. C-11:
    recompute the verdict from Cd_raw against test-side constants rather
    than asserting result["Cd_in_band"] (which the script that produced
    the JSON already wrote -- circular)."""
    cd = float(result["Cd_raw"])
    # Cross-check that the JSON's stored reference + band haven't drifted
    # from what this test pins -- otherwise the gate moves silently.
    assert abs(result["Cd_ref_clift_grace_weber"] - CGW_CD_REF) < 1e-9, (
        f"JSON reference {result['Cd_ref_clift_grace_weber']} drifted "
        f"from pinned CGW {CGW_CD_REF}; update the constant or the JSON."
    )
    assert abs(result["Cd_tolerance_band"] - CD_TOLERANCE_BAND) < 1e-9, (
        f"JSON band {result['Cd_tolerance_band']} drifted from pinned "
        f"{CD_TOLERANCE_BAND}; update the constant or the JSON."
    )
    err_abs = abs(cd - CGW_CD_REF)
    assert err_abs <= CD_TOLERANCE_BAND, (
        f"sphere Re=100 Cd = {cd:.3f} is outside the +/-{CD_TOLERANCE_BAND:.2f} "
        f"band around Clift-Grace-Weber {CGW_CD_REF:.3f} "
        f"(|Cd - ref| = {err_abs:.3f}). Re-bake or re-run "
        f"validate_3d_sphere_cd.py."
    )


def test_drag_is_physically_signed(result):
    """The drag force points downstream (+x) and is finite."""
    F_drag = result["F_drag_lattice"]
    assert F_drag > 0.0, (
        f"sphere Re=100 drag force is negative ({F_drag:+.4f}), "
        "i.e. pointing upstream -- a sign flip or formula bug."
    )


def test_drag_magnitude_is_physical(result):
    """Cd is within the order-of-magnitude band a sphere should live in."""
    cd = result["Cd_raw"]
    assert CD_PHYSICAL_MIN < cd < CD_PHYSICAL_MAX, (
        f"sphere Re=100 Cd = {cd:.3f} is outside the broad physical "
        f"envelope [{CD_PHYSICAL_MIN}, {CD_PHYSICAL_MAX}] -- check the "
        "force scaling or the momentum-exchange formula."
    )


def test_axisymmetric_forces_vanish(result):
    """For a sphere in axisymmetric flow, F_y and F_z should be tiny."""
    F_drag = result["F_drag_lattice"]
    F_lift = result["F_lift_lattice"]
    F_side = result["F_side_lattice"]
    assert abs(F_lift) / abs(F_drag) < 0.05, (
        f"|F_lift| / |F_drag| = {abs(F_lift)/abs(F_drag):.3%} -- "
        "axisymmetry should drive this below 5 %."
    )
    assert abs(F_side) / abs(F_drag) < 0.05, (
        f"|F_side| / |F_drag| = {abs(F_side)/abs(F_drag):.3%} -- "
        "axisymmetry should drive this below 5 %."
    )


def test_mass_drift_bounded(result):
    """Mass drift over the validation run stays small."""
    drift = abs(result["mass_drift_rel"])
    assert drift < 0.01, (
        f"|mass drift| = {drift:.2%} exceeds the 1 % budget "
        "-- the regularised outflow may be misbehaving."
    )


def test_advective_times_settled(result):
    """The run is long enough to be past the startup transient."""
    adv = result["advective_times"]
    assert adv >= 4.0, (
        f"advective times {adv:.1f} D/u is below the 4 D/u settling "
        "rule of thumb -- the wake hasn't stabilised."
    )
