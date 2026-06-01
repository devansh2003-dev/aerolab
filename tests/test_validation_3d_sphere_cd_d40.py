"""Gate the D = 40 sphere Re = 100 measurement (audit item #8, first half).

The headline JSON is produced by `scripts/validate_3d_sphere_cd_d40.py`
and committed at `data/validation_3d_sphere_re100_d40.json`. This gate
asserts the experiment that isolated the **grid-resolution contribution**
to the +44 - 56 % Ladd-1994-on-D=20 Cd bias.

Key result locked in by this gate
---------------------------------
At D = 40 (everything else held constant vs the D = 20 lowblock case):

    Cd_D40 = 1.528  (+40.2 % vs CGW 1978)
    Cd_D20 = 1.645  (+50.9 % vs CGW 1978, see test_validation_3d_sphere_cd_lowblock.py)
    Delta  = -0.117 (-7.1 percentage points of bias)

The doubling of grid resolution removed only ~ 7 percentage points
of the bias, NOT the bulk of it. This **falsifies the working
hypothesis** that grid resolution was the dominant residual bias
and re-anchors VALIDATION.md sec 8.8 on the MYSL 2002 momentum-
exchange upgrade as the highest-impact remaining item.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = ROOT / "data" / "validation_3d_sphere_re100_d40.json"

CGW_CD_REF = 1.09
CD_TOL_PCT = 60.0          # wider than the gate could be -- the script
                            # already gates at +-80 %; this is a tighter
                            # ceiling that locks in the measurement.
CD_FLOOR_VS_D20 = 1.4      # Cd must drop measurably vs D=20 (which was 1.645)
                            # but stay > 1.4 -- a value below that means
                            # grid was the dominant bias all along, which
                            # this experiment shows it is NOT.
F_TRANSVERSE_TOL_FRAC = 0.05


def _load() -> dict:
    assert PAYLOAD_PATH.exists(), (
        f"Missing {PAYLOAD_PATH.relative_to(ROOT)} -- run "
        "`python scripts/validate_3d_sphere_cd_d40.py` "
        "(takes ~ 2.5 hours on a 4-core laptop)."
    )
    return json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))


def test_payload_shape():
    p = _load()
    for key in ("Re_target", "Re_actual", "Cd_raw", "F_drag_lattice",
                "F_lift_lattice", "F_side_lattice", "mass_drift_rel",
                "grid", "tau", "momentum_exchange"):
        assert key in p, f"Missing key '{key}' in {PAYLOAD_PATH.name}"


def test_grid_is_d40():
    """Lock in that this is the D = 40 measurement, not some other resolution."""
    p = _load()
    assert p["grid"] == [320, 160, 160], (
        f"Grid = {p['grid']}, expected [320, 160, 160]. The D=40 "
        "audit item must run at D = 40 specifically; a different "
        "grid would not isolate the resolution contribution."
    )
    assert p["Re_target"] == 100
    re_actual = float(p["Re_actual"])
    assert abs(re_actual - 100.0) < 0.5, (
        f"Re_actual = {re_actual:.2f}, expected 100.0. Solver "
        "constants drifted."
    )


def test_uses_simplified_ladd():
    """First half of #8 must NOT use MYSL -- the point is to isolate D resolution."""
    p = _load()
    assert "Ladd" in p["momentum_exchange"], (
        f"momentum_exchange = {p['momentum_exchange']!r}, expected to "
        "include 'Ladd'. The D=40 isolation experiment must use the "
        "same Ladd 1994 simplified formula as the D=20 baseline so "
        "the only changed variable is grid resolution."
    )


def test_lift_side_negligible():
    p = _load()
    drag = abs(float(p["F_drag_lattice"]))
    lift = abs(float(p["F_lift_lattice"]))
    side = abs(float(p["F_side_lattice"]))
    assert lift < F_TRANSVERSE_TOL_FRAC * drag, (
        f"|F_lift| / |F_drag| = {lift/drag:.4f}, expected < "
        f"{F_TRANSVERSE_TOL_FRAC:.2f}. Sphere axisymmetry broken."
    )
    assert side < F_TRANSVERSE_TOL_FRAC * drag, (
        f"|F_side| / |F_drag| = {side/drag:.4f}, expected < "
        f"{F_TRANSVERSE_TOL_FRAC:.2f}."
    )


def test_mass_drift_small():
    p = _load()
    drift = abs(float(p["mass_drift_rel"]))
    assert drift < 0.01, (
        f"|mass drift| = {drift*100:.2f} %, expected < 1 %."
    )


def test_cd_in_band():
    p = _load()
    cd = float(p["Cd_raw"])
    err = 100.0 * (cd - CGW_CD_REF) / CGW_CD_REF
    assert abs(err) <= CD_TOL_PCT, (
        f"Cd = {cd:.3f} -> {err:+.1f} % vs CGW {CGW_CD_REF:.3f}. "
        f"Outside the +/-{CD_TOL_PCT:.0f} % gate. Either solver "
        "regressed or the MYSL upgrade landed and is now in scope "
        "for this script."
    )


def test_cd_improves_vs_d20_but_only_partially():
    """Locks in the falsification result.

    D = 20 lowblock gave Cd = 1.645 (+50.9 %); D = 40 must give Cd <
    that (improvement is real) but Cd > 1.40 (improvement is partial
    -- grid alone is not the dominant residual bias). The measured
    value is 1.528, comfortably in the (1.40, 1.645) interval.
    """
    p = _load()
    cd = float(p["Cd_raw"])
    assert cd < 1.645, (
        f"Cd at D=40 = {cd:.3f} is NOT lower than the D=20 lowblock "
        "value of 1.645. Either grid refinement made things worse "
        "(unphysical -- investigate the solver) or the D=20 baseline "
        "moved."
    )
    assert cd > CD_FLOOR_VS_D20, (
        f"Cd at D=40 = {cd:.3f} is at or below {CD_FLOOR_VS_D20}, "
        "meaning grid refinement closed most of the gap to CGW 1.09. "
        "If THIS test fails after a future change (say, MYSL is "
        "implemented and the script is re-run), update the floor -- "
        "but at v1.6.5.1 the experimental result is Cd = 1.528, well "
        "above this floor."
    )
