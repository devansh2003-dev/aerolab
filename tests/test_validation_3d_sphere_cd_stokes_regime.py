"""Gate the Re=20 steady-wake sphere validation (audit item #9).

The headline JSON is produced by
`scripts/validate_3d_sphere_cd_stokes_regime.py` and committed at
`data/validation_3d_sphere_re20_stokes_regime.json`. This gate
asserts the auditor's "3D rests on more than TGV + one bluff body"
bar is met: a SECOND 3D drag measurement at a regime where the
physics is qualitatively different (steady symmetric wake, viscous-
dominated) from the shipped Re = 100 case.

What this test gates
--------------------
1. The file exists and is well-formed.
2. The recorded Re matches the target (Re = 20).
3. Lift / side forces are within axisymmetry expectation (small
   relative to drag): |F_lift|, |F_side| < 5 % of |F_drag|.
4. Mass drift is < 1 % (the Zou-He outlet shouldn't have leaked
   significant mass).
5. Cd lands in a +/-60 % band around the CGW Re=20 reference
   (Cd_ref = 2.73). Why so wide: the shipped Ladd 1994 + D=20
   baseline at Re=100 lands at +44 - 51 % already; at Re=20 the
   viscous / pressure split is different so the bias may differ.
   The gate's job is "the measurement is in the physically-
   plausible Cd range and the bias is order-of-magnitude
   consistent with Re=100," not a percent-level claim.
6. Cd is strictly > 1.5 (a sphere at Re=20 must have substantially
   higher Cd than at Re=100 -- Cd_CGW falls monotonically with Re
   in this band; any measurement < 1.5 indicates a real regression).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = ROOT / "data" / "validation_3d_sphere_re20_stokes_regime.json"

CGW_CD_REF = 2.73
# **Exploratory gate, not a validated tolerance.** The Re=20 case ships
# at the OLD Ladd-1994 + D = 20 baseline (the same configuration that
# the §8.3.4 MYSL upgrade superseded for Re=100). The +56 % bias this
# CONFIG carries means the +/-60 % band below is a "the measurement
# landed somewhere physically plausible and stayed on the documented
# side of CGW" check, not a percent-level validation gate. A
# method-consistent re-run at MYSL + D = 40 (queued as VALIDATION.md
# §8.8 item #1) would tighten this to ~ +/-15 % once measured.
CD_TOL_PCT = 60.0
CD_FLOOR = 1.5  # below this, the Cd-vs-Re monotonicity is broken
F_TRANSVERSE_TOL_FRAC = 0.05  # |F_lift|, |F_side| < 5 % of |F_drag|


def _load() -> dict:
    assert PAYLOAD_PATH.exists(), (
        f"Missing {PAYLOAD_PATH.relative_to(ROOT)} -- run "
        "`python scripts/validate_3d_sphere_cd_stokes_regime.py` "
        "(takes ~ 15 - 25 min on a 4-core laptop)."
    )
    return json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))


def test_payload_shape():
    p = _load()
    for key in ("Re_target", "Re_actual", "Cd_raw", "F_drag_lattice",
                "F_lift_lattice", "F_side_lattice", "mass_drift_rel"):
        assert key in p, f"Missing key '{key}' in {PAYLOAD_PATH.name}"


def test_re_matches_target():
    p = _load()
    assert p["Re_target"] == 20, (
        f"Re_target = {p['Re_target']}, expected 20. The audit-item-#9 "
        "Stokes-regime validation must run at Re = 20 specifically; "
        "Re = 100 is the existing case."
    )
    re_actual = float(p["Re_actual"])
    assert abs(re_actual - 20.0) < 0.5, (
        f"Re_actual = {re_actual:.2f}, expected ~ 20.0. Solver "
        "constants (u_in, nu, D) drifted from the configured Re."
    )


def test_lift_side_negligible():
    """Sphere in axial flow: lift and side forces must be ~ 0."""
    p = _load()
    drag = abs(float(p["F_drag_lattice"]))
    lift = abs(float(p["F_lift_lattice"]))
    side = abs(float(p["F_side_lattice"]))
    assert lift < F_TRANSVERSE_TOL_FRAC * drag, (
        f"|F_lift| / |F_drag| = {lift/drag:.3f}, expected < "
        f"{F_TRANSVERSE_TOL_FRAC:.2f}. Sphere axisymmetry broken; "
        "investigate boundary conditions or mesh asymmetry."
    )
    assert side < F_TRANSVERSE_TOL_FRAC * drag, (
        f"|F_side| / |F_drag| = {side/drag:.3f}, expected < "
        f"{F_TRANSVERSE_TOL_FRAC:.2f}."
    )


def test_mass_drift_small():
    p = _load()
    drift = abs(float(p["mass_drift_rel"]))
    assert drift < 0.01, (
        f"|mass drift| = {drift*100:.2f} %, expected < 1 %. The Zou-He "
        "outlet should not leak this much."
    )


def test_cd_in_band():
    p = _load()
    cd = float(p["Cd_raw"])
    err = 100.0 * (cd - CGW_CD_REF) / CGW_CD_REF
    assert abs(err) <= CD_TOL_PCT, (
        f"Cd = {cd:.3f} -> {err:+.1f} % vs CGW {CGW_CD_REF:.3f}. "
        f"Outside the +/-{CD_TOL_PCT:.0f} % gate. The shipped Re=100 "
        "baseline carries a +44 - 51 % Ladd + D=20 bias; the Re=20 "
        "case should land in the same ballpark if the bias is "
        "Re-independent."
    )


def test_cd_monotone_with_re():
    """Cd at Re=20 must be substantially higher than the Re=100 baseline.

    CGW gives Cd(Re=20) = 2.73, Cd(Re=100) = 1.09 -- a 2.5x ratio.
    Even with the +44 % bias we carry on the shipped Re=100 case
    (Cd_measured ~ 1.57), the Re=20 measurement must clear ~ 1.5.
    A value below that breaks the Cd-monotonically-decreasing-with-Re
    property that bluff-body drag obeys in this regime.
    """
    p = _load()
    cd = float(p["Cd_raw"])
    assert cd > CD_FLOOR, (
        f"Cd = {cd:.3f} at Re=20 is at or below the shipped Re=100 "
        f"measurement (1.57). The Cd-vs-Re monotonicity is broken; "
        "real regression."
    )
