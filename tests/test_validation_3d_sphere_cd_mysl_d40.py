"""Gate the D = 40 + MYSL sphere Re = 100 measurement (audit item #8, second half).

The headline JSON is produced by `scripts/validate_3d_sphere_cd_mysl_d40.py`
and committed at `data/validation_3d_sphere_re100_d40_mysl.json`. This
gate asserts the MYSL Bouzidi-aware momentum-exchange formula closed
the bulk of the +40 % Ladd-1994 Cd bias measured at D = 40 (§8.3.3).

Key result locked in by this gate
---------------------------------
At D = 40 (same Bouzidi flow as §8.3.3, different force formula):

    Cd_LADD  = 1.528  (+40.2 % vs CGW 1.090)   <- §8.3.3 baseline
    Cd_MYSL  = 1.160  (+ 6.4 % vs CGW 1.090)   <- this run
    Delta    = -24.1 % (MYSL drag is 24.1 % lower than Ladd)
    Bias reduction: 33.8 percentage points

The momentum-exchange formula was the dominant residual bias, as
the §8.3.3 falsification predicted. The remaining +6.4 % is the
target for follow-on work (D = 40 / B <= 10 % bake to separate
residual blockage, Mei-Luo-Shyy D >= 60 guideline, or refined BB).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = (
    ROOT / "data" / "validation_3d_sphere_re100_d40_mysl.json"
)

CGW_CD_REF = 1.09
CD_TOL_PCT_MYSL = 10.0           # MYSL must land within 10 % of CGW
LADD_BASELINE_CD = 1.528         # from VALIDATION.md sec 8.3.3
MYSL_VS_LADD_REDUCTION_PCT = 15.0 # MYSL drag must be at least 15 % lower
F_TRANSVERSE_TOL_FRAC = 0.05


def _load() -> dict:
    assert PAYLOAD_PATH.exists(), (
        f"Missing {PAYLOAD_PATH.relative_to(ROOT)} -- run "
        "`python scripts/validate_3d_sphere_cd_mysl_d40.py` "
        "(takes ~ 1.9 hours on a 4-core laptop)."
    )
    return json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))


def test_payload_shape():
    p = _load()
    for key in ("Re_target", "Re_actual", "ladd", "mysl",
                "delta_mysl_vs_ladd_pct", "grid"):
        assert key in p, f"Missing key '{key}' in {PAYLOAD_PATH.name}"
    for k in ("Cd_raw", "Cd_error_pct", "F_drag_lattice",
              "F_lift_lattice", "F_side_lattice", "momentum_exchange"):
        assert k in p["ladd"], f"Missing ladd.{k}"
        assert k in p["mysl"], f"Missing mysl.{k}"


def test_grid_and_re_match_d40():
    p = _load()
    assert p["grid"] == [320, 160, 160], (
        f"Grid = {p['grid']}, expected [320, 160, 160] for the D=40 "
        "MYSL companion to sec 8.3.3."
    )
    re_actual = float(p["Re_actual"])
    assert abs(re_actual - 100.0) < 0.5


def test_mysl_uses_mysl_formula():
    p = _load()
    me = p["mysl"]["momentum_exchange"]
    assert "MYSL" in me, (
        f"mysl.momentum_exchange = {me!r}, expected to include 'MYSL'. "
        "The headline gate must lock in that the MYSL block actually "
        "uses MYSL -- protects against a future refactor silently "
        "swapping the implementation."
    )


def test_ladd_baseline_matches_section_8_3_3():
    """The Ladd-on-this-run number must match the sec 8.3.3 baseline.

    Both numbers come from the same physical case (D = 40, B = 25 %,
    same Bouzidi BB) so the Ladd post-stream Cd should agree to 4 dp
    between this MYSL bake and the sec 8.3.3 standalone bake. If they
    drift, one of the two ran with a different config.
    """
    p = _load()
    cd_ladd = float(p["ladd"]["Cd_raw"])
    assert abs(cd_ladd - LADD_BASELINE_CD) < 0.005, (
        f"Ladd Cd on this MYSL bake = {cd_ladd:.4f}, but the sec 8.3.3 "
        f"standalone Ladd baseline is {LADD_BASELINE_CD:.4f}. Either "
        "the two runs used different configs, or there is a Ladd "
        "regression. Investigate."
    )


def test_mysl_drag_substantially_lower_than_ladd():
    """MYSL must lower the drag by >= 15 % vs Ladd on this same flow.

    The whole point of the MYSL upgrade: at curved walls the q-aware
    formula should give a noticeably different (and more accurate)
    force. The measured reduction is 24 % at v1.6.5.1; the gate at
    15 % gives margin without making it trivial.
    """
    p = _load()
    delta = float(p["delta_mysl_vs_ladd_pct"])  # signed: negative = lower
    assert delta < -MYSL_VS_LADD_REDUCTION_PCT, (
        f"MYSL vs Ladd delta = {delta:+.2f} %, expected < "
        f"-{MYSL_VS_LADD_REDUCTION_PCT:.0f} %. Either MYSL is "
        "silently equivalent to Ladd, or this configuration happens "
        "to be a fixed point (look at sec 8.3.3 budget)."
    )


def test_mysl_cd_within_10pct_of_cgw():
    """The headline claim: MYSL D=40 Cd within 10 % of CGW 1.09."""
    p = _load()
    cd = float(p["mysl"]["Cd_raw"])
    err = 100.0 * (cd - CGW_CD_REF) / CGW_CD_REF
    assert abs(err) <= CD_TOL_PCT_MYSL, (
        f"MYSL Cd = {cd:.4f} -> {err:+.2f} % vs CGW {CGW_CD_REF:.3f}. "
        f"Outside the +/-{CD_TOL_PCT_MYSL:.0f} % headline gate. At "
        "v1.6.5.1 the measurement was +6.44 %; a drift outside +/-10 % "
        "is a real regression or a config mismatch."
    )


def test_mysl_transverse_forces_small():
    """Sphere axisymmetry: |F_lift|, |F_side| << |F_drag| under MYSL too."""
    p = _load()
    fdr = abs(float(p["mysl"]["F_drag_lattice"]))
    fli = abs(float(p["mysl"]["F_lift_lattice"]))
    fsi = abs(float(p["mysl"]["F_side_lattice"]))
    assert fli < F_TRANSVERSE_TOL_FRAC * fdr, (
        f"|F_lift| / |F_drag| = {fli/fdr:.4f} under MYSL, expected < "
        f"{F_TRANSVERSE_TOL_FRAC:.2f}. The q-aware formula is leaking "
        "transverse force somewhere."
    )
    assert fsi < F_TRANSVERSE_TOL_FRAC * fdr


def test_bias_reduction_at_least_25pp():
    """Lock the headline narrative: MYSL drops Cd error by >= 25 percentage points."""
    p = _load()
    err_ladd = abs(float(p["ladd"]["Cd_error_pct"]))
    err_mysl = abs(float(p["mysl"]["Cd_error_pct"]))
    reduction_pp = err_ladd - err_mysl
    assert reduction_pp >= 25.0, (
        f"Bias reduction = {reduction_pp:.1f} percentage points "
        "(ladd_err - mysl_err). The v1.6.5.1 measurement is "
        "33.8 pp; a drift below 25 pp means MYSL is no longer "
        "closing the bulk of the gap."
    )
