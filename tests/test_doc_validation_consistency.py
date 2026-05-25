"""Single-source-of-truth gate for headline validation numbers.

The headline validation moved from the Standard preset (B = 0.35,
2.6 x Allen-Vincenti rescale) to the low-blockage Validation preset
(B = 0.05, near-no-op correction) after the 2026-05-26 senior CFD
review. The reviewer pointed out that a fitted correction at the
Standard size absorbs solver error along with the blockage, so the
small corrected error there is a property of the correction, not of
the solver. The new headline is anchored to results_lowblockage.json,
scoped to the laminar-shedding band where 2D physics is still a
faithful representation (Cylinder Re = 100 - 200, Square Re = 150 -
200, both below the Williamson mode-A 3D transition at Re ~ 190).

This gate prevents drift between three sources of truth:

  * data/validation/results_lowblockage.json -- the headline source
    of truth (low-blockage Validation preset).
  * data/validation/results.json -- the Standard preset full sweep
    (kept as the transparency table in VALIDATION.md section 3.5).
  * README.md + VALIDATION.md headline tables and section 3.5
    transparency table.

If any of those drift, the relevant test fails with the exact
substitution needed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_JSON            = ROOT / "data" / "validation" / "results.json"
RESULTS_LOWBLOCKAGE_JSON = ROOT / "data" / "validation" / "results_lowblockage.json"
README     = ROOT / "README.md"
VALIDATION = ROOT / "VALIDATION.md"


# Validated band per shape, set by physics (Williamson mode-A 3D
# transition at Re ~ 190) and by literature (Mei-Luo-Shyy 1999 D >= 40
# guideline, which we approach but don't meet at D = 20). See
# VALIDATION.md section 2.4 for the full justification.
HEADLINE_BAND = {
    "Cylinder": (100, 200),
    "Square":   (150, 200),
}

# Standard-preset transparency block keeps the previously-cited full-Re
# bands -- they ARE the data, just demoted from "validation result" to
# "what the correction can fit at this blockage".
TRANSPARENCY_BAND = {
    "Cylinder": (100, 1000),
    "Square":   (150, 500),
}


def _abs_errs(results, shape, key, re_min, re_max):
    """Sorted abs(percent error) list, restricted to [re_min, re_max]."""
    return sorted(
        abs(r[key]) for r in results
        if r["shape"] == shape
        and re_min <= r["re"] <= re_max
        and r.get(key) is not None
    )


def _stats(errs):
    if not errs:
        return float("nan"), float("nan")
    return float(median(errs)), float(max(errs))


def _round1(x):
    return round(x, 1)


def _aggregate(json_path, band):
    """Recompute (median, max) Cd error per shape from a results.json."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]
    out = {}
    for shape, (re_min, re_max) in band.items():
        med, mx = _stats(_abs_errs(results, shape, "cd_error_pct", re_min, re_max))
        out[f"{shape.lower()}_cd_median"] = _round1(med)
        out[f"{shape.lower()}_cd_max"]    = _round1(mx)
    return out


# Headline row regex: matches a markdown table cell labelled e.g.
# "Cylinder Cd", optionally followed by qualifier cells, with the
# next two NUMERIC cells captured as (median, max). Handles both:
#
#   README/old format:  | Cylinder Cd | **8.0 %** | **13.8 %** | ... |
#   new VALIDATION:     | Cylinder Cd (laminar shedding) | 100 - 200 | **8.0 %** | **13.8 %** | ... |
#
# The "[^%|\n]*\|" cell-skip absorbs any non-numeric intermediate
# cells (Re band, blank, etc.) without consuming the median cell
# (which contains the `%` character).
_ROW = re.compile(
    r"\|\s*{label}\s*[^|]*\|"             # label cell  (incl. any qualifier)
    r"(?:\s*[^%|\n]*\|)*"                 # zero or more non-numeric cells
    r"\s*\**\s*(\d+\.\d+)\s*%\**\s*\|"    # median %
    r"\s*\**\s*(\d+\.\d+)\s*%\**\s*\|",   # max %
)


def _find_row(text, label):
    rx = re.compile(_ROW.pattern.format(label=re.escape(label)))
    m = rx.search(text)
    assert m, (
        f"Could not parse {label!r} headline row. Expected a markdown row "
        f"like `| {label} | ... | **X.Y %** | **Z.W %** |`."
    )
    return float(m.group(1)), float(m.group(2))


def _assert_match(declared, computed, where, label, kind, tol=0.2):
    assert abs(declared - computed) <= tol, (
        f"{where} declares {label} {kind} = {declared} %, but the "
        f"underlying results.json computes {computed} %. Re-run "
        f"`python scripts/validate_solver.py [--headline]` and update "
        f"{where} to match, or vice versa."
    )


# ---------------------------------------------------------------------------
# Headline gates: low-blockage data drives the headline tables
# in README and VALIDATION.md.
# ---------------------------------------------------------------------------

def test_lowblockage_results_json_exists():
    assert RESULTS_LOWBLOCKAGE_JSON.exists(), (
        f"{RESULTS_LOWBLOCKAGE_JSON.relative_to(ROOT)} not found. Re-run "
        f"`python scripts/validate_solver.py --headline` to regenerate it."
    )


def test_readme_headline_cylinder_cd_matches_lowblockage():
    stats = _aggregate(RESULTS_LOWBLOCKAGE_JSON, HEADLINE_BAND)
    assert stats is not None
    declared = _find_row(README.read_text(encoding="utf-8"), "Cylinder Cd")
    _assert_match(declared[0], stats["cylinder_cd_median"], "README.md",
                  "Cylinder Cd (headline)", "median")
    _assert_match(declared[1], stats["cylinder_cd_max"], "README.md",
                  "Cylinder Cd (headline)", "max")


def test_readme_headline_square_cd_matches_lowblockage():
    stats = _aggregate(RESULTS_LOWBLOCKAGE_JSON, HEADLINE_BAND)
    assert stats is not None
    declared = _find_row(README.read_text(encoding="utf-8"), "Square Cd")
    _assert_match(declared[0], stats["square_cd_median"], "README.md",
                  "Square Cd (headline)", "median")
    _assert_match(declared[1], stats["square_cd_max"], "README.md",
                  "Square Cd (headline)", "max")


def test_validation_headline_cylinder_cd_matches_lowblockage():
    stats = _aggregate(RESULTS_LOWBLOCKAGE_JSON, HEADLINE_BAND)
    assert stats is not None
    declared = _find_row(VALIDATION.read_text(encoding="utf-8"), "Cylinder Cd")
    _assert_match(declared[0], stats["cylinder_cd_median"], "VALIDATION.md",
                  "Cylinder Cd (headline)", "median")
    _assert_match(declared[1], stats["cylinder_cd_max"], "VALIDATION.md",
                  "Cylinder Cd (headline)", "max")


def test_validation_headline_square_cd_matches_lowblockage():
    stats = _aggregate(RESULTS_LOWBLOCKAGE_JSON, HEADLINE_BAND)
    assert stats is not None
    declared = _find_row(VALIDATION.read_text(encoding="utf-8"), "Square Cd")
    _assert_match(declared[0], stats["square_cd_median"], "VALIDATION.md",
                  "Square Cd (headline)", "median")
    _assert_match(declared[1], stats["square_cd_max"], "VALIDATION.md",
                  "Square Cd (headline)", "max")


# ---------------------------------------------------------------------------
# Transparency gate: VALIDATION.md section 3.5 quotes the Standard-preset
# aggregate medians ("4.3 %", "8.9 %", etc.) in prose as a transparency
# disclosure rather than as a headline validation. Those numbers must
# still match results.json so the doc doesn't drift silently if the
# Standard sweep is re-run.
# ---------------------------------------------------------------------------

def test_validation_sec35_standard_preset_matches_results_json():
    stats = _aggregate(RESULTS_JSON, TRANSPARENCY_BAND)
    if stats is None:
        # results.json may not exist in a sparse checkout; skip rather
        # than fail when the cause is missing input, not drift.
        import pytest
        pytest.skip(f"{RESULTS_JSON.relative_to(ROOT)} not present")

    text = VALIDATION.read_text(encoding="utf-8")
    # Section 3.5 prose: "median 4.3 % / max 11.6 % for cylinder Cd,
    # median 8.9 % / max 21.8 % for square Cd".
    m = re.search(
        r"median\s+(\d+\.\d+)\s*%\s*/\s*max\s+(\d+\.\d+)\s*%\s*for\s+cylinder\s+Cd",
        text, re.IGNORECASE,
    )
    assert m, (
        "VALIDATION.md section 3.5 missing the standard-preset transparency "
        "prose for cylinder Cd. Expected 'median X.Y % / max Z.W % for "
        "cylinder Cd'."
    )
    _assert_match(float(m.group(1)), stats["cylinder_cd_median"],
                  "VALIDATION.md section 3.5", "Cylinder Cd", "median")
    _assert_match(float(m.group(2)), stats["cylinder_cd_max"],
                  "VALIDATION.md section 3.5", "Cylinder Cd", "max")

    m = re.search(
        r"median\s+(\d+\.\d+)\s*%\s*/\s*max\s+(\d+\.\d+)\s*%\s*for\s+square\s+Cd",
        text, re.IGNORECASE,
    )
    assert m, (
        "VALIDATION.md section 3.5 missing the standard-preset transparency "
        "prose for square Cd. Expected 'median X.Y % / max Z.W % for "
        "square Cd'."
    )
    _assert_match(float(m.group(1)), stats["square_cd_median"],
                  "VALIDATION.md section 3.5", "Square Cd", "median")
    _assert_match(float(m.group(2)), stats["square_cd_max"],
                  "VALIDATION.md section 3.5", "Square Cd", "max")
