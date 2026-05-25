"""Single-source-of-truth gate for headline validation numbers.

The headline anchor moved twice during senior CFD review:

  Round 1 (2026-05-26): demoted the 35 %-blockage Standard preset
  from headline to transparency table, because the 2.6 x
  Allen-Vincenti rescale was absorbing solver error along with
  blockage. Headline moved to the low-blockage Validation preset
  (D = 20, B = 5 %).

  Round 2 (2026-05-27): the Resolved sweep (D = 40, B = 10 %) showed
  that the Validation cylinder Re = 200 number (+13.8 %) was
  grid-limited rather than solver-limited; at the literature-grade
  D = 40 it lands at +1.0 %. The Resolved sweep also exposed that
  the K = 1.00 square AV correction over-corrects at low blockage,
  so the square headline now uses the RAW measurement (no
  correction applied). Headline moved to results_resolved.json with
  the square row gated against raw error.

This gate prevents drift between three sources of truth:

  * data/validation/results_resolved.json -- the headline source
    of truth (D = 40, B = 10 %).
  * data/validation/results.json -- the Standard preset full sweep
    (kept as the transparency table in VALIDATION.md section 3.6).
  * README.md + VALIDATION.md headline tables and section 3.6
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

RESULTS_JSON           = ROOT / "data" / "validation" / "results.json"
RESULTS_RESOLVED_JSON  = ROOT / "data" / "validation" / "results_resolved.json"
README     = ROOT / "README.md"
VALIDATION = ROOT / "VALIDATION.md"


# Validated band per shape, set by physics (Williamson mode-A 3D
# transition at Re ~ 190) and by literature (Mei-Luo-Shyy 1999 D >= 40
# guideline, which the Resolved preset meets). See VALIDATION.md
# section 2.4 for the full justification.
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

# Headline conventions:
#   Cylinder -> the AV-corrected Cd at the Resolved preset (correction
#               factor 0.79, modest, consistent with the K=1.10
#               Mei-Luo-Shyy fit).
#   Square   -> the RAW Cd at the Resolved preset. The K=1.00 square
#               correction over-corrects at low blockage (VALIDATION.md
#               section 3.2); the raw measurement IS the solver
#               result and is inside Sohankar's 5 % bar without any
#               correction applied.
HEADLINE_ERROR_KIND = {
    "Cylinder": "corrected",
    "Square":   "raw",
}


def _abs_errs(results, shape, kind, re_min, re_max):
    """Sorted abs(percent error) list, restricted to [re_min, re_max].

    `kind` is 'corrected' (uses the pre-computed cd_error_pct field) or
    'raw' (computes cd_raw vs cd_ref directly).
    """
    out = []
    for r in results:
        if r["shape"] != shape or not (re_min <= r["re"] <= re_max):
            continue
        if kind == "corrected":
            v = r.get("cd_error_pct")
            if v is None:
                continue
            out.append(abs(v))
        elif kind == "raw":
            cd_raw = r.get("cd_raw")
            cd_ref = r.get("cd_ref")
            if cd_raw is None or cd_ref is None or cd_ref == 0:
                continue
            out.append(abs(100.0 * (cd_raw - cd_ref) / cd_ref))
        else:
            raise ValueError(f"unknown kind {kind!r}; expected 'corrected' or 'raw'")
    return sorted(out)


def _stats(errs):
    if not errs:
        return float("nan"), float("nan")
    return float(median(errs)), float(max(errs))


def _round1(x):
    return round(x, 1)


def _aggregate_headline(json_path):
    """Headline aggregate: cylinder uses corrected, square uses raw."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]
    out = {}
    for shape, (re_min, re_max) in HEADLINE_BAND.items():
        kind = HEADLINE_ERROR_KIND[shape]
        med, mx = _stats(_abs_errs(results, shape, kind, re_min, re_max))
        out[f"{shape.lower()}_cd_median"] = _round1(med)
        out[f"{shape.lower()}_cd_max"]    = _round1(mx)
    return out


def _aggregate_transparency(json_path):
    """Standard-preset transparency uses corrected Cd across the full
    documented band (Cyl 100-1000, Sq 150-500)."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data["results"]
    out = {}
    for shape, (re_min, re_max) in TRANSPARENCY_BAND.items():
        med, mx = _stats(_abs_errs(results, shape, "corrected", re_min, re_max))
        out[f"{shape.lower()}_cd_median"] = _round1(med)
        out[f"{shape.lower()}_cd_max"]    = _round1(mx)
    return out


# Headline row regex: matches a markdown table cell labelled e.g.
# "Cylinder Cd", optionally followed by qualifier cells, with the
# next two NUMERIC cells captured as (median, max). Handles both:
#
#   README/old format:  | Cylinder Cd | **8.0 %** | **13.8 %** | ... |
#   new VALIDATION:     | Cylinder Cd (corrected) | 100 - 200 | **5.6 %** | **10.2 %** | ... |
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
        f"underlying results JSON computes {computed} %. Re-run "
        f"`python scripts/validate_solver.py --resolved` and update "
        f"{where} to match, or vice versa."
    )


# ---------------------------------------------------------------------------
# Headline gates: Resolved (D = 40) data drives the headline tables
# in README and VALIDATION.md. Square uses raw error, cylinder uses
# corrected (see HEADLINE_ERROR_KIND).
# ---------------------------------------------------------------------------

def test_resolved_results_json_exists():
    assert RESULTS_RESOLVED_JSON.exists(), (
        f"{RESULTS_RESOLVED_JSON.relative_to(ROOT)} not found. Re-run "
        f"`python scripts/validate_solver.py --resolved` to regenerate it."
    )


def test_readme_headline_cylinder_cd_matches_resolved():
    stats = _aggregate_headline(RESULTS_RESOLVED_JSON)
    assert stats is not None
    declared = _find_row(README.read_text(encoding="utf-8"), "Cylinder Cd")
    _assert_match(declared[0], stats["cylinder_cd_median"], "README.md",
                  "Cylinder Cd (headline, corrected)", "median")
    _assert_match(declared[1], stats["cylinder_cd_max"], "README.md",
                  "Cylinder Cd (headline, corrected)", "max")


def test_readme_headline_square_cd_matches_resolved():
    stats = _aggregate_headline(RESULTS_RESOLVED_JSON)
    assert stats is not None
    declared = _find_row(README.read_text(encoding="utf-8"), "Square Cd")
    _assert_match(declared[0], stats["square_cd_median"], "README.md",
                  "Square Cd (headline, RAW)", "median")
    _assert_match(declared[1], stats["square_cd_max"], "README.md",
                  "Square Cd (headline, RAW)", "max")


def test_validation_headline_cylinder_cd_matches_resolved():
    stats = _aggregate_headline(RESULTS_RESOLVED_JSON)
    assert stats is not None
    declared = _find_row(VALIDATION.read_text(encoding="utf-8"), "Cylinder Cd")
    _assert_match(declared[0], stats["cylinder_cd_median"], "VALIDATION.md",
                  "Cylinder Cd (headline, corrected)", "median")
    _assert_match(declared[1], stats["cylinder_cd_max"], "VALIDATION.md",
                  "Cylinder Cd (headline, corrected)", "max")


def test_validation_headline_square_cd_matches_resolved():
    stats = _aggregate_headline(RESULTS_RESOLVED_JSON)
    assert stats is not None
    declared = _find_row(VALIDATION.read_text(encoding="utf-8"), "Square Cd")
    _assert_match(declared[0], stats["square_cd_median"], "VALIDATION.md",
                  "Square Cd (headline, RAW)", "median")
    _assert_match(declared[1], stats["square_cd_max"], "VALIDATION.md",
                  "Square Cd (headline, RAW)", "max")


# ---------------------------------------------------------------------------
# Transparency gate: VALIDATION.md section 3.6 quotes the Standard-preset
# aggregate medians ("4.3 %", "8.9 %", etc.) in prose as a transparency
# disclosure rather than as a headline validation. Those numbers must
# still match results.json so the doc doesn't drift silently if the
# Standard sweep is re-run.
# ---------------------------------------------------------------------------

def test_validation_sec36_standard_preset_matches_results_json():
    stats = _aggregate_transparency(RESULTS_JSON)
    if stats is None:
        import pytest
        pytest.skip(f"{RESULTS_JSON.relative_to(ROOT)} not present")

    text = VALIDATION.read_text(encoding="utf-8")
    m = re.search(
        r"median\s+(\d+\.\d+)\s*%\s*/\s*max\s+(\d+\.\d+)\s*%\s*for\s+cylinder\s+Cd",
        text, re.IGNORECASE,
    )
    assert m, (
        "VALIDATION.md section 3.6 missing the standard-preset transparency "
        "prose for cylinder Cd. Expected 'median X.Y % / max Z.W % for "
        "cylinder Cd'."
    )
    _assert_match(float(m.group(1)), stats["cylinder_cd_median"],
                  "VALIDATION.md section 3.6", "Cylinder Cd", "median")
    _assert_match(float(m.group(2)), stats["cylinder_cd_max"],
                  "VALIDATION.md section 3.6", "Cylinder Cd", "max")

    m = re.search(
        r"median\s+(\d+\.\d+)\s*%\s*/\s*max\s+(\d+\.\d+)\s*%\s*for\s+square\s+Cd",
        text, re.IGNORECASE,
    )
    assert m, (
        "VALIDATION.md section 3.6 missing the standard-preset transparency "
        "prose for square Cd. Expected 'median X.Y % / max Z.W % for "
        "square Cd'."
    )
    _assert_match(float(m.group(1)), stats["square_cd_median"],
                  "VALIDATION.md section 3.6", "Square Cd", "median")
    _assert_match(float(m.group(2)), stats["square_cd_max"],
                  "VALIDATION.md section 3.6", "Square Cd", "max")
