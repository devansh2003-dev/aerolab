# Validation results

Solver-output Cd, Cl, St vs published free-stream reference,
after Allen-Vincenti blockage correction (Standard preset, B = 0.350).

| Shape    | Re   | Cd raw | Cd corr | Cd ref | Cd err % | St raw | St corr | St ref | St err % | Cd pass | St pass |
|----------|------|--------|---------|--------|----------|--------|---------|--------|----------|---------|---------|
| Cylinder |  100 | 1.510 | 1.348 | 1.320 | +2.1 | 0.183 | 0.166 | 0.166 | -0.1 |   PASS  |   PASS  |
| Cylinder |  200 | 1.466 | 1.309 | 1.150 | +13.8 | 0.183 | 0.166 | 0.197 | -15.8 |   PASS  |   PASS  |
| Cylinder |  300 | 1.478 | 1.320 | 1.080 | +22.2 | 0.229 | 0.207 | 0.203 | +2.1 |  *FAIL* |   PASS  |
| Cylinder |  500 | 1.518 | 1.355 | 1.020 | +32.9 | 0.229 | 0.207 | 0.207 | +0.2 |  *FAIL* |   PASS  |
| Cylinder | 1000 | 1.521 | 1.358 | 0.990 | +37.2 | 0.229 | 0.207 | 0.210 | -1.3 |  *FAIL* |   PASS  |
| Square   |  150 | 1.681 | 1.517 | 1.550 | -2.1 | 0.183 | 0.166 | 0.146 | +13.6 |   PASS  |   PASS  |
| Square   |  200 | 1.729 | 1.560 | 1.600 | -2.5 | 0.183 | 0.166 | 0.148 | +12.1 |   PASS  |   PASS  |
| Square   |  300 | 1.852 | 1.671 | 1.850 | -9.7 | 0.183 | 0.166 | 0.142 | +16.8 |   PASS  |   PASS  |
| Square   |  500 | 1.975 | 1.782 | 2.000 | -10.9 | 0.137 | 0.124 | 0.135 | -7.9 |   PASS  |   PASS  |

### Aggregate statistics

- Cases run: 9
- Cd within +/- 25 %: **6 / 9** (median abs error 10.9 %, max 37.2 %)
- St within +/- 35 %: **9 / 9** (median abs error 7.9 %, max 16.8 %)

### Strouhal FFT diagnostics

- St-axis bin width: ~**0.038 - 0.046** (Validation preset D = 20, longer record than Standard).
- Captured cycles at the measured peak: **3 - 5** across all rows -- still below the 20-cycle "informative" threshold despite the longer record, because the shedding period scales with D/U and our D is small.
- The St columns above happen to land within +/-17 % of Williamson / Okajima, but that agreement is the FFT pinning the peak to the nearest discrete bin -- not a measurement of the solver's St(Re) curve. VALIDATION.md §3.4 has the long-form discussion.
- Per-row diagnostics live in the JSON sibling; `tests/test_validation_st.py` gates them.
