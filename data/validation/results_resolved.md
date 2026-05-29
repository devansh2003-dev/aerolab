# Validation results

Solver-output Cd, Cl, St vs published free-stream reference,
after Allen-Vincenti blockage correction (Standard preset, B = 0.350).

| Shape    | Re   | Cd raw | Cd corr | Cd ref | Cd err % | St raw | St corr | St ref | St err % | Cd pass | St pass |
|----------|------|--------|---------|--------|----------|--------|---------|--------|----------|---------|---------|
| Cylinder |  100 | 1.497 | 1.186 | 1.320 | -10.2 | 0.152 | 0.126 | 0.166 | -24.1 |   PASS  |   PASS  |
| Cylinder |  200 | 1.466 | 1.162 | 1.150 | +1.0 | 0.229 | 0.189 | 0.197 | -4.1 |   PASS  |   PASS  |
| Cylinder |  500 | 1.606 | 1.272 | 1.020 | +24.7 | 0.229 | 0.189 | 0.207 | -8.7 |  *FAIL* |   PASS  |
| Square   |  150 | 1.609 | 1.304 | 1.550 | -15.9 | 0.152 | 0.126 | 0.146 | -13.7 |   PASS  |   PASS  |
| Square   |  200 | 1.681 | 1.362 | 1.600 | -14.9 | 0.152 | 0.126 | 0.148 | -14.9 |   PASS  |   PASS  |

### Aggregate statistics

- Cases run: 5
- Cd within +/- 25 %: **4 / 5** (median abs error 14.9 %, max 24.7 %)
- St within +/- 35 %: **5 / 5** (median abs error 13.7 %, max 24.1 %)

### Strouhal FFT diagnostics

- FFT record length depends on n_steps; resolved-preset rows here used **~3675 lattice steps** for the FFT half.
- St-axis bin width: **0.076** across all rows (uniform record length and char_length D = 40).
- Captured cycles at the measured peak: **2 - 3** across all rows -- well below the 20-cycle "informative" threshold.
- Implication: same as the Standard table -- the FFT bin width is the same order as the Williamson St(Re) range across our Re band, so the displayed St error is dominated by bin quantisation, not by solver fidelity. The St columns are reported for completeness only; the validated quantity in this doc is Cd. VALIDATION.md §3.4 has the long-form discussion.
- Per-row diagnostics live in the JSON sibling; `tests/test_validation_st.py` gates them.
