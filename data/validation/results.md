# Validation results

Solver-output Cd, Cl, St vs published free-stream reference,
after Allen-Vincenti blockage correction (Standard preset, B = 0.350).

| Shape    | Re   | Cd raw | Cd corr | Cd ref | Cd err % | St raw | St corr | St ref | St err % | Cd pass | St pass |
|----------|------|--------|---------|--------|----------|--------|---------|--------|----------|---------|---------|
| Cylinder |  100 | 3.590 | 1.358 | 1.320 | +2.9 | 0.373 | 0.205 | 0.166 | +23.4 |   PASS  |   PASS  |
| Cylinder |  200 | 3.110 | 1.176 | 1.150 | +2.3 | 0.373 | 0.205 | 0.197 | +4.0 |   PASS  |   PASS  |
| Cylinder |  500 | 2.883 | 1.090 | 1.020 | +6.9 | 0.320 | 0.176 | 0.207 | -15.2 |   PASS  |   PASS  |
| Square   |  200 | 4.259 | 1.799 | 1.600 | +12.5 | 0.373 | 0.205 | 0.148 | +38.4 |   PASS  |  *FAIL* |
| Square   |  500 | 4.477 | 1.891 | 2.000 | -5.4 | 0.373 | 0.205 | 0.135 | +51.7 |   PASS  |  *FAIL* |

### Aggregate statistics

- Cases run: 5
- Cd within +/- 20 %: **5 / 5** (median abs error 5.4 %, max 12.5 %)
- St within +/- 25 %: **3 / 5** (median abs error 23.4 %, max 51.7 %)
