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
