# Validation results

Solver-output Cd, Cl, St vs published free-stream reference,
after Allen-Vincenti blockage correction (Standard preset, B = 0.350).

| Shape    | Re   | Cd raw | Cd corr | Cd ref | Cd err % | St raw | St corr | St ref | St err % | Cd pass | St pass |
|----------|------|--------|---------|--------|----------|--------|---------|--------|----------|---------|---------|
| Cylinder |   40 | 5.404 | 2.044 | 1.550 | +31.9 | 0.107 | 0.059 |   --   |   --   |  *FAIL* |   PASS  |
| Cylinder |   80 | 3.866 | 1.462 | 1.380 | +6.0 | 0.373 | 0.205 | 0.150 | +36.6 |   PASS  |  *FAIL* |
| Cylinder |  100 | 3.590 | 1.358 | 1.320 | +2.9 | 0.373 | 0.205 | 0.166 | +23.4 |   PASS  |   PASS  |
| Cylinder |  150 | 3.250 | 1.229 | 1.200 | +2.4 | 0.373 | 0.205 | 0.182 | +12.6 |   PASS  |   PASS  |
| Cylinder |  200 | 3.110 | 1.176 | 1.150 | +2.3 | 0.373 | 0.205 | 0.197 | +4.0 |   PASS  |   PASS  |
| Cylinder |  300 | 2.978 | 1.126 | 1.080 | +4.3 | 0.373 | 0.205 | 0.203 | +0.9 |   PASS  |   PASS  |
| Cylinder |  500 | 2.883 | 1.090 | 1.020 | +6.9 | 0.320 | 0.176 | 0.207 | -15.2 |   PASS  |   PASS  |
| Cylinder |  800 | 2.894 | 1.095 | 1.000 | +9.5 | 0.320 | 0.176 | 0.209 | -16.0 |   PASS  |   PASS  |
| Cylinder | 1000 | 2.921 | 1.105 | 0.990 | +11.6 | 0.320 | 0.176 | 0.210 | -16.4 |   PASS  |   PASS  |
| Square   |  100 | 4.978 | 2.103 | 1.500 | +40.2 | 0.373 | 0.205 | 0.143 | +43.2 |  *FAIL* |   PASS  |
| Square   |  150 | 4.469 | 1.888 | 1.550 | +21.8 | 0.373 | 0.205 | 0.146 | +40.3 |   PASS  |   PASS  |
| Square   |  200 | 4.259 | 1.799 | 1.600 | +12.5 | 0.373 | 0.205 | 0.148 | +38.4 |   PASS  |   PASS  |
| Square   |  300 | 4.191 | 1.771 | 1.850 | -4.3 | 0.373 | 0.205 | 0.142 | +44.3 |   PASS  |   PASS  |
| Square   |  500 | 4.477 | 1.891 | 2.000 | -5.4 | 0.373 | 0.205 | 0.135 | +51.7 |   PASS  |   PASS  |

### Aggregate statistics

- Cases run: 14
- Cd within +/- 25 %: **12 / 14** (median abs error 6.4 %, max 40.2 %)
- St within +/- 30 %: **13 / 14** (median abs error 23.4 %, max 51.7 %)
