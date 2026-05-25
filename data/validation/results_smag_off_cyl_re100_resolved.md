# Smagorinsky-off experiment, Cylinder Re = 100, Resolved preset

The reviewer (2026-05-27) flagged the Resolved Cylinder Re = 100
corrected Cd error of −10.2 % as a yellow flag: it is the most
laminar, cleanest case, and the error sign flipped vs the D = 20
Validation preset (+2.1 % → −10.2 %). The §4.4 LES-at-laminar-Re
note already named the Smagorinsky sub-grid eddy viscosity as a
candidate cause. This run isolates it.

## Setup

- Grid:        Resolved preset (1200 × 400, D = 40, B = 0.10)
- Re:          100 (Cylinder, AoA = 0)
- n_frames:    300 (10 500 lattice steps)
- All else:    identical to the on-record Resolved sweep
- Difference:  `src.lbm.C_SMAG = C_SMAG_SQ = 0` monkey-patched
               BEFORE the first JIT compile of the MRT kernel, so
               Numba captures the patched value at trace time.

Reproduce:

```bash
python scripts/dev_smag_off_resolved_re100.py
```

Runtime ≈ 16 min on the development laptop.

## Result

| Configuration            | Cd raw   | Cd corrected | Error vs Williamson (1.32) |
|--------------------------|----------|--------------|----------------------------|
| Smag-on  (C_smag = 0.17) | 1.4965   | 1.1859       | −10.16 %                   |
| Smag-off (C_smag = 0)    | 1.4914   | 1.1814       | −10.50 %                   |
| Δ                        | −0.0051  | −0.0045      | −0.34 pp                   |

Strouhal (per-step FFT) at this case:

| Configuration | St raw  | St corrected |
|---------------|---------|--------------|
| Smag-on       | 0.1524  | 0.1259       |
| Smag-off      | 0.1524  | 0.1259       |

Same FFT bin, as expected at this record length (Δ St ≈ 0.076 at the
Resolved preset, see VALIDATION.md §3.4).

## Interpretation

The 0.34 percentage-point shift is far smaller than the ~ 10 pp shift
that would be needed for the LES to be responsible for the
Resolved Re = 100 result. The Smagorinsky term is therefore NOT the
source of the −10 % bias. It pays a small (~ 0.3 % on raw Cd)
laminar-Re penalty for being on, but it is not the dominant error
contribution.

The remaining candidate causes for the Re = 100 error are:

1. **K-mismatch at low blockage** (most likely). The K = 1.10 AV
   factor for cylinder was fitted at the Standard B = 0.35 against
   Williamson. At B = 0.10 the same constant over-rescales: the raw
   measurement is +13 % over Williamson, the correction drops it by
   −24 %, net −10 %. This is the same calibration problem already
   exposed for the square in VALIDATION.md §3.2 -- just less
   dramatic. A K(B) recalibration would address both shapes.

2. **Grid-staircase residual at D = 40**. Mei-Luo-Shyy 1999
   recommend D ≥ 40 for free-stream Cd; we sit AT the recommended
   minimum, not above it. A D = 80 sweep would close this
   contribution; not run.

3. **Wall-interaction at B = 0.10**. The no-slip vertical walls
   inject a small additional drag contribution at this blockage; see
   VALIDATION.md §4.6.

K-recalibration is the highest-leverage next step. Out of scope for
this revision (would shift the Standard-preset numbers in §3.6 and
require a fresh full sweep).
