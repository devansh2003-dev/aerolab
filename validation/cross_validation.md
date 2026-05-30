# Three-way cross-validation: cylinder Re = 100

Sources:

- AeroLab: `data/validation/results_lowblockage.json` (Validation preset, D = 20, B = 5 %)
- OpenFOAM 11: `validation/openfoam/cylinder_re100/` (foamRun + incompressibleFluid, 2D laminar). Notes: 10001 samples in last 50 D/U (t_end = 400.000 s, equiv 200.0 D/U)
- Williamson 1996 ARFM 28: from `src/references.py:CYLINDER_FREESTREAM`

| Source | Cd | Deviation Cd vs Williamson | St | Deviation St vs Williamson |
|--------|----|-----------------------------|----|-----------------------------|
| AeroLab (Cd corrected, D=20) | 1.348 | +2.13 % | n/a | n/a |
| AeroLab (Cd raw, D=20) | 1.510 | +14.36 % | n/a | n/a |
| OpenFOAM foamRun (incompressibleFluid, laminar) | 1.183 | -10.36 % | 0.1200 | -27.72 % |
| Williamson 1996 ARFM 28 | 1.320 | 0 (reference) | 0.1660 | 0 (reference) |

**Notes on the OpenFOAM result.**

- **Cd at -10 % vs Williamson** is the under-resolved-wake
  signature. The shipped mesh has 6 480 cells (18 × 18 per
  block × 20 blocks); the inner annulus around the cylinder
  is well-resolved (~2.5 ° / cell tangentially, 0.023-cell
  radial spacing) but the outer downstream blocks span ~20
  lattice units in 18 cells → ~1.1 unit per cell, i.e.
  roughly D/2 per cell in the wake. That coarseness adds
  numerical dissipation which acts like a viscosity bump,
  dropping the effective Reynolds number.
- **St at -28 % vs Williamson** is the same story, more
  sensitive. Williamson 1996 reports St = 0.13 at Re ≈ 50
  and St = 0.166 at Re = 100; the measured 0.120 sits in
  the Re ≈ 40 - 50 band, consistent with the effective Re
  drop from numerical dissipation.
- **Cd mean is taken over t = 300 - 400 s** (last 50 D/U).
  Shedding bootstrapped slowly because the symmetric mesh +
  symmetric inflow locks the wake in a metastable steady
  state until floating-point asymmetry breaks it (the run
  was extended past t = 200 for this reason; the diagnostic
  in `openfoam/cylinder_re100/diagnose.py` shows Cl_std
  rising from ~0.0001 at t = 80 to ~0.32 at t = 400).
- **AeroLab's corrected Cd lands within 2.1 %** of
  Williamson without any numerical-dissipation refit, which
  is the headline 2D validation already gated in CI by
  `test_validation_benchmark.py`.

**What this comparison closes and does not close.**

- ✅ V2 from David Artemyev's 2026-05-27 review (third
  independent Cd number from a different numerical method)
  is **measured and documented**.
- ✅ AeroLab's MRT-LBM and OpenFOAM's finite-volume are on
  the same *side* of the Williamson reference for the
  headline Cd: AeroLab corrected is +2 %, OpenFOAM is -10 %.
  Different sign of error reflects different bias sources
  (LBM blockage correction over-shoots slightly; FV
  under-resolved wake under-shoots).
- ❌ The reviewer's 5 % gates do NOT pass. To pass them, the
  next iteration of the OpenFOAM case would refine the outer
  downstream blocks (target h/D ≤ 0.25 in the wake) and run
  to t ≥ 600 s so shedding amplitude is fully saturated.
  Expected effort: ~30 - 45 min wall-time; not done in this
  round.
