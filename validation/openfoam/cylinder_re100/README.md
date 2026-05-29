# OpenFOAM cross-validation: cylinder Re = 100

**Status: scaffolded, run pending.** This directory is the OpenFOAM
deliverable from David Artemyev's 2026-05-27 review (card #6). It
ships the case files and a comparison script; the actual solve is
runtime-deferred because the AeroLab dev tree is Windows and OpenFOAM
runs cleanly only on Linux / WSL / macOS.

## Goal

Produce a third independent number against the AeroLab Standard /
Validation / Resolved presets for cylinder Re = 100, so the three-way
comparison **AeroLab vs OpenFOAM vs Williamson 1996** becomes a single
table the reader can audit.

## Case parameters

Chosen to mirror the AeroLab Validation preset (D = 20 cells, B = 5 %)
so the blockage correction is small and the comparison is honest:

- 2D laminar incompressible Newtonian flow, `pisoFoam`
- Domain: `30 D` long × `20 D` tall, cylinder at `x = 10 D`
- Mesh: structured hex grid, refined near the cylinder (`y+ < 1`)
- Re = 100 (U = 1 m/s, D = 0.01 m, nu = 1e-4 m²/s)
- BCs: velocity inlet, pressure outlet, no-slip cylinder,
  symmetry top / bottom (cf. AeroLab's no-slip walls — see
  VALIDATION.md §4.6 for the wall-BC trade-off)
- CFL ≤ 0.5 throughout
- Run time: `t = 200 D/U` (≈ 33 shedding periods at St ≈ 0.165 — well
  past the 20-cycle FFT-resolution threshold from card #5)

## To run (Linux / WSL / macOS, OpenFOAM ≥ 11 installed)

```bash
cd validation/openfoam/cylinder_re100
./Allrun                         # mesh + solve + postProcess, ~20 - 40 min
python ../../compare_aerolab_vs_openfoam.py
```

`Allrun` is the standard OpenFOAM tutorial-style driver; it invokes
`blockMesh`, `pisoFoam`, and `postProcess -func forceCoeffs` in that
order. `forceCoeffs.dat` lands in `postProcessing/forceCoeffs/0/`.

## What the compare script does

`compare_aerolab_vs_openfoam.py` (lives one level up):

1. Reads `postProcessing/forceCoeffs/0/forceCoeffs.dat` and computes
   `Cd_mean` over the last 50 D/U.
2. Reads `data/validation/results_lowblockage.json` for AeroLab's
   `Cd_corrected` and `Cd_raw` at the same case.
3. Reads the Williamson 1996 ARFM 28 reference from
   `src/references.py:CYLINDER_FREESTREAM`.
4. Writes a 3-row comparison table to
   `validation/cross_validation.md` (committed) and prints the same
   table to stdout. Each row reports the value AND the percent
   deviation from Williamson, so the comparison is symmetric.

## Acceptance (per card #6)

- `Cd_AeroLab` within 5 % of `Cd_OpenFOAM`
- `Cd_OpenFOAM` within 5 % of `Williamson 1996`

If either gate fails on first solve, the failure is informative on its
own (different wall BCs, different inflow profile, different
turbulence treatment) and gets documented before retuning the case.

## Why this is not in CI

OpenFOAM is 600 MB+ of binary dependencies, a 20-40 minute solve, and
not available on the GitHub Actions runners we use. The compare
SCRIPT can run in CI as a sanity check once `forceCoeffs.dat` and
`cross_validation.md` are committed; the SOLVE itself stays an
offline / one-shot deliverable.
