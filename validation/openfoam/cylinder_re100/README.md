# OpenFOAM cross-validation: cylinder Re = 100

**Status: case implemented, runnable on OpenFOAM 11.** This directory
ships the OpenFOAM case files and an `Allrun` driver that, together
with the comparison script one level up
([`../../compare_aerolab_vs_openfoam.py`](../../compare_aerolab_vs_openfoam.py)),
close item V2 (third-party drag cross-check) from David Artemyev's
2026-05-27 review (card #6).

## Goal

Produce a third independent number against the AeroLab Validation
preset for cylinder Re = 100, so the comparison **AeroLab vs OpenFOAM
vs Williamson 1996** becomes a single table the reader can audit.

## Case parameters

Chosen to mirror the AeroLab Validation preset (D = 20 cells in a
400-cell-tall channel, B = D / Ly = 5 %) so the blockage correction
is small and the comparison is honest:

- 2D laminar incompressible Newtonian flow, **OpenFOAM 11
  `foamRun` driver with `solver incompressibleFluid`** (the modern
  OF11 replacement for the old standalone `pisoFoam` / `pimpleFoam`
  binaries).
- **Domain:** `60` long × `40` tall × `2` deep (single span cell,
  empty BCs → quasi-2D). Cylinder centred at the origin. With D = 2,
  this gives 10 D upstream, 20 D downstream, ±10 D on each side
  (**B = 2 / 40 = 5 %**).
- **Mesh:** structured 8-block planar O-grid around the cylinder
  (inner ring r ∈ [1, √2] then 4 outer corner blocks reaching the
  bounding box). 18 × 18 × 1 cells per block × 20 blocks ≈ **6 480
  cells total**. 144 tangential cells around the cylinder (≈ 2.5 °
  per cell), 18 radial cells in the inner O-grid annulus.
- **Re = 100** via U = 1 m/s, D = 2 m, ν = 0.02 m²/s.
- **Boundary conditions:**
  - Inlet: velocity inlet U = (1, 0, 0).
  - Outlet: pressure outlet p = 0 (zero-gradient on U).
  - Top / bottom: **slip walls**, NOT no-slip. At B = 5 % the
    Allen-Vincenti correction is negligible *only if* the outer
    walls do not grow boundary layers that close the channel. Slip
    keeps the outer walls effectively at u = U_∞ while still being
    a real BC (vs. periodic, which would couple the wake top and
    bottom and change the dynamics).
  - Cylinder: no-slip.
  - z faces: empty (quasi-2D).
- **Time integration:** dt = 0.01, endTime = 200. That's 100 D/U,
  with the wake fully developed past ~25 D/U at Re = 100 and ~6
  shedding cycles for FFT.
- **Solver settings:** PIMPLE 2 outer + 1 inner corrector. GAMG /
  DIC for pressure, PBiCGStab / DILU for momentum. CFL ≈ 0.27 at
  the finest cell.

## To run

The case path must contain **no spaces** (OpenFOAM tokenises on
whitespace). If your AeroLab checkout sits under `Study & Work/...`,
copy the case to a clean path first:

```bash
# Linux / WSL / macOS, with OpenFOAM 11 sourced
cp -r validation/openfoam/cylinder_re100 ~/aerolab_cyl_re100
cd ~/aerolab_cyl_re100
./Allrun                         # blockMesh + checkMesh + foamRun
# -> postProcessing/forceCoeffs/0/forceCoeffs.dat
```

`Allrun` invokes `blockMesh`, `checkMesh`, and `foamRun` in that
order; the `forceCoeffs` function object in `system/controlDict`
writes Cd, Cl every time step to
`postProcessing/forceCoeffs/0/forceCoeffs.dat` during the solve.

Then back at the project root:

```bash
cp -r ~/aerolab_cyl_re100/postProcessing validation/openfoam/cylinder_re100/
python validation/compare_aerolab_vs_openfoam.py
# -> writes validation/cross_validation.md
```

## What the compare script does

[`validation/compare_aerolab_vs_openfoam.py`](../../compare_aerolab_vs_openfoam.py):

1. Reads `postProcessing/forceCoeffs/0/forceCoeffs.dat`. Auto-detects
   the OF11 column order (`Time Cd Cs Cl ...`) from the header
   comment so it also works on older OpenFOAM versions
   (`time Cm Cd Cl ...`).
2. Averages Cd over the last 50 D/U (= last 100 s of case time at
   D = 2, U = 1). Computes Strouhal from an FFT of Cl over the
   same window.
3. Reads AeroLab's Validation-preset Cd from
   `data/validation/results_lowblockage.json`.
4. Reads Williamson 1996 Cd / St from
   `src/references.py:CYLINDER_FREESTREAM_{CD,ST}`.
5. Writes the four-row three-way comparison table to
   [`validation/cross_validation.md`](../../cross_validation.md)
   and prints it to stdout.

## Acceptance (per review card #6)

- `Cd_AeroLab` within 5 % of `Cd_OpenFOAM`
- `Cd_OpenFOAM` within 5 % of `Williamson 1996`
- `St_OpenFOAM` within 5 % of `Williamson 1996`

If either gate fails on first solve, the failure is informative on its
own (different wall BCs, different inflow profile, different
turbulence treatment) and gets documented in VALIDATION.md §8.4
before retuning the case.

## Why this is not in CI

OpenFOAM 11 is 600 MB+ of binary dependencies, a ~10–15 min solve, and
not available on the GitHub Actions runners we use. The compare
SCRIPT can run in CI off a committed `forceCoeffs.dat` and
`cross_validation.md`; the SOLVE itself stays an offline / one-shot
deliverable. To reproduce on a fresh machine, install OpenFOAM 11
(<https://openfoam.org/download/>) and run the steps above.

## Browser-only alternative

If you don't have OpenFOAM locally, the SimScale runbook at
[`../../simscale/cylinder_re100/RUNBOOK.md`](../../simscale/cylinder_re100/RUNBOOK.md)
runs the same case on SimScale's cloud OpenFOAM in ~30–45 min with no
install. Both paths produce a directly-comparable forceCoeffs record.
