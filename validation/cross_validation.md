# Three-way cross-validation: cylinder Re = 100

Sources:

- AeroLab: `data/validation/results_lowblockage.json` (Validation preset, D = 20, B = 5 %)
- OpenFOAM: `validation/openfoam/cylinder_re100/` (pisoFoam, 2D laminar). Notes: OpenFOAM `forceCoeffs.dat` not present (validation\openfoam\cylinder_re100\postProcessing\forceCoeffs\0\forceCoeffs.dat). Run `./Allrun` in the case directory first.
- Williamson 1996 ARFM 28: from `src/references.py:CYLINDER_FREESTREAM`

| Source | Cd | Deviation from Williamson |
|--------|----|----------------------------|
| AeroLab (Cd corrected, D=20) | 1.348 | +2.13 % |
| AeroLab (Cd raw, D=20) | 1.510 | +14.36 % |
| OpenFOAM pisoFoam (laminar) | n/a | n/a |
| Williamson 1996 ARFM 28 | 1.320 | 0 (reference) |

**Status: pending OpenFOAM solve.** Run `validation/openfoam/cylinder_re100/Allrun` on a Linux / WSL / macOS box with OpenFOAM >= 11, then re-run this script.
