# SimScale cross-validation: cylinder Re = 100 (runbook)

**Status:** ready to run, no Linux or local install required.

This is a browser-based alternative to the OpenFOAM scaffolding in
[`../../openfoam/cylinder_re100/`](../../openfoam/cylinder_re100/).
SimScale's Community plan uses OpenFOAM as its incompressible
finite-volume backend (`pimpleFoam` family), so the cross-validation
claim is identical: same numerical method, same problem, just on
their cloud compute. Sign-up to a Cd number is ~30–45 min of
wall-clock and ~5–15 min of human time.

## Why this exists

The original V2 deliverable from David Artemyev's 2026-05-27 review
required WSL / Linux / macOS to run OpenFOAM locally. AeroLab is
developed on Windows. SimScale removes the OS barrier without
changing the underlying solver.

## What this validates against

The AeroLab **Validation preset** (D = 20 cells, B = 5 %) at Re = 100
— the same preset the 2D `compare_aerolab_vs_openfoam.py` script in
the OpenFOAM scaffolding targets. Three numbers come out of the
SimScale run, all directly comparable to the AeroLab + Williamson
1996 numbers in VALIDATION.md §3.6:

| Quantity | Williamson 1996 | AeroLab Validation | SimScale (this run) |
|---|---|---|---|
| Cd (mean over shedding cycles) | 1.33 – 1.40 | gated in CI | TBD |
| St (Strouhal) | 0.164 – 0.166 | gated in CI | TBD |
| L/D (recirculation length) | 0.84 – 0.87 | not measured | TBD |

Two-line conclusion that goes in VALIDATION.md §8.4: "SimScale
(OpenFOAM finite-volume) at the same geometry gives Cd = X, St = Y.
AeroLab's LBM lands within ±Z % of both, on the same problem solved
by a different method — independent confirmation."

## Account setup (one-time, ~3 min)

1. Sign up at <https://www.simscale.com> on the **Community** plan
   (free). 3 000 core-hours/year quota; this case uses ~5–15.
2. Verify email.
3. Note: Community projects are public by default. That is desired
   here — it lets a reviewer (David) re-run the case from your URL.
   Add the URL to VALIDATION.md §8.4 once the run is done.

## Case setup (~10 min in the SimScale UI)

### Geometry

Use SimScale's built-in **CAD Edit** to create the 2D-like channel
with cylinder, then extrude thin in span. Or upload a STEP file you
generate locally (FreeCAD: half a day; one of the public SimScale
"cylinder in channel" templates is faster — search the public
project library).

Domain dimensions (mirrors AeroLab Validation preset and the
OpenFOAM scaffold in `../../openfoam/cylinder_re100/README.md`):

- Channel: 30 D long × 20 D tall × 0.05 D thick (thin span — this is
  how SimScale handles 2D-like problems; the side walls get periodic /
  symmetry BCs)
- Cylinder: D = 0.01 m at `(10 D, 10 D)` in the inlet plane
- Inlet 10 D upstream, outlet 20 D downstream of the cylinder centre
- Top/bottom walls 10 D from cylinder centre → blockage B = D / (20 D)
  = 5 % (Allen-Vincenti correction is negligible at this blockage)

### Analysis type

- **Incompressible** → **Transient**
  - **Do NOT pick** "Lattice Boltzmann (Beta)" — for a cross-check
    you specifically want a *different* numerical method from
    AeroLab's LBM.
- Solver: SimScale picks `pimpleFoam` under the hood. Confirm the
  OpenFOAM version in the solver log after the run (it has been v9
  / v10 / v11 in recent years; cite whichever you got).

### Material

- Newtonian fluid (water for SI ease, or air — Cd is dimensionless,
  it doesn't matter for the comparison)
- Set `nu = 1e-4 m²/s` so that with `U = 1 m/s` and `D = 0.01 m`,
  `Re = U·D/ν = 100`

### Boundary conditions

- Inlet (upstream face): Velocity inlet, U = (1, 0, 0) m/s
- Outlet (downstream face): Pressure outlet, p = 0
- Cylinder surface: No-slip wall (default)
- Top + bottom faces: **Slip** wall (not no-slip — slip is the
  symmetry equivalent at B = 5 %; matches the OpenFOAM scaffold)
- Front + back (span) faces: **Symmetry** (the cheap "2D" approximation)

### Mesh

- Standard automatic mesh, "Fineness: 4" (or move the slider toward
  "fine"). Refine near the cylinder via a local mesh refinement
  region (Box: ±2 D around cylinder, target cell size = D / 60).
- Target ~150 k – 300 k cells. Cost: ~5–10 core-hours on the
  Community plan.

### Numerics + time stepping

- Time step: `dt = 0.005 s` (CFL ≈ 0.5 at u_in = 1, dx = D/60 = 0.01/60 ≈ 1.7e-4)
  — SimScale's solver will adapt; this is just the initial guess.
- End time: `t = 2.0 s` = 200 D/U
  — ~33 shedding periods at St ≈ 0.165, well past the 20-cycle
  FFT threshold and the startup transient.
- Schemes: leave at SimScale defaults (PIMPLE + second-order in time
  and space) — those match the OpenFOAM scaffold's `system/fvSchemes`.

### Result control

Critical: turn on **Force on cylinder** (lift / drag time series).
Without this, no Cd or St extraction is possible.

- Result Control → Surface data → Forces → select the cylinder face,
  reference density 1000 (water) or 1.2 (air), reference area =
  `D × span = 0.01 × 0.0005 = 5e-6 m²`, write interval every step.
- Optionally: probe `p, U` at a point one cylinder diameter
  downstream for a velocity time series.

### Simulation control

- Number of cores: 8 (Community default)
- Maximum runtime: 2 hours (will finish in ~30 min)

## Solve (~20–30 min wall-clock)

Click **Start simulation**. SimScale queues the job, runs it on AWS,
and emails when it finishes. You can close the browser tab.

## Post-processing → Cd and St (~5–10 min)

### Pull the force-time-series CSV

Once the run finishes:

1. Results → Solution fields → **Forces on cylinder** → download
   CSV (columns: time, F_x, F_y, F_z, ...).

### Compute Cd, St locally

```python
import numpy as np
import pandas as pd

df = pd.read_csv("simscale_forces.csv")
t  = df["Time [s]"].values
Fx = df["F_x [N]"].values          # drag
Fy = df["F_y [N]"].values          # lift

# Reference dynamic pressure
rho   = 1000.0                     # water; match SimScale material setting
U_inf = 1.0
D     = 0.01
span  = 0.0005                     # thin-span thickness
A_ref = D * span
q     = 0.5 * rho * U_inf ** 2

Cd_t = Fx / (q * A_ref)
Cl_t = Fy / (q * A_ref)

# Throw away first 50 D/U of startup -> use the last 100+ shedding cycles
mask     = t > 0.5
Cd_mean  = Cd_t[mask].mean()
Cd_std   = Cd_t[mask].std()

# Strouhal: FFT of Cl_t, peak frequency * D / U
dt       = np.median(np.diff(t[mask]))
freqs    = np.fft.rfftfreq(mask.sum(), d=dt)
spectrum = np.abs(np.fft.rfft(Cl_t[mask] - Cl_t[mask].mean()))
f_peak   = freqs[1 + spectrum[1:].argmax()]
St       = f_peak * D / U_inf

print(f"Cd = {Cd_mean:.3f} +- {Cd_std:.3f}")
print(f"St = {St:.4f}")
```

For thin-span 2D-like cases the lift coefficient is the one that
oscillates at the shedding frequency; the drag coefficient
oscillates at **2 × f_shedding** and at lower amplitude — use Cl for
the Strouhal FFT.

### Recirculation length (optional but cheap)

From the velocity field: take a horizontal line through the
cylinder centre, find the x-position past the cylinder where
`u_x` first becomes positive again. That's L. Plot is one screenshot.

## What to record (paste into VALIDATION.md §8.4)

| Field | Value |
|---|---|
| SimScale project URL (public) | `https://www.simscale.com/projects/...` |
| OpenFOAM version | (read from solver log) |
| Mesh size | (final cell count) |
| Solve wall-time | (from SimScale dashboard) |
| Cd mean ± std | |
| St (Strouhal) | |
| L/D | |

## Williamson 1996 reference

Williamson, C. H. K. (1996). "Vortex dynamics in the cylinder wake."
Annual Review of Fluid Mechanics, 28, 477–539.

- Cd at Re = 100: ~1.33–1.40 (consensus across Park 1998,
  Posdziech & Grundmann 2007, Henderson 1995)
- St at Re = 100: ~0.164–0.166
- L/D at Re = 100: ~0.84–0.87 (steady recirculation length;
  unsteady-mean reads similar past startup)

## Failure modes / common gotchas

- **"Floating point exception" in the first few steps.** dt too large
  or mesh too coarse near the cylinder. Halve dt and re-run.
- **Cd reads ~0.7 instead of ~1.35.** You very likely used a 2D area
  reference (D × 1 m) instead of `D × span_thickness`. Re-divide.
- **St reads ~2× expected.** You FFT'd Cd instead of Cl. See above
  — Cd oscillates at 2× f_shedding.
- **Wake looks steady, not shedding.** Mesh too coarse in the
  near-wake, or end-time too short. Re-mesh with `D/80` near
  cylinder and re-run to `t = 3 s` if needed.

## Reciprocity with the OpenFOAM scaffold

`../../openfoam/cylinder_re100/` is the local-OpenFOAM path to the
same answer. If you ever do get WSL set up, that path runs `pisoFoam`
on the same geometry and produces a directly comparable Cd / St.
Either path closes the V2 reviewer item; SimScale is just the
no-install version.
