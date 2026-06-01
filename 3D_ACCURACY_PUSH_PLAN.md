# 3D accuracy push — plan as of v1.7.1 (2026-06-02)

User ask: *"from all the tests we did, can you make [the 3D solver] more accurate."*

This doc is a pair-programming plan. It is not yet executed. Each
item names the experiment, the expected outcome under each hypothesis,
and the wall-time cost.

## Where we are

| Configuration | Cd | Δ vs CGW |
|---|---|---|
| Re = 100, D = 20, Ladd | 1.65 | +51 % |
| Re = 100, D = 40, Ladd | 1.53 | +40 % |
| Re = 100, D = 40, **MYSL** | **1.16** | **+6.4 %** |
| Re = 20,  D = 20, Ladd | 4.27 | +56 % |
| Re = 20,  D = 40, Ladd | (not run — would be ~ +55 %) | — |
| Re = 20,  D = 40, **MYSL** | **4.02** | **+47 %** |

**What the data tells us:**

1. At Re = 100, the two known bias sources (grid + Ladd halfway-BB
   assumption) account for 51 pp out of the 51 % bias — fully closed
   to +6.4 % by D = 40 + MYSL. The remaining 6 % decomposes to
   ~3-5 % residual blockage + ~1-3 % D = 40 voxelisation + ~1 %
   higher-order BB. Known refinement path.
2. At Re = 20, the SAME upgrades (D = 40 + MYSL) only close 9 pp
   out of 56 %. **There is a residual bias source that does not
   appear at Re = 100.** The leading suspect is the LBM equilibrium
   itself at high τ (τ_Re=20 = 0.74 vs τ_Re=100 = 0.548).

This is the single most actionable signal in the project. The audit
called for "make the 3D solver more accurate" — the data points
directly at what to attack.

## Three investigation paths, ordered by cost

### Path A: Lower τ at Re = 20 by rescaling u_in (cheap test, ~ 2.5 h compute)

Same physical Re = 20, but ν / u_in scaled so τ ≈ 0.55 instead of
0.74. If the +47 % bias drops substantially with τ alone (no other
code change), the high-τ hypothesis is confirmed and the path
forward is clear: lower-τ-only operation or a τ-independent collision.

**Cost:** one ~ 2.3 h bake at lower u_in. ~ 1 h to write the script
and tests. **Total: half a day.**

**Outcomes:**

- Cd drops toward 3.0 (≈ +10 % vs CGW) → high-τ residual confirmed;
  proceed to Path B.
- Cd stays at 4.0 (+47 %) → high-τ hypothesis falsified; the residual
  is something else (slow viscous convergence? grid asymmetry at low
  Re?). Investigate further before committing to multi-day work.

### Path B: Implement cumulant collision (Geier et al. 2015) — multi-day code work

Cumulant LBM is Galilean-invariant by construction (the standard
equilibrium isn't). It removes the τ-dependent spurious cross-terms
that suspect Path A would have confirmed. It also extends the stable
Re ceiling (current TRT diverges at Re ≥ 200 on our grid).

**Cost:** ~ 2-3 days of code (collision-step kernel + equilibrium
moments + Numba), 1 day of verification (TGV decay, channel-flow
parabola), 1 night of bakes (both Re = 20 and Re = 100 with the
new collision).

**Outcomes:**

- Re = 20 Cd drops into the percent-level band → cumulant is the
  right tool; ship as the production 3D kernel.
- Re = 20 Cd does not drop → the bias is somewhere we haven't looked
  yet (e.g. spanwise periodicity, outflow BC at high viscosity).

### Path C: 3D OpenFOAM cross-check at Re = 20 sphere — multi-hour run

Independent third-method verification of the +47 %. If OpenFOAM also
lands at ~ Cd = 4.0 at Re = 20 with the same blockage / domain, the
bias is in the *physical setup* (blockage at low Re is well known
to bias upward, since the viscous wake is much longer than the
inertial one), not the LBM kernel — and the "fix" is to lower
blockage, not change the collision operator.

**Cost:** ~ 4-6 h OpenFOAM bake on 4 ranks (same case files as
v0.6.5, with ν scaled). ~ 1 h to set up the case + compare script.

**Outcomes:**

- OpenFOAM Cd ≈ 4.0 → AeroLab is RIGHT and CGW's tabulated 2.728 is
  the wrong reference at this blockage. Update VALIDATION.md to
  cite Allen-Vincenti / similar low-Re blockage correction; the
  3D claim is healthier than it looks.
- OpenFOAM Cd ≈ 2.8 → AeroLab is wrong, and Path A / B is
  unavoidable.

## Recommended sequence

1. **Path A first** (cheap, ~ half a day). It diagnoses the
   problem with one experiment.
2. Based on Path A result:
   - High-τ confirmed → **Path B** (the multi-day cumulant
     implementation).
   - High-τ falsified → **Path C** (OpenFOAM verification). The
     "physical setup" answer is the cheapest possible win — if true,
     no code changes needed.
3. After Path B or C lands, the §8.3.5 narrative either evolves to
   "cumulant fixes it" or "blockage correction explains it." Either
   way, the project has its general 3D claim.

## What this is NOT

- This plan does not commit to multi-day work yet. Path A is the
  smart first move; everything else is conditional.
- "Make the solver more accurate" is open-ended. Going to D = 60 /
  B = 10 % at Re = 100 (the §8.8 #2 item) tightens the validated
  configuration but doesn't address the **diagnostic** the audit
  surfaced. Closing the high-τ residual is higher-leverage.

## Sign-offs needed before committing real time

- User confirmation that the 3D-accuracy goal is the right place
  to invest. The alternative is finishing the UX side of AeroLab
  (per the visual-richness audience). Both are valid.
- User confirmation of which Path to start with.
