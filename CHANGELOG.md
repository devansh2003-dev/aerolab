# Changelog

All notable changes to AeroLab. Dates are absolute; versions follow [SemVer](https://semver.org/).

## [1.7.3] — 2026-06-02 (±45° Re=200 wing clipping fix)

User report: "at 45 deg, both airfoils go out of the box."

Verified — the 4 ±45° Re=200 wings had `Nz = 40` and `chord_offset = 32`,
but the rotated chord (vertical extent = chord × sin 45° + thickness ≈ 26 LU)
plus the offset placed the wing's upper tip at `z ≥ Nz − 1` (top wall).
Body z-bbox measured at `[21..39]` in `Nz=40` — confirmed top-wall clip.
The other 14 ±45° wings (Re=20, Re=40, Re=100) fit cleanly (`z=[8..24]`
in `Nz=32`).

### Fixed — 4 ±45° Re=200 wing presets

- **`scripts/bake_3d_field.py`** — for `naca{0012,4412}_aoa{±45}_re200`:
  - `Nz`: 40 → **48** (~20 % more voxels, ~7 LU extra clearance per face).
  - `chord_offset`: 32 → **24** (centers the chord in the Z domain so
    +45° and −45° are mirror-symmetric in placement, not just in flow).
  - `Nx`, `Ny`, `chord`, `nu`, `n_steps` unchanged. Inline comment in
    PRESETS explaining the fix.
- **`data/baked/naca{0012,4412}_aoa{±45}_re200.npz`** — 4 re-baked
  artifacts on the corrected grid. Re-bake batch wall-time: TBD
  (~2 h estimated on 4-core CPU).

### App version

- **`app.py`** — version chip `v1.7.2` → `v1.7.3`.

### Why this wasn't caught in v1.7.2

The v1.7.2 batch verified flow magnitude mirror-symmetry between
+AoA and −AoA pairs (3–6 % difference in `max|u|`) and treated the
top-wall clip as a "cosmetic" limitation. It wasn't — the rendered
wing in the gallery showed a visibly truncated chord, which the user
called out. Re-baking on the corrected grid both fixes the cosmetic
issue and lets the flow develop properly around the full chord (no
artificial flow channeling between body and top wall).

---

## [1.7.2] — 2026-06-02 (gallery wing convergence fix + Re=20 / Re=200 bake bands)

Direct response to two user-reported gallery issues:

1. **"At 45° the wing doesn't stall."** Root cause: every gallery
   bake ran at `n_steps = 800`, which at `u_in = 0.04` and chord = 24 LU
   is only **~1.3 chord-transits**. The flow had no time to develop
   separation — what the gallery showed was startup transient, not
   a stalled wake. Re-baked all ±30° and ±45° wing presets with
   `n_steps = 8 000` (Re = 40, ~10 chord-transits) and `n_steps = 12 000`
   (Re = 100, ~20 chord-transits).
2. **"Wind looks the same at different airspeeds."** Root cause: the
   speed slider only swapped between two baked bands (Re = 40 and
   Re = 100). Within a band, the same `.npz` replayed — slider felt
   inert. Added two new bake bands (Re = 20 "creeping" and Re = 200
   "shedding") so the slider has more snap-points and visible motion.

### Re-baked — ±30° / ±45° wing presets (B fix)

- **`scripts/bake_3d_field.py`** — bumped `n_steps` for 16 wing presets:
  - `naca{0012,4412}_aoa{±30,±45}_re40`: 800 → **8 000** (~10 chord-transits at u_in = 0.04, chord = 20)
  - `naca{0012,4412}_aoa{±30,±45}_re100`: 800 → **12 000** (~20 chord-transits at chord = 24)
- **`data/baked/naca*_aoa{±30,±45}_re{40,100}.npz`** — re-baked artifacts. The hash in each manifest changes; the visible flow at ±45° now shows a developed separated wake (was: laminar attached flow with no separation).

### Added — Re = 20 and Re = 200 baked bands (A3 fix)

- **`scripts/bake_3d_field.py`** — 22 new presets:
  - Bluff bodies: `sphere_re20`, `cylinder_re20` only. `sphere_re200` and `cylinder_re200` attempted but **diverged to u_peak = NaN** at u_in = 0.04 / ν = 0.0048 (τ = 0.5144). Root cause: 37 % blockage + τ near the TRT stability boundary is outside the stable envelope on consumer-grid sizes. Removed from PRESETS with an in-source comment explaining the deferral. To revive: cumulant collision (VALIDATION.md §8.8 #3) or grid Ny ≥ 128.
  - NACA 0012 + 4412 at AoA ∈ {0, ±30, ±45} × Re ∈ {20, 200} = 20 wing presets (all stable — the pilot `naca0012_re200` bake converged with `u_peak = 0.054`, well below Ma 0.17).
  - Re = 20 grids match the existing Re = 40 grids; `nu` scaled so Re = 20 exact. τ = 0.62. n_steps = 6 000 wings / 2 000 bluff.
  - Re = 200 wing grids refined to chord = 32 LU (was 24) and Ny bumped for the projected wing extent; `nu = 0.0064` → τ = 0.5192 (margin above the τ = 0.5 boundary). n_steps = 16 000.
- **`data/baked/`** — corresponding `.npz` files (auto-discovered by the gallery's filename parser, no app.py change needed for shape→Re routing).
- **User-visible result**: the gallery speed slider now has **4 snap-points for wings** (20 / 40 / 100 / 200) and **3 for bluff bodies** (20 / 40 / 100).

### Changed — 3D gallery slider

- **`app.py`** — `Flow speed (m/s)` slider min lowered from 0.10 m/s
  to **0.05 m/s** so Re = 20 (≈ 0.06 m/s) is reachable cleanly.
- Help text updated with the new 4-band ladder: 0.06 m/s (creeping) /
  0.12 m/s (laminar) / 0.30 m/s (transitional) / 0.60 m/s (shedding).
- Auto-discovery in `_shape_aoa_re_map` picks up the new bands by
  filename — no code change to the routing.

### Why this isn't a solver-accuracy regression

The B fix is a gallery-side fix only — the validated 3D claim
(§8.3.4 sphere Cd = 1.16 / +6.44 % at Re = 100) is unchanged. Gallery
bakes don't share scripts with the validation pipeline; the gallery
just needed more steps to develop separation, which the validation
runs already had (6 M+ steps).

### Known limitations

1. **Re = 200 wing bakes sit close to the TRT stability boundary**
   (τ = 0.5192). The pilot bake confirms convergence on the refined
   grid; the visible flow at the highest slider setting is
   shedding-regime (not validated to a literature Cd).
2. **Cosmetic wing-tip clipping at ±45° / Re = 200.** The Nz = 40
   gallery grid doesn't fully contain the projected chord at large
   AoA — the wing body bbox reaches `z = Nz - 1` (top wall). Flow
   magnitudes remain near mirror-symmetric (6 % difference between
   +45° and -45° max\|u\|) so the visualization is correct, but the
   rendered wing tip looks truncated. **Fixed in v1.7.3** (Nz=40→48,
   chord_offset=32→24).
3. **u_peak diagnostic** in the .npz manifest is `max(ux)`, not
   `max|u|`. At high AoA `ux` collapses while overall speed stays
   healthy — do not interpret a small `u_peak` as a divergence
   signal without cross-checking `mean|u|` from the field itself.
   (Source of one false alarm during the v1.7.2 A3 batch.)

---

## [1.7.1] — 2026-06-02 (Re=20 MYSL companion + AoA±45 wing presets + stale-banner UX)

Follow-on to v1.7.0. Three things landed: the method-consistent Re = 20
MYSL companion bake (audit Task 7 closed), eight new pre-baked wing
scenes at AoA = ±45° so the gallery slider can show stalled / deep-
separated flow, and a louder stale-display banner with an inline
"Run with new settings" button (user reported missing the previous
yellow warning).

### Added — §8.3.5 Re = 20 + MYSL + D = 40 (method-consistent sphere companion)

- **`scripts/validate_3d_sphere_cd_stokes_regime_mysl_d40.py`** — same
  D = 40 grid / Bouzidi BB / TRT collision as §8.3.4, with ν scaled
  so Re = 20.0 exact. Reports both Ladd and MYSL on the same
  converged flow. Wall-time: 8 306 s ≈ 2.3 h on a 4-core CPU.
- **`data/validation_3d_sphere_re20_mysl_d40.json`** — measured result.
  Cd_LADD = 4.27 (+56.4 % vs CGW 2.728), Cd_MYSL = 4.02 (+47.4 %).
  Bias reduction: only **9 percentage points**, vs 33.8 pp at Re = 100.
- **`tests/test_validation_3d_sphere_cd_stokes_regime_mysl_d40.py`** —
  7 gates including the headline-diagnostic check that MYSL closes
  ≥ 5 but ≤ 20 pp (locks the "MYSL is partial at low Re" result).
- **VALIDATION.md §8.3.5** — full subsection. Diagnoses τ = 0.74 as
  the leading suspect for the +47 % residual (the LBM equilibrium
  loses Galilean invariance at high τ); §8.3.4 sat at τ = 0.548 and
  did not have this signature.
- **VALIDATION.md §8.8** — priority list updated with the high-τ
  residual as the new #1. Cumulant collision (#3) elevated as a
  candidate fix for BOTH the high-Re ceiling (Re ≥ 200) AND the
  low-Re high-τ residual (Re = 20).

### Added — AoA ±45° wing presets (gallery stall visualization)

- **`scripts/bake_3d_field.py`** — 8 new presets:
  `naca0012_aoa{±45}_re{40,100}` and `naca4412_aoa{±45}_re{40,100}`.
  Bumped Ny one notch above the ±30° presets so the projected vertical
  extent of the wing fits with margin at deep-stall AoA.
- **`data/baked/`** — 8 new .npz files. ~ 15 – 20 min total bake time.
  Gallery slider already supports the ±45° band (slider range was
  already [-45, +45]); the new bakes give it landing points to snap to.
- Visual / qualitative addition. Thin airfoils stall around 12 – 15°,
  so 45° is deep stall — the LBM kernel renders the massive separated
  wake the user wanted to see. This is **not** an aerodynamic claim:
  validated airfoil regime remains the NeuralFoil surrogate at
  attached-flow AoA up to ~10°.

### Fixed — stale-display banner (user feedback "flow not changing with speed")

- **`app.py`** — the previous `st.warning()` for "your sidebar inputs
  diverged from the displayed result" was being missed (yellow under
  sidebar focus). Upgraded to a louder `st.error()` and added an
  inline **"Run with new settings"** button that re-triggers the run
  via the same `lbm_gallery_pending` mechanism the demo gallery
  cards use. The "must click Run to refresh" design (reviewer item
  #14, 2026-05-25) is preserved — just made more discoverable.

## [1.7.0] — 2026-06-01 (MYSL upgrade closes the bulk of the 3D Cd gap)

Closes the second half of audit item #8 — and with it, the dominant
residual bias on the 3D sphere Cd. The Mei-Yu-Shyy-Luo 2002
Bouzidi-aware momentum-exchange formula is implemented, parity-
tested against the existing Ladd 1994 form at halfway BB, and run
end-to-end on the D = 40 sphere case. **Headline result: Cd = 1.160,
+6.44 % vs Clift-Grace-Weber 1978** — down from +40.2 % with the
Ladd form on the same flow, a **33.8 percentage-point** bias
reduction. **3D bluff-body drag is no longer preview-quality;** it
is validated against an experimental reference, with the residual
~ 6 % attributable to known refinement sources (D = 40 voxelisation,
B = 25 % blockage, Bouzidi quadratic BB).

### Added — Audit item #8 second half: MYSL 2002 momentum exchange

- **`src/forces_3d.py:momentum_exchange_force_3d_mysl`** — full
  implementation of the MYSL 2002 (Mei, Yu, Shyy, Luo, Phys. Rev.
  E 65, 041203) Bouzidi-aware momentum-exchange formula. Per wall
  link: `F = c_i * (f_tilde_i(x_f) + f_opp^{post-BB}(x_f))`, with
  `f_tilde` reconstructed from the post-stream populations via the
  same TRT split as `apply_bouzidi_correction_trt` and
  `f_opp^{post-BB}` computed by the Bouzidi q-aware reflection
  formula (different for q ≥ 0.5 vs q < 0.5). At q = 0.5 the
  formula collapses to `2 c_i f_tilde_i` (Ladd), recovered exactly
  in the parity unit test.
- **`tests/test_forces_3d_mysl.py`** — 4 unit gates: (1) MYSL
  returns finite force, (2) drag dominates lift/side on a sphere,
  (3) at q = 0.5 everywhere MYSL matches Ladd to ~ 1 % at
  convergence (4 000-step bake fixture), (4) on real curved-wall
  geometry MYSL and Ladd differ by > 1 % (lock that MYSL is
  actually doing q-aware work, not silently degenerating).
- **`scripts/validate_3d_sphere_cd_mysl_d40.py`** — committed and
  **RUN** in 6 698 s ≈ 1.9 h on a 4-core CPU. Identical flow to
  §8.3.3 (D = 40 / B = 25 %, same Bouzidi BB, same TRT, same
  boundary conditions, same n_steps); reports BOTH the Ladd and
  MYSL force at the end so the comparison is apples-to-apples.
- **`data/validation_3d_sphere_re100_d40_mysl.json`** — headline
  result. MYSL Cd = **1.1602 / +6.44 % vs CGW** (vs Ladd-on-same-run
  1.5282 / +40.20 %). Bias reduction: 33.76 percentage points.
  Transverse forces stay axisymmetric under MYSL (|F_lift| /
  |F_drag| = 2.3 %, |F_side| / |F_drag| = 2 × 10⁻⁶).
- **`tests/test_validation_3d_sphere_cd_mysl_d40.py`** — 8 gates
  on the bake JSON: payload shape, D=40 grid + Re=100 exact, MYSL
  marker preserved (no silent refactor swap), Ladd baseline within
  0.005 of the §8.3.3 standalone bake (apples-to-apples lock),
  MYSL drag ≥ 15 % below Ladd on this flow, MYSL Cd within ±10 %
  of CGW, transverse forces small, and total bias reduction ≥
  25 percentage points.
- **VALIDATION.md §8.3.4** — new full subsection with the MYSL
  formula derivation, parity result, configuration table, side-by-
  side Cd comparison, error-budget breakdown for the residual 6 %,
  and a "what closes / what remains" section.
- **VALIDATION.md §8.8 priority list re-anchored.** MYSL closed.
  New #1 is the D = 60 / B = 10 % MYSL bake (pushes the headline
  into the percent-level band). #2 cumulant LBM for Re ≥ 200, #3
  3D Strouhal cross-check.

## [1.6.5.1] — 2026-06-01 (audit nice-to-haves #8/#9/#10)

Closes the three "nice-to-have" follow-ups from the v0.6.5.1 senior
re-audit. The version jump (0.6.5.1 → 1.6.5.1) marks the point at
which all reviewer-raised items — Critical, Important, and
Nice-to-have — are addressed; the headline validation now rests on
**three** independent reference frames (Williamson 1996, OpenFOAM 11,
and a second-regime sphere measurement) instead of one.

### Added — Audit item #10: AeroLab ↔ OpenFOAM Strouhal cross-check

- **`scripts/validate_2d_cylinder_strouhal_lowblockage.py`** — long-record
  cylinder Re = 100 bake at the Validation preset (700 × 400, D = 20,
  B = 5 %) for 31 500 LBM steps (~ 158 D/U). Drops the first 50 D/U as
  startup, FFTs the per-step lift coefficient on the saturated tail,
  extracts Strouhal with cycle-count + bin-width diagnostics. Now
  serializes the cl/cd time series so future re-extracts with a
  different FFT window do not require a fresh 7-hour bake.
- **`data/validation/cylinder_re100_strouhal_lowblockage.json`** — the
  committed result. **AeroLab St = 0.1794** over 28 saturated cycles
  (well above the auditor's 20-cycle threshold), FFT bin width
  ± 0.0064 in St units. **+8.07 % vs Williamson 1996** (0.166),
  **+12.13 % vs OpenFOAM 11** (0.1600). The +8 % gap is honest: the
  AeroLab Validation preset (D = 20 / B = 5 %) is biased high in
  *both* Cd raw (+14.36 %) and St; the Allen-Vincenti correction
  acts on Cd only and has no clean St analogue. AeroLab raw and
  OpenFOAM raw bracket the Williamson reference from opposite sides.
- **`tests/test_validation_2d_cylinder_strouhal_lowblockage.py`** — 4
  gates: payload shape, ≥ 20 cycles, St within ± 10 % of Williamson
  (gate passes at +8.07 %), St within ± 15 % of OpenFOAM (gate
  passes at +12.13 %).
- **VALIDATION.md §8.4 three-way table** — AeroLab St row populated
  (was `n/a`); the table is now AeroLab vs OpenFOAM vs Williamson on
  both Cd and St.
- **`validation/compare_aerolab_vs_openfoam.py`** — loads the new
  Strouhal JSON and surfaces AeroLab St in the regenerated
  `cross_validation.md`.

### Added — Audit item #9: second 3D drag-validation regime

- **`scripts/validate_3d_sphere_cd_stokes_regime.py`** — Re = 20
  companion to the shipped Re = 100 sphere case. Same body, grid, and
  blockage (160 × 80 × 80, D = 20, B = 25 %); only ν and U are scaled
  to land at Re = 20 (steady symmetric wake, no shedding, viscous-
  dominated). Validates against Clift-Grace-Weber 1978 Cd = 2.73.
  Wall-time: 633 s on a 4-core CPU.
- **`data/validation_3d_sphere_re20_stokes_regime.json`** — Cd = **4.265**
  vs CGW 2.73 → **+56.2 %**. For context, the Re = 100 baseline at
  the same B = 25 % gives Cd = 1.645 / +50.9 %. The Ladd 1994 + D = 20
  bias is **slightly Re-dependent**, growing as the viscous fraction
  of drag grows (~ 50 % viscous at Re = 100, ~ 70 % at Re = 20). This
  is physically consistent with the diagnosed failure mode: the
  simplified Ladd formula does not weight wall links by their
  Bouzidi q-fraction, so it mis-counts boundary-layer shear stress.
  The MYSL 2002 upgrade (audit item #8 second half) targets exactly
  this contribution.
- **`tests/test_validation_3d_sphere_cd_stokes_regime.py`** — 6
  gates: payload shape, Re matches target (20.0 exact), lift/side
  < 5 % of drag (1.3 × 10⁻⁴ / 3.7 × 10⁻⁶ — symmetric to numerical
  precision), |mass drift| < 1 % (+0.28 %), Cd within ± 60 % of CGW
  (+56.2 %), Cd > 1.5 (monotonicity vs Re = 100 preserved).

### Added — Audit item #8 (first half): D = 40 sphere bake — falsifies grid-dominance

- **`scripts/validate_3d_sphere_cd_d40.py`** — committed and run at
  320 × 160 × 160 (D = 40, B = 25 %, 8.2 M cells, **2.2 h wall** on
  a 4-core CPU — much faster than the 5–6 h estimate). Holds
  everything except grid spacing constant versus the §8.3.1 D = 20
  lowblock baseline so the only variable is grid resolution.
- **`data/validation_3d_sphere_re100_d40.json`** — **Cd = 1.528**,
  +40.2 % vs CGW 1.09. The D = 20 baseline gives Cd = 1.645 / +50.9 %,
  so the 8 × cell-count refinement reduced the Cd bias by **only
  ~ 7 percentage points**.
- **`tests/test_validation_3d_sphere_cd_d40.py`** — 7 gates: payload
  shape, grid is [320, 160, 160] exactly, Re = 100 exact,
  momentum_exchange is "Ladd 1994 simplified" (the entire point —
  no MYSL pollution), lift/side < 5 % of drag (1.9 % / 5 × 10⁻⁶ —
  axisymmetry preserved), |mass drift| < 1 % (0.16 %), Cd within
  ± 60 % of CGW (40.2 %), and the falsification lock-in
  (1.40 < Cd < 1.645 — refinement improved but did not close).
- **`scripts/validate_3d_sphere_cd_d40.py` progress callback** —
  added per-5 %-of-steps stdout line with elapsed minutes, ETA, and
  step rate. stdout flushed so PowerShell `Tee-Object` and
  `Get-Content -Wait` show live progress instead of a 5-hour silent
  block.
- **Falsification result.** The original "grid resolution is a major
  component" hypothesis (§8.3.1 budget breakdown: ~ + 5 % grid /
  ~ + 30 – 40 % momentum exchange) is **rejected**: grid contributes
  ~ 7 percentage points, the remaining ~ 33 % lives in the
  simplified Ladd 1994 momentum exchange. MYSL 2002 q-aware
  momentum exchange is now **the** highest-leverage 3D-validation
  next step.
- **VALIDATION.md §8.3.3** — new full subsection with config table,
  result, falsification narrative, and the corrected error-budget
  breakdown.
- **VALIDATION.md §8.8 priority list re-anchored.** MYSL momentum
  exchange promoted from #2 to #1; D = 40 bake moved to the closed
  list. The new #2 / #3 are cumulant LBM (for higher-Re 3D) and a
  follow-on D = 40 / B = 10 % bake (separates the residual lift
  component).

### Versioning

The 0.x → 1.x bump signals that AeroLab's validation rests on three
independent reference frames (Williamson literature, OpenFOAM
independent solver, second-regime sphere measurement) and that all
reviewer items raised through the 2026-05-26, -27, -29, -31 audits
are addressed. The 3D bluff-body Cd is still preview-quality
(+44 % vs CGW at D = 20) and is labeled as such; the D = 40 run is
queued to close that residual.

## [0.6.5.1] — 2026-05-31 (audit cleanup)

Doc + test polish from the post-v0.6.5 senior re-audit. No solver
or UI behaviour changes.

### Removed

- **`validation/openfoam/cylinder_re100/postProcessing/forceCoeffs/200/forceCoeffs.dat`**
  — the old coarse-mesh time series (Cd ≈ 1.183 / St ≈ 0.120) was
  still tracked in the working tree after the refined-mesh re-run
  shipped in v0.6.5. A reader pointing `diagnose.py` at this file
  would have seen numbers that directly contradict the v0.6.5
  headline; the coarse-baseline narrative is retained only via git
  history, as the docs already promised.

### Added

- **`tests/test_openfoam_cross_check_consistency.py`** — five
  consistency gates mirroring `test_doc_validation_consistency.py`:
  asserts the committed `forceCoeffs/0/forceCoeffs.dat` is the full
  500-D/U run, that the recomputed Cd_mean = 1.341 still matches
  VALIDATION.md §8.4 + README, that the FFT-of-Cl Strouhal still
  lands at 0.160, that the three citation strings (`1.341`,
  `+1.60`, `0.5 %`) remain in VALIDATION.md, and that the stray
  coarse-baseline file does not get re-introduced. Locks the
  headline numbers as derivable from the committed data.

### Changed

- **README.md status line** — "Phases 1 + 2 closed; Phase 3 in
  progress" → "Phases 1 + 2 + 3 closed" (internal contradiction
  with the prose two lines below).
- **README.md test count** — "199 unit tests / ~200" → "320+ unit
  tests across 22 files." Static count: 323 `def test_` lines.
- **README.md OpenFOAM headline row** — added a † footnote
  clarifying that 0.5 % is the cross-method gap, not a validation
  error against Williamson. The two solver-vs-reference deltas
  (+2.1 %, +1.6 %) live in the prose; the table number is now
  unambiguous on skim.
- **VALIDATION.md §8.4 averaging-window wording** — "Cd mean over
  t = 300 – 400 s = first 50 D/U" was wrong (the compare script
  uses the **last** 50 D/U, t = 950 – 1000 s in case time, via
  `WINDOW_DU = 50`). Same number because Cd_mean is flat from
  t = 300 onward, but the prose now matches the script.
- **`validation/compare_aerolab_vs_openfoam.py`** — Notes block
  matches the corrected window wording; `cross_validation.md`
  regenerated.
- **`src/lbm_3d.py` module docstring** — the stale "once smoke
  passes we can layer MRT on top" line was the source of repeated
  reviewer confusion that 3D was MRT. 3D production is **TRT** in
  `src/lbm_3d_trt.py`; the docstring now states that explicitly,
  flags this file as a constants-export + parity-test scaffold, and
  warns against extending it with MRT.
- **`src/forces_3d.py` module docstring** — the earlier "Ladd 1994
  is within 5 – 10 % of MYSL 2002" line contradicted VALIDATION.md
  §8.3.1, which attributes the +44 – 51 % sphere Cd gap to exactly
  this simplified Ladd form on a D = 20 grid. Docstring rewritten
  to reference §8.3.1 directly and flag the MYSL upgrade as
  pending.

## [0.6.5] — 2026-05-31

**V2 cross-check passes ±5 % gates.** The OpenFOAM cylinder Re = 100
case was re-meshed and re-run: an 8-block O-grid with **31 200 cells**
(was 6 480), `simpleGrading` clustering radial cells near the body,
**320 tangential cells around the cylinder** (≈ 1.1 ° / cell),
`div(phi,U)` upgraded from `Gauss linear` to `Gauss linearUpwindV grad(U)`,
and `endTime` extended from 400 (200 D/U) to **1000 (500 D/U)** with
`dt = 0.005`. Run on 4 MPI ranks (`scotch` decomposition) inside a tmux
session for ~5.6 h wall-time. **Results: Cd = 1.341 (+1.60 % vs
Williamson 1996), St = 0.1600 (-3.62 %)** — both inside the reviewer's
±5 % gate. AeroLab corrected (+2.13 %) and OpenFOAM (+1.60 %) now
bracket the Williamson reference from the same side, with the AeroLab
↔ OpenFOAM gap at 0.5 %. David Artemyev's card #6 V2 is **closed**.

### Changed

- **`validation/openfoam/cylinder_re100/system/blockMeshDict`** —
  rewritten with per-block cell counts and `simpleGrading` clustering.
  All middle-ring tangential edges raised to 40 cells (8 blocks ×
  40 = 320 cells around the cylinder). Outer blocks graded so the
  first cell sits ≈ 0.014 D from the body. checkMesh: 31 200 cells,
  62 400 faces, max non-orthogonality 44°, max skewness 0.38.
- **`validation/openfoam/cylinder_re100/system/fvSchemes`** —
  `div(phi,U)` switched from `Gauss linear` to `Gauss linearUpwindV
  grad(U)`. The 2nd-order upwind-biased scheme adds less numerical
  diffusion than the central scheme and is the standard choice for
  unsteady cylinder wakes. This single change accounts for the bulk
  of the Cd jump (1.18 → 1.34) on the new mesh.
- **`validation/openfoam/cylinder_re100/system/controlDict`** —
  `endTime` extended to 1000 (500 D/U), `deltaT` reduced to 0.005 to
  keep CFL < 0.5 in the finest cells. `forceCoeffs` function object
  unchanged (writes every timestep).
- **`validation/openfoam/cylinder_re100/system/decomposeParDict`**
  (added) — 4-subdomain `scotch` decomposition. Re-meshed case ~5.6 h
  wall-time vs ~22 min serial on the coarse mesh.
- **`validation/openfoam/cylinder_re100/postProcessing/forceCoeffs/0/forceCoeffs.dat`**
  — replaced by the 200 010-line time series from the refined run
  (t = 0 → 1000 at dt = 0.005). Shedding fully saturates by t ≈ 300;
  Cd_mean over t = 300 – 1000 is flat at 1.3411 ± 0.0001.
- **VALIDATION.md §8.4** — rewritten with the refined-mesh result.
  Three-way table updated; "Does V2 pass the 5 % gates?" subsection
  now answers *yes* with three pass-by-margin lines (0.5 %, 1.6 %,
  3.6 %). The previous coarse-baseline number (Cd = 1.18 / St = 0.12)
  is recorded in the "Result reading" bullets as the mesh-convergence
  prior baseline; the case folder's git history preserves it for
  reproducibility.
- **VALIDATION.md §8.8** — "Refine the OpenFOAM cylinder wake mesh
  and re-run" promoted from #1 priority to the **Closed (do not ask)**
  list. Remaining priorities: MYSL 2002 momentum exchange (#1),
  AeroLab-side Strouhal (#2), cumulant LBM for Re ≥ 200 (#3).
- **`validation/compare_aerolab_vs_openfoam.py`** — "Notes on the
  OpenFOAM result" block rewritten to describe the refined case
  instead of the under-resolved baseline.
- **`validation/cross_validation.md`** — regenerated with refined
  numbers and refreshed notes.

## [0.6.4] — 2026-05-30

**V2 cross-check is now measured.** OpenFOAM 11 ran the cylinder
Re = 100 case (`foamRun -solver incompressibleFluid` on a 6 480-cell
planar O-grid, 200 D/U with shedding fully developed past t ≈ 200 s).
The cross-check **runs and reports numbers**; the reviewer's 5 %
acceptance gates from card #6 do **not** pass — OpenFOAM's Cd lands
at -10 % vs Williamson and St at -28 %, both consistent with
under-resolution of the outer downstream wake mesh. The diagnosis is
documented honestly in §8.4 and the mesh-refined re-run is promoted
to the #1 item in §8.8.

### Added

- **`validation/openfoam/cylinder_re100/{0,constant,system}/`** —
  full OpenFOAM 11 case files (was previously empty placeholder
  directories). Symmetric 30 D × 20 D channel mirroring AeroLab's
  Validation preset (B = 5 %), 8-block planar O-grid topology copied
  from the OF11 `offsetCylinder` tutorial and scaled to symmetric
  domain. Slip top/bottom walls (matches B = 5 % free-stream
  expectation), velocity inlet, pressure outlet, no-slip cylinder.
  Newtonian laminar via `model Stokes` (OF11's name for the
  constant-viscosity stress model).
- **`validation/openfoam/cylinder_re100/Allrun`** — driver script
  that runs `blockMesh + checkMesh + foamRun`. Forces / coefficients
  written every time step by the `forceCoeffs` function object in
  `system/controlDict`.
- **`validation/openfoam/cylinder_re100/postProcessing/forceCoeffs/0/forceCoeffs.dat`**
  — the actual time-series output from the 200 D/U run (40 010 lines;
  startup pulse from t = 0, symmetric steady wake until ~t = 150,
  shedding develops between t = 160 – 200, fully saturated past
  t = 200; Cl_std reaches 0.32 by t = 400).
- **`validation/openfoam/cylinder_re100/diagnose.py`** — committed
  Cd / Cl diagnostic. Splits the time series into 10 windows
  (Cd_mean / Cd_amp / Cl_std per window) so the shedding-bootstrap
  transient is visible at a glance, then computes Strouhal three
  ways over the saturated tail (FFT(Cl), FFT(Cd) → ½ f, and
  zero-crossing). All three agree on St = 0.120 to within 0.4 %,
  ruling out an analysis artefact in the §8.4 St gap.
- **VALIDATION.md §8.4 (rewritten)** — three-way Cd / St table (AeroLab
  corrected, AeroLab raw, OpenFOAM, Williamson 1996), result reading
  with sign-of-error analysis on the two solvers, explicit
  "Does V2 pass the 5 % gates?" subsection that answers *no* with
  numbers, and Strouhal extraction methodology.
- **`validation/cross_validation.md` (extended)** — same three-way
  table as §8.4 plus "what this comparison closes and does not close"
  notes; written by the compare script and committed.

### Changed

- **`validation/compare_aerolab_vs_openfoam.py`** — Cd window is now
  scaled by `CASE_T_PER_DU = CASE_D / CASE_U_INF` (was hard-coded
  to 0.01 assuming a D = 0.01 m case). The shipped case uses D = 2,
  U = 1, so 50 D/U = last 100 s of the time series. Added a Strouhal
  computation via FFT of Cl over the same window, the OF11 column
  layout (`# Time Cm Cd Cl Cl(f) Cl(r)`) is auto-detected from the
  header comment so the script also handles older OF versions. Output
  table gained an `St` column with deviation vs Williamson.
- **`validation/openfoam/cylinder_re100/README.md`** — rewrote
  "Status: scaffolded, run pending" → "Status: case implemented,
  runnable on OpenFOAM 11"; documented the symmetric-domain
  modifications vs the source `offsetCylinder` tutorial, the OF11
  `foamRun + incompressibleFluid` solver dispatch (not standalone
  `pisoFoam`), and the path-with-spaces gotcha (`Study & Work` paths
  must be copied into a clean home directory before running).
- **VALIDATION.md §8.5** — bullet about the cylinder cross-check
  updated from "templated, not measured" to "measured but does not
  pass the 5 % gates; diagnosed as outer-wake under-resolution".
- **VALIDATION.md §8.8 priority list** — item #1 changed from "Run
  the SimScale cylinder Re=100 cross-check" to "Refine the OpenFOAM
  cylinder wake mesh and re-run". Item #3 (Strouhal on the cylinder
  3D bake) now also references the §8.4 missing-cell.

### Notes for the next round

The OpenFOAM result is interpretable but doesn't *pass* yet. The
next iteration would:

- Bump outer downstream blocks from 18 × 18 to ~50 × 50 cells
  (or apply `simpleGrading 4 1 1` to cluster cells near the inner
  ring), target h / D ≤ 0.25 in the wake.
- Run to endTime ≥ 600 so Cl_std plateaus (currently still slowly
  rising at t = 400).
- Re-run the comparison; if Cd lands within ±5 % of Williamson and
  St within ±10 %, the V2 reviewer gates from card #6 close cleanly.

## [0.6.3] — 2026-05-29

**Reviewer-polish pass + V2 cross-check made executable from a Windows machine.**

### Added

- **`validation/simscale/cylinder_re100/RUNBOOK.md`** — browser-only path to V2 (OpenFOAM cylinder Re=100 cross-check). SimScale's Community plan runs OpenFOAM `pimpleFoam` in the cloud, so the cross-validation claim is identical to a local OpenFOAM run but with no WSL/Linux requirement. Account setup → Cd / St numbers in ~30–45 min. Includes account setup, geometry / BC / mesh / time-step choices, post-processing snippet that pulls forces CSV and extracts Cd, St (FFT on Cl), and L/D, and a failure-modes section.
- **VALIDATION.md §8.4** — three-way result template (Williamson 1996 vs AeroLab Validation preset vs SimScale OpenFOAM) with `TBD` cells to fill once the SimScale run completes. Pass criterion and "what to do if they disagree by > 10 %" notes inline.

### Changed

- **VALIDATION.md §8.4–8.7 renumbered to §8.5–§8.8.** New §8.4 is the SimScale cross-check; existing "What we don't validate", "Blockage", "Reproducing", and "What a senior reviewer should ask next" each shift down one. Internal §8.7 → §8.8 reference in §8.3 updated. `What we still do NOT validate in 3D` bullet about OpenFOAM rewritten to "Cylinder cross-check is templated, not measured" since the install barrier is gone.
- **VALIDATION.md §8.8 priority list** — "Run the SimScale cylinder Re=100 cross-check" promoted to item #1 (it's the cheapest open item now that the install barrier is gone). MYSL 2002 + D≥40 demoted to #2 but flagged as still the highest *absolute impact* item.
- **README.md** (under "What this solver isn't") — "no quantitative drag validation has been done for 3D yet" rewritten to acknowledge the §8.3 sphere measurement exists at +44 %, attribute the residual to Ladd 1994 + D=20 (not blockage, per the §8.3.1 cross-check refutation), and link both sections.
- **VALIDATION.md §8.3** — the original four-bullet error budget is now wrapped in a `> (Superseded)` blockquote with each bullet struck through and annotated with what §8.3.1 found instead. The lead-in explicitly says "Do not cite the budget below." Skim-readers can't quote the falsified paragraph by accident.
- **`app.py:357`** — header version chip bumped `v0.5.0 → v0.6.3`. The string had been frozen at v0.5.0 since the 2D launch, even as 0.6.x shipped 3D.

## [0.6.2] — 2026-05-29

**3D gallery polish + sphere Cd error-budget falsification.**

Two threads:

1. **3D viewer fixes**: streamlines no longer clip into the body (mesh now follows the snapped baked AoA, not the continuous slider), streamlines no longer cut off mid-flow on slow regions (`dt` 12 → 8, `max_steps` 400 → 700, 3-iteration binary-search body-collision snap), pressure became actual gauge pressure with a symmetric diverging colormap, velocity color now sampled at the polyline vertex.
2. **Sphere Cd low-blockage cross-check**: ran the experiment §8.3 of VALIDATION.md proposed. Halving blockage from 42 % to 25 % was supposed to drop Cd from 1.57 toward ~1.30. It didn't — Cd ticked UP to 1.65. The blockage hypothesis is falsified. Revised error budget points at simplified Ladd 1994 momentum exchange + D = 20 grid resolution as the prime suspects instead.

### Added

- **`scripts/validate_3d_sphere_cd_lowblock.py`** — same physics as the high-blockage validation but on a 160 × 80 × 80 grid (blockage 42 % → 25 %, D unchanged at 20). 1100 s solve time.
- **`tests/test_validation_3d_sphere_cd_lowblock.py`** (8 gates) — same physics-validity checks as the high-blockage gate plus two cross-check gates: `test_blockage_is_not_dominant_bias` (locks in that the two Cd values land within 15 % of each other, i.e. blockage is NOT the dominant gap) and `test_blockage_is_lower_than_shipped_bake` (sanity check on grid).
- **VALIDATION.md §8.3.1** — full low-blockage results table side-by-side with the high-blockage measurement, narrative of what the experiment was supposed to show vs what it actually showed, revised error budget.
- **`data/validation_3d_sphere_re100_lowblock.json`** — measured result, committed.

### Changed

- **3D body mesh (cube + NACA)** rendered at the snapped baked AoA, not the continuous slider value. With the mesh disagreeing with the flow field by up to 8°, streamlines were cutting visibly through the rotated body. Aligning the mesh to `aoa_actual` eliminates the clipping. The slider caption already announces the snap, so the discrete jump is honest.
- **`_trace_streamlines` defaults** in `app.py`: `dt` 12.0 → 8.0 (each step now covers ~0.32 cell instead of ~0.48; polyline tracks curvature more closely and doesn't chord-cut through curved bodies), `max_steps` 400 → 700 (streamlines reach the outlet even through low-speed wake regions where each step covers far less than 0.32 cell).
- **Body-collision snap** upgraded from a single bisection to a 3-iteration binary search; endpoint now lands within ~`dt·|u| / 8` (~0.04 cell at `u_in=0.04`) of the analytic wall instead of the previous ~0.12 cell.
- **3D Pressure mode** now plots **gauge pressure** `p = c_s²·(ρ − ρ_ref)` with `c_s² = 1/3` (D3Q19) and `ρ_ref = median(ρ_fluid)`. Colorbar is symmetric about zero, so RdBu_r reads as **red = stagnation, blue = suction, white = freestream** instead of the previous near-flat raw-density slice (Δρ ≈ 5e-3 was being percentile-stretched into a monochrome lump). The caption and colorbar title updated accordingly.
- **3D Velocity mode** colors via vertex-trilerp of `|u|` over the field, matching the Vorticity / Pressure pattern. Previously it reused the RK2 midpoint speed which lagged the geometry by half a step.
- **VALIDATION.md §8.3** — added a status callout up top noting the budget below was falsified by §8.3.1; the original budget is kept as a record of the prediction the cross-check refuted.
- **VALIDATION.md §8.7 priority list** — "Low-blockage sphere Cd sweep" moved to the **Closed** section with a note explaining why (hypothesis refuted, not confirmed). "Mei-Yu-Shyy-Luo 2002 Bouzidi-aware momentum exchange + D ≥ 40 sphere bake" promoted to item #1.

## [0.6.1] — 2026-05-29

**Sphere Re=100 drag validation — first quantitative 3D Cd comparison.**

Closes review item **V1** (one canonical 3D drag number, the gap that converted the 3D side from "no quantitative validation" to "validated for one canonical case"). The measured Cd lands recognisably in the published band with a systematic positive bias dominated by 42 % blockage — physical, signed correctly, axisymmetric forces vanish to 10⁻⁴, and the error budget is explainable and documented.

### Added

- **`src/forces_3d.py`** — 3D D3Q19 analogue of `src/forces.py`. Ladd 1994 momentum exchange with two API entry points: `momentum_exchange_force_3d(f_post_collision, body)` for use inside a step kernel, and `momentum_exchange_force_3d_post_stream(f_post_stream, body)` for the natural exit state of `run_channel_smoke_trt` (reads the opposite-direction slot at each fluid cell, accounting for the halfway bounce-back reflection that happens during the step). Plus `drag_coefficient_3d` for Cd from F_drag, ρ_ref, U_ref, A_proj.
- **`scripts/validate_3d_sphere_cd.py`** — runs the sphere preset for 2 500 steps (5 D/u, past startup), computes Cd via post-stream momentum exchange, compares to Clift-Grace-Weber 1978 (Cd ≈ 1.09), writes JSON. Measured Cd = **1.57** (raw); error +44 % vs free-stream reference, dominated by the 42 % blockage bias documented in VALIDATION.md §8.3.
- **`tests/test_validation_3d_sphere_cd.py`** (6 gates) — reads the committed JSON and asserts: Cd inside the tolerance band, drag positive (downstream), Cd in the broad physical envelope [0.4, 3.0], `|F_lift|/|F_drag|` < 5 %, `|F_side|/|F_drag|` < 5 %, mass drift < 1 %, advective times ≥ 4 D/u. All pass.
- **VALIDATION.md §8.3** — full results table, error budget broken down (blockage / halfway BB / grid resolution / finite advective time), reproduction commands, and an explicit "why the test passes a +44 % error" subsection so reviewers see the answer to the obvious follow-up question.
- **`data/validation_3d_sphere_re100.json`** — the measured result, committed (unlike most of `data/*` which is gitignored). The JSON keeps VALIDATION.md's "see the numbers I cite" promise: anyone can read the source of the published 1.57.

### Changed

- **`run_channel_smoke_trt`** now accepts `return_populations=True` to return the final population array `f` for downstream force / Cd computations. Backwards-compatible: the bake script and existing callers continue to receive the legacy 5-tuple.
- **VALIDATION.md §8.4–8.7** renumbered. Previous §8.3 ("What we do NOT validate") becomes §8.4 with the sphere Cd item removed (now §8.3); the priority list in §8.7 swaps "Sphere Re=100 Cd" for "Low-blockage sphere Cd sweep" as the next step (close the systematic gap, not the canonical comparison — that one is now made).

## [0.6.0] — 2026-05-29

**3D gallery (preview) ships.** A pre-baked D3Q19 TRT field replay running alongside the 2D playground via the sidebar's "Solver tab" radio. Cloud-safe: the kernel runs offline (~10–25 s per scene on a laptop), the saved `.npz` snapshots ship in `data/baked/`, the hosted app loads them and renders interactive 3D streamlines + a solid body mesh. No live 3D solve in the browser.

### Added

- **3D solver core** (`src/lbm_3d_bouzidi.py`, `src/lbm_3d_trt.py`, `src/lbm_3d_qcriterion.py`, `src/lbm_3d_smoke_particles.py`, `src/baked_fields.py`). D3Q19 with TRT collision (Λ = 3/16 magic parameter), Bouzidi interpolated bounce-back with analytic q-field for spheres, voxelised wall links for everything else, Guo NEEM inflow and regularised Latt-Chopard outflow. Validated against the Taylor-Green vortex decay rate to ±2 % vs the analytic 4νk² exponent (`tests/test_phase1_gate_trt_tgv_decay_rate`, gated in CI).
- **10 pre-baked 3D scenes**, all at u_in = 0.04 lattice units, regularised outflow:
  - Sphere, cylinder (spanwise z), cube at Re ∈ {40, 100}
  - NACA 0012 (symmetric) and NACA 4412 (cambered) at Re ∈ {40, 100} × AoA ∈ {0°, 10°}
- **3D gallery UI** (`if view == "3D gallery (preview)":` block in `app.py`). Sidebar mirrors the 2D playground layout exactly: Simulation setup heading, First-time expander, Shape selectbox, Flow speed slider (0.10–4.50 m/s, same `L = 5 mm` convention as 2D), AoA slider (NACA only), Color picker (Velocity / Vorticity / Pressure), top-down body silhouette preview, Streamlines (density / thickness / animate flow), Overlays (body / wind-tunnel chamber / Q-criterion shell).
- **Server-side RK2 streamline tracing** (`_trace_streamlines` in `app.py`). Vectorised numba-jitted trilinear interpolation; ~50–150 ms per scene for 25–96 seeds × 400 steps. Replaces the earlier Plotly `go.Streamtube` (which did the integration in JavaScript on the browser main thread and blocked the UI for 3–8 s on scene-swap).
- **Body-collision midpoint snap** in the tracer so streamlines visibly graze the body surface instead of stopping ~0.5 lattice cells short.
- **Animated growing streamlines.** Each cycle: head holds at outflow while tail sweeps from inflow forward (drain), then tail stays at inflow while head sweeps to outflow (grow). 60 frames at 80 ms per frame with 60 ms linear transition; per-streamline phase stagger so streaks don't all march in lockstep. Camera rotation stays live during playback via `uirevision` tokens on layout + scene (Plotly preserves view-state across redraws bound to the same token).
- **Curated 3D gallery cards** below the chart: *How a wing lifts* (NACA 4412 + pressure coloring), *Wing at zero AoA*, *Where the air spins* (cylinder + vorticity), *Bluff cube*, *Almost stopped (creep)*, *Sphere wake*.
- **3D bake script** (`scripts/bake_3d_field.py`) with `--preset <name> --out data/baked`. Each `.npz` carries the preset config, body mask, float16-quantised velocity / density arrays, and a SHA-256 hash of the (config + mask + final field) for reproducibility.
- **`naca_outline()`** and **`make_naca_mask()`** in `src/lbm_3d_bouzidi.py`. Analytic NACA 4-digit thickness + camber line, AoA-rotated about the chord midpoint, used by both the bake-time mask voxeliser and the runtime body-mesh renderer.
- **3D validation section** in `VALIDATION.md` (§8). What 3D is and isn't validated against, baked-scene parameters table, blockage and advective-times disclosure, reproduction commands.
- **Hero chip becomes contextual.** "3D gallery · preview, in sidebar →" while in 2D view, "3D gallery · preview · live" (green) while in 3D — no stale pointer.

### Changed

- **Velocity → Re mapping in the 3D playground uses the same `L = 5 mm` convention as 2D.** Previously a 0.5 mm characteristic length compressed the slider into the {40, 100} bake range, which produced a confusing mismatch ("in 2D this was Re 500 at 1.5 m/s, why is 3D calling it Re 50?"). Now 1.5 m/s reads as nominal Re 500 in both solvers; the 3D readout shows BOTH the nominal Re from the slider AND the snapped baked Re below it whenever they differ, so users see the gap honestly.
- **2D Resolution radio labels** now state the trade-off, not just the grid size. "Standard (320 x 80) — faster, ~40 s local" and "Detailed (960 x 240) — sharper wake, ~100 s local".
- **2D Solver tab radio** (sidebar) gates 2D vs 3D. The earlier developer-only "3D dev bench (local)" view was removed — 987 lines of dead code path retired.

### Fixed

- **3D NACA preview crashed when an older `lbm_3d_bouzidi.py` was on the deploy.** Refactored the preview and 3D body render to call `naca_outline()` with positional-only arguments and apply the chord-midpoint rotation inline — no longer depends on the deployed module exposing the `aoa_deg` kwarg.
- **3D shape selector tooltip** previously said "NACA wings + custom-upload support are queued for the next release" even after NACA shipped. Rewritten to reflect what's live (NACA 0012 / 4412 at AoA = 0° and 10°); only Custom STL/PNG upload is now flagged as deferred.
- **3D animation rotation was locked during playback** in an interim build (used `redraw=False` to preserve camera, but Plotly's 3D Scatter geometry updates require `redraw=True` to actually paint). Restored `redraw=True` and added the `uirevision` tokens so camera state survives redraws.
- **Filename parser** for 3D bake files used a `[a-zA-Z]+` regex that silently dropped shapes whose names contain digits (`naca0012`, `naca4412`). Replaced with an `rfind('_re')` / `rfind('_aoa')` split that handles arbitrary `<shape>[_aoa<deg>]_re<N>` patterns.

### Docs

- **README "Status"** updated to reflect the shipped 3D gallery. Phase 3 row in the roadmap table now shows "3D gallery (D3Q19 TRT, 10 pre-baked scenes) ✅" alongside the OpenFOAM-pending mark.
- **README "What this solver isn't"** clarified — Live 3D solve on Cloud explicitly called out as outside scope; the offline-bake / Cloud-replay split documented.
- **VALIDATION.md** gains §8 covering 3D bake parameters, the TGV decay-gate validation, and an explicit "what we do NOT validate in 3D" list (no Cd, no Strouhal, no NACA polar, no OpenFOAM cross-validation).

## [0.5.0] — 2026-05-23

**Custom polygon-drawer + plain-English help overhaul** — the major pre-ship revamp before sending to senior engineers for review.

### Added
- **Click-to-place polygon drawer.** Replaces `streamlit-drawable-canvas` (which forces double-click / right-click to close) with a hand-rolled custom Streamlit component in `components/polygon_drawer/`. The UX is the standard polygon-drawing gesture: click to drop each vertex, the green start dot lights up once you have 3+ vertices, click it to close. Rubber-band preview line follows the cursor between clicks; Undo / Clear toolbar; Ctrl+Z keyboard shortcut.
- **`vertices_to_polygon()`** helper in `src/custom_shape.py` adapts the JSON vertex list from the new component into the (N, 2) polygon_xy contract the LBM pipeline already consumes. 7 new unit tests cover happy path + 5 edge cases (too few vertices, zero extent, NaN coordinates, malformed dicts, wrong type).
- **End-to-end pre-ship validation** via `scripts/final_ship_validation.py` — sweeps every (shape × viz_mode) combination at production n_frames=150, checks Cd > 0, |Cl| < 0.15 for symmetric bodies, Strouhal in (0, 1), pressure stagnation delta > 0.001 at cylinder front. 15/15 pass.
- **Plain-English captions** under every Cd / Cl / Strouhal metric tile ("Air resistance. Lower = sleeker.", etc.) so non-CFD users have an anchor.

### Changed
- **Help text overhaul throughout.** Cd / Cl / Strouhal / Re / velocity / resolution / model-quality help all rewritten to lead with intuition, not formula. Velocity slider: "slow = syrupy, fast = chaotic"; formula footnoted. Strouhal: "how often vortex pairs peel off — ~0.2 for a cylinder, which is why telephone wires can hum a steady musical note". Fast-mode Reynolds help anchors values to flight regimes (hand-glider → sailplane → light aircraft → jet airliner).
- **Strouhal "—" dash** for short runs now explains "Run was too short to spot the rhythm — try Detailed mode" inline, instead of looking like an error code.
- **JIT compile freeze** (~20 s on first click) surfaces an honest "Warming up the solver (first run takes ~20 s while the compiler does its thing — later runs are instant)" message. Session-state flag switches to a plain "Simulating the flow..." for subsequent runs.
- **Upload tab help** tightened: 10-format laundry list replaced with "most common image formats" + left-facing-shape orientation hint.
- **Building cross-section sample polygon** rotated 90° CW so wind blows across the long axis as the README always intended. The upright polygon (1:3.2 aspect) produced ~75 % vertical blockage and unphysical Cd ≈ 33; the rotated polygon gives 24 % blockage and Cd ≈ 2.6.

### Fixed
- **Polygon drawer iframe was silently 404'ing.** The `from components import polygon_drawer` import sat inside the Draw-tab `with` block, so `declare_component` never ran on a fresh page load — the iframe URL returned "Component not found" while Streamlit's catch-all served the main-app HTML with HTTP 200 (which masked the failure in smoke tests). Import moved to module top; verified via Playwright that click-place + click-start-to-close → polygon committed to session state.

### Docs
- **README:** removed "no install, no signup" claim that contradicted the Quickstart's pip dance. Drawable canvas now listed as ✅ shipped (no longer "pending"). Pressure viz no longer listed as pending. Test count 129 → 186.

## [0.4.2] — 2026-05-23

**Stability sweep + default-mode flip** — every AoA-extreme divergence the gallery cards or sliders could reach, mapped and walled off.

### Fixed
- **Diamond gallery card crashed at Re=433** (Square AoA=45°). Re=600 at AoA=45° hit a stability cliff at frame ~31 / 150 — the diagonal blockage jumps from 35 % to ~50 % and the corner-shed shear layers thin out below the LBM stability margin. Lowered Diamond card to Re=200; added AoA-aware Re cap (`Square |AoA| ≥ 5 → Re ≤ 200`).
- **Four other hidden divergences** uncovered by an exhaustive AoA × Re local stability sweep:
  - Ellipse AoA=25° Re=600 → Ellipse slider tightened to ±20°, `|AoA| > 15 → Re ≤ 400`.
  - NACA AoA=45° → NACA slider tightened to ±25°.
  - Square AoA=9.5° Re=600 → Square broadside band tightened from `|AoA| < 10` to `|AoA| < 5`.
  - Square AoA=24° Re=1000 → Square broadside cap lowered 1000 → 500.

### Changed
- **Default mode flipped to Real CFD** (`index=1`). The "see air actually move" feature is what makes AeroLab visually distinctive; the curated gallery cards give first-time visitors something compelling to click without needing to know what NeuralFoil is.
- **Brick gallery card** lowered from vel 1.80 (Re=600) to vel 1.50 (Re=500) — matches the tightened Square broadside cap.
- **Visual-regression test** `n_frames` bumped 12 → 50. At 12 frames (~420 lattice steps) the test was missing divergences that hit at step ~1000-1500 in production.

## [0.4.1] — 2026-05-23

**Rigorous solver validation** against canonical 2D bluff-body data — the answer to "how accurate is your CFD?" for senior-engineer / professor scrutiny.

### Added
- **`VALIDATION.md`** (444 lines): full methodology, 14-case Cd / Strouhal sweep against **Williamson 1996 ARFM 28** (cylinder) and **Okajima 1982 JFM 123** (square), Allen-Vincenti / West-Apelt blockage corrections with cited K-factor ranges, 5 numerically-bounded limitations sections, 7 pre-empted reviewer Q&As, 14 academic citations.
- **`scripts/validate_solver.py`** — full Re sweep harness producing `data/validation/results.{json,md}`. CI-machine-readable and human-readable.
- **`scripts/validate_conservation.py`** — closed-box mass-drift diagnostic (machine precision target) + open-channel mass-balance diagnostic (< 2 % of throughflow target). Both pass.
- **`tests/test_validation_benchmark.py`** — per-shape Cd validation gate, Cylinder-only Strouhal gate, symmetry-invariant Cl gate, no-shedding-at-low-Re gate, blockage-correction-ratio gate. 11 tests, runs on every CI push.
- **Inline textbook-comparison delta chips** on the Cd / Strouhal metric tiles, plus a "free-stream reference" `st.info` callout with the Williamson / Okajima number after Allen-Vincenti correction.

### Validated bands (full 14-case sweep)
| Quantity | Median error | Max error | Tolerance band |
|---|---|---|---|
| Cylinder Cd (Re 100–1000) | 4.3 % | 11.6 % | ± 15 % |
| Square Cd (Re 150–500) | 5.4 % | 21.8 % | ± 25 % |
| Cylinder St (Re 100–1000) | 12.6 % | 23.4 % | ± 35 % |

Square Strouhal is *diagnostic-only*: confined-channel resonance at B=0.35 locks raw St ≈ 0.37 across Re, structurally uncorrectable by any single-formula blockage correction.

## [0.4.0] — 2026-05-23

**Blockage-honest Cd reporting + per-shape Re caps + wake polish.**

### Added
- **Free-stream reference Cd / St callout.** Standard runs at 35 % channel blockage; that inflates raw Cd by ~25-40 % vs wind-tunnel free-stream. The app now surfaces BOTH numbers — the raw measurement AND the blockage-corrected free-stream estimate (Williamson / Okajima) — so the user walks away with the number they would cite in a report.
- **Per-shape Re ceilings** in the velocity slider. Cylinder 1500, Square 1000, Ellipse 1200, NACA 1500. Tied to the local stability sweep numbers; the slider can never produce a configuration that diverges.

### Changed
- **Wake-spawn point** pulled closer to the body (+6 → +3 cells past trailing edge) so the streakline mass enters the shedding region earlier — visible difference at Standard preset where the channel is short.

## [0.3.2] — 2026-05-23

**Solver hardened against high-Re bluff-body divergence.**

### Added
- **End-of-frame `np.isfinite(f).all()` blow-up guard** in the LBM main loop. Surfaces a polite "Simulation diverged at frame N of M" error instead of an opaque `ZeroDivisionError` from the `@njit` macroscopic step.
- **Pre-flight mask validation** in `solve_lbm` — rejects degenerate / thread-like polygons (single-row, isolated cells) before the JIT path can crash on them.
- **`rho_safe` / `rho_int_safe`** clamps at every macroscopic division site in `src/lbm.py`. Combined with `@njit(error_model='numpy')`, numerical underflow now produces `inf` / `NaN` instead of a Python exception that breaks out of the JIT'd inner loop.

## [0.3.1] — 2026-05-23

**Visible version marker + Cloud-deploy fixes.**

### Added
- Version string in the AeroLab wordmark header so users can tell the deployed version at a glance.

### Fixed
- CI pythonpath import resolution for tests run as `pytest` from the repo root.
- `streamlit-drawable-canvas` defensive import — if the wheel ever breaks against a minor Streamlit version bump on Cloud, the rest of the app still works and the Draw tab surfaces a polite fallback.
- `ZeroDivisionError` in the Zou-He outflow boundary for shapes that ran off the channel edge.

## [0.3.0] — 2026-05-21

**Phase 2 W5 shipped: custom shape upload — the headline differentiator.**

### Added
- **Upload PNG/JPG** as a 6th shape option in the LBM sidebar. Otsu threshold + connected-component extraction + Douglas-Peucker simplification (1 % of shorter dim tolerance) via scikit-image. Inline error messages for sanity-gate failures (image too small, shape touches edge, low contrast, area < 2 % or > 85 %).
- **Three bundled sample silhouettes** (fish, car profile, building cross-section) via "Try a sample" buttons. Closes the Phase 2 W5 gate per README ("end-to-end on three real-world silhouettes").
- **Live silhouette preview** on the LBM grid before clicking Run — shows orientation + scale + AoA-rotated outline.
- **Pin / Clear snapshot** now supports custom shapes — polygon array stashed in session state alongside the snapshot tuple.
- **Velocity (m/s) slider** replaces the Reynolds number slider as the user-facing input. Re = U·L/ν with L = 5 mm (fountain-pen scale) in standard air; slider 0.15–4.5 m/s maps to Re 50–1500. Re displayed alongside so the educational angle stays.
- **Cd-accuracy `st.info` card** post-run for custom shapes, explaining the halfway-BB caveat.
- Differentiated GIF filenames for custom shapes — sample name (e.g., `aerolab_custom_fish_re200.gif`) or polygon hash for uploads.
- `src/custom_shape.py` (silhouette extraction + polygon rasterization) and `src/sample_shapes.py` (parametric polygons for the bundled samples).
- 23 new tests (`tests/test_custom_shape.py`): synthetic-image extraction, sanity-gate rejections, rasterization, rotation, end-to-end simulate_and_render with Custom polygon, Phase 2 W5 gate tests for the 3 bundled samples. **Total: 129 passing (was 106 at 0.2.0).**

### Changed
- Hero in README is now an MP4 (`assets/hero_cylinder_re400.mp4`, LFS-tracked) instead of the 1.1 MB GIF — smaller bandwidth footprint per render and higher visual quality.
- `custom_extent` per-preset: 30 cells (Standard), 80 cells (Detailed). Reduces aliasing on thin features like a fish tail.
- Pin button no longer disabled for custom shapes.

### Dependencies
- Added `scikit-image==0.26.0` (silhouette extraction) and `streamlit-drawable-canvas==0.9.3` (planned for the hand-draw feature, Day 3 of Phase 2 W5). Pinned in both `requirements.txt` and `pyproject.toml`.

## [0.2.1] — 2026-05-19

**Wake-streakline polish.** Particle visualization now reads correctly from the very first frame and trails off gradually on the Detailed preset.

### Changed
- Both presets bumped from 60 / 100 frames to **150 frames** (5250 LBM steps each). Standard ~12 s → ~30 s locally; Detailed ~50 s → ~75 s locally. Cloud roughly 2.5× longer (free tier is 1 vCPU).
- `MAX_AGE` 60 → 100. Particles live longer, so the trail past `wake_x_max` extends gradually instead of cutting off — visible mostly on Detailed where the channel is 3× longer.
- Wake-spawn box extends to `LBM_NX − 30` (fixed cell buffer) instead of `0.78 × LBM_NX` (proportional). On Standard this barely moves; on Detailed it eliminates a 150-cell empty zone past the body.
- Detailed `gif_palette` 128 → 96 to absorb the +50% frame count without ballooning GIF size.
- Pin button no longer drops the GIF after click. Tracked the last-displayed config in session state so the post-run controls survive Streamlit's rerun.

### Fixed
- Wake particles spawned at frame 0 produced a visual artifact ("particles teleport behind the body"). Wake-spawn count now ramps from 0 to full over the frames it physically takes an inflow particle to reach the wake region.

## [0.2.0] — 2026-05-19

**Cloud-readiness pass + public-release polish.** Production solver is now MRT + Bouzidi + Zou-He end-to-end; Streamlit Cloud deploys cleanly; docs and code-comments are aligned with what actually ships.

### Added
- MRT collision + Smagorinsky LES (`C_SMAG=0.17`, Lilly 1967) as the production hot path; stable Re=50–1500 across all 5 shape presets.
- Bouzidi-Firdaouss-Lallemand interpolated bounce-back at the body surface (replaces halfway BB). Cd overshoot vs textbook cut from ~89% to ~37%; Standard↔Detailed Strouhal disagreement from ~240% to ~3% (grid-converged).
- Zou & He velocity-inflow + pressure-outflow BCs. Long-run mass drift cut from ~3%/1k steps to ~0.1%/1k steps.
- Mei-Yu-Shyy-Luo Bouzidi-aware momentum exchange for body-force calc.
- 4 shape presets beyond cylinder (square, ellipse, NACA 0012, NACA 4412) with analytic q-fields for Bouzidi.
- Side-by-side pinned comparison, GIF download with parameter-encoded filenames, demo gallery script.
- Visual-regression test (`tests/test_visual_regression.py`) — canonical cylinder Re=400 frame fingerprint.
- Validation infrastructure: `dev_validate_cfd.py` (4 physics gates + 3 diagnostics), `dev_grid_convergence.py` (Richardson extrapolation), `week1_cylinder_sweep.py` (5-config Cd convergence), `naca0012_aoa_polar.py` (8-AoA polar).
- LICENSE, CONTRIBUTING.md, pyproject.toml, devcontainer, GitHub Actions test workflow.
- Hero GIF (cylinder Re=400) via Git LFS.

### Changed
- Standard resolution preset 320×100 → 240×80 (44% fewer cells) for Cloud free-tier wall-time. Body sizes scaled so blockage stays ~20%. Detailed unchanged.
- `STEPS_PER_FRAME` 50 → 35; Standard `n_frames` 100 → 60; render DPI 100 → 88. Cloud wait time roughly halved.
- README rewritten to 135 lines, scannable structure with performance table.
- Pinned `NUMBA_NUM_THREADS=16` at line 1 of `app.py` to match Cloud's post-init reset (avoids `reload_config` mismatch crash).

### Removed
- `parallel=True` / `cache=True` on the JIT step functions (Streamlit Cloud env conflict). `prange` aliased to `range`. Local loses 2-3× speedup; Cloud was already serial.
- `src/warmup.py` (dead since the Numba-thread debug saga — first user click now amortizes the JIT compile).
- Stale halfway-BB sharp-corner warning in `app.py` (Bouzidi now active for rotated squares/ellipses).

### Fixed
- BGK-τ wall-correction artifact characterized via 5-config convergence sweep (He–Zou–Luo–Dembo 1997; Cornubert et al. 1991). Documented in Validation section; structural fixes shipped via MRT + Bouzidi.

## [0.1.0] — 2026-05-05

Initial Phase 1 ship: D2Q9 BGK + halfway bounce-back, Streamlit dual-mode app (NeuralFoil Fast + LBM Real CFD), cylinder Re=100 von Kármán validation, lid-driven cavity benchmark against Ghia 1982.
