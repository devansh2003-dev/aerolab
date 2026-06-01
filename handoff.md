# AeroLab handoff

> **Read this file first** at the start of every work session. **Update it at the end** of every substantive turn. Last updated: 2026-06-02, post-v1.7.2 work (bakes landed, awaiting commit + push approval).

Current tip of `main`: **`640d798`** (v1.7.1, 1 commit ahead of `origin/main`). v1.7.2 ready to commit (uncommitted in working tree).

App version chip: **v1.7.2** (bumped, ready to ship).

---

## 1. Landed in last edit (v1.7.2 — STAGED, NOT YET COMMITTED)

Responded to two user bugs in the 3D gallery:
- "At 45° the wing doesn't stall."
- "Wind looks the same at different airspeeds."

### Diagnosis (in CHANGELOG.md [1.7.2])

1. **No stall at ±45°** — every gallery bake ran `n_steps = 800` ≈ 1.3 chord-transits. Pure startup transient. **Fixed**: re-baked all ±30°/±45° wings at 8 000 (Re=40) / 12 000 (Re=100) steps. B batch took 60 min.
2. **Slider feels inert** — only two baked Re bands (40, 100). **Fixed**: added Re=20 (sphere, cylinder, wings) and Re=200 (wings only — bluff bodies diverged at high blockage + τ near boundary, deferred). A3 batch took 188 min.

### Files changed (uncommitted)

| File | Change |
|---|---|
| `scripts/bake_3d_field.py` | Bumped n_steps for 16 ±30°/±45° wings (B); added 22 new presets (A3: Re=20 sphere/cyl/wings + Re=200 wings); removed sphere_re200/cylinder_re200 from PRESETS with inline deferral note. |
| `app.py` (slider) | Min 0.10 → 0.05 m/s. Help text refreshed for 4 Re bands (20/40/100/200). |
| `app.py` (version chip) | v1.7.1 → v1.7.2. |
| `.gitignore` | Whitelist `wing_rebake_logs/` + `a3_bake_logs/`. |
| `CHANGELOG.md` | Full v1.7.2 entry: diagnosis + B fix + A3 scope + 3 honest known limitations. |
| `handoff.md` | This file. |

### Bake artifacts (uncommitted .npz files)

**B re-bake (overwrote existing):**
- 16 × `data/baked/naca{0012,4412}_aoa{±30,±45}_re{40,100}.npz`

**A3 new files (whitelisted by existing `!data/baked/`):**
- `data/baked/sphere_re20.npz`, `data/baked/cylinder_re20.npz` (2 bluff Re=20)
- 10 × `data/baked/naca{0012,4412}_aoa{0,±30,±45}_re20.npz` (10 wings Re=20)
- 10 × `data/baked/naca{0012,4412}_aoa{0,±30,±45}_re200.npz` (10 wings Re=200, including the pilot)

Total: 38 new/updated .npz files.

### Bake stability outcomes

- ✅ B re-bake (16/16 in 60 min): all wings converged, healthy `mean|u| ≈ 0.027`.
- ✅ Pilot Re=200 NACA0012 (803 s): stable with `u_peak = 0.054`.
- ✅ A3 Re=20 bluff bodies + 10 Re=20 wings: all clean, ~95 s each.
- ✅ A3 Re=200 wings (9 batch + 1 pilot = 10): all converged. Mirror-symmetric `max|u|` between +AoA and -AoA (verified inline via field load).
- ❌ A3 sphere_re200 + cylinder_re200: diverged to NaN at 37 % blockage + τ=0.5144. Removed from PRESETS, documented as needing cumulant collision or Ny ≥ 128.

### Known limitations (documented in CHANGELOG)

1. Re=200 wings sit close to TRT stability boundary (τ=0.5192).
2. Cosmetic wing-tip clipping at ±45°/Re=200 (Nz=40 too small for full projected chord). Flow magnitude correct; rendered wing looks truncated.
3. `u_peak` in manifest is `max(ux)` not `max|u|` — false-alarm at high AoA (caused one false divergence flag this session).

### Push status

**Two commits unpushed** to origin/main:
- `640d798` (v1.7.1) — already 1 commit ahead from last turn.
- The forthcoming v1.7.2 commit (38 .npz + 5 code/docs files).

Push command will be provided after the v1.7.2 commit lands locally. Per standing rule (`feedback_handoff_file.md`), I do not push myself — user runs the command.

---

## 2. To edit / pending decisions

### Immediate

- [ ] **User reviews v1.7.2 changes**, confirms commit + push.
- [ ] **Push command provided**: `git push origin main`.

### Follow-up improvements (low priority)

- [ ] **Fix wing-tip clipping**: bump Nz to ≥ 64 for ±45° Re=200 wings + re-bake (4 presets, ~80 min). Cosmetic only.
- [ ] **Lower-AoA wings at Re=20 / Re=200** (±5°, ±15°): 16 more bakes if user wants full coverage.
- [ ] **Cube AoA variants at Re=20 / Re=200**: deferred.
- [ ] **Bluff body Re=200 revival**: requires cumulant collision (VALIDATION.md §8.8 #3) or Ny ≥ 128 grid.

### High-leverage research (awaiting user direction, unchanged from prior turn)

- [ ] **3D accuracy push** — `3D_ACCURACY_PUSH_PLAN.md` Path A/B/C, still awaiting sign-off.

### Pre-planned bakes (`VALIDATION.md §8.8`)

- [ ] D=60 / B=10% MYSL sphere bake (Re=100).
- [ ] 3D Strouhal cross-check.
- [ ] Cumulant collision implementation.

### Audit Tasks 10/11/12

- [ ] Task 10: Split app.py monolith.
- [ ] Task 11: numpy 2 × numba pin compatibility check.
- [ ] Task 12: Cloud keep-warm ping.

### Release / publishing

- [ ] Create v1.7.0 GitHub Release.
- [ ] Optional v1.7.1 / v1.7.2 tags + releases.

### Strategic decision (open)

3D-accuracy direction vs UX/visual-richness pivot.

---

## 3. Perfected — do not touch without explicit user sign-off

### Solver kernels (verified against literature, gated in CI)

- `src/lbm.py` — D2Q9 MRT + LES + Zou-He + Bouzidi. 1423 lines.
- `src/forces.py` — 2D Ladd 1994 momentum exchange.
- `src/references.py` — All literature values verified.
- `src/lbm_3d_trt.py` — D3Q19 TRT, Λ=3/16. TGV gate passes at ±2%.
- `src/forces_3d.py:momentum_exchange_force_3d_mysl` — MYSL 2002 q-aware. Closes 33.8 pp at Re=100 / D=40.
- `src/lbm_3d_bouzidi.py` — Sphere q-field, Bouzidi quadratic BB.
- `src/shapes.py` — Analytic cylinder, ellipse, square; NACA4 decode.
- `src/custom_shape.py` — Multi-threshold Otsu/Triangle/Yen.

### Tests (47+ consistency gates, all green)

- `tests/test_doc_validation_consistency.py`
- `tests/test_openfoam_cross_check_consistency.py` (5 gates)
- `tests/test_validation_2d_cylinder_strouhal_lowblockage.py` (4)
- `tests/test_validation_3d_sphere_cd_d40.py` (7)
- `tests/test_validation_3d_sphere_cd_mysl_d40.py` (8)
- `tests/test_validation_3d_sphere_cd_stokes_regime.py` (6)
- `tests/test_validation_3d_sphere_cd_stokes_regime_mysl_d40.py` (7)
- `tests/test_forces_3d_mysl.py` (4)
- `tests/test_validation_3d_sphere_cd_lowblock.py` (8)

### OpenFOAM 11 cross-check infrastructure

- Full case files + 200 010-line forceCoeffs.dat: Cd=1.341, St=0.160 (±5% gate).

### VALIDATION.md headline numbers

- §3 2D Resolved: 5–10% gaps.
- §8.3 sphere Re=100 D=20 Ladd: Cd=1.57.
- §8.3.4 D=40 MYSL **headline**: Cd=1.160 (+6.44%).
- §8.3.5 Re=20 D=40 MYSL: Cd=4.02 (+47%).
- §8.4 OpenFOAM cross-check.

### UI behavior

- 3-tier Re banner; provenance badges; "Pre-baked snapshot" labels;
  OpenFOAM + 3D sphere callouts in Validation tab;
  Stash-on-sidebar-change (2D path, v1.7.1);
  AoA slider snaps to nearest baked value (gallery, scans `data/baked/`).

### Documentation

- README.md, CHANGELOG.md, VALIDATION.md (14 citations).

### Code hygiene

- Zero TODO/FIXME/HACK in `src/` or `app.py`.

---

## Workflow rule

This file is read at turn start and updated at turn end automatically — see `feedback_handoff_file.md` in memory. If a request would touch a "Perfected" item, raise it before doing it.
