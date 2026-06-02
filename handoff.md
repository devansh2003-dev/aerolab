# AeroLab handoff

> **Read this file first** at the start of every work session. **Update it at the end** of every substantive turn. Last updated: 2026-06-02, mid-v1.7.3 work (4 wings re-baking in background).

Current tip of `main`: **`a0d5900`** (v1.7.2, pushed to origin/main). v1.7.3 in-flight — staged code edits in working tree, awaiting bake completion before commit.

App version chip: **v1.7.3** (bumped this turn, not yet committed).

---

## 1. Landed in last edit (in-flight, v1.7.3 — NOT YET COMMITTED)

User report: "at 45 deg, both airfoils go out of the box."

### Diagnosis

Programmatically inspected 14 ±45° wing presets' body z-bbox vs `Nz`:
- 10 of 14 fit cleanly (body z-bbox `[8..24]` in `Nz=32`).
- **4 ±45° Re=200 wings clip** the top wall (body z-bbox `[21..39]` in `Nz=40`).

Root cause: v1.7.2 grew `chord` 24→32 for Re=200 wings but did **not** bump `Nz` proportionally. `chord_offset=32` placed the wing asymmetrically near the top, and the rotated chord's vertical extent (~26 LU) overshot the domain.

### Fix (uncommitted)

| File | Change |
|---|---|
| `scripts/bake_3d_field.py` | For 4 presets `naca{0012,4412}_aoa{±45}_re200`: `Nz` 40→48, `chord_offset` 32→24 (centers chord in Z). Inline comment explaining the fix. |
| `app.py` (version chip) | `v1.7.2` → `v1.7.3`. |
| `CHANGELOG.md` | New `[1.7.3]` section; amended v1.7.2 limitation #2 to mark "Fixed in v1.7.3". |
| `data/baked/naca{0012,4412}_aoa{±45}_re200.npz` | Deleted old (clipped) artifacts. Re-bakes in flight. |
| `handoff.md` | This file. |

### Background work in flight

- **v1.7.3 re-bake monitor** (`bl26ybpoq`, persistent): 4 wings sequentially. ~2 h ETA. First bake (`naca0012_aoa45_re200`) started 09:48.

### After bake lands

1. Verify body z-bbox now fits within `Nz=48` (inline `.npz` check).
2. Stage commit: code/docs + 4 .npz + handoff.
3. Provide push command to user (will not push self).

---

## 2. To edit / pending decisions

### Immediate

- [ ] Wait for v1.7.3 monitor `bl26ybpoq` to emit `v173_ALL_DONE`.
- [ ] Verify the 4 re-baked wings fit Z domain.
- [ ] Commit v1.7.3.
- [ ] Provide push command to user.

### Deferred for next turn (low priority)

- [ ] Lower-AoA wings (±5°, ±15°) at Re=20 and Re=200 if user wants full coverage.
- [ ] Cube AoA variants at Re=20 / Re=200.
- [ ] Bluff body Re=200 revival (requires cumulant collision or Ny ≥ 128).

### High-leverage research (awaiting user direction, unchanged)

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
- [ ] Optional v1.7.1 / v1.7.2 / v1.7.3 tags + releases.

### Strategic decision (open)

3D-accuracy direction vs UX/visual-richness pivot.

---

## 3. Perfected — do not touch without explicit user sign-off

### Solver kernels (verified against literature, gated in CI)

- `src/lbm.py` — D2Q9 MRT + LES + Zou-He + Bouzidi.
- `src/forces.py` — 2D Ladd 1994 momentum exchange.
- `src/references.py` — All literature values verified.
- `src/lbm_3d_trt.py` — D3Q19 TRT, Λ=3/16. TGV gate at ±2%.
- `src/forces_3d.py:momentum_exchange_force_3d_mysl` — MYSL 2002.
- `src/lbm_3d_bouzidi.py` — Sphere q-field, Bouzidi quadratic BB.
- `src/shapes.py` — Analytic shape q-fields; NACA4 decode.
- `src/custom_shape.py` — Multi-threshold thresholding.

### Tests (338 tests, all green as of v1.7.2 commit verify)

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

- Full case files + 200 010-line forceCoeffs.dat: Cd=1.341, St=0.160.

### VALIDATION.md headline numbers

- §3 2D Resolved: 5–10% gaps.
- §8.3 sphere Re=100 D=20 Ladd: Cd=1.57.
- §8.3.4 D=40 MYSL **headline**: Cd=1.160 (+6.44%).
- §8.3.5 Re=20 D=40 MYSL: Cd=4.02 (+47%).

### UI behavior

- 3-tier Re banner; provenance badges; "Pre-baked snapshot" labels;
  OpenFOAM + 3D sphere callouts;
  Stash-on-sidebar-change (2D path);
  AoA slider snaps to nearest baked value;
  Speed slider now spans 0.05–4.50 m/s, 4 snap-points for wings (Re=20/40/100/200).

### Documentation

- README.md, CHANGELOG.md, VALIDATION.md (14 citations).

### Code hygiene

- Zero TODO/FIXME/HACK in `src/` or `app.py`.

---

## Workflow rule

This file is read at turn start and updated at turn end automatically — see `feedback_handoff_file.md` in memory.
