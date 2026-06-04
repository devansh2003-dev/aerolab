# AeroLab handoff

> **Read this file first** at the start of every work session. **Update it at the end** of every substantive turn. Last updated: 2026-06-04, end of v1.7.4 pre-launch fix sprint (awaiting user manual card test before commit).

Current tip of `main`: **`a0d5900`** (v1.7.2, pushed to origin/main).
v1.7.3 was committed and pushed in a prior turn — that's why the working
tree starts the v1.7.4 sprint cleanly. v1.7.4 in-flight: code + docs +
file moves all staged in working tree, NOT YET COMMITTED.

App version chip: **v1.7.4** (bumped this turn, not yet committed).

---

## 1. Landed in last edit (in-flight, v1.7.4 — NOT YET COMMITTED)

User's 10-task pre-launch fix sprint. All 10 tasks complete + one
defensive cleanup from validator hypothesis. No solver changes — every
edit is in `app.py`, page config, internal-docs layout.

### Changes (uncommitted)

| File | Change |
|---|---|
| `app.py` (line 77-86) | TASK 7: `page_title="AeroLab — Browser-Based CFD"`, `page_icon="🌀"`. |
| `app.py` (line 358) | Version chip `v1.7.3` → `v1.7.4`. |
| `app.py` (line 1234-1257) | DEFENSIVE: flow-speed slider now uses `setdefault` + `key=` only (drops `value=_v_state`), aligning with selectbox pattern. Validator hypothesis for "silent no-op" on card clicks. |
| `app.py` (line 1273-1284) | TASK 5: snap caption mirrors AoA style — `Re ≈ 17 snapped to baked Re = 20`. |
| `app.py` (line 1290-1304) | DEFENSIVE: viz_mode radio now uses `setdefault` + `key=` only (drops `index=0`). Same fix as slider. |
| `app.py` (line 1479-1484) | TASK 6: `_trace_streamlines` wrapped in `st.spinner(":material/refresh: Tracing streamlines...")`. |
| `app.py` (~line 1995) | TASK 4: shape-dependent camera eye/lookat — sphere & cylinder pulled back (`1.50, 0.95, 0.55` vs `0.82, 0.62, 0.48` for wings) so wake visible from first render. |
| `app.py` (line 2071-2086) | TASK 6: progress bar `.empty()` deferred until AFTER `st.plotly_chart` (was: before). Bar holds at "Rendering in your browser…" while WebGL paints. |
| `app.py` (line 2075-2084) | TASK 3: `st.plotly_chart(..., key="gallery_3d_plotly")` for camera persistence across reruns. |
| `app.py` (line 2141-2180) | TASK 1: gallery cards use `on_click=_apply_3d_gallery_card` callback (was inline `if st.button(...):` which crashed with `StreamlitAPIException`). |
| `CHANGELOG.md` | New `[1.7.4]` section above v1.7.3. |
| `handoff.md` | This file. |
| `LAUNCH_CHECKLIST.md` (NEW) | TASK 9+10: pre-launch verification + UptimeRobot setup. |
| `RELEASE_NOTES_v1.7.4.md` (NEW) | TASK 2: GitHub release form markdown block. |
| `docs/internal/3D_RESEARCH_PLAN.md` | TASK 8: `git mv` from root. |
| `docs/internal/3D_PHASE0_DECISIONS.md` | TASK 8: `git mv` from root. |
| `docs/internal/3D_PHASE0_FINDINGS.md` | TASK 8: `git mv` from root. |
| `docs/internal/3D_ACCURACY_PUSH_PLAN.md` | TASK 8: `git mv` from root. |
| `docs/internal/future-ideas/cfd_convergence_predictor.md` | TASK 8: `git mv` from `future-ideas/`. `future-ideas/` deleted. |

### Validator-reported open question (pre-commit gate)

External validator confirmed the 3D card crash IS fixed. They flagged a
possible "silent no-op" (sidebar unchanged after click) but admitted
their evidence was inconclusive — their automated rapid-clicking wedged
the dev server before any rerun could complete cleanly. See
`feedback_external_reviewer_false_corruption.md` in memory.

Defensive code review verified:
- All 3 widget keys (`gallery_shape_select`, `gallery_velocity`,
  `gallery_viz_mode`) match between callback writes and widget defs.
- All 6 card shape labels match `_BUILTIN_SHAPES_3D` exactly.
- The slider + radio had `value=` / `index=` alongside `key=` (a known
  Streamlit warning pattern); aligned to the selectbox's `setdefault`
  pattern as a defensive fix even though the bug isn't confirmed.

**User is testing the cards manually.** Hold the commit until they
confirm the cards actually load their scenes (vs the validator's
inconclusive "silent no-op" claim).

### After user confirms (or denies) the card behavior

If cards work → commit + provide push command for v1.7.4.
If cards genuinely no-op → diagnose the specific failing card and
fix before commit.

### Push command (ready when user gives go)

```powershell
cd "C:\Users\USER\Desktop\Study & Work\Personal Projects\AeroLab"; git add -A; git commit -m @'
v1.7.4: pre-launch UX fix sprint

- Fix 3D preset card crash (on_click callback pattern)
- Align slider + radio with selectbox setdefault pattern (defensive)
- Plotly key for camera persistence across reruns
- Shape-dependent camera framing for sphere + cylinder
- Flow-speed snap caption mirrors AoA snap style
- Loading spinner during streamline trace
- Page title "AeroLab — Browser-Based CFD" + cyclone favicon
- Move 3D_*.md + future-ideas/ to docs/internal/
- New LAUNCH_CHECKLIST.md with UptimeRobot setup
- v1.7.4 release notes block

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
'@; git push origin main
```

---

## 2. To edit / pending decisions

### Immediate

- [ ] User manually verifies 3D card clicks load scenes (NOT no-ops).
- [ ] Commit + push v1.7.4.
- [ ] Tag `v1.7.4` + draft GitHub release with `RELEASE_NOTES_v1.7.4.md` content.
- [ ] Run through `LAUNCH_CHECKLIST.md` D-1 / D-0 items.
- [ ] Set up UptimeRobot per LAUNCH_CHECKLIST §D-0.

### Stashed (resume after launch)

- [ ] **`v1.8.0` upload-your-own-shape**. Paused 2026-06-04 mid-implementation
  per user direction. Stash: `git stash list | grep v1.8.0`. Includes
  STL bytes parser, polygon-to-extruded-mask voxelizer, upload runtime
  with Cloud-safe baking config, test coverage. ~70 % complete.

### Deferred for next turn (low priority)

- [ ] Lower-AoA wings (±5°, ±15°) at Re=20 and Re=200 if user wants full coverage.
- [ ] Cube AoA variants at Re=20 / Re=200.
- [ ] Bluff body Re=200 revival (requires cumulant collision or Ny ≥ 128).

### High-leverage research (awaiting user direction, unchanged)

- [ ] **3D accuracy push** — `docs/internal/3D_ACCURACY_PUSH_PLAN.md` Path A/B/C, still awaiting sign-off.

### Pre-planned bakes (`VALIDATION.md §8.8`)

- [ ] D=60 / B=10% MYSL sphere bake (Re=100).
- [ ] 3D Strouhal cross-check.
- [ ] Cumulant collision implementation.

### Audit Tasks 10/11/12

- [ ] Task 10: Split app.py monolith.
- [ ] Task 11: numpy 2 × numba pin compatibility check.
- [ ] Task 12: Cloud keep-warm ping.

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

### Tests (393 tests, all green as of v1.7.3 commit verify)

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
  Speed slider spans 0.05–4.50 m/s, 4 snap-points for wings (Re=20/40/100/200).
- v1.7.4 added: Plotly camera persistence via stable `key=`,
  shape-dependent default camera framing, parallel Re snap caption,
  loading spinner during trace, custom page title + favicon.

### Documentation

- README.md, CHANGELOG.md, VALIDATION.md (14 citations).
- LAUNCH_CHECKLIST.md (new v1.7.4).
- `docs/internal/` houses 3D research/planning markdown.

### Code hygiene

- Zero TODO/FIXME/HACK in `src/` or `app.py`.

---

## Workflow rule

This file is read at turn start and updated at turn end automatically — see `feedback_handoff_file.md` in memory.
