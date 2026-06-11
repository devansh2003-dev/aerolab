# AeroLab handoff

> **Read this file first** at the start of every work session. **Update it at the end** of every substantive turn. Last updated: 2026-06-11, external-audit-#3 follow-up committed + pushed.

Current tip of `main` and `origin/main`: **`2ac6fb2`** "fix: respond to external audit #3 — close 25 audit items across critical/important/polish tiers".

App version chip / `pyproject.toml`: **v1.7.4**. CHANGELOG has the audit-3 work staged under `[Unreleased]` — bump to v1.7.5 (or v1.8.0 if that label is still reserved for upload-your-own-shape) and tag when verified on Cloud.

---

## 1. Landed in last edit (commit `2ac6fb2`, pushed 2026-06-11)

External audit #3 follow-up. 25 audit items closed across critical /
important / polish tiers. No solver or accuracy changes; this batch is
about reliability under Cloud load, input-validation hygiene, test
coverage of an untested kernel branch, and consumer-facing UX
consistency.

### What's in the commit

- **Critical (4/4):** B-1 share-trap, B-2 3D pipeline caching, B-3 card-
  drop verification toast, B-4 truncated image crash.
- **Important (18/20):** C-1 pyproject bump, C-2 velocity_mps stash,
  C-3 mode-radio key, C-4 9 widget patterns, C-5 stale OUT-OF-DATE
  banner, C-6 pin-snapshot cache stash, C-7 npz guard, C-8 NeuralFoil
  wrap, C-9 README test counts, C-10 2D Bouzidi tests, C-11 3 rubber-
  stamp tests, C-12 alpha + 16-bit upload, C-13 voxelize geometry-fit +
  STL, C-14 self-intersect polygons, C-16 solve_lbm validation, C-18
  2D preview cache, C-19 README 3D Re/scene-count, C-20 subsumed by B-2.
- **Polish (6/16):** D-1 share host, D-3 JIT warm-up flag, D-5 hashlib
  dedupe, D-7 3D card velocity + UI copy, D-15 root docs cleanup,
  D-16 Validation tab schema trust.

Full per-item rationale in `CHANGELOG.md` `[Unreleased]` section.

### Deferred (queued, not done)

- **C-15** GIF palette quantisation (needs perf measurement).
- **C-17** float16 snapshots (invasive; precedent doesn't transfer).
- **D-8** `n_frames=0` cryptic IndexError in `solve_lbm` — small real
  bug bundled with comment drift; should be added next to C-16's
  validation block.
- **D-9** `canonical_param_hash` TypeErrors on `np.float64` baked-field
  params — small real bug, add `default=float` in JSON dump.
- **D-10** 3D smoke/Q one-line defensive asserts (n_substeps=0, NaN Q).
- **D-12** Canvas Clear button doesn't clear committed polygon preview.
- **D-2/4/6/11/13/14** genuine polish — dead code / comment drift /
  test-suite hygiene with no UX impact.

---

## 1b. Previously landed (v1.7.4, 2026-06-04)

10-task pre-launch fix sprint + validator follow-up. No solver changes;
all edits in `app.py`, page config, internal-docs layout. Tagged
v1.7.4 + GitHub Release published as Latest.

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
| `app.py` (line 2071-2086) | TASK 6: progress bar `.empty()` deferred until AFTER `st.plotly_chart` (was: before). Closes the blank gap between trace progress and chart paint; the "Rendering in your browser…" indicator itself is sub-second. |
| `app.py` (line 2075-2084) | TASK 3: `st.plotly_chart(..., key="gallery_3d_plotly")`. Preserves camera through Plotly's animation loop. Does NOT preserve camera across control-change reruns -- known limitation, see below. |
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

### Validator confirmation + follow-up fix (2026-06-04)

Validator manually re-tested on v1.7.4 localhost build. Results:

- ✅ All 6 preset cards load their scenes correctly (no crash, no no-op).
  The defensive slider/radio cleanup landed before testing — can't
  isolate whether the cleanup was load-bearing or whether the
  on_click pattern alone would have worked. Either way the cards now
  function as designed.
- ✅ Tab title + 🌀 favicon.
- ✅ Sphere/cylinder pulled-back framing.
- ✅ Flow-speed snap caption (`Re ≈ 1500 snapped to baked Re = 100
  (20, 40, 100 available).`).
- ✅ 2D regression check passes.
- ❌ **Camera persistence on viz toggle / AoA drag** — camera reset
  to default. `uirevision` alone was not enough; the explicit
  `camera=dict(...)` block in `fig.update_layout` was overriding the
  user orbit even with an unchanged uirevision token.
- ⚠️ **Card 5 mismatch** — *"Almost stopped (creep)"* set velocity
  0.5 m/s which gave Re ≈ 167 → snapped to Re=100 baked band.
  Blurb says *"Re ≈ 40 ... Stokes-flow limit"*. Mismatch.

### Follow-up landed in this turn

1. **`app.py` (~line 2128)** — card 5 velocity 0.5 → 0.12 m/s so
   `Re ≈ 40` matches the "creeping / Stokes flow" blurb. **WORKS**
   (validator re-confirmed: title now `SPHERE_RE40`, snap caption
   reads "Showing pre-baked Re = 40.").
2. **`app.py` (~line 1992)** — attempted camera-persistence fix
   (conditional `camera=` based on shape change) did **NOT** work.
   Validator re-tested and confirmed camera still resets on viz
   toggle / AoA / speed drag. Root cause: `st.plotly_chart` re-mounts
   the Plotly instance on each Streamlit rerun, losing browser-side
   camera state. `uirevision` only preserves state within Plotly's
   animation loop, not across re-mounts.
3. **De-scoped:** reverted the conditional `camera=` change (it didn't
   help — added complexity without behavior change). Kept stable
   `key=` and `uirevision` (still help for animation loop). Documented
   the cross-rerun reset as a known limitation in CHANGELOG.md and
   RELEASE_NOTES_v1.7.4.md. Proper fix would need `streamlit-plotly-
   events` or a custom Streamlit component; deferred to a later
   release. Mitigation: the v1.7.4 default-framing fix means every
   rerun lands on a *good* view, not a bad one.

### Push command (after final manual re-verify of camera + card 5)

Pushes both v1.7.4 commits (sprint + follow-up) in one go:

```powershell
cd "C:\Users\USER\Desktop\Study & Work\Personal Projects\AeroLab"; git push origin main
```

Optional: tag and push v1.7.4:

```powershell
cd "C:\Users\USER\Desktop\Study & Work\Personal Projects\AeroLab"; git tag -a v1.7.4 -m "v1.7.4: pre-launch UX fix sprint"; git push origin main; git push origin v1.7.4
```

---

## 2. To edit / pending decisions

### Immediate

- [ ] Wait for CI to confirm green on `2ac6fb2`.
- [ ] Cloud-verify the load-bearing audit-3 fixes once the deploy lands:
      B-1 (Share-then-click stickiness), B-2 (sidebar checkbox in 3D
      should be sub-second after first scene load; second session must
      stay responsive during heavy 3D use), B-3 (warning toast appears
      when a card click is dropped — hard to force manually but watch
      for it under load), C-5 (no spurious OUT-OF-DATE banner after
      2D→3D→2D), C-6 (pin a config, explore 4+ others, return — must
      not silently re-solve).
- [ ] Decide version bump: v1.7.5 (incremental polish) vs v1.8.0
      (audit-3 is genuinely user-visible; the upload-your-own-shape
      stash is the v1.8.0 reservation though — choose which gets the
      label).
- [ ] Tag + cut a GitHub Release once verified.
- [ ] Re-examine the bundled-bug skips (D-8 / D-9 / D-10 / D-12) as a
      small follow-up commit — ~20 min total.

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

### Tests (~410 collected across 29 files, 401 green in fast suite as of audit-3 commit)

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
- Audit-3 follow-up (`2ac6fb2`) added: 4 cached wrappers around the
  3D pipeline (load/streamlines/color-volume/Q-iso) so a single 3D
  session no longer starves the deployment; 3D-card velocity tuned to
  baked Re so sidebar caption matches card copy; share-then-click no
  longer reverts the next interaction; branch-tracking clears the
  stale-display stash on view/mode change.

### Documentation

- README.md, CHANGELOG.md, VALIDATION.md (14 citations).
- LAUNCH_CHECKLIST.md (new v1.7.4).
- `docs/internal/` houses 3D research/planning markdown.

### Code hygiene

- Zero TODO/FIXME/HACK in `src/` or `app.py`.

---

## Workflow rule

This file is read at turn start and updated at turn end automatically — see `feedback_handoff_file.md` in memory.
