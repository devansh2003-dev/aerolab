# Roadmap: make the 3D gallery "perfect"

**Date:** 2026-06-11 · **Source:** 15-agent audit-to-action workflow
(`wf_7ed7aeb3-98a`, ~688k tokens, 5 expert lenses → synthesis → adversarial
verification of the top 8 → final plan). This file is the durable roadmap so the
analysis survives compaction. Work it batch by batch, each change gated + diffed.

> **Headline:** the 3D gallery is genuinely well-built and honest (forward-only
> RK2 streamlines from one inflow plane, body-snapped meshes, percentile-clipped
> colormaps, body-zeroed Q field, disciplined cache layer). What it is NOT yet is
> a *studio render* or a frictionless first-ten-seconds. The path to "perfect" is
> a sequence of small, independent, gated diffs — visual richness + truthfulness,
> NOT new CFD features (audience wants to SEE air).

---

## Load-bearing guardrails (read before ANY 3D edit)

1. **ANIMATION INDEX-0 CONTRACT.** `app.py` hardcodes `_stream_trace_index = 0`
   and animation frames patch ONLY that index. Any new trace (glow, comet, lit-Q,
   ground grid) MUST be **appended after** the crisp streamline — never prepended.
   After each visual edit, confirm `scene_traces[0]` is still the crisp streamline.
2. **CACHE-KEY INTEGRITY.** `_trace_streamlines_cached` (and any new frame/seed
   cache) is keyed `(scene_name, n_seeds)` with underscore-prefixed array args
   that are NOT hashed. So the seed sampler / frame builder must be **pure
   deterministic functions of exactly those keys** (seed any RNG by
   `scene_name`; no AoA/viz/time inputs), or stale geometry returns silently.
3. **SINGLE COLORBAR.** The streamlines own the only colorbar (`showscale=True`).
   Every new `Scatter3d`/`Mesh3d`/`Surface` with color data must set
   `showscale=False`.
4. **DO NOT TOUCH the camera block** (`scene.camera` + `uirevision` tokens,
   ~app.py:2113-2139/2218) during visual/perf work — leaving it exactly as-is is
   the only zero-risk state until the component fix.
5. **BOUNDARY LAYER.** Clip all seeds to `[3, N-4]` so they stay out of the layer
   the 99.5-pctile colormap deliberately excludes.
6. **1-vCPU FREE TIER.** Keep new traces static (no per-frame payload doubling),
   bound any new cache with `max_entries`, cap glow width ~16px (WebGL quirk).
7. After each diff: load the default sphere + one wing-AoA scene, toggle
   Animate/Q, confirm motion + shell + (later) comet still render.

---

## Camera-reset verdict: DEFER (P3). Do NOT ship the "cheap mitigation".

The camera resetting on every viz/AoA/speed change is the #1 documented wart, but:
- The "emit `scene.camera` only on shape change" mitigation was **already tried,
  validator-tested, confirmed not to fix it, and reverted** (handoff.md:182-196).
- Root cause is structural: `st.plotly_chart` **re-mounts** the Plotly instance
  on every rerun, dropping ALL browser camera state; `uirevision` only persists
  within Plotly's own animation loop, never across re-mounts.
- Worse, as written the mitigation **REGRESSES**: dropping the `scene.camera`
  override on same-shape reruns leaves the re-mounted figure with no camera, so
  Plotly falls back to its generic centered default — losing the deliberately
  tuned sphere/cylinder pulled-back framing (+1.50 streamwise, look-at +0.25)
  that makes the wake visible. Zero upside, real downside.
- A sibling relayout-listener component is **architecturally impossible**
  (sandboxed iframes can't reach the `stPlotlyChart` iframe).
- The ONLY clean fix is **render-figure-in-component**: a self-contained static-
  iframe component (the sanctioned `components/polygon_drawer/` precedent, no pip
  dep) that owns its own Plotly.js, receives figure JSON, calls `Plotly.react`,
  and wires `plotly_relayout` → `setComponentValue` to persist/re-apply camera.
  **L effort, risk 4** (re-implementing the chart bridge + Streamlit theming +
  bundling a ~3.5 MB Plotly.js asset + version drift), and it is an *interaction*
  fix, not an art-direction win. Also: passing the full 3D figure JSON (tens of
  thousands of streamline verts) through the component bridge every rerun is a
  real perf risk — could be slower than today.
- **Schedule only as a standalone P3 AFTER the visual/perf wins land**, and after
  extracting scene assembly into helpers gives it a clean `_compute_camera` seam.

---

## Batch 1 — quick wins (DONE 2026-06-11, uncommitted, verified parse+ruff)

All three are single additive/local edits, verified `proceed`, with zero
interaction with the animation/camera/cache/baked-field contracts.

- [x] **denser-hero-sphere** (`app.py` sphere `_theta/_phi`): 33×17 → 64×33
  lat/long. Smooth round hero sphere instead of polygonal facets. Static trace,
  serialized once.
- [x] **guard-q-isosurface** (`app.py`, `if show_q:`): wrap the
  `_q_isosurface_cached` call in `try/except Exception` with a quiet caption.
  Defense-in-depth (mirrors C-7) so a transient MemoryError/skimage edge case on
  the free tier can't replace the whole scene with a red traceback.
- [x] **plotly-modebar-trim** (`app.py` `st.plotly_chart`): add
  `config=dict(displaylogo=False, displayModeBar="hover")`. Cleaner beauty-first
  frame. NOTE: `scrollZoom` is already default-true for gl3d, so intentionally
  not set.

---

## Batch 2 — high-impact visual + truthfulness (items 1-5 DONE 2026-06-11, uncommitted)

Implemented per the user's "implement, I'll review pushed" choice. Parse + ruff
clean; the two contract-sensitive pieces (seeding, frame builder) were
extracted via AST and **headless-tested** (count/bounds/determinism/body-
concentration for seeding; frame-count/shape/head-count/determinism/full-reveal
for the animation builder). **Still needs the owner's in-browser eyeball for
aesthetics** — the code is correct, the *look* is unverified headlessly.

- [x] **body-aware-seeding** — new `_inflow_seeds_3d` helper (deterministic via
  `hashlib` digest, NOT the salted builtin `hash()`); dilation bands (shape-
  agnostic, works for spanwise cylinder/wing, not just compact sphere); test
  shows 67-92% of seeds now land near the body.
- [x] **streamline-depth-glow-layer** — appended after the crisp trace
  (`showscale=False`, opacity 0.16, width cap 16px).
- [x] **lit-q-shell-material** — inline lighting + lightpos on the Q `Mesh3d`
  (no hoist needed; reuses the body light position), opacity 0.32 → 0.36.
- [x] **particle-head-on-animation** — warm comet-head markers trace, appended
  last, added to every `go.Frame` alongside the crisp streamline.
- [x] **cache-animation-frames** — new `_build_anim_frames_cached`
  (`max_entries=4`); the 60-frame build is memoised on `(scene_name, n_seeds)`.
- [ ] **arclength-stagnation-cap** — HELD (not in the user's listed Batch 2 scope;
  touches the shared tracer across all 4 shapes → needs the live-loop eyeball).

Original approach notes (kept for reference):

1. **body-aware-seeding** (`app.py` seed block, replaces uniform meshgrid).
   *Biggest truthfulness+beauty lever.* Replace the full-cross-section grid with:
   ~60% on a jittered dilated body-silhouette grid (`silhouette =
   field.body.any(axis=0)`), ~25% on a halo ring, ~15% sparse background. MUST:
   (a) make jitter a **deterministic function of `scene_name`**
   (`np.random.default_rng(seed=hash(scene_name)&0xFFFFFFFF)`) to keep the
   `(scene_name, n_seeds)` cache key valid; (b) `np.clip` seeds to `[3, N-4]`;
   (c) keep total count == `n_seeds` exactly; (d) read `field.body` (baked,
   AoA-correct). Payoff: streamlines actually wrap the body + trace the shear
   layer/wake instead of flying past as straight freestream lines. Upstream of
   everything below.
2. **streamline-depth-glow-layer** (append after the crisp trace, NOT prepend).
   Second `Scatter3d`, same `flat_x/y/z`, width `min(line_width*2.6, 16)`,
   `opacity~0.18`, `showscale=False`, reuse `flat_color`+`_colorscale`. Keep it
   STATIC (don't mirror per-frame). Luminous "lit air" ribbons. No recompute.
3. **lit-q-shell-material** (`app.py` Q `Mesh3d`). Add `lighting` +
   `lightposition`; HOIST `_body_lighting/_body_lightpos` defs above the
   `if show_q:` block (currently defined after it → NameError otherwise). Use a
   softened translucent material (ambient~0.5, diffuse~0.8, specular~0.2,
   fresnel~0.3), NOT the opaque-body specular. If adding `intensity=verts[:,2]`
   gradient, MUST `showscale=False`. Vortex shell reads as a form-revealing
   membrane.
4. **particle-head-on-animation** (comet head). One `Scatter3d` markers trace
   (≤96 markers), built only when `animate_flow`, APPENDED after existing traces;
   reuse the `_head_idx` math; add this trace's index to every `go.Frame`'s
   data+traces list. Do AFTER glow+seeding so the frame-trace plumbing is touched
   once with full context.
5. **cache-animation-frames.** Extract the per-frame array construction into an
   `st.cache_data` helper keyed `(scene_name, n_seeds)` with underscore array
   params (mirror `_trace_streamlines_cached`), `max_entries~8`. Every sidebar
   interaction while Animate is on gets snappier (stops re-doing 60 array copies
   on the shared vCPU). Do AFTER comet so the cached fn returns the final frame
   contract.
6. **arclength-stagnation-cap** (`_trace_streamlines` core). Accumulate per-seed
   arc length, terminate past ~3× domain length; raise effective stop to
   ~`0.02*u_in_meta`; keep `max_steps=700` hard cap. Eliminates ragged stub
   streamlines that stall in the recirculation bubble. **Touches the SHARED
   tracing core — regression-check all four shapes.**

---

## Batch 3 — P2 polish (kept in backlog, after Batch 2; real low-risk gains)

ground-grid depth plane · q-shell empty-feedback · drop-preview relabel ·
soften Re-snap captions · non-colour viz legend · background radial gradient ·
second fill light · velocity-colormap wake contrast · colorbar polish ·
vorticity signed/streamwise · q-level physical default · placeholder-shapes
clarity · onboarding expander · first-load hero scene · surface cards above
fold · suppress progressbar on cache hit.

## Deferred / dropped

- **render-figure-in-component** (the real camera fix) — P3, see verdict above.
- **extract-scene-assembly-helpers / body-overlay-dispatch / rename show_sphere**
  — P3 code-quality; no user payoff alone; do opportunistically, and sequence the
  scene-assembly extraction just before the component camera fix.
- **slim/adaptive animation frames** — highest-risk perf item; `cache-animation-
  frames` removes most felt cost at lower risk. Revisit only if payload still
  hurts.
- **fix-animation-double-control (copy)** — dropped as churn (caption already
  names the button).
