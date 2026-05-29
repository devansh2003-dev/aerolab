# AeroLab 3D Solver: Research and Implementation Plan

Status: research and planning. Phase 0 + Phase 1 complete (TGV gate
PASSED). Plan revised 2026-05-26 for the **consumer-product
re-scope** — 3D is no longer "validation tool", it is "smoke moving
around your uploaded shape that you can rotate with your mouse",
with validation moved to the backend credibility layer.

Author of original plan: external CFD review, 2026-05-23.
Revision authors: AeroLab + external review, 2026-05-26.

This document is the plan for extending AeroLab from a validated 2D
D2Q9 solver to a 3D solver. It is deliberately long: it is meant to
be the reference you build from and the thing you defend in an
interview.

---

## 0. Product framing, locked decisions, and the central tension

### 0.1 What the product is (revised 2026-05-26)

AeroLab is a **consumer-grade aerodynamics visualisation product**.
The audience is people who do not know what kinematic viscosity is
and never will. They want to see air move around a shape and play
with it. 3D extends that promise into three dimensions; the
deliverables are:

1. **See smoke particles flow around a 3D body**, rendered as a
   smooth animation.
2. **Rotate, zoom, and pan the 3D scene with the mouse**,
   interactively, inside Streamlit. No keyboard shortcuts, no
   modifier keys, no learning curve.
3. **Upload an STL / OBJ mesh** of their own shape (their drone, a
   car, a building, anything) and see smoke streaming around it.

The user interface mirrors the 2D mode the way the 2D mode itself
already works: a "Try one of these" gallery of built-in 3D shapes,
an Upload tab next to it, **one** wind-strength slider, a Run
button. No `nu`, no Reynolds, no lattice units anywhere in the UI.

The current `app.py` "3D (local, in development)" tab — with the
`Nx / Ny / Nz` select-sliders, the kinematic viscosity slider, the
Poiseuille-profile chart — is a developer **workbench**, not the
product. It stays in the branch as a debugging surface but does
not survive into the shipped UI.

### 0.2 Locked decisions

Four decisions from the original plan, all preserved:

1. **Deployment: inside the Streamlit web app.** 3D must be
   reachable from the hosted app.
2. **Compute: CPU now (Numba), GPU-ready structure.** Numba CPU
   kernel first, behind a backend seam so a GPU kernel can slot in
   later.
3. **First milestone: solver and a minimal 3D viewer in parallel.**
   (Phase 1 closed with TGV gate; Phase A now adds the viewer.)
4. **Validation: sphere first (Schiller-Naumann), then 3D cylinder
   (Williamson).** Now framed explicitly as the **backend
   credibility layer** that never appears in user-facing UI.

Five decisions added 2026-05-26 by the product re-scope:

5. **Viz: smoke-particle RK4 advection, rendered via Plotly
   Scatter3d.** Q-criterion isosurface, originally the headline
   viz, is deferred to v2 — it is the wrong abstraction for the
   "people who do not do CFD" audience. See §7.
6. **Pre-bake + replay architecture.** Built-in gallery shapes ship
   with pre-computed velocity fields. Click → instant smoke. The
   user never waits for an LBM solve on a gallery shape. See §2.
7. **Mesh upload via trimesh + voxelisation.** Honest UX tradeoff:
   built-in gallery is instant, uploaded meshes take a few minutes
   for the first solve, then play back instantly. See §2 and §3.7.
8. **Validation lives in `VALIDATION.md`, not the UI.** The sphere
   Cd vs Schiller-Naumann and cylinder Cd vs Williamson tracks
   stay. A 3D extension of `test_doc_validation_consistency.py`
   gates the doc against committed JSON. The end user never sees a
   Reynolds number; the reviewer still finds the proof.
9. **The "in-development" technical tab is retired before ship.**
   Replaced by the gallery + Upload UI mirror of 2D mode.

### 0.3 The central tension you must accept up front

A D3Q19 grid large enough to *validate* a sphere (D >= 40 cells,
low blockage) needs on the order of 0.5 to 1.2 GB just for the
population arrays. Streamlit Cloud's free tier gives the whole
process about 1 GB, and the existing app already resident
(streamlit, numpy, numba, scipy, matplotlib, plotly, scikit-image,
neuralfoil, aerosandbox, pandas, trimesh, the 2D solver) eats
300 to 500 MB of that before a single 3D array is allocated.

Conclusion, and it is non-negotiable: **3D cannot be both
interactive on Cloud and validation-grade in the same preset.**
The original plan resolved this with a two-preset split. The
product re-scope sharpens it further:

- **Gallery (pre-baked)**: solved offline on the dev laptop at the
  best grid the laptop can afford (~96³ to ~128³), velocity field
  shipped with the repo. Cloud spends zero solve time on these. A
  pre-baked sphere at 128³ in the repo is ~3 MB compressed.
- **Upload (live)**: solved live on Cloud at a coarse Interactive
  preset (~64³ to ~96³). Honest user expectation: *"first run
  takes a few minutes, then you can play with it."* This is the
  only path where the user waits.
- **Validation (offline only)**: large rectangular domain, D >= 40
  sphere, low blockage. Runs on the dev laptop. Results committed
  to the repo as JSON. Never on Cloud. Never on the UI.

Three presets, each with one job. The 2D validation saga is the
lesson: do not try to make a preset claim accuracy it cannot
deliver. The web upload preset is qualitative on purpose; that is
labelled honestly in the UI.

---

## 1. Constraints and budget analysis

This section is the governing arithmetic. Every later decision traces
back to it.

### 1.1 Memory

D3Q19 stores 19 populations per cell. At **float32** (4 bytes), the
population array `f` costs `Nx*Ny*Nz*19*4` bytes. Streaming needs either
a second buffer (`f_new`, doubling the cost) or an in-place scheme
(see 3.5).

| Grid        | cells   | f, single buffer | f, double buffer |
|-------------|---------|------------------|------------------|
| 48 cubed    | 0.11 M  | 8.4 MB           | 16.8 MB          |
| 64 cubed    | 0.26 M  | 19.9 MB          | 39.9 MB          |
| 96 cubed    | 0.88 M  | 67 MB            | 135 MB           |
| 128 cubed   | 2.10 M  | 159 MB           | 319 MB           |
| 320x160x160 | 8.19 M  | 622 MB           | 1.24 GB          |

Additional arrays: macroscopic snapshot fields (rho, ux, uy, uz) at
`4*cells*4` bytes, only needed at render time; a solid mask at
`cells*1` byte. These are small next to `f`.

**The q-field trap.** In 2D you stored a dense `(Nx,Ny,9)` q-field.
The 3D analogue `(Nx,Ny,Nz,19)` is *as large as the population array
itself* (159 MB at 128 cubed). Do not store it densely. See 3.4.

**Cloud verdict:** the interactive web preset realistically caps at
**96 cubed double-buffered (135 MB)** or **128 cubed single-buffered
(159 MB)**. 96 cubed is the safe target with comfortable margin.

**Offline validation verdict:** a 320x160x160 domain (622 MB single
buffer, 1.24 GB double) is fine on any 16 GB+ laptop and impossible on
Cloud. Validation is offline. Accept it.

### 1.2 Compute

A D3Q19 BGK step is roughly 19 reads, a local collision (~150 to 250
flops/cell), and 19 writes per cell. Single-core Numba throughput for
this kernel is realistically 50 to 150 Mcell-updates/s on a modern
laptop core, and Streamlit Cloud's shared 1-vCPU is slower, call it
20 to 50 Mcell-updates/s.

Per-run cost at ~5000 steps, Cloud at 30 Mcell/s:

| Grid      | per step | 5000 steps |
|-----------|----------|------------|
| 64 cubed  | 0.009 s  | ~45 s      |
| 96 cubed  | 0.030 s  | ~2.5 min   |
| 128 cubed | 0.070 s  | ~6 min     |

So **96 cubed is the interactive sweet spot**: it costs about what your
current 2D Standard preset already costs on Cloud (~3.3 min), which
users tolerate. 128 cubed is too slow for interactive use.

Offline validation, 320x160x160, ~10000 steps, local single-core at
80 Mcell/s is ~17 min/run. With Numba `parallel=True` (a `prange` over
the outer axis, which you disabled *only* because of a Cloud-specific
`NUMBA_NUM_THREADS` error, not a real limitation) you get 4 to 8x, so
2 to 4 min/run. A 5-case sphere sweep is well under an hour offline.

### 1.3 What the budget forces

- float32, not float64, for the population array (halves memory, and
  LBM tolerates it for these Re; verify in Phase 0).
- 96 cubed interactive ceiling; D ~ 24 sphere there; qualitative only.
- Validation is a separate offline preset, results committed as JSON.
- The offline path may use Numba `parallel=True`; the Cloud path stays
  serial. The backend seam (3.6) carries this difference.

---

## 2. Architecture: solve once, replay many

The product re-scope adds a SECOND axis to the split. Originally the
plan had just two presets (Interactive vs Validation). The revised
architecture has three roles, each separated by where the expensive
work happens.

### 2.1 The three presets

```
RESOLUTION_PRESETS_3D = {
    # Gallery: dev solves OFFLINE at the best grid the laptop affords,
    # commits the velocity field. Cloud never solves; it loads + replays.
    "Gallery":      Nx=Ny=Nz=128, float32, parallel, OFFLINE solve,
                    fields shipped as data/3d_fields/<shape>.npz

    # Upload: user solves LIVE on Cloud (or local) when they upload.
    # Coarser than Gallery because the user is waiting in their browser.
    # Cached per-mesh-hash via st.cache_data so re-uploads are instant.
    "Upload":       Nx=Ny=Nz=64-96, float32, serial,  LIVE on demand,
                    cached in st.cache_data keyed on mesh hash

    # Validation: D >= 40 sphere or spanwise cylinder, low blockage.
    # Dev runs offline, results committed to data/validation/ as JSON.
    # NEVER on Cloud. NEVER on the UI. Backend credibility only.
    "Validation":   Nx=320, Ny=Nz=160, float32, parallel, OFFLINE only,
                    JSON committed under data/validation/
}
```

### 2.2 The solve / replay split

The architectural decision that makes the product viable on Cloud's
1 GB / 1 vCPU is that **the LBM solve and the smoke render are two
separate stages**, not one pipeline:

```
  Solve stage (expensive — minutes)
    geometry mask
       |
       v
    LBM TRT kernel (Phase 1)
       |
       v
    steady velocity field  u(x, y, z) = (ux, uy, uz)
       |
       v
    save .npz  OR  pass to replay stage in-process

  Replay stage (cheap — milliseconds per frame)
    u(x, y, z)  +  particle seed scheme
       |
       v
    RK4 advection of N particles per frame
       |
       v
    Plotly Scatter3d  ->  Streamlit  ->  user mouse rotate/zoom
```

The replay stage is what the user actually sees. It runs in real
time, on every camera move, on every re-seed. The solve stage runs
**once** per shape — offline for gallery, on first upload for user
meshes — and never again unless the geometry or wind speed changes.

This is the same separation 2D has internally between the LBM solve
and the GIF render. 3D just makes it more explicit because the solve
side is heavier and the render side is interactive.

### 2.3 Storage and cost

- **Gallery `.npz`**: 3 components × 128³ × 4 bytes raw = 25 MB,
  ~6 MB with `np.savez_compressed` and float16 quantisation. Five
  gallery shapes = ~30 MB committed to the repo. Git LFS for
  `*.npz` if it grows past a comfortable size.
- **Upload cache**: `st.cache_data(max_entries=4)` keyed on the
  SHA-1 of the uploaded mesh bytes. Each entry ~25 MB. Total cap
  ~100 MB, well inside Cloud's process budget alongside the 2D
  caches.
- **Validation JSON**: ~kB per case, same as 2D. No memory concern.

### 2.4 Where the 2D UX patterns map

The 2D app already solved the "consumer wraps over technical solver"
problem. The 3D mode reuses every one of those patterns:

| 2D pattern (existing)                          | 3D analogue                                            |
|------------------------------------------------|--------------------------------------------------------|
| Sample-shape gallery cards (`sample_shapes.py`)| Built-in 3D gallery loading pre-baked `.npz` fields    |
| Custom image upload (`custom_shape.py`)        | STL/OBJ mesh upload (trimesh + voxelise, see §3.7)     |
| Velocity slider (m/s, hides ν / Re)            | One "wind strength" slider with the same mapping       |
| `@st.cache_data` keyed on shape + Re + AoA     | `@st.cache_data` keyed on shape + wind + mesh hash     |
| GIF render via Pillow                          | Plotly Scatter3d (interactive, no GIF for v1)          |
| `VALIDATION.md` headline tables                | 3D section in same file, gated against committed JSON  |

If 2D was the prototype, 3D is the same blueprint with a different
solver and renderer underneath.

---

## 3. Research phase: open questions and how to resolve each

Phase 0 exists to answer these before any production code. Each item
lists the decision, the options, a recommendation, and how to resolve.

### 3.1 Lattice: D3Q15 vs D3Q19 vs D3Q27

- D3Q15: cheapest, but known to suffer spurious anisotropy and
  instability. Reject.
- D3Q19: the standard memory/accuracy sweet spot for incompressible
  flow. 19 populations, well documented, every reference BC exists for
  it.
- D3Q27: better Galilean invariance and rotational isotropy at higher
  Mach/Re, but 42% more memory and flops.

**Recommendation: D3Q19.** Your Re band is laminar and low Mach
(U = 0.1, Ma ~ 0.17). D3Q27's advantages do not pay for themselves
against your memory budget. Resolve by: Krüger et al. (2017) ch. 3,
plus confirming the 96 cubed memory line in 1.1 with D3Q19 numbers
(already done above).

### 3.2 Collision model: BGK vs TRT vs MRT

Your 2D solver uses MRT plus Smagorinsky. For 3D the realistic options:

- **BGK**: one relaxation rate. Simplest. Unstable as tau approaches
  0.5. Keep it only as the reference path for the equivalence test.
- **TRT (two-relaxation-time, Ginzburg)**: two rates, one even
  (sets viscosity), one odd (free). The "magic parameter"
  Lambda = (1/s_e - 1/2)(1/s_o - 1/2). With **Lambda = 3/16** the
  bounce-back wall sits at exactly the mid-link location independent
  of viscosity. Nearly BGK cost. Far simpler than 3D MRT.
- **MRT (d'Humieres 2002, D3Q19)**: 19-moment transform. Best
  stability, heaviest to implement and test.

**Recommendation: TRT for the 3D production kernel, BGK as the
reference path.** Rationale you can defend in an interview: (1) the
validated band is laminar, Re <= ~300, where TRT is amply stable;
(2) TRT with Lambda = 3/16 gives viscosity-independent wall placement,
which directly improves Cd accuracy, the exact metric you validate;
(3) TRT is dramatically cheaper to implement, test and reason about
than 3D MRT, and on a memory and compute budget that simplicity is
worth real money. MRT stays documented as the upgrade path if you ever
push high-Re. **Do not port Smagorinsky LES into the 3D validated
path** — the 2D review already established it injects spurious
dissipation at laminar Re; make it opt-in for the interactive toy
only.

Resolve by: a Phase 0 prototype running TRT and BGK on a 3D
lid-driven cavity and a Taylor-Green vortex across the target tau
range, comparing stability and the TGV decay rate.

### 3.3 Boundary conditions in 3D

- **Obstacle wall**: bounce-back, with Bouzidi interpolated bounce-back
  for the sphere. **The sphere q-field is a quadratic root, identical
  in structure to your 2D `cylinder_q_field`.** |x_f + q*c_i - c|^2 = R^2
  expands to the same A q^2 + B q + C = 0 you already solve. This
  generalises with almost no new math.
- **Inflow / outflow**: full 3D Zou-He is messy (many unknown
  populations, awkward edge and corner cases). Do not port it. Use
  **non-equilibrium extrapolation (Guo, Zheng, Shi 2002)** or the
  **regularized BC (Latt & Chopard 2006)**. Both are simple, robust,
  and standard in 3D. Prototype both in Phase 0, pick on mass
  conservation and stability.
- **Lateral faces**: for the sphere, free-slip or periodic to minimise
  blockage interference in the validation preset; no-slip walls are
  acceptable in the interactive preset. For the cylinder, **spanwise
  periodic** (see 5.2).

### 3.4 q-field storage strategy

Dense `(Nx,Ny,Nz,19)` is forbidden (1.1). Two storage modes,
selected per geometry source:

- **Analytic on-the-fly** for sphere and spanwise-periodic cylinder
  (the validation-track geometries): compute q inside the kernel
  from the surface equation. Zero storage. Cheap (one quadratic).
  See 3D_PHASE0_DECISIONS.md D-4 for the formula.
- **Sparse wall-link list** for uploaded meshes (the product-track
  geometry, in scope as of 2026-05-26): a 1-D array of
  `(cell_x, cell_y, cell_z, direction_i, q)` tuples covering every
  fluid cell at the body's surface. Wall-link count is O(L²) for a
  body of linear size L cells; tens of thousands of entries on a
  96³ grid with a body filling ~half the domain. Tiny next to the
  population array.

**Recommendation (revised 2026-05-26):** analytic for the
validation track; sparse list for uploads. v1 uploads use **full-way
bounce-back** (q = 0.5 assumed for every wall link), matching what
2D's `custom_shape.py` does for arbitrary polygons. Bouzidi
interpolation against the voxelised mesh surface is the v1.1
follow-on; first ship something that works, then refine.

### 3.7 Mesh upload pipeline (added 2026-05-26)

A new sub-project the original plan deferred. v1 must accept user
STL / OBJ uploads and produce a usable solid mask + sparse
wall-link list. Steps and decision points:

- **Library: `trimesh`** (pure-Python, ~5 MB Cloud-image cost, no
  compiled binary on Cloud). Standard for STL/OBJ/PLY/GLB. Accept
  the dependency.
- **Repair pipeline**: many real-world meshes are non-watertight,
  flipped, mis-scaled. Pre-process with
  `trimesh.repair.fix_normals()`, then attempt
  `trimesh.repair.fill_holes()`. If still not watertight, do not
  refuse the upload — proceed with a "result may be approximate"
  badge in the UI. This matches the 2D silhouette extractor's
  approach (most of that module is validation code, not
  algorithm).
- **Normalise scale + position**: centre the mesh's bounding box
  on the LBM domain centre; rescale so the streamwise bounding-box
  dimension is `~0.3 * Lx`, matching the 2D blockage convention.
  Auto-align the longest axis to the wind direction; provide a
  flip / rotate button in the UI as the override.
- **Voxelise**: `trimesh.voxelized(pitch=lattice_spacing)` gives a
  `VoxelGrid` whose `.matrix` is the boolean solid-cell array
  the kernel needs. Voxelisation cost is O(triangles * N) where N
  is the grid size; a 1000-triangle mesh at 96³ takes ~1 s.
- **Decimation safety valve**: if voxelisation would take more
  than ~5 s, run
  `trimesh.simplify_quadric_decimation(target=2000_faces)` first.
  Quality loss is irrelevant since we are about to voxelise at a
  fixed grid pitch anyway.

Resolve by: a Phase A prototype that loads a known STL (a Stanford
bunny or similar), runs the full pipeline, and writes the mask to a
file the existing slice viewer can render.

### 3.5 Streaming pattern and memory

Double-buffering (`f` and `f_new`) is simple and is fine at 96 cubed
(135 MB). If you want the 128 cubed interactive option, the
**in-place AA-pattern / esoteric twist (Geier & Schonherr 2017)** uses
a single `f` array, halving population memory. Recommendation: start
with double-buffering for correctness and simplicity; evaluate the
in-place scheme in Phase 1 only if you decide 128 cubed is worth it.
Do not start with the clever scheme.

### 3.6 GPU-ready backend seam

You chose "CPU now, GPU-ready." Do not over-engineer this. The seam is:

- A `Solver3DBackend` interface: `allocate(grid, params) -> state`,
  `step(state) -> state`, `download_macroscopic(state) -> (rho, u)`.
- State arrays are ndarray-like. numpy now; cupy is API-compatible as
  a near drop-in later; NVIDIA Warp has its own array type and would
  need a thin adapter.
- Geometry, masks and q-fields are built host-side in numpy and
  uploaded once.
- The time loop, BC application order, and snapshot cadence live in
  backend-agnostic orchestration code.
- Implement `NumbaCPUBackend` now, with a serial variant (Cloud) and a
  `parallel=True` variant (offline). A future `CuPyBackend` is then a
  kernel transcription, not a redesign.

The single most useful GPU-ready habit: keep the per-cell collision
math as one small, heavily-tested, well-commented reference function.
A GPU port becomes a transcription of that function. Do not abstract
beyond this. A premature plugin architecture is wasted effort.

---

## 4. Implementation phases (solver and viewer in parallel)

Each phase has an explicit exit gate. Do not start the next phase until
the gate passes.

Phase ordering is now split into two tracks running in parallel: a
**product track** that builds toward "user sees smoke around their
upload" and a **credibility track** that builds toward "the
solver's Cd matches the literature." Each track has its own gate
sequence. The product track is what unblocks users; the credibility
track is what unblocks reviewers.

### Phase 0 — Research and decisions [DONE 2026-05-26]
Four throwaway prototypes confirmed every locked Phase 0 question.
See `3D_PHASE0_FINDINGS.md` for the numbers. **Gate met.**

### Phase 1 — Core TRT kernel + Taylor-Green gate [DONE 2026-05-26]
D3Q19 + TRT (Λ = 3/16) + BGK reference path, periodic box, no
geometry, AoS layout, double-buffered. **Gate met** (TGV decay
within ±0.25 % across ν ∈ {0.005, 0.01, 0.02}).

### Phase 1.5 — Kernel optimisation backlog [DEFERRED]
Equilibrium-de-duplication via 19 named scalars (saving the ~2x
redundant work the AoS kernel currently does per cell), plus
consolidation of the four kernel variants behind an
`@njit(inline='always')` per-cell helper. Lifts 96³ throughput
from ~1.8 Mcell/s toward the plan's ~30-50 Mcell/s target.
Scheduled to land alongside Phase 2 geometry so both can be
validated against the sphere Cd gate at once.

---

**Product track (the user-facing path):**

### Phase A — Smoke-particle viz + Plotly + pre-bake (~2 weeks)
The first thing the user actually sees. Three sub-deliverables:

1. **Pre-bake driver** (`scripts/prebake_3d_field.py`): solves a
   geometry offline using the Phase 1 TRT kernel, snapshots the
   steady (ux, uy, uz) field, writes
   `data/3d_fields/<shape>.npz` (compressed). Re-usable for every
   gallery shape.
2. **Smoke-particle advection** (`src/smoke_3d.py`): RK4 integrator
   of N tracer particles through a precomputed (ux, uy, uz) field,
   with trilinear interpolation. Continuous upstream re-seeding,
   lifetime-capped, deterministic seed pattern. ~50 ms per frame
   for 500 particles, pure numpy.
3. **Plotly viz integration**: a single `Scatter3d` per frame
   inside Streamlit, rotates / zooms natively, no GIF needed for
   v1. Wire into a new "3D" mode that mirrors the 2D mode's
   sidebar UX.

**Gate:** loading a pre-baked sphere field in Streamlit renders
smoke particles flowing past the sphere at interactive frame
rate; the user can rotate the camera with the mouse.

### Phase B — Mesh upload pipeline (~2 weeks)
The hard new piece. Sub-deliverables per §3.7:

1. **`src/mesh_upload_3d.py`**: trimesh load + repair + centre +
   rescale + voxelise → boolean mask + sparse wall-link list.
2. **Streamlit Upload tab**: file uploader, the same UX as 2D's
   custom-shape upload. Shows a preview of the voxelised body
   (slice view) so the user can confirm orientation before
   committing to a multi-minute solve.
3. **Live-solve hook**: when the user clicks Run on an upload,
   call the Phase 1 TRT kernel + Phase 2 boundary pass at the
   Upload-preset grid. Cache the resulting field in
   `st.cache_data` keyed on the mesh SHA-1 so re-uploads are
   instant.
4. **Honest UX wait message**: "Your shape is solving — about
   3 minutes the first time, then it is yours to play with"
   shown in the spinner.

**Gate:** uploading a Stanford-bunny STL produces a watertight
voxelised mask, the solver runs without crashing, the smoke
viz shows reasonable wake structure behind the bunny.

### Phase C — Gallery curation + UI polish (~1 week)
Pick and pre-bake the 3D gallery: sphere, simplified car,
building, wing. Each gets a card in the same style as the 2D
sample-shape gallery. The "3D (local, in development)" workbench
tab is removed; the 3D mode now looks exactly like the 2D mode.

**Gate:** a first-time visitor can click "Try a sphere" and see
smoke flowing around it without reading any documentation.

---

**Credibility track (the backend / portfolio path):**

### Phase 2 — Sphere geometry + Bouzidi q + Guo NEEM (~2 weeks)
Sphere mask, analytic Bouzidi q-field per D-4, Guo NEEM as a
separate boundary pass after `trt_periodic_step_aos`, momentum-
exchange force in 3D (direct generalisation of `src/forces.py`).
**Gate:** sphere at Re = 100 shows clean axisymmetric steady
wake; force routine returns a physically sane Cd; mean lift ≈ 0.

### Phase 3 — Sphere offline validation (~1 week)
`Validation` preset (320 × 160 × 160, D ≥ 40), sweep at Re = 20,
50, 100, 200, 300. Compare Cd to Schiller-Naumann.
**Gate:** Cd within ~10 % at D ≥ 40, low blockage; mean lift
~0 below Re ~ 210.

### Phase 4 — Spanwise-periodic cylinder validation (~1.5 weeks)
**The portfolio centrepiece.** Cd and St vs Williamson 1996,
mode-A wake instability vs Barkley & Henderson 1996.
**Gate:** at Re = 300 the 3D Cd sits below the 2D over-prediction
and nearer experiment; mode-A wake instability visible. This
directly closes the "2D ceiling" the 2D VALIDATION.md was
scoped around.

---

### Phase D — Validation writeup + 3D `VALIDATION.md` (~0.5 week)
3D section in `VALIDATION.md` to the same honesty standard as
the 2D rewrite: pre-baked gallery and live uploads explicitly
labelled qualitative; sphere + cylinder Validation-preset numbers
as the headline. New `tests/test_doc_validation_consistency.py`
checks gate the 3D doc against the committed JSON the same way
the 2D ones do.

---

**Total: roughly 10 to 12 weeks** (up from the original 8 to 10
because Phase B mesh upload is now in scope). Honest pacing for an
interview: 3D done properly is about three months, not three weeks.

**If the timeline gets tight**: the product track (Phases A + B +
C) is the ship-blocking work; the credibility track (Phases 2 + 3
+ 4 + D) is what keeps the validation story honest. Both must
land before claiming "3D works." If you must triage, Phase 4
(the spanwise cylinder) is the protected milestone — it is the
single most defensible 3D claim, and without it the answer to
"is your 3D real" is hand-wavy.

---

## 5. Validation methodology

### 5.1 Sphere (primary, Phase 3)

- Drag coefficient: Cd = F_drag / (0.5 * rho * U^2 * A), frontal area
  A = pi * R^2.
- Reference: the **Schiller-Naumann (1933) correlation**,
  Cd = (24/Re) * (1 + 0.15 * Re^0.687), valid to Re ~ 800. Cross-check
  against the standard experimental sphere drag curve (Clift, Grace &
  Weber; Roos & Willmarth 1971).
- Flow regimes, used as physics sanity checks (Johnson & Patel 1999):
  steady axisymmetric below Re ~ 210; steady non-axisymmetric
  (double-thread wake) ~210 to 270; unsteady hairpin shedding above
  ~270. Below Re ~ 210 the mean lift must be ~0 and the wake
  axisymmetric. That is a strong, free correctness test.
- Why sphere first: Schiller-Naumann is a tight, universally accepted
  correlation, the geometry has no span ambiguity, and the analytic
  q-field is trivial.

### 5.2 3D cylinder, spanwise-periodic (Phase 4)

- Geometry: a cylinder spanning the full domain in z, with **periodic
  spanwise boundaries**. This models the infinite cylinder that
  Williamson's data describes, and it isolates the 3D wake instability
  without finite-end effects.
- The spanwise domain length matters: it must be at least one mode-A
  wavelength (~3 to 4 diameters) or the instability cannot develop.
  Set Lz >= ~6D to be safe. Cite Williamson 1996 and Barkley &
  Henderson 1996 (Floquet analysis of the wake transition).
- The money result: a 2D solver over-predicts cylinder Cd above
  Re ~ 190 because it cannot shed the 3D mode-A and mode-B
  instabilities that relieve the load. A correct spanwise-periodic 3D
  solver should reproduce the Cd drop and the St discontinuity at the
  mode-A transition. If it does, you have directly closed the
  "2D ceiling" limitation the 2D VALIDATION.md is scoped around. That
  is a genuinely strong, defensible portfolio claim.
- Keep blockage low in the validation preset (free-slip or far lateral
  walls). The 2D blockage saga is the lesson: design the validation
  domain for low blockage from the start.

### 5.3 Validation discipline

Reuse the 2D machinery. Offline sweep writes JSON. README and
VALIDATION.md 3D headline tables are gated against that JSON by an
extension of `test_doc_validation_consistency.py`. Tolerance bands set
from the literature first, not drawn around the data. Scope the claim
to where the comparison is honest. You already did this once; do it
the same way.

---

## 6. Testing strategy

Extend, do not abandon, the 2D test discipline.

1. **3D Taylor-Green vortex decay** (the core verification). The TGV
   has an analytic decaying solution; total kinetic energy decays at a
   rate set by nu. Run it, fit the decay rate, compare to the
   prescribed viscosity. This is the rigorous "is the viscosity
   correct" test and it gates Phase 1.
2. **3D Poiseuille duct flow**: analytic parabolic profile, checks the
   wall BC.
3. **Conservation**: closed periodic box, mass to machine precision.
4. **BGK kernel vs pure-numpy reference**: bit-level-ish equivalence,
   mirroring your 2D `test_lbm.py` discipline.
5. **Analytic momentum-exchange force** on a hand-computable config.
6. **Sphere symmetry**: zero mean lift below Re ~ 210.
7. **Isotropy spot checks** on the D3Q19 weights and lattice vectors.
8. **Grid convergence** on the sphere (two resolutions, Richardson).

CI keeps the same shape: a fast regression guard on a tiny grid every
push; the offline validation sweep run by hand and committed.

---

## 7. Rendering plan

The original plan had Q-criterion isosurfaces as the headline viz.
The product re-scope makes that wrong for the audience: people who
do not do CFD do not know what a Q-criterion isosurface is. They
know what smoke in a wind tunnel looks like. The revised viz plan
inverts the priority.

### 7.1 The v1 viz: smoke-particle streams (Phase A)

**Algorithm.** Once the LBM solve has produced a steady velocity
field `u(x, y, z) = (ux, uy, uz)`, integrate massless tracer
particles through it with RK4:

```
  p(t + dt) = p(t) + dt * RK4(u, p(t))
```

with **trilinear interpolation** of `u` at the particle position.
`dt` is chosen so a particle moves <~ 0.5 cell per step at the
inflow speed.

**Seeding.** Continuous re-seeding from a small region just
upstream of the body, lifetime-capped at `Lx / u_in` so particles
don't pile up at the outflow. ~300-500 particles per scene at v1,
deterministic seed pattern so the visual repeats cleanly. The
seed scheme matches the 2D streakline seeds — a row of dots
upstream, no wake injection (per the
[[feedback_streamline_design]] memory).

**Why this is cheap.** The expensive work (the LBM solve) produced
a static (ux, uy, uz) array, ~25 MB at 128³ float32. Advecting 500
particles per frame for ~100 frames is ~50 ms of pure-numpy work
total. The user can spin the camera, change the seed pattern, or
re-run the animation — all free, no solver in the loop.

**Why this is the right viz for the audience.** Everyone has seen
a wind-tunnel smoke test. Nobody outside CFD knows what a
Q-criterion isosurface is. The smoke particles ARE the 3D analogue
of the 2D streaklines AeroLab already ships, and they read the
same way visually: trails that bend around the body, get sucked
into the wake, recover downstream.

**Implementation** (`src/smoke_3d.py`, Phase A):
- `init_particles(seed_box, n_particles) -> (3, N) positions`
- `advect_step(positions, u_field, dt) -> new_positions` (numba
  @njit, RK4, trilinear interp; ~30-50 lines)
- `render_plotly_scene(positions, body_mesh, t) -> Figure` —
  Plotly `Scatter3d` of particles + `Mesh3d` of the body surface
  + camera defaults set to a nice 3/4 view.

### 7.2 Render technology: Plotly Scatter3d (Phase A)

`plotly.graph_objects.Scatter3d` with `mode="markers"`, ~500
points per frame. Streamlit renders it via `st.plotly_chart`,
which gives the user **rotate / zoom / pan with the mouse,
natively, no extra dependency, no GIF**. Plotly is already in
`requirements.txt`.

Animation: replay through frames either by updating the chart in
a Streamlit loop or by attaching `frames=[...]` to the figure and
letting Plotly's built-in animation slider do it. The latter is
cheaper on Cloud (one figure send instead of N).

Body surface: `trimesh.Trimesh` (for uploads) or a parametric
sphere / cylinder triangulation, fed to `plotly.graph_objects.
Mesh3d`. Light grey with low alpha so smoke is the visual hero.

### 7.3 Slice viewer (debugging only, Phase 1 done)

Already exists at `scripts/dev_3d_phase1_slice.py`. Stays in
`scripts/` as a developer debugging tool — extracts the z-mid
plane of `u` and renders with matplotlib. Not user-facing. Useful
for confirming a TGV decay or an uploaded mask voxelised
correctly before committing to a multi-minute solve.

### 7.4 Q-criterion isosurface (DEFERRED to v2)

The original viz plan. Mathematically interesting, visually
impressive to a CFD audience, **wrong for this audience**.
Deferred. If we ever ship a "advanced view" toggle for technical
users, this is what it shows. v1 ships smoke only.

Algorithm if/when we come back to it: compute
`Q = 0.5 * (|Omega|^2 - |S|^2)` on the 3D velocity field, extract
the isosurface with `skimage.measure.marching_cubes`, render with
Plotly `Mesh3d`. scikit-image is already in `requirements.txt`.

### 7.5 GIF export (DEFERRED to v2)

The original plan called for a rotating-camera GIF. With
interactive Plotly the user can rotate themselves, so a GIF is
nice-to-have, not essential. Defer until users specifically ask
for it.

### 7.6 No new dependencies for v1 rendering

- Plotly: already a dep.
- numba: already a dep.
- numpy: already a dep.
- (trimesh for the upload pipeline — see §3.7 — is one new dep,
  but for parsing not rendering.)

---

## 8. Risk register (revised 2026-05-26 for the product re-scope)

| ID  | Risk                                                    | Mitigation                                                                                              |
|-----|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| R1  | Cloud out-of-memory at solve time                       | float32, Upload preset ≤ 96³, gallery shapes pre-baked offline, `st.cache_data` per-mesh-hash           |
| R2  | Upload solve too slow on Cloud's 1 vCPU                 | Upload preset capped at 64³ if needed; honest "first run is ~3 min" UI message; cache per upload        |
| R3  | q-field memory blowup                                   | analytic for sphere/cylinder, sparse wall-link list for uploads, never dense                            |
| R4  | Rendering scope creep                                   | smoke particles only for v1; Q-criterion + GIF export deferred to v2                                    |
| R5  | Validation needs resolution Cloud cannot give           | validation runs offline only, results committed as JSON; never on Cloud, never in the UI                |
| R6  | Numba 3D JIT time / register pressure                   | keep kernels lean, `cache=True` set, named-scalars rewrite scheduled Phase 1.5                          |
| R7  | Timeline: 3D is ~3 months with the upload sub-project   | phase gates, Phase 4 cylinder is the protected portfolio milestone, product track can ship without it   |
| R8  | "Web 3D" oversells as validated                         | gallery + Upload UI labels both as "qualitative visualisation"; never call them validated               |
| R9  | **Upload pipeline robustness** (NEW)                    | trimesh repair, auto-rescale, auto-orient, never refuse an upload; warn if mask is degenerate           |
| R10 | **Pre-bake fields drift out of date** (NEW)             | regenerate via `scripts/prebake_3d_field.py` if the kernel changes, gate on a checksum in `data/3d_fields/` |
| R11 | **Plotly Scatter3d perf on Cloud** (NEW)                | cap at ~500 particles per frame; if Plotly struggles, fall back to a frame-by-frame static GIF          |
| R12 | **Mesh upload abuse vector** (NEW)                      | trimesh load on user input — `trimesh` rejects malformed binaries cleanly; cap file size at 25 MB       |

R8 + R9 are the ones to watch.

R8: you spent five review rounds making the 2D validation honest.
Do not let "smoke around your mesh in the browser" quietly become
a claim that the smoke is quantitatively accurate. It will not be
at Cloud-grid resolution. Say "visualisation" not "simulation
result" in the UI from day one.

R9: the 2D custom-shape upload taught the lesson that the
robustness layer is where most of the engineering goes. Budget
real time for it; the algorithm is easy, the
weird-meshes-users-actually-upload part is not.

---

## 9. References for the research phase

- Kruger et al. (2017), *The Lattice Boltzmann Method: Principles and
  Practice* — the primary text, already cited in `src/lbm.py`. 3D, D3Q19,
  TRT, boundary conditions.
- d'Humieres, Ginzburg, Krafczyk, Lallemand, Luo (2002),
  "Multiple-relaxation-time lattice Boltzmann models in three
  dimensions" — D3Q19 MRT (the upgrade path).
- Ginzburg, Verhaeghe, d'Humieres (2008), two-relaxation-time scheme —
  TRT and the magic parameter.
- Guo, Zheng, Shi (2002) — non-equilibrium extrapolation BC.
- Latt & Chopard (2006) — regularized boundary conditions.
- Bouzidi, Firdaouss, Lallemand (2001) — interpolated bounce-back,
  already used in 2D, generalises to 3D.
- Geier & Schonherr (2017) — in-place "esoteric twist" streaming.
- Schiller & Naumann (1933) — sphere drag correlation.
- Johnson & Patel (1999), "Flow past a sphere up to Re = 300" — sphere
  wake regimes.
- Williamson (1996) — cylinder, already cited.
- Barkley & Henderson (1996) — Floquet analysis of the cylinder wake
  3D transition (mode A and B).

---

- `trimesh` documentation (https://trimsh.org/) — mesh load,
  repair, voxelisation. (Added 2026-05-26 for the upload sub-project.)

---

## 10. Decisions locked (revised 2026-05-26)

**Originally locked (2026-05-27, before the product re-scope):**

1. v1 geometry scope: sphere plus spanwise-periodic cylinder only.
   Arbitrary uploaded 3D shapes explicitly deferred.
2. Timeline: validation-first, ~8 to 10 weeks. Phase 4 (cylinder)
   protected.
3. 2D K-recalibration: 2D backlog, not 3D-blocking.

**Revised + added (2026-05-26, product re-scope):**

1. **v1 geometry: GALLERY (built-in shapes) + UPLOAD (user STL/OBJ).**
   The original "sphere + cylinder only" decision is preserved for the
   *credibility* track. The *product* track adds a 4-6 shape gallery
   (sphere, simplified car, building, wing, + room to grow) and a
   trimesh-backed upload pipeline. See §3.7 and the new Phase B.
2. **v1 viz: smoke particles, not Q-criterion isosurface.** The
   audience does not know what Q is; everyone knows smoke. See §7.
   Q-criterion isosurface deferred to v2 if/when a "technical view"
   toggle ships.
3. **Pre-bake + replay architecture.** Gallery shapes solved
   offline, fields shipped as `.npz` in the repo. Uploads solve
   live, cached per-mesh-hash. Validation always offline. See §2.
4. **Timeline: ~10 to 12 weeks** (up from 8-10 because of the
   upload sub-project). Product track (Phases A, B, C) is
   ship-blocking; credibility track (Phases 2, 3, 4, D) is
   reviewer-blocking. Both must land before claiming "3D works."
5. **The "in development" technical 3D tab is retired before ship.**
   Stays as developer workbench during build; gallery + Upload UI
   replaces it for users.
6. **2D K-recalibration: still 2D backlog**, unchanged.

The full technical resolution of every Phase 0 research question is
in the companion document `3D_PHASE0_DECISIONS.md`. Phase 0 + Phase
1 closed. The next coding action is Phase A: the smoke-particle
advection prototype + pre-bake driver + Plotly viz integration.
