# AeroLab 3D Solver: Research and Implementation Plan

Status: research and planning only. No code written yet.
Author of plan: external CFD review, 2026-05-27.

This document is the plan for extending AeroLab from a validated 2D D2Q9
solver to a 3D solver. It is deliberately long: it is meant to be the
reference you build from and the thing you defend in an interview.

---

## 0. Locked decisions and the central tension

Four decisions were fixed before this plan was written:

1. **Deployment: inside the Streamlit web app.** 3D must be reachable
   from the hosted app.
2. **Compute: CPU now (Numba), GPU-ready structure.** Numba CPU kernel
   first, behind a backend seam so a GPU kernel can slot in later.
3. **First milestone: solver and a minimal 3D viewer in parallel.**
4. **Validation: sphere first (Schiller-Naumann), then 3D cylinder
   (Williamson).**

**The central tension you must accept up front.** A D3Q19 grid large
enough to *validate* a sphere (D >= 40 cells, low blockage) needs on the
order of 0.5 to 1.2 GB just for the population arrays. Streamlit Cloud's
free tier gives the whole process about 1 GB, and the existing app
already resident (streamlit, numpy, numba, scipy, matplotlib, plotly,
scikit-image, neuralfoil, aerosandbox, pandas, the 2D solver) eats
300 to 500 MB of that before a single 3D array is allocated.

Conclusion, and it is non-negotiable: **3D cannot be both interactive
on Cloud and validation-grade in the same preset.** The plan therefore
defines two 3D presets, and this is the same pattern you already shipped
in 2D:

- **`Interactive3D` preset** (web app): ~96 cubed grid, float32,
  coarse sphere (D ~ 24). A qualitative wake toy. Not validation-grade.
  This is the 3D analogue of your 2D Standard preset.
- **`Validation3D` preset** (offline only): large rectangular domain,
  D >= 40 sphere, low blockage. Runs on your machine, results committed
  to the repo as JSON. This is the 3D analogue of your 2D Resolved
  preset.

Anyone who internalised the 2D validation saga will accept this
immediately. Do not try to make the web preset validation-grade. It
will not work and pretending it does is the exact mistake the 2D
review spent five rounds unwinding.

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

## 2. Architecture: the two-preset split

```
RESOLUTION_PRESETS_3D = {
    "Interactive3D":  Nx=Ny=Nz~96,  D_sphere~24,  float32, serial,   web app
    "Validation3D":   Nx=320 Ny=Nz=160, D_sphere=40, float32, parallel, offline
}
```

This mirrors `Standard` vs `Resolved` in 2D. The web UI exposes only
`Interactive3D`. `scripts/validate_solver_3d.py` uses `Validation3D`
and writes `data/validation/results_3d_sphere.{json,md}`. The 3D
section of VALIDATION.md is anchored to the offline JSON, gated by an
extension of `test_doc_validation_consistency.py`. You already have
this exact machinery for 2D; reuse it.

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

Dense `(Nx,Ny,Nz,19)` is forbidden (1.1). Options:

- **Analytic on-the-fly** for the sphere: compute q inside the kernel
  from the sphere equation. Zero storage. Cheap (one quadratic).
- **Sparse wall-link list** for arbitrary geometry: store only the
  cells that have a wall link, as `(cell_index, direction, q)`. For a
  sphere surface in a 128 cubed box the wall-link count is O(D^2),
  a few thousand entries. Tiny.

**Recommendation:** analytic on-the-fly for the sphere (fast path);
sparse list as the general mechanism. This also decides scope: v1 is
sphere plus cylinder only, both analytic. Arbitrary uploaded 3D meshes
are explicitly out of scope for v1 (they need 3D voxelisation and a
sparse q-field builder; defer).

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

### Phase 0 — Research and decisions (~1 week)
Resolve every item in section 3 with small throwaway prototypes.
Deliverable: a one-page decision memo plus prototype scripts.
**Gate:** lattice, collision, inflow BC, q-strategy, streaming pattern
all decided and written down.

### Phase 1 — Core kernel + verification + slice viewer (~2 weeks)
Build the D3Q19 kernel: BGK (reference) and TRT (production), periodic
box, no geometry yet. In parallel, build the slice-plane viewer: extract
the z-mid (and optionally x/y-mid) plane and feed it straight into your
existing 2D matplotlib render pipeline. Near-zero new render code, and
it lets you watch the solver while you debug it.
**Gate:** the 3D Taylor-Green vortex decays at the analytic rate set by
nu, to within ~2% (see 6.1). This proves the viscosity is right.

### Phase 2 — Geometry, BCs, force, isosurface viewer (~2 weeks)
Sphere mask plus analytic Bouzidi q-field. Inflow/outflow BC chosen in
Phase 0. Momentum-exchange force in 3D (a direct generalisation of your
2D `forces.py`). In parallel, build the isosurface viewer: compute the
Q-criterion field, run `skimage.measure.marching_cubes` (scikit-image
is already a dependency), render the mesh with Plotly `Mesh3d` (Plotly
is already a dependency and is interactive in Streamlit).
**Gate:** sphere at Re = 100 renders a clean, steady, axisymmetric
wake; the force routine returns a physically sane Cd; mean lift is ~0.

### Phase 3 — Sphere validation, offline (~1 week)
`Validation3D` preset, `scripts/validate_solver_3d.py`, sphere sweep at
Re = 20, 50, 100, 200, 300. Compare Cd to Schiller-Naumann.
**Gate:** Cd within ~10% of Schiller-Naumann at D >= 40 and low
blockage; zero mean lift below Re ~ 210.

### Phase 4 — 3D cylinder validation (~1.5 weeks)
Spanwise-periodic cylinder, Cd and St vs Williamson. This is the
portfolio centrepiece: it should demonstrate the 2D-ceiling fix.
**Gate:** at Re = 300 the 3D Cd sits below the 2D over-prediction and
nearer experiment, and a 3D wake instability (mode A) is visible.

### Phase 5 — Web app integration (~1 week)
`Interactive3D` preset (~96 cubed) wired into the Streamlit UI: a 3D
shape selector (sphere, cylinder), the slice and isosurface viewers,
GIF export via a rotating camera. Finalise the backend seam.
**Gate:** a sphere run completes on Cloud in a tolerable time
(target < ~4 min, matching the current 2D Standard wait) without an
out-of-memory error.

### Phase 6 — Tests, docs, validation writeup (ongoing + ~0.5 week)
Full 3D test suite (section 6). A 3D section in VALIDATION.md written
to the same honest standard as the 2D rewrite: scoped band, offline
preset as the headline, interactive preset labelled a convenience.

**Total: roughly 8 to 10 weeks of focused work.** Be honest with
yourself and in interviews: 3D done properly is about two months, not
two weeks. If the original end-of-July target is tight, Phase 4 (the
cylinder) is the natural thing to land first as a milestone and Phase 5
(web integration) can trail it.

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

3D visualisation is a real sub-project. Two viewers, built across
Phases 1 and 2, both reusing existing dependencies.

- **Slice planes (Phase 1).** Extract orthogonal planes (z-mid first)
  and render with the *existing* 2D matplotlib pipeline (vorticity,
  velocity, pressure). Almost no new code, immediately interpretable,
  and it is your debugging window into the solver.
- **Q-criterion isosurface (Phase 2).** Compute the Q-criterion
  (Q = 0.5 * (|Omega|^2 - |S|^2)) on the 3D field, extract the
  isosurface with `skimage.measure.marching_cubes`, render the
  triangle mesh with Plotly `Mesh3d`. This is the "wow" view: hairpin
  vortices behind the sphere, vortex tubes behind the cylinder. Plotly
  3D is interactive inside Streamlit (rotate, zoom).
- **GIF export**: rotate the camera around the isosurface, capture
  frames, assemble with the existing Pillow pipeline.
- Optional later: Plotly `Streamtube` for 3D streamlines.

Everything here uses scikit-image, Plotly and Pillow, all already
dependencies. No new packages.

---

## 8. Risk register

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Cloud out-of-memory | float32, 96 cubed cap, measure resident memory in Phase 0, single-buffer fallback |
| R2 | Interactive 3D too slow on Cloud's 1 vCPU | 64 to 96 cubed cap, GIF not live, be honest it is a coarse toy; GPU backend is the real fix, deferred |
| R3 | q-field memory blowup | analytic sphere q, sparse wall-link list, decided in Phase 0 |
| R4 | Rendering scope creep | slice planes first (cheap), isosurface second, streamlines optional |
| R5 | Validation needs resolution Cloud cannot give | validation is a separate offline preset, results committed; do not fight this |
| R6 | Numba 3D JIT time / register pressure | keep kernels lean, measure compile time, split kernels if needed |
| R7 | Timeline: 3D is ~2 months | phase gates, land Phase 4 (cylinder) as the headline milestone if time is short |
| R8 | "Interactive 3D in the web app" oversells | label the web preset a qualitative convenience, exactly like the 2D Standard preset; never call it validated |

R8 is the one to watch. You spent five review rounds making the 2D
validation honest. Do not let "3D in the browser" quietly become a
claim that the browser 3D is accurate. It will not be. Say so in the UI
and in the docs from day one.

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

## 10. Decisions locked (2026-05-27)

All three open decisions resolved to the recommended option:

1. **v1 geometry scope: sphere plus spanwise-periodic cylinder only.**
   Arbitrary uploaded 3D shapes (3D mesh voxelisation, sparse q-field
   builder) are explicitly deferred to a later version.
2. **Timeline: validation-first, roughly 8 to 10 weeks.** Phase 4 (the
   3D cylinder) is the protected milestone. Phase 5 (web integration)
   may trail it if time is short.
3. **2D K-recalibration: carried as documented 2D backlog.** Not a
   blocker for 3D; 3D gets its own blockage treatment regardless.

The full technical resolution of every Phase 0 research question is in
the companion document `3D_PHASE0_DECISIONS.md`. Phase 0 is decided on
paper; the next action is the first prototype code.
