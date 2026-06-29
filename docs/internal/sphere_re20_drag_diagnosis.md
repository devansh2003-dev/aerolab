# Diagnosis: the 3D sphere Re=20 +47% drag discrepancy

**Date:** 2026-06-11 · **Status:** diagnosis complete, low confidence on the
confinement-vs-force split, one experiment outstanding to resolve it.
**Method:** 9-agent adversarial diagnostic workflow (4 independent diagnostic
angles → synthesis → 3 adversarial skeptics → costed recommendation;
~364k tokens, run `wf_c264991f-910`). This file is the durable capture so the
analysis survives context compaction.

> **Read this before touching the sphere validation or building a sphere
> Cd-vs-Re chart.** The headline finding corrects a conclusion currently
> stated as settled in `src/forces_3d.py` (docstring lines ~14-25) and
> `VALIDATION.md` §8.3.x.

---

## 1. The puzzle

The 3D D3Q19 TRT sphere drag validates well at Re=100 (+6.4% vs reference) but
is far off at Re=20 (+47%), using the **same** solver, grid family, and force
method. Last turn (the Turblyze comparison) we flagged the Re=20 point as the
open accuracy bug gating a sphere Cd-vs-Re chart. This is the diagnosis.

### The data (all committed under `data/validation_3d_sphere_*.json`)

| Case | grid | D | B (diam) | force | nu | tau | Cd_raw | err vs CGW |
|---|---|---|---|---|---|---|---|---|
| Re=100 MYSL D40 | 320³-ish | 40 | 25% | Ladd | 0.016 | 0.548 | 1.528 | +40.2% |
| **Re=100 MYSL D40** | 320×160×160 | 40 | 25% | **MYSL** | 0.016 | 0.548 | **1.160** | **+6.4%** ✓ |
| Re=100 D20 | 160×80×80 | 20 | 25% | Ladd | 0.008 | — | 1.645 | +50.9% |
| Re=100 D20 hi-block | 96×48×48 | 20 | 41.7% | Ladd | 0.008 | — | 1.572 | +44.3% |
| Re=20 D20 Stokes | 160×80×80 | 20 | 25% | Ladd | 0.04 | 0.62 | 4.265 | +56.2% |
| Re=20 MYSL D40 | 320×160×160 | 40 | 25% | Ladd | 0.08 | 0.74 | 4.268 | +56.4% |
| **Re=20 MYSL D40** | 320×160×160 | 40 | 25% | **MYSL** | 0.08 | 0.74 | **4.022** | **+47.4%** ✗ |

The two error levers that explained Re=100 barely move Re=20:
- **Grid refinement** (D20→D40): Re=100 helps ~7pp; Re=20 helps **~0pp**
  (4.265→4.268).
- **MYSL q-aware force** (Ladd→MYSL): Re=100 closes **33.8pp**; Re=20 closes
  only **9.0pp** (56.4→47.4).

So a third, Re-dependent mechanism that is large at Re=20 and small at Re=100
dominates the Re=20 error. The whole diagnosis is about identifying it.

---

## 2. Verdict (honest, low confidence on the split)

**The Re=20 +47% is NOT one clean cause and NOT a discrete bug.** Two real,
roughly co-equal contributors, and the data on hand cannot cleanly say which is
larger:

1. **Finite-Re wall confinement** (sphere is fat relative to the duct,
   a/R_duct = 0.25). Stokes-limit wall amplification at this ratio is genuinely
   large (~1.98 on-axis cylindrical tube; ~1.82 square duct — independently
   re-derived two ways). Inertia screens it down at finite Re. Estimated share:
   **~10–30 pp of the +47%** (wide band). This term **also accounts for
   essentially all of the Re=100 +6.4%** (implied K≈1.064 sits inside a
   defensible small-confinement estimate).
2. **Low-Re limitation of the simplified momentum-exchange force.** The MYSL
   q-correction lever is the TRT antisymmetric post-collision asymmetry, set by
   `s_minus` (= 0.78 at Re=20 vs 0.23 at Re=100), so it is intrinsically ~3–4×
   weaker at low Re. Combined with the viscous-dominated split at Re=20 (~70%
   viscous vs ~50/50 at Re=100), the simplified force leaves more drag
   mis-estimated. Estimated share: **~10–25 pp**. Effectively **~0 at Re=100**
   (MYSL already drove it there). **NOT a bug** — verified-correct code.

**Confidently ruled out** (each ~0 pp; high confidence):
- **Reference value** — CGW independently reconstructed at 2.715 (Re=20) /
  1.087 (Re=100), matching the repo refs within 0.5%. Every alternative
  correlation (Schiller-Naumann 2.61, Morrison 2.56) is *lower*, so a different
  reference makes the error *larger*. Reference is correct, if anything generous.
- **Temporal under-convergence** — Re=20 Cd is invariant (+0.07%) between
  D20/2500-steps and D40/5000-steps at the same 5 advective times. By the
  viscous-time yardstick the *better-validating* Re=100 run is *less* converged
  — the opposite of an under-convergence signature.
- **Grid / boundary-layer resolution** — the Re=20 momentum BL is ~9 cells at
  D40 vs only ~4 cells for the validated Re=100 point. The "broken" point is
  *better* resolved; 8× refinement moved its Cd ~0%.
- **Compressibility / Mach** — Ma≈0.08, Ma²≈0.6%, identical at both Re (same
  u_in). Cannot create a 47%-vs-6.4% split.

### Why "low confidence"

The confinement-vs-force split hinges on the **finite-Re attenuation of the
confinement factor**, which no agent computed from a rigorous finite-Re
sphere-in-duct solution — all used Oseen-screening scaling arguments. Under the
adversarial pass, two of three skeptics argued the screening logic, applied
consistently, could push K(Re=20) as low as ~1.1 (confinement small, force-bias
dominant), while the repo-defender skeptic could not break the confinement
channel at all. The honest position: **both terms are real; their ratio is
genuinely undetermined without the experiment in §4.**

---

## 3. The correction to the repo's documented conclusion

`src/forces_3d.py` (docstring ~lines 14–25) and `VALIDATION.md` §8.3.x state, as
settled, that **"blockage does not dominate; the simplified Ladd force on a
coarse grid does."** That conclusion is **legitimate at Re=100 but was
over-generalized to Re=20, where it does not hold:**

1. **It was only ever measured at Re=100.** No blockage sweep was ever run at
   Re=20. At Re=100 confinement really is small (~6%), near the sweep's <5%
   sensitivity floor — so even where it was measured the null result is weak.
2. **The Re=100 sweep is confounded.** `scripts/validate_3d_sphere_cd_lowblock.py`
   (lines ~54–56) changes the streamwise length `Nx` 96→160 (4.6D→8D downstream)
   *at the same time* it drops blockage 41.7%→25%. So the observed "+4.6% Cd
   went UP when blockage went down" mixes lateral confinement against streamwise
   wake-truncation relief and **cannot cleanly isolate confinement** even at
   Re=100.
3. **Confinement at Re=20 is ~5–8× stronger than at Re=100** (Stokes reservoir
   ~1.98 vs the screened ~1.06), so the Re=100 null cannot transfer.
4. **The repo's own §8.3.4 already concedes "+3–5% from blockage" at Re=100** —
   consistent with the implied K(100)≈1.064 here. So the Re=100 "+6.4%
   validated" point is itself *mostly residual confinement*, not a clean
   confinement-free anchor.
5. **`VALIDATION.md` §8.3.5 attributes the Re=20 residual to a high-tau (0.74)
   kernel effect** — but that hypothesis was never given a quantitative
   magnitude and was never tested against the (untested) confinement channel,
   which sits at roughly the right size.

Net: the repo named a real contributor (force method) but reached for an
untested tau-story for the Re=20 residual while the confinement channel — large
at low Re, never measured at Re=20 — was sitting right there.

---

## 4. The one experiment that resolves it (Option A)

A **Re=20 lateral-blockage sweep** is the single clean test that separates
confinement from force-bias. It was never run. Recipe (cheap, clean,
falsifiable):

- Two runs, **identical except lateral domain width**, both at **D=20**
  (the diagnosis established Re=20 Cd is grid-independent D20↔D40, so D=20 is
  valid and ~4× cheaper than D=40):
  - **Run 1 (B=25%, control):** `Nx=160, Ny=Nz=80, R=10`, MYSL force.
    (Re-measures the existing Stokes point with the MYSL force for a clean match.)
  - **Run 2 (B=12.5%, treatment):** `Nx=160, Ny=Nz=160, R=10`, MYSL force —
    **only the lateral width doubles**; streamwise layout, sphere resolution,
    nu, u_in, steps all held fixed. This avoids the `Nx` confound that
    contaminated the Re=100 sweep.
- Clone `scripts/validate_3d_sphere_cd_stokes_regime_mysl_d40.py`; set the grid
  per above; keep `u_in=0.04`, `nu=0.04` (Re = 0.04·20/0.04 = 20), `n_steps`
  scaled to ≥5 D/U.
- **Cost:** ~1h CPU per run on a 4-core machine (4.1M cells), ~2h total. No GPU.

**Falsifiable prediction:**
- If **Cd drops sharply** toward CGW (≈2.7) when blockage halves →
  **confinement-dominated**. Relabel Re=20 accordingly; you may show an
  *independently-predicted* (not back-solved) confinement-corrected companion
  point.
- If **Cd barely moves** → **force-method/low-Re limitation dominates**. Leave
  Re=20 uncorrected and label it a low-Re force-method limitation.

Either way you learn the truth *before* deciding how to label the point — the
opposite of a tautology.

---

## 5. What NOT to do

**Do not apply a back-solved blockage "correction"** `K_wall = 4.022/2.728 =
1.474` to make Re=20 hit CGW. This is the exact tautology trap that got the
cylinder square-Cd / Allen-Vincenti K correction demoted (fitted at the wrong
blockage, so the "corrected" number just reproduced the target by construction —
see the validation-scoping lesson). Three independent things condemn it:
1. K=1.474 is **above any defensible independent confinement estimate** (~1.25–1.34 max).
2. Applied **consistently it breaks the good point** — over-predicts Re=100 to
   Cd≈1.28 (+11%).
3. Two of three adversarial reviewers showed confinement **may not even be the
   largest** Re=20 term, so a confinement-only correction could be attributing
   the error to the wrong physics.

Also: do **not** present Re=20 as "validated"; do **not** chase the untested
high-tau kernel story as if established.

---

## 6. Recommendation for the sphere Cd-vs-Re chart (the Turblyze analog)

You **can** build it honestly now, but it is a thin two-point story, not the
dense multi-decade curve Turblyze showed. Two presentation tiers (exactly like
the cylinder chart's Re=500 tail):

- **Anchor at Re=100** (+6.4%) — the one genuinely validated 3D sphere point —
  drawn on the CGW 1978 reference curve.
- **Plot Re=20 as a visually distinct marker** (open/dashed, different colour)
  annotated *"reported, not validated — confinement + low-Re force bias."* Never
  silently corrected or hidden.

**Strategic fork for the owner** (this is the decision to make):
- **(i) Build the thin 2-point chart now** — honest, instructive for a SEE-the-air
  audience, zero new compute, but visually sparse next to Turblyze's curve.
- **(ii) Run Option A first (~2h), then decide** — resolves the Re=20 label and,
  if confinement-dominated, lets you add a defensible corrected companion point.
- **(iii) Invest in a real sphere curve** — several more Re bakes (~2h each) to
  get 4–5 points across decades. Closest to matching Turblyze, but ~10h CPU and
  each new low-Re point inherits the same confinement caveat.

Recommended: **(ii)** — one cheap experiment buys the truth, then build the
chart with a correctly-labelled Re=20. Until then, the honest claim is
**"3D sphere drag validated at Re=100 (+6.4%); characterised but not validated
at Re=20."** A smaller honest claim beats a flattering one that paints every
point green.

---

## 7. Side finding: reference-value consistency (minor, worth a cleanup later)

Three slightly different "CGW Re=20" values float around, all within ~1%:
- `scripts/validate_3d_sphere_cd_stokes_regime_mysl_d40.py` **hardcodes**
  `CGW_CD_REF = 2.728`, but its **docstring formula** (Clift-Gauvin single
  expression, line 17) actually evaluates to **~2.70** at Re=20.
- The diagnostic reference-audit reconstructed the CGW piecewise low-Re branch
  as **2.715**.
- At Re=100 the analogous spread is 1.09 (hardcoded) / 1.095 (Clift-Gauvin
  formula) / 1.087 (piecewise) — again <1%.

The +47% error is robust to all of these (every correlation lands 2.56–2.73, so
the error stays +47% to +57%). **Recommendation:** when a sphere drag
correlation is eventually added to `src/references.py`, implement the single
documented formula and let the hardcoded constants derive from it, so there is
one canonical curve instead of three near-duplicates. Deferred — needs the
owner's call on which variant is canonical (no rush; sub-1% and doesn't move any
conclusion).

---

## 8. Provenance

- Diagnostic workflow run `wf_c264991f-910`, 9 agents, 4 phases, ~364k tokens,
  full structured output archived in the session task output for run
  `w19yz8vhf`.
- Refutation tally on the strong "confinement is THE single largest Re=20 term"
  claim: **1 holds / 2 refuted** → confidence downgraded to *low*, verdict
  reframed as "two co-equal causes, split undetermined."
- Code verified during the audit: `src/forces_3d.py`
  (`momentum_exchange_force_3d_mysl`, `_trt_post_collide_at`) — no bug found;
  `s_plus`/`s_minus` stable and valid at both nu; the two `s_minus` derivations
  algebraically identical; q<0.5 fallback mirrors the BB kernel.
