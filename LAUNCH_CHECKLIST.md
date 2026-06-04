# AeroLab — Pre-Launch Test Checklist

Comprehensive verification before flipping the public share switch for v1.7.4.
Tick top-to-bottom; sections are ordered from most-critical (blocking) to
nice-to-have (polish).

> **Setup:** Ctrl-C any stale server, then `streamlit run app.py` from the
> `.venv311` activated PowerShell. Use Chrome incognito for fresh-session
> testing. DevTools → Console open for the entire run.

---

## §0 — Pre-flight gates (in the repo, before launching the server)

### Code quality
- [ ] `python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read())"` → `app.py: syntax OK`
- [ ] `pytest -q` → **393 passed, 17 deselected** (or current target). No failures, no errors.
- [ ] `git status -s` → clean working tree (no uncommitted edits)
- [ ] `git log origin/main..HEAD` → empty (everything pushed) OR explicitly ready to push
- [ ] Grep for leftover debug: `grep -rn "print(\|breakpoint(\|pdb\|TODO\|FIXME\|XXX\|HACK" src/ app.py` → only documented hits

### Dependencies + assets
- [ ] `requirements.txt` versions match the local `.venv311` (no pin drift):
  - `streamlit==1.57.0`, `plotly==6.7.0`, `numpy==2.4.4`, `numba==0.65.1`, etc.
- [ ] All baked `.npz` files present under `data/baked/` for every preset the gallery references (no `FileNotFoundError` on first card click)
- [ ] No accidental large binary check-ins (`git ls-files | xargs -I{} ls -la {} | sort -k5 -n -r | head -5`)
- [ ] `.gitignore` actually excludes `wing_rebake_logs/`, `a3_bake_logs/`, `.venv*/`, `__pycache__/`

### Secret + privacy scan
- [ ] No API keys / tokens in tracked files: `grep -rni "api[_-]?key\|secret\|token\|password\|bearer" --include="*.py" --include="*.md" .` → only false-positives (docstrings, variable names)
- [ ] `.env` and any `.streamlit/secrets.toml` are git-ignored
- [ ] No personally identifiable data in logs or sample files

---

## §1 — 2D playground smoke test

Start at: sidebar **Mode = CFD (LBM solver)**, default state.

### 1.1 Mode toggle + landing
- [ ] Mode radio defaults to **CFD (LBM solver)** (index=1, not Fast)
- [ ] Switching to **Fast (ML surrogate)** updates the main area without solver run
- [ ] Switching to **Validation (benchmarks)** shows tables / charts, no solver
- [ ] Switching back to **CFD (LBM solver)** preserves last shape / Re / AoA

### 1.2 Built-in shapes (CFD mode)
- [ ] **NACA 0012** at AoA=0°, Re=400 → run completes, GIF visible, polar plot renders
- [ ] **NACA 4412** at AoA=5°, Re=200 → run completes, polar shows positive Cl
- [ ] **Cylinder** at Re=100 → vortex shedding visible in GIF (after enough timesteps)
- [ ] **Cube / square** at Re=200 → bluff-body wake visible
- [ ] **Ellipse** (if shipped) → renders cleanly

### 1.3 Custom shape — upload PNG/JPG
- [ ] Upload a simple silhouette (e.g. a black square on white background)
- [ ] Preview shows the extracted polygon overlaid on the LBM grid
- [ ] Run completes without exception; flow visualizes around the uploaded shape
- [ ] **Bad upload** (e.g. all-white image, no contrast) → graceful error message, no traceback

### 1.4 Custom shape — draw polygon
- [ ] Draw a closed polygon on the canvas → preview shows it correctly
- [ ] Run completes without exception
- [ ] **Empty / single-point polygon** → graceful error, no traceback

### 1.5 Sliders + edge cases
- [ ] AoA at extremes (-90° / +90°) → renders or refuses gracefully
- [ ] Re at extremes (Re=10 creeping, Re=800 unphysical-banner) → run completes; **unphysical Re banner** appears above slider when Re > 200
- [ ] Resolution toggle (**Standard** / **Detailed**) → both complete; Detailed clearly higher fidelity but takes ~3× longer
- [ ] **Cancel** button (mid-solve) → aborts cleanly, no zombie process

### 1.6 Snapshot / pin / compare
- [ ] After a run, click **Pin as snapshot** → confirmation toast appears
- [ ] Change shape or Re → side-by-side view shows pinned snapshot on left, new run on right
- [ ] **Clear snapshot** button restores single-run view
- [ ] Snapshot survives viz-mode change (Vorticity → Velocity → Pressure)

### 1.7 Share link
- [ ] Generate share link → copy URL
- [ ] Paste URL in fresh incognito tab → app loads with the same shape / Re / AoA / viz
- [ ] Share link includes custom polygon if one was drawn/uploaded

### 1.8 Export
- [ ] **GIF download** → saves file, plays in image viewer
- [ ] **Polar CSV export** (Fast mode + NeuralFoil) → CSV opens in Excel, headers clear
- [ ] Filename includes shape + Re (so multiple downloads don't collide)

### 1.9 Fast (ML surrogate) mode
- [ ] NACA 0012 → polar shows in **<1 second** (NeuralFoil is fast)
- [ ] AoA sweep produces smooth Cl/Cd curves
- [ ] Switching from Fast → CFD preserves shape, doesn't lose state

### 1.10 Validation (benchmarks) mode
- [ ] All tables render without solver runs
- [ ] Bar charts show AeroLab vs literature (Williamson, Okajima, OpenFOAM)
- [ ] Citation list at bottom is complete + clickable

---

## §2 — 3D gallery smoke test

Switch to **3D gallery (preview)** in sidebar. Wait for first render.

### 2.1 First-load state
- [ ] Sidebar shape defaults to **Sphere (round)** (first in list)
- [ ] Flow speed defaults to **0.30 m/s**, Re display reads **Re ≈ 100**
- [ ] Color radio defaults to **Velocity**
- [ ] Main title reads `AeroLab 3D · SPHERE_RE100`
- [ ] Sphere is centered-left with **wake visible behind it** (camera pulled back per v1.7.4 framing)

### 2.2 All shape options (verify each loads)
For each shape: dropdown change → no exception → title updates → preset preview thumbnail shows.
- [ ] **Sphere (round)** → SPHERE_RE100 default
- [ ] **Cylinder (round pipe)** → CYLINDER_RE100, wake visible
- [ ] **Cube (block)** → CUBE_RE100, sharp-edge separation
- [ ] **NACA 0012 wing (symmetric)** → NACA0012_RE100, AoA slider appears
- [ ] **NACA 4412 wing (cambered)** → NACA4412_RE100, AoA slider appears
- [ ] **Upload your own** → placeholder info card (deferred to v1.8.0)

### 2.3 Re bands per shape (drag flow speed through all values)
- [ ] **Sphere**: bands {20, 40, 100} — drag through 0.06 / 0.12 / 0.30 / 4.50 m/s
- [ ] **Cylinder**: bands {20, 40, 100}
- [ ] **Cube**: bands {20, 40, 100}
- [ ] **NACA 0012**: bands {20, 40, 100, 200}
- [ ] **NACA 4412**: bands {20, 40, 100, 200}
- [ ] **Snap caption** updates correctly for off-band values (e.g. `Re ≈ 17 snapped to baked Re = 20 (20, 40, 100 available).`)
- [ ] **On-band caption** reads `Showing pre-baked Re = 40.` exactly

### 2.4 AoA slider (wings + cube only)
- [ ] AoA slider hidden for **Sphere** and **Cylinder** (rotationally symmetric note shown instead)
- [ ] Slider visible for **Cube**, **NACA 0012**, **NACA 4412**
- [ ] Drag to +15° → caption: `+15° snapped to baked AoA = +30°.`
- [ ] Drag to +30° → caption: `Showing pre-baked AoA = +30°.`
- [ ] Drag to ±45° → wing renders without clipping the top wall (v1.7.3 fix)
- [ ] Drag to -45° → mirror-symmetric of +45°

### 2.5 Color (viz mode) — Velocity / Vorticity / Pressure
For each shape, toggle through all three:
- [ ] **Velocity** → Plasma colormap, streamlines purple→yellow
- [ ] **Vorticity** → Viridis colormap, bright where |curl| high
- [ ] **Pressure** → RdBu colormap, blue=suction / red=stagnation, white=freestream
- [ ] Colorbar title updates per mode (`speed` / `|ω|` / `p`)

### 2.6 Overlays
- [ ] **Body** toggle off → sphere/wing surface disappears, streamlines still render
- [ ] **Wind-tunnel chamber outline** toggle off → wireframe box disappears
- [ ] **Q-criterion vortex shell** toggle on → translucent cyan shell appears where vortices live
- [ ] Q threshold slider → moving toward 1% expands shell, toward 50% contracts it

### 2.7 Streamline controls
- [ ] **Density** slider 12 → 96 → seed count visibly changes
- [ ] **Thickness** slider 1 → 8 → line width visibly changes
- [ ] **Animate flow** toggle off → static snapshot; toggle on → ▶/⏸ buttons appear bottom-left of chart

### 2.8 Animation play/pause
- [ ] Click **▶ Animate** → streamline streaks travel through the field
- [ ] Click **⏸ Pause** → freezes mid-animation
- [ ] Camera orbit DURING animation is preserved (this is the uirevision-works case)
- [ ] Switching shape during playback → stops cleanly, new scene takes over

### 2.9 Curated gallery cards — THE LAUNCH-CRITICAL TEST
For each of the 6 cards under **Try one of these**:
1. Click *Show me*
2. Confirm toast appears: `:material/play_arrow: Loading: <card title>`
3. Sidebar widgets update (shape dropdown, flow speed, color)
4. Main title bar updates (e.g. `AeroLab 3D · NACA4412_RE200`)
5. Scene re-renders to the advertised state

- [ ] **How a wing lifts** → NACA 4412, 4.50 m/s, Pressure, `NACA4412_RE200`
- [ ] **Wing at zero AoA** → NACA 0012, 4.50 m/s, Velocity, `NACA0012_RE200`
- [ ] **Where the air spins** → Cylinder, 4.50 m/s, Vorticity, `CYLINDER_RE100`
- [ ] **Bluff cube** → Cube, 4.50 m/s, Velocity, `CUBE_RE100`
- [ ] **Almost stopped (creep)** → Sphere, **0.12 m/s**, Velocity, `SPHERE_RE40` (v1.7.4 follow-up fix)
- [ ] **Sphere wake** → Sphere, 4.50 m/s, Velocity, `SPHERE_RE100`

### 2.10 Camera behavior (known-limitation aware)
- [ ] Orbit sphere camera → **toggle Velocity → Vorticity** → camera **resets** to default framing (known limitation, lands on GOOD view)
- [ ] Orbit camera → **drag flow speed** → camera resets to default (known limitation)
- [ ] Orbit camera → **drag AoA** → camera resets to default (known limitation)
- [ ] Camera reset always lands on a **wake-visible** framing (sphere/cylinder pulled-back; wings close)
- [ ] Camera survives the **animation loop** (▶ / frame cycle) — preserved within Plotly's Animate frames

### 2.11 Loading indicators
- [ ] Fresh shape switch → spinner icon appears next to *"Tracing streamlines through the flow..."*
- [ ] Trace progress bar ticks through ~10% → 25% → 40% → 50-90% → 88% → 92% → 100%
- [ ] No blank gap between progress bar at 100% and chart appearing (v1.7.4 fix)
- [ ] Cached-field load (same shape, AoA change) is near-instant

### 2.12 Edge cases
- [ ] Switch to a shape, then immediately switch to a different shape mid-trace → second click cancels first cleanly, no double-render
- [ ] Refresh browser mid-render → app reloads to a clean state
- [ ] Network throttle to "Slow 3G" in DevTools → chart still loads (maybe slow), no broken state

---

## §3 — Cross-mode + cross-feature regression

- [ ] **2D → 3D → 2D**: shape preference preserved within each mode (don't bleed across)
- [ ] **CFD → Fast → CFD**: sidebar shape stays where you left it
- [ ] **Mode switch mid-solve** in 2D: cancels cleanly, no zombie process
- [ ] **Browser refresh during 3D scene swap**: reloads to default state, no half-rendered chart
- [ ] **Two tabs same app**: each holds its own session_state; no cross-tab leakage
- [ ] **Browser back / forward** doesn't crash the app

---

## §4 — Visual + polish

### 4.1 Browser chrome
- [ ] Tab title: **"AeroLab — Browser-Based CFD"**
- [ ] Tab favicon: 🌀 cyclone glyph (not Streamlit's red dot)
- [ ] Streamlit's "Manage app" / "Hosted with Streamlit" footer chrome is hidden via CSS
- [ ] Sidebar collapse chevron present and works

### 4.2 Layout at common viewport widths
- [ ] **1920×1080** (desktop): no horizontal scroll, no overflow
- [ ] **1366×768** (laptop): sidebar + main both fit, gallery cards reflow to 3+3 grid
- [ ] **1024×768** (tablet landscape): sidebar still usable, gallery cards single-column or 2+4
- [ ] **414×896** (mobile portrait): sidebar collapses by default; the warning is acceptable for v1.7.4 ("desktop recommended")
- [ ] Console: zero red errors (yellow warnings from Streamlit framework are OK)

### 4.3 Typography + readability
- [ ] All sidebar labels readable (color contrast not gray-on-gray)
- [ ] Material icons render (no broken square placeholders)
- [ ] Code blocks / monospace look intentional
- [ ] Long help-tooltip text wraps cleanly, doesn't overflow

### 4.4 Color / theme
- [ ] Dark background (`#0a0a0a`) is consistent across 2D + 3D + Validation
- [ ] Plotly colormaps (Plasma, Viridis, RdBu_r) render correctly
- [ ] Streamlit primary color theme matches the AeroLab brand if set

---

## §5 — Performance + memory (local)

### 5.1 Cold-start timings (fresh process)
- [ ] First `streamlit run` to first paint: **<10 s** locally
- [ ] First 2D LBM solve (NACA 0012, Re=400, Standard): **<35 s** locally
  (Cloud is 3× slower; this is the local baseline)
- [ ] First 3D gallery card click (cold trace): **<5 s** locally
- [ ] Repeat clicks on cached field: **<2 s**

### 5.2 Memory budget
- [ ] Open Resource Manager / Task Manager → Python process memory during a 2D Detailed run: **<800 MB** locally
- [ ] 3D scene swap doesn't leak memory (run 6 cards in sequence; memory should plateau, not grow)

### 5.3 Cache behavior
- [ ] `@st.cache_data(persist="disk")` survives a `streamlit` server restart
  (check `.streamlit/cache/` exists and has content)
- [ ] LRU eviction visible after >4 distinct 2D solve configs (oldest evicted)

---

## §6 — Streamlit Cloud deploy verification

**After pushing `aa808bc + 6405bdd` to `main`:**

- [ ] Streamlit Cloud auto-deploys within ~2 min (watch the **Manage app → Deploy** badge)
- [ ] App URL loads in incognito; no auth gate
- [ ] First cold-start: **<60 s** (first visitor pays JIT-compile tax on numba)
- [ ] Memory cap: peak stays **<500 MB** during a typical session (Free tier cap)
- [ ] Logs (Manage app → Logs) show no ImportError, no missing-file errors
- [ ] At least one full 2D LBM run completes on Cloud without OOM
- [ ] At least one 3D gallery card click renders without crash
- [ ] Two concurrent sessions (two browsers) don't crash each other
- [ ] Cache persists across reloads within the same container lifetime

---

## §7 — Documentation accuracy

- [ ] `README.md` references the correct Cloud URL (if mentioned)
- [ ] `README.md` install / run instructions actually work in a fresh clone
- [ ] `VALIDATION.md` headline numbers match what the app shows
  - §3 2D Resolved 5–10% gaps
  - §8.3.4 D=40 MYSL Cd=1.160 (+6.44%)
- [ ] `CHANGELOG.md` v1.7.4 entry is accurate (no claims that don't hold)
- [ ] `RELEASE_NOTES_v1.7.4.md` matches what users will actually experience
- [ ] In-app help tooltips don't reference removed features
- [ ] `LICENSE` file present (MIT or whichever you intended)

---

## §8 — Browser compatibility

Quick sanity on the major browsers (incognito each time):
- [ ] **Chrome** (latest) — primary target, full UX works
- [ ] **Firefox** (latest) — full UX works, WebGL renders 3D scene
- [ ] **Edge** (latest) — full UX works
- [ ] **Safari** (if Mac available) — full UX works; check the Plotly 3D scene specifically (Safari WebGL has quirks)

---

## §9 — External monitoring (UptimeRobot)

Free-tier uptime + downtime alerts so we know if Streamlit Cloud spins the
container down or returns 5xx.

### Setup steps (one-time)
1. Sign up at <https://uptimerobot.com> (free tier: 50 monitors, 5-min interval)
2. Confirm the email address used (alerts go here)
3. **Add New Monitor**:
   - **Monitor Type:** `HTTP(s)`
   - **Friendly Name:** `AeroLab — Streamlit Cloud`
   - **URL:** the deployed app URL (`https://<app-slug>.streamlit.app`)
   - **Monitoring Interval:** 5 minutes
   - **Monitor Timeout:** 30 seconds
     (Streamlit Cloud cold-starts can exceed the default 10 s — bumping
     timeout avoids false-positive alerts on first request after idle)
4. **Alert Contacts to Notify:** the email confirmed in step 2
5. Save. First check fires within 5 min; status flips to green on the
   first 2xx response

### Verification
- [ ] Monitor created and showing **"Up"** (green) in the dashboard
- [ ] **Test alert:** temporarily change URL to `https://<app-slug>.streamlit.app/nonexistent` → wait one check cycle → confirm email alert lands → revert URL
- [ ] (Optional) Public status-page badge — Settings → Public Status Pages → Add → grab embed URL for `README.md`

### Alert hygiene
- A **single** "Down" alert from Streamlit Cloud is usually a cold-start spike past 30 s — confirm the URL works in a browser before treating it as real
- Repeated alerts (>2 in 30 min) → likely a real outage; check the Streamlit Cloud status page and app logs (Manage app → Logs)

---

## §10 — Tag, release, announce

- [ ] `git tag -a v1.7.4 -m "v1.7.4: pre-launch UX fix sprint"`
- [ ] `git push origin main`
- [ ] `git push origin v1.7.4`
- [ ] GitHub Release drafted with content from `RELEASE_NOTES_v1.7.4.md`
- [ ] Share link works in incognito (rules out cookie-cached auth)
- [ ] Social-share preview (paste URL into Twitter/LinkedIn compose box) shows the title + 🌀 favicon correctly

---

## §11 — Post-launch first-24h watch

- [ ] UptimeRobot dashboard: no sustained outages
- [ ] Streamlit Cloud **Manage app → Resources**: memory not climbing toward 500 MB
- [ ] Streamlit Cloud **Manage app → Logs**: scan for unexpected tracebacks
- [ ] Browser-console screenshot from one external visitor (ask a friend) — confirm no errors visible there
- [ ] Stay near keyboard for the first few hours in case a hot-fix is needed

---

## Rollback plan

If something breaks badly post-launch:

1. **Identify the bad commit** — `git log --oneline -10` on `main`
2. **Revert that commit** — `git revert <sha>` → push to `main`
3. **Streamlit Cloud redeploys automatically** within ~2 min
4. **Verify** — load the app in incognito, confirm the broken behaviour is gone
5. **Don't force-push to main** — revert commits leave a clean audit trail

If the rollback target is multiple commits back, a `git revert <oldest>..<newest>` range is safer than amending or resetting.

For a *full* rollback to a known-good earlier release:
```powershell
cd "C:\Users\USER\Desktop\Study & Work\Personal Projects\AeroLab"; git revert --no-commit <bad_first_sha>..HEAD; git commit -m "Rollback to v1.7.3"; git push origin main
```

---

## Known limitations to document (NOT bugs)

These are deliberate v1.7.4 scope decisions; don't surprise yourself by testing them as failures:

- **3D camera orbit resets on viz/AoA/speed change.** The reset lands on the (good) shape-specific default. Proper fix needs `streamlit-plotly-events` or a custom component; deferred.
- **3D upload-your-own-shape is a placeholder.** Cloud free tier can't run a fresh LBM bake within the request budget. Stub UI present, deferred to v1.8.0.
- **Bluff-body Re=200 (sphere, cylinder) diverged** at τ near 0.5 boundary. Not in baked-band list. Cumulant collision or Ny ≥ 128 needed.
- **2D solver above Re=200 marked "exploratory"** — banner warns of 3D-instability regime. This is the right framing, not a bug.

---

## Final go / no-go decision

If §0 + §1 + §2 + §6 all green → **safe to push public link.**
§3–§5 + §7–§8 should be green but a single yellow item in those sections isn't blocking.
§9 ideally green before public launch, but can lag the first few hours.
§11 starts at T+0 and runs continuously.
