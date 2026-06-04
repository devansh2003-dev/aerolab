# AeroLab — Launch Checklist

Pre-launch verification for the public v1.7.4 release. Tick each item before
flipping the share switch.

---

## D-1: Code quality gates

- [ ] `app.py` parses cleanly (`python -c "import ast; ast.parse(open('app.py').read())"`)
- [ ] Full test suite passes: `pytest -q` (target: 393 passed, 17 deselected)
- [ ] No uncommitted edits to `src/` or `app.py` (`git status -s`)
- [ ] `requirements.txt` matches `.venv311` (no drift from local pins)
- [ ] All baked `.npz` files present under `data/baked_fields_3d/`

---

## D-1: UX smoke test (local)

Run `streamlit run app.py` and click through:

### 2D playground
- [ ] Mode toggle defaults to **CFD (LBM solver)**
- [ ] Pick NACA 0012 → slider → run → GIF renders, polar matches reference
- [ ] Switch to **Fast (ML surrogate)** → NeuralFoil polar updates in <1 s
- [ ] **Validation (benchmarks)** tab renders without solver runs

### 3D gallery
- [ ] Browser tab title reads **"AeroLab — Browser-Based CFD"** + cyclone favicon
- [ ] Sphere loads with wake visible in default camera framing
- [ ] Cylinder loads with wake visible in default camera framing
- [ ] Cube + NACA wings load with their existing framing (no regression)
- [ ] Drag AoA slider → caption reads `+15° snapped to baked AoA = +30°`
- [ ] Set flow-speed to 0.05 m/s → caption reads `Re ≈ 17 snapped to baked Re = 20`
- [ ] Switch shapes → spinner appears during streamline trace
- [ ] Trace progress flows into a brief "Rendering in your browser…" indicator with no blank gap before the chart appears (the indicator itself is sub-second — the win is no gap, not a long hold)
- [ ] Orbit camera → drag AoA slider → camera survives the rerun
- [ ] Click each of the **6 gallery cards** → toast appears, sidebar updates,
      title bar updates, scene re-renders to the advertised shape/Re/colour

### Visual polish
- [ ] No orphan tracebacks in browser console (DevTools → Console)
- [ ] Streamlit "Deploy" pill in top-right shows local URL (not the cloud chrome)
- [ ] Sidebar opens by default on first load (no manual click needed)

---

## D-1: Cloud deploy verification

After pushing to `main`:

- [ ] Streamlit Cloud picks up the commit (watch the deploy badge)
- [ ] Cold-start cold takes <60 s (first visitor pays JIT-compile tax)
- [ ] Memory stays under 500 MB during a typical session
  (Streamlit Cloud free tier cap; monitor via Manage app → Resources)
- [ ] At least one full 2D LBM run completes without OOM
- [ ] At least one 3D gallery card click renders without crash

---

## D-0: External monitoring (UptimeRobot)

Free-tier uptime + downtime alerts so we know if Streamlit Cloud spins the
container down or returns 5xx.

### Setup steps (one-time)
1. Sign up at <https://uptimerobot.com> (free tier: 50 monitors, 5-min interval).
2. Confirm the email address used (alerts go here).
3. **Add New Monitor**:
   - **Monitor Type:** `HTTP(s)`
   - **Friendly Name:** `AeroLab — Streamlit Cloud`
   - **URL:** the deployed app URL (`https://<app-slug>.streamlit.app`)
   - **Monitoring Interval:** 5 minutes
   - **Monitor Timeout:** 30 seconds
     (Streamlit Cloud cold-starts can exceed the default 10 s — bumping
     timeout avoids false-positive alerts on first request after idle)
4. **Alert Contacts to Notify:** the email confirmed in step 2.
5. Save. First check fires within 5 min; status flips to green on the
   first 2xx response.

### Verification
- [ ] Monitor created and showing **"Up"** (green) in the dashboard
- [ ] Test alert: temporarily change URL to `https://<app-slug>.streamlit.app/nonexistent`
      → wait one check cycle → confirm email alert lands → revert URL
- [ ] (Optional) Add a public status-page badge — Settings → Public Status
      Pages → Add → grab the embed URL for `README.md`

### What to watch
- A **single** "Down" alert from Streamlit Cloud is usually a cold-start spike
  past 30 s — confirm the URL works in a browser before treating it as real.
- Repeated alerts (>2 in 30 min) → likely a real outage; check the
  Streamlit Cloud status page and app logs (Manage app → Logs).

---

## D-0: Tag + announce

- [ ] `CHANGELOG.md` has a v1.7.4 entry with all the pre-launch fixes
- [ ] `git tag -a v1.7.4 -m "v1.7.4: pre-launch UX fixes"`
- [ ] `git push origin v1.7.4`
- [ ] GitHub release drafted with the release-notes block (TASK 2 output)
- [ ] Share link works in incognito (rules out cookie-cached auth)

---

## Rollback plan

If something breaks badly post-launch:

1. **Identify the bad commit** — `git log --oneline -10` on `main`.
2. **Revert that commit** — `git revert <sha>` → push to `main`.
3. **Streamlit Cloud redeploys automatically** within ~2 min.
4. **Verify** — load the app in incognito, confirm the broken behaviour is gone.
5. **Don't force-push to main** — revert commits leave a clean audit trail.

If the rollback target is multiple commits back, a `git revert <oldest>..<newest>`
range is safer than amending or resetting.
