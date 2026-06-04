# v1.7.4 — Pre-launch UX fix sprint

Polish pass before flipping the public share switch. No solver or accuracy
changes — every edit is in the UI surface.

## 🎯 Headline

The 3D gallery preset cards no longer crash. **"Try one of these"** cards
now load cleanly with a one-click experience that survives camera orbits,
shape switches, and slider drags.

## 🛠️ Fixes

- **3D preset cards no longer crash.** Clicking *Show me* on any of the six
  curated scenes used to throw `StreamlitAPIException`; now it just loads the
  scene with a confirmation toast.
- **Camera holds during animation playback.** The orbit you set is preserved
  through Plotly's animation frame loop (it used to reset every animation
  cycle). *Cross-rerun camera persistence — i.e. holding your orbit when
  you change viz mode or drag a slider — needs a custom component, deferred
  to a later release.*

## 💅 Polish

- **Sphere and cylinder default camera.** Bluff bodies sit centered in the
  chamber, so the old wing-centric framing hid their wakes. Pulled the
  camera back so the wake is visible from first render.
- **Flow-speed snap caption now shows both values.** Reads
  `Re ≈ 17 snapped to baked Re = 20 (20, 40, 100 available).` instead
  of just announcing the snapped value. Matches the AoA caption style.
- **Loading spinner during streamline trace.** A rotating icon during the
  2–4 s trace step makes the load feel responsive instead of frozen. The
  trace progress now flows continuously into a brief *"Rendering in your
  browser…"* indicator with no blank gap before the chart appears (it used
  to vanish, leaving a blank spot).
- **Browser tab title and favicon.** Tab now reads
  *"AeroLab — Browser-Based CFD"* with a 🌀 favicon, easier to find in a
  crowded browser window.

## 🧹 Internal

- Repo-root research markdown moved to `docs/internal/` (no functional
  change — paths in Python docstrings stay grep-able).
- New `LAUNCH_CHECKLIST.md` with pre-launch verification steps and
  UptimeRobot setup notes for post-launch monitoring.

## What's next

`v1.8.0` reopens the **upload-your-own-shape** work (paused mid-implementation
to ship this fix sprint). STL + PNG/SVG silhouette extrusion paths are
stashed and ready to resume.

---

**Full diff:** `git log v1.7.3..v1.7.4`
**Cloud deploy:** auto-rolls within ~2 min of the tag push.
