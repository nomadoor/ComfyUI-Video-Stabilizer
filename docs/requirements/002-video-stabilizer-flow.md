# ComfyUI Video Stabilizer — Requirements (Flow Variant / Dense Optical Flow)

Last updated: 2025‑11‑03 (UTC+9)

> This file lists **delta requirements** relative to the Classic (feature‑tracking) spec. UI and I/O remain **identical**; only internals differ unless noted.

---

## A. Purpose & Scope — Delta

* Use **dense optical flow (CPU)** as the primary signal for stabilization instead of sparse feature tracks
* Default backend: **OpenCV DIS (Dense Inverse Search)** for speed on CPU; optional fallback: **TV‑L1** when `cv2.optflow` is available

---

## B. User‑Facing Controls — No Change

* Reuse the exact same controls and behavior from the Classic file: `framing_mode`, `transform_mode`, `camera_lock`, `strength`, `smooth`, `keep_fov`（Cropモード専用）, `padding_color`（`crop_and_pad` / `expand` の境界色）
* Flow vs. Classic is surfaced as **different nodes** or an internal backend toggle — the UI schema remains the same

---

## C. I/O — No Change

* Always emit `padding_mask` (all‑zeros for `crop`)
* In `meta`, add fields such as `flow_backend: "DIS"|"TVL1"`, `flow_pyramid_levels`, and summary confidences if useful
* Also emit the shared `meta.stabilization_warp` contract from the Classic spec, using the exact matrices passed to `cv2.warpPerspective`

---

## D. Internal Design — Delta

1. **Estimation**

   * Compute dense optical flow `F(x,y)` for each adjacent pair (default: DIS with image pyramid)
   * Fit the requested **`transform_mode`** (Translation/Similarity/Perspective) **from the flow** using RANSAC/IRLS to down‑weight outliers (moving objects)
   * Optional: provide a **coarse mesh warp** path and smooth only its low‑frequency control lattice to reduce local content pull
2. **Camera Lock**

   * Bias the global component of flow toward **zero velocity**; apply temporal penalties on ∆ and ∆² of the global parameters
   * Detect strong global pans; locally reduce lock bias to avoid suppressing intended motion
3. **Normal Stabilization**

   * `strength`: removal gain applied to the fitted global motion (after flow→model fitting)
   * `smooth`: temporal smoothing **on the parameter sequence** (θ, log s, tx, ty), identical to Classic formulas
4. **Framing** (same logic as Classic)

   * `crop`: zoom according to `keep_fov` to hide edges
   * `crop_and_pad`: zoom within the `keep_fov` limit, pad the remainder; mark pad=1 and **exclude pad from flow computation** in subsequent steps
   * `expand`: zero zoom; enlarge canvas to minimally enclose the stabilized path
5. **Numerical Robustness**

   * Use pyramid levels (e.g., 3–5). If the **lowest‑resolution level** still yields poor inliers/outlier ratio, **auto‑degrade `transform_mode`**
   * Apply Huber/L1 robust loss during model fitting
6. **Borders & Padding**

   * Prefer `reflect`/`edge‑blur` padding for the warping stage; keep a **safe border mask** so the flow solver does not sample synthetic pads

---

## E. Defaults — Delta

* `flow_backend = DIS`
* `flow_pyramid_levels = 3–5` (auto‑select by resolution)
* `confidence_threshold` for outlier suppression: internal default (e.g., 0.2)

---

## F. Failure & Fallback — Delta

* When low‑texture or subject‑dominant frames reduce flow reliability, **degrade `transform_mode`** exactly like Classic
* If DIS backend creation fails, try **TV‑L1** only when `cv2.optflow` is available; otherwise fall back to optical-flow-independent translation estimation, then identity if translation estimation fails

---

## G. Performance & Non‑Functional — Delta

* CPU target at 1080p30: **≥3–5 fps** with DIS; slower when falling back to TV‑L1
* Memory overhead vs. Classic: **+20–40%** due to flow pyramids

---

## H. Tests — Delta

* Synthetic sequences with known θ/tx/ty/log_s and perspective components to validate flow→model fitting
* Subject‑dominant scenes (e.g., a person covers 80% of the frame) to verify outlier suppression and stability
