# ComfyUI Video Stabilizer — Requirements (Classic / Feature Tracking)

Last updated: 2025‑11‑03 (UTC+9)

---

## 01. Purpose & Scope

* **Goal:** Robust, long‑run video stabilization that remains numerically stable on CPU (no GPU required) while keeping the UX minimal and predictable
* **Scope:** Global‑motion stabilization, output framing control, and padding mask emission
* **Out of scope:** Inpainting / super‑resolution inside this node. Padding areas are **output as masks** to be filled by downstream models (e.g., Wan2.1, VACE)

---

## 02. User‑Facing Controls (Resolve‑style, minimal)

* **framing_mode**: `crop | crop_and_pad | expand`

  * `crop`: Hide edges via auto‑zoom
  * `crop_and_pad`: Zoom within a limit first, then use padding for the remainder (always emit mask)
  * `expand`: No zoom; expand canvas to enclose the full stabilized trajectory (mask is expected)
* **transform_mode**: `translation | similarity | perspective`

  * `translation`: X/Y translation only (most robust)
  * `similarity`: translation + rotation + uniform scale
  * `perspective`: full projective model (most expressive; less robust on some content)
* **camera_lock**: `bool`

  * `true`: “Tripod” look (separate, lock‑specific solver)
  * `false`: Normal stabilization (strength/smooth active)
* **strength**: `float 0..1` (active when `camera_lock=false`)

  * How much estimated camera motion to **remove** (removal gain)
* **smooth**: `float 0..1` (active when `camera_lock=false`)

  * Temporal **trajectory smoothing** strength (higher = fewer jitters but more damping of intentional motion)
* **keep_fov**: `float 0..1` (used when `framing_mode=crop|crop_and_pad`)

  * `1.0 = ズーム0 (入力FOVを完全死守)`, `0.0 = 最大ズーム許容` (internally clamped to safe limits)
  * In `crop_and_pad`: respect `keep_fov` first, then pad anything that still exceeds the frame

> Design note: **Do not fold `camera_lock` into `framing_mode`**. Lock and framing are orthogonal.

---

## 03. I/O

### Inputs

* `frames`: VIDEO / IMAGE_SEQUENCE (raw frames prior to stabilization)
* `framing_mode`: ENUM = `crop | crop_and_pad | expand`
* `transform_mode`: ENUM = `translation | similarity | perspective`
* `camera_lock`: BOOL (enables lock solver; hides/disables strength & smooth)
* `strength`: FLOAT 0..1 (effective only when `camera_lock=false`)
* `smooth`: FLOAT 0..1 (effective only when `camera_lock=false`)
* `keep_fov`: FLOAT 0..1 (treated as `1.0` a.k.a. **no zoom** when `framing_mode=expand`)

### Outputs

* `frames_stabilized`: VIDEO / IMAGE_SEQUENCE
* `padding_mask`: IMAGE_SEQUENCE (**always emitted**; all‑zeros for `crop`, padding=1.0 for `crop_and_pad`/`expand`)
* `meta`: JSON (per‑frame parameters such as θ, tx, ty, log_s, confidences, applied vs. estimated transforms)

---

## 04. Internal Design Guidance (implementation‑free contract)

1. **Separate estimation and application**

   * Track/estimate **on the original frames only**
   * Apply warps/framing afterwards in a single pass
2. **Camera‑lock (distinct solver)**

   * Objective emphasizes **zero velocity**, **low acceleration**, and **identity pull‑back**
   * Long runs: enable start/end (and/or periodic) anchors internally
   * Optionally detect strong global pans and relax lock locally to avoid over‑constraining intent
3. **Normal stabilization**

   * `strength` = removal gain; `smooth` = temporal filtering (penalize Δ and Δ²)
   * Allow internal auto‑fallback of `transform_mode` (perspective→similarity→translation) when unstable
4. **Framing**

   * `crop`: hide edges via zoom governed by `keep_fov`
   * `crop_and_pad`: zoom within `keep_fov`, then pad the remainder (mark pad=1 in mask; exclude pad from tracking)
   * `expand`: zero zoom; enlarge canvas to the minimal enclosing rectangle (or keep input size + extend outward—implementation choice)
5. **Numerical robustness**

   * Parameterize similarity as `[θ, log s, tx, ty]` to avoid compounding matrix drift
   * Use Huber/L1 robust fitting; use second‑difference regularization and a drift term
6. **Borders & padding**

   * Pad regions are always `mask=1` and **excluded** from feature tracking / flow estimation

---

## 05. Defaults

* `framing_mode = crop_and_pad`
* `transform_mode = similarity`
* `camera_lock = false`
* `strength = 0.7`
* `smooth = 0.5`
* `keep_fov = 0.6`

---

## 06. Failure & Fallback Policy

* Low confidence: degrade `transform_mode` (perspective→similarity→translation)
* Monotonic growth of missing area: in `crop` clip zoom to max; in `crop_and_pad` push overflow to padding; in `expand` update bounding canvas
* Long‑run divergence: auto‑increase lock/anchors internally or flag via `meta`

---

## 07. Tests & Metrics

* Synthetic tests with known θ/tx/ty/log_s; report reconstruction errors
* Long‑run (≥500f): track zoom factor & missing‑area curves; fail on monotonic divergence
* Natural videos: pan/tilt/zoom/handheld/low‑texture scenarios; visual QA

---

## 08. Non‑Functional Requirements

* **Performance:** CPU‑only, target ≥5 fps at 1080p30 (baseline); algorithm scales predictably
* **Dependencies:** OpenCV (GFTT/LK, RANSAC, etc.); avoid GPU‑only dependencies
* **Maintainability:** clear separation of estimation/apply, test suite, CI long‑run checks
