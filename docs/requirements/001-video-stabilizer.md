##00 ##overview - Purpose
- Provide a lightweight, robust **video stabilization** node for ComfyUI using classic CV only (OpenCV + NumPy; CPU-friendly).
- Key features:
  - Transformation models: `translation / similarity / perspective` (akin to DaVinci Resolve’s idea space).
  - Framing behaviors: `CROP / CROP_and_PAD / NO_CROP_PAD`.
  - **Always emits a missing-area mask** for downstream outpainting (e.g., Wan2.1 VACE).
  - Two intuitive sliders: **smoothness** and **stabilize_zoom**.

##01 ##id - Identification
- ID: `001-video-stabilizer`
- Status: Draft
- Updated: 2025-10-29
- Repository: **ComfyUI-Video-Stabilizer**

##02 ##scope - In/Out of Scope
- In scope: Typical handheld jitter—pan/tilt, rotation, uniform scale, mild perspective.
- Out of scope (v1):
  - Learning-based warping and heavy dense optical flow.
  - Explicit rolling-shutter modeling.
  - IMU/Gyro ingestion.
  - Outpainting itself (node provides mask + padded color only).

##03 ##usecases - Use Cases
- Reduce shake while preserving field-of-view (FOV) as much as possible.
- Replace crop-loss with outpainting (gray padding + mask).
- Run efficiently on CPU with minimal dependencies.

##04 ##params - Input Parameters (minimal, final)
- **method** (default: `similarity`)
  - `translation` — X/Y only
  - `similarity` — translation + rotation + uniform scale
  - `perspective` — homography (projective)
- **smoothness** *0.00–1.00* (default: `0.25`)
  - Higher → smoother trajectory (but borders are more likely to appear).
  - Internally maps to **OneEuro** (`f_min`, `β`).
- **stabilize_zoom** *0.00–1.00* (default: `0.35`)
  - Maximum auto-zoom (i.e., effective crop) used to hide borders.
  - Applies to **CROP** and **CROP_and_PAD**; ignored in **NO_CROP_PAD**.
- **framing** (default: `CROP_and_PAD`)
  - `CROP` — zero borders via fixed common crop.
  - `CROP_and_PAD` — minimal crop; fill residual gaps with pad color; always output mask (FOV-friendly).
  - `NO_CROP_PAD` — no crop; fill all gaps; always output mask.
- **pad_color_rgb** (default: `(128,128,128)`)
  - Fill color for missing areas (gray by default for outpainting workflows).

##05 ##io - I/O Specification
- Input: `frames` (IMAGE list, or VIDEO decoded internally to frames)
- Output:
  - `stabilized_frames` (IMAGE list)
  - `mask` (IMAGE list: white = missing, black = valid)  <!-- renamed per instruction -->

##06 ##behavior - Internal Flow
- Feature detection: GoodFeaturesToTrack (ORB as fallback if needed).
- Tracking: Pyramidal Lucas–Kanade.
- Outlier rejection: RANSAC / LMEDS.
- Transform estimation:
  - `translation / similarity`: `estimateAffinePartial2D`  
    (for `translation`, zero out rotation & scale; apply translation only)
  - `perspective`: `findHomography`
- Trajectory smoothing: **OneEuro** (map `smoothness → f_min, β`).
- Target path interpolation: `T_target = (1–s)*T_raw + s*T_smooth` in parameter space.
- Warping: `warpAffine` / `warpPerspective`.
- Missing-area detection: warp a full-ones mask with identical transform; zeros → missing → white in `mask`.
- Framing:
  - **CROP**: intersection of valid regions across frames → largest inscribed rectangle; apply required zoom clamped by `stabilize_zoom`. If still insufficient, relax smoothing to maintain zero borders.
  - **CROP_and_PAD**: apply minimal zoom ≤ `stabilize_zoom`; pad remaining gaps with `pad_color_rgb`; always output `mask`.
  - **NO_CROP_PAD**: no zoom/crop; pad all gaps; always output `mask`.

##07 ##ux - UI Strings (ComfyUI)
- method: Translation / Similarity / Perspective (what geometric model is stabilized)
- smoothness: Smoothing (0–1) — higher = smoother camera path
- stabilize_zoom: Zoom allowance (0–1) — used in CROP / CROP_and_PAD
- framing: CROP / CROP_and_PAD / NO_CROP_PAD (with concise tooltips)
- pad_color_rgb: Padding Color (default gray; color picker)

##08 ##defaults - Defaults
- method: `similarity`
- smoothness: `0.25`
- stabilize_zoom: `0.35`
- framing: `CROP_and_PAD`
- pad_color_rgb: `(128,128,128)`

##09 ##accept - Acceptance Criteria
- Methods:
  - `translation` reduces jitter with minimal distortion.
  - `similarity` handles rotation/zoom naturally without perspective artifacts.
  - `perspective` tolerates mild parallax without catastrophic skew.
- Framing:
  - `CROP`: zero white in `mask`; fixed FOV across frames.
  - `CROP_and_PAD`: FOV ≥ `CROP`; residual gaps padded; `mask` correct (white where missing).
  - `NO_CROP_PAD`: no crop; `mask` covers all out-of-bounds regions.
- Controls:
  - Increasing **smoothness** increases perceived stability.
  - Increasing **stabilize_zoom** increases achievable stability (at FOV cost).
- Determinism: same input/settings → identical output.

##10 ##tests - Test Plan
- Unit: extremes `smoothness=0/1`, `stabilize_zoom=0/1`, method switching, mask correctness.
- Functional:
  - `CROP`: zero borders; constant FOV; `mask` ~ all black.
  - `CROP_and_PAD`: minimal crop; only gaps white in `mask`.
  - `NO_CROP_PAD`: zero crop; all gaps white in `mask`.
- Golden: short reference clips with frame-hash checks.
- Scenarios: mild handheld jitter / with rotation / mild perspective / fast pan.

##11 ##perf - Performance
- FHD (1920×1080) should be practical on CPU (>10 fps target; environment-dependent).
- Stream processing to limit peak memory.

##12 ##reliability - Reliability & Exceptions
- Feature scarcity / estimation failure: gracefully downgrade `perspective → similarity → translation`, or soften smoothing.
- Mask generation always uses the exact same warp, guaranteeing geometric consistency.
- Log warnings without crashing the graph.

##13 ##repo - Repository Layout (proposed)
- `README.md` — overview, install, quickstart, examples
- `docs/requirements.md` — this document
- `nodes/CameraStabilize.py` — node implementation
- `nodes/utils/oneeuro.py` — OneEuro filter
- `examples/` — sample workflows and clips
- `tests/` — unit and golden tests

##14 ##roadmap - Future Work
- Dense optical-flow global fit; simple rolling-shutter compensation.
- GyroFlow/IMU backend.
- Learning-based flow (e.g., RAFT) as optional backend (not default).
- Temporal consistency smoothing for outpainted boundaries.
