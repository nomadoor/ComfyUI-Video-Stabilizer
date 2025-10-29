##00 ##overview - Purpose
- Deliver a higher fidelity **video stabilization** node that leverages dense optical flow and global motion consensus to minimise residual shake while preserving scene structure.
- Key features:
  - Dense optical flow derived from OpenCV DIS with adaptive downsampling for speed and robustness.
  - Dual-stage motion model: per-frame dense flow → global similarity/perspective fit with outlier rejection and bundle smoothing.
  - Adaptive padding: smart crop budget with deterministic masks for downstream inpainting workflows.

##01 ##id - Identification
- ID: `VideoStabilizerFlow`
- Status: Draft
- Updated: 2025-10-29
- Repository: **ComfyUI-Video-Stabilizer**

##02 ##scope - In/Out of Scope
- In scope:
  - Handheld footage with complex parallax and rolling shutter approximations.
  - Dense optical flow computation (OpenCV contrib DIS) and global agreement via least squares / bundle adjustment.
  - CPU execution with adaptive flow resolution to balance quality and runtime.
- Out of scope (v1):
  - Alternative flow backends (DeepFlow, RAFT, etc.).
  - GPU acceleration paths.
  - Feathered mask generation (mask remains binary to reflect missing regions exactly).

##03 ##usecases - Use Cases
- Dramatically reduce residual shake on footage with significant depth variation.
- Feed stabilized frames to video diffusion/outpainting pipelines with deterministic missing-region masks.
- Offer a “quality-first” alternative for users willing to trade compute time for stability.

##04 ##params - Input Parameters (current implementation)
- **method** (default: `similarity`)
  - `translation` | `similarity` | `perspective` — global motion model fit post-flow.
- **smoothness** (default: `0.5`, range `0.0`–`1.0`)
  - Controls trajectory regularisation (higher = smoother motion, more border loss).
- **stabilize_zoom** (default: `0.5`, range `0.0`–`1.0`)
  - Maximum automatic zoom (crop) applied to hide borders.
- **framing** (default: `CROP_and_PAD`)
  - `CROP` enforces a consistent crop window; `CROP_and_PAD` pads exposed areas with the chosen colour.
- **pad_color_rgb** (default: `128,128,128`)
  - Padding colour (comma-separated R,G,B) used when framing requires fill.

##05 ##io - I/O Specification
- Input: `frames` (IMAGE tensor batch).
- Output:
  - `stabilized_frames` (IMAGE batch)
  - `missing_mask` (MASK batch, binary float)

##06 ##behavior - Internal Flow
1. Convert frames to float tensors and numpy arrays as needed.
2. Generate grayscale frames and compute dense DIS flow with adaptive downsampling and post blur.
3. Sample flows onto a sparse lattice and fit the selected motion model per frame.
4. Apply One-Euro smoothing to trajectory parameters.
5. Enforce zoom allowance via iterative blending when `framing=CROP`.
6. Warp frames with the stabilisation transform and optional zoom; build masks by warping a ones tensor.
7. Pad or crop outputs based on `framing`; emit deterministic masks.
8. Update progress via `get_progress_state().update_progress(...)`.

##07 ##ux - UI Strings (ComfyUI)
- method: Motion Model
- smoothness: Smoothing
- stabilize_zoom: Zoom Allowance
- framing: Framing
- pad_color_rgb: Padding Color

##08 ##defaults - Defaults
- method: `similarity`
- smoothness: `0.5`
- stabilize_zoom: `0.5`
- framing: `CROP_and_PAD`
- pad_color_rgb: `128,128,128`

##09 ##accept - Acceptance Criteria
- DIS flow computation succeeds for typical 720p–4K footage with deterministic results.
- Output video exhibits lower jitter than Classic on parallax-heavy clips given the same settings.
- Binary mask accurately marks missing regions (no false positives).
- Progress bar updates consistently across long sequences.

##10 ##tests - Test Plan
- Unit: verify parameter validation, mask dimensions, deterministic behaviour.
- Regression: compare stabilisation metrics (camera path variance) vs Classic on sample videos.
- Performance: ensure default DIS path completes 1080p@120 frames within acceptable time on reference hardware.
- CROP framing: confirm no padding bleed and consistent crop window.

##11 ##perf - Performance
- Expect slower runtime than Classic; target ≥5 fps 1080p with DIS on modern CPU.
- Adaptive downsampling keeps work proportional to frame size without degrading results excessively.

##12 ##reliability - Reliability & Exceptions
- If optical flow fails, fall back to identity transform for that frame to avoid blank outputs.
- When zoom budget is exceeded, iteratively relax smoothing until coverage fits.
- Missing dependencies (opencv-contrib) halt the node with a clear error message.

##13 ##repo - Repository Notes
- Node file: `nodes/video_stabilizer_flow.py`
- Shared utilities: reuse helpers under `nodes/utils/`.
- Update README / CHANGELOG / ADR after major behaviour changes.
- Requirements additions: `opencv-contrib-python>=4.9,<4.13`.

##14 ##roadmap - Future Work
- Investigate optional DeepFlow or learning-based flow backends once dependency risk is acceptable.
- Explore GPU acceleration (CUDA DIS or PyTorch RAFT) behind a feature flag.
- Evaluate feathered mask output for downstream blending when quality requirements demand it.
