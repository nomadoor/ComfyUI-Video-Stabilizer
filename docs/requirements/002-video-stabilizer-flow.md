##00 ##overview - Purpose
- Deliver a higher fidelity **video stabilization** node that leverages dense optical flow and global motion consensus to minimise residual shake while preserving scene structure.
- Key features:
  - Optical-flow backends: `DIS` (fast) / `DeepFlow` (robust) with auto-selection.
  - Dual-stage motion model: per-frame dense flow → global similarity/perspective fit with outlier rejection and bundle smoothing.
  - Adaptive padding: smart crop budget with feathered mask for seamless inpainting or frame blending.
  - Optional GPU acceleration when available (Torch/NumPy fallback otherwise).

##01 ##id - Identification
- ID: `VideoStabilizerFlow`
- Status: Draft
- Updated: 2025-10-29
- Repository: **ComfyUI-Video-Stabilizer**

##02 ##scope - In/Out of Scope
- In scope:
  - Handheld footage with complex parallax and rolling shutter approximations.
  - Dense optical flow computation (OpenCV contrib) and global agreement via least squares / bundle adjustment.
  - GPU acceleration via PyTorch tensors when available; CPU compatibility retained.
- Out of scope (v1):
  - Learning-based or transformer stabilizers.
  - Direct IMU/Gyro ingestion (future extension).
  - Keyframe editing UI; output remains deterministic per parameters.

##03 ##usecases - Use Cases
- Dramatically reduce residual shake on footage with significant depth variation.
- Feed stabilized frames to video diffusion/outpainting pipelines with accurate feather masks.
- Offer “quality-first” alternative for users willing to trade compute time for stability.

##04 ##params - Input Parameters (initial proposal)
- **flow_backend** (default: `DIS`)  
  - `DIS`	 Fast pyramidal dense flow (balanced quality).  
  - `DeepFlow`	 High-quality dense flow; slower but more tolerant of large motion.
- **pyramid_levels** *1 E0* (default: `5`)  
  - Number of pyramid levels passed to the flow backend.
- **smoothing_strength** *0.0 E.* (default: `0.65`)  
  - Amount of trajectory regularisation (higher = smoother). Uses windowed bundle smoothing.
- **crop_budget** *0.0 E.0* (default: `0.6`)  
  - Max zoom allowance (1.0 adds 100% crop budget). Shared with feather mask logic.
- **mask_feather_px** *0 E0* (default: `12`)  
  - Feather radius in pixels for mask falloff.
- **reference_model** (default: `similarity`)  
  - `translation` | `similarity` | `perspective` – global motion model fit post-flow.
- **use_gpu** (default: `auto`)  
  - `auto` / `force_cpu` / `force_gpu` – toggles Torch acceleration when available.

##05 ##io - I/O Specification
- Input: `frames` (IMAGE list / video tensor).
- Output:
  - `stabilized_frames` (IMAGE list)
  - `mask` (IMAGE list, float 0–1 feathered)
  - `motion_debug` (optional JSON/string with per-frame stats; optional output controlled via parameter)

##06 ##behavior - Internal Flow
1. Convert frames to GPU tensors if enabled; otherwise NumPy arrays.
2. Compute dense flow between consecutive frames using selected backend (DIS or DeepFlow) with confidence maps.
3. Downsample trajectories to sparse control points via global least squares; compute similarity/perspective transforms per frame.
4. Run sliding-window bundle adjustment with `smoothing_strength` controlling regularisation strength.
5. Apply resulting transforms via warpPerspective/warpAffine; track per-frame zoom required relative to `crop_budget`.
6. Generate feathered masks by warping a full-one mask and applying Gaussian/box blur with `mask_feather_px`.
7. Update progress after each frame via `get_progress_state().update_progress(...)`.

##07 ##ux - UI Strings (ComfyUI)
- flow_backend: Flow Backend (DIS / DeepFlow)
- pyramid_levels: Flow Pyramid Levels
- smoothing_strength: Trajectory Smoothing
- crop_budget: Crop Budget
- mask_feather_px: Mask Feather (px)
- reference_model: Global Motion Model
- use_gpu: GPU Acceleration
- debug_output (bool): Emit Motion Debug Output

##08 ##defaults - Defaults
- flow_backend: `DIS`
- pyramid_levels: `5`
- smoothing_strength: `0.65`
- crop_budget: `0.6`
- mask_feather_px: `12`
- reference_model: `similarity`
- use_gpu: `auto`
- debug_output: `false`

##09 ##accept - Acceptance Criteria
- Flow backend selection works; fallback to DIS if DeepFlow unavailable.
- Stabilized output exhibits measurably lower jitter than Classic on parallax-heavy clips.
- Mask feathering produces smooth transitions with no hard edges.
- GPU path matches CPU output within tolerance.
- Progress bar updates consistently across long sequences.
- Deterministic results for same inputs and parameters.

##10 ##tests - Test Plan
- Unit: verify flow backend selection, mask feather radius, deterministic behaviour.
- Regression: compare stabilization metrics (e.g., variance of camera path) vs baseline on sample videos.
- GPU/CPU parity tests for identical seeds.
- Performance: ensure DIS backend completes 1080p@120 frames within acceptable time on reference hardware.

##11 ##perf - Performance
- Expect slower runtime than Classic; target >5 fps 1080p with DIS on modern CPU, >15 fps with GPU.
- DeepFlow path flagged as high quality; may drop below 2 fps CPU.
- Optimize by caching pyramids and using shared memory buffers.

##12 ##reliability - Reliability & Exceptions
- If optical flow fails (confidence below threshold), fallback to Classic similarity estimation to avoid blank frames.
- When crop budget exceeded, reduce smoothing strength adaptively to retain frame coverage.
- Missing GPU: gracefully fall back to CPU without crashing.
- Provide clear warnings when DeepFlow backend unavailable (requires OpenCV contrib).

##13 ##repo - Repository Notes
- Node file: `nodes/video_stabilizer_flow.py`
- Shared utilities can live under `nodes/utils/flow.py`
- Update README / CHANGELOG / ADR after implementation.
- Requirements additions: `opencv-contrib-python>=4.9`, optional torch GPU detection.

##14 ##roadmap - Future Work
- Investigate temporal consistency mask refinement using flow confidence.
- Optional IMU fusion backend.
- Adaptive crop budgeting driven by scene content.
- Batch mode for precomputed flows to re-run with different smoothing settings.
