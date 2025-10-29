# ADR: Video Stabilizer Node Implementation

- **Date:** 2025-10-29
- **Status:** Accepted
- **Context:** Requirement document `docs/requirements/001-video-stabilizer.md` calls for a classic CV video stabilization node that outputs both stabilized frames and a missing-area mask aligned with ComfyUI V3 schemas.
- **Decision:** Implemented `VideoStabilizerNode` (ID `VideoStabilizerClassic`) using OpenCV feature tracking (GoodFeaturesToTrack with ORB fallback), LK optical flow, affine/homography estimation, and Gaussian-smoothed trajectory parameters. Final design exposes framing modes (`CROP`, `CROP_and_PAD`), per-frame masks, and progress bar updates via `comfy_execution.progress`.
- **Consequences:** Provides a CPU-friendly stabilization path without additional dependencies beyond OpenCV/NumPy. Downstream workflows receive a dedicated mask for outpainting and clear UI feedback during long jobs. NO_CROP_PAD was removed to reduce overlap with CROP_and_PAD. Default smoothing/zoom (0.5/0.5) deliver visible stabilization out of the box.

## 2025-10-29 Updates

- Replaced the One Euro filter with Gaussian smoothing over translation/similarity/perspective parameters; no separate `oneeuro` utility required.
- Adjusted defaults (`smoothness = 0.5`, `stabilize_zoom = 0.5`) and removed `NO_CROP_PAD`; use `CROP_and_PAD` with `stabilize_zoom = 0` for the same behaviour.
- Added node-head progress indication by calling `get_progress_state().update_progress(...)` during frame iteration so users see time-to-completion.
- Repository hygiene: expanded `.gitignore`, added MIT `LICENSE`, refreshed README, documented minimal `requirements.txt`.
- Future work: prototype a higher-quality stabilizer using dense optical flow + global consensus (e.g. OpenCV DeepFlow/DIS) on a dedicated branch.
