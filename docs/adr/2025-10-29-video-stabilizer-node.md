# ADR: Video Stabilizer Node Implementation

- **Date:** 2025-10-29
- **Status:** Accepted
- **Context:** Requirement document `docs/requirements/001-video-stabilizer.md` calls for a classic CV video stabilization node that outputs both stabilized frames and a missing-area mask aligned with ComfyUI V3 schemas.
- **Decision:** Implemented `VideoStabilizerNode` (ID `001-video-stabilizer`) using OpenCV feature tracking (GoodFeaturesToTrack with ORB fallback), LK optical flow, affine/homography estimation, and One Euro filtering for trajectory smoothing. Added deterministic framing modes (`CROP`, `CROP_and_PAD`, `NO_CROP_PAD`) and a reusable One Euro filter utility under `nodes/utils/oneeuro.py`.
- **Consequences:** Provides a CPU-friendly stabilization path without additional dependencies beyond OpenCV/NumPy. Downstream workflows receive a dedicated mask for outpainting. Future enhancements can build on the shared One Euro utility or extend framing heuristics without breaking the schema.

