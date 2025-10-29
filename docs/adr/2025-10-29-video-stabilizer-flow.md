# ADR: Video Stabilizer Flow Node Implementation

- **Date:** 2025-10-29
- **Status:** Draft
- **Context:** Requirement document docs/requirements/002-video-stabilizer-flow.md specifies a high-fidelity stabiliser leveraging dense optical flow and global consensus for shots with significant parallax.
- **Decision:** Build VideoStabilizerFlow using OpenCV DIS/DeepFlow dense flow, global similarity/perspective fitting, Gaussian trajectory smoothing, and feathered masks with adaptive crop budgeting. Provide optional debug telemetry and progress updates via comfy_execution.progress.
- **Consequences:** Higher compute cost vs. Classic but improved stability on complex scenes; relies on OpenCV contrib for DeepFlow, with graceful fallback to DIS.

## Notes

- Future work: GPU acceleration, optical-flow confidence-aware masks, IMU fusion.
