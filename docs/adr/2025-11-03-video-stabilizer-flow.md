# 2025-11-03 — Flow-Based Video Stabilizer Node

## Context
- Requirement file `docs/requirements/002-video-stabilizer-flow.md` calls for a dense optical flow variant that reuses the classic node schema.
- We already exposed the classic (feature-tracking) node via the shared extension; the flow node should sit alongside it without altering the UI contract.

## Decision
- Added `nodes/video_stabilizer_flow.py` implementing `VideoStabilizerFlow`.
  - Estimates motion with OpenCV DIS optical flow (auto-falling back to TV-L1 when DIS is unavailable).
  - Fits the requested global transform (translation/similarity/perspective) on subsampled dense correspondences with the same fallback policy as the classic node.
  - Shares framing logic, progress reporting, and metadata shape with the feature-tracking node.
- Updated `__init__.py` to register both stabilizer nodes through a unified extension.

## Consequences
- Users now have a drop-in alternative better suited to low-texture or subject-dominant footage while keeping identical graph wiring.
- Metadata includes the effective flow backend so downstream tools can audit which solver ran.
