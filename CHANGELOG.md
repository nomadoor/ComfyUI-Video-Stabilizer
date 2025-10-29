# Changelog

## 0.2.0 - 2025-10-29

- Added VideoStabilizerFlow dense optical-flow stabilizer with DIS/DeepFlow backends and feathered mask output (see docs/requirements/002-video-stabilizer-flow.md).
- Updated README/ADR, expanded .gitignore, added MIT LICENSE, and bumped defaults (smoothness=0.5, stabilize_zoom=0.5).
- Removed One Euro dependency in favour of Gaussian trajectory smoothing; progress bar updates for long runs.

## 0.1.0 - 2025-10-29

- Added VideoStabilizerNode (ID VideoStabilizerClassic) implementing classic CV stabilization per docs/requirements/001-video-stabilizer.md.
