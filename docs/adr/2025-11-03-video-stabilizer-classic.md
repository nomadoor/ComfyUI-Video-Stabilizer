# 2025-11-03 — Classic Video Stabilizer Node

## Context
- Requirements file `docs/requirements/001-video-stabilizer.md` defines a CPU-only, feature-tracking stabilizer node that must follow the ComfyUI V3 schema, emit deterministic transforms, expose framing controls, and always output a padding mask plus metadata.
- The repository previously had no nodes implementing this behavior or any V3 schema scaffolding.

## Decision
- Added `nodes/video_stabilizer_classic.py` providing a `VideoStabilizerClassic` V3 node and its `ComfyExtension` entry point.
  - Uses OpenCV corner detection + LK optical flow to estimate frame-to-frame motion.
  - Supports translation, similarity, and perspective transforms with automatic fallback and confidence tracking.
  - Separates estimation from application, smooths the cumulative path, applies strength/camera lock logic, and handles framing modes (`crop`, `crop_and_pad`, `expand`) while emitting deterministic padding masks.
  - Returns structured JSON metadata that captures per-transition confidence, fallback events, motion paths, and framing choices.
- Schema ports are aligned with common ComfyUI image workflows: inputs accept standard `IMAGE` batches, and outputs provide `IMAGE` (stabilized frames) plus `MASK` (padding map) sockets so downstream nodes can connect without adapters.
- Renamed the zoom limiter control to `keep_fov`, restrict it to `crop` mode only, and record `keep_fov_requested` / `keep_fov_effective` in metadata when applied; other framing modes simply document the padding/centering region.
- Introduced a string-based `padding_color` input (`"R, G, B"`) to control border fills for `crop_and_pad` / `expand`, and propagate the chosen color into both processing and metadata.
- Adhered to spec defaults (strength 0.7, smooth 0.5, etc.) and remained dependency-neutral beyond OpenCV (already allowed by the requirements).

## Consequences
- The project now exposes a stable Classic stabilizer node ready for `codex-nodegen`/`codex-validator` flows.
- Downstream docs/tests can rely on the emitted metadata to audit confidence, padding ratios, and applied transforms.
- Future flow-based variants (`docs/requirements/002-video-stabilizer-flow.md`) can reuse the schema and wrapper with alternative internal estimators.
