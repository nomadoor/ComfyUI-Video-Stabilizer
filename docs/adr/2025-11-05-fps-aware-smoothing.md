# 2025-11-05 – Introduce FPS Input For Stabilizers

## Status
Accepted

## Context
- The classic/flow stabilizer nodes previously treated the `smooth` value as a fixed-sized moving window (3–13 frames).
- This caused higher-FPS material (e.g. 60/120 fps) to receive very little smoothing compared with low-FPS footage.
- Users reported inconsistent stabilisation strength and requested explicit control over the assumed frame rate.

## Decision
- Added a required `frame_rate` input (default 16.0 fps, step 0.1) to both classic and flow nodes.
- Converted the internal smoothing kernel to be time-based: the window now scales with `frame_rate`, keeping perceived smoothing constant across frame rates.
- Recorded both the requested and effective FPS in the output metadata so downstream tooling can audit the chosen value.

## Consequences
- Graphs now need to supply (or accept the default for) `Input FPS`. Existing workflows without saved parameter order may need to reconnect this input.
- Stabilisation behaviour is more consistent across different FPS sources, reducing surprises when switching from low-FPS to high-FPS footage.
- Further work is still needed to resolve crop-mode padding issues; these will be tackled separately.
