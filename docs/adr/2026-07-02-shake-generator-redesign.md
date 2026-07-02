# ADR: Shake Generator Redesign

Date: 2026-07-02

## Decision

Replace the original `preset`, `strength`, `speed`, and `detail` controls with a two-node design:

- `Video Stabilizer Shake Generator` for style-based generation.
- `Video Stabilizer Shake Generator Manual` for absolute recipe editing.

Generate shake in angle space (`pan_deg`, `tilt_deg`, `roll_deg`, `zoom_log`) and project it to the existing absolute pixel-coordinate 3x3 matrix convention.

Move motion blur to Video Stabilizer Motion Apply.

## Rationale

The old presets were mostly strength levels, which overlapped with `strength` and made the UI hard to reason about. The new `style` control describes the character of the motion, while `amount` controls volume and `pace` controls speed.

Putting style presets and manual controls in one Python-only node was also rejected. ComfyUI Python nodes cannot update other widget values when one widget changes without a custom frontend extension. Style presets and manual editing therefore need separate nodes.

Angle-space motion better matches camera behavior. Pan and tilt become resolution-independent camera angles, and `virtual_fov` naturally creates wider or telephoto-feeling shake without a separate telephoto preset.

The old generator subtracted the clip mean from each noise channel. On short clips this removed much of the low-frequency drift. The new model subtracts the first frame instead, preserving drift while making the first generated matrix identity.

Motion blur is implemented in Motion Apply because it depends on how matrices are applied to pixels and should also work for estimated or legacy motion metadata.

The resolved `generator.recipe` block makes the style preset visible and copyable into Manual mode.

## Consequences

The Shake Generator schema changes are allowed because the node is unreleased. Existing `motion_meta` v2 semantics remain unchanged: Motion Apply consumes `input_to_output` matrices as-is.
