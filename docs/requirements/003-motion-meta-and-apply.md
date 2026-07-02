# Motion Meta v2 and Motion Apply Requirements

Last updated: 2026-07-02 (UTC+9)

## Purpose

`motion_meta` v2 is the shared producer/consumer contract for camera-motion metadata. Matrices are applied as-is by Video Stabilizer Motion Apply, using `input_to_output` absolute pixel coordinates.

## Contract

`meta.motion_meta` has:

- `version: 2`
- `source: estimated_classic | estimated_flow | generated_shake | legacy_stabilization`
- `frame_count`
- `fps`
- `input_size: [width, height]`
- `output_size: [width, height]`
- `matrix_convention: input_to_output`
- `per_frame[].index`
- `per_frame[].matrix`
- `generator` only for `generated_shake`

Classic and Flow still emit the existing `meta.stabilization_warp` unchanged. Their `motion_meta` block is additive and stores the same input-to-output matrices that were applied during stabilization, so original frames can be connected directly to Motion Apply with the emitted metadata.

## Motion Apply

Inputs:

- `frames`
- `motion_meta` as `io.Custom("JSON")`
- `framing_mode: crop_and_pad | crop | expand`
- `interpolation: bilinear | bicubic`
- `padding_color`
- `motion_blur: 0.0..1.0`
- `motion_blur_quality: Draft | Standard | High | Ultra`

Outputs:

- `frames`
- `padding_mask`
- `meta`

Behavior:

- Validate frame size and frame count against `motion_meta`.
- `crop_and_pad` warps each frame to `motion_meta.output_size`, clips outside that canvas, pads uncovered pixels, and emits a padding mask.
- `crop` finds a shared valid region, applies an aspect-preserving center crop, and emits an all-zero mask.
- `expand` computes the union bounding box of all transformed frame corners, prepends a translation, and expands the output canvas so no transformed source content is clipped. Individual frames may still have padding inside the union canvas.
- If `crop` cannot find a usable shared region without more than 4x scale, fall back to `crop_and_pad` and record `framing_fallback: crop_and_pad`.
- Legacy `stabilization_warp` metadata resolves to `motion_meta` by inverting `applied_matrix`, so Motion Apply with `crop_and_pad` and `bilinear` matches the deprecated Inverse behavior.
- If both top-level `motion_meta` and legacy `stabilization_warp` are present, Motion Apply selects the block whose input size matches the connected frames. This lets Classic/Flow metadata work for both original-frame reapplication and stabilized-frame inverse restoration.
- `motion_blur=0.0` uses the same path as unblurred apply.
- `motion_blur>0.0` linearly interpolates adjacent frame matrices, warps multiple shutter samples, averages frames, and emits a soft padding mask based on mean coverage. Blur Quality maps to shutter sample counts: Draft=5, Standard=9, High=17, Ultra=33.
- Motion Apply reports progress during both crop-mask analysis and frame warping. With motion blur enabled, progress advances per shutter sample.
- The deprecated Inverse wrapper always calls Motion Apply with blur disabled.

## Compatibility

The JSON socket type remains `io.Custom("JSON")`. Existing `stabilization_warp` metadata remains valid and unchanged.
