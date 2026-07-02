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

Classic and Flow still emit the existing `meta.stabilization_warp` unchanged. Their `motion_meta` block is additive and stores the removed camera motion by inverting each recorded stabilization matrix.

## Motion Apply

Inputs:

- `frames`
- `motion_meta` as `io.Custom("JSON")`
- `framing_mode: pad | crop`
- `interpolation: bilinear | bicubic`
- `padding_color`
- `motion_blur: 0.0..1.0`
- `motion_blur_samples: 3..33`

Outputs:

- `frames`
- `padding_mask`
- `meta`

Behavior:

- Validate frame size and frame count against `motion_meta`.
- `pad` warps each frame to `output_size` and emits a padding mask.
- `crop` finds a shared valid region, applies an aspect-preserving center crop, and emits an all-zero mask.
- If `crop` cannot find a usable shared region without more than 4x scale, fall back to `pad` and record `framing_fallback: pad`.
- Legacy `stabilization_warp` metadata resolves to `motion_meta` by inverting `applied_matrix`, so Motion Apply with `pad` and `bilinear` matches the deprecated Inverse behavior.
- `motion_blur=0.0` uses the same path as unblurred apply.
- `motion_blur>0.0` linearly interpolates adjacent frame matrices, warps multiple shutter samples, averages frames, and emits a soft padding mask based on mean coverage.
- Motion Apply reports progress during both crop-mask analysis and frame warping. With motion blur enabled, progress advances per shutter sample.
- The deprecated Inverse wrapper always calls Motion Apply with blur disabled.

## Compatibility

The JSON socket type remains `io.Custom("JSON")`. Existing `stabilization_warp` metadata remains valid and unchanged.
