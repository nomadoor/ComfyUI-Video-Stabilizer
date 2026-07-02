# ComfyUI Video Stabilizer

[![日本語版](https://img.shields.io/badge/README-日本語版-gray.svg)](README_ja.md)

A ComfyUI custom node package for CPU-friendly video stabilization, padding mask generation, reusable motion metadata, and motion restore workflows.

It provides six nodes:

- **Classic**: sparse feature tracking with OpenCV / NumPy
- **Flow**: dense optical flow using OpenCV DIS by default
- **Motion Apply**: applies motion metadata to frames
- **Shake Generator**: creates deterministic style-based motion metadata without changing pixels
- **Shake Generator Manual**: creates the same kind of metadata from explicit recipe values

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

## Installation

- Install via ComfyUI Manager.

## Nodes

| Node | Purpose |
| --- | --- |
| `Video Stabilizer Classic` | Lightweight general-purpose stabilization using feature tracking. |
| `Video Stabilizer Flow` | Higher-accuracy stabilization using DIS optical flow. TV-L1 is optional when `cv2.optflow` is available. |
| `Video Stabilizer Motion Apply` | Applies `motion_meta` JSON to frames with pad/crop framing and optional motion blur. |
| `Video Stabilizer Shake Generator` | Emits deterministic shake `motion_meta`; `style` chooses motion character and `amount` controls strength. |
| `Video Stabilizer Shake Generator Manual` | Emits deterministic shake `motion_meta` from explicit pan/tilt/roll/zoom recipe values. |
| `Video Stabilizer Inverse` | Deprecated compatibility node for restoring removed camera shake. |

Flow normally uses DIS optical flow. If unavailable, it automatically falls back through TV-L1, translation estimation, and identity.

## Usage

Input a video or batched images into either `Video Stabilizer Classic` or `Video Stabilizer Flow`.

Use `padding_mask` when you want VACE or another outpainting workflow to fill borders created by stabilization.

Connect Classic/Flow `meta` to `Video Stabilizer Motion Apply` to restore the removed motion after editing stabilized frames. Connect Shake Generator output to Motion Apply when you want to add generated handheld motion.

For tuning, start with `Video Stabilizer Shake Generator`, inspect `motion_meta.generator.recipe`, then enter those values in `Video Stabilizer Shake Generator Manual` and adjust the absolute pan/tilt/roll/zoom recipe.

## Parameters

Shared by Classic / Flow:

| Parameter | Default | Notes |
| --- | ---: | --- |
| `frame_rate` | `16.0` | Input FPS used to scale the temporal smoothing window. |
| `framing_mode` | `crop_and_pad` | `crop`, `crop_and_pad`, or `expand`. |
| `transform_mode` | `similarity` | `translation`, `similarity`, or `perspective`. |
| `camera_lock` | `false` | Pulls motion toward a tripod-like result. |
| `strength` | `0.7` | Removal gain for estimated camera motion. Ignored while `camera_lock` is on. |
| `smooth` | `0.5` | Temporal smoothing amount. Ignored while `camera_lock` is on. |
| `keep_fov` | `0.6` | Crop-mode FOV preservation. `1.0` means no zoom; `0.0` allows maximum zoom. |
| `padding_color` | `#7F7F7F` | HEX fill color for padded areas. Can use the core Color Picker `hex` output. |

Framing modes:

| Mode | Behavior |
| --- | --- |
| `crop` | Hides borders with an aspect-ratio-preserving zoom/crop, reducing FOV. |
| `crop_and_pad` | Limits zoom and pads remaining empty regions. |
| `expand` | Does not crop at all; expands the canvas as needed. |

Shake Generator:

| Parameter | Default | Notes |
| --- | ---: | --- |
| `style` | `handheld` | `tripod`, `handheld`, `walking`, `action`, or `vibration`. |
| `amount` | `1.0` | Overall shake amplitude. |
| `speed` | `1.0` | Overall shake speed. |
| `seed` | `0` | Deterministic seed. |

Shake Generator Manual exposes the resolved recipe directly: `pan`, `tilt`, `roll`, `zoom`, `drift_freq`, `tremor`, `tremor_freq`, `jitter_rate`, `step`, `randomness`, and `virtual_fov`.

Motion Apply:

| Parameter | Default | Notes |
| --- | ---: | --- |
| `framing_mode` | `pad` | `pad` or `crop`. |
| `interpolation` | `bilinear` | `bilinear` or `bicubic`. |
| `motion_blur` | `0.0` | Shutter fraction. `0.5` is roughly a 180-degree shutter. |
| `motion_blur_samples` | `9` | Displayed as Blur Quality. Higher values average more shutter samples and are smoother but slower. |

## Outputs

| Output | Notes |
| --- | --- |
| `frames_stabilized` | Stabilized video frames. |
| `padding_mask` | Mask of padded / missing regions. |
| `meta` | JSON diagnostics, including estimated motion, applied stabilization matrices, and additive `motion_meta`. |

## Inverse Stabilization

`Video Stabilizer Motion Apply` replaces the old inverse workflow. `Video Stabilizer Inverse` is deprecated and kept only for compatibility.

With `crop` / `crop_and_pad`, gaps will almost always appear at the end because pixels were cropped or padded during stabilization. Use `expand` when you plan to use Inverse.

## Example Workflows

- [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
- [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)
- [Sample Video (Pexels)](https://www.pexels.com/ja-jp/video/29507473/)

## License

MIT
