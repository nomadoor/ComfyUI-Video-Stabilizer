# ComfyUI Video Stabilizer

[![日本語版](https://img.shields.io/badge/README-日本語版-gray.svg)](README_ja.md)

A ComfyUI custom node package for video stabilization.

It mainly provides three functions:

- Stabilize camera shake in a video
- Restore the original shake from a stabilized video
- Add artificial shake

https://github.com/user-attachments/assets/7da060c1-d775-47b7-91e6-f7a2ce147389

## Nodes

| Node | Purpose |
| --- | --- |
| `Video Stabilizer Classic` | Lightweight stabilizer using feature tracking. |
| `Video Stabilizer Flow` | Higher-accuracy stabilizer using dense optical flow. |
| `Video Stabilizer Motion Apply` | Applies `motion_meta` to frames (reapply, restore, or add generated shake). |
| `Video Stabilizer Shake Generator` | Generates deterministic shake `motion_meta`. Does not touch the frames at all. |
| `Video Stabilizer Shake Generator Manual` | Same as above, with pan/tilt/roll/zoom specified directly for fine control. |

## Installation

- Install via ComfyUI Manager.

## Usage

### Stabilization

Just feed a video or a batch of images into `Video Stabilizer Classic` or `Video Stabilizer Flow` to stabilize it.
`Classic` is a lightweight method using feature tracking; `Flow` uses dense optical flow, so it's heavier but more accurate. Unless processing speed is a real concern, `Flow` is the safe default.

Main parameters:

| Parameter | Description |
| --- | --- |
| `framing_mode` | How to handle the areas exposed by stabilization. `crop` (zooms in to hide them, narrowing the field of view) / `crop_and_pad` (limits zoom and pads the remaining gaps) / `expand` (doesn't crop at all, expands the canvas as needed). |
| `strength` | How much of the camera motion to remove. Closer to `1.0` gives a more locked-off result. |
| `smooth` | How strongly to smooth the motion over time. Higher values leave only slow, sweeping motion. |
| `camera_lock` | When on, pushes harder toward a tripod-like result. Used instead of `strength` / `smooth`. |
| `keep_fov` | How much field of view to sacrifice in `crop`-style modes. `1.0` means no zoom, `0.0` allows maximum zoom. |
| `padding_color` | The fill color for padded areas. |

### Restoring stabilized shake

The `meta` output of `Classic` / `Flow` records the exact camera motion that was removed. Connect it to `Video Stabilizer Motion Apply` and the same motion can be reproduced.

For example, use this when you've stabilized footage, regenerated or edited it with a video generation AI, and want to put the original camera work back on top of the result — removing the shake to make editing easier, then restoring the natural camera motion at the end.

### Adding artificial shake

Conversely, to add a handheld feel to static footage, generate shake `motion_meta` with `Video Stabilizer Shake Generator` and connect it to `Video Stabilizer Motion Apply` to apply it.

Main parameters:

| Parameter | Description |
| --- | --- |
| `style` | The character of the shake. `tripod` (nearly static) / `handheld` (natural handheld) / `walking` (up-and-down motion from walking) / `action` (intense motion) / `vibration` (mechanical vibration, e.g. inside a vehicle). |
| `amount` | Overall shake strength. |
| `speed` | Overall shake speed. |
| `seed` | The same value always reproduces the same shake. |

For finer control, use `Video Stabilizer Shake Generator Manual`. Instead of picking a `style`, you can directly specify numeric values such as pan / tilt / roll / zoom and the shake frequencies.

### Combining with video generation AI

Stabilization generally exposes areas at the edges, which are normally hidden by cropping — at the cost of field of view.
This node outputs those exposed areas as a mask, so a video generation AI can outpaint them, letting you stabilize footage without sacrificing the field of view.

Example workflows:

- [Wan2.1_VACE_outpainting_VideoStabilizer.json](example_workflows/Wan2.1_VACE_outpainting_VideoStabilizer.json)
- [Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json](example_workflows/Wan2.2-VACE-Fun_outpainting_VideoStabilizer.json)

([Sample video](https://www.pexels.com/ja-jp/video/29507473/))

## Learn more

For a closer look at how the parameters behave and some simple workflows, see:

- [日本語](https://comfyui.nomadoor.net/ja/notes/comfyui-video-stabilizer/)
- [English](https://comfyui.nomadoor.net/en/notes/comfyui-video-stabilizer/)
- [中文](https://comfyui.nomadoor.net/zh/notes/comfyui-video-stabilizer/)

## License

MIT
