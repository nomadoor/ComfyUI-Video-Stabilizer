# ComfyUI Video Stabilizer

Two complementary stabiliser nodes for ComfyUI:

| Node | ID | Highlights |
| --- | --- | --- |
| Video Stabilizer (Classic) | VideoStabilizerClassic | Feature tracking + Gaussian-smoothed affine/perspective motion. Fast, CPU friendly. |
| Video Stabilizer (Flow) | VideoStabilizerFlow | Dense optical-flow consensus with feathered mask output and optional debug stats. Higher quality, heavier compute. |

## Classic Node (VideoStabilizerClassic)

- Motion models: **translation**, **similarity**, or **perspective**.
- Gaussian trajectory smoothing (smoothness 0.0?1.0) with adaptive crop budget (stabilize_zoom).
- Framing control via CROP (max common crop) or CROP_and_PAD (minimal crop plus padding colour).
- Per-frame binary masks for downstream inpainting/compositing.
- Deterministic CPU pipeline (OpenCV + NumPy).

## Flow Node (VideoStabilizerFlow)

- Dense optical flow backends: DIS (fast) or DeepFlow (robust, requires opencv-contrib).
- Bundle smoothing over similarity/perspective transforms with adjustable strength.
- Feathered mask generation (mask_feather_px) and adaptive crop budget (crop_budget).
- Optional motion debug output (JSON) and progress bar updates for long clips.
- GPU auto-detection (Torch) for future acceleration paths; graceful CPU fallback.

## Installation

`ash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-org>/ComfyUI-Video-Stabilizer.git
pip install -r requirements.txt  # includes NumPy / OpenCV (contrib recommended for DeepFlow)
`

## Quick Reference

| Parameter | Classic | Flow | Notes |
|-----------|---------|------|-------|
| smoothness / smoothing_strength | ? | ? | Controls trajectory smoothing strength. |
| stabilize_zoom / crop_budget | ? | ? | Crop allowance (1.0 ¨ 2~ zoom budget). |
| low_backend | ? | ? | DIS or DeepFlow optical flow. |
| mask_feather_px | ? | ? | Feather radius for flow masks. |
| emit_debug | ? | ? | Emit JSON diagnostics for analysis. |
| pad_color_rgb | ? | ? | Padding colour for uncovered regions. |

> Want a no-crop result? Use CROP_and_PAD with stabilize_zoom = 0 (Classic) or set crop_budget = 0 (Flow).

## Development

- Python 3.10+ recommended (matching ComfyUI baseline).
- Run python -m compileall nodes before committing to validate syntax.
- Contributions welcome?especially around GPU acceleration and dense-flow refinements.
