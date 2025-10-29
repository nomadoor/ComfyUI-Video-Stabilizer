# ComfyUI Video Stabilizer

Two complementary stabiliser nodes for ComfyUI:

| Node | ID | Highlights |
| --- | --- | --- |
| Video Stabilizer (Classic) | VideoStabilizerClassic | Feature tracking + Gaussian-smoothed affine/perspective motion. Fast, CPU friendly. |
| Video Stabilizer (Flow) | VideoStabilizerFlow | Dense DIS optical flow with global motion smoothing. Higher quality, heavier compute. |

## Classic Node (VideoStabilizerClassic)

- Motion models: **translation**, **similarity**, or **perspective**.
- Gaussian trajectory smoothing (smoothness 0.0?1.0) with adaptive crop budget (stabilize_zoom).
- Framing control via CROP (max common crop) or CROP_and_PAD (minimal crop plus padding colour).
- Per-frame binary masks for downstream inpainting/compositing.
- Deterministic CPU pipeline (OpenCV + NumPy).

## Flow Node (VideoStabilizerFlow)

- Dense optical flow via OpenCV DIS with adaptive downsampling for speed/robustness.
- Motion models: **translation**, **similarity**, or **perspective** with bundle smoothing.
- Adaptive zoom allowance and framing options (CROP / CROP_and_PAD).
- Deterministic CPU pipeline that outputs binary missing-area masks aligned with the stabilized frames.
