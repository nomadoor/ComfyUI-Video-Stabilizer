# ComfyUI Video Stabilizer

## Features

- Choice of **translation**, **similarity**, or **perspective** motion models.
- Gaussian filtered camera trajectories with adjustable `smoothness` (0.0–1.0).
- Framing control via `CROP` (max common crop) or `CROP_and_PAD` (minimal crop plus colour padding).
- Per-frame missing-area masks for downstream inpainting or compositing.
- Deterministic CPU-friendly implementation (OpenCV + NumPy).