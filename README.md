# ComfyUI Video Stabilizer (Classic)

Classic computer-vision based camera stabilisation node for ComfyUI. The node focuses on feature tracking, geometric smoothing, and deterministic image outputs without relying on neural-video models. It pairs well with outpainting/backfill workflows thanks to its missing-area mask output.

## Features

- Choice of **translation**, **similarity**, or **perspective** motion models.
- Gaussian filtered camera trajectories with adjustable `smoothness` (0.0–1.0).
- Framing control via `CROP` (max common crop) or `CROP_and_PAD` (minimal crop plus colour padding).
- Per-frame missing-area masks for downstream inpainting or compositing.
- Deterministic CPU-friendly implementation (OpenCV + NumPy).

## Installation

Clone into your ComfyUI `custom_nodes` folder (or add as a submodule):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-org>/ComfyUI-Video-Stabilizer.git
```

Dependencies rely on the base ComfyUI stack (PyTorch, NumPy). If you run the node outside the full distribution, install the essentials with:

```bash
pip install -r requirements.txt
```

## Node Parameters

| Parameter        | Type    | Default | Description                                                                 |
|------------------|---------|---------|-----------------------------------------------------------------------------|
| `method`         | Combo   | `similarity` | Motion model (`translation`, `similarity`, `perspective`).                 |
| `smoothness`     | Float   | `0.5`   | Trajectory smoothing strength (0.0 = raw motion, 1.0 = heavy smoothing).   |
| `stabilize_zoom` | Float   | `0.5`   | Max auto-zoom allowance. `0.5` allows up to 1.5× crop to hide borders.     |
| `framing`        | Combo   | `CROP_and_PAD` | View handling: `CROP` or `CROP_and_PAD`.                                   |
| `pad_color_rgb`  | String  | `128,128,128` | Padding colour for uncovered regions (comma-separated RGB).               |

### Outputs

1. **stabilized_frames** – Image batch with stabilised content and optional padding.
2. **missing_mask** – Per-frame mask (1.0 = padded area) for compositing/inpainting.

## Usage Notes

- `smoothness` and `stabilize_zoom` interact: higher smoothness may require more zoom to avoid missing areas. `CROP_and_PAD` will fall back to padding if the zoom headroom is exhausted.
- For aggressive stabilization without zoom, keep `stabilize_zoom` at 0 and switch to `CROP_and_PAD`; padded borders will collect any residual motion.
- `pad_color_rgb` accepts tuple-style strings (`"(64, 64, 64)"`) or plain comma separated values (`"64,64,64"`).

## Development

- Python 3.10+ recommended (matching ComfyUI baseline).
- Run a quick syntax sanity check via `python -m compileall nodes`.
- Linting/tests are not bundled yet; contributions welcome.

## License

MIT License. See `LICENSE` if present, or add one before distribution.
