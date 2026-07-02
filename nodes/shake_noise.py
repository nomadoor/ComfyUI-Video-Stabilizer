from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .motion_meta import build_motion_meta_v2


@dataclass(frozen=True)
class ShakePreset:
    tx_amp: float
    ty_amp: float
    rot_amp: float
    zoom_amp: float
    weights: Tuple[float, float, float]
    freqs: Tuple[float, float, float]


PRESETS: Dict[str, ShakePreset] = {
    "handheld_subtle": ShakePreset(0.003, 0.003, 0.10, 0.001, (1.0, 0.5, 0.3), (0.3, 1.1, 3.5)),
    "handheld": ShakePreset(0.006, 0.006, 0.20, 0.002, (1.0, 0.7, 0.4), (0.3, 1.1, 3.5)),
    "handheld_rough": ShakePreset(0.012, 0.010, 0.45, 0.004, (0.9, 1.0, 0.7), (0.4, 1.3, 4.0)),
    "tripod_micro": ShakePreset(0.0008, 0.0008, 0.03, 0.0003, (1.0, 0.3, 0.2), (0.15, 0.9, 3.0)),
    "telephoto": ShakePreset(0.002, 0.002, 0.35, 0.001, (1.0, 0.6, 0.5), (0.25, 1.0, 3.5)),
    "vibration": ShakePreset(0.004, 0.004, 0.08, 0.001, (0.2, 0.5, 1.5), (0.5, 2.5, 7.0)),
}


def _catmull_rom(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: np.ndarray) -> np.ndarray:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _smooth_value_noise(
    rng: np.random.Generator,
    frame_count: int,
    fps: float,
    frequency: float,
) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((0,), dtype=np.float64)
    frequency = float(min(max(frequency, 1e-6), max(fps * 0.5, 1e-6)))
    duration = max((frame_count - 1) / fps, 0.0)
    control_count = max(4, int(math.ceil(duration * frequency)) + 5)
    controls = rng.standard_normal(control_count).astype(np.float64)
    positions = np.arange(frame_count, dtype=np.float64) * frequency / fps
    base = np.floor(positions).astype(np.int64) + 1
    u = positions - np.floor(positions)
    max_base = control_count - 3
    base = np.clip(base, 1, max_base)
    return _catmull_rom(
        controls[base - 1],
        controls[base],
        controls[base + 1],
        controls[base + 2],
        u,
    )


def _channel(
    rng: np.random.Generator,
    frame_count: int,
    fps: float,
    preset: ShakePreset,
    speed: float,
    detail: float,
) -> np.ndarray:
    # RNG consumption order is compatibility-sensitive: channel order, then low/mid/high.
    low = _smooth_value_noise(rng, frame_count, fps, preset.freqs[0] * speed)
    mid = _smooth_value_noise(rng, frame_count, fps, preset.freqs[1] * speed)
    high = _smooth_value_noise(rng, frame_count, fps, preset.freqs[2] * speed)
    values = low * preset.weights[0] + mid * preset.weights[1] + high * preset.weights[2] * detail
    if values.size:
        values = values - float(np.mean(values))
    return values


def _matrix(width: int, height: int, tx: float, ty: float, rotation_deg: float, log_scale: float) -> np.ndarray:
    cx = width * 0.5
    cy = height * 0.5
    angle = math.radians(rotation_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    scale = math.exp(log_scale)
    translate_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    transform = np.array(
        [
            [scale * cos_a, -scale * sin_a, 0.0],
            [scale * sin_a, scale * cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    translate_back = np.array([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    return translate_back @ transform @ translate_to_origin


def generate_shake_motion_meta(
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    preset: str,
    strength: float,
    speed: float,
    detail: float,
    seed: int,
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown shake preset {preset!r}.")
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    fps = float(max(1.0, fps))
    strength = float(np.clip(strength, 0.0, 3.0))
    speed = float(np.clip(speed, 0.1, 3.0))
    detail = float(np.clip(detail, 0.0, 1.0))
    if frame_count < 0 or width <= 0 or height <= 0:
        raise ValueError("frame_count must be non-negative and width/height must be positive.")

    preset_def = PRESETS[preset]
    rng = np.random.default_rng(int(seed))
    tx_noise = _channel(rng, frame_count, fps, preset_def, speed, detail)
    ty_noise = _channel(rng, frame_count, fps, preset_def, speed, detail)
    rot_noise = _channel(rng, frame_count, fps, preset_def, speed, detail)
    scale_noise = _channel(rng, frame_count, fps, preset_def, speed, detail)
    base = float(min(width, height))

    matrices = []
    for idx in range(frame_count):
        matrices.append(
            _matrix(
                width,
                height,
                tx_noise[idx] * preset_def.tx_amp * base * strength,
                ty_noise[idx] * preset_def.ty_amp * base * strength,
                rot_noise[idx] * preset_def.rot_amp * strength,
                scale_noise[idx] * preset_def.zoom_amp * strength,
            )
        )

    return build_motion_meta_v2(
        source="generated_shake",
        frame_count=frame_count,
        fps=fps,
        input_size=(width, height),
        output_size=(width, height),
        matrices=matrices,
        generator={
            "preset": preset,
            "strength": strength,
            "speed": speed,
            "detail": detail,
            "seed": int(seed),
        },
    )
