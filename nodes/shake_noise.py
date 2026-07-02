from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .motion_meta import build_motion_meta_v2


@dataclass(frozen=True)
class ShakeStyle:
    pan_amp: float
    tilt_amp: float
    roll_amp: float
    zoom_amp: float
    drift_freq: float
    drift_enabled: bool
    tremor_ratio: float
    tremor_freq: float
    jitter_rate: float
    step_enabled: bool


STYLES: Dict[str, ShakeStyle] = {
    "tripod": ShakeStyle(0.03, 0.03, 0.02, 0.0002, 0.20, True, 0.15, 4.0, 0.0, False),
    "handheld": ShakeStyle(0.40, 0.33, 0.50, 0.0030, 0.35, True, 0.35, 5.0, 0.0, False),
    "walking": ShakeStyle(0.46, 0.60, 0.70, 0.0040, 0.30, True, 0.30, 5.0, 0.0, True),
    "action": ShakeStyle(0.80, 0.66, 1.00, 0.0060, 0.50, True, 0.80, 6.0, 0.5, False),
    "vibration": ShakeStyle(0.15, 0.15, 0.10, 0.0010, 0.00, False, 1.00, 8.0, 0.0, False),
}


@dataclass(frozen=True)
class ShakeComponents:
    pan_deg: np.ndarray
    tilt_deg: np.ndarray
    roll_deg: np.ndarray
    zoom_log: np.ndarray


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


def _zero_start(values: np.ndarray) -> np.ndarray:
    if values.size:
        return values - float(values[0])
    return values


def _modulated_noise(
    rng: np.random.Generator,
    frame_count: int,
    fps: float,
    frequency: float,
    pace: float,
    randomness: float,
) -> np.ndarray:
    base = _smooth_value_noise(rng, frame_count, fps, frequency * pace)
    if frame_count <= 0 or randomness <= 0.0:
        return base
    modulation = _smooth_value_noise(rng, frame_count, fps, 0.2 * pace)
    modulation = modulation / max(float(np.max(np.abs(modulation))), 1e-6)
    envelope = np.clip(1.0 + modulation * randomness, 0.0, 2.0)
    return base * envelope


def _jitter_events(
    rng: np.random.Generator,
    frame_count: int,
    fps: float,
    rate: float,
    pace: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros = np.zeros((frame_count,), dtype=np.float64)
    if frame_count <= 0 or rate <= 0.0:
        return zeros.copy(), zeros.copy(), zeros.copy()

    duration = frame_count / fps
    event_count = int(rng.poisson(rate * pace * duration))
    pan = zeros.copy()
    tilt = zeros.copy()
    roll = zeros.copy()
    if event_count <= 0:
        return pan, tilt, roll

    times = rng.uniform(0.0, duration, size=event_count)
    amplitudes = rng.standard_normal((event_count, 3)).astype(np.float64)
    t = np.arange(frame_count, dtype=np.float64) / fps
    tau = 0.1
    for event_time, amplitude in zip(times, amplitudes, strict=True):
        dt = t - float(event_time)
        envelope = np.where(dt >= 0.0, np.exp(-dt / tau), 0.0)
        pan += amplitude[0] * envelope
        tilt += amplitude[1] * envelope
        roll += amplitude[2] * envelope
    return pan, tilt, roll


def _walking_step(
    rng: np.random.Generator,
    frame_count: int,
    fps: float,
    pace: float,
    randomness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros = np.zeros((frame_count,), dtype=np.float64)
    if frame_count <= 0:
        return zeros.copy(), zeros.copy(), zeros.copy()

    t = np.arange(frame_count, dtype=np.float64) / fps
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    amp_noise = _smooth_value_noise(rng, frame_count, fps, 0.25 * pace)
    amp_noise = amp_noise / max(float(np.max(np.abs(amp_noise))), 1e-6)
    envelope = np.clip(1.0 + amp_noise * randomness, 0.0, 2.0)
    step_freq = 1.9 * pace
    tilt = np.sin(2.0 * math.pi * step_freq * t + phase) * envelope
    sway = np.sin(2.0 * math.pi * (step_freq * 0.5) * t + phase * 0.73) * envelope
    roll = np.sin(2.0 * math.pi * (step_freq * 0.5) * t + phase * 1.31) * envelope
    return sway, tilt, roll


def generate_shake_components(
    *,
    frame_count: int,
    fps: float,
    style: str,
    amount: float,
    pace: float,
    seed: int,
    pan_amount: float = 1.0,
    tilt_amount: float = 1.0,
    roll_amount: float = 1.0,
    zoom_amount: float = 1.0,
    drift_amount: float = 1.0,
    tremor_amount: float = 1.0,
    jitter_amount: float = 1.0,
    randomness: float = 0.3,
) -> ShakeComponents:
    if style not in STYLES:
        raise ValueError(f"Unknown shake style {style!r}.")
    frame_count = int(frame_count)
    fps = float(max(1.0, fps))
    amount = float(np.clip(amount, 0.0, 3.0))
    pace = float(np.clip(pace, 0.1, 3.0))
    pan_amount = float(np.clip(pan_amount, 0.0, 3.0))
    tilt_amount = float(np.clip(tilt_amount, 0.0, 3.0))
    roll_amount = float(np.clip(roll_amount, 0.0, 3.0))
    zoom_amount = float(np.clip(zoom_amount, 0.0, 3.0))
    drift_amount = float(np.clip(drift_amount, 0.0, 3.0))
    tremor_amount = float(np.clip(tremor_amount, 0.0, 3.0))
    jitter_amount = float(np.clip(jitter_amount, 0.0, 3.0))
    randomness = float(np.clip(randomness, 0.0, 1.0))
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative.")

    style_def = STYLES[style]
    rng = np.random.default_rng(int(seed))
    zeros = np.zeros((frame_count,), dtype=np.float64)

    # RNG consumption order is compatibility-sensitive:
    # drift pan/tilt/roll/zoom, tremor pan/tilt/roll/zoom, jitter, walking step.
    if style_def.drift_enabled and drift_amount > 0.0:
        drift_pan = _modulated_noise(rng, frame_count, fps, style_def.drift_freq, pace, randomness)
        drift_tilt = _modulated_noise(rng, frame_count, fps, style_def.drift_freq, pace, randomness)
        drift_roll = _modulated_noise(rng, frame_count, fps, style_def.drift_freq, pace, randomness)
        drift_zoom = _modulated_noise(rng, frame_count, fps, style_def.drift_freq, pace, randomness)
    else:
        drift_pan = drift_tilt = drift_roll = drift_zoom = zeros

    tremor_pan = _modulated_noise(rng, frame_count, fps, style_def.tremor_freq, pace, randomness)
    tremor_tilt = _modulated_noise(rng, frame_count, fps, style_def.tremor_freq, pace, randomness)
    tremor_roll = _modulated_noise(rng, frame_count, fps, style_def.tremor_freq, pace, randomness)
    tremor_zoom = _modulated_noise(rng, frame_count, fps, style_def.tremor_freq, pace, randomness)

    jitter_pan, jitter_tilt, jitter_roll = _jitter_events(rng, frame_count, fps, style_def.jitter_rate, pace)
    if style_def.step_enabled:
        step_pan, step_tilt, step_roll = _walking_step(rng, frame_count, fps, pace, randomness)
    else:
        step_pan = step_tilt = step_roll = zeros

    pan = (
        drift_pan * style_def.pan_amp * drift_amount
        + tremor_pan * style_def.pan_amp * style_def.tremor_ratio * tremor_amount
        + jitter_pan * style_def.pan_amp * jitter_amount
        + step_pan * style_def.pan_amp * 0.5
    )
    tilt = (
        drift_tilt * style_def.tilt_amp * drift_amount
        + tremor_tilt * style_def.tilt_amp * style_def.tremor_ratio * tremor_amount
        + jitter_tilt * style_def.tilt_amp * jitter_amount
        + step_tilt * style_def.tilt_amp
    )
    roll = (
        drift_roll * style_def.roll_amp * drift_amount
        + tremor_roll * style_def.roll_amp * style_def.tremor_ratio * tremor_amount
        + jitter_roll * style_def.roll_amp * jitter_amount
        + step_roll * style_def.roll_amp * 0.5
    )
    zoom = (
        drift_zoom * style_def.zoom_amp * drift_amount
        + tremor_zoom * style_def.zoom_amp * style_def.tremor_ratio * tremor_amount
    )

    return ShakeComponents(
        pan_deg=_zero_start(pan * pan_amount * amount),
        tilt_deg=_zero_start(tilt * tilt_amount * amount),
        roll_deg=_zero_start(roll * roll_amount * amount),
        zoom_log=_zero_start(zoom * zoom_amount * amount),
    )


def _matrix(
    width: int,
    height: int,
    pan_deg: float,
    tilt_deg: float,
    roll_deg: float,
    zoom_log: float,
    virtual_fov: float,
) -> np.ndarray:
    cx = width * 0.5
    cy = height * 0.5
    fov_rad = math.radians(float(np.clip(virtual_fov, 10.0, 120.0)))
    focal_px = 0.5 * min(width, height) / math.tan(fov_rad * 0.5)
    tx = focal_px * math.tan(math.radians(pan_deg))
    ty = focal_px * math.tan(math.radians(tilt_deg))
    angle = math.radians(roll_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    scale = math.exp(zoom_log)
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
    style: str,
    amount: float,
    pace: float,
    seed: int,
    pan_amount: float = 1.0,
    tilt_amount: float = 1.0,
    roll_amount: float = 1.0,
    zoom_amount: float = 1.0,
    drift_amount: float = 1.0,
    tremor_amount: float = 1.0,
    jitter_amount: float = 1.0,
    randomness: float = 0.3,
    virtual_fov: float = 60.0,
) -> dict:
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    fps = float(max(1.0, fps))
    virtual_fov = float(np.clip(virtual_fov, 10.0, 120.0))
    if frame_count < 0 or width <= 0 or height <= 0:
        raise ValueError("frame_count must be non-negative and width/height must be positive.")

    components = generate_shake_components(
        frame_count=frame_count,
        fps=fps,
        style=style,
        amount=amount,
        pace=pace,
        seed=seed,
        pan_amount=pan_amount,
        tilt_amount=tilt_amount,
        roll_amount=roll_amount,
        zoom_amount=zoom_amount,
        drift_amount=drift_amount,
        tremor_amount=tremor_amount,
        jitter_amount=jitter_amount,
        randomness=randomness,
    )

    matrices = [
        _matrix(
            width,
            height,
            components.pan_deg[idx],
            components.tilt_deg[idx],
            components.roll_deg[idx],
            components.zoom_log[idx],
            virtual_fov,
        )
        for idx in range(frame_count)
    ]

    return build_motion_meta_v2(
        source="generated_shake",
        frame_count=frame_count,
        fps=fps,
        input_size=(width, height),
        output_size=(width, height),
        matrices=matrices,
        generator={
            "style": style,
            "amount": float(np.clip(amount, 0.0, 3.0)),
            "pace": float(np.clip(pace, 0.1, 3.0)),
            "seed": int(seed),
            "pan_amount": float(np.clip(pan_amount, 0.0, 3.0)),
            "tilt_amount": float(np.clip(tilt_amount, 0.0, 3.0)),
            "roll_amount": float(np.clip(roll_amount, 0.0, 3.0)),
            "zoom_amount": float(np.clip(zoom_amount, 0.0, 3.0)),
            "drift_amount": float(np.clip(drift_amount, 0.0, 3.0)),
            "tremor_amount": float(np.clip(tremor_amount, 0.0, 3.0)),
            "jitter_amount": float(np.clip(jitter_amount, 0.0, 3.0)),
            "randomness": float(np.clip(randomness, 0.0, 1.0)),
            "virtual_fov": virtual_fov,
        },
    )
