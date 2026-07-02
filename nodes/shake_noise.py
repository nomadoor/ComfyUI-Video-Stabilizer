from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .motion_meta import build_motion_meta_v2


@dataclass(frozen=True)
class ShakeRecipe:
    pan: float
    tilt: float
    roll: float
    zoom: float
    drift_freq: float
    tremor: float
    tremor_freq: float
    jitter_rate: float
    step: float
    randomness: float
    virtual_fov: float


STYLES: Dict[str, ShakeRecipe] = {
    "tripod": ShakeRecipe(0.03, 0.03, 0.02, 0.0002, 0.20, 0.15, 4.0, 0.0, 0.0, 0.3, 60.0),
    "handheld": ShakeRecipe(0.40, 0.33, 0.50, 0.0030, 0.35, 0.35, 5.0, 0.0, 0.0, 0.3, 60.0),
    "walking": ShakeRecipe(0.46, 0.60, 0.70, 0.0040, 0.30, 0.30, 5.0, 0.0, 0.60, 0.3, 60.0),
    "action": ShakeRecipe(0.80, 0.66, 1.00, 0.0060, 0.50, 0.80, 6.0, 0.5, 0.0, 0.3, 60.0),
    "vibration": ShakeRecipe(0.15, 0.15, 0.10, 0.0010, 0.00, 1.00, 8.0, 0.0, 0.0, 0.3, 60.0),
}


@dataclass(frozen=True)
class ShakeComponents:
    pan_deg: np.ndarray
    tilt_deg: np.ndarray
    roll_deg: np.ndarray
    zoom_log: np.ndarray


def recipe_to_dict(recipe: ShakeRecipe) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(recipe).items()}


def clamp_recipe(recipe: ShakeRecipe) -> ShakeRecipe:
    return ShakeRecipe(
        pan=float(np.clip(recipe.pan, 0.0, 5.0)),
        tilt=float(np.clip(recipe.tilt, 0.0, 5.0)),
        roll=float(np.clip(recipe.roll, 0.0, 5.0)),
        zoom=float(np.clip(recipe.zoom, 0.0, 0.05)),
        drift_freq=float(np.clip(recipe.drift_freq, 0.0, 2.0)),
        tremor=float(np.clip(recipe.tremor, 0.0, 2.0)),
        tremor_freq=float(np.clip(recipe.tremor_freq, 1.0, 15.0)),
        jitter_rate=float(np.clip(recipe.jitter_rate, 0.0, 3.0)),
        step=float(np.clip(recipe.step, 0.0, 2.0)),
        randomness=float(np.clip(recipe.randomness, 0.0, 1.0)),
        virtual_fov=float(np.clip(recipe.virtual_fov, 10.0, 120.0)),
    )


def recipe_from_mapping(value: dict[str, object]) -> ShakeRecipe:
    return clamp_recipe(
        ShakeRecipe(
            pan=float(value["pan"]),
            tilt=float(value["tilt"]),
            roll=float(value["roll"]),
            zoom=float(value["zoom"]),
            drift_freq=float(value["drift_freq"]),
            tremor=float(value["tremor"]),
            tremor_freq=float(value["tremor_freq"]),
            jitter_rate=float(value["jitter_rate"]),
            step=float(value["step"]),
            randomness=float(value["randomness"]),
            virtual_fov=float(value["virtual_fov"]),
        )
    )


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
    recipe: ShakeRecipe,
    frame_count: int,
    fps: float,
    amount: float,
    pace: float,
    seed: int,
) -> ShakeComponents:
    recipe = clamp_recipe(recipe)
    frame_count = int(frame_count)
    fps = float(max(1.0, fps))
    amount = float(np.clip(amount, 0.0, 3.0))
    pace = float(np.clip(pace, 0.1, 3.0))
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative.")

    rng = np.random.default_rng(int(seed))
    zeros = np.zeros((frame_count,), dtype=np.float64)

    # RNG consumption order is compatibility-sensitive:
    # drift pan/tilt/roll/zoom, tremor pan/tilt/roll/zoom, jitter, walking step.
    if recipe.drift_freq > 0.0:
        drift_pan = _modulated_noise(rng, frame_count, fps, recipe.drift_freq, pace, recipe.randomness)
        drift_tilt = _modulated_noise(rng, frame_count, fps, recipe.drift_freq, pace, recipe.randomness)
        drift_roll = _modulated_noise(rng, frame_count, fps, recipe.drift_freq, pace, recipe.randomness)
        drift_zoom = _modulated_noise(rng, frame_count, fps, recipe.drift_freq, pace, recipe.randomness)
    else:
        drift_pan = drift_tilt = drift_roll = drift_zoom = zeros

    tremor_pan = _modulated_noise(rng, frame_count, fps, recipe.tremor_freq, pace, recipe.randomness)
    tremor_tilt = _modulated_noise(rng, frame_count, fps, recipe.tremor_freq, pace, recipe.randomness)
    tremor_roll = _modulated_noise(rng, frame_count, fps, recipe.tremor_freq, pace, recipe.randomness)
    tremor_zoom = _modulated_noise(rng, frame_count, fps, recipe.tremor_freq, pace, recipe.randomness)

    jitter_pan, jitter_tilt, jitter_roll = _jitter_events(rng, frame_count, fps, recipe.jitter_rate, pace)
    if recipe.step > 0.0:
        step_pan, step_tilt, step_roll = _walking_step(rng, frame_count, fps, pace, recipe.randomness)
    else:
        step_pan = step_tilt = step_roll = zeros

    pan = drift_pan * recipe.pan + tremor_pan * recipe.pan * recipe.tremor + jitter_pan * recipe.pan + step_pan * recipe.step * 0.5
    tilt = drift_tilt * recipe.tilt + tremor_tilt * recipe.tilt * recipe.tremor + jitter_tilt * recipe.tilt + step_tilt * recipe.step
    roll = drift_roll * recipe.roll + tremor_roll * recipe.roll * recipe.tremor + jitter_roll * recipe.roll + step_roll * recipe.step * 0.5
    zoom = drift_zoom * recipe.zoom + tremor_zoom * recipe.zoom * recipe.tremor

    return ShakeComponents(
        pan_deg=_zero_start(pan * amount),
        tilt_deg=_zero_start(tilt * amount),
        roll_deg=_zero_start(roll * amount),
        zoom_log=_zero_start(zoom * amount),
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
    recipe: ShakeRecipe,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    amount: float,
    pace: float,
    seed: int,
    node: str = "shake_generator",
    style: str = "manual",
) -> dict:
    recipe = clamp_recipe(recipe)
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    fps = float(max(1.0, fps))
    if frame_count < 0 or width <= 0 or height <= 0:
        raise ValueError("frame_count must be non-negative and width/height must be positive.")

    amount = float(np.clip(amount, 0.0, 3.0))
    pace = float(np.clip(pace, 0.1, 3.0))
    components = generate_shake_components(
        recipe=recipe,
        frame_count=frame_count,
        fps=fps,
        amount=amount,
        pace=pace,
        seed=seed,
    )

    matrices = [
        _matrix(
            width,
            height,
            components.pan_deg[idx],
            components.tilt_deg[idx],
            components.roll_deg[idx],
            components.zoom_log[idx],
            recipe.virtual_fov,
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
            "node": node,
            "style": style,
            "amount": amount,
            "pace": pace,
            "seed": int(seed),
            "recipe": recipe_to_dict(recipe),
        },
    )
