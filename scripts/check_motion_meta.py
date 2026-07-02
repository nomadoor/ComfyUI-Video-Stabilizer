#!/usr/bin/env python3
"""Validate motion_meta helpers without importing ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes.motion_apply import apply_motion  # noqa: E402
from nodes.motion_meta import (  # noqa: E402
    applied_motion_meta_from_stabilization_warp,
    build_motion_meta_v2,
    motion_meta_from_stabilization_warp,
    resolve_motion_meta,
)
from nodes.shake_noise import (  # noqa: E402
    STYLES,
    ShakeRecipe,
    generate_shake_components,
    generate_shake_motion_meta,
    recipe_from_mapping,
)
from nodes.stabilizer_utils import _build_stabilization_warp_meta, _normalize_video_input  # noqa: E402


def _frames(count: int, width: int = 32, height: int = 24) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for idx in range(count):
        frame = np.zeros((height, width, 3), dtype=np.float32)
        frame[..., 0] = (xx + idx) / max(width + count - 1, 1)
        frame[..., 1] = yy / max(height - 1, 1)
        frame[..., 2] = ((xx + yy + idx) % 7) / 6.0
        frames.append(frame)
    return frames


def _generator_block() -> dict[str, object]:
    return {
        "node": "shake_generator",
        "style": "handheld",
        "amount": 1.0,
        "speed": 1.0,
        "seed": 0,
        "recipe": {
            "pan": 0.40,
            "tilt": 0.33,
            "roll": 0.50,
            "zoom": 0.003,
            "drift_freq": 0.35,
            "tremor": 0.35,
            "tremor_freq": 5.0,
            "jitter_rate": 0.0,
            "step": 0.0,
            "randomness": 0.3,
            "virtual_fov": 60.0,
        },
    }


def _translation_energy(block: dict) -> float:
    matrices = [np.asarray(entry["matrix"], dtype=np.float64) for entry in block["per_frame"]]
    translations = np.array([[matrix[0, 2], matrix[1, 2]] for matrix in matrices], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(translations * translations, axis=1))))


def _high_frequency_energy(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    second_diff = np.diff(values, n=2)
    return float(np.sqrt(np.mean(second_diff * second_diff)))


def _dominant_frequency(values: np.ndarray, fps: float) -> float:
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fps)
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    return float(freqs[int(np.argmax(spectrum))])


def _rotation_degrees(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def _shake(
    style: str = "handheld",
    *,
    frame_count: int = 5,
    width: int = 64,
    height: int = 48,
    fps: float = 16.0,
    amount: float = 1.0,
    speed: float = 1.0,
    seed: int = 123,
) -> dict:
    return generate_shake_motion_meta(
        recipe=STYLES[style],
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        amount=amount,
        speed=speed,
        seed=seed,
        node="shake_generator",
        style=style,
    )


def _manual_shake(
    recipe: ShakeRecipe,
    *,
    frame_count: int = 5,
    width: int = 64,
    height: int = 48,
    fps: float = 16.0,
    amount: float = 1.0,
    speed: float = 1.0,
    seed: int = 123,
) -> dict:
    return generate_shake_motion_meta(
        recipe=recipe,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        amount=amount,
        speed=speed,
        seed=seed,
        node="shake_generator_manual",
        style="manual",
    )


def main() -> int:
    matrices = [
        np.eye(3, dtype=np.float64),
        np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.25], [0.0, 0.0, 1.0]], dtype=np.float64),
    ]
    block = build_motion_meta_v2(
        source="generated_shake",
        frame_count=2,
        fps=16.0,
        input_size=(64, 48),
        output_size=(64, 48),
        matrices=matrices,
        generator=_generator_block(),
    )
    resolved = resolve_motion_meta({"motion_meta": block})
    if resolved.frame_count != 2 or resolved.input_size != (64, 48) or resolved.output_size != (64, 48):
        raise AssertionError("motion_meta block did not resolve with expected dimensions")

    warp = _build_stabilization_warp_meta(
        source_size=(80, 50),
        output_size=(96, 60),
        framing_mode="expand",
        applied_matrices=matrices,
    )
    legacy = motion_meta_from_stabilization_warp(warp, fps=24.0, source="legacy_stabilization")
    if legacy is None:
        raise AssertionError("invertible stabilization_warp unexpectedly failed")
    resolved_legacy = resolve_motion_meta({"stabilization_warp": warp})
    if resolved_legacy.input_size != (96, 60) or resolved_legacy.output_size != (80, 50):
        raise AssertionError("legacy stabilization_warp did not resolve to inverse input/output sizes")
    if not np.allclose(resolved_legacy.per_frame[1].matrix, np.linalg.inv(matrices[1])):
        raise AssertionError("legacy stabilization_warp matrix was not inverted")
    applied = applied_motion_meta_from_stabilization_warp(warp, fps=24.0, source="estimated_flow")
    resolved_applied = resolve_motion_meta({"motion_meta": applied})
    if resolved_applied.input_size != (80, 50) or resolved_applied.output_size != (96, 60):
        raise AssertionError("applied stabilization motion_meta resolved with wrong dimensions")
    if not np.allclose(resolved_applied.per_frame[1].matrix, matrices[1]):
        raise AssertionError("applied stabilization motion_meta should preserve applied matrices")
    frames_for_apply = _frames(2, width=80, height=50)
    combined_meta = {"stabilization_warp": warp, "motion_meta": applied}
    direct_apply = apply_motion(_normalize_video_input(frames_for_apply), combined_meta, (127, 127, 127))
    if direct_apply.frames.shape[1:3] != (60, 96):
        raise AssertionError(f"direct applied motion_meta output shape mismatch: {direct_apply.frames.shape}")
    inverse_apply = apply_motion(_normalize_video_input(direct_apply.frames), combined_meta, (127, 127, 127))
    if inverse_apply.frames.shape[1:3] != (50, 80):
        raise AssertionError(f"context-selected inverse motion output shape mismatch: {inverse_apply.frames.shape}")

    try:
        resolve_motion_meta({})
    except ValueError as exc:
        if "motion_meta or stabilization_warp" not in str(exc):
            raise
    else:
        raise AssertionError("missing meta blocks should fail")

    shake_a = _shake(seed=123)
    shake_b = _shake(seed=123)
    shake_c = _shake(seed=124)
    if shake_a != shake_b:
        raise AssertionError("shake generation is not deterministic for identical inputs")
    if shake_a["per_frame"] == shake_c["per_frame"]:
        raise AssertionError("shake generation did not vary across seeds")
    if len(shake_a["per_frame"]) != 5 or shake_a["frame_count"] != 5:
        raise AssertionError("shake frame_count/per_frame length mismatch")
    if not np.allclose(np.asarray(shake_a["per_frame"][0]["matrix"], dtype=np.float64), np.eye(3), atol=1e-9):
        raise AssertionError("shake first frame is not identity")

    _shake("vibration", frame_count=4, fps=8.0, speed=3.0, seed=0)

    manual_default = _manual_shake(STYLES["handheld"], seed=33)
    simple_default = _shake("handheld", seed=33)
    if manual_default["per_frame"] != simple_default["per_frame"]:
        raise AssertionError("manual handheld defaults must match simple handheld matrices")
    if manual_default["generator"]["recipe"] != simple_default["generator"]["recipe"]:
        raise AssertionError("manual/default recipe metadata should match simple handheld recipe")

    for style in STYLES:
        simple = _shake(style, frame_count=32, width=96, height=54, seed=17)
        first_matrix = np.asarray(simple["per_frame"][0]["matrix"], dtype=np.float64)
        if not np.allclose(first_matrix, np.eye(3), atol=1e-9):
            raise AssertionError(f"{style} first frame is not identity")
        round_trip_recipe = recipe_from_mapping(simple["generator"]["recipe"])
        manual = _manual_shake(round_trip_recipe, frame_count=32, width=96, height=54, seed=17)
        if simple["per_frame"] != manual["per_frame"]:
            raise AssertionError(f"{style} recipe did not round-trip through manual generation")

    handheld_components = generate_shake_components(
        recipe=STYLES["handheld"],
        frame_count=128,
        fps=16.0,
        amount=1.0,
        speed=1.0,
        seed=77,
    )
    vibration_components = generate_shake_components(
        recipe=STYLES["vibration"],
        frame_count=128,
        fps=16.0,
        amount=1.0,
        speed=1.0,
        seed=77,
    )
    walking_components = generate_shake_components(
        recipe=STYLES["walking"],
        frame_count=128,
        fps=16.0,
        amount=1.0,
        speed=1.0,
        seed=77,
    )
    tripod_block = _shake("tripod", frame_count=128, width=128, height=72, seed=77)
    handheld_block = _shake("handheld", frame_count=128, width=128, height=72, seed=77)
    if _high_frequency_energy(vibration_components.pan_deg) <= _high_frequency_energy(handheld_components.pan_deg):
        raise AssertionError("vibration should have more high-frequency pan energy than handheld")
    walking_peak = _dominant_frequency(walking_components.tilt_deg, fps=16.0)
    if abs(walking_peak - 1.9) > 0.35:
        raise AssertionError(f"walking tilt peak should be near step frequency, got {walking_peak:.3f} Hz")
    if _translation_energy(tripod_block) * 10.0 >= _translation_energy(handheld_block):
        raise AssertionError("tripod total translation amplitude should be at least one order below handheld")

    no_roll = _manual_shake(
        ShakeRecipe(
            pan=STYLES["action"].pan,
            tilt=STYLES["action"].tilt,
            roll=0.0,
            zoom=STYLES["action"].zoom,
            drift_freq=STYLES["action"].drift_freq,
            tremor=STYLES["action"].tremor,
            tremor_freq=STYLES["action"].tremor_freq,
            jitter_rate=STYLES["action"].jitter_rate,
            step=STYLES["action"].step,
            randomness=STYLES["action"].randomness,
            virtual_fov=STYLES["action"].virtual_fov,
        ),
        frame_count=32,
        width=96,
        height=54,
        seed=5,
    )
    rotations = [
        abs(_rotation_degrees(np.asarray(entry["matrix"], dtype=np.float64)))
        for entry in no_roll["per_frame"]
    ]
    if max(rotations) > 1e-9:
        raise AssertionError("manual roll=0 should zero generated roll")

    frames = _frames(3)
    identity_meta = {
        "motion_meta": build_motion_meta_v2(
            source="generated_shake",
            frame_count=3,
            fps=16.0,
            input_size=(32, 24),
            output_size=(32, 24),
            matrices=[np.eye(3, dtype=np.float64) for _ in frames],
            generator=_generator_block(),
        )
    }
    baseline_result = apply_motion(_normalize_video_input(frames), identity_meta, (127, 127, 127))
    identity_result = apply_motion(_normalize_video_input(frames), identity_meta, (127, 127, 127), motion_blur=0.0)
    if not np.allclose(identity_result.frames, np.stack(frames, axis=0), atol=1e-6):
        raise AssertionError("identity motion apply changed frames")
    if float(np.max(identity_result.masks)) != 0.0:
        raise AssertionError("identity motion apply produced padding")
    if not np.array_equal(identity_result.frames, baseline_result.frames) or not np.array_equal(
        identity_result.masks,
        baseline_result.masks,
    ):
        raise AssertionError("motion_blur=0 should use the exact baseline path")

    blur_matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    blur_meta = {
        "motion_meta": build_motion_meta_v2(
            source="generated_shake",
            frame_count=3,
            fps=16.0,
            input_size=(32, 24),
            output_size=(32, 24),
            matrices=[np.eye(3, dtype=np.float64), blur_matrix, blur_matrix @ blur_matrix],
            generator=_generator_block(),
        )
    }
    blur_a = apply_motion(
        _normalize_video_input(frames),
        blur_meta,
        (127, 127, 127),
        motion_blur=0.5,
        motion_blur_samples=7,
    )
    blur_b = apply_motion(
        _normalize_video_input(frames),
        blur_meta,
        (127, 127, 127),
        motion_blur=0.5,
        motion_blur_samples=7,
    )
    if not np.array_equal(blur_a.frames, blur_b.frames) or not np.array_equal(blur_a.masks, blur_b.masks):
        raise AssertionError("motion blur apply should be deterministic")
    if blur_a.meta.get("motion_apply", {}).get("motion_blur") != 0.5:
        raise AssertionError("motion blur metadata missing")
    progress_ticks = 0

    def tick() -> None:
        nonlocal progress_ticks
        progress_ticks += 1

    apply_motion(
        _normalize_video_input(frames),
        blur_meta,
        (127, 127, 127),
        motion_blur=0.5,
        motion_blur_samples=7,
        progress_callback=tick,
    )
    if progress_ticks != len(frames) * 7:
        raise AssertionError(f"motion blur progress tick count mismatch: {progress_ticks}")

    progress_ticks = 0
    apply_motion(
        _normalize_video_input(frames),
        blur_meta,
        (127, 127, 127),
        framing_mode="crop",
        motion_blur=0.5,
        motion_blur_samples=7,
        progress_callback=tick,
    )
    if progress_ticks != len(frames) + len(frames) * 7:
        raise AssertionError(f"crop motion blur progress tick count mismatch: {progress_ticks}")

    shift = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    fallback_meta = {
        "motion_meta": build_motion_meta_v2(
            source="generated_shake",
            frame_count=3,
            fps=16.0,
            input_size=(32, 24),
            output_size=(32, 24),
            matrices=[shift for _ in frames],
            generator=_generator_block(),
        )
    }
    crop_result = apply_motion(
        _normalize_video_input(frames),
        fallback_meta,
        (127, 127, 127),
        framing_mode="crop",
    )
    if crop_result.meta.get("framing_fallback") != "pad":
        raise AssertionError("extreme crop did not fall back to pad")

    forbidden_patterns = (
        "np.random.seed",
        "np.random.rand",
        "np.random.randn",
        "np.random.random",
        "import random",
        "from random",
        "random.seed",
        "random.random",
        "import time",
        "from time",
        "time.time",
        "advanced=True",
    )
    for relative_path in (
        "nodes/shake_noise.py",
        "nodes/video_stabilizer_shake_generator.py",
        "nodes/video_stabilizer_shake_generator_manual.py",
        "nodes/video_stabilizer_motion_apply.py",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                raise AssertionError(f"forbidden pattern {pattern!r} in {relative_path}")

    print("Motion meta check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
