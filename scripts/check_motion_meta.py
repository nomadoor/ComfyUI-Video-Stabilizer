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
    build_motion_meta_v2,
    motion_meta_from_stabilization_warp,
    resolve_motion_meta,
)
from nodes.shake_noise import generate_shake_motion_meta  # noqa: E402
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
        generator={"preset": "handheld_subtle", "strength": 1.0, "speed": 1.0, "detail": 0.35, "seed": 0},
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

    try:
        resolve_motion_meta({})
    except ValueError as exc:
        if "motion_meta or stabilization_warp" not in str(exc):
            raise
    else:
        raise AssertionError("missing meta blocks should fail")

    shake_a = generate_shake_motion_meta(
        frame_count=5,
        width=64,
        height=48,
        fps=16.0,
        preset="handheld_subtle",
        strength=1.0,
        speed=1.0,
        detail=0.35,
        seed=123,
    )
    shake_b = generate_shake_motion_meta(
        frame_count=5,
        width=64,
        height=48,
        fps=16.0,
        preset="handheld_subtle",
        strength=1.0,
        speed=1.0,
        detail=0.35,
        seed=123,
    )
    shake_c = generate_shake_motion_meta(
        frame_count=5,
        width=64,
        height=48,
        fps=16.0,
        preset="handheld_subtle",
        strength=1.0,
        speed=1.0,
        detail=0.35,
        seed=124,
    )
    if shake_a != shake_b:
        raise AssertionError("shake generation is not deterministic for identical inputs")
    if shake_a["per_frame"] == shake_c["per_frame"]:
        raise AssertionError("shake generation did not vary across seeds")
    if len(shake_a["per_frame"]) != 5 or shake_a["frame_count"] != 5:
        raise AssertionError("shake frame_count/per_frame length mismatch")

    generate_shake_motion_meta(
        frame_count=4,
        width=64,
        height=48,
        fps=8.0,
        preset="vibration",
        strength=1.0,
        speed=3.0,
        detail=1.0,
        seed=0,
    )

    frames = _frames(3)
    identity_meta = {
        "motion_meta": build_motion_meta_v2(
            source="generated_shake",
            frame_count=3,
            fps=16.0,
            input_size=(32, 24),
            output_size=(32, 24),
            matrices=[np.eye(3, dtype=np.float64) for _ in frames],
            generator={"preset": "handheld_subtle", "strength": 0.0, "speed": 1.0, "detail": 0.35, "seed": 0},
        )
    }
    identity_result = apply_motion(_normalize_video_input(frames), identity_meta, (127, 127, 127))
    if not np.allclose(identity_result.frames, np.stack(frames, axis=0), atol=1e-6):
        raise AssertionError("identity motion apply changed frames")
    if float(np.max(identity_result.masks)) != 0.0:
        raise AssertionError("identity motion apply produced padding")

    shift = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    fallback_meta = {
        "motion_meta": build_motion_meta_v2(
            source="generated_shake",
            frame_count=3,
            fps=16.0,
            input_size=(32, 24),
            output_size=(32, 24),
            matrices=[shift for _ in frames],
            generator={"preset": "handheld_subtle", "strength": 3.0, "speed": 1.0, "detail": 0.35, "seed": 0},
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
    )
    for relative_path in ("nodes/shake_noise.py", "nodes/video_stabilizer_shake_generator.py"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                raise AssertionError(f"forbidden hidden randomness pattern {pattern!r} in {relative_path}")

    print("Motion meta check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
