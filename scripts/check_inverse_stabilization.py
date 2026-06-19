#!/usr/bin/env python3
"""Validate inverse stabilization helper behavior without importing ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes.stabilizer_utils import (  # noqa: E402
    _apply_inverse_stabilization,
    _build_stabilization_warp_meta,
    _compute_bounding_boxes,
    _ensure_rgb,
    _normalize_video_input,
    _prepare_expand_transform,
)


def make_base_frame(width: int = 73, height: int = 45) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[..., 0] = xx / max(width - 1, 1)
    frame[..., 1] = yy / max(height - 1, 1)
    frame[..., 2] = (((xx // 6) + (yy // 5)) % 2).astype(np.float32)
    cv2.rectangle(frame, (8, 7), (30, 24), (1.0, 0.2, 0.1), -1)
    cv2.circle(frame, (width - 19, height - 13), 7, (0.1, 0.9, 0.3), -1)
    return frame


def make_source_frames(count: int = 7) -> list[np.ndarray]:
    base = make_base_frame()
    height, width = base.shape[:2]
    center = (width * 0.5, height * 0.5)
    frames: list[np.ndarray] = []
    for idx in range(count):
        matrix_2x3 = cv2.getRotationMatrix2D(center, idx * 0.45, 1.0 + idx * 0.0015)
        matrix_2x3[0, 2] += idx * 0.65
        matrix_2x3[1, 2] += idx * -0.35
        frame = cv2.warpAffine(
            base,
            matrix_2x3,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(_ensure_rgb(frame.astype(np.float32)))
    return frames


def make_expand_stabilization(frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
    height, width = frames[0].shape[:2]
    matrices: list[np.ndarray] = []
    for idx in range(len(frames)):
        matrix = np.array(
            [
                [1.0, 0.0, -idx * 0.7],
                [0.0, 1.0, idx * 0.4],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        matrices.append(matrix)

    mins, maxs = _compute_bounding_boxes(matrices, width, height)
    expand_translate, output_size = _prepare_expand_transform(mins, maxs)
    final_matrices = [expand_translate @ matrix for matrix in matrices]
    stabilized = [
        cv2.warpPerspective(
            frame,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.5, 0.5, 0.5),
        ).astype(np.float32)
        for frame, matrix in zip(frames, final_matrices)
    ]
    meta = {
        "frames": len(frames),
        "framing": {"mode": "expand"},
        "stabilization_warp": _build_stabilization_warp_meta(
            source_size=(width, height),
            output_size=output_size,
            framing_mode="expand",
            applied_matrices=final_matrices,
        ),
    }
    return stabilized, meta


def make_crop_stabilization(frames: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
    height, width = frames[0].shape[:2]
    crop_matrix = np.array(
        [
            [1.12, 0.0, -0.06 * width],
            [0.0, 1.12, -0.06 * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    matrices = [crop_matrix.copy() for _ in frames]
    stabilized = [
        cv2.warpPerspective(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.5, 0.5, 0.5),
        ).astype(np.float32)
        for frame, matrix in zip(frames, matrices)
    ]
    meta = {
        "frames": len(frames),
        "framing": {"mode": "crop"},
        "stabilization_warp": _build_stabilization_warp_meta(
            source_size=(width, height),
            output_size=(width, height),
            framing_mode="crop",
            applied_matrices=matrices,
        ),
    }
    return stabilized, meta


def assert_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise AssertionError(f"{name}: shape {value.shape} != {expected}")


def main() -> int:
    source_frames = make_source_frames()
    stabilized_frames, meta = make_expand_stabilization(source_frames)
    context = _normalize_video_input(stabilized_frames)
    result = _apply_inverse_stabilization(context, meta, (127, 127, 127))

    restored = np.stack(result.frames, axis=0)
    masks = np.stack(result.masks, axis=0)
    source = np.stack(source_frames, axis=0)

    assert_shape("restored", restored, source.shape)
    assert_shape("padding mask", masks, (len(source_frames), source.shape[1], source.shape[2], 1))
    if restored.dtype != np.float32 or masks.dtype != np.float32:
        raise AssertionError(f"unexpected dtypes: restored={restored.dtype}, masks={masks.dtype}")

    warp_meta = result.meta.get("stabilization_warp")
    inverse_meta = result.meta.get("inverse_stabilization")
    if not isinstance(warp_meta, dict) or warp_meta.get("matrix_convention") != "source_to_stabilized":
        raise AssertionError("missing source_to_stabilized stabilization_warp metadata")
    if not isinstance(inverse_meta, dict) or inverse_meta.get("matrix_convention") != "stabilized_to_source":
        raise AssertionError("missing stabilized_to_source inverse metadata")
    if len(warp_meta.get("per_frame", [])) != len(source_frames):
        raise AssertionError("per-frame applied matrix count mismatch")

    max_diff = float(np.max(np.abs(restored - source)))
    mean_diff = float(np.mean(np.abs(restored - source)))
    p99_diff = float(np.quantile(np.abs(restored - source), 0.99))
    if p99_diff > 0.3 or mean_diff > 0.035:
        raise AssertionError(f"inverse reconstruction drift too high: max={max_diff:.6f}, mean={mean_diff:.6f}")

    crop_frames, crop_meta = make_crop_stabilization(source_frames)
    crop_result = _apply_inverse_stabilization(_normalize_video_input(crop_frames), crop_meta, (127, 127, 127))
    crop_masks = np.stack(crop_result.masks, axis=0)
    assert_shape("crop padding mask", crop_masks, (len(source_frames), source.shape[1], source.shape[2], 1))
    if float(np.max(crop_masks)) <= 0.0:
        raise AssertionError("expected crop inverse padding mask to mark unrecoverable pixels")

    print(
        "Inverse stabilization check passed "
        f"(frames={len(source_frames)}, restored_shape={restored.shape}, "
        f"mask_shape={masks.shape}, max_diff={max_diff:.6f}, "
        f"p99_diff={p99_diff:.6f}, mean_diff={mean_diff:.6f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
