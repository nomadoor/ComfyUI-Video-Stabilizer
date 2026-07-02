from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import cv2
import numpy as np

from .motion_meta import MotionMeta, resolve_motion_meta
from .stabilizer_utils import VideoContext, _ensure_rgb

ApplyFramingMode = Literal["pad", "crop"]
ApplyInterpolation = Literal["bilinear", "bicubic"]


@dataclass
class MotionApplyResult:
    frames: np.ndarray
    masks: np.ndarray
    meta: Dict[str, Any]


def _interpolation_flag(interpolation: ApplyInterpolation) -> int:
    if interpolation == "bilinear":
        return cv2.INTER_LINEAR
    if interpolation == "bicubic":
        return cv2.INTER_CUBIC
    raise ValueError(f"Unsupported interpolation {interpolation!r}; expected 'bilinear' or 'bicubic'.")


def _validate_context(context: VideoContext, motion: MotionMeta) -> None:
    if (context.width, context.height) != motion.input_size:
        raise ValueError(
            "Input frames must match motion_meta.input_size "
            f"{motion.input_size}, got {(context.width, context.height)}."
        )
    if len(context.frames) != motion.frame_count:
        raise ValueError(
            "Frame count mismatch: "
            f"got {len(context.frames)} frame(s), metadata has {motion.frame_count} matrix entry/entries."
        )


def _frame_border_value(context: VideoContext, padding_rgb: Tuple[int, int, int]) -> Any:
    padding_array = np.array(padding_rgb, dtype=np.float32) / 255.0
    return float(np.mean(padding_array)) if context.channels == 1 else padding_array.tolist()


def _warp_with_matrices(
    context: VideoContext,
    matrices: list[np.ndarray],
    output_size: Tuple[int, int],
    interpolation_flag: int,
    padding_rgb: Tuple[int, int, int],
    *,
    masks_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = output_size
    frame_count = len(matrices)
    frames_out = np.empty((frame_count, out_h, out_w, 3), dtype=np.float32)
    masks_out = np.zeros((frame_count, out_h, out_w, 1), dtype=np.float32)
    border_value = _frame_border_value(context, padding_rgb)
    ones = np.ones((context.height, context.width), dtype=np.float32)

    for idx, matrix in enumerate(matrices):
        matrix32 = np.asarray(matrix, dtype=np.float32)
        warped = cv2.warpPerspective(
            context.frames[idx],
            matrix32,
            output_size,
            flags=interpolation_flag,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        frames_out[idx] = _ensure_rgb(warped.astype(np.float32))
        if masks_zero:
            continue

        content = cv2.warpPerspective(
            ones,
            matrix32,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask = 1.0 - (content > 0.5).astype(np.float32)
        mask[mask < 1e-3] = 0.0
        masks_out[idx, ..., 0] = mask

    return frames_out, masks_out


def _blurred_matrix_samples(matrices: list[np.ndarray], idx: int, motion_blur: float, sample_count: int) -> list[np.ndarray]:
    if len(matrices) <= 1:
        return [matrices[idx]]
    base = np.asarray(matrices[idx], dtype=np.float64)
    if idx < len(matrices) - 1:
        delta = np.asarray(matrices[idx + 1], dtype=np.float64) - base
    else:
        delta = base - np.asarray(matrices[idx - 1], dtype=np.float64)
    ts = np.linspace(0.0, float(motion_blur), int(sample_count), dtype=np.float64)
    return [base + delta * t for t in ts]


def _warp_with_motion_blur(
    context: VideoContext,
    matrices: list[np.ndarray],
    output_size: Tuple[int, int],
    interpolation_flag: int,
    padding_rgb: Tuple[int, int, int],
    motion_blur: float,
    motion_blur_samples: int,
    *,
    masks_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if motion_blur <= 0.0 or motion_blur_samples <= 1:
        return _warp_with_matrices(
            context,
            matrices,
            output_size,
            interpolation_flag,
            padding_rgb,
            masks_zero=masks_zero,
        )

    out_w, out_h = output_size
    frame_count = len(matrices)
    frames_out = np.empty((frame_count, out_h, out_w, 3), dtype=np.float32)
    masks_out = np.zeros((frame_count, out_h, out_w, 1), dtype=np.float32)
    border_value = _frame_border_value(context, padding_rgb)
    ones = np.ones((context.height, context.width), dtype=np.float32)
    sample_count = int(np.clip(motion_blur_samples, 3, 33))

    for idx in range(frame_count):
        frame_accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        coverage_accum = np.zeros((out_h, out_w), dtype=np.float32)
        for matrix in _blurred_matrix_samples(matrices, idx, motion_blur, sample_count):
            matrix32 = np.asarray(matrix, dtype=np.float32)
            warped = cv2.warpPerspective(
                context.frames[idx],
                matrix32,
                output_size,
                flags=interpolation_flag,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            frame_accum += _ensure_rgb(warped.astype(np.float32))
            if not masks_zero:
                coverage = cv2.warpPerspective(
                    ones,
                    matrix32,
                    output_size,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                coverage_accum += (coverage > 0.5).astype(np.float32)

        frames_out[idx] = frame_accum / float(sample_count)
        if not masks_zero:
            mean_coverage = coverage_accum / float(sample_count)
            mask = 1.0 - mean_coverage
            mask[mask < 1e-3] = 0.0
            masks_out[idx, ..., 0] = mask

    return frames_out, masks_out


def _common_valid_mask(
    input_size: Tuple[int, int],
    output_size: Tuple[int, int],
    matrices: list[np.ndarray],
) -> np.ndarray:
    in_w, in_h = input_size
    out_w, out_h = output_size
    ones = np.ones((in_h, in_w), dtype=np.float32)
    common = np.ones((out_h, out_w), dtype=bool)
    for matrix in matrices:
        content = cv2.warpPerspective(
            ones,
            np.asarray(matrix, dtype=np.float32),
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        common &= content > 0.5
    return common


def _center_crop_matrix_from_common(common: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray | None:
    out_w, out_h = output_size
    center_x = (out_w - 1) * 0.5
    center_y = (out_h - 1) * 0.5
    target_aspect = out_w / float(out_h)
    lo = 0.0
    hi = 1.0

    def fits(scale: float) -> bool:
        crop_w = max(1.0, out_w / scale)
        crop_h = crop_w / target_aspect
        if crop_h > out_h:
            crop_h = out_h / scale
            crop_w = crop_h * target_aspect
        x0 = int(np.ceil(center_x - crop_w * 0.5))
        y0 = int(np.ceil(center_y - crop_h * 0.5))
        x1 = int(np.floor(center_x + crop_w * 0.5))
        y1 = int(np.floor(center_y + crop_h * 0.5))
        if x0 < 0 or y0 < 0 or x1 >= out_w or y1 >= out_h or x1 <= x0 or y1 <= y0:
            return False
        return bool(common[y0 : y1 + 1, x0 : x1 + 1].all())

    if fits(1.0):
        hi = 1.0
    else:
        hi = 1.0
        while hi <= 4.0 and not fits(hi):
            hi *= 1.25
        if hi > 4.0:
            return None

    for _ in range(32):
        mid = (lo + hi) * 0.5
        if mid < 1.0:
            mid = 1.0
        if fits(mid):
            hi = mid
        else:
            lo = mid

    scale = float(hi)
    crop_w = out_w / scale
    crop_h = crop_w / target_aspect
    if crop_h > out_h:
        crop_h = out_h / scale
        crop_w = crop_h * target_aspect
    x0 = center_x - crop_w * 0.5
    y0 = center_y - crop_h * 0.5
    return np.array(
        [
            [scale, 0.0, -scale * x0],
            [0.0, scale, -scale * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_motion(
    context: VideoContext,
    meta: Dict[str, Any],
    padding_rgb: Tuple[int, int, int],
    *,
    framing_mode: ApplyFramingMode = "pad",
    interpolation: ApplyInterpolation = "bilinear",
    motion_blur: float = 0.0,
    motion_blur_samples: int = 9,
) -> MotionApplyResult:
    motion = resolve_motion_meta(meta)
    _validate_context(context, motion)

    matrices = [transform.matrix for transform in motion.per_frame]
    output_size = motion.output_size
    interp_flag = _interpolation_flag(interpolation)
    result_meta = dict(meta)
    effective_framing = framing_mode
    motion_blur = float(np.clip(motion_blur, 0.0, 1.0))
    motion_blur_samples = int(np.clip(motion_blur_samples, 3, 33))
    warp_fn = _warp_with_matrices if motion_blur <= 0.0 else _warp_with_motion_blur

    if framing_mode == "pad":
        if motion_blur <= 0.0:
            frames, masks = warp_fn(context, matrices, output_size, interp_flag, padding_rgb)
        else:
            frames, masks = warp_fn(
                context,
                matrices,
                output_size,
                interp_flag,
                padding_rgb,
                motion_blur,
                motion_blur_samples,
            )
    elif framing_mode == "crop":
        common = _common_valid_mask(motion.input_size, output_size, matrices)
        crop_matrix = _center_crop_matrix_from_common(common, output_size)
        if crop_matrix is None:
            if motion_blur <= 0.0:
                frames, masks = warp_fn(context, matrices, output_size, interp_flag, padding_rgb)
            else:
                frames, masks = warp_fn(
                    context,
                    matrices,
                    output_size,
                    interp_flag,
                    padding_rgb,
                    motion_blur,
                    motion_blur_samples,
                )
            result_meta["framing_fallback"] = "pad"
            effective_framing = "pad"
        else:
            cropped_matrices = [crop_matrix @ matrix for matrix in matrices]
            if motion_blur <= 0.0:
                frames, masks = warp_fn(
                    context,
                    cropped_matrices,
                    output_size,
                    interp_flag,
                    padding_rgb,
                    masks_zero=True,
                )
            else:
                frames, masks = warp_fn(
                    context,
                    cropped_matrices,
                    output_size,
                    interp_flag,
                    padding_rgb,
                    motion_blur,
                    motion_blur_samples,
                    masks_zero=True,
                )
    else:
        raise ValueError(f"Unsupported framing_mode {framing_mode!r}; expected 'pad' or 'crop'.")

    result_meta["motion_apply"] = {
        "input_size": [int(motion.input_size[0]), int(motion.input_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "framing_mode": effective_framing,
        "interpolation": interpolation,
        "motion_blur": motion_blur,
        "motion_blur_samples": motion_blur_samples,
        "source": motion.source,
    }
    return MotionApplyResult(frames, masks, result_meta)
