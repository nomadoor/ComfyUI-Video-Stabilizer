from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for some utilities
    torch = None

TransformMode = Literal["translation", "similarity", "perspective"]

ParamsToMatrix = Callable[[np.ndarray, TransformMode], np.ndarray]

__all__ = [
    "_largest_axis_aligned_rectangle",
    "_compute_crop_with_keep_fov_parametric",
    "_refine_no_padding_crop",
    "_parse_padding_color",
    "_compute_bounding_boxes",
    "_min_content_ratio",
    "_convert_masks_for_output",
]


def _largest_axis_aligned_rectangle(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute the largest axis-aligned rectangle fully contained in ``binary_mask``.

    Args:
        binary_mask: 2D uint8/float mask with values in {0, 1}.

    Returns:
        (x0, y0, w, h) describing the rectangle of ones with the maximum area.
    """
    height, width = binary_mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, width, height)

    for y in range(height):
        row = binary_mask[y]
        heights = (heights + 1) * (row > 0)
        stack: List[int] = []
        x = 0
        while x <= width:
            curr_h = heights[x] if x < width else 0
            if not stack or curr_h >= heights[stack[-1]]:
                stack.append(x)
                x += 1
            else:
                top = stack.pop()
                h = heights[top]
                left = stack[-1] + 1 if stack else 0
                w = x - left
                area = h * w
                if area > best_area:
                    best_area = area
                    x0 = left
                    y0 = y - h + 1
                    best_rect = (x0, y0, w, h)
    return best_rect


def _scale_deltas_parametric(
    params_to_matrix: ParamsToMatrix,
    base_mode: TransformMode,
    deltas: Sequence[np.ndarray],
    scale: float,
) -> List[np.ndarray]:
    """Convert per-frame parameter deltas into matrices after scaling by ``scale``."""
    scale = float(np.clip(scale, 0.0, 1.0))
    return [params_to_matrix(delta * scale, base_mode) for delta in deltas]


def _compute_crop_with_keep_fov_parametric(
    params_to_matrix: ParamsToMatrix,
    base_mode: TransformMode,
    delta_params: Sequence[np.ndarray],
    width: int,
    height: int,
    keep_fov_target: float,
    safety_margin_px: float,
    max_iterations: int = 18,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    float,
    str,
    str | None,
    float,
    List[float],
    List[float],
]:
    """
    Solve for a stabilisation scale that satisfies the requested ``keep_fov``.

    This performs a binary search over a scalar ``s`` applied to the motion removal
    strength. For each candidate the intersection of warped bounds is measured, a
    safety margin is applied, and the resulting crop is evaluated using binary masks.
    """
    keep_fov_clamped = float(np.clip(keep_fov_target, 0.0, 1.0))
    target_ratio = keep_fov_clamped
    eps = 1e-4

    def evaluate_bbox_only(scale: float) -> Tuple[float, Dict[str, object]]:
        mats = _scale_deltas_parametric(params_to_matrix, base_mode, delta_params, scale)
        mins, maxs = _compute_bounding_boxes(mats, width, height)
        x0 = float(np.max(mins[:, 0]))
        y0 = float(np.max(mins[:, 1]))
        x1 = float(np.min(maxs[:, 0]))
        y1 = float(np.min(maxs[:, 1]))

        inter_w = max(0.0, x1 - x0)
        inter_h = max(0.0, y1 - y0)
        ratio_est = 0.0 if inter_w <= 0.0 or inter_h <= 0.0 else min(inter_w / width, inter_h / height)

        margin = min(safety_margin_px, inter_w * 0.25, inter_h * 0.25)
        crop_w = max(1.0, min(width, inter_w - 2.0 * margin))
        crop_h = max(1.0, min(height, inter_h - 2.0 * margin))
        crop_x0 = x0 + max(0.0, (inter_w - crop_w) * 0.5)
        crop_y0 = y0 + max(0.0, (inter_h - crop_h) * 0.5)
        crop_x0 = float(np.clip(crop_x0, 0.0, max(width - crop_w, 0.0)))
        crop_y0 = float(np.clip(crop_y0, 0.0, max(height - crop_h, 0.0)))

        scale_x = width / max(1.0, crop_w)
        scale_y = height / max(1.0, crop_h)
        crop_matrix = np.array(
            [
                [scale_x, 0.0, -scale_x * crop_x0],
                [0.0, scale_y, -scale_y * crop_y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        final_mats = [crop_matrix @ mat for mat in mats]
        return ratio_est, {
            "scale": scale,
            "pre_crop": mats,
            "final": final_mats,
            "crop_origin": [crop_x0, crop_y0],
            "crop_size": [crop_w, crop_h],
        }

    def finalize_with_masks(candidate: Dict[str, object]) -> Dict[str, object]:
        ones = np.ones((height, width), dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)
        masks: List[np.ndarray] = []
        min_ratio = 1.0
        best_origin: List[float] = [0.0, 0.0]
        best_size: List[float] = [float(width), float(height)]

        for matrix in candidate["final"]:
            content = cv2.warpPerspective(
                ones,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            content = (content > 0.5).astype(np.float32)
            content = cv2.dilate(content, kernel, iterations=1)
            content = cv2.erode(content, kernel, iterations=1)
            masks.append(content[..., None])

            coords = np.argwhere(content > 0.5)
            if coords.size == 0:
                ratio = 0.0
                origin = [0.0, 0.0]
                size = [0.0, 0.0]
            else:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                origin = [float(x_min), float(y_min)]
                size = [float(max(1, x_max - x_min + 1)), float(max(1, y_max - y_min + 1))]
                ratio = min(size[0] / width, size[1] / height)

            if ratio < min_ratio:
                min_ratio = ratio
                best_origin = origin
                best_size = size

        candidate = dict(candidate)
        candidate.update(
            {
                "content_masks": masks,
                "ratio_final": float(min_ratio),
                "crop_origin": best_origin,
                "crop_size": best_size,
            }
        )
        return candidate

    if keep_fov_clamped <= eps:
        _, raw = evaluate_bbox_only(1.0)
        candidate = finalize_with_masks(raw)
        return (
            candidate["final"],
            raw["pre_crop"],
            candidate["content_masks"],
            candidate["ratio_final"],
            "disabled",
            None,
            1.0,
            candidate["crop_origin"],
            candidate["crop_size"],
        )

    ratio_full, raw_full = evaluate_bbox_only(1.0)
    if ratio_full >= target_ratio - eps:
        candidate = finalize_with_masks(raw_full)
        return (
            candidate["final"],
            raw_full["pre_crop"],
            candidate["content_masks"],
            candidate["ratio_final"],
            "met",
            None,
            1.0,
            candidate["crop_origin"],
            candidate["crop_size"],
        )

    low, high = 0.0, 1.0
    best_candidate: Dict[str, object] | None = None
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        ratio_mid, raw_mid = evaluate_bbox_only(mid)
        if ratio_mid >= target_ratio - eps:
            best_candidate = raw_mid
            low = mid
        else:
            high = mid

    if best_candidate is None:
        _, raw_zero = evaluate_bbox_only(0.0)
        candidate_zero = finalize_with_masks(raw_zero)
        note = (
            None
            if keep_fov_clamped <= eps
            else f"keep_fov target {keep_fov_clamped:.3f} could not be satisfied even with zero stabilisation."
        )
        return (
            candidate_zero["final"],
            raw_zero["pre_crop"],
            candidate_zero["content_masks"],
            candidate_zero["ratio_final"],
            "failed" if keep_fov_clamped > eps else "disabled",
            note,
            0.0,
            candidate_zero["crop_origin"],
            candidate_zero["crop_size"],
        )

    candidate_final = finalize_with_masks(best_candidate)
    status = "met" if candidate_final["ratio_final"] >= target_ratio - eps else "clamped"
    note = None
    scale_best = float(best_candidate["scale"])
    if status == "clamped" and keep_fov_clamped > eps:
        note = (
            f"keep_fov target {keep_fov_clamped:.3f} reduced to {candidate_final['ratio_final']:.3f} "
            f"at stabilisation scale {scale_best:.3f}."
        )

    return (
        candidate_final["final"],
        best_candidate["pre_crop"],
        candidate_final["content_masks"],
        candidate_final["ratio_final"],
        status,
        note,
        scale_best,
        candidate_final["crop_origin"],
        candidate_final["crop_size"],
    )


def _refine_no_padding_crop(
    final_matrices: Sequence[np.ndarray],
    width: int,
    height: int,
    safety_shrink_px: int = 1,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], float]:
    """
    Post-process crop matrices to guarantee padding-free output in crop mode.
    """
    ones = np.ones((height, width), dtype=np.float32)
    masks_bin: List[np.ndarray] = []
    for matrix in final_matrices:
        content = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        masks_bin.append((content > 0.5).astype(np.uint8))

    common = masks_bin[0].copy()
    for mask in masks_bin[1:]:
        common &= mask

    if safety_shrink_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2 * safety_shrink_px, 1 + 2 * safety_shrink_px))
        common = cv2.erode(common, kernel, iterations=1)

    if common.max() == 0:
        return (
            list(final_matrices),
            [mask[..., None].astype(np.float32) for mask in masks_bin],
            [0.0, 0.0],
            [float(width), float(height)],
            0.0,
        )

    x0, y0, w, h = _largest_axis_aligned_rectangle(common)
    crop_w = max(1, w)
    crop_h = max(1, h)
    scale_x = width / float(crop_w)
    scale_y = height / float(crop_h)
    crop_matrix = np.array(
        [
            [scale_x, 0.0, -scale_x * float(x0)],
            [0.0, scale_y, -scale_y * float(y0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    refined_mats = [crop_matrix @ matrix for matrix in final_matrices]
    refined_masks: List[np.ndarray] = []
    min_ratio = 1.0
    origin_best = [float(x0), float(y0)]
    size_best = [float(crop_w), float(crop_h)]

    for matrix in refined_mats:
        content = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask = (content > 0.5).astype(np.float32)
        refined_masks.append(mask[..., None])
        coords = np.argwhere(mask > 0.5)
        if coords.size == 0:
            ratio = 0.0
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            size_best = [float(max(1, x_max - x_min + 1)), float(max(1, y_max - y_min + 1))]
            origin_best = [float(x_min), float(y_min)]
            ratio = min(size_best[0] / width, size_best[1] / height)
        min_ratio = min(min_ratio, ratio)

    return refined_mats, refined_masks, origin_best, size_best, float(min_ratio)


def _parse_padding_color(value: str) -> Tuple[int, int, int]:
    """Parse user-provided padding color string into RGB tuple."""
    parts = value.replace("/", ",").replace(" ", ",").split(",")
    ints = [int(p) for p in parts if p != ""]
    if len(ints) == 1:
        ints = [ints[0]] * 3
    if len(ints) != 3:
        raise ValueError("Padding color must be one value (grayscale) or three comma-separated values.")
    return tuple(int(np.clip(c, 0, 255)) for c in ints)


def _compute_bounding_boxes(
    matrices: Sequence[np.ndarray],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width, 0.0, 1.0],
            [0.0, height, 1.0],
            [width, height, 1.0],
        ],
        dtype=np.float64,
    ).T

    mins = []
    maxs = []
    for matrix in matrices:
        warped = matrix @ corners
        warped /= warped[2, :]
        xs = warped[0, :]
        ys = warped[1, :]
        mins.append([xs.min(), ys.min()])
        maxs.append([xs.max(), ys.max()])
    return np.array(mins), np.array(maxs)


def _min_content_ratio(
    mins: np.ndarray,
    maxs: np.ndarray,
    width: int,
    height: int,
) -> float:
    x0 = np.max(mins[:, 0])
    y0 = np.max(mins[:, 1])
    x1 = np.min(maxs[:, 0])
    y1 = np.min(maxs[:, 1])

    intersection_w = max(1.0, x1 - x0)
    intersection_h = max(1.0, y1 - y0)
    return max(1e-6, min(intersection_w / width, intersection_h / height))


def _convert_masks_for_output(masks: Iterable[np.ndarray]) -> Any:
    """Convert internal mask list into tensor/array payload for Comfy outputs."""
    masks_2d: List[np.ndarray] = []
    for mask in masks:
        mask_2d = mask[..., 0] if mask.ndim == 3 else mask
        masks_2d.append(mask_2d.astype(np.float32))

    if not masks_2d:
        stacked = np.zeros((1, 1, 1), dtype=np.float32)
    else:
        stacked = np.stack(masks_2d, axis=0)

    stacked = np.ascontiguousarray(stacked)
    if torch is not None:
        return torch.from_numpy(stacked)
    return stacked
