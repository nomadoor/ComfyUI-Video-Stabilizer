"""
Dense optical flow-based video stabilizer node for ComfyUI V3.

Implements the requirements in docs/requirements/002-video-stabilizer-flow.md:
 - Reuses the same public schema as the classic tracker node
 - Estimates motion using dense optical flow (DIS by default, TV-L1 fallback)
 - Shares framing/metadata semantics with the classic stabilizer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

try:
    import cv2.optflow  # type: ignore[attr-defined]
    _HAS_OPTFLOW = True
except Exception:  # pragma: no cover - optional contrib module
    _HAS_OPTFLOW = False

try:
    import torch
except ImportError:  # pragma: no cover - torch expected at runtime
    torch = None

from comfy_api.latest import ComfyExtension, io
from comfy.utils import ProgressBar

FramingMode = Literal["crop", "crop_and_pad", "expand"]
TransformMode = Literal["translation", "similarity", "perspective"]

JSONType = io.Custom("JSON")


@dataclass
class FrameAdapter:
    dtype: np.dtype
    channel_first: bool
    value_range: Literal["0_1", "0_255"]
    origin: Literal["numpy", "torch"]
    squeeze_last_dim: bool


@dataclass
class VideoContext:
    frames: List[np.ndarray]
    adapter: FrameAdapter
    width: int
    height: int
    channels: int
    fps: float | None
    template_kind: Literal["dict", "sequence"]
    template_meta: Dict[str, Any]


@dataclass
class StabilizationResult:
    frames: List[np.ndarray]
    masks: List[np.ndarray]
    meta: Dict[str, Any]


def _to_numpy_frame(frame: Any) -> Tuple[np.ndarray, FrameAdapter]:
    origin: Literal["numpy", "torch"] = "numpy"
    if torch is not None and isinstance(frame, torch.Tensor):
        origin = "torch"
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    channel_first = False
    squeeze_last_dim = False
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        channel_first = True
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
        squeeze_last_dim = True
    elif arr.ndim == 3 and arr.shape[2] == 1:
        squeeze_last_dim = True

    dtype = arr.dtype
    arr = arr.astype(np.float32)

    if dtype == np.uint8 or arr.max() > 1.5:
        value_range: Literal["0_1", "0_255"] = "0_255"
        arr /= 255.0
    else:
        value_range = "0_1"

    adapter = FrameAdapter(
        dtype=dtype,
        channel_first=channel_first,
        value_range=value_range,
        origin=origin,
        squeeze_last_dim=squeeze_last_dim,
    )
    return arr, adapter


def _normalize_video_input(value: Any) -> VideoContext:
    if isinstance(value, dict):
        candidates = ("frames", "images", "video")
        frames_seq = None
        for key in candidates:
            if key in value:
                frames_seq = value[key]
                break
        if frames_seq is None:
            raise ValueError("Video input dictionary must contain 'frames'.")
        template_kind: Literal["dict", "sequence"] = "dict"
        template_meta = {k: v for k, v in value.items() if k not in candidates}
        fps = template_meta.get("fps")
    else:
        frames_seq = value
        template_kind = "sequence"
        template_meta = {}
        fps = None

    frames_np: List[np.ndarray] = []
    adapter_ref: FrameAdapter | None = None
    width = height = channels = 0

    for frame in frames_seq:
        arr, adapter = _to_numpy_frame(frame)
        if adapter_ref is None:
            adapter_ref = adapter
        else:
            if adapter.channel_first != adapter_ref.channel_first or adapter.origin != adapter_ref.origin:
                raise ValueError("Mixed tensor layouts within the same video sequence are not supported.")
        arr = _ensure_rgb(arr)
        frames_np.append(arr)

    if not frames_np:
        raise ValueError("The input video sequence is empty.")

    height, width, channels = frames_np[0].shape
    return VideoContext(
        frames=frames_np,
        adapter=adapter_ref,  # type: ignore[arg-type]
        width=width,
        height=height,
        channels=channels,
        fps=fps,
        template_kind=template_kind,
        template_meta=template_meta,
    )


def _reconstruct_video(frames: Iterable[np.ndarray], context: VideoContext) -> Any:
    frame_list = list(frames)
    if not frame_list:
        stacked = np.zeros((1, context.height, context.width, 3), dtype=np.float32)
    else:
        stacked = np.stack(frame_list, axis=0).astype(np.float32)
    stacked = np.ascontiguousarray(stacked)
    if torch is not None:
        tensor = torch.from_numpy(stacked)
    else:
        tensor = stacked

    if context.template_kind == "dict":
        payload = dict(context.template_meta)
        payload["frames"] = tensor
        return payload
    return tensor


def _convert_masks_for_output(masks: Iterable[np.ndarray]) -> Any:
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


def _parse_padding_color(value: str) -> Tuple[int, int, int]:
    default = (127, 127, 127)
    if not isinstance(value, str):
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        return default
    rgb: List[int] = []
    for part in parts:
        try:
            rgb.append(int(part))
        except ValueError:
            return default
    return tuple(int(np.clip(channel, 0, 255)) for channel in rgb)


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[..., np.newaxis]
    channels = frame.shape[2]
    if channels == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif channels > 3:
        frame = frame[..., :3]
    return frame


def _make_gray(frame: np.ndarray) -> np.ndarray:
    if frame.shape[2] == 1:
        gray = frame[..., 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


def _create_flow_backend(backend: Literal["DIS", "TVL1"]) -> Any:
    if backend == "TVL1":
        if not _HAS_OPTFLOW:
            raise RuntimeError("OpenCV TV-L1 optical flow is unavailable (cv2.optflow missing).")
        return cv2.optflow.DualTVL1OpticalFlow_create()

    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(2)
    dis.setPatchSize(8)
    dis.setPatchStride(4)
    dis.setUseSpatialPropagation(True)
    return dis


def _estimate_motion_flow(
    backend_obj: Any,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    requested_mode: TransformMode,
    sample_step: int = 8,
) -> Tuple[np.ndarray, TransformMode, float, float]:
    flow = backend_obj.calc(prev_gray, curr_gray, None)
    h, w = prev_gray.shape
    ys = np.arange(0, h, sample_step, dtype=np.int32)
    xs = np.arange(0, w, sample_step, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    prev_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    flow_samples = flow[grid_y, grid_x].reshape(-1, 2)
    curr_points = prev_points + flow_samples

    # Remove invalid points (NaN / inf)
    valid = np.isfinite(curr_points).all(axis=1)
    prev_valid = prev_points[valid]
    curr_valid = curr_points[valid]
    if len(prev_valid) < 12:
        return np.eye(3, dtype=np.float32), "translation", 0.0, 0.0

    mode_priority: Dict[TransformMode, List[TransformMode]] = {
        "perspective": ["perspective", "similarity", "translation"],
        "similarity": ["similarity", "translation"],
        "translation": ["translation"],
    }
    for mode in mode_priority[requested_mode]:
        if mode == "perspective" and len(prev_valid) >= 4:
            matrix, inliers = cv2.findHomography(
                prev_valid,
                curr_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None and inliers is not None:
                confidence = float(inliers.sum()) / float(len(prev_valid))
                if confidence >= 0.15:
                    residuals = np.abs((prev_valid @ matrix[:2, :2].T + matrix[:2, 2]) - curr_valid)
                    return matrix.astype(np.float32), "perspective", confidence, float(residuals.mean())
        elif mode == "similarity" and len(prev_valid) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                prev_valid,
                curr_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(prev_valid)) if inliers is not None else 0.0
                if confidence >= 0.1:
                    affine3x3 = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
                    residuals = np.abs((prev_valid @ matrix[:, :2].T + matrix[:, 2]) - curr_valid)
                    return affine3x3.astype(np.float32), "similarity", confidence, float(residuals.mean())
        elif mode == "translation":
            shifts = curr_valid - prev_valid
            if shifts.ndim > 2:
                shifts = shifts.reshape(-1, shifts.shape[-1])
            delta = np.median(shifts, axis=0).reshape(-1).astype(np.float32)
            tx = float(delta[0]) if delta.size > 0 else 0.0
            ty = float(delta[1]) if delta.size > 1 else 0.0
            matrix = np.array(
                [
                    [1.0, 0.0, tx],
                    [0.0, 1.0, ty],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            confidence = float(len(prev_valid)) / float(len(prev_points))
            residuals = np.abs((prev_valid + np.array([tx, ty], dtype=np.float32)) - curr_valid)
            return matrix, "translation", confidence, float(residuals.mean())

    return np.eye(3, dtype=np.float32), "translation", 0.0, 0.0


def _matrix_to_params(matrix: np.ndarray, base_mode: TransformMode) -> np.ndarray:
    if base_mode == "translation":
        return np.array([matrix[0, 2], matrix[1, 2]], dtype=np.float64)
    if base_mode == "similarity":
        tx, ty = matrix[0, 2], matrix[1, 2]
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        scale = math.sqrt(max(a * a + c * c, 1e-10))
        theta = math.atan2(c, a)
        log_scale = math.log(scale)
        return np.array([tx, ty, theta, log_scale], dtype=np.float64)
    return np.array(
        [
            matrix[0, 0] - 1.0,
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 0],
            matrix[1, 1] - 1.0,
            matrix[1, 2],
            matrix[2, 0],
            matrix[2, 1],
        ],
        dtype=np.float64,
    )


def _params_to_matrix(params: np.ndarray, base_mode: TransformMode) -> np.ndarray:
    if base_mode == "translation":
        return np.array(
            [
                [1.0, 0.0, params[0]],
                [0.0, 1.0, params[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if base_mode == "similarity":
        tx, ty, theta, log_scale = params
        scale = math.exp(log_scale)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array(
            [
                [scale * cos_t, -scale * sin_t, tx],
                [scale * sin_t, scale * cos_t, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            [params[0] + 1.0, params[1], params[2]],
            [params[3], params[4] + 1.0, params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float32,
    )


def _smooth_path(path: np.ndarray, smooth: float, fps: float) -> np.ndarray:
    smooth = float(np.clip(smooth, 0.0, 1.0))
    if smooth <= 0.0 or len(path) <= 2:
        return path.copy()

    fps = float(max(1.0, fps))
    min_seconds = 3.0 / 16.0
    max_seconds = 13.0 / 16.0
    window_seconds = min_seconds + smooth * (max_seconds - min_seconds)
    window = int(round(window_seconds * fps))
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)

    smoothed = np.zeros_like(path)
    for dim in range(path.shape[1]):
        series = path[:, dim]
        padded = np.pad(series, (pad, pad), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    return smoothed


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


def _prepare_expand_transform(
    mins: np.ndarray,
    maxs: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Compute translation to keep all frames within the expanded canvas."""
    x_min = float(np.min(mins[:, 0]))
    y_min = float(np.min(mins[:, 1]))
    x_max = float(np.max(maxs[:, 0]))
    y_max = float(np.max(maxs[:, 1]))

    out_w = int(math.ceil(x_max - x_min))
    out_h = int(math.ceil(y_max - y_min))
    translate = np.array(
        [
            [1.0, 0.0, -x_min],
            [0.0, 1.0, -y_min],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return translate, (max(out_w, 1), max(out_h, 1))


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
    base_mode: TransformMode,
    deltas: Sequence[np.ndarray],
    scale: float,
) -> List[np.ndarray]:
    """Convert per-frame parameter deltas into matrices after scaling by ``scale``."""
    scale = float(np.clip(scale, 0.0, 1.0))
    return [_params_to_matrix(delta * scale, base_mode) for delta in deltas]


def _compute_crop_with_keep_fov_parametric(
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

    def evaluate_bbox_only(scale: float) -> Tuple[float, Dict[str, Any]]:
        mats = _scale_deltas_parametric(base_mode, delta_params, scale)
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

    def finalize_with_masks(candidate: Dict[str, Any]) -> Dict[str, Any]:
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
    best_candidate: Dict[str, Any] | None = None
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
    Post-process the crop by intersecting binary masks and extracting the largest
    rectangle of guaranteed content. This ensures crop mode never exposes padding.
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
    for matrix in refined_mats:
        content = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        refined_masks.append((content > 0.5).astype(np.float32)[..., None])

    ratio = min(float(crop_w) / float(width), float(crop_h) / float(height))
    return refined_mats, refined_masks, [float(x0), float(y0)], [float(crop_w), float(crop_h)], ratio


def _stabilize_frames(
    context: VideoContext,
    framing_mode: FramingMode,
    transform_mode: TransformMode,
    camera_lock: bool,
    strength: float,
    smooth: float,
    keep_fov: float,
    padding_rgb: Tuple[int, int, int],
    frame_rate: float,
) -> StabilizationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the video stabilizer node.")

    frames = context.frames
    total_frames = len(frames)

    fps_candidate = frame_rate
    if not isinstance(fps_candidate, (int, float)) or not np.isfinite(fps_candidate) or fps_candidate <= 0.0:
        fps_candidate = (
            context.fps
            if isinstance(context.fps, (int, float)) and np.isfinite(context.fps) and context.fps > 0.0
            else 16.0
        )
    fps_effective = float(max(1.0, fps_candidate))
    fps_requested = float(frame_rate) if isinstance(frame_rate, (int, float)) and frame_rate > 0.0 else None
    flow_backend: Literal["DIS", "TVL1"] = "DIS"

    if total_frames == 0:
        meta = {
            "frames": 0,
            "note": "Empty frame sequence; nothing to stabilise.",
            "transform_mode_requested": transform_mode,
            "transform_mode_applied": "identity",
            "camera_lock": camera_lock,
            "strength": strength,
            "strength_effective": 0.0,
            "smooth": smooth,
            "fps_requested": fps_requested,
            "fps_effective": None,
            "framing": {
                "mode": framing_mode,
                "input_size": [context.width, context.height],
                "padding_color_rgb": [int(c) for c in padding_rgb],
            },
            "keep_fov_applied": False,
            "padding_color_rgb": [int(c) for c in padding_rgb],
            "flow_backend": flow_backend,
            "estimated_motion": {"per_transition": [], "path": [], "target_path": [], "target_path_effective": []},
            "padding_fraction_mean": 0.0,
            "padding_fraction_max": 0.0,
        }
        return StabilizationResult([], [], meta)

    pbar = ProgressBar(total_frames)

    if len(frames) == 1:
        zero_mask = np.zeros((context.height, context.width, 1), dtype=np.float32)
        frame_rgb = _ensure_rgb(frames[0])
        meta = {
            "frames": 1,
            "note": "Single-frame input; bypassed stabilization.",
            "transform_mode": transform_mode,
            "framing_mode": framing_mode,
            "keep_fov_applied": False,
            "flow_backend": flow_backend,
            "fps_requested": fps_requested,
            "fps_effective": fps_effective,
        }
        pbar.update(total_frames)
        return StabilizationResult([frame_rgb], [zero_mask], meta)

    gray_frames = [_make_gray(frame) for frame in frames]
    base_mode = transform_mode

    try:
        backend_obj = _create_flow_backend(flow_backend)
    except Exception:
        flow_backend = "TVL1"
        backend_obj = _create_flow_backend(flow_backend)

    delta_params: List[np.ndarray] = []
    matrices: List[np.ndarray] = []
    confidences: List[float] = []
    residuals: List[float] = []
    modes_used: List[TransformMode] = []

    active_mode = transform_mode
    for idx in range(1, len(frames)):
        matrix, used_mode, confidence, residual = _estimate_motion_flow(
            backend_obj,
            gray_frames[idx - 1],
            gray_frames[idx],
            active_mode,
        )
        if used_mode != active_mode:
            active_mode = used_mode
        matrices.append(matrix)
        confidences.append(confidence)
        residuals.append(residual)
        modes_used.append(used_mode)
        delta_params.append(_matrix_to_params(matrix, base_mode))

    path = np.zeros((len(frames), delta_params[0].shape[0]), dtype=np.float64)
    for idx, delta in enumerate(delta_params, start=1):
        path[idx] = path[idx - 1] + delta

    strength = float(np.clip(strength, 0.0, 1.0))
    smooth = float(np.clip(smooth, 0.0, 1.0))

    if camera_lock:
        smooth = max(smooth, 0.85)
        target_path = np.zeros_like(path)
    else:
        smooth_path = _smooth_path(path, smooth, fps_effective)
        target_path = path + strength * (smooth_path - path)

    diffs = target_path - path
    delta_params_full: List[np.ndarray] = [diff.copy() for diff in diffs]

    keep_fov_clamped = float(np.clip(keep_fov, 0.0, 1.0))
    keep_fov_applied = framing_mode == "crop" and keep_fov_clamped > 1e-6
    stabilization_scale = 1.0
    keep_fov_status = "disabled"
    keep_fov_note: str | None = None
    keep_fov_effective_value = 1.0
    crop_origin: List[float] = [0.0, 0.0]
    crop_size: List[float] = [float(context.width), float(context.height)]

    final_matrices: List[np.ndarray]
    apply_matrices: List[np.ndarray]
    final_content_masks: List[np.ndarray] | None = None

    if framing_mode == "crop":
        if keep_fov_clamped >= 0.9999:
            zero_mask = np.zeros((context.height, context.width, 1), dtype=np.float32)
            meta = {
                "frames": len(frames),
                "note": "keep_fov~=1.0 in crop mode; returning original frames.",
                "transform_mode_requested": transform_mode,
                "transform_mode_applied": "identity",
                "camera_lock": camera_lock,
                "strength": strength,
                "strength_effective": 0.0,
                "smooth": smooth,
                "fps_requested": fps_requested,
                "fps_effective": fps_effective,
                "framing": {
                    "mode": framing_mode,
                    "input_size": [context.width, context.height],
                    "keep_fov_requested": keep_fov_clamped,
                    "keep_fov_effective": 1.0,
                    "min_content_ratio": 1.0,
                    "padding_color_rgb": list(int(c) for c in padding_rgb),
                    "stabilization_scale": 0.0,
                },
                "keep_fov_applied": False,
                "flow_backend": flow_backend,
                "estimated_motion": {
                    "per_transition": [],
                    "path": path.tolist(),
                    "target_path": target_path.tolist(),
                    "target_path_effective": path.tolist(),
                },
                "padding_fraction_mean": 0.0,
                "padding_fraction_max": 0.0,
            }
            pbar.update(total_frames)
            frames_rgb = [_ensure_rgb(frame) for frame in frames]
            return StabilizationResult(frames_rgb, [zero_mask] * len(frames_rgb), meta)

        safety_margin_px = max(0.5, 0.02 * max(context.width, context.height))
        (
            final_matrices,
            apply_matrices,
            final_content_masks,
            keep_fov_effective_value,
            keep_fov_status,
            keep_fov_note,
            stabilization_scale,
            crop_origin,
            crop_size,
        ) = _compute_crop_with_keep_fov_parametric(
            base_mode,
            delta_params_full,
            context.width,
            context.height,
            keep_fov_clamped,
            safety_margin_px,
        )
        (
            final_matrices,
            final_content_masks,
            crop_origin,
            crop_size,
            keep_fov_effective_value,
        ) = _refine_no_padding_crop(final_matrices, context.width, context.height, safety_shrink_px=1)
        output_size = (context.width, context.height)
    else:
        apply_matrices = _scale_deltas_parametric(base_mode, delta_params_full, 1.0)
        final_matrices = apply_matrices
        output_size = (context.width, context.height)

    mins, maxs = _compute_bounding_boxes(apply_matrices, context.width, context.height)

    framing_meta: Dict[str, Any] = {
        "mode": framing_mode,
        "input_size": [context.width, context.height],
        "padding_color_rgb": list(int(c) for c in padding_rgb),
        "min_content_ratio": _min_content_ratio(mins, maxs, context.width, context.height),
    }

    if framing_mode == "crop":
        framing_meta.update(
            {
                "keep_fov_status": keep_fov_status,
                "keep_fov_effective": keep_fov_effective_value,
                "crop_origin": crop_origin,
                "crop_size": crop_size,
                "actual_content_ratio": keep_fov_effective_value,
                "stabilization_scale": float(stabilization_scale),
            }
        )
        if keep_fov_applied:
            framing_meta["keep_fov_requested"] = keep_fov_clamped
        if keep_fov_note:
            framing_meta["keep_fov_note"] = keep_fov_note
    elif framing_mode == "crop_and_pad":
        x0 = float(np.max(mins[:, 0]))
        y0 = float(np.max(mins[:, 1]))
        x1 = float(np.min(maxs[:, 0]))
        y1 = float(np.min(maxs[:, 1]))
        intersection_w = max(1.0, x1 - x0)
        intersection_h = max(1.0, y1 - y0)
        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        frame_center_x = context.width * 0.5
        frame_center_y = context.height * 0.5
        offset_x = frame_center_x - center_x
        offset_y = frame_center_y - center_y
        translate_matrix = np.array(
            [
                [1.0, 0.0, offset_x],
                [0.0, 1.0, offset_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta.update(
            {
                "safe_region_origin": [x0, y0],
                "safe_region_size": [intersection_w, intersection_h],
                "actual_content_ratio": min(intersection_w / context.width, intersection_h / context.height),
                "center_offset": [offset_x, offset_y],
            }
        )
    else:
        translate_matrix, output_size = _prepare_expand_transform(mins, maxs)
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta["expanded_size"] = list(output_size)

    effective_diffs = (
        np.array([_matrix_to_params(mat, base_mode) for mat in apply_matrices])
        if framing_mode == "crop"
        else np.array(delta_params_full)
    )
    stabilization_scale = float(np.clip(stabilization_scale, 0.0, 1.0))
    strength_effective = strength * stabilization_scale
    effective_target_path = path + effective_diffs

    stabilized_frames: List[np.ndarray] = []
    padding_masks: List[np.ndarray] = []
    padded_ratios: List[float] = []
    padding_detected = False

    padding_array = np.array(padding_rgb, dtype=np.float32) / 255.0
    frame_border_value: Any = float(np.mean(padding_array)) if context.channels == 1 else padding_array.tolist()

    for idx, (frame, matrix) in enumerate(zip(frames, final_matrices)):
        warped = cv2.warpPerspective(
            frame,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=frame_border_value,
        )
        stabilized_frames.append(_ensure_rgb(warped.astype(np.float32)))

        if framing_mode == "crop" and final_content_masks is not None and idx < len(final_content_masks):
            content = final_content_masks[idx][..., 0]
        else:
            content = cv2.warpPerspective(
                np.ones((context.height, context.width), dtype=np.float32),
                matrix,
                output_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        mask = (1.0 - (content > 0.5).astype(np.float32))[..., np.newaxis]
        mask[mask < 1e-3] = 0.0
        if not padding_detected and float(np.max(mask)) > 1e-3:
            padding_detected = True
        padded_ratios.append(float(mask.mean()))
        padding_masks.append(mask)
        pbar.update(1)

    framing_meta["padding_detected"] = padding_detected

    meta = {
        "frames": len(frames),
        "transform_mode_requested": transform_mode,
        "transform_mode_applied": active_mode,
        "camera_lock": camera_lock,
        "strength": strength,
        "strength_effective": strength_effective,
        "smooth": smooth,
        "fps_requested": fps_requested,
        "fps_effective": fps_effective,
        "framing": framing_meta,
        "keep_fov_applied": keep_fov_applied,
        "padding_color_rgb": list(int(c) for c in padding_rgb),
        "flow_backend": flow_backend,
        "estimated_motion": {
            "per_transition": [
                {
                    "index": idx,
                    "mode": mode,
                    "confidence": confidence,
                    "residual": residual,
                    "matrix": matrices[idx].astype(np.float32).tolist()
                    if idx < len(matrices)
                    else np.eye(3, dtype=np.float32).tolist(),
                }
                for idx, (mode, confidence, residual) in enumerate(zip(modes_used, confidences, residuals))
            ],
            "path": path.tolist(),
            "target_path": target_path.tolist(),
            "target_path_effective": effective_target_path.tolist(),
        },
        "padding_fraction_mean": float(np.mean(padded_ratios)),
        "padding_fraction_max": float(np.max(padded_ratios)),
    }

    return StabilizationResult(stabilized_frames, padding_masks, meta)


class VideoStabilizerFlow(io.ComfyNode):
    """Dense optical flow-based stabilizer aligned with the V3 schema."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_flow",
            display_name="Video Stabilizer (Flow)",
            category="Video/Stabilization",
            description=(
                "CPU-friendly video stabilization using dense optical flow with configurable transforms "
                "and framing, emitting stabilized frames, a padding mask, and motion diagnostics."
            ),
        )
        schema.inputs = [
            io.Image.Input("frames", display_name="Frames"),
            io.Float.Input(
                "frame_rate",
                default=16.0,
                min=1.0,
                step=0.1,
                display_name="Input FPS",
                tooltip="Frame rate in frames per second used to scale smoothing window.",
            ),
            io.Combo.Input(
                "framing_mode",
                options=["crop", "crop_and_pad", "expand"],
                default="crop_and_pad",
                display_name="Framing Mode",
                tooltip="Choose how borders produced by stabilization are handled.",
            ),
            io.Combo.Input(
                "transform_mode",
                options=["translation", "similarity", "perspective"],
                default="similarity",
                display_name="Transform Mode",
                tooltip="Select the geometric model fitted to the optical flow.",
            ),
            io.Boolean.Input(
                "camera_lock",
                default=False,
                display_name="Camera Lock",
                tooltip="Aggressively pull the motion curve toward a locked tripod-like solution.",
            ),
            io.Float.Input(
                "strength",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Strength",
                tooltip="Removal gain (0 keeps original motion, 1 removes it using the smoothed motion curve).",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "smooth",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Smooth",
                tooltip="Temporal smoothing amount applied to the motion curve before removal.",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "keep_fov",
                default=0.6,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Keep FOV",
                tooltip=(
                    "[Crop only] How much of the original FOV to preserve (1.0 = no zoom, 0.0 = maximum zoom). "
                    "Ignored when framing_mode is crop_and_pad or expand."
                ),
                display_mode=io.NumberDisplay.slider,
            ),
            io.String.Input(
                "padding_color",
                default="127, 127, 127",
                display_name="Padding RGB",
                tooltip="Comma separated padding color applied in crop_and_pad / expand (e.g. 64, 64, 64).",
            ),
        ]
        schema.outputs = [
            io.Image.Output("frames_stabilized", display_name="Stabilized Frames"),
            io.Mask.Output("padding_mask", display_name="Padding Mask"),
            JSONType.Output("meta", display_name="Meta"),
        ]
        return schema

    @classmethod
    def execute(
        cls,
        frames: Any,
        frame_rate: float,
        framing_mode: FramingMode,
        transform_mode: TransformMode,
        camera_lock: bool,
        strength: float,
        smooth: float,
        keep_fov: float,
        padding_color: str,
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        padding_rgb = _parse_padding_color(padding_color)
        result = _stabilize_frames(
            context=context,
            framing_mode=framing_mode,
            transform_mode=transform_mode,
            camera_lock=camera_lock,
            strength=strength,
            smooth=smooth,
            keep_fov=keep_fov,
            padding_rgb=padding_rgb,
            frame_rate=frame_rate,
        )

        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)
        return io.NodeOutput(video_payload, mask_payload, result.meta)
