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


def _smooth_path(path: np.ndarray, smooth: float) -> np.ndarray:
    smooth = float(np.clip(smooth, 0.0, 1.0))
    if smooth <= 0.0 or len(path) <= 2:
        return path.copy()

    window = int(3 + smooth * 10)
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


def _crop_safety_margin(strength: float, smooth: float, keep_fov: float) -> float:
    base_margin = 0.005  # 0.5% baseline to hide interpolation fringing
    strength_term = 0.02 * np.clip(strength, 0.0, 1.0)
    smooth_term = 0.01 * np.clip(smooth, 0.0, 1.0)
    keep_term = 0.03 * np.clip(1.0 - keep_fov, 0.0, 1.0)
    margin = base_margin + strength_term + smooth_term + keep_term
    return float(np.clip(margin, 0.0, 0.08))


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


def _stabilize_frames(
    context: VideoContext,
    framing_mode: FramingMode,
    transform_mode: TransformMode,
    camera_lock: bool,
    strength: float,
    smooth: float,
    keep_fov: float,
    padding_rgb: Tuple[int, int, int],
) -> StabilizationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the video stabilizer node.")

    frames = context.frames
    total_frames = len(frames)
    pbar = ProgressBar(total_frames) if total_frames > 0 else None
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
        }
        if pbar is not None:
            pbar.update(total_frames)
        return StabilizationResult([frame_rgb], [zero_mask], meta)

    gray_frames = [_make_gray(frame) for frame in frames]
    base_mode = transform_mode

    flow_backend: Literal["DIS", "TVL1"] = "DIS"
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
        smooth_path = _smooth_path(path, smooth)
        target_path = path + strength * (smooth_path - path)

    diffs = target_path - path
    effective_diffs = diffs.copy()
    strength_effective_factor = 1.0
    keep_fov_clamped = float(np.clip(keep_fov, 0.0, 1.0))
    keep_fov_applied = False

    apply_matrices = [_params_to_matrix(diff, base_mode) for diff in effective_diffs]
    mins, maxs = _compute_bounding_boxes(apply_matrices, context.width, context.height)

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
                "framing": {
                    "mode": framing_mode,
                    "input_size": [context.width, context.height],
                    "keep_fov_requested": keep_fov_clamped,
                    "keep_fov_effective": 1.0,
                    "min_content_ratio": 1.0,
                    "padding_color_rgb": list(padding_rgb),
                },
                "keep_fov_applied": False,
                "flow_backend": flow_backend,
                "estimated_motion": {
                    "path": path.tolist(),
                    "target_path": target_path.tolist(),
                    "target_path_effective": path.tolist(),
                },
                "padding_fraction_mean": 0.0,
                "padding_fraction_max": 0.0,
            }
            if pbar is not None:
                pbar.update(total_frames)
            frames_rgb = [_ensure_rgb(frame) for frame in frames]
            return StabilizationResult(frames_rgb, [zero_mask] * len(frames_rgb), meta)

        keep_fov_applied = True
        allowed_ratio = max(keep_fov_clamped, 1e-3)
        max_iterations = 10
        for _ in range(max_iterations):
            needed_ratio = _min_content_ratio(mins, maxs, context.width, context.height)
            if needed_ratio >= allowed_ratio - 1e-6:
                break
            if allowed_ratio <= 1e-6:
                break
            scale = float(np.clip(needed_ratio / allowed_ratio, 0.0, 1.0))
            if scale <= 1e-6:
                break
            effective_diffs *= scale
            strength_effective_factor *= scale
            apply_matrices = [_params_to_matrix(diff, base_mode) for diff in effective_diffs]
            mins, maxs = _compute_bounding_boxes(apply_matrices, context.width, context.height)

    effective_target_path = path + effective_diffs
    final_matrices: List[np.ndarray] = []
    framing_meta: Dict[str, Any] = {
        "mode": framing_mode,
        "input_size": [context.width, context.height],
        "padding_color_rgb": list(padding_rgb),
    }
    if keep_fov_applied:
        framing_meta["keep_fov_requested"] = keep_fov_clamped

    if framing_mode == "crop":
        x0 = float(np.max(mins[:, 0]))
        y0 = float(np.max(mins[:, 1]))
        x1 = float(np.min(maxs[:, 0]))
        y1 = float(np.min(maxs[:, 1]))

        intersection_w = max(1.0, x1 - x0)
        intersection_h = max(1.0, y1 - y0)
        margin_ratio = _crop_safety_margin(
            strength * strength_effective_factor,
            smooth,
            keep_fov_clamped if keep_fov_applied else 1.0,
        )
        shrink_factor = max(0.0, 1.0 - margin_ratio)
        crop_w = max(1.0, intersection_w * shrink_factor)
        crop_h = max(1.0, intersection_h * shrink_factor)
        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        crop_x0 = center_x - crop_w * 0.5
        crop_y0 = center_y - crop_h * 0.5
        crop_x0 = float(np.clip(crop_x0, 0.0, max(context.width - crop_w, 0.0)))
        crop_y0 = float(np.clip(crop_y0, 0.0, max(context.height - crop_h, 0.0)))

        scale_x = context.width / crop_w
        scale_y = context.height / crop_h
        crop_matrix = np.array(
            [
                [scale_x, 0.0, -scale_x * crop_x0],
                [0.0, scale_y, -scale_y * crop_y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        final_matrices = [crop_matrix @ mat for mat in apply_matrices]
        actual_ratio = min(crop_w / context.width, crop_h / context.height)
        framing_meta.update(
            {
                "min_content_ratio": _min_content_ratio(mins, maxs, context.width, context.height),
                "crop_origin": [crop_x0, crop_y0],
                "crop_size": [crop_w, crop_h],
                "actual_content_ratio": actual_ratio,
                "keep_fov_effective": actual_ratio,
                "crop_safety_margin_ratio": margin_ratio,
            }
        )
        output_size = (context.width, context.height)
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
        actual_ratio = min(intersection_w / context.width, intersection_h / context.height)
        framing_meta.update(
            {
                "safe_region_origin": [x0, y0],
                "safe_region_size": [intersection_w, intersection_h],
                "actual_content_ratio": actual_ratio,
                "center_offset": [offset_x, offset_y],
                "min_content_ratio": _min_content_ratio(mins, maxs, context.width, context.height),
            }
        )
        output_size = (context.width, context.height)
    else:
        translate_matrix, output_size = _prepare_expand_transform(mins, maxs)
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta["expanded_size"] = list(output_size)
        framing_meta["min_content_ratio"] = _min_content_ratio(mins, maxs, context.width, context.height)

    stabilized_frames: List[np.ndarray] = []
    padding_masks: List[np.ndarray] = []
    padded_ratios: List[float] = []
    crop_adjusted_frames = 0

    padding_array = np.array(padding_rgb, dtype=np.float32) / 255.0
    if context.channels == 1:
        frame_border_value: Any = float(np.mean(padding_array))
    else:
        frame_border_value = padding_array.tolist()

    for idx, (frame, matrix) in enumerate(zip(frames, final_matrices)):
        warped = cv2.warpPerspective(
            frame,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=frame_border_value,
        )
        warped_rgb = _ensure_rgb(warped.astype(np.float32))
        stabilized_frames.append(warped_rgb)

        if framing_mode == "crop":
            content = cv2.warpPerspective(
                np.ones((context.height, context.width), dtype=np.float32),
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            content = np.clip(content, 0.0, 1.0)
            coverage = float(content.mean())
            if coverage < 0.999:
                coords = np.argwhere(content > 0.5)
                if coords.size > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    content_w = max(1.0, float(x_max - x_min + 1))
                    content_h = max(1.0, float(y_max - y_min + 1))
                    scale_x = context.width / content_w
                    scale_y = context.height / content_h
                    scale = min(scale_x, scale_y)
                    scale = min(scale, 1.05)
                    if scale > 1.0005:
                        center_x = float(x_min + x_max) * 0.5
                        center_y = float(y_min + y_max) * 0.5
                        translate_to_center = np.array(
                            [
                                [1.0, 0.0, -center_x],
                                [0.0, 1.0, -center_y],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                        scale_mat = np.array(
                            [
                                [scale, 0.0, 0.0],
                                [0.0, scale, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                        translate_back = np.array(
                            [
                                [1.0, 0.0, context.width * 0.5],
                                [0.0, 1.0, context.height * 0.5],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                        matrix = translate_back @ scale_mat @ translate_to_center @ matrix
                    warped = cv2.warpPerspective(
                        frame,
                        matrix,
                        output_size,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=frame_border_value,
                    )
                    stabilized_frames[-1] = _ensure_rgb(warped.astype(np.float32))
                    crop_adjusted_frames += 1
            mask = np.zeros((output_size[1], output_size[0], 1), dtype=np.float32)
        else:
            content = cv2.warpPerspective(
                np.ones((context.height, context.width), dtype=np.float32),
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            content = np.clip(content, 0.0, 1.0)
            mask = (1.0 - content)[..., np.newaxis].astype(np.float32)
        padded_ratios.append(float(mask.mean()))
        padding_masks.append(mask)
        if pbar is not None:
            pbar.update(1)

    meta = {
        "frames": len(frames),
        "transform_mode_requested": transform_mode,
        "transform_mode_applied": active_mode,
        "camera_lock": camera_lock,
        "strength": strength,
        "strength_effective": strength * strength_effective_factor,
        "smooth": smooth,
        "framing": framing_meta,
        "keep_fov_applied": keep_fov_applied,
        "flow_backend": flow_backend,
        "estimated_motion": {
            "per_transition": [
                {
                    "index": idx,
                    "mode": mode,
                    "confidence": confidence,
                    "residual": residual,
                    "matrix": matrices[idx].astype(np.float32).tolist(),
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
    if crop_adjusted_frames > 0:
        meta["crop_adjusted_frames"] = crop_adjusted_frames

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
        )

        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)
        return io.NodeOutput(video_payload, mask_payload, result.meta)
