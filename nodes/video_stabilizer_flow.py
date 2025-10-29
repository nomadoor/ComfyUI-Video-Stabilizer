"""Dense optical-flow based video stabilizer for ComfyUI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from comfy_api.latest import io
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context

MASK_THRESHOLD = 0.995


@dataclass
class BoundingBox:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)


def _tensor_to_numpy(frames: torch.Tensor) -> np.ndarray:
    return frames.detach().cpu().numpy().astype(np.float32)


def _numpy_to_tensor(frames: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(frames).to(device=reference.device, dtype=reference.dtype)
    return tensor.contiguous()


def _prepare_gray_sequences(
    frames: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    frames_clipped = np.clip(frames, 0.0, 1.0)
    frames_u8 = (frames_clipped * 255.0).round().astype(np.uint8)
    gray_u8 = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.uint8) for frame in frames_u8
    ]
    frames_f32 = frames_clipped.astype(np.float32)
    gray_f32 = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) for frame in frames_f32
    ]
    return gray_u8, gray_f32


def _mask_to_box(mask: np.ndarray) -> BoundingBox:
    valid = mask >= MASK_THRESHOLD
    if not np.any(valid):
        height, width = mask.shape
        return BoundingBox(0, height, 0, width)
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y0, y1 = int(y_indices[0]), int(y_indices[-1] + 1)
    x0, x1 = int(x_indices[0]), int(x_indices[-1] + 1)
    return BoundingBox(y0, y1, x0, x1)


def _compute_zoom_needed(box: BoundingBox, width: int, height: int) -> float:
    if box.width == 0 or box.height == 0:
        return 1.0
    width_ratio = width / max(1, box.width)
    height_ratio = height / max(1, box.height)
    return float(max(width_ratio, height_ratio))


def _compute_base_masks(
    transforms: List[np.ndarray], width: int, height: int
) -> List[np.ndarray]:
    base_masks: List[np.ndarray] = []
    identity_mask = np.ones((height, width), dtype=np.float32)
    for transform in transforms:
        mask = cv2.warpPerspective(
            identity_mask,
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        base_masks.append(np.clip(mask, 0.0, 1.0))
    return base_masks


def _params_from_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "translation":
        return np.array([matrix[0, 2], matrix[1, 2]], dtype=np.float64)
    if method == "similarity":
        tx, ty = matrix[0, 2], matrix[1, 2]
        theta = math.atan2(matrix[1, 0], matrix[0, 0])
        scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        return np.array([tx, ty, theta, math.log(max(scale, 1e-6))], dtype=np.float64)
    tx, ty = matrix[0, 2], matrix[1, 2]
    theta = math.atan2(matrix[1, 0], matrix[0, 0])
    scale = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
    hx, hy = matrix[2, 0], matrix[2, 1]
    return np.array(
        [tx, ty, theta, math.log(max(scale, 1e-6)), hx, hy], dtype=np.float64
    )


def _matrix_from_params(params: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method == "translation":
        tx, ty = params
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = tx
        matrix[1, 2] = ty
        return matrix
    if method == "similarity":
        tx, ty, theta, log_scale = params
        scale = math.exp(log_scale)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        matrix = np.array(
            [
                [scale * cos_t, -scale * sin_t, tx],
                [scale * sin_t, scale * cos_t, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return matrix
    tx, ty, theta, log_scale, hx, hy = params
    matrix = _matrix_from_params(np.array([tx, ty, theta, log_scale]), "similarity")
    matrix[2, 0] = hx
    matrix[2, 1] = hy
    return matrix


def _smooth_params_sequence(
    params_list: List[np.ndarray], smoothness: float, method: str
) -> List[np.ndarray]:
    if not params_list:
        return params_list
    smoothness_clamped = float(np.clip(smoothness, 0.0, 1.0))
    if smoothness_clamped <= 0.0:
        return [params.copy() for params in params_list]
    params = np.stack(params_list, axis=0)
    method = method.lower()
    if method in ("similarity", "perspective"):
        params[:, 2] = np.unwrap(params[:, 2])

    radius = max(1, int(round(5 + smoothness_clamped * 55)))
    sigma = radius / 2.0
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(params, ((radius, radius), (0, 0)), mode="edge")
    smoothed_array = np.empty_like(params)
    for dim in range(params.shape[1]):
        smoothed_array[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")

    smoothed_array[0] = params[0]
    if method in ("similarity", "perspective"):
        smoothed_array[:, 2] = np.unwrap(smoothed_array[:, 2])
    return [smoothed_array[i].copy() for i in range(smoothed_array.shape[0])]


def _blend_params(
    raw_params: List[np.ndarray], target_params: List[np.ndarray], alpha: float
) -> List[np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended: List[np.ndarray] = []
    for raw, target in zip(raw_params, target_params):
        blended.append(raw + alpha * (target - raw))
    if blended:
        blended[0] = raw_params[0]
    return blended


def _enforce_motion_model(matrix: np.ndarray, method: str) -> np.ndarray:
    corrected = matrix.astype(np.float64, copy=True)
    if abs(corrected[2, 2]) > 1e-9:
        corrected /= corrected[2, 2]
    corrected[2, 2] = 1.0
    method = method.lower()
    if method == "translation":
        corrected[0, 0] = 1.0
        corrected[0, 1] = 0.0
        corrected[1, 0] = 0.0
        corrected[1, 1] = 1.0
        corrected[2, 0] = 0.0
        corrected[2, 1] = 0.0
        return corrected
    if method == "similarity":
        a, b, c, d = corrected[0, 0], corrected[0, 1], corrected[1, 0], corrected[1, 1]
        scale_col0 = math.hypot(a, c)
        scale_col1 = math.hypot(b, d)
        scale = max(1e-6, (scale_col0 + scale_col1) * 0.5)
        theta = math.atan2(c, a)
        corrected[0, 0] = scale * math.cos(theta)
        corrected[0, 1] = -scale * math.sin(theta)
        corrected[1, 0] = scale * math.sin(theta)
        corrected[1, 1] = scale * math.cos(theta)
        corrected[2, 0] = 0.0
        corrected[2, 1] = 0.0
        return corrected
    corrected = corrected / max(1e-6, corrected[2, 2])
    corrected[2, 0] = float(np.clip(corrected[2, 0], -0.0015, 0.0015))
    corrected[2, 1] = float(np.clip(corrected[2, 1], -0.0015, 0.0015))
    linear = corrected[:2, :2]
    u, s, vh = np.linalg.svd(linear)
    s_clamped = np.clip(s, 0.3, 3.5)
    corrected[:2, :2] = (u * s_clamped) @ vh
    return corrected


def _flow_to_points(flow: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    ys = np.arange(step / 2, height, step, dtype=np.float32)
    xs = np.arange(step / 2, width, step, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    prev_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    flow_samples = cv2.remap(
        flow,
        grid_x.astype(np.float32),
        grid_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
    )
    curr_points = prev_points + flow_samples.reshape(-1, 2)
    return prev_points, curr_points


def _fit_transform(
    prev_points: np.ndarray, curr_points: np.ndarray, method: str
) -> np.ndarray:
    method = method.lower()
    if len(prev_points) < 6:
        return np.eye(3, dtype=np.float64)
    if method == "translation":
        shift = np.median(curr_points - prev_points, axis=0)
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = shift[0]
        matrix[1, 2] = shift[1]
        return matrix
    if method == "similarity":
        affine, mask = cv2.estimateAffinePartial2D(
            prev_points,
            curr_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
        )
        if affine is None:
            return np.eye(3, dtype=np.float64)
        return _enforce_motion_model(
            np.vstack([affine, np.array([0.0, 0.0, 1.0])]), method
        )
    if len(prev_points) < 8:
        return np.eye(3, dtype=np.float64)
    homography, mask = cv2.findHomography(
        prev_points, curr_points, cv2.RANSAC, 4.0, maxIters=6000
    )
    if homography is None:
        return np.eye(3, dtype=np.float64)
    return homography


def _build_stabilization_transforms(
    raw_transforms: List[np.ndarray],
    raw_params: List[np.ndarray],
    smoothed_params: List[np.ndarray],
    method: str,
    width: int,
    height: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    smoothed_mats = [
        _enforce_motion_model(_matrix_from_params(params, method), method)
        for params in smoothed_params
    ]
    stabilization_transforms: List[np.ndarray] = []
    for raw, smooth in zip(raw_transforms, smoothed_mats):
        try:
            correction = smooth @ np.linalg.inv(raw)
        except np.linalg.LinAlgError:
            correction = np.eye(3, dtype=np.float64)
        if not np.isfinite(correction).all():
            correction = np.eye(3, dtype=np.float64)
        stabilization_transforms.append(_enforce_motion_model(correction, method))
    base_masks = _compute_base_masks(stabilization_transforms, width, height)
    zoom_needed = [
        _compute_zoom_needed(_mask_to_box(mask), width, height) for mask in base_masks
    ]
    return stabilization_transforms, smoothed_mats, base_masks, zoom_needed


def _create_flow_backend(name: str, pyramid_levels: int):
    name = name.lower()
    if name == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis.setUseMeanNormalization(True)
        dis.setFinestScale(0)
        return dis
    if not hasattr(cv2, "optflow") or not hasattr(cv2.optflow, "createOptFlow_DeepFlow"):
        raise RuntimeError(
            "DeepFlow backend requires opencv-contrib-python with cv2.optflow available."
        )
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    return deepflow


class VideoStabilizerFlowNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VideoStabilizerFlow",
            display_name="Video Stabilizer (Flow)",
            category="image/video",
            description=(
                "High-quality video stabilizer using dense optical flow and global motion consensus."
            ),
            inputs=[
                io.Image.Input(
                    "frames",
                    display_name="Frames",
                    tooltip="Input frames (B, H, W, C).",
                ),
                io.Combo.Input(
                    "method",
                    options=["translation", "similarity", "perspective"],
                    default="similarity",
                    display_name="Motion Model",
                    tooltip="Select the geometric transform estimated between frames.",
                ),
                io.Float.Input(
                    "smoothness",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="Smoothing",
                    tooltip="Higher values yield smoother motion at the expense of larger missing borders.",
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Float.Input(
                    "stabilize_zoom",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="Zoom Allowance",
                    tooltip="Maximum automatic zoom (crop) applied to hide borders.",
                    display_mode=io.NumberDisplay.slider,
                ),
                io.Combo.Input(
                    "framing",
                    options=["CROP", "CROP_and_PAD"],
                    default="CROP_and_PAD",
                    display_name="Framing",
                    tooltip="Choose how to treat missing borders after stabilization.",
                ),
                io.Combo.Input(
                    "flow_backend",
                    options=["DIS", "DeepFlow"],
                    default="DIS",
                    display_name="Flow Backend",
                    tooltip="Dense optical-flow solver: DIS is faster, DeepFlow is higher quality.",
                ),
                io.Int.Input(
                    "pyramid_levels",
                    default=5,
                    min=1,
                    max=10,
                    display_name="Flow Pyramid Levels",
                    tooltip="Number of multi-scale levels used by the dense optical-flow solver.",
                ),
                io.String.Input(
                    "pad_color_rgb",
                    default="128,128,128",
                    display_name="Padding Color",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="stabilized_frames"),
                io.Mask.Output(display_name="missing_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames: torch.Tensor,
        flow_backend: str,
        pyramid_levels: int,
        smoothness: float,
        stabilize_zoom: float,
        method: str,
        framing: str,
        pad_color_rgb: str,
    ) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError("Expected frames tensor with shape [N, H, W, C].")

        frame_count = frames.shape[0]
        if frame_count <= 1:
            mask = torch.zeros(
                (frame_count, frames.shape[1], frames.shape[2]),
                dtype=frames.dtype,
                device=frames.device,
            )
            return io.NodeOutput(frames, mask)

        numpy_frames = _tensor_to_numpy(frames)
        frames_linear = np.clip(numpy_frames, 0.0, 1.0)
        gray_u8, gray_f32 = _prepare_gray_sequences(frames_linear)

        backend_lower = flow_backend.lower()
        flow_calc = _create_flow_backend(flow_backend, pyramid_levels)
        flows: List[np.ndarray] = []
        for idx in range(1, frame_count):
            if backend_lower == "dis":
                prev = np.ascontiguousarray(gray_u8[idx - 1])
                curr = np.ascontiguousarray(gray_u8[idx])
            else:
                prev = np.ascontiguousarray(gray_f32[idx - 1])
                curr = np.ascontiguousarray(gray_f32[idx])
            try:
                flow = flow_calc.calc(prev, curr, None)
            except cv2.error:
                flow = None
            if flow is None or flow.size == 0:
                flow = np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)
            flows.append(flow.astype(np.float32, copy=False))

        height, width = frames_linear.shape[1], frames_linear.shape[2]
        step_divisor = max(6, int(pyramid_levels) * 6)
        step = max(4, max(1, min(height, width) // step_divisor))

        raw_transforms: List[np.ndarray] = [np.eye(3, dtype=np.float64)]

        for flow in flows:
            prev_points, curr_points = _flow_to_points(flow, step)
            if prev_points.shape[0] < 4:
                transform = np.eye(3, dtype=np.float64)
            else:
                transform = _fit_transform(prev_points, curr_points, method)
            combined = raw_transforms[-1] @ transform
            if not np.isfinite(combined).all() or abs(np.linalg.det(combined)) < 1e-6:
                combined = raw_transforms[-1]
            raw_transforms.append(_enforce_motion_model(combined, method))

        raw_params = [_params_from_matrix(mat, method) for mat in raw_transforms]
        smooth_params = _smooth_params_sequence(raw_params, smoothness, method)
        smoothness_target = float(np.clip(smoothness, 0.0, 1.0))
        framing_mode = str(framing).upper()
        max_zoom = 1.0 + float(np.clip(stabilize_zoom, 0.0, 1.0))

        if framing_mode == "CROP":
            attempt = 0
            max_zoom_needed = 1.0
            stabilization_transforms: List[np.ndarray] = []
            base_masks: List[np.ndarray] = []
            zoom_needed: List[float] = []
            smoothed_mats: List[np.ndarray] = []
            working_smoothed = smooth_params
            while attempt < 6:
                (
                    stabilization_transforms,
                    smoothed_mats,
                    base_masks,
                    zoom_needed,
                ) = _build_stabilization_transforms(
                    raw_transforms,
                    raw_params,
                    working_smoothed,
                    method,
                    width,
                    height,
                )
                max_zoom_needed = max(zoom_needed) if zoom_needed else 1.0
                if max_zoom_needed <= max_zoom + 1e-3 or smoothness_target <= 0.0:
                    smooth_params = working_smoothed
                    break
                blend = min(1.0, max_zoom / (max_zoom_needed + 1e-6))
                working_smoothed = _blend_params(raw_params, working_smoothed, blend)
                attempt += 1

            if max_zoom_needed > max_zoom + 1e-3:
                (
                    stabilization_transforms,
                    smoothed_mats,
                    base_masks,
                    zoom_needed,
                ) = _build_stabilization_transforms(
                    raw_transforms,
                    raw_params,
                    raw_params,
                    method,
                    width,
                    height,
                )
                max_zoom_needed = max(zoom_needed) if zoom_needed else 1.0
                smooth_params = raw_params
            else:
                smooth_params = working_smoothed
        else:
            (
                stabilization_transforms,
                smoothed_mats,
                base_masks,
                zoom_needed,
            ) = _build_stabilization_transforms(
                raw_transforms,
                raw_params,
                smooth_params,
                method,
                width,
                height,
            )
            max_zoom_needed = max(zoom_needed) if zoom_needed else 1.0

        pad_color = _parse_pad_color(pad_color_rgb)
        pad_color_normalized = tuple(channel / 255.0 for channel in pad_color)

        if framing_mode == "CROP":
            crop_zoom = min(max(zoom_needed) if zoom_needed else 1.0, max_zoom)
            zooms = [crop_zoom] * frame_count
        elif framing_mode == "CROP_AND_PAD":
            zooms = [min(value, max_zoom) for value in zoom_needed]
        else:
            zooms = [1.0] * frame_count
            framing_mode = "CROP_AND_PAD"

        progress_context = get_executing_context()
        progress_node_id = getattr(progress_context, "node_id", None)

        stabilized_frames: List[np.ndarray] = []
        stabilized_masks: List[np.ndarray] = []
        border_mode = cv2.BORDER_CONSTANT
        border_value = pad_color_normalized
        if framing_mode == "CROP":
            border_mode = cv2.BORDER_REPLICATE
            border_value = pad_color_normalized  # unused by replicate, keeps signature explicit
        for idx, (frame, transform, zoom) in enumerate(
            zip(frames_linear, stabilization_transforms, zooms)
        ):
            zoom_matrix = _zoom_matrix(zoom, width, height)
            full_transform = zoom_matrix @ transform
            stabilized = cv2.warpPerspective(
                frame,
                full_transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=border_value,
            )
            mask = cv2.warpPerspective(
                np.ones((height, width), dtype=np.float32),
                full_transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            mask = np.clip(mask, 0.0, 1.0)
            stabilized_frames.append(stabilized)
            stabilized_masks.append(mask)
            if progress_node_id is not None:
                get_progress_state().update_progress(
                    node_id=progress_node_id,
                    value=idx + 1,
                    max_value=frame_count,
                )

        if framing_mode == "CROP":
            intersection = np.ones_like(stabilized_masks[0], dtype=bool)
            for mask in stabilized_masks:
                intersection &= mask >= MASK_THRESHOLD
            if np.any(intersection):
                rows = np.where(np.any(intersection, axis=1))[0]
                cols = np.where(np.any(intersection, axis=0))[0]
                y0, y1 = int(rows[0]), int(rows[-1] + 1)
                x0, x1 = int(cols[0]), int(cols[-1] + 1)
            else:
                y0, y1, x0, x1 = 0, height, 0, width
            cropped_frames = [frame[y0:y1, x0:x1, :] for frame in stabilized_frames]
            output_mask = [
                np.zeros((y1 - y0, x1 - x0), dtype=np.float32) for _ in stabilized_masks
            ]
            frames_tensor = _numpy_to_tensor(np.stack(cropped_frames, axis=0), frames)
            mask_tensor = _numpy_to_tensor(np.stack(output_mask, axis=0), frames)
            if progress_node_id is not None:
                get_progress_state().update_progress(
                    node_id=progress_node_id,
                    value=frame_count,
                    max_value=frame_count,
                )
            return io.NodeOutput(frames_tensor, mask_tensor)

        filled_frames: List[np.ndarray] = []
        output_masks: List[np.ndarray] = []
        for frame, mask in zip(stabilized_frames, stabilized_masks):
            missing = np.clip(1.0 - mask, 0.0, 1.0)
            if np.any(missing > 0.0):
                frame = frame.copy()
                frame[missing > 0.0] = pad_color_normalized
            filled_frames.append(frame)
            output_masks.append(missing)

        frames_tensor = _numpy_to_tensor(np.stack(filled_frames, axis=0), frames)
        mask_tensor = _numpy_to_tensor(np.stack(output_masks, axis=0), frames)
        if progress_node_id is not None:
            get_progress_state().update_progress(
                node_id=progress_node_id,
                value=frame_count,
                max_value=frame_count,
            )
        return io.NodeOutput(frames_tensor, mask_tensor)


def _zoom_matrix(factor: float, width: int, height: int) -> np.ndarray:
    cx = width / 2.0
    cy = height / 2.0
    zoom = np.array(
        [
            [factor, 0.0, (1.0 - factor) * cx],
            [0.0, factor, (1.0 - factor) * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return zoom


def _parse_pad_color(raw: object) -> Tuple[int, int, int]:
    if isinstance(raw, str):
        candidate = raw.strip()
        if candidate.startswith("(") and candidate.endswith(")"):
            candidate = candidate[1:-1]
        parts = [segment.strip() for segment in candidate.split(",") if segment.strip()]
        if len(parts) == 3:
            try:
                values = [float(part) for part in parts]
                return tuple(int(np.clip(round(value), 0, 255)) for value in values)
            except ValueError:
                pass
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 3:
        try:
            return tuple(int(np.clip(float(v), 0, 255)) for v in raw)
        except (TypeError, ValueError):
            pass
    if isinstance(raw, dict):
        keys = ("r", "g", "b")
        if all(k in raw for k in keys):
            return tuple(int(np.clip(raw[k], 0, 255)) for k in keys)
    return 128, 128, 128
