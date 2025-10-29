"""ComfyUI Video Stabilizer node based on classic CV primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math

import cv2
import numpy as np
import torch
from typing_extensions import override

from comfy_api.latest import io
from comfy_execution.utils import get_executing_context
from comfy_execution.progress import get_progress_state

DEFAULT_FPS = 30.0
MIN_FEATURES = 12
HOMOGRAPHY_MIN_POINTS = 4
AFFINE_MIN_POINTS = 3
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


def _detect_keypoints(gray_frame: np.ndarray) -> np.ndarray | None:
    features = cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=1200,
        qualityLevel=0.01,
        minDistance=6,
        blockSize=7,
    )
    if features is not None:
        return features.reshape(-1, 2)

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray_frame, None)
    if not keypoints:
        return None
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def _track_keypoints(
    previous: np.ndarray, current: np.ndarray
) -> Tuple[np.ndarray, np.ndarray] | None:
    keypoints_prev = _detect_keypoints(previous)
    if keypoints_prev is None or len(keypoints_prev) < MIN_FEATURES:
        return None

    keypoints_prev = keypoints_prev.reshape(-1, 1, 2).astype(np.float32)
    keypoints_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        keypoints_prev,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if keypoints_curr is None or status is None:
        return None

    status_flat = status.reshape(-1)
    prev_points = keypoints_prev.reshape(-1, 2)[status_flat == 1]
    curr_points = keypoints_curr.reshape(-1, 2)[status_flat == 1]
    if len(prev_points) < MIN_FEATURES:
        return None
    return prev_points, curr_points


def _affine_to_homography(matrix: np.ndarray) -> np.ndarray:
    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = matrix
    return homography


def _estimate_transform(
    previous_gray: np.ndarray, current_gray: np.ndarray, mode: str
) -> np.ndarray:
    tracked = _track_keypoints(previous_gray, current_gray)
    if tracked is None:
        return np.eye(3, dtype=np.float64)

    prev_points, curr_points = tracked
    attempts = {
        "perspective": ("perspective", "similarity", "translation"),
        "similarity": ("similarity", "translation"),
        "translation": ("translation",),
    }
    for candidate in attempts.get(mode, ("similarity", "translation")):
        if candidate == "perspective" and len(prev_points) >= HOMOGRAPHY_MIN_POINTS:
            homography, _ = cv2.findHomography(
                prev_points, curr_points, cv2.RANSAC, 4.0, maxIters=6000
            )
            if homography is not None and np.isfinite(homography).all():
                return homography.astype(np.float64)
        elif candidate in {"similarity", "translation"} and len(prev_points) >= AFFINE_MIN_POINTS:
            affine, _ = cv2.estimateAffinePartial2D(
                prev_points,
                curr_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=6000,
                confidence=0.999,
            )
            if affine is None or not np.isfinite(affine).all():
                continue
            if candidate == "translation":
                affine[0, 0] = 1.0
                affine[0, 1] = 0.0
                affine[1, 0] = 0.0
                affine[1, 1] = 1.0
            return _affine_to_homography(affine.astype(np.float64))

    return np.eye(3, dtype=np.float64)


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
    # perspective -> treat as similarity + projective tail
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

    # perspective
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

    radius = max(1, int(round(3 + smoothness_clamped * 45)))
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
    for raw, smooth in zip(raw_params, target_params):
        blended.append(raw + alpha * (smooth - raw))
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

    # perspective: clamp projective terms and excessive scale drift
    corrected = corrected / max(1e-6, corrected[2, 2])
    corrected[2, 0] = float(np.clip(corrected[2, 0], -0.0015, 0.0015))
    corrected[2, 1] = float(np.clip(corrected[2, 1], -0.0015, 0.0015))
    linear = corrected[:2, :2]
    u, s, vh = np.linalg.svd(linear)
    s_clamped = np.clip(s, 0.3, 3.0)
    corrected[:2, :2] = (u * s_clamped) @ vh
    return corrected


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


def _log_trajectory(label: str, transforms: List[np.ndarray]) -> None:
    # Debug utility intentionally left blank; retained for future instrumentation if needed.
    return


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


class VideoStabilizerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VideoStabilizerClassic",
            display_name="Video Stabilizer (Classic)",
            category="image/video",
            description=(
                "Stabilize handheld footage using classic CV (feature tracking + One Euro smoothing). "
                "Outputs stabilized frames plus a missing-area mask for downstream outpainting."
            ),
            inputs=[
                io.Image.Input(
                    "frames",
                    display_name="Frames",
                    tooltip="Video frames as an image batch (B, H, W, C).",
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
                io.String.Input(
                    "pad_color_rgb",
                    display_name="Padding Color",
                    default="128,128,128",
                    tooltip="Color used to fill missing areas when padding is required (comma-separated R,G,B like 244,124,5).",
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
        method: str,
        smoothness: float,
        stabilize_zoom: float,
        framing: str,
        pad_color_rgb: object,
    ) -> io.NodeOutput:
        if frames.ndim != 4:
            raise ValueError("Expected frames tensor with shape [N, H, W, C].")

        frame_count = frames.shape[0]
        height, width = int(frames.shape[1]), int(frames.shape[2])
        method = str(method).lower()

        if frame_count <= 1:
            mask = torch.zeros((frame_count, height, width), dtype=frames.dtype, device=frames.device)
            return io.NodeOutput(frames, mask)

        numpy_frames = _tensor_to_numpy(frames)
        grayscale_frames: List[np.ndarray] = []
        for frame in numpy_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if gray.dtype != np.uint8:
                gray = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
            grayscale_frames.append(gray)

        raw_transforms = [np.eye(3, dtype=np.float64)]
        for index in range(1, frame_count):
            transform = _estimate_transform(
                grayscale_frames[index - 1],
                grayscale_frames[index],
                method,
            )
            combined = raw_transforms[-1] @ transform
            if not np.isfinite(combined).all() or abs(np.linalg.det(combined)) < 1e-6:
                combined = raw_transforms[-1]
            raw_transforms.append(_enforce_motion_model(combined, method))

        smoothness_target = float(np.clip(smoothness, 0.0, 1.0))
        framing_mode = "CROP" if str(framing).upper() == "CROP" else "CROP_AND_PAD"
        zoom_clamped = float(np.clip(stabilize_zoom, 0.0, 1.0))
        max_zoom = 1.0 + zoom_clamped

        raw_params = [_params_from_matrix(mat, method) for mat in raw_transforms]
        smoothed_params = _smooth_params_sequence(raw_params, smoothness_target, method)

        _log_trajectory("raw-cumulative", raw_transforms)

        if method == "translation":
            (
                stabilization_transforms,
                smoothed_mats,
                base_masks,
                zoom_needed,
            ) = _build_stabilization_transforms(
                raw_transforms,
                raw_params,
                smoothed_params,
                method,
                width,
                height,
            )
            max_required_zoom = max(zoom_needed) if zoom_needed else 1.0
        elif framing_mode == "CROP":
            attempt = 0
            max_required_zoom = 1.0
            stabilization_transforms = []
            base_masks = []
            zoom_needed = []
            smoothed_mats = []

            working_smoothed = smoothed_params
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
                max_required_zoom = max(zoom_needed) if zoom_needed else 1.0
                if max_required_zoom <= max_zoom + 1e-3 or smoothness_target <= 0.0:
                    smoothed_params = working_smoothed
                    break
                blend = min(1.0, max_zoom / (max_required_zoom + 1e-6))
                working_smoothed = _blend_params(raw_params, working_smoothed, blend)
                attempt += 1

            if max_required_zoom > max_zoom + 1e-3:
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
                max_required_zoom = max(zoom_needed) if zoom_needed else 1.0
                smoothed_params = raw_params
            else:
                smoothed_params = working_smoothed
        else:
            (
                stabilization_transforms,
                smoothed_mats,
                base_masks,
                zoom_needed,
            ) = _build_stabilization_transforms(
                raw_transforms,
                raw_params,
                smoothed_params,
                method,
                width,
                height,
            )
            max_required_zoom = max(zoom_needed) if zoom_needed else 1.0

        _log_trajectory("smoothed", smoothed_mats)
        _log_trajectory("stabilization", stabilization_transforms)
        if framing_mode == "CROP":
            effective_zoom = min(max(zoom_needed), max_zoom)
            zooms = [effective_zoom] * frame_count
        else:
            zooms = [min(value, max_zoom) for value in zoom_needed]

        pad_color = _parse_pad_color(pad_color_rgb)
        pad_color_normalized = tuple(channel / 255.0 for channel in pad_color)
        image_border_mode = cv2.BORDER_CONSTANT
        image_border_value = pad_color_normalized
        if framing_mode == "CROP":
            image_border_mode = cv2.BORDER_REPLICATE

        executing_context = get_executing_context()
        progress_node_id = getattr(executing_context, "node_id", None)

        stabilized_frames: List[np.ndarray] = []
        stabilized_masks: List[np.ndarray] = []
        for idx, (frame, transform, zoom) in enumerate(
            zip(numpy_frames, stabilization_transforms, zooms)
        ):
            zoom_matrix = _zoom_matrix(zoom, width, height)
            full_transform = zoom_matrix @ transform
            stabilized = cv2.warpPerspective(
                frame,
                full_transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=image_border_mode,
                borderValue=image_border_value,
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
            inverted = np.clip(1.0 - mask, 0.0, 1.0)
            missing = inverted > 0.0
            if missing.any():
                frame = frame.copy()
                frame[missing] = pad_color_normalized
            filled_frames.append(frame)
            output_masks.append(inverted)

        frames_tensor = _numpy_to_tensor(np.stack(filled_frames, axis=0), frames)
        mask_tensor = _numpy_to_tensor(np.stack(output_masks, axis=0), frames)
        if progress_node_id is not None:
            get_progress_state().update_progress(
                node_id=progress_node_id,
                value=frame_count,
                max_value=frame_count,
            )
        return io.NodeOutput(frames_tensor, mask_tensor)


