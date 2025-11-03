"""
Feature-tracking video stabilizer node (Classic backend) for ComfyUI V3.

The implementation follows the requirements in docs/requirements/001-video-stabilizer.md:
 - CPU-only workflow using sparse feature tracking (GFTT + LK optical flow)
 - Deterministic transforms (translation/similarity/perspective) with automatic fallback
 - Framing controls (crop, crop+pad, expand) and mandatory padding mask emission
 - Metadata emission covering confidence, fallback events, and applied transforms
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected at runtime but optional for typing
    torch = None

from comfy_api.latest import ComfyExtension, io

FramingMode = Literal["crop", "crop_and_pad", "expand"]
TransformMode = Literal["translation", "similarity", "perspective"]

JSONType = io.Custom("JSON")


@dataclass
class FrameAdapter:
    """Keeps enough context to project numpy frames back to the original container type."""

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
    """Convert an incoming frame to float32 numpy in RGB order with range 0..1."""
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
        # Some pipelines wrap frames as [1, H, W, C]; remove the batch axis.
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


def _from_numpy_frame(frame: np.ndarray, adapter: FrameAdapter) -> Any:
    """Project the stabilized numpy frame back to the original container representation."""
    output = np.clip(frame, 0.0, 1.0)
    if adapter.value_range == "0_255":
        output = np.rint(output * 255.0).astype(adapter.dtype)
    else:
        output = output.astype(adapter.dtype)

    if adapter.channel_first:
        output = np.moveaxis(output, -1, 0)

    if adapter.squeeze_last_dim and output.ndim >= 3:
        output = output[..., 0]

    if adapter.origin == "torch" and torch is not None:
        return torch.from_numpy(output)
    return output


def _normalize_video_input(value: Any) -> VideoContext:
    """Extract list of numpy frames plus metadata required for reconstruction."""
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

    for idx, frame in enumerate(frames_seq):
        arr, adapter = _to_numpy_frame(frame)
        if adapter_ref is None:
            adapter_ref = adapter
        else:
            # Ensure adapter consistency; mismatches can cause runtime surprises.
            if adapter.channel_first != adapter_ref.channel_first or adapter.origin != adapter_ref.origin:
                raise ValueError("Mixed tensor layouts within the same video sequence are not supported.")
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
    """Pack numpy frames back into the incoming container style."""
    converted = [_from_numpy_frame(frame, context.adapter) for frame in frames]
    if context.template_kind == "dict":
        payload = dict(context.template_meta)
        payload["frames"] = converted
        return payload
    return converted


def _make_gray(frame: np.ndarray) -> np.ndarray:
    """Convert frame to uint8 grayscale for tracking."""
    if frame.shape[2] == 1:
        gray = frame[..., 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return gray_uint8


def _estimate_motion_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    requested_mode: TransformMode,
) -> Tuple[np.ndarray, TransformMode, float]:
    """Estimate geometric transform between consecutive frames with graceful fallback."""
    feature_mask = None
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=21,
        mask=feature_mask,
    )
    if features is None or len(features) < 12:
        # Lack of tracks – return identity transform and let smoothing handle it.
        return np.eye(3, dtype=np.float32), "translation", 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
    )

    status = status.reshape(-1)
    good_prev = features[status == 1]
    good_curr = next_pts[status == 1]
    if len(good_prev) < 8:
        return np.eye(3, dtype=np.float32), "translation", 0.0

    # The order defines fallback precedence.
    mode_priority: Dict[TransformMode, List[TransformMode]] = {
        "perspective": ["perspective", "similarity", "translation"],
        "similarity": ["similarity", "translation"],
        "translation": ["translation"],
    }
    tried_modes = mode_priority[requested_mode]

    for mode in tried_modes:
        if mode == "perspective" and len(good_prev) >= 4:
            matrix, inliers = cv2.findHomography(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(good_prev))
                if confidence < 0.15:
                    continue
                return matrix.astype(np.float32), "perspective", confidence
        elif mode == "similarity" and len(good_prev) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(good_prev)) if inliers is not None else 0.0
                if confidence < 0.1:
                    continue
                affine3x3 = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
                return affine3x3.astype(np.float32), "similarity", confidence
        elif mode == "translation":
            delta = np.median(good_curr - good_prev, axis=0)
            matrix = np.array(
                [
                    [1.0, 0.0, float(delta[0])],
                    [0.0, 1.0, float(delta[1])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            confidence = float(len(good_prev)) / float(len(features))
            return matrix, "translation", confidence

    return np.eye(3, dtype=np.float32), "translation", 0.0


def _matrix_to_params(matrix: np.ndarray, base_mode: TransformMode) -> np.ndarray:
    """Project a transform matrix into the parameter space used for smoothing."""
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

    # Perspective: store deviation from identity for the top 2 rows + projective terms.
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
    """Convert parameter offset back to a homogeneous transform matrix."""
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
    """Apply symmetric moving-average smoothing to the path."""
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
    """Return arrays of min and max XY coordinates after transformation."""
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


def _prepare_crop_transform(
    mins: np.ndarray,
    maxs: np.ndarray,
    width: int,
    height: int,
    keep_fov: float,
) -> Tuple[np.ndarray, dict]:
    """Determine crop transform, effective ratio, and metadata."""
    x0 = np.max(mins[:, 0])
    y0 = np.max(mins[:, 1])
    x1 = np.min(maxs[:, 0])
    y1 = np.min(maxs[:, 1])

    intersection_w = max(1.0, x1 - x0)
    intersection_h = max(1.0, y1 - y0)
    min_ratio = max(1e-3, min(intersection_w / width, intersection_h / height))

    limit = float(np.clip(keep_fov, 0.0, 1.0))
    target_ratio = float(np.clip(max(min_ratio, limit), 1e-3, 1.0))

    crop_w = width * target_ratio
    crop_h = height * target_ratio

    center_x = np.clip((x0 + x1) * 0.5, crop_w * 0.5, width - crop_w * 0.5)
    center_y = np.clip((y0 + y1) * 0.5, crop_h * 0.5, height - crop_h * 0.5)

    crop_x0 = center_x - crop_w * 0.5
    crop_y0 = center_y - crop_h * 0.5

    scale_x = width / crop_w
    scale_y = height / crop_h
    crop_matrix = np.array(
        [
            [scale_x, 0.0, -scale_x * crop_x0],
            [0.0, scale_y, -scale_y * crop_y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    framing_meta = {
        "min_content_ratio": min_ratio,
        "keep_fov_effective": target_ratio,
        "crop_origin": [crop_x0, crop_y0],
        "crop_size": [crop_w, crop_h],
    }
    return crop_matrix, framing_meta


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
) -> StabilizationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the video stabilizer node.")

    frames = context.frames
    if len(frames) == 1:
        zero_mask = np.zeros((context.height, context.width, 1), dtype=np.float32)
        meta = {
            "frames": 1,
            "note": "Single-frame input; bypassed stabilization.",
            "transform_mode": transform_mode,
            "framing_mode": framing_mode,
        }
        return StabilizationResult(frames.copy(), [zero_mask], meta)

    gray_frames = [_make_gray(frame) for frame in frames]
    base_mode = transform_mode

    delta_params: List[np.ndarray] = []
    matrices: List[np.ndarray] = []
    confidences: List[float] = []
    modes_used: List[TransformMode] = []

    active_mode = transform_mode
    for idx in range(1, len(frames)):
        matrix, used_mode, confidence = _estimate_motion_pair(
            gray_frames[idx - 1],
            gray_frames[idx],
            active_mode,
        )
        if used_mode != active_mode:
            active_mode = used_mode
        matrices.append(matrix)
        confidences.append(confidence)
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
    apply_matrices = [_params_to_matrix(diff, base_mode) for diff in diffs]

    mins, maxs = _compute_bounding_boxes(apply_matrices, context.width, context.height)
    final_matrices: List[np.ndarray] = []
    framing_meta: Dict[str, Any] = {
        "mode": framing_mode,
        "input_size": [context.width, context.height],
        "keep_fov_requested": float(np.clip(keep_fov, 0.0, 1.0)),
    }

    if framing_mode in ("crop", "crop_and_pad"):
        crop_matrix, crop_meta = _prepare_crop_transform(mins, maxs, context.width, context.height, keep_fov)
        final_matrices = [crop_matrix @ mat for mat in apply_matrices]
        framing_meta.update(crop_meta)
        output_size = (context.width, context.height)
    else:
        translate_matrix, output_size = _prepare_expand_transform(mins, maxs)
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta["expanded_size"] = list(output_size)

    stabilized_frames: List[np.ndarray] = []
    padding_masks: List[np.ndarray] = []
    padded_ratios: List[float] = []

    for idx, (frame, matrix) in enumerate(zip(frames, final_matrices)):
        warped = cv2.warpPerspective(
            frame,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        stabilized_frames.append(warped.astype(np.float32))

        if framing_mode == "crop":
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

    meta = {
        "frames": len(frames),
        "transform_mode_requested": transform_mode,
        "transform_mode_applied": active_mode,
        "camera_lock": camera_lock,
        "strength": strength,
        "smooth": smooth,
        "framing": framing_meta,
        "estimated_motion": {
            "per_transition": [
                {
                    "index": idx,
                    "mode": mode,
                    "confidence": confidence,
                    "matrix": matrices[idx].astype(np.float32).tolist() if idx < len(matrices) else np.eye(3).tolist(),
                }
                for idx, (mode, confidence) in enumerate(zip(modes_used, confidences))
            ],
            "path": path.tolist(),
            "target_path": target_path.tolist(),
        },
        "padding_fraction_mean": float(np.mean(padded_ratios)),
        "padding_fraction_max": float(np.max(padded_ratios)),
    }

    return StabilizationResult(stabilized_frames, padding_masks, meta)


class VideoStabilizerClassic(io.ComfyNode):
    """Classic feature-tracking video stabilizer aligned with the V3 schema."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="codex.video_stabilizer.classic.v1",
            display_name="Video Stabilizer (Classic)",
            category="Video/Stabilization",
            description=(
                "CPU-friendly video stabilization using sparse feature tracking with configurable transforms "
                "and framing, emitting both stabilized frames and a padding mask."
            ),
        )
        schema.inputs = [
            io.Image.Input("frames", display_name="Frames"),
            io.Combo.Input(
                "framing_mode",
                options=["crop", "crop_and_pad", "expand"],
                default="crop_and_pad",
                display_name="Framing Mode",
                tooltip="Choose how to handle borders produced by stabilization.",
            ),
            io.Combo.Input(
                "transform_mode",
                options=["similarity", "perspective", "translation"],
                default="similarity",
                display_name="Transform Mode",
                tooltip="Select the geometric model used to estimate camera motion.",
            ),
            io.Boolean.Input(
                "camera_lock",
                default=False,
                display_name="Camera Lock",
                tooltip="Treat the shot as tripod-like by aggressively damping motion.",
            ),
            io.Float.Input(
                "strength",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Strength",
                tooltip="Removal gain (0 keeps original motion, 1 removes it based on smoothing).",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "smooth",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Smooth",
                tooltip="Temporal smoothing amount applied to the estimated motion path.",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "keep_fov",
                default=0.6,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Keep FOV",
                tooltip="入力FOVの死守度（1.0 = ズーム0, 0.0 = 最大ズーム）",
                display_mode=io.NumberDisplay.slider,
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
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        result = _stabilize_frames(
            context=context,
            framing_mode=framing_mode,
            transform_mode=transform_mode,
            camera_lock=camera_lock,
            strength=strength,
            smooth=smooth,
            keep_fov=keep_fov,
        )

        video_payload = _reconstruct_video(result.frames, context)
        mask_context = VideoContext(
            frames=result.masks,
            adapter=FrameAdapter(
                dtype=np.float32,
                channel_first=False,
                value_range="0_1",
                origin=context.adapter.origin,
                squeeze_last_dim=True,
            ),
            width=context.width,
            height=context.height,
            channels=1,
            fps=context.fps,
            template_kind=context.template_kind,
            template_meta=dict(context.template_meta),
        )
        mask_payload = _reconstruct_video(result.masks, mask_context)

        return io.NodeOutput(video_payload, mask_payload, result.meta)


class VideoStabilizerClassicExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerClassic]
