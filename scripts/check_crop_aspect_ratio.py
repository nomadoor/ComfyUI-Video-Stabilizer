#!/usr/bin/env python3
"""Validate crop framing preserves aspect ratio without introducing padding."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes.stabilizer_utils import (  # noqa: E402
    _compute_crop_with_keep_fov_parametric,
    _params_to_matrix,
    _refine_no_padding_crop,
)


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=tolerance):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def install_comfy_stubs() -> None:
    """Load the node modules without requiring a running ComfyUI server."""
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        Custom=lambda name: name,
    )
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = type(
        "ProgressBar",
        (),
        {"__init__": lambda self, total: None, "update": lambda self, amount: None},
    )
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils


def make_synthetic_frames(width: int, height: int, count: int = 6) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = xx / max(width - 1, 1)
    base[..., 1] = yy / max(height - 1, 1)
    base[..., 2] = ((xx // 9 + yy // 7) % 2).astype(np.float32)
    cv2.rectangle(base, (12, 10), (44, 34), (1.0, 0.2, 0.1), -1)
    cv2.circle(base, (width - 22, height - 16), 9, (0.1, 0.9, 0.3), -1)

    frames: list[np.ndarray] = []
    for index in range(count):
        matrix = np.array([[1.0, 0.0, index * 1.1], [0.0, 1.0, -index * 0.35]], dtype=np.float32)
        frames.append(
            cv2.warpAffine(
                base,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return frames


def check_node_crop(
    node_module: Any,
    frames: list[np.ndarray],
    width: int,
    height: int,
    transform_mode: str,
    keep_fov: float,
) -> None:
    context = node_module._normalize_video_input(frames)
    result = node_module._stabilize_frames(
        context,
        "crop",
        transform_mode,
        False,
        1.0,
        0.5,
        keep_fov,
        (127, 127, 127),
        24.0,
    )
    if len(result.frames) != len(frames) or len(result.masks) != len(frames):
        raise AssertionError(f"{node_module.__name__} {transform_mode} keep_fov={keep_fov}: unexpected output frame count")
    for index, mask in enumerate(result.masks):
        if mask.shape != (height, width, 1) or float(np.max(mask)) > 0.0:
            raise AssertionError(f"{node_module.__name__} {transform_mode} keep_fov={keep_fov}: frame {index} has crop padding")

    crop_width, crop_height = result.meta["framing"]["crop_size"]
    assert_close(
        f"{node_module.__name__} {transform_mode} keep_fov={keep_fov} crop aspect ratio",
        crop_width / crop_height,
        width / height,
    )
    for entry in result.meta["stabilization_warp"]["per_frame"]:
        matrix = np.asarray(entry["applied_matrix"], dtype=np.float64)
        assert_close(
            f"{node_module.__name__} {transform_mode} keep_fov={keep_fov} uniform crop scale",
            float(matrix[0, 0]),
            float(matrix[1, 1]),
        )


def main() -> int:
    width, height = 121, 73
    deltas = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([5.5, -1.5], dtype=np.float64),
        np.array([-3.0, 2.5], dtype=np.float64),
        np.array([4.0, 1.0], dtype=np.float64),
        np.array([-2.5, -2.0], dtype=np.float64),
    ]
    final_matrices, pre_crop_matrices, _, _, _, _, _, _, _ = _compute_crop_with_keep_fov_parametric(
        _params_to_matrix,
        "translation",
        deltas,
        width,
        height,
        keep_fov_target=0.0,
        safety_margin_px=2.0,
    )
    refined_matrices, content_masks, crop_origin, crop_size, coverage = _refine_no_padding_crop(
        final_matrices,
        width,
        height,
        safety_shrink_px=1,
    )

    crop_width, crop_height = crop_size
    assert_close("crop aspect ratio", crop_width / crop_height, width / height)
    assert_close("coverage", coverage, 1.0)
    if crop_origin[0] < 0.0 or crop_origin[1] < 0.0:
        raise AssertionError(f"crop origin must remain on the canvas, got {crop_origin}")

    # The first pre-crop matrix is identity, so this isolates the crop scale.
    crop_matrix = refined_matrices[0] @ np.linalg.inv(pre_crop_matrices[0])
    assert_close("uniform crop scale", float(crop_matrix[0, 0]), float(crop_matrix[1, 1]))
    assert_close("crop x shear", float(crop_matrix[0, 1]), 0.0)
    assert_close("crop y shear", float(crop_matrix[1, 0]), 0.0)

    for index, (matrix, content_mask) in enumerate(zip(refined_matrices, content_masks, strict=True)):
        expected_mask = cv2.warpPerspective(
            np.ones((height, width), dtype=np.float32),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        if float(np.min(expected_mask)) <= 0.5 or float(np.min(content_mask)) <= 0.5:
            raise AssertionError(f"frame {index}: crop framing introduced padding")

    no_overlap_deltas = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([width * 2.0, 0.0], dtype=np.float64),
    ]
    no_overlap = _compute_crop_with_keep_fov_parametric(
        _params_to_matrix,
        "translation",
        no_overlap_deltas,
        width,
        height,
        keep_fov_target=0.0,
        safety_margin_px=2.0,
    )
    no_overlap_matrices, _, no_overlap_masks, _, _, no_overlap_note, no_overlap_scale, _, _ = no_overlap
    assert_close("no-overlap stabilization scale", no_overlap_scale, 0.0)
    if no_overlap_note is None:
        raise AssertionError("no-overlap crop fallback must report why stabilization was disabled")
    for index, (matrix, mask) in enumerate(zip(no_overlap_matrices, no_overlap_masks, strict=True)):
        assert_close(f"no-overlap matrix {index} x scale", float(matrix[0, 0]), float(matrix[1, 1]))
        if float(np.min(mask)) <= 0.5:
            raise AssertionError(f"no-overlap frame {index}: fallback introduced missing content")

    install_comfy_stubs()
    frames = make_synthetic_frames(width, height)
    for module_name in ("nodes.video_stabilizer_classic", "nodes.video_stabilizer_flow"):
        module = importlib.import_module(module_name)
        for transform_mode in ("translation", "similarity"):
            for keep_fov in (0.0, 0.6):
                check_node_crop(module, frames, width, height, transform_mode, keep_fov)

    print("Crop aspect-ratio and no-padding checks passed for helper, Classic, and Flow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
