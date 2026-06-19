#!/usr/bin/env python3
"""Compare stabilizer behavior between two git refs.

This is a lightweight regression harness for refactors. It loads node modules
from two refs into isolated package names, stubs the small ComfyUI API surface
needed at import time, and compares helper behavior plus Classic/Flow outputs on
synthetic video inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("OpenCV is required to run this comparison script.") from exc


ROOT = Path(__file__).resolve().parents[1]
NODE_FILES = [
    "nodes/stabilizer_utils.py",
    "nodes/video_stabilizer_classic.py",
    "nodes/video_stabilizer_flow.py",
    "nodes/video_stabilizer_inverse.py",
]
ATOL = 1e-5
RTOL = 1e-5


class FakeSchema:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.inputs: list[Any] = []
        self.outputs: list[Any] = []


class _TypedPortFamily:
    port_type = "UNKNOWN"

    @classmethod
    def Input(cls, name: str, **kwargs: Any) -> dict[str, Any]:
        return {"direction": "input", "type": cls.port_type, "name": name, **kwargs}

    @classmethod
    def Output(cls, name: str, **kwargs: Any) -> dict[str, Any]:
        return {"direction": "output", "type": cls.port_type, "name": name, **kwargs}


def port_family(port_type: str) -> type[_TypedPortFamily]:
    return type(f"{port_type.title()}PortFamily", (_TypedPortFamily,), {"port_type": port_type})


class _CustomType:
    def __init__(self, name: str) -> None:
        self.custom_name = name

    def Input(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return {"direction": "input", "custom_type": self.custom_name, "name": name, **kwargs}

    def Output(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return {"direction": "output", "custom_type": self.custom_name, "name": name, **kwargs}


def install_comfy_stubs() -> None:
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        Custom=lambda name: _CustomType(name),
        Image=port_family("IMAGE"),
        Mask=port_family("MASK"),
        Color=port_family("COLOR"),
        Float=port_family("FLOAT"),
        Combo=port_family("COMBO"),
        Boolean=port_family("BOOLEAN"),
        String=port_family("STRING"),
        Int=port_family("INT"),
        Schema=FakeSchema,
        NodeOutput=lambda *args: args,
        NumberDisplay=types.SimpleNamespace(slider="slider"),
    )
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = type(
        "ProgressBar",
        (),
        {"__init__": lambda self, total: None, "update": lambda self, amount: None},
    )
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils


def git_show(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def git_show_optional(ref: str, path: str) -> str | None:
    try:
        return git_show(ref, path)
    except subprocess.CalledProcessError:
        return None


def materialize_ref(ref: str, temp_root: Path) -> Path:
    package_root = temp_root / ref.replace("/", "_").replace("-", "_")
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "nodes").mkdir(exist_ok=True)
    (package_root / "nodes" / "__init__.py").write_text("", encoding="utf-8")
    for relative_path in NODE_FILES:
        content = git_show_optional(ref, relative_path)
        if content is None:
            continue
        destination = package_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return package_root


def load_version(ref: str, temp_root: Path) -> dict[str, Any]:
    package_root = materialize_ref(ref, temp_root)
    package_name = f"compare_{package_root.name}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    nodes_package = types.ModuleType(f"{package_name}.nodes")
    nodes_package.__path__ = [str(package_root / "nodes")]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.nodes"] = nodes_package

    modules: dict[str, Any] = {}
    for short_name in (
        "stabilizer_utils",
        "video_stabilizer_classic",
        "video_stabilizer_flow",
        "video_stabilizer_inverse",
    ):
        module_name = f"{package_name}.nodes.{short_name}"
        path = package_root / "nodes" / f"{short_name}.py"
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[short_name] = module
    return modules


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def assert_close(name: str, left: Any, right: Any, atol: float = ATOL, rtol: float = RTOL) -> None:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        raise AssertionError(f"{name}: shape mismatch {left_arr.shape} != {right_arr.shape}")
    if left_arr.dtype != right_arr.dtype:
        raise AssertionError(f"{name}: dtype mismatch {left_arr.dtype} != {right_arr.dtype}")
    if not np.allclose(left_arr, right_arr, atol=atol, rtol=rtol, equal_nan=True):
        diff = float(np.max(np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))))
        raise AssertionError(f"{name}: values differ (max abs diff {diff})")


def compare_nested(name: str, left: Any, right: Any) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"{name}: key mismatch {sorted(left)} != {sorted(right)}")
        for key in sorted(left):
            compare_nested(f"{name}.{key}", left[key], right[key])
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{name}: length mismatch {len(left)} != {len(right)}")
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            compare_nested(f"{name}[{idx}]", left_item, right_item)
        return
    if isinstance(left, float) or isinstance(right, float):
        if not math.isclose(float(left), float(right), rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"{name}: float mismatch {left!r} != {right!r}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert_close(name, left, right)
        return
    if left != right:
        raise AssertionError(f"{name}: value mismatch {left!r} != {right!r}")


def make_synthetic_frames(frame_count: int = 8, height: int = 45, width: int = 73) -> list[np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[..., 0] = xx / max(width - 1, 1)
    base[..., 1] = yy / max(height - 1, 1)
    base[..., 2] = (((xx // 5) + (yy // 7)) % 2).astype(np.float32)
    cv2.rectangle(base, (9, 8), (31, 24), (1.0, 0.2, 0.1), -1)
    cv2.circle(base, (width - 20, height - 14), 7, (0.1, 1.0, 0.3), -1)

    frames: list[np.ndarray] = []
    center = (width * 0.5, height * 0.5)
    for idx in range(frame_count):
        matrix = cv2.getRotationMatrix2D(center, idx * 0.6, 1.0 + idx * 0.002)
        matrix[0, 2] += idx * 0.7
        matrix[1, 2] += idx * 0.35
        frame = cv2.warpAffine(
            base,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(frame.astype(np.float32))
    return frames


def schema_signature(schema: FakeSchema) -> dict[str, Any]:
    return {
        "schema": schema.kwargs,
        "inputs": schema.inputs,
        "outputs": schema.outputs,
    }


def node_schema_signature(module: Any, node_name: str) -> dict[str, Any]:
    if node_name.endswith("classic"):
        return schema_signature(module.VideoStabilizerClassic.define_schema())
    return schema_signature(module.VideoStabilizerFlow.define_schema())


def input_by_name(schema: dict[str, Any], name: str) -> dict[str, Any]:
    for item in schema["inputs"]:
        if item.get("name") == name:
            return item
    raise AssertionError(f"schema input not found: {name}")


def normalize_expected_schema_delta(node_name: str, base_schema: dict[str, Any], head_schema: dict[str, Any]) -> None:
    base_padding = input_by_name(base_schema, "padding_color")
    head_padding = input_by_name(head_schema, "padding_color")
    base_type = base_padding.get("type")
    head_type = head_padding.get("type")
    if base_type == "STRING" and head_type == "COLOR":
        head_inputs = head_schema["inputs"]
        padding_index = next(idx for idx, item in enumerate(head_inputs) if item.get("name") == "padding_color")
        head_inputs[padding_index] = dict(base_padding)
        return
    if base_type != head_type:
        raise AssertionError(f"schema.{node_name}.padding_color: unexpected type change {base_type!r} -> {head_type!r}")


def compare_schema(base: dict[str, Any], head: dict[str, Any]) -> None:
    for node_name in ("video_stabilizer_classic", "video_stabilizer_flow"):
        base_schema = node_schema_signature(base[node_name], node_name)
        head_schema = node_schema_signature(head[node_name], node_name)
        normalize_expected_schema_delta(node_name, base_schema, head_schema)
        compare_nested(f"schema.{node_name}", base_schema, head_schema)


def compare_helpers(module_name: str, base_module: Any, head_module: Any, frames: list[np.ndarray]) -> None:
    cases: dict[str, Any] = {
        "list": frames,
        "batch": np.stack(frames, axis=0),
        "dict": {"frames": np.stack(frames, axis=0), "fps": 24.0},
        "wrapped_frames": [frame[np.newaxis, ...] for frame in frames],
    }
    for case_name, value in cases.items():
        base_context = base_module._normalize_video_input(value)
        head_context = head_module._normalize_video_input(value)
        for attr in ("width", "height", "channels", "fps", "template_kind", "template_meta"):
            compare_nested(f"{module_name}.normalize.{case_name}.{attr}", getattr(base_context, attr), getattr(head_context, attr))
        for idx, (base_frame, head_frame) in enumerate(zip(base_context.frames, head_context.frames)):
            assert_close(f"{module_name}.normalize.{case_name}.frame[{idx}]", base_frame, head_frame)

        base_reconstructed = base_module._reconstruct_video(base_context.frames, base_context)
        head_reconstructed = head_module._reconstruct_video(head_context.frames, head_context)
        base_payload = to_numpy(base_reconstructed["frames"] if isinstance(base_reconstructed, dict) else base_reconstructed)
        head_payload = to_numpy(head_reconstructed["frames"] if isinstance(head_reconstructed, dict) else head_reconstructed)
        assert_close(f"{module_name}.reconstruct.{case_name}", base_payload, head_payload)

    matrices = {
        "translation": np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.25], [0.0, 0.0, 1.0]], dtype=np.float32),
        "similarity": np.array([[1.02, -0.03, 2.0], [0.03, 1.02, -3.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        "perspective": np.array([[1.01, 0.02, 2.0], [-0.01, 0.99, -1.0], [0.0002, -0.0001, 1.0]], dtype=np.float32),
    }
    for mode, matrix in matrices.items():
        base_params = base_module._matrix_to_params(matrix, mode)
        head_params = head_module._matrix_to_params(matrix, mode)
        assert_close(f"{module_name}.matrix_to_params.{mode}", base_params, head_params)
        assert_close(
            f"{module_name}.params_to_matrix.{mode}",
            base_module._params_to_matrix(base_params, mode),
            head_module._params_to_matrix(head_params, mode),
        )

    path = np.stack(
        [
            np.linspace(0.0, 4.0, 8),
            np.linspace(1.0, -2.0, 8),
            np.sin(np.linspace(0.0, 1.5, 8)),
            np.cos(np.linspace(0.0, 1.5, 8)),
        ],
        axis=1,
    )
    for smooth in (0.0, 0.5, 1.0):
        for fps in (16.0, 24.0, 60.0):
            assert_close(
                f"{module_name}.smooth.{smooth}.{fps}",
                base_module._smooth_path(path, smooth, fps),
                head_module._smooth_path(path, smooth, fps),
            )

    mins = np.array([[-2.0, 1.0], [0.5, -3.0], [1.5, 0.0]], dtype=np.float32)
    maxs = np.array([[73.5, 47.0], [75.0, 45.5], [72.0, 49.0]], dtype=np.float32)
    base_matrix, base_size = base_module._prepare_expand_transform(mins, maxs)
    head_matrix, head_size = head_module._prepare_expand_transform(mins, maxs)
    assert_close(f"{module_name}.expand.matrix", base_matrix, head_matrix)
    compare_nested(f"{module_name}.expand.size", base_size, head_size)


def compare_result(name: str, base_result: Any, head_result: Any) -> None:
    if len(base_result.frames) != len(head_result.frames):
        raise AssertionError(f"{name}: frame count mismatch")
    if len(base_result.masks) != len(head_result.masks):
        raise AssertionError(f"{name}: mask count mismatch")
    for idx, (base_frame, head_frame) in enumerate(zip(base_result.frames, head_result.frames)):
        assert_close(f"{name}.frames[{idx}]", base_frame, head_frame, atol=2e-5, rtol=2e-5)
    for idx, (base_mask, head_mask) in enumerate(zip(base_result.masks, head_result.masks)):
        assert_close(f"{name}.masks[{idx}]", base_mask, head_mask, atol=2e-5, rtol=2e-5)
        if not (0.0 <= float(np.min(head_mask)) <= float(np.max(head_mask)) <= 1.0):
            raise AssertionError(f"{name}.masks[{idx}]: mask values outside 0..1")
    compare_nested(f"{name}.meta", base_result.meta, head_result.meta)


def compare_stabilizers(base: dict[str, Any], head: dict[str, Any], frames: list[np.ndarray]) -> None:
    scenarios = [
        ("crop_and_pad_similarity", "crop_and_pad", "similarity", 0.6),
        ("expand_translation", "expand", "translation", 0.6),
        ("crop_keep_fov_bypass", "crop", "translation", 1.0),
    ]
    for module_name in ("video_stabilizer_classic", "video_stabilizer_flow"):
        for scenario_name, framing_mode, transform_mode, keep_fov in scenarios:
            base_context = base[module_name]._normalize_video_input(frames)
            head_context = head[module_name]._normalize_video_input(frames)
            args = (framing_mode, transform_mode, False, 0.7, 0.5, keep_fov, (127, 127, 127), 24.0)
            base_result = base[module_name]._stabilize_frames(base_context, *args)
            head_result = head[module_name]._stabilize_frames(head_context, *args)
            compare_result(f"{module_name}.{scenario_name}", base_result, head_result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--head-ref", default="HEAD")
    args = parser.parse_args()

    install_comfy_stubs()
    with tempfile.TemporaryDirectory(prefix="stabilizer-compare-") as temp_dir:
        temp_root = Path(temp_dir)
        base_modules = load_version(args.base_ref, temp_root)
        head_modules = load_version(args.head_ref, temp_root)
        frames = make_synthetic_frames()

        compare_schema(base_modules, head_modules)
        for module_name in ("video_stabilizer_classic", "video_stabilizer_flow"):
            compare_helpers(module_name, base_modules[module_name], head_modules[module_name], frames)
        compare_stabilizers(base_modules, head_modules, frames)

    print(f"Behavior comparison passed: {args.base_ref} == {args.head_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
