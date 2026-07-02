#!/usr/bin/env python3
"""Static checks for ComfyUI node schema declarations."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NODES = {
    "nodes/video_stabilizer_classic.py": {
        "node_id": "video_stabilizer_classic",
        "display_name": "Video Stabilizer Classic",
        "inputs": [
            "frames",
            "frame_rate",
            "framing_mode",
            "transform_mode",
            "camera_lock",
            "strength",
            "smooth",
            "keep_fov",
            "padding_color",
        ],
        "outputs": ["frames_stabilized", "padding_mask", "meta"],
    },
    "nodes/video_stabilizer_flow.py": {
        "node_id": "video_stabilizer_flow",
        "display_name": "Video Stabilizer Flow",
        "inputs": [
            "frames",
            "frame_rate",
            "framing_mode",
            "transform_mode",
            "camera_lock",
            "strength",
            "smooth",
            "keep_fov",
            "padding_color",
        ],
        "outputs": ["frames_stabilized", "padding_mask", "meta"],
    },
    "nodes/video_stabilizer_inverse.py": {
        "node_id": "video_stabilizer_inverse",
        "display_name": "Video Stabilizer Inverse",
        "is_deprecated": True,
        "inputs": ["frames", "meta", "padding_color"],
        "outputs": ["frames_restored", "padding_mask", "meta"],
    },
    "nodes/video_stabilizer_motion_apply.py": {
        "node_id": "video_stabilizer_motion_apply",
        "display_name": "Video Stabilizer Motion Apply",
        "inputs": [
            "frames",
            "motion_meta",
            "framing_mode",
            "interpolation",
            "padding_color",
            "motion_blur",
            "motion_blur_quality",
        ],
        "outputs": ["frames", "padding_mask", "meta"],
    },
    "nodes/video_stabilizer_shake_generator.py": {
        "node_id": "video_stabilizer_shake_generator",
        "display_name": "Video Stabilizer Shake Generator",
        "inputs": ["frames_context", "frame_rate", "style", "amount", "speed", "seed"],
        "outputs": ["motion_meta"],
    },
    "nodes/video_stabilizer_shake_generator_manual.py": {
        "node_id": "video_stabilizer_shake_generator_manual",
        "display_name": "Video Stabilizer Shake Generator Manual",
        "inputs": [
            "frames_context",
            "frame_rate",
            "pan",
            "tilt",
            "roll",
            "zoom",
            "drift_freq",
            "tremor",
            "tremor_freq",
            "jitter_rate",
            "step",
            "randomness",
            "virtual_fov",
            "amount",
            "speed",
            "seed",
        ],
        "outputs": ["motion_meta"],
    },
}


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _literal_bool(node: ast.AST) -> bool | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) else None


def _schema_keywords(tree: ast.AST) -> dict[str, object]:
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "Schema":
            continue
        for keyword in node.keywords:
            if keyword.arg in {"node_id", "display_name", "category", "description"}:
                value = _literal_string(keyword.value)
                if value is not None:
                    found[keyword.arg] = value
            elif keyword.arg == "is_deprecated":
                value = _literal_bool(keyword.value)
                if value is not None:
                    found[keyword.arg] = value
    return found


def _io_names(tree: ast.AST, suffix: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != suffix:
            continue
        if node.args:
            value = _literal_string(node.args[0])
            if value is not None:
                names.append(value)
    return names


def main() -> int:
    failures: list[str] = []
    for relative_path, expected in EXPECTED_NODES.items():
        path = ROOT / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            keywords = _schema_keywords(tree)
            inputs = _io_names(tree, "Input")
            outputs = _io_names(tree, "Output")
        except FileNotFoundError:
            failures.append(f"{relative_path}: file not found")
            continue
        except SyntaxError as exc:
            failures.append(f"{relative_path}: invalid Python syntax: {exc.msg} at line {exc.lineno}")
            continue

        for key in ("node_id", "display_name"):
            expected_value = expected[key]
            actual = keywords.get(key)
            if actual != expected_value:
                failures.append(f"{relative_path}: {key} expected {expected_value!r}, got {actual!r}")
        if "is_deprecated" in expected and keywords.get("is_deprecated") != expected["is_deprecated"]:
            failures.append(
                f"{relative_path}: is_deprecated expected {expected['is_deprecated']!r}, "
                f"got {keywords.get('is_deprecated')!r}"
            )

        if inputs != expected["inputs"]:
            failures.append(f"{relative_path}: input order mismatch: {inputs!r}")
        if outputs != expected["outputs"]:
            failures.append(f"{relative_path}: output order mismatch: {outputs!r}")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Validated {len(EXPECTED_NODES)} node schema declaration(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
