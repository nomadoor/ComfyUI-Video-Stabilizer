#!/usr/bin/env python3
"""Static checks for ComfyUI node schema declarations."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NODES = {
    "nodes/video_stabilizer_classic.py": {
        "node_id": "video_stabilizer_classic",
        "display_name": "Video Stabilizer (Classic)",
    },
    "nodes/video_stabilizer_flow.py": {
        "node_id": "video_stabilizer_flow",
        "display_name": "Video Stabilizer (Flow)",
    },
}
EXPECTED_INPUTS = [
    "frames",
    "frame_rate",
    "framing_mode",
    "transform_mode",
    "camera_lock",
    "strength",
    "smooth",
    "keep_fov",
    "padding_color",
]
EXPECTED_OUTPUTS = ["frames_stabilized", "padding_mask", "meta"]


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _schema_keywords(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
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

        for key, expected_value in expected.items():
            actual = keywords.get(key)
            if actual != expected_value:
                failures.append(f"{relative_path}: {key} expected {expected_value!r}, got {actual!r}")

        if inputs != EXPECTED_INPUTS:
            failures.append(f"{relative_path}: input order mismatch: {inputs!r}")
        if outputs != EXPECTED_OUTPUTS:
            failures.append(f"{relative_path}: output order mismatch: {outputs!r}")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Validated {len(EXPECTED_NODES)} node schema declaration(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
