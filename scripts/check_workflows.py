#!/usr/bin/env python3
"""Validate bundled ComfyUI workflow JSON files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "example_workflows"
KNOWN_STABILIZER_IDS = {"video_stabilizer_classic", "video_stabilizer_flow"}


def _node_type(node: object) -> str | None:
    if not isinstance(node, dict):
        return None
    value = node.get("type") or node.get("class_type")
    return value if isinstance(value, str) else None


def main() -> int:
    paths = sorted(WORKFLOW_DIR.glob("*.json"))
    if not paths:
        print("No workflow JSON files found.")
        return 1

    failures: list[str] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            failures.append(f"{path}: invalid JSON: {exc}")
            continue

        nodes = data.get("nodes") if isinstance(data, dict) else None
        if not isinstance(nodes, list):
            failures.append(f"{path}: missing top-level nodes list")
            continue

        node_types = {node_type for node in nodes if (node_type := _node_type(node))}
        if not (node_types & KNOWN_STABILIZER_IDS):
            failures.append(f"{path}: no stabilizer node id found")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Validated {len(paths)} workflow JSON file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
