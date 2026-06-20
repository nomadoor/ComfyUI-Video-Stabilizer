#!/usr/bin/env python3
"""Run lightweight repository validation checks."""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = [
    "__init__.py",
    "nodes/stabilizer_utils.py",
    "nodes/video_stabilizer_classic.py",
    "nodes/video_stabilizer_flow.py",
    "nodes/video_stabilizer_inverse.py",
    "scripts/compare_refactor_behavior.py",
    "scripts/check_crop_aspect_ratio.py",
    "scripts/check_inverse_stabilization.py",
    "scripts/check_node_schema.py",
    "scripts/check_workflows.py",
]


def check_py_compile() -> int:
    for relative_path in PYTHON_FILES:
        py_compile.compile(str(ROOT / relative_path), doraise=True)
    print(f"Compiled {len(PYTHON_FILES)} Python file(s).")
    return 0


def run_script(relative_path: str) -> int:
    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path)],
        cwd=ROOT,
        check=False,
        text=True,
    )
    return result.returncode


def main() -> int:
    checks = [
        ("python syntax", check_py_compile),
        ("node schema", lambda: run_script("scripts/check_node_schema.py")),
        ("workflow json", lambda: run_script("scripts/check_workflows.py")),
        ("crop aspect ratio", lambda: run_script("scripts/check_crop_aspect_ratio.py")),
        ("inverse stabilization", lambda: run_script("scripts/check_inverse_stabilization.py")),
    ]

    failed: list[str] = []
    for name, check in checks:
        try:
            code = check()
        except Exception as exc:
            print(f"{name}: {exc}")
            code = 1
        if code != 0:
            failed.append(name)

    if failed:
        print("Failed checks: " + ", ".join(failed))
        return 1

    print("All lightweight repository checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
