---
name: dependency-audit
description: Use when checking OpenCV/Kornia/ComfyUI dependencies, import failures, pyproject metadata, or environment-specific behavior.
---

# Dependency Audit

Use this skill when investigating runtime dependencies or environment drift.

## Focus Areas

- Direct imports in `nodes/` and `__init__.py`.
- Declared dependencies in `pyproject.toml`.
- What ComfyUI itself installs versus what this package must declare.
- Difference between standard OpenCV and contrib OpenCV.
- Shared venv accidents: a package may import locally because another custom node
  installed it, not because this repo declared it.

## OpenCV Notes

- `cv2.DISOpticalFlow` may exist in standard OpenCV builds.
- `cv2.optflow` is part of OpenCV contrib/extra modules in common Python wheels.
- TV-L1 fallback should not assume `cv2.optflow` exists.

## Useful Commands

```bash
python3 -m pip show opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless kornia
python3 -c "import cv2; print(cv2.__version__, cv2.__file__, hasattr(cv2, 'DISOpticalFlow'), hasattr(cv2, 'optflow'))"
python3 scripts/validate_repo.py
```
