---
name: stabilizer-backend-change
description: Use when changing Classic/Flow motion estimation, OpenCV backend selection, fallback behavior, or stabilization math.
---

# Stabilizer Backend Change

Use this skill for work on feature tracking, dense optical flow, DIS/TV-L1, transform
fitting, smoothing, crop handling, or fallback behavior.

## Context

- Classic estimates sparse feature motion with OpenCV GFTT/LK.
- Flow normally uses OpenCV DIS optical flow.
- `cv2.optflow` is not available in standard OpenCV wheels; TV-L1 requires contrib
  OpenCV or an equivalent available API.
- Existing sample workflows use the Flow node.

## Rules

- Do not assume `cv2` or `cv2.optflow` exists unless declared and checked.
- If a stronger backend is unavailable, prefer safe fallback:
  perspective -> similarity -> translation -> identity.
- Keep fallback metadata auditable (`flow_backend`, applied transform mode,
  confidence/residual data where available).
- Preserve deterministic behavior for identical frames and inputs.
- Avoid GPU-only dependencies; this node is CPU-oriented.

## Useful Checks

```bash
python3 scripts/validate_repo.py
python3 -c "import cv2; print(cv2.__version__, hasattr(cv2, 'DISOpticalFlow'), hasattr(cv2, 'optflow'))"
```

If using a ComfyUI venv, run the `cv2` check inside that venv too.
