---
name: workflow-update
description: Use when editing example_workflows JSON files or changing node schema in ways that affect sample workflows.
---

# Workflow Update

Use this skill for `example_workflows/*.json` and README workflow links.

## Rules

- Keep workflow JSON valid and formatted as ComfyUI can load it.
- Preserve sample intent: VACE/outpainting examples should demonstrate using
  `padding_mask` from the stabilizer.
- Check that workflows reference current node ids:
  `video_stabilizer_classic` and/or `video_stabilizer_flow`.
- Avoid embedding local machine paths unless they are already part of the sample
  and the user intentionally wants them preserved.
- Update README and README_ja links when sample filenames change.

## Checks

```bash
python3 scripts/check_workflows.py
python3 scripts/validate_repo.py
```
