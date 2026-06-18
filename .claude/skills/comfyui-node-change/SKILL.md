---
name: comfyui-node-change
description: Use when changing ComfyUI node classes, schemas, inputs, outputs, metadata, or registration.
---

# ComfyUI Node Change

Use this skill for changes under `nodes/` or `__init__.py` that affect node behavior
or ComfyUI schema.

## Read First

- `AGENTS.md`
- Relevant `docs/requirements/*.md`
- Relevant `docs/adr/*.md`
- Existing Classic and Flow implementations if the change touches shared behavior

## Rules

- Preserve graph compatibility unless the user explicitly approves a break.
- Keep `node_id`, socket names, socket order, defaults, categories, and metadata
  stable when possible.
- Prefer shared helpers only when both Classic and Flow genuinely use the same
  behavior.
- Keep ComfyUI imports at the node boundary where possible. Pure math and parsing
  helpers should remain testable without a ComfyUI runtime.
- Validate inputs early and return clear errors or safe fallback output.

## Checks

Run:

```bash
python3 scripts/validate_repo.py
```

When a ComfyUI environment is available, also load the node in ComfyUI and exercise
at least one small workflow.
