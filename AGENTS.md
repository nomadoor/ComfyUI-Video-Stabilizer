# AGENTS.md - ComfyUI Video Stabilizer

This repository is a Python-only ComfyUI custom node package. Keep this file short:
repo-wide rules live here, task-specific procedures live in `.claude/skills/`, and
mechanical checks live in `scripts/`.

## Repository Map

- `nodes/`: ComfyUI V3 node implementations and shared stabilization helpers.
- `docs/requirements/`: behavior contracts for Classic and Flow stabilizers.
- `docs/adr/`: decisions made while changing node behavior.
- `example_workflows/`: ComfyUI workflow JSON samples that should keep working.
- `.github/workflows/publish_action.yml`: Comfy Registry publish workflow.

## Core Rules

- Preserve existing graph compatibility unless the user explicitly accepts a break.
  Keep node ids, socket names, socket order, and metadata shape stable.
- Follow ComfyUI V3 schema conventions for node classes and `define_schema()`.
- Keep outputs deterministic for the same input. Avoid hidden randomness.
- Do not add network I/O, file I/O, subprocess calls, or new dependencies unless
  the relevant requirement/spec says so.
- Declare runtime dependencies in `pyproject.toml`. OpenCV use must not rely on
  whatever happens to be installed in a shared ComfyUI environment.
- Fail softly with clear errors or safe fallback behavior. Media nodes should
  prefer translation/identity fallback over crashing when possible.
- Update requirements/ADR/README/workflows when user-facing behavior changes.
- Do not put `[codex]`, `[claude]`, or similar tool labels in PR titles.

## Verification

Run focused checks after changes:

```bash
python3 scripts/validate_repo.py
```

If a ComfyUI checkout/venv is available, also test node loading and runtime there.
This repository's lightweight scripts intentionally avoid importing ComfyUI so they
can run in a plain Python environment.

## Task Skills

Use the relevant skill before substantial work:

- `.claude/skills/comfyui-node-change/SKILL.md`
- `.claude/skills/stabilizer-backend-change/SKILL.md`
- `.claude/skills/workflow-update/SKILL.md`
- `.claude/skills/docs-sync/SKILL.md`
- `.claude/skills/release-check/SKILL.md`
- `.claude/skills/dependency-audit/SKILL.md`
- `.claude/skills/article-or-announcement/SKILL.md`
