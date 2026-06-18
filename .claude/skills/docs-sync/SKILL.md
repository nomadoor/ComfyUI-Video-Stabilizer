---
name: docs-sync
description: Use when changing README, README_ja, requirements, ADRs, or user-facing behavior.
---

# Docs Sync

Use this skill when behavior, schema, dependencies, workflow samples, or release
metadata changes.

## What To Sync

- `README.md` and `README_ja.md` should describe the same public behavior.
- `docs/requirements/` should state the intended behavior and acceptance criteria.
- `docs/adr/` should record significant design decisions, especially compatibility
  or fallback changes.
- `pyproject.toml` version/dependencies should match release and runtime behavior.

## Rules

- Keep README concise and user-facing.
- Put implementation rationale in ADRs, not the README.
- Do not document fallback behavior that the code does not actually implement.
- Keep local workflow links valid.

## Checks

```bash
python3 scripts/validate_repo.py
```
