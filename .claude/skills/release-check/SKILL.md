---
name: release-check
description: Use before version bumps, Comfy Registry publishing, release PRs, tags, or public release notes.
---

# Release Check

Use this skill before release-oriented work.

## Checklist

- Confirm `pyproject.toml` version.
- Confirm runtime dependencies are declared.
- Confirm README / README_ja reflect current behavior.
- Confirm `example_workflows/*.json` parse and reference current node ids.
- Confirm `.github/workflows/publish_action.yml` still publishes only intended
  changes.
- Run `python3 scripts/validate_repo.py`.
- If creating a PR, do not prefix the title with `[codex]`, `[claude]`, or any
  other tool label.

## Notes

This repo currently publishes to Comfy Registry when `pyproject.toml` changes on
`main`, or via manual workflow dispatch.
